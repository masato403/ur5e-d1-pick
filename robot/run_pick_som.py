"""学習なしで、カメラ画像を見ながら UR5e で積み木を箱に運ぶ（MuJoCo のシミュレーター、30FPS）。

  見る係　：最初に撮った「何も置いていない作業台」の画像と比べて、色のついた違う部分を「物」として切り出す
            （何の物かは区別しない。YOLO などの学習はしない）
  決める係：物に番号を書き込んだ画像＋指示を d1-3B に渡し、選択肢（番号＋プログラムで計算した事実）から選ばせる
            （Jev と同じ「状況と選択肢を渡すだけ」。追加学習なし）。判断は別スレッドで、ロボットの制御は止めない
  動かす係：選ばれた物の位置へ逆運動学で動かし、吸着して箱へ

  python robot/run_pick_som.py --instr "Put only the blue block into the box." --video robot/som_blue.mp4
"""
import argparse
import json
import math
import os
import sys
import threading
import time

import cv2
import mujoco
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
os.chdir(os.path.join(HERE, ".."))

from run_pick import DOWN, GRIP_GAP, IK, SAFE_Z, SLOTS, SPEED  # noqa: E402
from sim_env import BIN, BIN_HALF, BLOCKS, HALF, HOME, Env  # noqa: E402

FPS = 30
DT = 1 / FPS


# ------------------------------------------------------------ 見る係（学習なし）
class SegPerception:
    """作業台だけの画像（最初に 1 枚撮る）と違って、色がついている部分を物とする。
    ロボットの腕は灰色・白・黒なので、色の濃さ（彩度）で除く。腕に隠れた物は、最後に見えた位置を覚えておく。"""

    def __init__(self, ref_rgb, diff=0.15, sat=70, min_area=120):
        self.ref = self._chroma(ref_rgb)
        self.diff, self.sat, self.min_area = diff, sat, min_area
        self.objs = {}          # 番号 → {"xy": 机の上の位置, "uv": 画像の位置, "seen": 最後に見えたフレーム}
        self.next_id = 1
        self.held = None        # ロボットが運んでいる物の番号

    @staticmethod
    def _chroma(rgb):
        """明るさを除いた色の割合（影になっても変わらない）。"""
        f = rgb.astype(np.float32) + 1.0
        return f / f.sum(2, keepdims=True)

    def update(self, rgb, frame_no, robot_mask=None):
        d = np.abs(self._chroma(rgb) - self.ref).sum(2)
        s = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
        m = ((d > self.diff) & (s > self.sat)).astype(np.uint8)
        if robot_mask is not None:            # ロボット自身は除く（関節の角度とロボットの形から、画像のどこに映るかが分かる）
            m[robot_mask] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        hue = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 0].astype(np.float32)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        found = []
        for k in range(1, n):
            x, y, w, h, a = st[k]
            if a < self.min_area:
                continue
            ys, xs = np.nonzero(lab == k)
            hk = hue[ys, xs]
            ang = hk * (np.pi / 90)                       # 色相は輪になっているので、角度で扱う
            pts = np.stack([np.cos(ang), np.sin(ang)], 1)
            parts = [(xs, ys)]
            if np.linalg.norm(pts.mean(0)) < 0.9:         # 違う色が混ざっている（くっついた別々の物）→ 色で 2 つに分ける
                _, lb, _ = cv2.kmeans(pts.astype(np.float32), 2, None,
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3), 3, cv2.KMEANS_PP_CENTERS)
                lb = lb.ravel()
                parts = [(xs[lb == j], ys[lb == j]) for j in (0, 1)]
            for px, py in parts:
                if len(px) < self.min_area:
                    continue
                pw, ph = px.max() - px.min() + 1, py.max() - py.min() + 1
                if len(px) <= 2000 and max(pw, ph) <= 45:
                    found.append((float(px.mean()), float(py.mean()), int(pw), int(ph)))
        # 前に見えた物と近ければ同じ物（番号を引き継ぐ）。新しく見えた物には新しい番号
        used = set()
        for u, v, w, h in found:
            xy = Env.pixel_to_table(u, v)
            best, bd = None, 0.04
            for k, o in self.objs.items():
                dd = np.linalg.norm(o["xy"] - xy)
                if k not in used and k != self.held and dd < bd:
                    best, bd = k, dd
            if best is None:
                best = self.next_id
                self.next_id += 1
            used.add(best)
            self.objs[best] = {"xy": xy, "uv": (u, v, w, h), "seen": frame_no}
        return found

    def visible(self, frame_no, within=3):
        return {k: o for k, o in self.objs.items() if frame_no - o["seen"] <= within}


def in_box(xy):
    return bool(np.all(np.abs(xy - BIN) < BIN_HALF))


def draw_marks(rgb, objs):
    im = rgb.copy()
    for k, o in objs.items():
        u, v, w, h = o["uv"]
        x, y = int(u + w / 2 + 12), int(v - h / 2 - 10)
        cv2.circle(im, (x, y), 13, (255, 255, 255), -1)
        cv2.circle(im, (x, y), 13, (0, 0, 0), 2)
        cv2.putText(im, str(k), (x - 7 if k < 10 else x - 12, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    return im


# ------------------------------------------------------------ 決める係（d1-3B、Jev と同じ形）
class SomBrain:
    """作業台の上の物 1 つずつに「指示どおりなら箱に入れるべき？」と はい／いいえ で聞く（1 回の呼び出しでまとめて聞く）。
    いちばん「はい」が高い物を次に運ぶ。どれも 0.5 未満なら終わり。箱の中の物は聞かない（状況の文章で伝える）。"""
    STATE = ("Camera image of a robot work table seen from above. The yellow square is the box. "
             "White circles with numbers mark the objects.")

    def __init__(self, instr):
        from d1_fast import D1Fast
        self.m = D1Fast(quant="fp8").model
        self.instr = instr
        self.result, self.busy, self.ms, self.last = None, False, [], None

    def ask(self, rgb, objs):
        from PIL import Image
        table = sorted(k for k, o in objs.items() if not in_box(o["xy"]))
        inbox = sorted(k for k, o in objs.items() if in_box(o["xy"]))
        state = self.STATE + (f" Objects already inside the box: {', '.join(map(str, inbox))}." if inbox else
                              " The box is empty.")
        qs = {f"obj{k}": {"type": "noul",
                          "instructions": f"Instruction: {self.instr} Is object number {k} one of the objects this "
                                          "instruction asks to put into the box?"} for k in table}
        img = Image.fromarray(cv2.resize(draw_marks(rgb, objs), (512, 384)))
        self.busy, self.result = True, None

        def work():
            t = time.perf_counter()
            ans = {}
            for k, q in qs.items():         # 1 つずつ聞く（まとめて聞く方法は、この PC の PyTorch にない機能を使うため）
                o = self.m.system_one(state, {k: q}, images=[img])
                ans[k] = round(o["answers"][k]["noul"], 3)
            self.ms.append((time.perf_counter() - t) * 1000)
            best = max(ans, key=ans.get) if ans else None
            self.last = {"yes": ans, "in_box": inbox}
            self.result = best if best and ans[best] >= 0.5 else "done"
            self.busy = False
        threading.Thread(target=work, daemon=True).start()


def empty_table(env):
    """何も置いていない作業台を撮る。積み木は遠くへどけ、ロボットの腕はカメラに映らない向き（後ろ）へ回す。
    実機では、物を置く前に腕をよけて 1 枚撮っておけばよい。"""
    for i, b in enumerate(BLOCKS):
        env.set_block(b, np.array([5.0 + i, 5.0]), HALF)
    env.d.qpos[:6] = HOME + np.array([np.pi, 0, 0, 0, 0, 0])
    mujoco.mj_forward(env.m, env.d)
    return env.render()


# ------------------------------------------------------------ 本体
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instr", default="Put all blocks into the box.")
    ap.add_argument("--want", default="red,blue,green", help="答え合わせ用：指示どおりなら箱に入るべき色")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--video", default="")
    ap.add_argument("--log", default="robot/logs/run_pick_som.jsonl")
    a = ap.parse_args()

    env = Env()
    # 最初に「何も置いていない作業台」を撮る（積み木を遠くへ動かして撮る。実機では物を置く前に 1 枚撮る）
    env.reset(np.random.default_rng(a.seed))
    keep = {b: env.block_pos(b) for b in BLOCKS}
    per = SegPerception(empty_table(env))
    for b in BLOCKS:
        env.set_block(b, keep[b][:2], HALF)
    env.d.qpos[:6] = HOME
    mujoco.mj_forward(env.m, env.d)

    ik = IK(env)
    seg = mujoco.Renderer(env.m, 480, 640)
    seg.enable_segmentation_rendering()
    robot_geoms = np.array([g for g in range(env.m.ngeom) if env.m.geom_bodyid[g] not in
                            [mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in BLOCKS] + [0]])

    def robot_mask():
        """ロボットが画像のどこに映っているか（実機でも、関節の角度とロボットの形から計算できる）。"""
        seg.update_scene(env.d, camera="top")
        ids = seg.render()[..., 0]
        mk = np.isin(ids, robot_geoms).astype(np.uint8)
        return cv2.dilate(mk, np.ones((7, 7), np.uint8)).astype(bool)

    brain = SomBrain(a.instr)
    n_sub = round(DT / env.m.opt.timestep)
    q_cmd = HOME.copy()
    goal = env.tool_pos()
    cmd = goal.copy()
    phase, target, held, slot_i = "decide", None, None, 0
    slot = BIN
    events, picks = [], []
    writer = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 480)) if a.video else None
    t_loop, late, frame_no = [], 0, 0
    wall0 = time.perf_counter()
    next_t = wall0
    held_body = None
    while frame_no < a.seconds * FPS:
        t0 = time.perf_counter()
        for _ in range(n_sub):
            env.d.qfrc_applied[:6] = env.d.qfrc_bias[:6]
            mujoco.mj_step(env.m, env.d)
            if held_body:
                p = env.tool_pos()
                env.set_block(held_body, p[:2], p[2] - GRIP_GAP - HALF)
        mujoco.mj_forward(env.m, env.d)
        frame = env.render()
        per.update(frame, frame_no, robot_mask())
        tool = env.tool_pos()
        sim_t = frame_no * DT
        if phase == "decide":
            if not brain.busy and brain.result is None:
                vis = per.visible(frame_no)
                if vis:
                    brain.ask(frame, vis)
            elif brain.result is not None:
                ch, brain.result = brain.result, None
                events.append({"t": round(sim_t, 2), "decide": ch, **(brain.last or {})})
                if ch == "done":
                    phase = "finished"
                elif ch.startswith("obj") and int(ch[3:]) in per.objs:
                    target, phase = int(ch[3:]), "approach"
            goal = np.array([tool[0], tool[1], SAFE_Z]) if tool[2] < SAFE_Z - 0.02 else goal
        elif phase in ("approach", "descend"):
            xy = per.objs[target]["xy"]                      # 見えている間は、見えた位置で目標を直す
            z = SAFE_Z if phase == "approach" else 2 * HALF + GRIP_GAP
            goal = np.array([xy[0], xy[1], z])
            if np.linalg.norm(cmd - goal) < 1e-3 and np.linalg.norm(tool - goal) < (0.01 if phase == "approach" else 0.004):
                if phase == "approach":
                    phase = "descend"
                else:
                    # 吸着：工具の先のすぐ下にある積み木をつかむ（実機では吸着パッドの ON）
                    near = min(BLOCKS, key=lambda b: np.linalg.norm(env.block_pos(b)[:2] - tool[:2]))
                    if np.linalg.norm(env.block_pos(near)[:2] - tool[:2]) < 0.03:
                        held_body = near
                        per.held = target                    # 運んでいる間は、この番号を見えなくても消さない
                        env.hold(near, True)
                        picks.append(near)
                    phase = "lift"
        elif phase == "lift":
            goal = np.array([tool[0], tool[1], SAFE_Z]) if goal[2] < SAFE_Z else goal
            if np.linalg.norm(tool - goal) < 0.01:
                phase = "carry"
        elif phase == "carry":
            slot = BIN + SLOTS[slot_i % len(SLOTS)]
            goal = np.array([slot[0], slot[1], SAFE_Z])
            if np.linalg.norm(cmd - goal) < 1e-3 and np.linalg.norm(tool - goal) < 0.01:
                phase = "lower"
        elif phase == "lower":
            goal = np.array([slot[0], slot[1], 2 * HALF + GRIP_GAP + 0.002])
            if np.linalg.norm(cmd - goal) < 1e-3 and np.linalg.norm(tool - goal) < 0.005:
                per.objs[target]["xy"] = slot.copy()     # 置いた場所に同じ番号で付け直す（置いたあとも同じ物として追う）
                per.objs[target]["seen"] = frame_no
                per.held = None
                if held_body:
                    env.hold(held_body, False)
                    events.append({"t": round(sim_t, 2), "placed": held_body})
                held_body = None
                slot_i += 1
                phase = "retreat"
        elif phase == "retreat":
            goal = np.array([slot[0], slot[1], SAFE_Z])
            if np.linalg.norm(tool - goal) < 0.01:
                phase = "decide"
        elif phase == "finished":
            goal = np.array([tool[0], tool[1], SAFE_Z])
        step = goal - cmd
        dist = np.linalg.norm(step)
        cmd = goal.copy() if dist < SPEED * DT else cmd + step / dist * SPEED * DT
        q_cmd = ik.solve(q_cmd, cmd)
        env.d.ctrl[:6] = q_cmd
        t2 = time.perf_counter()
        t_loop.append((t2 - t0) * 1000)
        if writer is not None:
            img = draw_marks(frame, per.visible(frame_no))[:, :, ::-1].copy()
            cv2.putText(img, f"{phase} {('obj' + str(target)) if target else ''}  loop {(t2 - t0) * 1000:.1f}ms", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            writer.write(img)
        frame_no += 1
        if phase == "finished":
            break
        next_t += DT
        wait = next_t - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        else:
            late += 1
            next_t = time.perf_counter()
    if writer is not None:
        writer.release()
    wall = time.perf_counter() - wall0
    truth = {b for b in BLOCKS if in_box(env.block_pos(b)[:2])}
    want = set(a.want.split(",")) if a.want else set()
    res = {"instr": a.instr, "seed": a.seed, "success": truth == want, "in_box": sorted(truth), "want": sorted(want),
           "picks": picks, "finished": phase == "finished", "sim_seconds": round(frame_no * DT, 1),
           "real_fps": round(frame_no / wall, 1), "late_frames": late, "loop_ms_avg": round(float(np.mean(t_loop)), 1),
           "d1_ms_median": round(float(np.median(brain.ms)), 1) if brain.ms else None, "events": events}
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    with open(a.log, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "events"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
