"""カメラの映像を AI で見ながら、UR5e で積み木を箱に入れる（MuJoCo のシミュレーター上で 30FPS）。

  毎フレーム（33ms ごと）：
    カメラ画像 → YOLO（シミュレーターの画像で学習したもの）で積み木と箱の位置 → 机の上の位置に変換
    → 逆運動学で関節の目標角度を計算 → ロボットへ送る（本物では ur_rtde の servoJ に当たる）
  次にどれを運ぶか：
    d1-3B が「カメラ画像＋指示文＋箱に入っている積み木」から判断する（別スレッドで動かし、ループは止めない）

  python robot/run_pick.py --instr "Put all blocks into the yellow box." --video robot/out.mp4
  python robot/run_pick.py --brain order --order red,green,blue      # d1-3B を使わず決まった順番で
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
os.chdir(os.path.join(HERE, ".."))          # models/ などの場所を jev からの相対で書いているため

from sim_env import BIN, BIN_HALF, BLOCKS, HALF, HOME, Env  # noqa: E402
from yolo_fast import YoloFast  # noqa: E402

FPS = 30
DT = 1 / FPS
SAFE_Z = 0.20            # 運ぶときの高さ
GRIP_GAP = 0.012         # 吸着するときの、積み木の上面と工具の先のすき間
SPEED = 0.45             # 工具の先の最高速度（m/s）
SLOTS = np.array([[-0.035, -0.035], [0.035, -0.035], [-0.035, 0.035], [0.035, 0.035]])   # 箱の中の置き場所
DOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)      # 工具を真下に向けた姿勢


# ------------------------------------------------------------ 逆運動学（工具の先の位置と向き → 関節角度）
class IK:
    def __init__(self, env):
        self.m, self.site = env.m, env.site
        self.d = mujoco.MjData(env.m)
        self.jp = np.zeros((3, self.m.nv))
        self.jr = np.zeros((3, self.m.nv))

    def solve(self, q0, pos, rot=DOWN, iters=8):
        d = self.d
        d.qpos[:6] = q0
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            p = d.site_xpos[self.site]
            R = d.site_xmat[self.site].reshape(3, 3)
            ep = pos - p
            er = 0.5 * (np.cross(R[:, 0], rot[:, 0]) + np.cross(R[:, 1], rot[:, 1]) + np.cross(R[:, 2], rot[:, 2]))
            err = np.concatenate([ep, er])
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            mujoco.mj_jacSite(self.m, d, self.jp, self.jr, self.site)
            J = np.vstack([self.jp[:, :6], self.jr[:, :6]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            d.qpos[:6] += dq
        return d.qpos[:6].copy()


# ------------------------------------------------------------ 次に運ぶ積み木を決める AI
class D1Brain:
    """d1-3B に「カメラ画像（作業台の部分）＋指示＋箱に入っている積み木」を見せて、次に運ぶ積み木を選ばせる。
    文が変わると CUDA グラフの形が変わるので、箱の中身の組み合わせ（8通り）ごとにグラフを作っておく。"""
    CHOICES = {"red": "the red block", "blue": "the blue block", "green": "the green block",
               "finished": "nothing more to do, the instruction is complete"}
    CROP = (148, 68, 492, 412)          # 作業台と箱が入る部分（u1, v1, u2, v2）

    def __init__(self, instr, size=256, d1=None):
        from d1_fast import D1Fast
        from d1_image_fast import D1ImageFast
        self.d1 = d1 or D1Fast(quant="fp8")
        self.size = size
        q = {"type": "choice",
             "instructions": f"Operator instruction: {instr} Which block should the robot pick up and put into the yellow box next?",
             "criteria": self.CHOICES}
        self.heads = {}
        for k in range(8):
            done = tuple(b for i, b in enumerate(BLOCKS) if k >> i & 1)
            state = ("Camera image of the robot work table seen from above. The yellow square is the box. "
                     f"Blocks already in the box: {', '.join(done) if done else 'none'}.")
            self.heads[done] = D1ImageFast(state, q, size=(size, size), quant="fp8", graph="cuda", d1=self.d1)
        self.result, self.busy, self.ms = None, False, []

    def ask(self, frame_rgb, done):
        """別スレッドで判断を始める（答えは self.result に入る）。"""
        u1, v1, u2, v2 = self.CROP
        crop = np.ascontiguousarray(frame_rgb[v1:v2, u1:u2])
        head = self.heads[tuple(b for b in BLOCKS if b in done)]
        self.busy, self.result = True, None

        def work():
            choice, probs, ms = head.decide(crop)
            self.ms.append(ms)
            self.result = (choice, probs)
            self.busy = False
        threading.Thread(target=work, daemon=True).start()


class OrderBrain:
    """AI を使わない比較用：決まった順番で運ぶ。"""
    def __init__(self, order):
        self.order, self.result, self.busy, self.ms = order, None, False, []

    def ask(self, frame_rgb, done):
        left = [b for b in self.order if b not in done]
        self.result = (left[0] if left else "finished", None)


# ------------------------------------------------------------ 見えたもの → 机の上の位置
class Perception:
    def __init__(self, weights):
        self.yolo = YoloFast(weights, size=(640, 480), half=True, graph="cuda", conf=0.4)
        names = self.yolo.names
        self.cls = {names[k].replace("_block", ""): k for k in names}
        self.pos = {}             # 最後に見えた位置（ロボットの腕で隠れたときはこれを使う）
        self.seen = {}

    def update(self, frame_rgb, frame_no):
        boxes, ms = self.yolo.detect(frame_rgb[:, :, ::-1])      # YoloFast は BGR を受け取る
        best = {}
        for x1, y1, x2, y2, s, c in boxes:
            if c not in best or s > best[c][4]:
                best[c] = (x1, y1, x2, y2, s)
        for b in BLOCKS:
            k = self.cls[b]
            if k in best:
                x1, y1, x2, y2, s = best[k]
                w, h = x2 - x1, y2 - y1
                if 14 <= w <= 45 and 14 <= h <= 45:              # 腕に一部隠れた枠は位置がずれるので使わない
                    self.pos[b] = Env.pixel_to_table((x1 + x2) / 2, (y1 + y2) / 2)
                    self.seen[b] = frame_no
        return boxes, ms

    def in_box(self, b):
        return b in self.pos and np.all(np.abs(self.pos[b] - BIN) < BIN_HALF)


# ------------------------------------------------------------ 本体
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instr", default="Put all blocks into the yellow box.")
    ap.add_argument("--brain", default="d1", choices=["d1", "order"])
    ap.add_argument("--order", default="red,green,blue")
    ap.add_argument("--weights", default="robot/blocks_yolo26n.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--video", default="")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--log", default="robot/logs/run_pick.jsonl")
    a = ap.parse_args()

    env = Env()
    env.reset(np.random.default_rng(a.seed))
    ik = IK(env)
    per = Perception(a.weights)
    brain = D1Brain(a.instr) if a.brain == "d1" else OrderBrain(a.order.split(","))
    n_sub = round(DT / env.m.opt.timestep)

    # 準備運動（CUDA グラフの記録など）
    f0 = env.render()
    for _ in range(3):
        per.update(f0, 0)
    q_cmd = HOME.copy()
    goal = env.tool_pos()
    cmd = goal.copy()
    phase, target, held, phase_t, slot = "decide", None, None, 0.0, BIN
    events, done = [], set()
    writer = None
    if a.video:
        writer = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 480))
    t_loop, t_yolo, t_ctrl, late = [], [], [], 0
    wall0 = time.perf_counter()
    next_t = wall0
    frame_no = 0
    while frame_no < a.seconds * FPS:
        t0 = time.perf_counter()
        # 1) ロボットを 1/30 秒動かす（本物ではこの間にロボットが動く）
        for _ in range(n_sub):
            env.d.qfrc_applied[:6] = env.d.qfrc_bias[:6]      # 重力の打ち消し（本物の UR5e は自分で行っている）
            mujoco.mj_step(env.m, env.d)
            if held:
                p = env.tool_pos()
                env.set_block(held, p[:2], p[2] - GRIP_GAP - HALF)
        mujoco.mj_forward(env.m, env.d)
        # 2) カメラ → YOLO
        frame = env.render()
        boxes, ms_yolo = per.update(frame, frame_no)
        t1 = time.perf_counter()
        # 3) 何をするか → 工具の先の目標
        tool = env.tool_pos()
        sim_t = frame_no * DT
        if phase == "decide":
            done = {b for b in BLOCKS if per.in_box(b)}
            if brain.result is None and not brain.busy:
                brain.ask(frame, done)
            elif brain.result is not None:
                choice = brain.result[0]
                events.append({"t": round(sim_t, 2), "decide": choice, "done": sorted(done)})
                brain.result = None
                if choice == "finished":
                    phase = "finished"
                elif choice in per.pos and choice not in done:
                    target, phase, phase_t = choice, "approach", sim_t
                # 箱に入っている物や見えない物を選んだときは、もう一度聞く
            goal = np.array([tool[0], tool[1], SAFE_Z]) if tool[2] < SAFE_Z - 0.02 else goal
        elif phase == "approach":
            xy = per.pos[target]                     # 毎フレーム、見えた位置で目標を直す
            goal = np.array([xy[0], xy[1], SAFE_Z])
            if np.linalg.norm(cmd - goal) < 1e-3 and np.linalg.norm(tool - goal) < 0.01:
                phase = "descend"
        elif phase == "descend":
            xy = per.pos[target]
            goal = np.array([xy[0], xy[1], 2 * HALF + GRIP_GAP])
            if np.linalg.norm(cmd - goal) < 1e-3 and np.linalg.norm(tool - goal) < 0.004:
                held = target                        # 吸着する（シミュレーターでは工具の先にくっつける）
                env.hold(held, True)
                phase = "lift"
        elif phase == "lift":
            goal = np.array([tool[0], tool[1], SAFE_Z]) if goal[2] < SAFE_Z else goal
            if np.linalg.norm(tool - goal) < 0.01:
                phase = "carry"
        elif phase == "carry":
            slot = BIN + SLOTS[len(done) % len(SLOTS)]
            goal = np.array([slot[0], slot[1], SAFE_Z])
            if np.linalg.norm(cmd - goal) < 1e-3 and np.linalg.norm(tool - goal) < 0.01:
                phase = "lower"
        elif phase == "lower":
            goal = np.array([slot[0], slot[1], 2 * HALF + GRIP_GAP + 0.002])
            if np.linalg.norm(cmd - goal) < 1e-3 and np.linalg.norm(tool - goal) < 0.005:
                env.hold(held, False)
                held = None                          # 離す
                events.append({"t": round(sim_t, 2), "placed": target, "took_s": round(sim_t - phase_t, 2)})
                phase = "retreat"
        elif phase == "retreat":
            goal = np.array([slot[0], slot[1], SAFE_Z])
            if np.linalg.norm(tool - goal) < 0.01:
                phase = "decide"
        elif phase == "finished":
            goal = np.array([tool[0], tool[1], SAFE_Z])
        # 速度を制限しながら目標へ近づける → 逆運動学 → 関節の目標角度
        step = goal - cmd
        dist = np.linalg.norm(step)
        cmd = goal.copy() if dist < SPEED * DT else cmd + step / dist * SPEED * DT
        q_cmd = ik.solve(q_cmd, cmd)
        env.d.ctrl[:6] = q_cmd                       # 本物では rtde_c.servoJ(q_cmd, ...)
        t2 = time.perf_counter()
        # 4) 記録と表示
        if writer is not None or a.show:
            img = np.ascontiguousarray(frame[:, :, ::-1])
            for x1, y1, x2, y2, s, c in boxes:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)
                cv2.putText(img, f"{per.yolo.names[int(c)]} {s:.2f}", (int(x1), int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
            cv2.putText(img, f"{phase} {target or ''}  loop {(t2 - t0) * 1000:.1f}ms  yolo {ms_yolo:.1f}ms", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if writer is not None:
                writer.write(img)
            if a.show:
                cv2.imshow("UR5e", img)
                cv2.waitKey(1)
        t_loop.append((t2 - t0) * 1000)
        t_yolo.append(ms_yolo)
        t_ctrl.append((t2 - t1) * 1000)
        frame_no += 1
        if phase == "finished" and not brain.busy:
            break
        # 5) 30FPS に合わせて待つ
        next_t += DT
        wait = next_t - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        else:
            late += 1
            next_t = time.perf_counter()
    wall = time.perf_counter() - wall0
    if writer is not None:
        writer.release()
    truth = {b: bool(np.all(np.abs(env.block_pos(b)[:2] - BIN) < BIN_HALF)) for b in BLOCKS}
    res = {"brain": a.brain, "instr": a.instr if a.brain == "d1" else a.order, "seed": a.seed,
           "frames": frame_no, "real_fps": round(frame_no / wall, 1), "late_frames": late,
           "loop_ms_avg": round(float(np.mean(t_loop)), 2), "loop_ms_p99": round(float(np.percentile(t_loop, 99)), 2),
           "yolo_ms_avg": round(float(np.mean(t_yolo)), 2), "yolo_ms_max": round(float(np.max(t_yolo)), 2),
           "d1_ms": [round(x, 1) for x in brain.ms], "in_box_truth": truth, "sim_seconds": round(frame_no * DT, 1),
           "events": events}
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    with open(a.log, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(res) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
