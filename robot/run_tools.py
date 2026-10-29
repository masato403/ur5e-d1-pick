"""ドライバー・ネジ・ペンがばらばらに置かれた作業台から、指示どおりの物を UR5e で箱に運ぶ（学習なし、30FPS）。

  見る係　：何もない作業台の画像と比べて、色か明るさが違う部分を「物」として切り出す（学習なし）
  決める係：d1-3B（Jev と同じ「状況と選択肢を渡すだけ」、追加学習なし）。2 つの聞き方を比べる
    --brain twostep：①画像で「〇番は何？」（選択肢＝部品の一覧）②文章だけで「この指示は〇〇を箱に入れる指示？」
                     → 両方の答えをプログラムで突き合わせて、次に運ぶ物を決める
    --brain onestep：画像で、物 1 つずつに「この物は指示どおりなら箱に入れるべき？」（はい・いいえ）
  動かす係：選ばれた物の位置へ逆運動学で動かし、吸着して箱へ。判断するときは、腕をカメラの視野の外へよける

  python robot/run_tools.py --instr "Put all the screws into the box." --want screw --video robot/tools_screws.mp4
"""
import argparse
import json
import os
import sys
import threading
import time

import cv2
import mujoco
import mujoco.viewer  # noqa: F401（--show のとき使う）
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
os.chdir(os.path.join(HERE, ".."))

from run_pick import IK, SAFE_Z, SPEED  # noqa: E402
from run_pick_som import draw_marks  # noqa: E402
from sim_env import HOME  # noqa: E402
from sim_tools import BIN, BIN_HALF, KIND, OBJECTS, ToolEnv  # noqa: E402

FPS = 30
DT = 1 / FPS
PARTS = ["screwdriver", "screw", "pen"]                 # 作業場の部品の一覧（選択肢になる）
PART_NAME = {"screwdriver": "the screwdriver", "screw": "the screws", "pen": "the pen", "eraser": "the eraser"}
PART_DESC = {"screwdriver": "a screwdriver", "screw": "a screw", "pen": "a pen", "eraser": "an eraser", "other": "something else"}
PARK_Q = HOME + np.array([np.pi / 2, 0, 0, 0, 0, 0])   # 判断するときの腕の姿勢（横へ回して、カメラの視野の外へよける）
JOINT_SPEED = 1.5                                       # 腕をよけるときの関節の速さ（rad/s）
GRIP_Z = 0.034                                          # 吸着するときの工具の先の高さ
WALL_H = 0.03                                           # 箱の壁の高さ（ur5e_tools.xml）
SLOTS = np.array([[0.0, -0.05], [0.0, 0.05], [-0.05, 0.0], [0.05, 0.0]])   # 置く場所（物の中心をここに合わせる）


def in_box(xy):
    return bool(np.all(np.abs(xy - BIN) < BIN_HALF - 0.01))


# ------------------------------------------------------------ 見る係（学習なし）
class SegPerception:
    def __init__(self, ref_rgb, chroma=0.15, sat=70, bright=35, min_area=30):
        self.ref = ref_rgb.astype(np.int16)
        self.ref_c = self._chroma(ref_rgb)
        self.ref_v = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2HSV)[..., 2].astype(np.int16)
        self.chroma, self.sat, self.bright, self.min_area = chroma, sat, bright, min_area
        self.objs, self.next_id, self.held = {}, 1, None
        self.placed = {}        # 置いた直後の物：番号 → (置き場所, このフレームまで)
        self.births = []        # 調査用：新しい番号がついたときの記録

    @staticmethod
    def _chroma(rgb):
        f = rgb.astype(np.float32) + 1.0
        return f / f.sum(2, keepdims=True)

    def foreground(self, rgb):
        """物らしい画素（1）の画像。この版は、何もない机の画像との違いで求める（置き換えられるように分けてある）。"""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        colour = (np.abs(self._chroma(rgb) - self.ref_c).sum(2) > self.chroma) & (hsv[..., 1] > self.sat)
        light = np.abs(hsv[..., 2].astype(np.int16) - self.ref_v) > self.bright      # 色のない金属（ネジなど）は明るさの違いで見つける
        return (colour | light).astype(np.uint8)

    def update(self, rgb, frame_no, robot_mask=None, tool_xy=None):
        m = self.foreground(rgb)
        self.last_fg = m
        # 腕が見えている間は、腕に一部を隠された細長い物が 2 つに分かれて見えることがあるので、新しい番号はつけない
        arm_in_view = robot_mask is not None and int(robot_mask.sum()) > 1000     # 待機の姿勢でも画面の端に少し映るので、少しは許す
        if robot_mask is not None:
            m[robot_mask] = 0
        # 近くにある部分（ドライバーの持ち手と軸など）は 1 つの物としてまとめる。位置は元の画素の重心
        grp = cv2.dilate(m, np.ones((9, 9), np.uint8))
        n, lab = cv2.connectedComponents(grp)
        found = []
        for k in range(1, n):
            ys, xs = np.nonzero((lab == k) & (m > 0))
            if len(xs) < self.min_area:
                continue
            w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            if len(xs) <= 6000 and max(w, h) <= 140:
                found.append((float(xs.mean()), float(ys.mean()), int(w), int(h), (lab == k) & (m > 0)))
        used = set()
        # 番号の付け方：① 運んでいる物は、工具の先の近くに見えた物 ② 置いた直後の物は、置き場所の近くに見えた物
        # ③ それ以外は、いちばん近い既知の物（8cm 以内。いま見えていない物＝隠れていた物を先に当てはめる）
        # ④ どれでもなければ新しい番号（腕が見えていない時だけ）
        rest = []
        for u, v, w, h, fp in found:
            xy = ToolEnv.pixel_to_table(u, v)
            k = None
            if self.held is not None and tool_xy is not None and np.linalg.norm(xy - tool_xy) < 0.10:
                k = self.held
            for pk, (pxy, until) in list(self.placed.items()):
                if k is None and frame_no <= until and np.linalg.norm(xy - pxy) < 0.10 and pk not in used:
                    k = pk
            if k is not None:
                used.add(k)
                self.objs[k] = {"xy": xy, "uv": (u, v, w, h), "seen": frame_no}
            else:
                rest.append((u, v, w, h, xy, fp))
        for u, v, w, h, xy, fp in rest:
            cands = [(np.linalg.norm(o["xy"] - xy), o["seen"] == frame_no, k) for k, o in self.objs.items()
                     if k not in used and k != self.held]
            cands = [c for c in cands if c[0] < 0.08]
            if cands:
                best = min(cands, key=lambda c: (c[1], c[0]))[2]      # 見えていなかった物を優先し、その中で近い物
                if arm_in_view:
                    # 腕に一部を隠された物は、見えている端の位置にずれてしまうので、位置は覚えたまま「見えた」ことだけ残す
                    used.add(best)
                    self.objs[best]["seen"] = frame_no
                    continue
            elif arm_in_view:
                continue                                              # 腕が見えなくなってから、改めて番号をつける
            else:
                best = self.next_id
                self.next_id += 1
                near = sorted((round(float(np.linalg.norm(o["xy"] - xy)), 3), k, frame_no - o["seen"]) for k, o in self.objs.items())
                self.births.append({"frame": frame_no, "id": best, "xy": [round(float(c), 3) for c in xy], "size": [w, h],
                                    "near_known": near[:3], "held": self.held, "used": sorted(used)})
            used.add(best)
            if self.objs.get(best, {}).get("pinned"):
                # 手首のカメラで見に行って分けた物：固定カメラではまだくっついて見えるので、位置は覚えたまま「見えた」ことだけ残す
                self.objs[best]["seen"] = frame_no
                continue
            self.objs[best] = {"xy": xy, "uv": (u, v, w, h), "seen": frame_no}
            if not arm_in_view:
                self.objs[best]["fp"] = fp        # 固定カメラで見えた輪郭（手首で見直した切れ端が、どの物の一部かを確かめるのに使う）
        return [f[:4] for f in found]

    def visible(self, frame_no, within=3):
        # 見に行って分けた物（pinned）は、固定カメラで別々に見えなくても、そこにあるものとして扱う
        return {k: o for k, o in self.objs.items() if frame_no - o["seen"] <= within or o.get("pinned")}


# ------------------------------------------------------------ 決める係（d1-3B、Jev と同じ形）
STATE = ("Camera image of a robot work table seen from above. The yellow square is the box. "
         "White circles with numbers mark the objects.")


def crop(rgb, o, size=128):
    """物のまわりだけを切り出す（番号を書き込むより、見分けの正解率が高かった：98.8% 対 91.2%）。"""
    u, v, w, h = o["uv"]
    half = int(max(w, h) / 2 + 20)
    x1, y1 = max(0, int(u) - half), max(0, int(v) - half)
    return cv2.resize(rgb[y1:int(v) + half, x1:int(u) + half], (size, size))


_MODEL = None                    # 何回も続けて試すとき、d1-3B を読み込み直さない（メモリを節約）


class Brain:
    def __init__(self, instr, mode):
        global _MODEL
        if _MODEL is None:
            from d1_fast import D1Fast
            _MODEL = D1Fast(quant="fp8").model
        self.m = _MODEL
        self.instr, self.mode = instr, mode
        self.result, self.busy, self.ms, self.last = None, False, [], None
        self.what = {}                       # 番号 → 何の物か（twostep：一度見分けた物は聞き直さない）
        self.wanted = None                   # twostep：指示が箱に入れるように言っている部品
        self.what_size, self.reask = {}, False    # 見分けた時の物の大きさ／形が変わったら聞き直すか

    def _img(self, rgb, objs):
        return Image.fromarray(cv2.resize(draw_marks(rgb, objs), (640, 480)))

    def _twostep(self, rgb, objs, table):
        if self.wanted is None:              # ② 文章だけ：この指示はどの部品を箱に入れるように言っているか
            self.wanted = {}
            for p in PARTS:                  # 「入れる／残す」を比べて選ばせる（はい・いいえ より「〜以外」に強かった：14/14 対 8/14）
                o = self.m.system_one(f"Instruction to the robot: {self.instr}",
                                      {"q": {"type": "choice", "instructions": f"Following this instruction, what should the robot do with {PART_NAME[p]}?",
                                             "criteria": {"put": "put it into the box", "leave": "leave it on the table"}}})
                self.wanted[p] = round(o["answers"]["q"]["probabilities"]["put"], 3)
        for k in table:                      # ① 画像：その物だけを切り出して「これは何？」（部品の一覧から選ぶ）
            w, h = objs[k]["uv"][2:]
            if k in self.what:
                # reask：前に聞いた時から形（大きさ）が 3 割以上変わった物は聞き直す
                # （触れていた 2 つの物が 1 つに見えていて、片方を運んだら残りの形が変わった、など）
                w0, h0 = self.what_size.get(k, (w, h))
                if not (self.reask and max(abs(w - w0), abs(h - h0)) > 0.3 * max(w0, h0)):
                    continue
            self.what_size[k] = (w, h)
            o = self.m.system_one("Close-up camera image of one object on a work table.",
                                  {"q": {"type": "choice", "instructions": "What is this object?",
                                         "criteria": {p: PART_DESC[p] for p in PARTS + ["other"]}}},
                                  images=[Image.fromarray(crop(rgb, objs[k]))])
            a = o["answers"]["q"]
            self.what[k] = a["choice"]
        todo = [k for k in table if self.wanted.get(self.what[k], 0) >= 0.5]
        self.last = {"what": {k: self.what[k] for k in table}, "wanted": self.wanted}
        return f"obj{todo[0]}" if todo else "done"

    def _onestep(self, rgb, objs, table, inbox):
        state = STATE + (f" Objects already inside the box: {', '.join(map(str, inbox))}." if inbox else " The box is empty.")
        img = self._img(rgb, objs)
        ans = {}
        for k in table:
            o = self.m.system_one(state, {"q": {"type": "noul", "instructions": f"Instruction: {self.instr} Is object number "
                                                f"{k} one of the objects this instruction asks to put into the box?"}}, images=[img])
            ans[k] = round(o["answers"]["q"]["noul"], 3)
        self.last = {"yes": ans}
        best = max(ans, key=ans.get) if ans else None
        return f"obj{best}" if best is not None and ans[best] >= 0.5 else "done"

    def ask(self, rgb, objs):
        table = sorted(k for k, o in objs.items() if not in_box(o["xy"]))
        inbox = sorted(k for k, o in objs.items() if in_box(o["xy"]))
        self.busy, self.result = True, None

        def work():
            t = time.perf_counter()
            r = self._twostep(rgb, objs, table) if self.mode == "twostep" else self._onestep(rgb, objs, table, inbox)
            self.ms.append((time.perf_counter() - t) * 1000)
            self.result = r
            self.busy = False
        threading.Thread(target=work, daemon=True).start()


def seg_dist(env, g, tool):
    """工具の先（真上から見た位置）と、部品（円柱・カプセル）の軸の線分との距離。
    細長い物は中心から離れた所を吸っても持てるので、中心だけで比べない。"""
    c = env.d.geom_xpos[g][:2]
    ax = env.d.geom_xmat[g].reshape(3, 3)[:2, 2] * env.m.geom_size[g][1]
    t = np.clip(np.dot(tool[:2] - c, ax) / max(np.dot(ax, ax), 1e-9), -1, 1)
    return float(np.linalg.norm(tool[:2] - (c + t * ax)))


# ------------------------------------------------------------ 本体
def topmost_windows(title_part, done=None):
    """題名に title_part を含む窓を、出た時に一度だけ手前に出す（ずっと最前面に固定はしない）。
    いったん最前面にしてから、すぐに固定を外すことで、ほかの窓と同じように後ろへ回せるようにする。
    done（set）を渡すと、一度手前に出した題名は二度目からは何もしない。
    窓の番号（HWND）は 64 ビットなので、関数の引数の型を指定しないと命令が効かない。"""
    if os.name != "nt" or (done is not None and title_part in done):
        return
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    user32.SetWindowPos.restype = wintypes.BOOL
    found = []

    def cb(hwnd, _):
        n = user32.GetWindowTextLengthW(hwnd)
        if n and user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if title_part in buf.value:
                found.append(hwnd)
        return True
    user32.EnumWindows(ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(cb), 0)
    for h in found:
        flags = 0x0001 | 0x0002 | 0x0040                                   # 大きさと位置は変えない、表示する
        user32.SetWindowPos(h, wintypes.HWND(-1), 0, 0, 0, 0, flags)      # いったん最前面にして手前に出す
        user32.SetWindowPos(h, wintypes.HWND(-2), 0, 0, 0, 0, flags)      # すぐに固定を外す（手前に出たまま）
    if found and done is not None:
        done.add(title_part)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--instr", default="Put all the screws into the box.")
    ap.add_argument("--want", default="screw", help="答え合わせ用：箱に入るべき部品（, 区切り）")
    ap.add_argument("--brain", default="twostep", choices=["twostep", "onestep"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--video", default="")
    ap.add_argument("--show", action="store_true", help="ロボットの 3D 表示と、AI が見ている画像を窓に出す")
    ap.add_argument("--log", default="robot/logs/run_tools.jsonl")
    a = ap.parse_args(argv)

    env = ToolEnv()
    env.reset(np.random.default_rng(a.seed))
    per = SegPerception(env.empty_table())
    ik = IK(env)
    # 腕を視野の外（PARK）に置いてから始める
    q_cmd = PARK_Q.copy()
    env.d.qpos[:6] = q_cmd
    env.d.ctrl[:6] = q_cmd
    mujoco.mj_forward(env.m, env.d)
    seg = mujoco.Renderer(env.m, 480, 640)
    seg.enable_segmentation_rendering()
    obj_geoms = set(env.object_geoms())
    robot_geoms = np.array([g for g in range(env.m.ngeom) if g not in obj_geoms and env.m.geom_bodyid[g] != 0])

    def robot_mask():
        seg.update_scene(env.d, camera="top")
        return cv2.dilate(np.isin(seg.render()[..., 0], robot_geoms).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)

    brain = Brain(a.instr, a.brain)
    viewer = None
    raised = set()                                     # 一度手前に出した窓の題名
    if a.show:
        viewer = mujoco.viewer.launch_passive(env.m, env.d, show_left_ui=False, show_right_ui=False)
        viewer.cam.lookat[:] = [-0.05, 0.35, 0.1]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, -60, -35
        cv2.namedWindow("AI view", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI view", 640, 560)
    n_sub = round(DT / env.m.opt.timestep)
    cmd = env.tool_pos()
    goal = cmd.copy()
    phase, target, held, offset, slot_i, slot = "park", None, None, None, 0, BIN
    events, picks = [], []
    writer = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 480)) if a.video else None
    t_loop, late, frame_no = [], 0, 0
    wall0 = time.perf_counter()
    next_t = wall0
    while frame_no < a.seconds * FPS:
        t0 = time.perf_counter()
        for _ in range(n_sub):
            env.d.qfrc_applied[:6] = env.d.qfrc_bias[:6]
            mujoco.mj_step(env.m, env.d)
            if held:
                env.set_obj(held, env.tool_pos()[:2] + offset[:2], env.tool_pos()[2] + offset[2])
        mujoco.mj_forward(env.m, env.d)
        frame = env.render()
        nb = len(per.births)
        per.update(frame, frame_no, robot_mask(), tool_xy=env.tool_pos()[:2] if held else None)
        for b in per.births[nb:]:                               # 調査用：新しい番号がついた瞬間の状況を残す
            bxy = np.array(b["xy"])
            real = min(OBJECTS, key=lambda o: np.linalg.norm(env.obj_pos(o)[:2] - bxy))
            b.update({"phase": phase, "target": target, "carrying": held, "tool": [round(float(c), 3) for c in env.tool_pos()],
                      "real": real, "real_dist": round(float(np.linalg.norm(env.obj_pos(real)[:2] - bxy)), 3),
                      "real_z": round(float(env.obj_pos(real)[2]), 3)})
            if os.environ.get("BIRTHS") and b["frame"] > 0:
                os.makedirs(os.environ["BIRTHS"], exist_ok=True)
                img = draw_marks(frame, per.visible(frame_no))[:, :, ::-1].copy()
                cv2.putText(img, f"NEW {b['id']} {phase}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(os.environ["BIRTHS"], f"s{a.seed}_{a.want.replace(',', '+')}_f{b['frame']}_id{b['id']}.png"), img)
        tool = env.tool_pos()
        sim_t = frame_no * DT
        joint_mode = phase in ("park", "decide", "finished")      # 腕をよけている間は、関節の角度で動かす
        if phase == "park":                                      # 腕を視野の外へよけてから判断する
            if np.abs(env.d.qpos[:6] - PARK_Q).max() < 0.03:
                phase = "decide"
        elif phase == "decide":
            if not brain.busy and brain.result is None:
                brain.ask(frame, per.visible(frame_no))
            elif brain.result is not None:
                ch, brain.result = brain.result, None
                events.append({"t": round(sim_t, 2), "decide": ch, **(brain.last or {})})
                if ch == "done":
                    phase = "finished"
                else:
                    target, phase = int(ch[3:]), "approach"
                    # 腕を近づけると物の一部が隠れて、見える位置（重心）がずれる。腕をよけて見た時の位置で決めておく
                    target_xy = per.objs[target]["xy"].copy()
                    cmd = tool.copy()                            # ここから先は、工具の先の位置で動かす
        elif phase in ("approach", "descend"):
            xy = target_xy
            goal = np.array([xy[0], xy[1], SAFE_Z if phase == "approach" else GRIP_Z])
            if np.linalg.norm(cmd - goal) < 1e-3 and np.linalg.norm(tool - goal) < (0.01 if phase == "approach" else 0.004):
                if phase == "approach":
                    phase = "descend"
                else:
                    # 吸着：工具の先から 2.5cm 以内に部品の一部がある物をつかむ
                    dist = {o: min(seg_dist(env, g, tool) for g in env.geoms[o]) for o in OBJECTS}
                    near = min(dist, key=dist.get)
                    if dist[near] < 0.025 and not in_box(env.obj_pos(near)[:2]):
                        held, offset = near, env.obj_pos(near) - tool
                        env.hold(near, True)
                        per.held = target
                        picks.append(near)
                    phase = "lift"
        elif phase == "lift":
            goal = np.array([tool[0], tool[1], SAFE_Z]) if goal[2] < SAFE_Z else goal
            if np.linalg.norm(tool - goal) < 0.01:
                phase = "carry"
        elif phase == "carry":
            slot = BIN + SLOTS[slot_i % len(SLOTS)]
            sxy = slot - (offset[:2] if held else 0)        # 物の中心が置き場所に来るように、つかんだ位置のずれを引く
            goal = np.array([sxy[0], sxy[1], SAFE_Z])
            if np.linalg.norm(cmd - goal) < 1e-3 and np.linalg.norm(tool - goal) < 0.01:
                phase = "lower"
        elif phase == "lower":
            sxy = slot - (offset[:2] if held else 0)
            # 箱には壁（高さ 3cm）があるので、物の底が壁より上にある高さで放して、箱の中へ落とす
            goal = np.array([sxy[0], sxy[1], max(GRIP_Z + 0.01, WALL_H + 0.02 - (offset[2] if held else 0))])
            if np.linalg.norm(cmd - goal) < 1e-3 and np.linalg.norm(tool - goal) < 0.005:
                if held:
                    env.hold(held, False)
                    events.append({"t": round(sim_t, 2), "placed": held})
                if target in per.objs:
                    per.objs[target]["xy"] = slot.copy()
                    per.objs[target]["seen"] = frame_no
                    per.placed[target] = (slot.copy(), frame_no + 3 * FPS)   # 置いてから 3 秒は、置き場所の近くの物を同じ番号に
                held, per.held = None, None
                slot_i += 1
                phase = "retreat"
        elif phase == "retreat":
            goal = np.array([slot[0], slot[1], SAFE_Z])
            if np.linalg.norm(tool - goal) < 0.01:
                phase = "park"
        if joint_mode or phase == "park":
            dq = PARK_Q - q_cmd
            q_cmd = q_cmd + np.clip(dq, -JOINT_SPEED * DT, JOINT_SPEED * DT)
        else:
            step = goal - cmd
            dist = np.linalg.norm(step)
            cmd = goal.copy() if dist < SPEED * DT else cmd + step / dist * SPEED * DT
            q_cmd = ik.solve(q_cmd, cmd)
        env.d.ctrl[:6] = q_cmd
        t2 = time.perf_counter()
        t_loop.append((t2 - t0) * 1000)
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            show = draw_marks(frame, per.visible(frame_no))[:, :, ::-1].copy()
            panel = np.full((80, 640, 3), 255, np.uint8)
            last = brain.last or {}
            what = last.get("what", {})
            cv2.putText(panel, f"instruction: {a.instr}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(panel, "seen: " + "  ".join(f"{k}={v}" for k, v in what.items()), (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 0, 0), 1)
            put = [p for p, v in last.get("wanted", {}).items() if v >= 0.5]
            cv2.putText(panel, f"to box: {', '.join(put)}   now: {phase} {('#' + str(target)) if target and phase != 'finished' else ''}",
                        (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1)
            cv2.imshow("AI view", np.vstack([show, panel]))
            if frame_no in (0, 15, 60, 150):             # 窓が出た時に一度だけ手前に出す（3D 表示の窓は開くのが少し遅いので、出るまで何度か確かめる）
                topmost_windows("MuJoCo", raised)
                topmost_windows("AI view", raised)
            cv2.waitKey(1)
        if writer is not None:
            img = draw_marks(frame, per.visible(frame_no))[:, :, ::-1].copy()
            cv2.putText(img, f"{phase} {('obj' + str(target)) if target else ''}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            writer.write(img)
        frame_no += 1
        if phase == "finished" and np.abs(env.d.qpos[:6] - PARK_Q).max() < 0.03:
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
    if viewer is not None:
        time.sleep(2)
        viewer.close()
        cv2.destroyAllWindows()
    wall = time.perf_counter() - wall0
    truth = sorted(o for o in OBJECTS if in_box(env.obj_pos(o)[:2]))
    want = sorted(o for o in OBJECTS if KIND[o] in set(a.want.split(",")))
    res = {"brain": a.brain, "instr": a.instr, "seed": a.seed, "success": truth == want, "in_box": truth, "want": want,
           "picks": picks, "finished": phase == "finished", "ids_used": per.next_id - 1, "sim_seconds": round(frame_no * DT, 1),
           "real_fps": round(frame_no / wall, 1), "loop_ms_avg": round(float(np.mean(t_loop)), 1),
           "d1_ms_median": round(float(np.median(brain.ms)), 1) if brain.ms else None, "events": events,
           "births": per.births}
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    with open(a.log, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "events"}, ensure_ascii=False), flush=True)


BATCH = [("Put all the screws into the box.", "screw"),
         ("Put the screwdriver into the box.", "screwdriver"),
         ("Put everything except the pen into the box.", "screwdriver,screw"),
         ("Put the pen and the screwdriver into the box.", "pen,screwdriver")]

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":      # まとめて試す：python robot/run_tools.py --batch 20 21 22
        import gc
        done = set()                                          # 途中で止まったとき、終わった分は飛ばして続きから
        if os.path.exists(os.environ.get("LOG", "robot/logs/run_tools.jsonl")) and os.environ.get("RESUME", "1") == "1":
            for l in open(os.environ.get("LOG", "robot/logs/run_tools.jsonl"), encoding="utf-8"):
                r = json.loads(l)
                done.add((r["brain"], r["instr"], str(r["seed"])))
        for instr, want in BATCH:
            for b in os.environ.get("BRAINS", "twostep,onestep").split(","):
                for seed in sys.argv[2:]:
                    if (b, instr, seed) in done:
                        continue
                    main(["--brain", b, "--instr", instr, "--want", want, "--seed", seed] +
                         (["--log", os.environ["LOG"]] if os.environ.get("LOG") else []))
                    gc.collect()
    else:
        main()
