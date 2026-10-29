"""追いかけてつかむ（10/20 の目標：マットの上で物を動かしても、追いかけて、その場ですぐ取りに行く）の考え方を、シミュレーターで確かめる。
考え方：
  (1) 手首のカメラの 1 枚から物の塊を見つけ（実機と同じ measure.find_objects）、塊の中心の画素を「机から物の高さの半分の面」に写せば、
      腕が動いていても物の位置（根元の座標 mm）が分かる（height_by_motion.on_plane）
  (2) 見えている間は位置と速さを追い（α-β の追い方）、撮ってから使うまでの遅れの分だけ先を予測して狙えば、動く物にも指を合わせられる
  (3) 指先が近づいて物が見えなくなった後は、最後の速さで予測して閉じる
物の動かし方（line：一定の速さでまっすぐ、マットの端で跳ね返る／stop：1 秒動いて止まる／jump：途中で 60mm 置き直す）、物の速さ、
撮る速さ（1 秒に何枚）、遅れ、予測の有無を変えて、追う誤差・閉じた時のずれ・持てたかを数える。腕の動きは毎コマ目標を直す（実機の servoL に当たる）。
閉じ始めたら物を動かすのをやめる（人が手を離す、とみなす）。
使い方:  python robot/track_sim.py [条件ごとの回数]   （条件は下の GRID。結果は robot/logs/track_sim.json と画面の表）
         TRACK_GRID=quick で少しだけ動かす
"""
import base64
import itertools
import json
import math
import contextlib
import io
import os
import sys
import time
import urllib.request

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_grasp as TG  # noqa: E402（撮り方・つかみ方の差し替えを使い回す）
import test_sim_real as T  # noqa: E402
import sim_real as S  # noqa: E402
import sim_rtde  # noqa: E402
from sim_rtde import SimHand, SimRobot, rotvec_to_R, R_to_rotvec  # noqa: E402
import height_by_motion as HM  # noqa: E402
import measure as M  # noqa: E402
import pick_any as PA  # noqa: E402
import pick_by_camera as PB  # noqa: E402
import plan_ai as PL  # noqa: E402（d1-3B：つかむ高さ・やり直し方）
from plan_ai import ENG  # noqa: E402

DT = 0.02                    # 1 コマ（sim_rtde.frame）
VMAX = 0.25                  # 横に追う速さの上限（m/s。実機の PB.SPEED と同じ）
ZSPEED = 0.08                # 下りる・上がる速さ（m/s）
GRIP_MIN = float(os.environ.get("TRACK_GRIP_MIN", "3.0"))   # 下ろす高さの下限（爪の先と机の間 mm。d1-3B の候補 deep 2.0mm もここで止める）
SLOW_ZONE, SLOW_K = 0.015, 0.3                                # 目標の 15mm 上からは、下ろす速さを 3 割にする
GRIP_CLEAR = float(os.environ.get("TRACK_GRIP_CLEAR", "1.5"))   # 閉じる時の、爪と机の間の最小（mm）
FLOOR_MM = float(os.environ.get("TRACK_FLOOR_MM", "1.5"))   # ハンドの一番低い所が机からこれ（mm）より下がったら、上へ逃げる（10/20：ハンドが机に当たっていた）
VLOW = float(os.environ.get("TRACK_VLOW", "0.15"))          # 指先が低い時（LOST_H より下）の横の速さの上限（m/s。速く横へ動くと指が傾いて机をこする）
_HAND_G = None


def hand_low(env):
    """ハンド（本体・指・爪）の一番低い所の、机からの高さ（mm）。実機では TCP の位置と姿勢とハンドの形から同じ値を出せる。"""
    global _HAND_G
    m, d = env.m, env.d
    if _HAND_G is None:
        _HAND_G = [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("g_")
                   and (m.geom_contype[g] | m.geom_conaffinity[g])]
    low = min(d.geom_xpos[g][2] + d.geom_xmat[g].reshape(3, 3)[2] @ m.geom_aabb[g][:3]   # 形の箱の中心は部品の原点からずれている（爪は 3.8mm。10/20 に気づいて直した）
              - np.abs(d.geom_xmat[g].reshape(3, 3)[2]) @ m.geom_aabb[g][3:] for g in _HAND_G)
    return (low - S.MAT_Z) * 1000


ALONG_SPLIT = os.environ.get("TRACK_ALONG", "1") == "1"
ALONG_FRAC = 0.15                                                 # 長い向きのずれは、物の長さの 15%（ペン 140mm → 21mm、消しゴム 43mm → 6.5mm）まで許す
TOL_DOWN = 4.0               # 横のずれがこれ（mm）より小さい時だけ下りる
TOL_CLOSE = 2.5              # 閉じる時の横のずれの上限（mm）
LOST_H = 45.0                # 指先がこれ（mm、机から）より低いと、物が指に隠れるので写真で追わない（予測だけ）
CUT_HAND = os.environ.get("TRACK_CUT_HAND", "1") == "1"   # 0 で前と同じ（指だけで欠けを見る）
CUT_PERP = os.environ.get("TRACK_CUT_PERP", "1") == "1"   # 0 で前と同じ（欠けた写真の中心もそのまま使う）
VEL_MAX = 200.0              # 推定の速さの上限（mm/秒。実験で動かす速さは 100mm/秒まで）
STOP_FIX = os.environ.get("TRACK_STOP_FIX", "1") == "1"   # 物が止まった（何かに当たった）と分かったら、速さをすぐ 0 にする（10/20：止まった物を通り過ぎて腕が行き過ぎた）
STOP_WIN, STOP_MM, FLIP_MM = 0.3, 4.0, 8.0   # 最近これ（秒）の測り値の動きが STOP_MM 以内なら止まった／速さと逆向きに FLIP_MM より外れたら速さを捨てる
ALPHA, BETA = 0.6, 0.3       # α-β の追い方（位置と速さを、見えた位置でどれだけ直すか）
PRED_MAX = float(os.environ.get("TRACK_PRED_MAX", "99"))   # 最後に見えた時から、この秒数より先へは予測を伸ばさない（止まった物を通り過ぎない）
LEN_W = float(os.environ.get("TRACK_LEN_W", "0"))     # 長さの違い 1mm を、位置の違い何 mm とみなすか（0 で長さを見ない）
AVOID = os.environ.get("TRACK_AVOID", "1") == "1"   # 止まっているほかの物の所の塊を追わない。0 で前と同じ
AVOID_R = 35.0               # その物の真ん中からこれ（mm）以内の塊を外す
GATE = 60.0                  # 予測からこれ（mm）より離れた塊は、別の物とみなす
GATE_GROW, GATE_MAX = 150.0, 400.0   # 見えない時間 1 秒ごとに、探す範囲をこれ（mm）広げる（物は 100mm/秒まで動く）・上限（10/20：範囲の外へ出て二度と見つからなかった）


def gate_now(t_unseen):
    return min(GATE_MAX, GATE + GATE_GROW * max(0.0, t_unseen))


LOST_T, GIVEUP_T = 0.6, 6.0     # 高い所から見て、これ（秒）見えなければ見失った（予測を止めて上で待つ）・あきらめる
WRIST_DEAD = 8.0             # 手首の向きの違いがこれ（度）より小さければ回さない
QJUMP = 0.12                 # 1 コマ（0.02 秒）で関節がこれ（rad、約 7°）より動く答えは使わない
_cx, _cy = S.MAT_CENTER
WORK_LO = np.array([(_cx - S.MAT_MM[0] / 2000 + 0.02) * 1000, (_cy - S.MAT_MM[1] / 2000 + 0.02) * 1000])   # 追う範囲（mm）：マットの内側 20mm
WORK_HI = np.array([(_cx + S.MAT_MM[0] / 2000 - 0.02) * 1000, (_cy + S.MAT_MM[1] / 2000 - 0.02) * 1000])
TIMEOUT = 15.0               # 1 回の上限（シミュレーションの秒）
LIFT_BASE = PB.LIFT_SPEED
NAMES = {"eraser": "消しゴム", "r_pen": "ペン", "r_rod": "銀の棒"}
KIND_H = {NAMES[t]: float(S.TOOL_H[t]) for t in NAMES}   # 種類ごとの高さ（mm。物の一覧の値。1 枚で探す時に使う）


# ---- 物を動かす（人が押す・置き直すのまね）
SEP_MM = float(os.environ.get("TRACK_SEP", "90"))
END_FIX = os.environ.get("TRACK_END_FIX", "1") == "1"   # 指に隠れた物は、見えている方の端から長さの半分の所を中心とする（0 で前と同じ。10/20）
MERGE_MM = float(os.environ.get("TRACK_MERGE", "25"))   # 欠けていない塊が追う物よりこれ（mm）長ければ、くっついたとみなす（0 で前と同じ）
START_SEP = os.environ.get("TRACK_START_SEP", "1") == "1"   # 場面を作る時も、動かす物をほかの物から SEP_MM 離して置く（0 で前と同じ）   # 動く物の中心は、ほかの物の中心にこれ（mm）より近づかない


class Mover:
    def __init__(self, env, tool, mode, speed_mm, rng):
        self.env, self.tool, self.mode, self.on = env, tool, mode, True
        a = env.tool_q[tool]
        self.qa = a
        self.dof = int(env.m.jnt_dofadr[env.m.body_jntadr[mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_BODY, tool)]])
        ang = rng.uniform(0, 2 * math.pi)
        self.v = np.array([math.cos(ang), math.sin(ang)]) * speed_mm / 1000.0
        self.t0 = env.d.time
        self.jumped = False
        self.rng = rng
        cx, cy = S.MAT_CENTER
        hx, hy = S.MAT_MM[0] / 2000 - 0.06, S.MAT_MM[1] / 2000 - 0.06
        self.lo, self.hi = np.array([cx - hx, cy - hy]), np.array([cx + hx, cy + hy])
        self.others = [t for t in S.TOOLS if t != tool and np.all(np.abs(env.d.qpos[env.tool_q[t]:env.tool_q[t] + 2] - np.array([cx, cy]))
                                                                   < np.array([hx, hy]) + 0.06)]   # マットの上にあるほかの物
        out = getattr(S, "BOX_OUT", S.BOX_IN)                    # 箱（マットの左の端）には入れない：箱の外形＋物の長さの半分＋余裕の所を壁にする
        self.lo[0] = max(self.lo[0], S.BOX_XY[0] + out[0] / 2 + S.TOOL_LEN.get(tool, 140) / 2000 + 0.02) if hasattr(S, "TOOL_LEN") else             max(self.lo[0], S.BOX_XY[0] + out[0] / 2 + 0.09)

    def step(self, dt):
        if not self.on:
            return
        d, a = self.env.d, self.qa
        t = d.time - self.t0
        if self.mode == "jump":
            if not self.jumped and t > 0.6:
                ang = self.rng.uniform(0, 2 * math.pi)
                p = d.qpos[a:a + 2] + 0.06 * np.array([math.cos(ang), math.sin(ang)])
                d.qpos[a:a + 2] = np.clip(p, self.lo, self.hi)
                self.jumped = True
            return
        if self.mode == "stop" and t > 1.0:
            return
        p = d.qpos[a:a + 2] + self.v * dt
        for i in range(2):                                       # マットの端で跳ね返る
            if p[i] < self.lo[i] or p[i] > self.hi[i]:
                self.v[i] = -self.v[i]
                p[i] = min(max(p[i], self.lo[i]), self.hi[i])
        for t_ in self.others:                                   # ほかの物に近づきすぎたら跳ね返る（10/20、物が近すぎて取れない場面は、まず作らない）
            b = self.env.tool_q[t_]
            q_ = d.qpos[b:b + 2]
            n_ = p - q_
            dist = float(np.linalg.norm(n_))
            if dist < SEP_MM / 1000 and dist > 1e-6:
                n_ /= dist
                if float(self.v @ n_) < 0:
                    self.v = self.v - 2 * float(self.v @ n_) * n_
                p = d.qpos[a:a + 2] + self.v * dt
        p = np.clip(p, self.lo, self.hi)                         # ほかの物から跳ね返った後も、マットの外へは出さない（10/20、消しゴムがマットの外へ行った）
        d.qpos[a:a + 2] = p
        d.qvel[self.dof:self.dof + 6] = 0.0                      # 物理で滑らないように（押されている間は、押す手が位置を決める）

    def true_xy(self):
        return self.env.d.qpos[self.qa:self.qa + 2] * 1000.0


MOVER = None


_LIVE_N = 0


def frame(env):
    """sim_rtde.frame の代わり：物理の 1 刻みごとに物を動かす。"""
    t0 = time.perf_counter()
    for _ in range(sim_rtde.STEPS_PER_FRAME):
        if MOVER is not None:
            MOVER.step(env.m.opt.timestep)
        mujoco.mj_step(env.m, env.d)
    if sim_rtde.VIEWER is not None:
        sim_rtde.VIEWER.sync()
        global _LIVE_N
        _LIVE_N += 1
        if FIXED and _LIVE_N % 2 == 0 and hasattr(sim_rtde.VIEWER, "image"):   # 三脚のカメラの写真は、測っていない時（つかむ・運ぶ）もページに出す（10/20：つかむと止まって見えた）
            sim_rtde.VIEWER.image("三脚のカメラ", env.render_tripod()[0], width=640)
        wait = env.m.opt.timestep * sim_rtde.STEPS_PER_FRAME - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)


# ---- 三脚の固定カメラ（10/20、ざっくり）
FIXED = os.environ.get("TRACK_FIXED", "0") == "1"           # 1 で、三脚のカメラでも測る（SIM_TRIPOD=1 が要る）
ROBOT_BODIES = None
_TRI_N = 0


def robot_bodies(env):
    global ROBOT_BODIES
    if ROBOT_BODIES is None:
        base = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        ROBOT_BODIES = set(np.flatnonzero(env.m.body_rootid == env.m.body_rootid[base]).tolist())
    return ROBOT_BODIES


def tripod_background(env):
    """物の無いマットの写真（実機では、始める前に 1 回撮っておく）。"""
    keep = env.d.qpos.copy()
    for i, t in enumerate(S.TOOLS):
        env.set_tool(t, (1.5, 1.0 + 0.1 * i), 0.0)
    mujoco.mj_forward(env.m, env.d)
    img, _ = env.render_tripod()
    env.d.qpos[:] = keep
    mujoco.mj_forward(env.m, env.d)
    return img


TRI_CUT_AREA = 0.8
TRI_RECT = os.environ.get("TRACK_TRI_RECT", "1")   # 三脚で測る物の中心：0＝重心、1＝囲む四角の真ん中、both＝机の面と上の面に写した形の重なりの重心


def both_center(p0, ph, c_fallback, res=2.0):
    """机の面に写した形 p0 と、物の上の面に写した形 ph（mm）の重なりの重心（mm）。重ならなければ c_fallback。"""
    lo = np.minimum(p0.min(0), ph.min(0)) - 2
    hi = np.maximum(p0.max(0), ph.max(0)) + 2
    W, H = int((hi[0] - lo[0]) * res) + 1, int((hi[1] - lo[1]) * res) + 1
    m0, mh = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
    cv2.fillPoly(m0, [np.round((p0 - lo) * res).astype(np.int32)], 255)
    cv2.fillPoly(mh, [np.round((ph - lo) * res).astype(np.int32)], 255)
    mm = cv2.moments(m0 & mh, binaryImage=True)
    if mm["m00"] < 4:
        return c_fallback
    return lo + np.array([mm["m10"] / mm["m00"], mm["m01"] / mm["m00"]]) / res
_TRI_AREA = 0.0                                                  # 今追っている物を三脚で見た、一番大きい面積（机の上 mm²。回ごとに 0 に戻す）
TRI_OFF = os.environ.get("TRACK_TRI_OFF", "1") == "1"            # 高い所の手首の値で、三脚のずれを覚えて足す（10/20：消しゴムが三脚で 約 4mm 奥にずれ、爪が上に乗った）
TRI_FIRST = os.environ.get("TRACK_TRI_FIRST", "1") == "1"        # 三脚で欠けずに測れている間は、手首の測り値を使わない（手首は下りると高さの誤りでずれる。10/20）


def detect_tripod(env, bg, h_mm, pred, avoid=(), gate=None):
    """三脚のカメラの写真で、物の無いマットとの差の塊のうち、予測に近い物の中心（机の上 mm）と長い向き、欠けたか。
    ロボット（形と姿勢は分かっている）が写っている所は除く。"""
    img, rmask = env.render_tripod(robot_bodies(env))
    global _TRI_N
    _TRI_N += 1
    d = cv2.absdiff(img, bg).max(axis=2)
    fg = (d > 40).astype(np.uint8) * 255
    hx, hy = S.MAT_MM[0] / 2 - 5, S.MAT_MM[1] / 2 - 5
    cx, cy = S.MAT_CENTER[0] * 1000, S.MAT_CENTER[1] * 1000
    poly = env.tripod_px([[cx - hx, cy - hy, S.MAT_Z * 1000], [cx + hx, cy - hy, S.MAT_Z * 1000],
                          [cx + hx, cy + hy, S.MAT_Z * 1000], [cx - hx, cy + hy, S.MAT_Z * 1000]])
    mat = np.zeros_like(fg)
    cv2.fillPoly(mat, [np.round(poly).astype(np.int32)], 255)
    rob = cv2.dilate(rmask, np.ones((7, 7), np.uint8))
    fg = fg & mat & ~rob
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None
    for cnt in cnts:
        if cv2.contourArea(cnt) < 60:
            continue
        pts = env.tripod_on_plane(cnt.reshape(-1, 2), h_mm / 2)[:, :2]
        m_ = cv2.moments(pts.astype(np.float32))
        if m_["m00"] <= 1:
            continue
        c = np.array([m_["m10"] / m_["m00"], m_["m01"] / m_["m00"]])
        if TRI_RECT == "both":                                   # 写った形を机の面と物の上の面に写すと、どちらにも物の底（＝上の面）が入る。重なった所が物の形
            c = both_center(env.tripod_on_plane(cnt.reshape(-1, 2), 0.0)[:, :2], env.tripod_on_plane(cnt.reshape(-1, 2), h_mm)[:, :2], c)
        elif TRI_RECT == "1":                                             # 低い角度から見ると手前の側面も写り、重心は奥へずれる（消しゴム 約 4.7mm。10/20）。
            c = np.array(cv2.minAreaRect(pts.astype(np.float32))[0], float)   # 写った形を囲む四角の真ん中：手前の下の縁と奥の上の縁の真ん中になり、ずれが打ち消し合う
        dd = float(np.hypot(*(c - pred)))
        if dd >= (gate or GATE) or any(np.hypot(*(c - a)) < AVOID_R for a in avoid):
            continue
        if best is None or dd < best[0]:
            best = (dd, cnt, c, pts)
    if best is None:
        return None
    _, cnt, c, pts = best
    (_, _), (w_, h_), ang = cv2.minAreaRect(pts.astype(np.float32))
    a = math.radians(ang if w_ >= h_ else ang + 90)
    lv = np.array([math.cos(a), math.sin(a)])
    m1 = np.zeros_like(fg)
    cv2.drawContours(m1, [cnt], -1, 255, -1)
    x_, y_, w2, h2 = cv2.boundingRect(cnt)
    # 欠けたか：腕に触れているだけでは欠けとしない（低い所から見るので、腕が物の上にあるといつも触れる。10/20：止まったペンの正しい値を捨てていた）。
    # 面積がこれまでの一番大きい面積の TRI_CUT_AREA 倍より小さい時、または写真の縁にかかる時だけ欠けとする
    area = float(cv2.contourArea(pts.astype(np.float32)))
    global _TRI_AREA
    _TRI_AREA = max(_TRI_AREA, area)
    cut = (area < TRI_CUT_AREA * _TRI_AREA
           or x_ <= 2 or y_ <= 2 or x_ + w2 >= img.shape[1] - 2 or y_ + h2 >= img.shape[0] - 2)
    return c, lv, cut


# ---- 写真から物の位置を測る（考え方の (1)）
def detect(img, tcp, h_mm, pred, L0=None, avoid=(), gate=None):
    """予測 pred（mm）に近い塊の、机から h_mm/2 の面の上の中心（mm）と長い辺の向き。無ければ None。
    L0（追う物の長さ mm）があれば、欠けていない塊は長さの違いも点に入れる（ほかの物に乗り移らないように。10/20）。"""
    objs, _ = M.find_objects(img, PA.box_mask(img, tcp))
    best = None
    for cnt in objs:
        mm = cv2.moments(cnt)
        if mm["m00"] < 200:
            continue
        c = HM.on_plane(np.array([[mm["m10"] / mm["m00"], mm["m01"] / mm["m00"]]]), tcp, h_mm / 2)[0]
        d = float(np.hypot(*(c[:2] - pred)))
        if d >= (gate or GATE):
            continue
        if any(np.hypot(*(c[:2] - a)) < AVOID_R for a in avoid):
            continue                                             # 止まっているほかの物（最初の 3 枚で測った）の所の塊は追わない（10/20：ペンから銀の棒に乗り移った）
        score = d
        if L0 is not None:
            x_, y_, w_, h_ = cv2.boundingRect(cnt)
            if not (x_ <= 2 or y_ <= 2 or x_ + w_ >= img.shape[1] - 2 or y_ + h_ >= img.shape[0] - 2):
                bx = cv2.boxPoints(cv2.minAreaRect(cnt))
                pts = HM.on_plane(bx, tcp, h_mm / 2)[:, :2]
                ln = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2]))
                score += LEN_W * abs(ln - L0)
        if best is None or score < best[0]:
            best = (score, cnt, c)
    if best is None:
        return None
    fingers = HM.hand_mask(img) if CUT_HAND else M.finger_mask(img)   # 指だけでなく、画面の下のハンドの本体に隠れた時も「欠け」
    _, cnt, c = best
    box = cv2.boxPoints(cv2.minAreaRect(cnt))
    e = sorted(range(4), key=lambda i: np.linalg.norm(box[i] - box[(i + 1) % 4]))[-1]   # 長い辺
    ends = HM.on_plane(np.array([(box[e] + box[(e + 3) % 4]) / 2, (box[(e + 1) % 4] + box[(e + 2) % 4]) / 2]), tcp, h_mm / 2)
    lv = ends[1][:2] - ends[0][:2]
    lv = lv / max(np.linalg.norm(lv), 1e-9)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    x_, y_, w_, h_ = cv2.boundingRect(cnt)
    cut = (cv2.countNonZero(cv2.dilate(mask, np.ones((9, 9), np.uint8)) & fingers) > 0
           or x_ <= 2 or y_ <= 2 or x_ + w_ >= img.shape[1] - 2 or y_ + h_ >= img.shape[0] - 2)
    merged = False
    if MERGE_MM > 0 and L0 is not None and not cut and float(np.linalg.norm(ends[1] - ends[0])) > L0 + MERGE_MM:
        cut = merged = True                                      # 追う物より長い塊＝近くの物とくっついた：中心がずれるので「欠け」と同じ扱い（10/20：ペンが 35mm 隣の物とくっつき、推定が遅れた）
    c_end = None
    if END_FIX and cut and not merged and L0 is not None:
        # 欠けた塊の両端のうち、指・写真の縁に触れていない方は本当の端：そこから長さの半分だけ内側を中心とする
        # （10/20：速い銀の棒が指の下で跳ね返り、長い向きの位置が分からず 98mm ずれて閉じた）
        fd = cv2.dilate(fingers, np.ones((15, 15), np.uint8))
        pe = [(box[e] + box[(e + 3) % 4]) / 2, (box[(e + 1) % 4] + box[(e + 2) % 4]) / 2]
        H_, W_ = img.shape[:2]
        def hit(q):
            x0, y0 = int(round(q[0])), int(round(q[1]))
            if x0 <= 4 or y0 <= 4 or x0 >= W_ - 5 or y0 >= H_ - 5:
                return True
            return fd[y0, x0] > 0
        h0, h1 = hit(pe[0]), hit(pe[1])
        if h0 != h1:
            free, other = (ends[1][:2], ends[0][:2]) if h0 else (ends[0][:2], ends[1][:2])
            u = (free - other) / max(np.linalg.norm(free - other), 1e-9)
            c_end = free - u * L0 / 2
    return c[:2], lv, cut, c_end


# ---- 腕の目標の向き（pick_by_camera.grasp と同じ：指を結ぶ向きを長い辺と直角に）
def grip_R(R_now, longv):
    cur_y = R_now @ np.array(PB.GEOM["close_dir_tool"])
    want = np.array([-longv[1], longv[0], 0.0])
    if np.dot(want, cur_y) < 0:
        want = -want
    a_ = math.atan2(np.cross(cur_y, want)[2], np.dot(cur_y, want))
    Rz = np.array([[math.cos(a_), -math.sin(a_), 0], [math.sin(a_), math.cos(a_), 0], [0, 0, 1]])
    return Rz @ R_now


def tool_target(xy_mm, R):
    """物の中心 xy（mm）を狙う時の、道具の点の xy（mm）。実機のハンドの横のずれを補正（pick_by_camera.grasp と同じ）。"""
    bias = R @ np.array(PB.GEOM["bias_tool_mm"])
    return np.array([xy_mm[0] + PB.BIAS_SIGN * bias[0], xy_mm[1] + PB.BIAS_SIGN * bias[1]])


def grip_height(tool):
    """指先を下ろす高さ（mm、机から）。pick_any の決まり（高さの半分、丸い物は 0.3 倍）に近い値。"""
    h = S.TOOL_H[tool]
    return max(PA.ROUND_MIN, h * 0.3) if tool != "eraser" else min(max(h / 2, PA.GRIP_MIN), PA.GRIP_MAX)


# ---- 1 回
SURVEY3 = os.environ.get("TRACK_SURVEY", "1") == "1"   # 0：最初に 3 枚撮らず、1 枚で探す。高さは d1-3B が決めた種類ごとの値（10/20、3 枚なしの版）
H_GUESS = 10.0                                          # 1 枚の時、種類が決まるまでの仮の高さ（mm）


def search_one(env, robot, obs):
    """見張りの位置で 1 枚だけ撮り、塊を探して d1-3B に種類を聞く。高さは横へずらした写真が無いので測らず、種類ごとの値にする。"""
    t0, w0 = env.d.time, time.perf_counter()
    pose = robot.getActualTCPPose()
    robot.moveL([obs[0], obs[1], S.MAT_Z + HM.OBS / 1000] + list(pose[3:]), PB.SPEED, PB.ACCEL)
    for _ in range(int(0.2 / DT)):                               # 止まって揺れが収まるのを少し待つ
        frame(env)
    img, tcp = env.render_real(), env.tcp_like()
    t_move = env.d.time - t0
    if hasattr(sim_rtde.VIEWER, "image"):
        sim_rtde.VIEWER.image("撮った写真（1 枚）", img)
    w1 = time.perf_counter()
    keep = (HM.best_height, HM.best_height_multi, PA.part_heights)
    fake = lambda *a, **k: (H_GUESS, None, np.ones(1))          # noqa: E731（高さは測らない）
    HM.best_height, HM.best_height_multi, PA.part_heights = fake, fake, (lambda *a, **k: [])
    try:
        found = PA.survey(img, tcp, [(img, tcp)])
    finally:
        HM.best_height, HM.best_height_multi, PA.part_heights = keep
    TG.KA.decide(img, found, TG.kind_of, want=lambda f: f["on_mat"] >= PA.MAT_RING)
    for f in found:
        if f.get("kind") in KIND_H:
            f["h"] = KIND_H[f["kind"]]
    t_calc = time.perf_counter() - w1
    print(f"   1 枚で探した：{t_move + t_calc:.1f} 秒（腕の動きと待ち {t_move:.1f} 秒、計算 {t_calc:.1f} 秒）", flush=True)
    return img, tcp, [], found


ALONE = os.environ.get("TRACK_ALONE", "0") == "1"   # 追う物だけをマットに置く（d1-3B を使う時も。10/20、まずペン 1 本で）
CHAIN = os.environ.get("TRACK_CHAIN", "1") == "1"   # 箱に入れたら場面をリセットせず、残りの物でそのまま次へ（10/20）
_PLACED, KEPT = None, False                                              # 続けている場面で、マットに置いた物


def trial(env, robot, hand, rng, tool, mode, speed, fps, lat, pred_on, keep=False):
    global MOVER, _PLACED, KEPT
    KEPT = bool(keep and _PLACED is not None and tool in _PLACED and not TG.in_box(env, tool)[0])
    if KEPT:
        placed = _PLACED                                         # 前の物を箱に入れた場面のまま続ける（残りの物はそのまま）
    else:
        env.reset(rng)
        for i, o in enumerate(env.objects):
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        TG.go_to(env, robot.ik, np.array(S.OBS_XY))
        placed = _PLACED = T.place_tools(env, rng)
        for k_ in range(60 if START_SEP else 0):                 # 動かす物がほかの物に近すぎる場面は作らない（10/20）
            full = len(placed) == len(S.TOOLS)                   # 置き直しで物がマットの外に出たままにならないように、全部置けた場面だけ採る（10/20、銀の棒がマットの外に出た）
            sep_ok = tool in placed and all(np.hypot(*(placed[tool][0] - placed[u][0])) * 1000 >= SEP_MM for u in placed if u != tool)
            if full and (sep_ok or k_ >= 30):
                break
            with contextlib.redirect_stdout(io.StringIO()):     # 置き直しの途中の「置ける所が無い」は出さない
                placed = _PLACED = T.place_tools(env, rng)
    if not D1 or ALONE:
        for t in S.TOOLS:                                        # 追う物だけマットの上に残し、ほかはどける（d1-3B を使わない時。追うことだけ確かめる）
            if t != tool:
                env.set_tool(t, (1.5, -0.5 + 0.1 * S.TOOLS.index(t)), 0.0)
    mujoco.mj_forward(env.m, env.d)
    hand.env.set_finger(False)
    for _ in range(40):
        frame(env)
    robot.moveL(list(S.OBS_XY) + [S.MAT_Z + HM.OBS / 1000] + list(robot.getActualTCPPose()[3:]), PB.SPEED, PB.ACCEL)
    want, f, kind_ok = tool, None, None
    if tool not in placed:                                       # 追う物がマットに置けなかった（マットの外にある）：この場面は飛ばす
        return {"tool": tool, "mode": mode, "speed": speed, "fps": fps, "lat": lat, "pred": bool(pred_on), "held": False, "boxed": False,
                "why": "置けなかったので飛ばした", "kind_ok": None, "track_err": None, "aim_err": None, "t": 0.0, "n_ask": 0, "skip": True}
    if D1:                                                       # 実機の通しと同じ：3 枚撮って測り、種類は d1-3B が決める。頼まれた種類の物を追う
        imgS, tcpS, _, found = (TG.search if SURVEY3 else search_one)(env, robot, S.OBS_XY)
        cands = [g for g in found if g["kind"] == NAMES[want]
                 and not PA.why_not(g, math.hypot(g["center"][0] - tcpS[0] * 1000, g["center"][1] - tcpS[1] * 1000))]
        if not cands:                                            # 診断：何を見て、なぜ選ばなかったか（10/20：2 個目のペンが「それ以外」になった）
            for g in found:
                c_ = np.array(g["center"][:2], float)
                near = min(S.TOOLS, key=lambda t: np.hypot(*(env.d.qpos[env.tool_q[t]:env.tool_q[t] + 2] * 1000 - c_)))
                d_ = float(np.hypot(*(env.d.qpos[env.tool_q[near]:env.tool_q[near] + 2] * 1000 - c_)))
                print(f"   見つけた物：{g.get('kind')}（本当は {NAMES[near]}、中心のずれ {d_:.0f}mm）大きさ {np.round(g.get('size', []), 1)} 高さ {g.get('h')} "
                      f"マットの上 {g.get('on_mat')} 外す理由 {PA.why_not(g, math.hypot(g['center'][0] - tcpS[0] * 1000, g['center'][1] - tcpS[1] * 1000))}", flush=True)
            tr_ = env.d.qpos[env.tool_q[want]:env.tool_q[want] + 2] * 1000
            print(f"   本物の {NAMES[want]} の位置 {np.round(tr_)}", flush=True)
            cv2.imwrite(os.path.join(HERE, "logs", f"notfound_{want}_{int(time.time())}.jpg"), imgS)
            return {"tool": want, "mode": mode, "speed": speed, "fps": fps, "lat": lat, "pred": bool(pred_on), "held": False, "boxed": False,
                    "why": f"d1-3B が {NAMES[want]} を見つけられない", "kind_ok": False, "track_err": None, "aim_err": None, "t": 0.0, "n_ask": 0}
        f = max(cands, key=lambda g: g["match"])
        tool = TG.nearest_tool(env, f)                           # d1-3B が選んだ物（違う物を選んだら、それを追う。種類の間違いとして数える）
        kind_ok = tool == want
    h_mm = float(f["h"]) if f is not None else S.TOOL_H[tool]
    drop, empty = 0.0, False
    grip = PL.grip(f, drop, empty) if f is not None else grip_height(tool)   # 指先を下ろす高さは d1-3B が候補から選ぶ
    MOVER = Mover(env, tool, mode, speed, rng)
    every = max(1, int(round(1.0 / (fps * DT))))
    L0 = float(max(f["size"])) if (f is not None and LEN_W > 0) else None   # 3 枚で測った長さ
    avoid = [np.array(g["center"][:2], float) for g in found if g is not f and g.get("kind")] if (f is not None and AVOID) else []
    if f is not None:
        est = np.array(f["center"][:2], float)                   # 最初の位置は 3 枚で測った値（物は撮った後から動き出す）
        longv = np.array(f["longv"][:2], float) / max(np.linalg.norm(f["longv"][:2]), 1e-9)
    else:
        est = MOVER.true_xy().copy()                             # d1-3B を使わない時：位置だけ本物から
        longv = None
    vel = np.zeros(2)
    recent = []                   # 止まったかを見るための、最近の欠けていない測り値（時刻・位置）
    t_est = env.d.time
    queue = []                                                   # (使える時刻, 撮った時刻, 測った位置, 向き, 欠け)
    errs, det_ms, n_img, n_cut, n_miss = [], [], 0, 0, 0
    n_tri = 0                                                    # 三脚のカメラで測れた回数
    bg_t = tripod_background(env) if FIXED else None
    global _TRI_AREA
    _TRI_AREA, tri_ok_t = 0.0, -9.0
    wrist_ok, tri_off = None, None                               # 高い所の手首で最後に正しく測れた (時刻, 位置)・三脚のずれ（手首 − 三脚 mm）
    detail = []                                                  # 測るごとの（誤差 mm, 欠け, いちばん近いほかの物 mm, 指先の高さ mm, 長い向きと直角の誤差 mm）
    q = env.d.qpos[:6].copy()
    lead = (-env.m.actuator_biasprm[:6, 2] / env.m.actuator_gainprm[:6, 0]) if sim_rtde.FEEDFWD else np.zeros(6)
    p_cmd = np.array(env.d.site_xpos[env.site])
    R_cmd = env.d.site_xmat[env.site].reshape(3, 3).copy()
    z_hover = S.MAT_Z + PB.HOVER / 1000
    n_ask = 0
    L_obj = float(max(f["size"])) if f is not None else 40.0
    along_tol = max(TOL_DOWN, ALONG_FRAC * L_obj)                 # 長い向きに許すずれ（mm）：物の長さの ALONG_FRAC
    for attempt in range(1 + (PA.RETRY if D1 else 0)):
        z_grip = S.MAT_Z + max(grip, GRIP_MIN) / 1000           # 爪の先がこれより机に近づく高さへは、そもそも下ろさない（10/20、下ろしすぎると壊れる）
        p_cmd = np.array(env.d.site_xpos[env.site])
        q = env.d.qpos[:6].copy()
        k, t0 = 0, env.d.time
        last_seen, n_jump, why_wrong = env.d.time, 0, ""
        min_clr, n_floor = 999.0, 0
        up_until, n_no = (-1 if attempt == 0 else int(0.6 / DT)), 0   # やり直しは、一度上がって見直してから
        why = ""
        while True:
            if env.d.time - t0 > TIMEOUT:
                why = "時間切れ"
                break
            tip_h = (env.d.site_xpos[env.site][2] - S.MAT_Z) * 1000
            # 撮る（決まった間隔ごと。指先が低い時は、物が指に隠れるので撮らない）
            if k % every == 0 and tip_h > LOST_H:
                img = env.render_real()
                tcp = env.tcp_like()
                t_img = env.d.time
                w0 = time.perf_counter()
                p_pred = est + (vel * (t_img - t_est) if pred_on else 0)
                r = detect(img, tcp, h_mm, p_pred, L0, avoid, gate_now(env.d.time - last_seen))
                det_ms.append((time.perf_counter() - w0) * 1000)
                n_img += 1
                true_at_img = MOVER.true_xy().copy()
                if r is None:
                    n_miss += 1
                else:
                    last_seen = env.d.time
                    errs.append(float(np.hypot(*(r[0] - true_at_img))))
                    others = [np.hypot(*(env.d.qpos[env.tool_q[t]:env.tool_q[t] + 2] * 1000 - true_at_img)) for t in S.TOOLS if t != tool]
                    perp = np.array([-longv[1], longv[0]]) if longv is not None else np.zeros(2)
                    detail.append((round(errs[-1], 1), int(r[2]), round(float(min(others)), 0), round(tip_h), round(abs(float(perp @ (r[0] - true_at_img))), 1)))
                    n_cut += r[2]
                    high = tip_h > PB.HOVER - 5                     # 下り始める前（真上に近い所から、決まった高さで見ている）の手首の値だけを正とする
                    if r[3] is not None:                         # 見えている端から出した中心：長い向きも使える測り値として入れる
                        queue.append((t_img + lat, t_img, r[3], r[1], False, "wrist_end"))
                    else:
                        queue.append((t_img + lat, t_img, r[0], r[1], r[2], "wrist" if high else "wrist_low"))
                    if high and not r[2]:
                        wrist_ok = (t_img, r[0].copy())
            if FIXED and k % every == 0:                         # 三脚の固定カメラ：指が下りていても、横から見える（10/20）
                t_img = env.d.time
                w0 = time.perf_counter()
                p_pred = est + (vel * (t_img - t_est) if pred_on else 0)
                r = detect_tripod(env, bg_t, h_mm, p_pred, avoid, gate_now(env.d.time - last_seen))
                det_ms.append((time.perf_counter() - w0) * 1000)
                n_img += 1
                if r is None:
                    n_miss += 1
                else:
                    last_seen = env.d.time
                    true_at_img = MOVER.true_xy().copy()
                    errs.append(float(np.hypot(*(r[0] - true_at_img))))
                    n_tri += 1
                    n_cut += r[2]
                    zt = r[0].copy()
                    if TRI_OFF and wrist_ok is not None and wrist_ok[0] == t_img and not r[2]:   # 同じ時に手首でも正しく測れた：三脚のずれを覚える
                        d_ = wrist_ok[1] - zt
                        tri_off = d_ if tri_off is None else 0.7 * tri_off + 0.3 * d_
                    if TRI_OFF and tri_off is not None:
                        zt = zt + tri_off                            # 三脚の値に、覚えたずれを足す（低い角度から見た形の取り方のずれ。物ごとにほぼ一定）
                    queue.append((t_img + lat, t_img, zt, r[1], r[2], "tripod"))
                    if not r[2]:
                        tri_ok_t = t_img
                    queue.sort(key=lambda e: e[0])
            # 遅れて届いた測り値で、位置と速さを直す（考え方の (2)）
            while queue and queue[0][0] <= env.d.time:
                _, t_img, z, lv, cut, src = queue.pop(0)
                if TRI_FIRST and src == "wrist_low" and t_img - tri_ok_t < 0.15:
                    continue                                         # 下りている時の手首の値は、高さの誤りでずれる：三脚で欠けずに測れていれば使わない
                if TRI_OFF and src == "tripod" and wrist_ok is not None and t_img - wrist_ok[0] < 0.15:
                    continue                                         # 高い所の手首で測れている間は、手首の値を使う（三脚はずれを覚えるだけ）
                dt_ = t_img - t_est
                if dt_ < 0.02:                                       # 同じ時刻（最初の 1 枚など）：位置だけ合わせる。速さを割り算で出すと大きく狂う
                    est = 0.5 * (est + z)
                else:
                    p_ = est + vel * dt_
                    res = z - p_
                    if cut and CUT_PERP and longv is not None:      # 欠けた写真の中心は長い向きにずれる：長い向きと直角の分だけ使う
                        nv = np.array([-longv[1], longv[0]])
                        res = nv * float(nv @ res)
                    est = p_ + ALPHA * res
                    if STOP_FIX and pred_on and float(res @ vel) < 0 and np.linalg.norm(res) > FLIP_MM:
                        vel = np.zeros(2)                            # 予測より手前で止まった・跳ね返った：古い速さで先へ行かない
                    vel = vel + BETA * res / dt_ if pred_on else np.zeros(2)
                    if STOP_FIX and not cut:
                        recent.append((t_img, z.copy()))
                        recent[:] = [e for e in recent if t_img - e[0] <= STOP_WIN]
                        if len(recent) >= 3 and recent[-1][0] - recent[0][0] > 0.6 * STOP_WIN and                                 max(np.hypot(*(e[1] - z)) for e in recent) < STOP_MM:
                            vel = np.zeros(2)                        # 最近の測り値がほとんど動いていない＝止まっている
                            est = z.copy()
                    sp = np.linalg.norm(vel)
                    if sp > VEL_MAX:                                 # 速さの上限（測り違いで暴れない）
                        vel *= VEL_MAX / sp
                t_est = t_img
                if TRACE:
                    print(f"    測り値 {np.round(z)}{' 欠け' if cut else ''}{' 端から' if src == 'wrist_end' else ''}（撮った {t_img - t0:.2f}秒） 推定 {np.round(est)} 速さ {np.round(vel)}mm/秒", flush=True)
                if not cut:
                    longv = lv
            lost = tip_h > LOST_H and env.d.time - last_seen > LOST_T   # 高い所から見ているのに、しばらく見えない＝見失った
            if lost:
                vel = np.zeros(2)                                    # 予測で先へ進まない（腕が暴走しないように。10/20）
                if env.d.time - last_seen > GIVEUP_T:
                    tr_ = MOVER.true_xy()
                    img_ = env.render_real()
                    px_ = HM.project_fast(np.array([[tr_[0], tr_[1], S.MAT_Z * 1000 + h_mm / 2]]), env.tcp_like())[0]
                    inview = 0 <= px_[0] < img_.shape[1] and 0 <= px_[1] < img_.shape[0]
                    near_o = min(np.hypot(*(np.array(a) - tr_)) for a in avoid) if avoid else 999
                    why = (f"見失った（本物は写真の{'中' if inview else '外'}・推定とのずれ {np.hypot(*(est - tr_)):.0f}mm・"
                           f"ほかの物まで {near_o:.0f}mm）")
                    cv2.imwrite(os.path.join(HERE, "logs", f"lost_{tool}_{mode}_{speed}.jpg"), img_)
                    break
            now_xy = est + (vel * min(env.d.time - t_est, PRED_MAX) if pred_on else 0)   # 見えない間（指が低い時）は先へ伸ばしすぎない
            now_xy = np.clip(now_xy, WORK_LO, WORK_HI)               # 作業の範囲（マット＋余裕）の外へは追わない
            if longv is None:
                longv = np.array([1.0, 0.0])
            R_goal = grip_R(R_cmd, longv) if tip_h > LOST_H else R_cmd   # 低い所では向きを変えない
            if np.linalg.norm(R_to_rotvec(R_goal @ R_cmd.T)) < math.radians(WRIST_DEAD):
                R_goal = R_cmd                                       # 小さな向きの揺れでは手首を回さない（グネグネしない）
            goal = tool_target(now_xy, R_goal) / 1000
            dxy = goal - p_cmd[:2]
            dist = np.linalg.norm(dxy)
            step = min(dist, (VMAX if tip_h > LOST_H else VLOW) * DT)
            p_cmd[:2] += dxy / max(dist, 1e-9) * step
            e_ = (goal - p_cmd[:2]) * 1000
            if ALONG_SPLIT and longv is not None:                # 細長い物は長い向きに多少ずれても掴める：直角の向き（指が閉じる向き）だけ厳しく見る（10/20、つかめるのに見逃していた）
                e_al = abs(float(e_ @ longv))
                err_mm = max(abs(float(e_ @ np.array([-longv[1], longv[0]]))), e_al * TOL_DOWN / along_tol)
            else:
                err_mm = float(np.linalg.norm(e_))
            z_goal = z_grip if err_mm < TOL_DOWN and k >= up_until and not lost else z_hover
            vz = ZSPEED if p_cmd[2] - z_goal > SLOW_ZONE else ZSPEED * SLOW_K   # 下の方では、ゆっくり下ろす（勢いで行き過ぎて机に当たらない）
            p_cmd[2] += float(np.clip(z_goal - p_cmd[2], -vz * DT, ZSPEED * DT))
            rv = R_to_rotvec(R_goal @ R_cmd.T)
            ang = np.linalg.norm(rv)
            if ang > 1e-6:
                R_cmd = rotvec_to_R(rv / ang * min(ang, math.radians(120) * DT)) @ R_cmd
            q_prev = q
            q_new = robot.ik.solve(q, p_cmd, R_cmd, iters=8)
            if np.max(np.abs(q_new - q_prev)) > QJUMP:               # 関節が 1 コマで大きく飛ぶ答え（姿勢の入れ替わり）は使わず、止まる
                n_jump += 1
                if n_jump > 5:
                    why = "腕の計算が飛んだので止めた"
                    break
                q_new = q_prev
            q = q_new
            env.d.ctrl[:6] = q + lead * (q - q_prev) / DT
            frame(env)
            k += 1
            clr = hand_low(env)
            min_clr = min(min_clr, clr)
            if clr < FLOOR_MM and not abs(p_cmd[2] - z_grip) < 0.0005:   # 下りる途中で机に近すぎる（傾き・遅れ）：少し上がってから下り直す
                up_until = k + int(0.3 / DT)
                n_floor += 1
            if TRACE and k % 10 == 0:
                tip = env.d.site_xpos[env.site] * 1000
                print(f"    t {env.d.time - t0:5.2f} 指令 {np.round(p_cmd * 1000)} 本当 {np.round(tip)} 推定 {np.round(est)} 本物 {np.round(MOVER.true_xy())}"
                      f" 関節 {np.round(np.degrees(q)).astype(int)}"
                      f" 物の軸 {math.degrees(math.atan2(longv[1], longv[0])):.0f}° 手首 {math.degrees(env.tool_yaw()):.0f}°"
                      f" 物の向き {math.degrees(2 * math.atan2(env.d.qpos[env.tool_q[tool] + 6], env.d.qpos[env.tool_q[tool] + 3])):.0f}°", flush=True)
            if robot.hand_force() > sim_rtde.STOP_FORCE:
                f_, parts = robot.hand_force(detail=True)
                why = "保護停止（" + "・".join(f"{a}→{b} {f:.0f}N" for f, a, b in parts[:2]) + f"、指先 {tip_h:.1f}mm）"
                break
            if abs(p_cmd[2] - z_grip) < 0.0005 and err_mm < TOL_CLOSE:
                if not D1_CLOSE:
                    break
                if n_no >= CLOSE_NO_MAX:
                    why = "d1-3B が閉じないと答え続けた"
                    break
                yes, ms_ = ask_close(env.render_real(), NAMES[tool])  # 閉じてよいかは d1-3B が決める
                n_ask += 1
                for _ in range(max(1, int(math.ceil(ms_ / 1000 / DT)))):   # 答えを待つ間も物は動く（腕は止めたまま）
                    frame(env)
                if yes:
                    break
                n_no += 1
                up_until = k + int(0.4 / DT)                         # 「まだ」なら少し上がって追い直す
                print(f"   d1-3B：まだ閉じない（{n_no} 回目）", flush=True)
        # 閉じる（考え方の (3)：ここまで予測で狙った）
        # 腕を今の位置で止める。最後のコマの「動きの先回り」（lead × 速さ）が指令に残ると、下りる向きに押し続け、
        # 閉じる間に爪が机に押し付けられた（爪 −1.3mm、摩擦で閉じるのに毎回 3 秒。10/20）
        env.d.ctrl[:6] = q
        for _ in range(3):                                      # 爪と机の間を GRIP_CLEAR mm 以上あけてから閉じる（机に押し付けると壊れるおそれ・保護停止。10/20）
            clr = hand_low(env)
            if clr >= GRIP_CLEAR:
                break
            p_cmd[2] += (GRIP_CLEAR - clr + 0.3) / 1000
            q = robot.ik.solve(q, p_cmd, R_cmd, iters=8)
            env.d.ctrl[:6] = q
            for _ in range(10):
                frame(env)
        clr_close = hand_low(env)
        true_close = MOVER.true_xy().copy()
        pr, pl = env.d.geom_xpos[env.pads[0]], env.d.geom_xpos[env.pads[1]]
        ax_c = (pl - pr)[:2] / max(np.linalg.norm((pl - pr)[:2]), 1e-9)        # 指が閉じる向き
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, env.d.qpos[env.tool_q[tool] + 3:env.tool_q[tool] + 7])
        ax_o = Rq.reshape(3, 3)[:2, 0] / max(np.linalg.norm(Rq.reshape(3, 3)[:2, 0]), 1e-9)   # 物の長い向き
        dmid = ((pr + pl) / 2)[:2] * 1000 - env.d.qpos[env.tool_q[tool]:env.tool_q[tool] + 2] * 1000
        close_diag = {"across": round(float(dmid @ ax_c), 1), "along": round(float(dmid @ ax_o), 1),
                      "yaw": round(math.degrees(math.asin(min(1.0, abs(float(ax_c @ ax_o))))), 1), "open": round(hand.pos_mm(), 1)}
        print(f"   閉じる前：指の向きのずれ {close_diag['across']:+.1f}mm・物の長い向きのずれ {close_diag['along']:+.1f}mm・"
              f"指と物の向きの違い {close_diag['yaw']:.1f}°・開き {close_diag['open']}mm", flush=True)
        aim_close = now_xy.copy()
        MOVER.on = False
        held, closed = False, None
        if not why:
            try:
                ph = [env.d.time, time.perf_counter()]
                hand.grip(100 if tool != "eraser" else 60)
                closed = hand.pos_mm()
                ph += [env.d.time, time.perf_counter()]
                robot.moveL(list(env.d.site_xpos[env.site][:2]) + [z_hover] + list(R_to_rotvec(R_cmd)), PB.LIFT_SPEED, PB.HOLD_ACCEL)
                ph += [env.d.time, time.perf_counter()]
                print(f"   閉じる {ph[2] - ph[0]:.2f}秒（本当の時間 {ph[3] - ph[1]:.2f}秒）・持ち上げる {ph[4] - ph[2]:.2f}秒（{ph[5] - ph[3]:.2f}秒）", flush=True)
                obj_z = (env.d.qpos[env.tool_q[tool] + 2] - S.MAT_Z) * 1000
                held = hand.pos_mm() > PB.HOLD_MM and obj_z > 40
                got = [t for t in S.TOOLS if (env.d.qpos[env.tool_q[t] + 2] - S.MAT_Z) * 1000 > 40]
                if held and tool not in got:                     # 追う物でなく、ほかの物を持ち上げた（乗り移り）
                    held = False
                    why_wrong = "・".join(NAMES[t] for t in got)
                why = "持てた" if held else (f"別の物（{why_wrong}）をつかんだ" if why_wrong else f"持てない（閉じた {closed:.1f}mm・物の高さ {obj_z:.0f}mm）")
                if held:                                             # 箱まで運んで離し、入ったかを確かめてから次の場面へ（10/20）
                    rv_ = list(R_to_rotvec(R_cmd))
                    z_box = S.MAT_Z + S.BOX_H + 0.03 + grip / 1000           # 物の底が縁の上 30mm
                    robot.moveL([S.BOX_XY[0], S.BOX_XY[1], max(z_hover, z_box)] + rv_, PB.SPEED, PB.HOLD_ACCEL)
                    robot.moveL([S.BOX_XY[0], S.BOX_XY[1], z_box] + rv_, PB.SPEED, PB.HOLD_ACCEL)
                    hand.open()
                    for _ in range(60):
                        frame(env)
                    boxed, _, _ = TG.in_box(env, tool)
                    robot.moveL([S.BOX_XY[0], S.BOX_XY[1], max(z_hover, z_box) + 0.03] + rv_, PB.SPEED, PB.ACCEL)
                    why = "箱に入った" if boxed else "持てたが箱に入らない"
            except SystemExit as e:
                why = f"保護停止（{str(e)[:30]}）"
        if held or why.startswith("保護停止") or why.startswith("見失った") or not D1 or attempt >= PA.RETRY:   # 見失ったのは、つかみ方の失敗ではないので、やり直し方は聞かない
            break
        empty = closed is not None and closed <= PB.HOLD_MM
        how = PL.retry(f, attempt, grip, empty)                  # やり直し方は d1-3B が選ぶ
        print(f"   {why} → d1-3B のやり直し方：{how}", flush=True)
        if how == "skip":
            break
        if how == "lower":
            drop += PA.GRIP_STEP
            grip = PL.grip(f, drop, empty)
        if how == "slow":
            PB.LIFT_SPEED = LIFT_BASE / 2
        hand.open()
        MOVER.on = True
        why = ""
    PB.LIFT_SPEED = LIFT_BASE
    hand.env.set_finger(False)
    MOVER = None
    return {"tool": tool, "mode": mode, "speed": speed, "fps": fps, "lat": lat, "pred": bool(pred_on), "held": bool(held), "boxed": why == "箱に入った", "kind_ok": kind_ok, "n_ask": n_ask, "grip": round(float(grip), 1), "detail": detail, "min_clr": round(float(min_clr), 1), "clr_close": None if "clr_close" not in dir() else round(float(clr_close), 1), "n_floor": int(n_floor), "n_tripod": n_tri, "why": why,
            "track_err": float(np.mean(errs)) if errs else None, "track_err_max": float(np.max(errs)) if errs else None,
            "aim_err": float(np.hypot(*(aim_close - true_close))), "t": round(env.d.time - t0, 2),
            "det_ms": float(np.median(det_ms)) if det_ms else None, "n_img": n_img, "n_cut": n_cut, "n_miss": n_miss}


TRACE = os.environ.get("TRACK_TRACE", "0") == "1"
D1 = os.environ.get("TRACK_D1", "1") == "1"            # d1-3B が種類・つかむ高さ・やり直し方を決める（10/20）。0 で前と同じ（プログラムだけ）
D1_CLOSE = D1 and os.environ.get("TRACK_D1_CLOSE", "1") == "1"   # 閉じる前に、閉じてよいかを d1-3B に聞く
CLOSE_NO_MAX = 2                                       # 「まだ」がこれだけ続いたら、閉じる
ASK_URL = "http://127.0.0.1:5006/ask"


def ask_close(img, name):
    """指の間に物が入っていて、今閉じればつかめるかを d1-3B に聞く。(はい か, かかった ms)"""
    small = cv2.resize(img, (768, int(img.shape[0] * 768 / img.shape[1])), interpolation=cv2.INTER_AREA)
    b = base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode()
    state = (f"A camera on a robot gripper looks down. The two red fingers are open and lowered to the mat to pick up {ENG.get(name, name)}. "
             "The robot is about to close the fingers.")
    qs = {"close": {"type": "noul", "instructions": f"Is {ENG.get(name, name)} between the two red fingers, so that closing them now will grip it?"}}
    w = time.perf_counter()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(ASK_URL, data=json.dumps({"state": state, "questions": qs, "images": [b]}).encode(),
                                                                 headers={"Content-Type": "application/json"}), timeout=30).read())
    return r["answers"]["close"]["noul"] > 0.5, (time.perf_counter() - w) * 1000
TOOLS_ONLY = [t for t in os.environ.get("TRACK_TOOLS", "").split(",") if t]   # この種類だけ試す（例 r_pen）
GRIDS = {
    "one": dict(mode=["line"], speed=[0], fps=[4.3], lat=[0.1], pred=[1]),
    "wide": dict(mode=["line", "stop"], speed=[50, 100], fps=[25], lat=[0.1], pred=[1]),   # 動く物だけ（マットを広げた時）
    "jump50": dict(mode=["jump"], speed=[50], fps=[25], lat=[0.1], pred=[1]),
    "pen": dict(mode=["line"], speed=[0, 50], fps=[25], lat=[0.1], pred=[1]),
    "still": dict(mode=["line"], speed=[0], fps=[25], lat=[0.1], pred=[1]),
    "full": dict(mode=["line", "stop", "jump"], speed=[0, 20, 50, 100], fps=[4.3, 10, 25], lat=[0.1], pred=[1, 0]),
    "quick": dict(mode=["line"], speed=[0, 50], fps=[4.3, 25], lat=[0.1], pred=[1]),
    "lat": dict(mode=["line"], speed=[50], fps=[25], lat=[0.0, 0.1, 0.2, 0.4], pred=[1, 0]),
}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 1
    grid = GRIDS[os.environ.get("TRACK_GRID", "full")]
    sim_rtde.frame = frame                                       # 物を動かしながら進める（moveL・ハンドの中の frame もこれになる）
    T.patch_real_code()                                          # 実機の校正の値を、シミュレーターのカメラの値にする（test_sim_grasp と同じ）
    env = S.RealLikeEnv()
    TG.patch_grasp(env)
    TG.patch_capture(env)
    if "--show" in sys.argv:                                     # Chrome の 1 枚のページに映す（test_sim_grasp と同じ）
        from web_view import WebView
        sim_rtde.VIEWER = WebView(env.m, env.d, title="動く物を追いかける（シミュレーター）")
        sim_rtde.VIEWER.cam.lookat[:] = [S.MAT_CENTER[0], S.MAT_CENTER[1] - 0.05, 0.05]
        sim_rtde.VIEWER.cam.distance, sim_rtde.VIEWER.cam.elevation, sim_rtde.VIEWER.cam.azimuth = 0.9, -35, 120
    robot, hand = SimRobot(env), SimHand(env, log=lambda s: None)
    rng = np.random.default_rng(7)
    rows = []
    tools = ["eraser", "r_pen", "r_rod"]
    combos = list(itertools.product(grid["mode"], grid["speed"], grid["fps"], grid["lat"], grid["pred"]))
    for i, (mode, speed, fps, lat, pred) in enumerate(combos):
        if speed == 0 and mode != "line":
            continue
        for j in range(n):
            if TOOLS_ONLY and tools[(i + j) % 3] not in TOOLS_ONLY:
                continue
            tool = tools[(i + j) % 3]
            r = trial(env, robot, hand, rng, tool, mode, speed, fps, lat, pred, keep=CHAIN and bool(rows) and rows[-1].get("boxed"))
            r["kept"] = KEPT
            rows.append(r)
            te = "―" if r["track_err"] is None else f"{r['track_err']:.1f}"
            print(("（続き）" if r["kept"] else "") + f"{NAMES[tool]} 動き {mode} {speed}mm/秒 撮る {fps}枚/秒 遅れ {lat}秒 予測 {pred}：{r['why']}、"
                  f"追う誤差 {te}mm、閉じた時のずれ {'―' if r['aim_err'] is None else format(r['aim_err'], '.1f')}mm、{r['t']}秒、測る {r.get('det_ms') or 0:.0f}ms"
                  + f"、ハンドと机の最小 {r.get('min_clr', 999):.1f}mm（逃げた {r.get('n_floor', 0)} 回）" + (f"、三脚で測れた {r.get('n_tripod', 0)} 回" if FIXED else "") + ("" if r.get("kind_ok") is None else f"、d1-3B 種類 {'正' if r['kind_ok'] else '誤'}・高さ {r.get('grip')}mm・閉じてよいか {r.get('n_ask')} 回"), flush=True)
    out = os.path.join(HERE, "logs", f"track_sim_{os.environ.get('TRACK_GRID', 'full')}.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("---- まとめ（持てた数／回数、閉じた時のずれの平均 mm）")
    keyf = lambda r: (r["mode"], r["speed"], r["fps"], r["lat"], r["pred"])   # noqa: E731
    for key in sorted({keyf(r) for r in rows}):
        g = [r for r in rows if keyf(r) == key]
        print(f"  {key[0]:4s} {key[1]:3d}mm/秒 {key[2]:4}枚/秒 遅れ {key[3]} 予測 {key[4]}：持てた {sum(r['held'] for r in g)}/{len(g)}・箱に入った {sum(r['boxed'] for r in g)}/{len(g)}、"
              f"ずれ {np.mean([r['aim_err'] for r in g if r['aim_err'] is not None] or [0]):.1f}mm")
    print(f"結果：{out}")


if __name__ == "__main__":
    main()
