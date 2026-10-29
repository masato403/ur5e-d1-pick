"""ROBOTIQ 2F-85（指で挟むハンド）で、ドライバー・ペン・ネジを指示どおりに箱へ入れる（MuJoCo、30FPS）。

  見る係・決める係は run_tools.py と同じ（学習なし：机の画像との違いで物を見つける → d1-3B が「何か」と「どれを運ぶか」）。
  違うのはつかみ方：
    ① 腕をよけて判断した時の画像から、物の長い軸の向きを求める（物の画素の主成分分析）
    ② 物の真上で、指が閉じる向きが物の軸と直角になるように手首を回す
    ③ 下ろす → 指を閉じる → 指が途中で止まったら（全部閉じなかったら）つかめた → 持ち上げる
       物は「くっつける」処理を使わず、指と物の接触・摩擦だけで持つ
    ④ 箱の置き場所の上へ運び、物の軸が箱の壁と平行になるように手首を回して、壁より上で指を開いて落とす

  python robot/run_grip.py --instr "Put all the screws into the box." --want screw --seed 31 --show
  python robot/run_grip.py --batch 31 32          （probe_ids.py と同じ 5 場面、窓なし）
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import mujoco
# mujoco.viewer は、窓を出す時（--show で SIM_WEB=0）だけ読み込む（10/19：スマートアプリコントロールが窓の部品の DLL を止めることがある）
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
os.chdir(os.path.join(HERE, ".."))

from run_pick import DOWN, IK, SAFE_Z, SPEED  # noqa: E402
from run_tools import (DT, FPS, JOINT_SPEED, PARK_Q, SLOTS, WALL_H, Brain, SegPerception,  # noqa: E402
                       in_box)
import run_tools as RT  # noqa: E402（部品の一覧は差し替えられるように、使う時に読む）
from PIL import Image  # noqa: E402
from sim_grip import GripEnv  # noqa: E402
from perception_fastsam import FastSamPerception  # noqa: E402
from test_detect import grain, realistic_fast  # noqa: E402
from sim_tools import BIN, KIND, OBJECTS  # noqa: E402

GRIP_Z = float(os.environ.get("GRIP_GZ", 0.012))            # つかむ時の「つかむ点」の高さ。0.008 だと触れて置いたネジで指先が机にこすれて止まり、持ち上げで落とした（10/15：0.012 で落下 0、ふつうの5場面 5/5）
RELEASE_Z = WALL_H + 0.035  # 放す時の高さ（持った物の下の端が箱の壁より上）
YAW_SPEED = 3.0           # 手首を回す速さ（rad/s）
HOLD_GAP = 0.003          # 指先のすき間がこれより広ければ、物を挟んでいる（何もないと約 0.2mm まで閉じる。細いネジの軸でも約 6.5mm）
CLOSE_TIME = 0.6          # 指を開くのを待つ時間（秒）。閉じる時は、指先が止まるまで（最短 0.3 秒、最長 1.5 秒）待つ


def wrap_half(a):
    """角度を -π/2〜π/2 にそろえる（挟む向きは 180° 回しても同じ）。"""
    return (a + math.pi / 2) % math.pi - math.pi / 2


WRIST_LIMIT = math.pi       # 手首（第 6 関節）を回してよい範囲：±180°


WRIST_MARGIN = math.radians(10)   # 根元の回り方の見込みの誤差（肩のずれ）の分の余裕


def base_turn(tool_xy, goal_xy):
    """手先が tool_xy から goal_xy へ動く間に、腕の根元（第 1 関節）が回る角度の見込み（根元は原点）。"""
    return wrap_pi(math.atan2(goal_xy[1], goal_xy[0]) - math.atan2(tool_xy[1], tool_xy[0]))


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def nearest_yaw(goal, now, wrist=None, turn=0.0):
    """挟む向き goal（180° ごとに同じ）のうち、今の手首の向き now にいちばん近いもの。
    wrist（今の第 6 関節の角度）を渡すと、回した後の第 6 関節が ±180° を超えない向きの中から選ぶ
    （第 6 関節を +1 回すと手首の向きは -1 回る、根元を +1 回すと手首の向きは +1 回る：シミュレーターで測って確かめた）。
    つまり 第 6 関節 ＝ 根元 − 手首の向き ＋ 一定。turn（行き先までに根元が回る角度）も見込んで、着いた時の第 6 関節で判断する。"""
    best = now + wrap_half(goal - now)
    if wrist is None:
        return best
    cands = [best + k * math.pi for k in range(-3, 4)]
    ok = [c for c in cands if abs(wrist + turn - (c - now)) <= WRIST_LIMIT - WRIST_MARGIN]
    return min(ok, key=lambda c: abs(c - now)) if ok else best


def rot_yaw(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ DOWN


def object_axis(per, rgb, o, robot_mask=None):
    """物の画素の主成分分析で、長い軸の向き（作業台の上の角度、rad）を求める。見る係と同じ方法で物の画素を取り出す。"""
    if "mask" in o:                             # 拡大して見直して分けた物は、その輪郭だけで向きを求める
        m = o["mask"].astype(np.uint8)
    else:
        m = per.foreground(rgb).copy()
    if robot_mask is not None:
        m[robot_mask] = 0
    u, v, w, h = o["uv"]
    half = int(max(w, h) / 2 + 6)
    x1, y1 = max(0, int(u) - half), max(0, int(v) - half)
    win = m[y1:int(v) + half + 1, x1:int(u) + half + 1]
    n, lab = cv2.connectedComponents(cv2.dilate(win, np.ones((9, 9), np.uint8)))
    cu, cv_ = int(u) - x1, int(v) - y1
    best, bd = None, 1e9
    for k in range(1, n):                       # 物の中心にいちばん近いかたまり（隣の物の端が入っても混ぜない）
        ys, xs = np.nonzero((lab == k) & (win > 0))
        if len(xs) < 5:
            continue
        dd = (xs.mean() - cu) ** 2 + (ys.mean() - cv_) ** 2
        if dd < bd:
            best, bd = (xs, ys), dd
    if best is None:
        return 0.0, 1.0
    xs, ys = best
    c = np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()]))
    ev, evec = np.linalg.eigh(c)
    du, dv = evec[:, 1]                          # いちばん長く広がっている向き（画像の上で）
    elong = float(math.sqrt(ev[1] / max(ev[0], 1e-6)))
    return math.atan2(-dv, du), elong            # 画像の下向き（v）は作業台の -y 向き


def wrist_pixel_to_table(env, cam_id, u, v, z=0.01, W=640, H=480, fovy=60.0):
    """手首のカメラの画素 (u, v) → 作業台の上（高さ z）の位置。カメラの位置と向きから、視線と作業台の交点を求める。
    実環境では、カメラとハンドの位置関係（ハンドアイ校正）と、ロボットの関節の角度から同じ計算をする。"""
    f = (H / 2) / math.tan(math.radians(fovy) / 2)
    ray_c = np.array([(u - W / 2) / f, -(v - H / 2) / f, -1.0])      # カメラの中での視線の向き（MuJoCo のカメラは -z を見る）
    R = env.d.cam_xmat[cam_id].reshape(3, 3)
    p = env.d.cam_xpos[cam_id]
    d = R @ ray_c
    t = (z - p[2]) / d[2]
    return (p + t * d)[:2]


ABSENT_P = 0.15      # 手首で見た物が、狙った種類である確率がこれ未満なら「狙った物はここにない」とみなす


def wrist_look(env, per, brain, wr, wseg, robot_geoms, cam_id, target_xy, want_kind, pinned=False):
    """手首のカメラで撮って、物を切り出し直す。狙っていた位置の近く（6cm 以内）の物のうち、
    狙っていた種類（AI が前に答えた種類）と同じ答えの物を選び、その位置と長い向きを返す。"""
    import test_real as T
    wr.update_scene(env.d, camera="wrist")
    img = wr.render()
    wseg.update_scene(env.d, camera="wrist")
    own = cv2.dilate(np.isin(wseg.render()[..., 0], robot_geoms).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    with torch.inference_mode():
        objs, _ = T.detect(per.model, img, "small")
    cands = []
    for o in objs:
        mk = o["mask"] & ~own                       # 自分の指が写った所は除く
        if mk.sum() < 80:
            continue
        ys, xs = np.nonzero(mk)
        xy = wrist_pixel_to_table(env, cam_id, xs.mean(), ys.mean())
        if np.linalg.norm(xy - target_xy) > 0.06:
            continue
        o = {"mask": mk, "uv": (float(xs.mean()), float(ys.mean())), "box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))}
        cands.append((o, xy))
    info = {"n_near": len(cands)}
    if os.environ.get("WRIST_DEBUG"):                 # 調査用：手首のカメラの画像と、選んだ輪郭を保存
        dbg = img[:, :, ::-1].copy()
        for o, xy in cands:
            cnts, _ = cv2.findContours(o["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dbg, cnts, -1, (0, 0, 255), 2)
        os.makedirs("robot/wrist", exist_ok=True)
        cv2.imwrite(f"robot/wrist/look_{int(time.time() * 1000)}.png", dbg)
    if not cands:
        return None, None, img, info
    best = None
    if want_kind:                                   # それぞれを AI に見せて、狙っていた種類の物を選ぶ（1 つだけでも、本当に狙った物かを確かめる）
        crit = {p: RT.PART_DESC[p] for p in RT.PARTS + ["other"]}
        kinds = []
        for o, xy in cands:
            q = brain.m.system_one("Close-up camera image of one object on a work table.",
                                   {"q": {"type": "choice", "instructions": "What is this object?", "criteria": crit}},
                                   images=[Image.fromarray(T.crop(img, o))])["answers"]["q"]
            kinds.append((q["choice"], q["probabilities"].get(want_kind, 0.0)))
        info["kinds"] = [k[0] for k in kinds]
        info["p_want"] = [round(k[1], 3) for k in kinds]
        if pinned and all(k[0] != want_kind and k[1] < ABSENT_P for k in kinds):
            # 手首で見直して作った番号（pinned）なのに、近くに狙った種類の物がない（幽霊の番号）：別の物をつかまないよう、つかむのをやめる。
            # 固定カメラで見つけた番号には使わない（手首から見ると、本物のドライバーでも「その他」と答えることがあり、運ばなくなった）
            info["absent"] = True
            return None, None, img, info
        same = [i for i, k in enumerate(kinds) if k[0] == want_kind]
        if same:                                     # 狙った種類と同じ答えの物（2 つ以上なら、狙った位置に近い方）
            best = min((cands[i] for i in same), key=lambda c: np.linalg.norm(c[1] - target_xy))
    if best is None:
        best = min(cands, key=lambda c: np.linalg.norm(c[1] - target_xy))
    o, xy = best
    ys, xs = np.nonzero(o["mask"])
    c = np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()]))
    ev, evec = np.linalg.eigh(c)
    du, dv = evec[:, 1]
    cx, cy = xs.mean(), ys.mean()
    # つかむ場所を、物の長い向きに沿ってずらして選ぶ：指の通り道（指の幅 × 開いた幅）に、ほかの物が入らない所
    others = np.zeros(img.shape[:2], bool)
    for q, _ in cands:
        if q is not o:
            others |= q["mask"]
    for q in objs:
        if q["mask"].sum() and not (q["mask"] & o["mask"]).any():
            others |= q["mask"] & ~own
    dist = env.d.cam_xpos[cam_id][2] - 0.01
    px_m = (img.shape[0] / 2) / math.tan(math.radians(60) / 2) / dist        # 1m が何画素か（作業台の高さで）
    fw, fo = 0.025 * px_m, 0.10 * px_m                                       # 指の幅、開いた指の外側どうしの幅（画素）
    t_all = (xs - cx) * du + (ys - cy) * dv
    best_t, best_s = 0.0, None
    yy, xx = np.nonzero(others)
    for t in np.linspace(t_all.min() * 0.7, t_all.max() * 0.7, 15):
        gx, gy = cx + t * du, cy + t * dv
        a_ = (xx - gx) * du + (yy - gy) * dv                                 # 物の長い向きの距離
        b_ = -(xx - gx) * dv + (yy - gy) * du                                # 指が閉じる向きの距離
        hit = int(((np.abs(a_) < fw / 2) & (np.abs(b_) < fo / 2)).sum())
        score = (hit, abs(t))
        if best_s is None or score < best_s:
            best_t, best_s = t, score
    gx, gy = cx + best_t * du, cy + best_t * dv
    info["grasp_shift_mm"] = round(abs(best_t) / px_m * 1000, 1)
    info["finger_hits_px"] = best_s[0]
    if best_s[0] > 30:
        # どこをつかんでも指の通り道に別の物が入る：いちばん多く入る物を「邪魔な物」として返す
        def in_path(mk):
            yy2, xx2 = np.nonzero(mk)
            a2 = (xx2 - gx) * du + (yy2 - gy) * dv
            b2 = -(xx2 - gx) * dv + (yy2 - gy) * du
            return int(((np.abs(a2) < fw / 2) & (np.abs(b2) < fo / 2)).sum())
        obs = [(in_path(q["mask"]), i) for i, (q, _) in enumerate(cands) if q is not o]
        if obs and max(obs)[0] > 30:
            i = max(obs)[1]
            q, qxy = cands[i]
            qy, qx = np.nonzero(q["mask"])
            cq = np.cov(np.vstack([qx - qx.mean(), qy - qy.mean()]))
            evq, vq = np.linalg.eigh(cq)
            r0 = wrist_pixel_to_table(env, cam_id, qx.mean(), qy.mean())
            r1 = wrist_pixel_to_table(env, cam_id, qx.mean() + 20 * vq[0, 1], qy.mean() + 20 * vq[1, 1])
            info["obstacle"] = {"xy": [float(qxy[0]), float(qxy[1])], "kind": (info.get("kinds") or [None] * len(cands))[i],
                                "ang": float(math.atan2(r1[1] - r0[1], r1[0] - r0[0]))}
    xy = wrist_pixel_to_table(env, cam_id, gx, gy)
    p0 = wrist_pixel_to_table(env, cam_id, cx, cy)
    p1 = wrist_pixel_to_table(env, cam_id, cx + 20 * du, cy + 20 * dv)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])  # 物の長い向き（作業台の上の角度）
    return xy, ang, img, info


def draw_names(rgb, objs, what):
    """人が見る窓・録画用：番号の代わりに、AI が答えた物の名前を札にして出す（まだ聞いていない物は「?」）。
    AI に見せる画像は番号のまま（AI は番号で物を選ぶ）。"""
    im = rgb.copy()
    for k, o in objs.items():
        u, v, w, h = o["uv"]
        name = what.get(k, "?")
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = int(u + w / 2 + 4), int(v - h / 2 - 6)
        x = min(max(x, 0), im.shape[1] - tw - 8)
        y = min(max(y, th + 6), im.shape[0] - 2)
        cv2.rectangle(im, (x, y - th - 6), (x + tw + 8, y + 2), (255, 255, 255), -1)
        cv2.rectangle(im, (x, y - th - 6), (x + tw + 8, y + 2), (0, 0, 0), 1)
        cv2.putText(im, name, (x + 4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return im


ISOLATE = os.environ.get("GRIP_ISOLATE", "0") == "1"          # 見直しの切り抜きで、聞きたい物の外側を机の色で塗る
MERGE_OTHER = os.environ.get("GRIP_MERGE_OTHER", "0") == "1"  # 「その他」の切れ端が、くっついた物の切れ端ならまとめる
SAVE_INSPECT = os.environ.get("GRIP_SAVE_INSPECT", "")   # 記録用：手首で見直した切り抜きを保存する場所


def nearest_obj(env, xy):
    """記録用（判断には使わない）：机の上の点 xy にいちばん近い本当の物の名前と距離（mm）。"""
    o = min(OBJECTS, key=lambda o: np.linalg.norm(env.d.xpos[env.m.body(o).id][:2] - xy))
    return o, round(float(np.linalg.norm(env.d.xpos[env.m.body(o).id][:2] - xy)) * 1000, 1)


def grasp_diag(env):
    """記録用（判断には使わない）：指のいちばん近くの物について、つかんだ場所（物の中心から軸に沿った距離）・指の閉じる向きと物の軸の角度・物の傾き。"""
    R = env.d.site_xmat[env.site].reshape(3, 3)
    pinch, close_dir = env.d.site_xpos[env.site], R[:, 1]
    o = min(OBJECTS, key=lambda o: np.linalg.norm(env.d.xpos[env.m.body(o).id] - pinch))
    b = env.m.body(o).id
    ax = env.d.xmat[b].reshape(3, 3)[:, 0]
    rel = pinch - env.d.xpos[b]
    return {"obj": o, "along_mm": round(float(rel @ ax) * 1000, 1),
            "side_mm": round(float(np.linalg.norm(rel - (rel @ ax) * ax)) * 1000, 1),
            "close_vs_axis_deg": round(math.degrees(math.acos(min(1.0, abs(float(close_dir @ ax))))), 1),
            "tilt_deg": round(math.degrees(math.asin(min(1.0, abs(float(ax[2]))))), 1),
            "obj_z_mm": round(float(env.d.xpos[b][2]) * 1000, 1), "pinch_z_mm": round(float(pinch[2]) * 1000, 1), "touching": finger_contacts(env)}


def finger_contacts(env):
    """記録用：ハンドの部品が、いま何に触れているか（ハンドの部品名 → 相手の物の名前）。"""
    hits = set()
    for c in env.d.contact[:env.d.ncon]:
        for g1, g2 in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            b1, b2 = env.m.body(env.m.geom_bodyid[g1]).name, env.m.body(env.m.geom_bodyid[g2]).name
            if b1.startswith("g_") and not b2.startswith("g_"):
                hits.add(f"{b1}>{b2 or 'world'}")
    return sorted(hits)


def table_to_pixel(xy):
    """作業台の上の位置 → 真上の固定カメラの画素（ToolEnv.pixel_to_table の逆）。"""
    from sim_tools import CAM_Z
    f = (480 / 2) / math.tan(math.radians(45.0) / 2)
    dist = CAM_Z - 0.01
    return 640 / 2 + xy[0] * f / dist, 480 / 2 - (xy[1] - 0.5) * f / dist


def count_objects(brain, rgb, o):
    """真上の固定カメラの切り抜きを AI に見せて、物がいくつ写っているかを聞く（2 つ以上なら、くっついている疑い）。"""
    from run_tools import crop
    a = brain.m.system_one("Close-up camera image of a work table.",
                           {"q": {"type": "choice", "instructions": "How many separate objects are in this image?",
                                  "criteria": {"one": "one object", "two": "two objects", "three": "three or more objects"}}},
                           images=[Image.fromarray(crop(rgb, o))])["answers"]["q"]["choice"]
    return 1 if a == "one" else 2


def overlap_fixed(env, cam_id, mk, per, k):
    """手首の画像の切れ端 mk を固定カメラの画素へ写し、物 k の輪郭と、ほかの番号の物の輪郭に入る割合を返す。"""
    ys, xs = np.nonzero(mk)
    step = max(1, len(xs) // 400)
    ys, xs = ys[::step], xs[::step]
    W, H, fovy = 640, 480, 60.0
    f = (H / 2) / math.tan(math.radians(fovy) / 2)
    ray = np.stack([(xs - W / 2) / f, -(ys - H / 2) / f, -np.ones(len(xs))], 1)
    R = env.d.cam_xmat[cam_id].reshape(3, 3)
    p = env.d.cam_xpos[cam_id]
    d = ray @ R.T
    t = (0.01 - p[2]) / d[:, 2]
    xy = p[:2] + t[:, None] * d[:, :2]
    u, v = table_to_pixel((xy[:, 0], xy[:, 1]))
    u = np.clip(np.round(u).astype(int), 0, 639)
    v = np.clip(np.round(v).astype(int), 0, 479)
    ker = np.ones((7, 7), np.uint8)
    def frac(m):
        return float((cv2.dilate(m.astype(np.uint8), ker) > 0)[v, u].mean())
    def shape(ob):
        return ob.get("mask", ob.get("fp"))
    own_k = frac(shape(per.objs[k])) if shape(per.objs[k]) is not None else 0.0
    other = max([frac(shape(ob)) for j, ob in per.objs.items() if j != k and shape(ob) is not None], default=0.0)
    return own_k, other


def inspect_split(env, per, brain, wr, wseg, robot_geoms, cam_id, k):
    """怪しい物 k の真上から手首のカメラで撮り、近く（8cm 以内）の物を切り出し直して、それぞれ何の物かを AI に聞く。
    2 つ以上に分かれたら、k を消して、分かれた物に新しい番号（pinned：固定カメラでは位置を上書きしない）をつける。"""
    import test_real as T
    xy0 = per.objs[k]["xy"]
    wr.update_scene(env.d, camera="wrist")
    img = wr.render()
    wseg.update_scene(env.d, camera="wrist")
    own = cv2.dilate(np.isin(wseg.render()[..., 0], robot_geoms).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    with torch.inference_mode():
        objs, _ = T.detect(per.model, img, "small")
    crit = {p: RT.PART_DESC[p] for p in RT.PARTS + ["other"]}
    pieces, ov, truth = [], [], []
    # 机の色（どの物の輪郭にも入らない所の中央値）。切り抜きで、聞きたい物の外側をこの色で塗る
    anyobj = np.zeros(img.shape[:2], bool)
    for o in objs:
        anyobj |= o["mask"]
    table_rgb = np.median(img[~anyobj & ~own], axis=0).astype(np.uint8) if (~anyobj & ~own).any() else np.array([200, 195, 185], np.uint8)
    for o in objs:
        mk = o["mask"] & ~own
        if mk.sum() < 80:
            continue
        ys, xs = np.nonzero(mk)
        xy = wrist_pixel_to_table(env, cam_id, xs.mean(), ys.mean())
        if np.linalg.norm(xy - xy0) > 0.08:
            continue
        # 別の番号の物の方が近い切れ端は、その物が写り込んだだけ（分かれた切れ端ではない）
        if any(np.linalg.norm(xy - ob["xy"]) < np.linalg.norm(xy - xy0) for j, ob in per.objs.items() if j != k):
            continue
        # 切れ端の輪郭を固定カメラの画素へ写し、ほかの番号の物の輪郭と重なる方が多ければ、その物の一部とみなす
        # （手首では隣の物が一部しか写らず、切れ端の中心が k の方に寄ることがある）
        own_k, other = overlap_fixed(env, cam_id, mk, per, k)
        ov.append((round(own_k, 2), round(other, 2)))
        if other > 0.3 and other > own_k:
            continue
        o = {"mask": mk, "uv": (float(xs.mean()), float(ys.mean())), "box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))}
        src = img
        if ISOLATE:                                    # 隣の物が写り込むと、ねじを「ドライバー」「その他」と答えた（10/15）。その物だけを見せる
            src = img.copy()
            src[~cv2.dilate(mk.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)] = table_rgb
        q = brain.m.system_one("Close-up camera image of one object on a work table.",
                               {"q": {"type": "choice", "instructions": "What is this object?", "criteria": crit}},
                               images=[Image.fromarray(T.crop(src, o))])["answers"]["q"]
        pieces.append((xy, q["choice"], o["mask"]))
        truth.append((q["choice"], nearest_obj(env, xy)))
        if SAVE_INSPECT:
            os.makedirs(SAVE_INSPECT, exist_ok=True)
            cv2.imwrite(os.path.join(SAVE_INSPECT, f"t{env.d.time:06.2f}_k{k}_{q['choice']}_true-{truth[-1][1][0]}.png"),
                        cv2.cvtColor(np.asarray(T.crop(src, o)), cv2.COLOR_RGB2BGR))
    # くっついていて同じ種類と答えた切れ端（ドライバーの持ち手と軸など）は 1 つにまとめる
    merged = True
    while merged:
        merged = False
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                same = pieces[i][1] == pieces[j][1]
                if not same and not (MERGE_OTHER and "other" in (pieces[i][1], pieces[j][1])):
                    continue
                if not (cv2.dilate(pieces[i][2].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)[pieces[j][2]].any():
                    continue
                mk = pieces[i][2] | pieces[j][2]
                ys, xs = np.nonzero(mk)
                kind = pieces[j][1] if pieces[i][1] == "other" else pieces[i][1]   # 「その他」の切れ端は、くっついている物の一部（ドライバーの軸など）
                pieces[i] = (wrist_pixel_to_table(env, cam_id, xs.mean(), ys.mean()), kind, mk)
                del pieces[j]
                merged = True
                break
            if merged:
                break
    # 小さな切れ端（いちばん大きい物の 15% 未満）で「その他」と答えた物は、隣の物の一部とみなして捨てる
    if pieces:
        big = max(p[2].sum() for p in pieces)
        pieces = [p for p in pieces if not (p[1] == "other" and p[2].sum() < 0.15 * big)]
    info = {"inspect": k, "pieces": [p[1] for p in pieces], "overlap": ov,
            "truth": [[a_, n, d] for a_, (n, d) in truth], "truth_k": nearest_obj(env, xy0)}
    if len(pieces) < 2:
        # 1 つだけ：分かれなかった。種類は上のカメラの答えを残す（手首の答えで上書きすると、正しい答えを壊すことがあった）。
        # 上のカメラで「その他」だった時だけ、近くから見た答えを使う
        if len(pieces) == 1 and brain.what.get(k) == "other":
            brain.what[k] = pieces[0][1]
        return [], info
    del per.objs[k]
    brain.what.pop(k, None)
    new = []
    for xy, kind, _ in pieces:
        nid = per.next_id
        per.next_id += 1
        u, v = table_to_pixel(xy)
        per.objs[nid] = {"xy": np.array(xy), "uv": (u, v, 20, 20), "seen": 0, "pinned": True}
        brain.what[nid] = kind
        brain.what_size[nid] = (20, 20)
        new.append(nid)
    info["new_ids"] = new
    return new, info


def free_spot(per, exclude_xy, rng_seed=0):
    """机の上の空いている所：作業範囲の中で、知っている物から 12cm 以上、箱から 15cm 以上離れた所（いちばん離れた候補）。"""
    from sim_tools import AREA
    xs = np.linspace(AREA[0][0], AREA[0][1], 9)
    ys = np.linspace(AREA[1][0], AREA[1][1], 8)
    known = [o["xy"] for o in per.objs.values()] + [np.array(exclude_xy)]
    best, bd = None, -1
    for x in xs:
        for y in ys:
            p = np.array([x, y])
            if np.linalg.norm(p - BIN) < 0.15:
                continue
            d = min(np.linalg.norm(p - k) for k in known)
            if d > bd:
                best, bd = p, d
    return best, bd


def touch_layout(env, rng):
    """触れている物の場面：ドライバーとペンを離して置き、それぞれの横にネジをくっつける（すき間 2〜3mm）。"""
    sd = np.array([rng.uniform(-0.05, 0.05), rng.uniform(0.33, 0.40)])
    pen = np.array([rng.uniform(0.15, 0.25), rng.uniform(0.50, 0.60)])
    for body, xy, side_gap, yaw in (("screwdriver", sd, 0.014 + 0.009 + 0.003, rng.uniform(0, math.pi)),
                                    ("pen", pen, 0.006 + 0.009 + 0.002, rng.uniform(0, math.pi))):
        env.set_obj(body, xy, yaw=yaw)
        perp = np.array([-math.sin(yaw), math.cos(yaw)])
        along = np.array([math.cos(yaw), math.sin(yaw)])
        screw = "screw1" if body == "screwdriver" else "screw2"
        env.set_obj(screw, xy + perp * side_gap + along * rng.uniform(-0.02, 0.02), yaw=yaw + rng.uniform(-0.3, 0.3))
    mujoco.mj_forward(env.m, env.d)


WEB = os.environ.get("SIM_WEB", "1") == "1"   # --show の時、窓ではなく Chrome の 1 枚のページで見せる（10/19）


def other_display(user32):
    """主でないディスプレイの作業域 (左, 上, 右, 下)。1 つしか無ければ None。"""
    import ctypes
    from ctypes import wintypes

    class MI(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]
    out = []

    def cb(hmon, _hdc, _rc, _l):
        mi = MI()
        mi.cbSize = ctypes.sizeof(MI)
        user32.GetMonitorInfoW(hmon, ctypes.byref(mi))
        if not mi.dwFlags & 1:                                             # 1 ＝ 主のディスプレイ
            w = mi.rcWork
            out.append((w.left, w.top, w.right, w.bottom))
        return 1
    user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.c_void_p]
    user32.EnumDisplayMonitors(None, None, ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR, wintypes.HDC,
                                                              ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)(cb), 0)
    return max(out, key=lambda r: (r[2] - r[0]) * (r[3] - r[1])) if out else None


def topmost_windows(title_part, done=None):
    """題名に title_part を含む窓を、出た時に一度だけ手前に出す（ずっと最前面に固定はしない）。
    もう 1 つのディスプレイがつながっていれば、そちらへ移して最大化する（SIM_DISPLAY=keep で前と同じ場所）。
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
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
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
    mon = other_display(user32) if os.environ.get("SIM_DISPLAY", "other") == "other" else None
    for h in found:
        if mon is not None:
            # もう 1 つのディスプレイ（主でない方）へ移して最大化する（10/16：別のディスプレイにつないでいる時はそちらに大きく出す）
            user32.ShowWindow(h, 9)                                        # 最大化などを一度戻す（移せるように）
            user32.SetWindowPos(h, wintypes.HWND(0), mon[0] + 40, mon[1] + 40, 1280, 800, 0x0004 | 0x0040)
            user32.ShowWindow(h, 3)                                        # 最大化（今いるディスプレイいっぱいに広がる）
        flags = 0x0001 | 0x0002 | 0x0040                                   # 大きさと位置は変えない、表示する
        user32.SetWindowPos(h, wintypes.HWND(-1), 0, 0, 0, 0, flags)      # いったん最前面にして手前に出す
        user32.SetWindowPos(h, wintypes.HWND(-2), 0, 0, 0, 0, flags)      # すぐに固定を外す（手前に出たまま）
    if found and done is not None:
        done.add(title_part)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--instr", default="Put all the screws into the box.")
    ap.add_argument("--want", default="screw", help="答え合わせ用：箱に入るべき部品（, 区切り）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--video", default="")
    ap.add_argument("--show", action="store_true", help="ロボットの 3D 表示と、AI が見ている画像を窓に出す")
    ap.add_argument("--log", default="robot/logs/run_grip.jsonl")
    ap.add_argument("--detector", default="ref", choices=["ref", "fastsam"],
                    help="物の見つけ方：ref＝何もない机の画像との違い（シミュレーターならでは）／fastsam＝FastSAM（実環境向け）")
    ap.add_argument("--touch", action="store_true", help="物どうしを触れさせて置く（ネジ 1 をドライバーの横、ネジ 2 をペンの横）")
    ap.add_argument("--wrist", action="store_true", help="物の真上に来たら手首のカメラで撮り、物を分け直してから、つかむ位置と向きを決め直す")
    ap.add_argument("--inspect", default="sus", choices=["sus", "all", "need"],
                    help="手首のカメラで見に行く物：sus＝怪しい物だけ（その他・2 つ以上写っている）／all＝机の上の物を最初に全部 1 回ずつ"
                         "／need＝怪しい物のうち、運ぶ予定でない物だけ（運ぶ物は、つかむ直前の手首の見直しで確かめるので、別に往復しない）")
    ap.add_argument("--clear", default="both", choices=["none", "order", "move", "both"],
                    help="つかむ物の横に邪魔な物がある時：order＝それも運ぶ物なら先に運ぶ／move＝机の空いている所へどける／both＝両方")
    ap.add_argument("--zoom", action="store_true", help="判断の前に、物ごとに拡大してもう一度切り出す（触れている物を分ける）")
    ap.add_argument("--reask", action="store_true", help="物の形が 3 割以上変わったら、何の物かを AI に聞き直す")
    ap.add_argument("--camera", default="clean", choices=["clean", "real"],
                    help="カメラ画像：clean＝描いたまま／real＝実環境ふう（木目・照明の変化・ノイズ）")
    a = ap.parse_args(argv)

    env = GripEnv()
    env.reset(np.random.default_rng(a.seed))
    if a.touch:
        touch_layout(env, np.random.default_rng(a.seed))
    empty = env.empty_table()
    cam_rng = np.random.default_rng(1000 + a.seed)
    table = np.all(np.abs(empty.astype(int) - empty[240, 600].astype(int)) < 6, axis=2)    # 木目を乗せる所（画像を作る側の都合）
    wood = grain(np.random.default_rng(1))
    light_base = (cam_rng.uniform(0.8, 1.15), tuple(cam_rng.uniform(0.93, 1.07, 3)))

    def camera(rgb, t):
        """カメラの画像。real のときは、照明がゆっくり変わり続ける（雲や人の出入りのような変化）。"""
        if a.camera == "clean":
            return rgb
        lt = (light_base[0] * (1 + 0.12 * math.sin(t / 4.0)), light_base[1])
        return realistic_fast(rgb, table, wood, lt)

    if a.detector == "fastsam":
        per = FastSamPerception()
    else:
        per = SegPerception(camera(empty, 0.0) if a.camera == "real" else empty)   # 机を撮った時の照明は、始めの照明
    ik = IK(env)
    q_cmd = PARK_Q.copy()
    env.d.qpos[:6] = q_cmd
    env.d.ctrl[:6] = q_cmd
    env.set_finger(False)
    mujoco.mj_forward(env.m, env.d)
    seg = mujoco.Renderer(env.m, 480, 640)
    seg.enable_segmentation_rendering()
    obj_geoms = set(env.object_geoms())
    robot_geoms = np.array([g for g in range(env.m.ngeom) if g not in obj_geoms and env.m.geom_bodyid[g] != 0])

    def robot_mask():
        seg.update_scene(env.d, camera="top")
        return cv2.dilate(np.isin(seg.render()[..., 0], robot_geoms).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)

    wr = mujoco.Renderer(env.m, 480, 640)              # 手首のカメラの画像
    wseg = mujoco.Renderer(env.m, 480, 640)            # 手首のカメラに写った自分の指を見つける用
    wseg.enable_segmentation_rendering()
    wcam = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    looked, wrist_max = False, 0.0
    inspected, insp_target, insp_xy = set(), None, None   # 手首のカメラで見に行った物
    fails = {}                                             # 番号ごとの失敗の回数（空振り・取り落とし・近くに何もない）
    alone = set()                                          # それ自体を手首のカメラで見に行った物（分けて出てきた物は入らない）
    recheck_other = os.environ.get("GRIP_RECHECK_OTHER", "0") == "1"   # 触れた場面用（後回し、10/15）。普段は使わない
    forgot_at = [-99.0]                                    # 最後に番号を忘れた時刻

    def give_up(k, why):
        """同じ番号で 3 回失敗したら、その番号を忘れる。固定カメラで見え直した所から、新しい番号として判断し直す。"""
        fails[k] = fails.get(k, 0) + 1
        if fails[k] >= 3 and k in per.objs:
            per.objs.pop(k, None)
            brain.what.pop(k, None)
            inspected.discard(k)
            events.append({"t": round(sim_t, 2), "forget": k, "why": why})
            forgot_at[0] = sim_t
    clears, relocate = 0, None                            # 邪魔な物を片付けた回数、どける先（どけている間だけ）
    raised = set()                                     # 一度手前に出した窓の題名
    brain = Brain(a.instr, "twostep")
    brain.reask = a.reask
    viewer = None
    if a.show:
        if WEB:                                        # 窓ではなく Chrome の 1 枚のページで見せる（10/19。SIM_WEB=0 で前と同じ窓）
            from web_view import WebView
            viewer = WebView(env.m, env.d, title="片付け（シミュレーター）")
        else:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.m, env.d, show_left_ui=False, show_right_ui=False)
            cv2.namedWindow("AI view", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("AI view", 640, 560)
        viewer.cam.lookat[:] = [-0.05, 0.35, 0.1]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, -60, -35
    n_sub = round(DT / env.m.opt.timestep)
    cmd = env.tool_pos()
    goal = cmd.copy()
    yaw_cmd = yaw_goal = env.tool_yaw()
    phase, target, target_xy, slot_i, slot = "park", None, None, 0, BIN
    wait_until, lifted, gaps = 0, None, []
    events, picks, misses = [], [], 0
    writer = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 480)) if a.video else None
    t_loop, late, frame_no = [], 0, 0
    wall0 = time.perf_counter()
    next_t = wall0
    while frame_no < a.seconds * FPS:
        t0 = time.perf_counter()
        for _ in range(n_sub):
            env.d.qfrc_applied[:6] = env.d.qfrc_bias[:6]
            mujoco.mj_step(env.m, env.d)
        wrist_max = max(wrist_max, abs(float(env.d.qpos[5])))   # 手首（第 6 関節）がいちばん回った角度
        frame = camera(env.render(), frame_no * DT)
        mask = robot_mask()
        holding = phase in ("lift", "carry", "lower") and lifted is not None
        if phase != "decide":                    # 判断の間は腕を止めていて何も動かないので、番号の当て直しはしない（拡大して分けた結果を保つ）
            per.update(frame, frame_no, mask, tool_xy=env.tool_pos()[:2] if holding else None)
        tool = env.tool_pos()
        sim_t = frame_no * DT
        joint_mode = phase in ("park", "decide", "finished")
        if phase == "park":
            if np.abs(env.d.qpos[:6] - PARK_Q).max() < 0.03:
                phase = "decide"
        elif phase == "decide":
            if not brain.busy and brain.result is None and frame_no >= wait_until:
                if a.zoom and hasattr(per, "refine"):
                    ch_ = per.refine(frame, frame_no, mask)
                    if ch_:
                        events.append({"t": round(sim_t, 2), "zoom_split": {str(k): v for k, v in ch_.items()}})
                vis = {k: o for k, o in per.objs.items() if frame_no - o["seen"] <= 3 or "mask" in o or o.get("pinned")}
                brain.ask(frame, vis)
            elif brain.result is not None:
                ch, brain.result = brain.result, None
                events.append({"t": round(sim_t, 2), "decide": ch, **(brain.last or {})})
                # 怪しい物（AI が「その他」と答えた物、または「物が 2 つ以上写っている」と答えた物）は、運ぶ前に手首のカメラで見に行く
                sus = None
                if a.wrist:
                    asked = (brain.last or {}).get("what", {})              # AI に見せた物（判断の間は番号の当て直しを止めているので、この一覧を使う）
                    for k, o in [(k, per.objs[k]) for k in asked if k in per.objs]:
                        if k in inspected or in_box(o["xy"]):
                            continue
                        if a.inspect == "need" and (brain.last or {}).get("wanted", {}).get(brain.what.get(k), 0) >= 0.5:
                            continue                                    # 運ぶ予定の物は、近づいた時の手首の見直しで確かめる
                        inspected.add(k)
                        if a.inspect == "all" or brain.what.get(k) == "other" or count_objects(brain, frame, o) >= 2:
                            sus = k
                            break
                if sus is None and ch == "done" and a.wrist and recheck_other:
                    # 「終わり」の前に：分けて出てきた「その他」を、それ自体はまだ近くで見ていなければ見に行く（ねじの取り残し対策）
                    for k in (brain.last or {}).get("what", {}):
                        if k in per.objs and k not in alone and brain.what.get(k) == "other" and not in_box(per.objs[k]["xy"]):
                            sus = k
                            break
                if sus is not None:
                    alone.add(sus)
                    insp_target, insp_xy = sus, per.objs[sus]["xy"].copy()
                    phase = "inspect_go"
                    events.append({"t": round(sim_t, 2), "go_inspect": sus, "what": brain.what.get(sus)})
                    cmd = tool.copy()
                    yaw_cmd = yaw_goal = env.tool_yaw()
                elif ch == "done" and sim_t - forgot_at[0] < 3.0:
                    events.append({"t": round(sim_t, 2), "recheck": True})   # 番号を忘れた直後の「終わり」は、固定カメラで見え直してから判断し直す
                    wait_until = frame_no + int(1.0 * FPS)
                elif ch == "done":
                    phase = "finished"
                else:
                    target, phase = int(ch[3:]), "approach"
                    target_xy = per.objs[target]["xy"].copy()
                    looked = False
                    ang, elong = object_axis(per, frame, per.objs[target], mask)
                    yaw_goal = nearest_yaw(ang, env.tool_yaw(), env.d.qpos[5], base_turn(tool, target_xy))   # 指が閉じる向きが、物の軸と直角になる
                    events.append({"t": round(sim_t, 2), "axis_deg": round(math.degrees(ang), 1), "elong": round(elong, 2)})
                    cmd = tool.copy()
                    yaw_cmd = env.tool_yaw()
                    env.set_finger(False)
        elif phase in ("approach", "descend"):
            goal = np.array([target_xy[0], target_xy[1], SAFE_Z if phase == "approach" else GRIP_Z])
            near = np.linalg.norm(tool - goal) < (0.01 if phase == "approach" else 0.004)
            if np.linalg.norm(cmd - goal) < 1e-3 and near and abs(yaw_cmd - yaw_goal) < 1e-3:
                if phase == "approach" and a.wrist and not looked:
                    phase, wait_until = "look", frame_no + int(0.3 * FPS)     # 揺れが収まるまで少し待ってから撮る
                elif phase == "approach":
                    phase = "descend"
                else:
                    env.set_finger(True)
                    phase, wait_until, gaps = "close", frame_no, []
        elif phase == "inspect_go":
            goal = np.array([insp_xy[0], insp_xy[1], SAFE_Z])
            if np.linalg.norm(cmd - goal) < 1e-3 and np.linalg.norm(tool - goal) < 0.01:
                phase, wait_until = "inspect_look", frame_no + int(0.3 * FPS)
        elif phase == "inspect_look":
            if frame_no >= wait_until:
                new, info = inspect_split(env, per, brain, wr, wseg, robot_geoms, wcam, insp_target)
                inspected.update(new)
                events.append({"t": round(sim_t, 2), **info})
                goal = np.array([tool[0], tool[1], SAFE_Z])
                phase = "park"                                 # 腕をよけてから、改めて判断する
        elif phase == "look":
            if frame_no >= wait_until:
                kind = brain.what.get(target) if hasattr(brain, "what") else None
                nxy, nang, wimg, info = wrist_look(env, per, brain, wr, wseg, robot_geoms, wcam, target_xy, kind,
                                                    pinned=bool(per.objs.get(target, {}).get("pinned")))
                ev = {"t": round(sim_t, 2), "wrist": target, "want": kind, **info}
                obs = info.get("obstacle")
                cleared = False
                if info.get("n_near") == 0:
                    info["absent"] = True                 # 近くに何も見えない（もう動いた物・幽霊）：その場所ではつかまない
                    give_up(target, "empty")
                if info.get("absent"):
                    # 狙った物がない：手首で見直して作った番号（pinned）は消す。固定カメラの番号は、見えた種類に直す
                    if per.objs.get(target, {}).get("pinned"):
                        per.objs.pop(target, None)
                        brain.what.pop(target, None)
                        ev["dropped_id"] = target
                        # 運ぶはずの物が見当たらない＝どこかで見間違えている：「その他」の物をもう一度見に行けるようにする
                        again = [k for k in per.objs if brain.what.get(k) == "other" and k in inspected]
                        inspected.difference_update(again)
                        forgot_at[0] = sim_t
                        if again:
                            ev["reinspect"] = again
                    elif info.get("kinds"):
                        brain.what[target] = info["kinds"][0]
                        ev["rekind"] = info["kinds"][0]
                if obs and a.clear != "none" and clears < 3:
                    # 邪魔な物を先に片付ける：運ぶ予定の物なら先に運ぶ（順番を変える）、運ばない物なら机の空いている所へどける
                    oxy = np.array(obs["xy"])
                    near_ids = [k for k in per.objs if k != target]          # 狙っている物自身は、邪魔な物の候補から外す（取り違えて、邪魔な物を箱に入れていた）
                    oid = min(near_ids, key=lambda k: np.linalg.norm(per.objs[k]["xy"] - oxy)) if near_ids else None
                    if oid is None or np.linalg.norm(per.objs[oid]["xy"] - oxy) > 0.03:     # 知っている物に当てはまらなければ、新しい番号をつける
                        oid = per.next_id
                        per.next_id += 1
                        per.objs[oid] = {"xy": oxy, "uv": (*table_to_pixel(oxy), 20, 20), "seen": frame_no, "pinned": True}
                        if obs["kind"]:
                            brain.what[oid] = obs["kind"]
                    okind = brain.what.get(oid) or obs["kind"]
                    wanted_obs = (brain.wanted or {}).get(okind, 0) >= 0.5
                    mode = "order" if (wanted_obs and a.clear in ("both", "order")) else ("move" if a.clear in ("both", "move") else None)
                    if mode:
                        clears += 1
                        relocate = None
                        if mode == "move":
                            relocate, _ = free_spot(per, target_xy)
                            ev["move_to"] = [round(float(relocate[0]), 3), round(float(relocate[1]), 3)]
                        ev["clear"] = {"mode": mode, "obstacle": oid, "kind": okind}
                        target, target_xy = oid, oxy.copy()
                        yaw_goal = nearest_yaw(obs["ang"], yaw_cmd, env.d.qpos[5], base_turn(tool, oxy))
                        cleared = True
                if not cleared and nxy is not None:
                    ev["moved_mm"] = round(float(np.linalg.norm(nxy - target_xy)) * 1000, 1)
                    target_xy = nxy
                    yaw_goal = nearest_yaw(nang, yaw_cmd, env.d.qpos[5])
                events.append(ev)
                if wimg is not None and a.show and WEB:
                    viewer.image("手首のカメラ", wimg[:, :, ::-1])
                elif wimg is not None and a.show:
                    cv2.imshow("wrist camera", wimg[:, :, ::-1])
                    cv2.waitKey(1)
                    topmost_windows("wrist camera", raised)          # 初めて出た時だけ手前に出す
                looked = not cleared                      # 邪魔な物に切り替えた時は、その真上でもう一度撮ってからつかむ
                if info.get("n_near") == 0:
                    info["absent"] = True                 # 近くに何も見えない（もう動いた物・幽霊）：その場所ではつかまない
                    give_up(target, "empty")
                if info.get("absent"):
                    target = None
                    goal = np.array([tool[0], tool[1], SAFE_Z])
                    phase = "park"                        # 腕をよけてから、改めて判断する
                else:
                    phase = "approach"                    # 決め直した位置の真上へ動き直してから下ろす
        elif phase == "close":
            gaps.append(env.pad_gap())
            n = frame_no - wait_until
            stopped = n >= int(0.3 * FPS) and max(gaps[-4:]) - min(gaps[-4:]) < 0.0002
            if stopped or n >= int(1.5 * FPS):
                events.append({"t": round(sim_t, 2), "grasp": target, "gap_mm": round(env.pad_gap() * 1000, 1), **grasp_diag(env)})
                if env.pad_gap() > HOLD_GAP:             # 指先が途中で止まった＝物を挟んでいる
                    per.held = target
                    lifted = env.pad_gap()
                    phase = "lift"
                else:                                    # 何も挟めなかった：指を開いて上がり、判断からやり直す
                    misses += 1
                    events.append({"t": round(sim_t, 2), "miss": target, "gap_mm": round(env.pad_gap() * 1000, 1)})
                    give_up(target, "miss")
                    env.set_finger(False)
                    lifted = None
                    phase, wait_until = "retreat_miss", frame_no + int(CLOSE_TIME * FPS)
        elif phase == "lift":
            goal = np.array([tool[0], tool[1], SAFE_Z]) if goal[2] < SAFE_Z else goal
            if np.linalg.norm(tool - goal) < 0.01:
                # 持ち上げた後も指が途中で止まっていれば、本当に持てている（滑り落ちたら指が閉じきる）
                real = [o for o in OBJECTS if env.obj_pos(o)[2] > 0.08]
                if env.pad_gap() > HOLD_GAP:                 # 判断はすき間だけ（物の本当の高さは記録用）
                    picks.append(real[0] if real else "?")
                    if relocate is not None:                  # 邪魔な物をどける：箱ではなく、机の空いている所へ（向きはそのまま）
                        slot = relocate.copy()
                        yaw_goal = nearest_yaw(yaw_cmd, yaw_cmd, env.d.qpos[5], base_turn(tool, slot))   # 向きは同じ（180° 違いも同じ）で、手首が回りすぎない方
                    else:
                        slot = BIN + SLOTS[slot_i % len(SLOTS)]
                        s = SLOTS[slot_i % len(SLOTS)]
                        yaw_goal = nearest_yaw(math.atan2(s[1], s[0]) + math.pi / 2, yaw_cmd, env.d.qpos[5], base_turn(tool, slot))   # 物の軸を壁と平行に
                    phase = "carry"
                else:
                    misses += 1
                    events.append({"t": round(sim_t, 2), "dropped": target, "gap_mm": round(env.pad_gap() * 1000, 1), **grasp_diag(env)})
                    give_up(target, "dropped")
                    env.set_finger(False)
                    per.held, lifted = None, None
                    phase, wait_until = "retreat_miss", frame_no
        elif phase == "carry":
            goal = np.array([slot[0], slot[1], SAFE_Z])
            if np.linalg.norm(cmd - goal) < 1e-3 and np.linalg.norm(tool - goal) < 0.01 and abs(yaw_cmd - yaw_goal) < 1e-3:
                phase = "lower"
        elif phase == "lower":
            goal = np.array([slot[0], slot[1], RELEASE_Z if relocate is None else GRIP_Z + 0.01])   # どける時は机の上にそっと置く
            if np.linalg.norm(cmd - goal) < 1e-3 and np.linalg.norm(tool - goal) < 0.005:
                env.set_finger(False)
                events.append({"t": round(sim_t, 2), ("moved_aside" if relocate is not None else "placed"): picks[-1] if picks else None})
                if target in per.objs:
                    per.objs[target]["xy"] = slot.copy()
                    per.objs[target]["seen"] = frame_no
                    per.objs[target].pop("pinned", None)
                    per.placed[target] = (slot.copy(), frame_no + 3 * FPS)
                per.held, lifted = None, None
                if relocate is None:
                    slot_i += 1
                relocate = None
                phase, wait_until = "release", frame_no + int(CLOSE_TIME * FPS)
        elif phase == "release":
            if frame_no >= wait_until:
                phase = "retreat"
        elif phase in ("retreat", "retreat_miss"):
            goal = np.array([tool[0], tool[1], SAFE_Z]) if goal[2] < SAFE_Z else goal
            if np.linalg.norm(tool - goal) < 0.01:
                phase = "park"
        if joint_mode or phase == "park":
            dq = PARK_Q - q_cmd
            q_cmd = q_cmd + np.clip(dq, -JOINT_SPEED * DT, JOINT_SPEED * DT)
        else:
            # 指を閉じる・開く間は、腕を止めて待つ（つかみ損ねた時も、指が開ききってから上がる）
            if not (phase in ("close", "release") or (phase == "retreat_miss" and frame_no < wait_until)):
                step = goal - cmd
                dist = np.linalg.norm(step)
                cmd = goal.copy() if dist < SPEED * DT else cmd + step / dist * SPEED * DT
                yaw_cmd += float(np.clip(yaw_goal - yaw_cmd, -YAW_SPEED * DT, YAW_SPEED * DT))
            q_cmd = ik.solve(q_cmd, cmd, rot_yaw(yaw_cmd))
        env.d.ctrl[:6] = q_cmd
        t2 = time.perf_counter()
        t_loop.append((t2 - t0) * 1000)
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            show = draw_names(frame, per.visible(frame_no), getattr(brain, "what", {}))[:, :, ::-1].copy()
            panel = np.full((80, 640, 3), 255, np.uint8)
            last = brain.last or {}
            what = last.get("what", {})
            cv2.putText(panel, f"instruction: {a.instr}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(panel, "seen: " + ", ".join(sorted(map(str, what.values()))), (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 0, 0), 1)
            put = [p for p, v in last.get("wanted", {}).items() if v >= 0.5]
            cv2.putText(panel, f"to box: {', '.join(put)}   now: {phase} {getattr(brain, 'what', {}).get(target, '?') if target and phase != 'finished' else ''}"
                        f"   gap {env.pad_gap() * 1000:.0f}mm", (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1)
            if WEB:
                if frame_no % 5 == 0:                    # ページへは 1 秒に 4 回ほど送る（毎回送ると重い）
                    viewer.image("AI が見ている画像", np.vstack([show, panel]))
            else:
                cv2.imshow("AI view", np.vstack([show, panel]))
            if not WEB and frame_no in (0, 15, 60, 150):             # 窓が出た時に一度だけ手前に出す（3D 表示の窓は開くのが少し遅いので、出るまで何度か確かめる）
                topmost_windows("MuJoCo", raised)
                topmost_windows("AI view", raised)
            if not WEB:
                cv2.waitKey(1)
        if writer is not None:
            img = draw_names(frame, per.visible(frame_no), getattr(brain, "what", {}))[:, :, ::-1].copy()
            cv2.putText(img, f"{phase} {getattr(brain, 'what', {}).get(target, '?') if target else ''}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
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
        if not WEB:
            cv2.destroyAllWindows()
    wall = time.perf_counter() - wall0
    truth = sorted(o for o in OBJECTS if in_box(env.obj_pos(o)[:2]))
    want = sorted(o for o in OBJECTS if KIND[o] in set(a.want.split(",")))
    res = {"brain": "twostep", "hand": "2f85", "detector": a.detector, "camera": a.camera, "touch": a.touch, "reask": a.reask, "zoom": a.zoom, "wrist": a.wrist, "inspect": a.inspect, "clear": a.clear, "wrist_max_deg": round(math.degrees(wrist_max), 1), "instr": a.instr, "seed": a.seed, "success": truth == want, "in_box": truth,
           "want": want, "picks": picks, "misses": misses, "finished": phase == "finished", "ids_used": per.next_id - 1,
           "sim_seconds": round(frame_no * DT, 1), "real_fps": round(frame_no / wall, 1), "late_frames": late,
           "loop_ms_avg": round(float(np.mean(t_loop)), 1), "events": events, "births": per.births}
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    with open(a.log, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(res, ensure_ascii=False, default=float) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("events", "births")}, ensure_ascii=False), flush=True)


CASES = [("Put all the screws into the box.", "screw"),
         ("Put everything except the pen into the box.", "screwdriver,screw"),
         ("Put the pen and the screwdriver into the box.", "pen,screwdriver")]
PROBE = [(0, 31), (0, 32), (1, 32), (2, 31), (2, 32)]      # probe_ids.py と同じ 5 場面

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        import gc
        extra = sys.argv[2:]                                    # --show や --detector fastsam などを、そのまま各場面に渡す
        for ci, seed in PROBE:
            instr, want = CASES[ci]
            main(["--instr", instr, "--want", want, "--seed", str(seed)] + extra)
            gc.collect()
    else:
        main()
