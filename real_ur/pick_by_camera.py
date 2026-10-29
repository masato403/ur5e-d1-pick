"""手首カメラで消しゴムを見つけ、その真上へ行ってつかみ、持ち上げる（実機）。

  1) 机から OBS mm の見張りの高さへ上げる（真下向き）  2) 手首カメラで白い物（消しゴム）を探し、
     上の面の中心と長い辺の向きを机の上の座標で求める（校正 calib2/calib_result.json）
  3) 指を結ぶ向き（道具の y 軸）が、長い辺と直角になるように手首の先を回す（回す量が小さい方）
  4) 道具の点の横のずれ（grip_geom.json）を足して、真上 100mm へ → 開く → GRIP mm まで下ろす → 力 FORCE% で閉じる → 持ち上げる
  ハンドは hand_server.py を通す（動かす前に位置を確認する）。
  5) --place x y があれば：持てているのを確かめてから、その真上 100mm へ運び、指先を GRIP+1 mm まで下ろして離し、100mm へ上げる
使い方:  .venv_ur\\Scripts\\python.exe real_ur\\pick_by_camera.py --ip 192.0.2.10 --go
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

import locate as L
from hand import HandClient
from plan_hover import R_to_rotvec, TABLE_Z, rotvec_to_R

HERE = os.path.dirname(os.path.abspath(__file__))
GEOM = json.load(open(os.path.join(HERE, "grip_geom.json")))
OBS, HOVER, ERASER_H = 150.0, 100.0, 12.0
# 腕の速さ（m/s）と加速度。10/16 に上げた（前は 0.05・0.02・0.2・持ち上げ 0.03）：「腕を動かすのが遅すぎる」
# 物の近くで下ろす時・持ち上げる時だけ遅めにする（ぶつかった時の勢いを小さく、つかんだ物を振り落とさないように）
# 腕の速さ（m/秒）と加速度（m/秒²）。10/16 夕方に上げた（前は 0.15・0.05・0.08・0.5、下りる時は全部 0.05）。
# 下りる速い所・持ち上げるは 0.15 → 0.25 に（段階ごとの時間で、下りる 1.9 秒・持ち上げる 1.4 秒だった）
SPEED = float(os.environ.get("PB_SPEED", 0.25))           # 運ぶ・真上へ行く
DOWN_SPEED = float(os.environ.get("PB_DOWN_SPEED", 0.05))  # 物に近づく最後の SLOW_MM だけ、ゆっくり（触れる時の強さは前と同じ）
DOWN_FAST = float(os.environ.get("PB_DOWN_FAST", 0.25))    # そこまでは速く下りる（前は 0.05 で全部、約 2 秒かかっていた）
LIFT_SPEED = float(os.environ.get("PB_LIFT_SPEED", 0.25))  # 持ち上げる
ACCEL = float(os.environ.get("PB_ACCEL", 0.8))
# 物を持っている間（持ち上げる・運ぶ・置く所へ下ろす）の加速度。加速度を全部 1.2 にすると、銀の棒が持ち上げる途中で滑って落ちた
# （10/19 夜の実機。0.8 では落とさない）。持っていない動きだけ速くするために分けた。無ければ ACCEL と同じ
HOLD_ACCEL = float(os.environ.get("PB_HOLD_ACCEL", ACCEL))
SLOW_MM = 20.0
# 離す時に開く量：つかんでいた開き量 ＋ これ（mm）。全開（24.8mm）まで開くと 1.0 秒かかった（10/16）。0 なら全開
RELEASE_GAP = float(os.environ.get("PB_RELEASE_GAP", 6.0))
TIMING = os.environ.get("PB_TIMING", "0") == "1"   # 1 なら、つかむ・運ぶ・置くの段階ごとの時間を出す
HOLD_MM = 2.0
BIAS_SIGN = float(os.environ.get("PB_BIAS_SIGN", -1))  # 横のずれの補正の向き。シミュレーター（10/16）で -1 だと指の真ん中のずれ 10mm → 0.5mm 以下。+1 が前の向き。実機で未確認
BOX_FAR = float(os.environ.get("PB_BOX_FAR", 600.0))   # 手で覚えさせた箱へ運ぶ時の、置く場所までの距離の上限（mm）。箱はマットの横にある
MAX_SIZE, MAX_FAR = 70.0, 200.0   # 消しゴムの長い辺の上限、今の位置から離れてよい距離（mm）
WRIST_MAX = float(os.environ.get("WRIST_MAX", 180))       # 手首の先の関節の範囲（度）。決まり：±180° の中で回す
MAT_V_MAX = 205          # マットの明るさの上限（棚・窓の白は 240 前後）
BLUE_MIN = 0.01           # 候補の中の青い部分の割合の下限（カバー付きの消しゴム）
MIN_AREA = 6000          # 見張りの高さで消しゴムは約 18000 画素。マットの白いラベル（約 3800）は外す


def mat_area(img):
    """マット（暗い紫）の部分。校正の mat_mask より明るさの上限を広げる（カメラが明るさを上げてもマットが外れない）。"""
    hsv = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (0, 0), 9)
    m = cv2.inRange(hsv, (0, 0, 0), (180, 45, MAT_V_MAX))
    m[int(img.shape[0] * 0.72):] = 0                          # 下の方はハンドの本体
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))


def find_object(img, tcp):
    """白い物の上の面の中心と、長い辺の向き（根元の座標、単位ベクトル）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 190), (180, 60, 255))
    blue = cv2.inRange(hsv, (100, 120, 80), (130, 255, 255))
    white |= blue                                                # 消しゴムの青い紙も消しゴムの一部（青い面が上でも形が欠けない）
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))   # 方眼の細い線を消す（明るく写ると白に入る）
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # 青い帯や文字で切れないように
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mat = mat_area(img)
    # マットの縁の白い目盛りは、大きさも形も消しゴムに近い（10/15）。何でも用の物の見つけ方（マットの縁から 25mm 以内・床は外す）で
    # 見つかった物の中に中心がある候補だけにする
    import measure
    inner_objs, _ = measure.find_objects(img)
    best = None
    for c in cnts:
        c = cv2.convexHull(c)                                    # 青い帯で白が欠けても、外側の形で見る
        area = cv2.contourArea(c)
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if area < MIN_AREA or cy > img.shape[0] * 0.72 or area / max(1.0, w * h) < 0.8:
            continue
        inside = np.zeros(white.shape, np.uint8)
        cv2.drawContours(inside, [c], -1, 255, -1)
        ring = cv2.dilate(inside, np.ones((41, 41), np.uint8)) & ~cv2.dilate(inside, np.ones((11, 11), np.uint8))
        if (mat[ring > 0] > 0).mean() < 0.6:                       # まわりがマットでない（棚・枠・窓）は外す
            continue
        if not any(cv2.pointPolygonTest(o, (float(cx), float(cy)), False) >= 0 for o in inner_objs):
            continue                                                # マットの縁の近く・マットの外
        if (blue[inside > 0] > 0).mean() < BLUE_MIN:
            continue                                                # カバー付きの消しゴムには青い部分がある。縁の白い目盛りは真っ白
        # 長い辺が MAX_SIZE を超える物（マットの縁の白い目盛りなど）は、選ぶ前に外す（10/15：一番大きい白い物として選んで止まった）
        bp = cv2.boxPoints(cv2.minAreaRect(c))
        bmm = np.array([L.pixel_to_table(p, tcp) for p in bp])
        if max(np.linalg.norm(bmm[1] - bmm[0]), np.linalg.norm(bmm[2] - bmm[1])) > MAX_SIZE:
            continue
        if best is None or area > best[1]:
            best = (c, area)
    if best is None:
        return None
    c = best[0]
    box = cv2.boxPoints(cv2.minAreaRect(c))
    top = L.G0.copy()
    L.G0 = top + np.array([0, 0, ERASER_H])                     # 上の面の高さで机の座標に直す
    pts = np.array([L.pixel_to_table(p, tcp) for p in box])
    L.G0 = top
    center = pts.mean(0)
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    longv = e1 if np.linalg.norm(e1) > np.linalg.norm(e2) else e2
    size = sorted([np.linalg.norm(e1[:2]), np.linalg.norm(e2[:2])])
    longv = longv / np.linalg.norm(longv)
    return center, longv, size, box


LAST = {"closed": None}      # 最後につかんだ時の、閉じた直後の開き量（mm）。ほぼ 0 なら空振り（物より上か横で閉じた）


QUICK_CLOSE = os.environ.get("TR_QUICK_CLOSE", "0") == "1"   # 1：from_here の時、もう開いていれば開き直さず、つかむ所にいれば下り直さず、すぐ閉じる
OPEN_FULL_MM = 24.0


def descend(c, top, bottom, accel=None):
    """top から bottom へ真下に下りる。bottom の SLOW_MM 上までは DOWN_FAST で、そこからは DOWN_SPEED でゆっくり。
    accel：加速度（無ければ ACCEL。物を持って下ろす時は HOLD_ACCEL）"""
    a = ACCEL if accel is None else accel
    if top[2] - bottom[2] > SLOW_MM / 1000 + 1e-4 and DOWN_FAST > DOWN_SPEED:
        c.moveL(list(bottom[:2]) + [bottom[2] + SLOW_MM / 1000] + list(bottom[3:]), DOWN_FAST, a)
    c.moveL(bottom, DOWN_SPEED, a)


def aim(c, r, tcp, center, longv, grip, place_xy=None):
    """つかむ時の手首の向きと道具の点の行き先。(回す角度°, 回転ベクトル, x, y mm, 横のずれの補正, 手首の先の関節の角度の並び, 今の関節) を返す。
    grasp から取り出した（10/20 夜。下りながら向きを合わせる track_real.follow でも使う）。"""
    # 指を結ぶ向き（道具の y 軸）を、長い辺と直角に：道具の y 軸 = 長い辺を 90° 回した向き（±どちらか回す量が小さい方）
    Rt = rotvec_to_R(tcp[3:])
    cur_y = Rt @ np.array(GEOM["close_dir_tool"])
    want = np.array([-longv[1], longv[0], 0.0])
    if np.dot(want, cur_y) < 0:
        want = -want
    turn0 = math.degrees(math.atan2(np.cross(cur_y, want)[2], np.dot(cur_y, want)))
    # 手首の先の関節が ±WRIST_MAX° を超えるなら、反対向き（180° 違い）に回す。どちらも超えるなら止める
    q_now = r.getActualQ()
    for turn in (turn0, turn0 - 180 if turn0 > 0 else turn0 + 180):
        a_ = math.radians(turn)
        Rz = np.array([[math.cos(a_), -math.sin(a_), 0], [math.sin(a_), math.cos(a_), 0], [0, 0, 1]])
        Rg = Rz @ Rt
        bias = Rg @ np.array(GEOM["bias_tool_mm"])
        tx, ty = center[0] + BIAS_SIGN * bias[0], center[1] + BIAS_SIGN * bias[1]
        rv = list(R_to_rotvec(Rg))
        hover = [tx / 1000, ty / 1000, (TABLE_Z + HOVER) / 1000] + rv
        down = [tx / 1000, ty / 1000, (TABLE_Z + grip) / 1000] + rv
        route = [hover, down] + ([[place_xy[0] / 1000, place_xy[1] / 1000, (TABLE_Z + HOVER) / 1000] + rv] if place_xy else [])
        qs, qn = [], q_now
        for p in route:
            qn = c.getInverseKinematics(p, qn)
            qs.append(math.degrees(qn[5]))
        if all(abs(v) <= WRIST_MAX for v in qs):
            break
        print(f"   手首の先を {turn:+.0f}° 回すと、関節が {max(qs, key=abs):+.0f}° になり ±{WRIST_MAX:.0f}° を超えるので、反対向きを試す")
    else:
        raise SystemExit(f"どちら向きに回しても手首の先の関節が ±{WRIST_MAX:.0f}° を超えるので止めます")
    return turn, rv, tx, ty, bias, qs, q_now


def grasp(c, r, tcp, center, longv, grip, force, place_xy=None, look=False, place_h=None, miss="stop", place_far=None, from_here=False, closed_w=None):
    """見つけた物（中心 center、長い辺の向き longv、根元の座標 mm）を、上から指先を grip mm（机から）まで下ろしてつかみ、持ち上げる。
    place_xy があれば、そこへ運んで置く。look なら行き先を出すだけ。消しゴム用（main）と、何でも用（pick_any.py）で共通。
    place_h：離す時の指先の高さ（机から mm）。無ければ grip+1（机の上に置く）。箱に入れる時は、箱の高さ ＋ grip ＋ すき間（落として入れる）
    place_far：置く場所とつかむ所の距離の上限（mm）。無ければ MAX_FAR。手で覚えさせた箱（teach_box）はマットの横にあって遠いので広げる
    miss：持てなかった時。"stop" は止める（SystemExit）。"retry" は指を開いて真上へ戻り False を返す（やり直しは呼ぶ側、10/16）。
    from_here：運ぶ高さ（HOVER）へ上がらず、今の位置（物の真上の低い所）からそのまま下ろす（track_real で下りながら追った後。10/22）。
    持てた時（と、運んで置いた時）は True を返す。"""
    turn, rv, tx, ty, bias, qs, q_now = aim(c, r, tcp, center, longv, grip, place_xy)
    hover = [tx / 1000, ty / 1000, (TABLE_Z + HOVER) / 1000] + rv
    down = [tx / 1000, ty / 1000, (TABLE_Z + grip) / 1000] + rv
    print(f"   手首の先の関節: 今 {math.degrees(q_now[5]):+.0f}° → つかむ時 {qs[0]:+.0f}°"
          + (f" → 置く時 {qs[-1]:+.0f}°" if place_xy else ""))
    if look:
        print(f"   （--look なので、ここまで。行き先は x {tx:.1f}  y {ty:.1f} mm、手首の先を {turn:+.0f}° 回す予定）")
        return
    print(f"3) 手首の先を {turn:+.0f}° 回し、道具の点を x {tx:.1f}  y {ty:.1f} mm へ（横のずれの補正 {bias[0]:+.1f}, {bias[1]:+.1f}）")
    place = []
    if place_xy:
        far_max = MAX_FAR if place_far is None else place_far
        if math.hypot(place_xy[0] - tx, place_xy[1] - ty) > far_max:
            raise SystemExit(f"置く場所がつかむ所から {far_max:.0f}mm より遠いので止めます")
        rel = grip + 1.0 if place_h is None else place_h
        if rel > HOVER:
            raise SystemExit(f"離す高さ {rel:.0f}mm が運ぶ高さ {HOVER:.0f}mm より高いので止めます")
        place = [[place_xy[0] / 1000, place_xy[1] / 1000, (TABLE_Z + z) / 1000] + rv for z in (HOVER, rel)]
        print(f"   置く場所: x {place_xy[0]:.1f}  y {place_xy[1]:.1f} mm（指先を机から {rel:.0f}mm まで下ろして離す）")
    for p in [hover, down] + place:
        if not c.isPoseWithinSafetyLimits(p):
            raise SystemExit("行き先が安全設定の範囲の外なので止めます")
    h = HandClient()
    t_last = [time.perf_counter()]

    def lap(what):
        if TIMING:
            now = time.perf_counter()
            print(f"   ⏱ {what} {now - t_last[0]:.1f} 秒")
            t_last[0] = now
    if from_here:
        top = [tx / 1000, ty / 1000, max(tcp[2], down[2])] + rv
    else:
        top = hover
        c.moveL(hover, SPEED, ACCEL)
        lap("真上へ")
    w0_ = h.pos_mm() if (from_here and QUICK_CLOSE and closed_w is None) else None
    if closed_w is not None:                                     # 追いながらもう閉じた（track_real の CLOSE_MOVING）：開く・下りる・閉じるを飛ばす
        w = closed_w
        print(f"5) 追いながら閉じた: 開き量 {w:.1f}mm")
    else:
        w = None
    if w is None and w0_ is not None and w0_ >= OPEN_FULL_MM - 0.5:   # もう開いている：開き直さない（止まってためてから閉じるように見えるため。10/22）
        print(f"4) 開いている: {w0_:.1f} mm（開き直さない）")
    elif w is None:
        print("4) 開く:", round(h.open(), 1), "mm")
    if w is None:
        lap("開く")
        if not (from_here and QUICK_CLOSE and abs(top[2] - down[2]) < 1e-3 and np.hypot(top[0] - down[0], top[1] - down[1]) < 1e-3):
            descend(c, top, down)                                # QUICK_CLOSE で、もうつかむ所にいれば下り直さない
        lap("下りる")
        w = h.grip(force)
        lap("閉じる")
        print(f"5) 閉じた: 開き量 {w:.1f}mm")
    LAST["closed"] = w
    c.moveL(hover, LIFT_SPEED, HOLD_ACCEL)
    lap("持ち上げる")
    w2 = h.pos_mm()
    print(f"6) 持ち上げた: 机から {r.getActualTCPPose()[2]*1000 - TABLE_Z:.0f}mm、開き量 {w2:.1f}mm →",
          "持てている" if w2 and w2 > HOLD_MM else "持てていない")
    if not (w2 and w2 > HOLD_MM) and miss == "retry":
        print("   持てていないので、指を開いて真上へ戻る（やり直す）")
        h.open()
        h.close_port()
        return False
    if place:
        if not (w2 and w2 > HOLD_MM):
            raise SystemExit("持てていないので、運ばずに止めます")
        c.moveL(place[0], SPEED, HOLD_ACCEL)
        lap("運ぶ・上がる")
        descend(c, place[0], place[1], HOLD_ACCEL)
        lap("下ろす")
        print(f"7) 置く高さ: 机から {r.getActualTCPPose()[2]*1000 - TABLE_Z:.1f}mm")
        wr = h.move_mm(min(24.8, w2 + RELEASE_GAP)) if RELEASE_GAP > 0 else h.open()
        print("8) 離した: 開き量", round(wr, 1), "mm")
        lap("離す")
        c.moveL(place[0], SPEED, ACCEL)
        lap("運ぶ・上がる")
        print(f"9) 上げた: 机から {r.getActualTCPPose()[2]*1000 - TABLE_Z:.0f}mm")
    h.close_port()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--grip", type=float, default=6.0)
    ap.add_argument("--force", type=int, default=60)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--at", type=float, nargs=2, help="見張りの位置 x y（mm）。無ければ今の真上")
    ap.add_argument("--turn", type=float, default=0.0, help="見張りの時に手首の先を回す角度（度、±90 まで）")
    ap.add_argument("--place", type=float, nargs=2, help="置く場所 x y（mm、根元の座標）。つかんだ所から 200mm 以内")
    ap.add_argument("--look", action="store_true", help="見張りの高さへ行って探すだけ（つかまない）")
    a = ap.parse_args()
    import web_page
    web_page.tee_stdout()                                         # print の文字を、cam_view が出す Chrome のタブにも出す（10/19）
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(a.ip)
    c = RTDEControlInterface(a.ip)
    try:
        p0 = r.getActualTCPPose()
        R0 = rotvec_to_R(p0[3:])
        x = R0[:, 0].copy(); x[2] = 0; x /= np.linalg.norm(x)
        if abs(a.turn) > 90:
            raise SystemExit("--turn は ±90° まで")
        t_ = math.radians(a.turn)
        x = np.array([x[0] * math.cos(t_) - x[1] * math.sin(t_), x[0] * math.sin(t_) + x[1] * math.cos(t_), 0.0])
        Rdown = np.column_stack([x, np.cross([0, 0, -1.0], x), [0, 0, -1.0]])
        ox, oy = (a.at[0] / 1000, a.at[1] / 1000) if a.at else (p0[0], p0[1])
        if math.hypot(ox - p0[0], oy - p0[1]) > 0.4:
            raise SystemExit("見張りの位置が今から 40cm 以上離れているので止めます")
        obs = [ox, oy, (TABLE_Z + OBS) / 1000] + list(R_to_rotvec(Rdown))
        if not c.isPoseWithinSafetyLimits(obs):
            raise SystemExit("見張りの位置が安全設定の範囲の外なので止めます")
        print(f"1) 見張りの高さ（机から {OBS:.0f}mm）へ")
        if not a.go:
            print("--go が無いので、ここまで（何も送っていません）")
            return
        c.moveL(obs, SPEED, ACCEL)
        tcp = r.getActualTCPPose()
        img = L.grab_now()                                       # カメラの窓に撮る合図を出してすぐ撮る（前は 6 秒待っていた、10/16）
        L.archive(img, "pick_obs", {"tcp": list(tcp)})
        found = find_object(img, tcp)
        if found is None:
            raise SystemExit("消しゴムが見つからない")
        center, longv, size, box = found
        dbg = img.copy()
        cv2.drawContours(dbg, [box.astype(int)], -1, (0, 255, 0), 4)
        cv2.imwrite(os.path.join(HERE, "cam_shots", "pick_found.jpg"), cv2.resize(dbg, (960, 540)))
        print(f"2) 見つけた: 中心 x {center[0]:.1f}  y {center[1]:.1f} mm、大きさ {size[0]:.0f}×{size[1]:.0f} mm、"
              f"長い辺の向き {math.degrees(math.atan2(longv[1], longv[0])):.0f}°")
        far = math.hypot(center[0] - tcp[0] * 1000, center[1] - tcp[1] * 1000)
        if size[1] > MAX_SIZE or far > MAX_FAR:
            raise SystemExit(f"見つけた物が消しゴムらしくない（大きさ {size[1]:.0f}mm、今の位置から {far:.0f}mm）ので止めます")
        grasp(c, r, tcp, center, longv, a.grip, a.force, a.place, a.look)
    finally:
        c.stopScript()
        c.disconnect()
        r.disconnect()


if __name__ == "__main__":
    main()
