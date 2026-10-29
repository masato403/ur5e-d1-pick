r"""手首と三脚のカメラのずれを確かめる（10/23 夕方、配置を変えた後）。ハンドは動かさない。物は押さない。
mat_grid_points.json（指先をマス目の交点の上に置いて読んだ 4 点）から、マス目の目盛りを根元の座標に直し、
指先を目盛りの交点の真上（机から H_MM）へ順に動かして、手首の写真（1920×1080）と三脚の写真と腕の位置を残す。
手先は真下に向け直す（手で動かした後は 2〜6° 傾いている）。手首の回りは今のまま。解析は後でオフラインでする。
使い方:  .venv_ur\Scripts\python.exe real_ur\check_cams.py            （行き先を出すだけ）
         .venv_ur\Scripts\python.exe real_ur\check_cams.py --go       （動かす）
結果: real_ur/logs_check_cams_1023/（p01_wrist.jpg・p01_tripod.jpg・points.json）"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import locate as L
from plan_hover import TABLE_Z

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "logs_check_cams_1023")
IP = "192.0.2.10"
SPEED, ACCEL = 0.05, 0.2
H_MM = 50.0                                               # 机から指先まで（calib_tripod.py と同じく 50mm 以上。物は 12mm 以下）
TICKS = [(x, y) for y in (5, 15, 25) for x in (5, 15, 25, 35)]   # 目盛り（cm）


def mat_fit():
    """目盛り（mm）→ 根元の xy。指先の傾きの分（道具の軸に沿って L mm）も一緒に求める（calib_tripod_grid.py と同じ）。"""
    d = json.load(open(os.path.join(HERE, "mat_grid_points.json"), encoding="utf-8"))
    A, b = [], []
    for v in d.values():
        g = np.array(v["grid_cm"], float) * 10
        z = cv2.Rodrigues(np.array(v["rv"]))[0][:2, 2]
        A += [[g[0], -g[1], 1, 0, -z[0]], [g[1], g[0], 0, 1, -z[1]]]
        b += v["tcp_mm"][:2]
    x = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    yaw = np.arctan2(x[1], x[0])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return lambda t: R @ (np.array(t, float) * 10) + x[2:4]


def vertical(rv):
    """手先の向き rv を、いちばん小さく回して真下（道具の z が -Z）に向け直す。"""
    Rc = cv2.Rodrigues(np.array(rv, float))[0]
    z = Rc[:, 2]
    axis = np.cross(z, [0, 0, -1.0])
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return list(rv)
    ang = np.arctan2(s, float(np.dot(z, [0, 0, -1.0])))
    Ra = cv2.Rodrigues(axis / s * ang)[0]
    return cv2.Rodrigues(Ra @ Rc)[0].ravel().tolist()


def tripod_after(t0, timeout=12.0):
    p = os.path.join(HERE, "cam_shots", "tripod_latest.jpg")
    while time.time() - t0 < timeout:
        if os.path.exists(p) and os.path.getmtime(p) > t0:
            time.sleep(0.1)
            img = cv2.imread(p)
            if img is not None:
                return img
        time.sleep(0.05)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--heights", type=float, nargs="+", default=[H_MM])   # 机から指先まで（mm、30 以上）
    ap.add_argument("--yaws", type=float, nargs="+", default=[0.0])       # 手首を今の向きから回す角度（°、±90 以内）
    ap.add_argument("--ticks", default="")                                # 目盛り "5,5 25,15 ..."（空なら TICKS）
    ap.add_argument("--xy", default="")                                   # 根元の座標 "x,y x,y ..."（mm）。あれば目盛りの代わりに、物ごとに高い順に撮る（手で隠れた時の合わせ、10/23）
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    to_xy = mat_fit()
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(IP)
    p0 = r.getActualTCPPose()
    rot = vertical(p0[3:])
    tilt = np.degrees(np.arccos(-cv2.Rodrigues(np.array(p0[3:]))[0][2, 2]))
    ticks = [tuple(float(v) for v in s_.split(",")) for s_ in a.ticks.split()] if a.ticks else TICKS
    assert min(a.heights) >= 30 and max(abs(y_) for y_ in a.yaws) <= 90
    Rv = cv2.Rodrigues(np.array(rot))[0]
    rots = {yw: cv2.Rodrigues(cv2.Rodrigues(np.array([0, 0, np.radians(yw)]))[0] @ Rv)[0].ravel().tolist() for yw in a.yaws}   # 真下の軸（根元の z）まわりに回す
    goals = [((t, h_, yw), to_xy(t)) for yw in a.yaws for h_ in a.heights for t in ticks]
    if a.xy:
        pts = [tuple(float(v) for v in s_.split(",")) for s_ in a.xy.split()]
        goals = [((p_, h_, yw), np.array(p_)) for p_ in pts for yw in a.yaws for h_ in sorted(a.heights, reverse=True)]
    print(f"今の指先 x {p0[0] * 1000:.0f} y {p0[1] * 1000:.0f} 机から {p0[2] * 1000 - TABLE_Z:.0f}mm、傾き {tilt:.1f}° → 真下に直す")
    print(f"  {len(goals)} か所：{'座標 ' + a.xy if a.xy else '目盛り ' + str(ticks)}、高さ {a.heights}、回り {a.yaws}°")
    if not a.go:
        print("--go が無いので、ここまで（何も送っていません）")
        return
    from rtde_control import RTDEControlInterface
    OUT_ = a.out
    os.makedirs(OUT_, exist_ok=True)
    c = RTDEControlInterface(IP)
    rows = []
    try:
        up = [p0[0], p0[1], (TABLE_Z + 100.0) / 1000] + rot     # まず真上へ上がる（手で置いた所は机から数 mm）
        c.moveL(up, SPEED, ACCEL)
        for k, ((t, h_, yw), xy) in enumerate(goals):
            goal = [xy[0] / 1000, xy[1] / 1000, (TABLE_Z + h_) / 1000] + rots[yw]
            if not c.isPoseWithinSafetyLimits(goal):
                print(f"  {k + 1}: 安全設定の範囲の外なので飛ばす", flush=True)
                continue
            c.moveL(goal, SPEED, ACCEL)
            if r.getSafetyMode() != 1:
                print("  安全状態が普通でないので止めます", flush=True)
                break
            p = r.getActualTCPPose()
            t0 = time.time()
            w = L.grab_now(settle=0.5)
            tri = tripod_after(t0)
            if w is not None:
                cv2.imwrite(os.path.join(OUT_, f"p{k + 1:02d}_wrist.jpg"), w)
            if tri is not None:
                cv2.imwrite(os.path.join(OUT_, f"p{k + 1:02d}_tripod.jpg"), tri)
            rows.append({"tick_cm": list(t), "h": h_, "yaw": yw, "goal_mm": [float(xy[0]), float(xy[1])], "tcp_mm": [v * 1000 for v in p[:3]], "rv": list(p[3:]),
                         "wrist": w is not None, "tripod": tri is not None})
            print(f"  {k + 1}/{len(goals)}：目盛り {t}・高さ {h_:.0f}・回り {yw:+.0f}°、指先 x {p[0] * 1000:.1f} y {p[1] * 1000:.1f} 机から {p[2] * 1000 - TABLE_Z:.0f}mm、"
                  f"手首 {'○' if w is not None else '×'} 三脚 {'○' if tri is not None else '×'}", flush=True)
        c.moveL(up, SPEED, ACCEL)                               # 始めの所の真上へ戻る
    finally:
        json.dump(rows, open(os.path.join(a.out, "points.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        c.stopScript()
        c.disconnect()
    print("終わり:", a.out)


if __name__ == "__main__":
    main()
