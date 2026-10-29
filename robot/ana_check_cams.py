"""check_cams.py の写真から、手首と三脚のカメラのずれを出す（10/23 夕方）。
手首：写真のマス目の線の画素を、wrist_xy で机の面（マットの上、机から 2mm）の位置に直し、目盛りの座標で 10mm の線からどれだけずれているか（x・y、10mm で割った余り）。
三脚：赤い指の先（calib_tripod.finger_tip）と、腕の指先を校正で写真に映した所の差（新しい校正・前の校正）。"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "real_ur")
sys.path.insert(0, HERE)
os.chdir(HERE)
import track_real as TR  # noqa: E402
import calib_tripod as CT  # noqa: E402
from plan_hover import TABLE_Z  # noqa: E402

D = os.path.join(HERE, "logs_check_cams_1023")
rows = json.load(open(os.path.join(D, "points.json"), encoding="utf-8"))
# 目盛り（mm）⇔ 根元の xy（check_cams.mat_fit と同じ）
d = json.load(open("mat_grid_points.json", encoding="utf-8"))
A, b = [], []
for v in d.values():
    g = np.array(v["grid_cm"], float) * 10
    z = cv2.Rodrigues(np.array(v["rv"]))[0][:2, 2]
    A += [[g[0], -g[1], 1, 0, -z[0]], [g[1], g[0], 0, 1, -z[1]]]
    b += v["tcp_mm"][:2]
x = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
M2 = np.array([[x[0], -x[1]], [x[1], x[0]]])
to_tick = lambda p: np.linalg.solve(M2, (np.asarray(p, float) - x[2:4]).T).T   # 根元 → 目盛り mm

cals = {"新": json.load(open("tripod_calib_grid_1023b.json", encoding="utf-8")), "前": json.load(open("tripod_calib.json", encoding="utf-8"))}


def proj(cal, P):
    uv, _ = cv2.projectPoints(np.array(P, float).reshape(-1, 1, 3), np.array(cal["rvec"]), np.array(cal["tvec"]), np.array(cal["K"]), np.array(cal["dist"]))
    return uv.reshape(-1, 2)


def circ_mean(r):
    a = r / 10.0 * 2 * np.pi
    m = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    return m / (2 * np.pi) * 10.0, np.hypot(np.sin(a).mean(), np.cos(a).mean())


print("点  目盛り    手首のずれ x・y（mm、強さ）        三脚：指先の写真のずれ 新・前（画素）")
for k, rw in enumerate(rows):
    tcp = rw["tcp_mm"]
    tcp_m = [v / 1000 for v in tcp] + rw["rv"]
    out = f"{k + 1:2d}  {str(tuple(rw['tick_cm'])):9s} "
    w = cv2.imread(os.path.join(D, f"p{k + 1:02d}_wrist.jpg"))
    if w is not None:
        g = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)
        th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, np.ones((15, 15), np.uint8))
        hsv = cv2.cvtColor(w, cv2.COLOR_BGR2HSV)
        line = (th > 30) & (hsv[..., 0] < 30) & (hsv[..., 1] > 30)
        line[int(w.shape[0] * 0.66):] = False
        line[:, :int(w.shape[1] * 0.05)] = False
        ys, xs = np.nonzero(line[::3, ::3])
        ys, xs = ys * 3, xs * 3
        pick = np.random.default_rng(0).choice(len(xs), min(4000, len(xs)), replace=False)
        P = np.array([TR.wrist_xy((float(xs[i]), float(ys[i])), tcp_m, 2.0) for i in pick])
        T = to_tick(P)
        near = np.linalg.norm(P - np.array(tcp[:2]), axis=1) < 80          # 指先から 80mm 以内
        T = T[near]
        rx, sx = circ_mean(T[:, 0] % 10)
        ry, sy = circ_mean(T[:, 1] % 10)
        out += f"  x {rx:+5.1f}（{sx:.2f}）y {ry:+5.1f}（{sy:.2f}）  "
    tri = cv2.imread(os.path.join(D, f"p{k + 1:02d}_tripod.jpg"))
    f = CT.finger_tip(tri) if tri is not None else None
    if f is not None:
        uv = f[0]
        es = [np.linalg.norm(proj(c, [tcp])[0] - uv) for c in cals.values()]
        out += f"        新 {es[0]:5.1f}・前 {es[1]:5.1f}"
    print(out, flush=True)
