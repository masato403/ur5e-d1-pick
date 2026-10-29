"""三脚の校正（calib_tripod_grid の出力）で、腕で読んだマス目の交点（mat_grid_points.json）を写真に描く（10/23 夕方、目盛り 0 の位置を確かめる）。"""
import json
import os
import sys

import cv2
import numpy as np

REAL_UR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "real_ur")
sys.path.insert(0, REAL_UR)
from plan_hover import TABLE_Z  # noqa: E402

cal = json.load(open(sys.argv[1], encoding="utf-8"))
img = cv2.imread(sys.argv[2])
d = json.load(open(os.path.join(REAL_UR, "mat_grid_points.json"), encoding="utf-8"))
L = float(sys.argv[4]) if len(sys.argv) > 4 else -67.2
P = []
for v in d.values():
    z = cv2.Rodrigues(np.array(v["rv"]))[0][:, 2]
    p = np.array(v["tcp_mm"][:2]) + L * z[:2]
    P.append([p[0], p[1], TABLE_Z + 2.0])
uv, _ = cv2.projectPoints(np.array(P).reshape(-1, 1, 3), np.array(cal["rvec"]), np.array(cal["tvec"]), np.array(cal["K"]), np.array(cal["dist"]))
big = cv2.resize(img, None, fx=2, fy=2)
for (u, v), k in zip(uv.reshape(-1, 2), d):
    cv2.circle(big, (int(u * 2), int(v * 2)), 8, (0, 255, 255), 2)
    cv2.putText(big, k, (int(u * 2) + 10, int(v * 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    print(k, np.round([u, v], 1))
cv2.imwrite(sys.argv[3], big[200:950, 300:1650])
