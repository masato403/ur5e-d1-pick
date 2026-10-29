r"""手で隠れた時の合わせ（10/23）：腕をよけた今の三脚の写真で、マットの上の物の位置と大きさを出す（基準）。腕は動かさない。
使い方:  set TRI_DX=0& set TRI_DY=0& .venv_ur\Scripts\python.exe real_ur\hide_base.py   → real_ur/logs_hide_1023/base.json・base_tripod.jpg"""
import json
import os

import cv2
import numpy as np
from rtde_receive import RTDEReceiveInterface

import tripod_view as TV

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.environ.get("HIDE_OUT", "logs_hide_1023"))
os.makedirs(OUT, exist_ok=True)
tcp = RTDEReceiveInterface("192.0.2.10").getActualTCPPose()
objs, img = TV.objects(tcp)
if img is None:
    raise SystemExit("三脚の写真が来ない")
cv2.imwrite(os.path.join(OUT, "base_tripod.jpg"), img)
rows = []
for i, o in enumerate(objs):
    w, l = TV.size_mm(o["contour"])
    rows.append({"xy": [float(v) for v in o["xy"]], "px": [float(v) for v in o["px"]], "size": [w, l]})
    print(f"  {i + 1}: x {o['xy'][0]:.1f} y {o['xy'][1]:.1f}、幅 {w:.1f} 長さ {l:.1f}mm、画素 ({o['px'][0]:.0f},{o['px'][1]:.0f})")
    cv2.drawContours(img, [o["contour"]], -1, (0, 255, 255), 1)
    cv2.putText(img, str(i + 1), tuple(int(v) for v in o["px"]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.imwrite(os.path.join(OUT, "base_marked.jpg"), img)
json.dump({"tcp": list(tcp), "objs": rows, "fix": TV.FIX.tolist()}, open(os.path.join(OUT, "base.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
