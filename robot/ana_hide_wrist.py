r"""手で隠れた時の合わせ（10/23）の解析・手首の側：real_ur/logs_hide_1023 の手首の写真で、真下の物の位置と大きさを、三脚の基準（腕をよけた時）と比べる。
.venv_ur で real_ur から動かす。物の上の面は机から H_OBJ mm とする。"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "real_ur"))
os.environ.setdefault("TRI_DX", "0")
os.environ.setdefault("TRI_DY", "0")
import measure as M  # noqa: E402
import track_real as TR  # noqa: E402

H_OBJ = float(os.environ.get("H_OBJ", 10.0))
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "real_ur", "logs_hide_1023")
base = json.load(open(os.path.join(D, "base.json"), encoding="utf-8"))["objs"][:3]
NAME = ["ペン", "消しゴム", "銀の棒"]
NOM = [TR.SIZE_NOM["ペン"], TR.SIZE_NOM["消しゴム"], TR.SIZE_NOM["銀の棒"]]
rows = json.load(open(os.path.join(D, "points.json"), encoding="utf-8"))
for k, r in enumerate(rows):
    img = cv2.imread(os.path.join(D, f"p{k + 1:02d}_wrist.jpg"))
    tcp = [v / 1000 for v in r["tcp_mm"]] + r["rv"]
    tgt = int(np.argmin([np.linalg.norm(np.array(b["xy"]) - r["tcp_mm"][:2]) for b in base]))
    b = np.array(base[tgt]["xy"])
    H_, W_ = img.shape[:2]
    found = []
    for cnt in M.find_objects(img)[0]:
        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        cut = x_ <= 3 or y_ <= 3 or x_ + w_ >= W_ - 3 or y_ + h_ >= H_ - 3
        pts = np.array([TR.wrist_xy(p, tcp, H_OBJ) for p in cnt.reshape(-1, 2)[::4].astype(float)], np.float32)
        if len(pts) < 5:
            continue
        (cx, cy), (a, bb), _ = cv2.minAreaRect(pts)
        found.append((np.array([cx, cy]), sorted((a, bb)), cut, cnt))
    near = [f for f in found if np.linalg.norm(f[0] - b) < 80]
    if near:
        f = min(near, key=lambda f: np.linalg.norm(f[0] - b))
        d = f[0] - b
        msg = (f"ずれ x {d[0]:+.1f} y {d[1]:+.1f}、{f[1][0]:.0f}×{f[1][1]:.0f}mm（本当 {NOM[tgt][0]}×{NOM[tgt][1]}、三脚 {base[tgt]['size'][0]:.0f}×{base[tgt]['size'][1]:.0f}）"
               + ("、写真の端で切れる" if f[2] else ""))
        cv2.drawContours(img, [f[3]], -1, (0, 255, 255), 3)
    else:
        msg = f"見つからない（塊 {len(found)} 個）"
    cv2.imwrite(os.path.join(D, f"p{k + 1:02d}_wrist_ana.jpg"), cv2.resize(img, None, fx=0.5, fy=0.5))
    print(f"{k + 1}：{NAME[tgt]}・高さ {r['h']:.0f}  {msg}")
