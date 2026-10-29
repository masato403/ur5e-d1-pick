r"""手で隠れた時の合わせ（10/23）の解析：real_ur/logs_hide_1023 の三脚の写真で、腕の真下の物が基準（腕をよけた時）からどれだけずれて見えるか。
cut なし（今の標準：真ん中が腕の所に入る塊は捨てる）と cut あり（腕の所の画素を消して探す）の両方。.venv_ur で real_ur から動かす。"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "real_ur"))
os.environ.setdefault("TRI_DX", "0")
os.environ.setdefault("TRI_DY", "0")
import tripod_view as TV  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "real_ur", "logs_hide_1023")
base = json.load(open(os.path.join(D, "base.json"), encoding="utf-8"))["objs"][:3]
NAME = ["ペン", "消しゴム", "銀の棒"]
rows = json.load(open(os.path.join(D, "points.json"), encoding="utf-8"))
for k, r in enumerate(rows):
    img = cv2.imread(os.path.join(D, f"p{k + 1:02d}_tripod.jpg"))
    tcp = [v / 1000 for v in r["tcp_mm"]] + r["rv"]
    tgt = int(np.argmin([np.linalg.norm(np.array(b["xy"]) - r["tcp_mm"][:2]) for b in base]))
    out = []
    for cut in (False, True):
        rm = TV.robot_mask(img.shape, tcp, TV.CUT_BOXES if cut else None)
        found = []
        for c, cnt, L in TV.blobs(img, rm if cut else None):
            if not cut and rm[int(c[1]), int(c[0])]:
                continue
            found.append((TV.table_center(cnt), TV.size_mm(cnt), cnt))
        # 狙った物に近い塊（80mm 以内）
        b = np.array(base[tgt]["xy"])
        near = [f for f in found if np.linalg.norm(f[0] - b) < 80]
        if near:
            f = min(near, key=lambda f: np.linalg.norm(f[0] - b))
            d = f[0] - b
            out.append(f"{'cut' if cut else '標準'}：ずれ x {d[0]:+.1f} y {d[1]:+.1f}、{f[1][0]:.0f}×{f[1][1]:.0f}mm（基準 {base[tgt]['size'][0]:.0f}×{base[tgt]['size'][1]:.0f}）")
            cv2.drawContours(img, [f[2]], -1, (0, 255, 255) if cut else (255, 0, 255), 1)
        else:
            out.append(f"{'cut' if cut else '標準'}：見つからない")
        # ほかの物が消えていないか
        others = [i for i in range(3) if i != tgt and not any(np.linalg.norm(f[0] - np.array(base[i]["xy"])) < 15 for f in found)]
        if others:
            out[-1] += f"（ほかに見えない物：{'・'.join(NAME[i] for i in others)}）"
    cv2.imwrite(os.path.join(D, f"p{k + 1:02d}_ana.jpg"), img)
    print(f"{k + 1}：{NAME[tgt]}・高さ {r['h']:.0f}  " + " ／ ".join(out))
