"""実機で残した写真（cam_shots/archive/日付）の、つかむ候補の物すべてについて、種類を d1-3B（kind_ai.ask）に聞き、
長さと幅の表（kind_ai.kind_of。10/19 に写真 49 組で確かめた答え）と比べる。切り抜いた絵は robot/logs/kind_ai_<日付>/ に残す。
判断の係（real_ur/d1_server.py）を先に動かしておく。
使い方:  MAT_V_MIN=90 MAT_S_MAX=60 python robot/eval_kind_ai.py [日付 20261016]
"""
import os
import sys
from collections import Counter

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scan_archive as SA  # noqa: E402（実機のプログラムの読み込みと、3 枚の組の探し方を使い回す）
import kind_ai as KA  # noqa: E402
import pick_any as PA  # noqa: E402
from kind_ai import kind_of  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import results_table  # noqa: E402（結果を robot/logs/results/kind_eval.json に残し、仕様書の表に入れる）
import time  # noqa: E402


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else "20261016"
    folder = os.path.join(SA.REAL, "cam_shots", "archive", day)
    out = os.path.join(HERE, "logs", f"kind_ai_{day}")
    os.makedirs(out, exist_ok=True)
    if not KA.ready():
        sys.exit("判断の係（real_ur/d1_server.py）が動いていない")
    agree, total, ms, trip = Counter(), 0, [], 0
    for ta, tb, tc in SA.triplets(folder):
        trip += 1
        A, tcpA = SA.load(folder, ta, "A")
        found = PA.survey(A, tcpA, [SA.load(folder, tb, "B"), SA.load(folder, tc, "C")])
        for i, f in enumerate(found):
            if f["on_mat"] < PA.MAT_RING or f["h"] < PA.MIN_H:
                continue
            rule = kind_of(f)
            ai, probs, t = KA.ask(A, f)
            ms.append(t)
            total += 1
            agree[(rule, ai)] += 1
            p = max(probs.values())
            mark = "" if rule == ai else "  ← 違う"
            print(f"{ta} {i}: 表 {rule or 'それ以外'} ／ d1 {ai or 'それ以外'}（{p:.2f}）"
                  f" 長さ {f['size'][1]:.0f} 幅 {f['size'][0]:.0f} 高さ {f['h']:.0f}{mark}", flush=True)
            cv2.imwrite(os.path.join(out, f"{ta}_{i}_{ {v: k for k, v in KA.NAME.items()}.get(ai, 'other')}.jpg"), KA.crop_upright(A, f["contour"]))   # OpenCV は日本語のファイル名を書けない
    same = sum(v for (a, b), v in agree.items() if a == b)
    print(f"==== 表と d1 の答えが同じ {same}/{total}、d1 の 1 回の時間 平均 {np.mean(ms):.0f}ms（最大 {np.max(ms):.0f}ms）")
    pairs = sorted(agree.items(), key=lambda kv: -kv[1])
    for (a, b), v in pairs:
        print(f"   表 {a or 'それ以外'} → d1 {b or 'それ以外'}：{v}")
    results_table.save("kind_eval", {
        "when": time.strftime("%Y-%m-%d %H:%M"), "day": f"{day[4:6]}/{day[6:]}", "triplets": trip, "total": total, "same": same,
        "pairs": [[a or "それ以外", b or "それ以外", v] for (a, b), v in pairs],
        "ms_mean": float(np.mean(ms)), "ms_max": float(np.max(ms))})


if __name__ == "__main__":
    main()
