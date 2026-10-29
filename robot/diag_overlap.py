"""先の計算（pick_any.pre_scores、BH_OVERLAP）を使うと、合い具合がわずかに変わる原因を探す（10/19）。
実機で残した写真で、(1) ふつうに survey、(2) 先に pre_scores をしてから survey、を比べ、少しでも違う値を出す。腕は使わない。
使い方:  MAT_V_MIN=90 MAT_S_MAX=60 python robot/diag_overlap.py [日付 20261016] [組の数 49]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scan_archive as SA  # noqa: E402
import pick_any as PA  # noqa: E402
import height_by_motion as HM  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import results_table  # noqa: E402
import time  # noqa: E402

KEYS = ("h", "h_whole", "match")


def run(A, tA, others, pre):
    HM.clear_memo()
    if pre:
        PA.pre_scores(A, tA, others[0][0], others[0][1])
    return PA.survey(A, tA, others)


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else "20261016"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 49
    folder = os.path.join(SA.REAL, "cam_shots", "archive", day)
    diff, total = 0, 0
    for ta, tb, tc in list(SA.triplets(folder))[:n]:
        A, tA = SA.load(folder, ta, "A")
        others = [SA.load(folder, tb, "B"), SA.load(folder, tc, "C")]
        r0, r1 = run(A, tA, others, False), run(A, tA, others, True)
        for i, (f, g) in enumerate(zip(r0, r1)):
            total += 1
            bad = [(k, f[k], g[k]) for k in KEYS if f.get(k) != g.get(k)]
            if not np.array_equal(f["center"], g["center"]):
                bad.append(("center", f["center"], g["center"]))
            if bad:
                diff += 1
                print(f"{ta} {i}: " + "、".join(f"{k} {a} → {b}" for k, a, b in bad), flush=True)
        if len(r0) != len(r1):
            print(f"{ta}: 見つかった数が違う {len(r0)} → {len(r1)}")
    print(f"==== 違った物 {diff}/{total}")
    results_table.save("overlap", {"when": time.strftime("%Y-%m-%d %H:%M"), "what": "先の計算 BH_OVERLAP の有り無し", "n": total, "diff": diff})


if __name__ == "__main__":
    main()
