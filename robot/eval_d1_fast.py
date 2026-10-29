"""判断の係の速い動かし方（d1_judge_fast）が、元の動かし方（model.system_one）と同じ答えを出すかを確かめる（10/19）。
実機の写真の種類の問い（eval_kind_ai と同じ 113 個）と、4 つの判断の問い（eval_plan_ai と同じ）を集め、同じ問いを
"path": "ref"（元）と "fast"（速い）で聞き比べる。選んだ答えが 1 つでも違えば、速い動かし方は使わない。
結果を robot/logs/results/d1_fast.json に書き、仕様書 19 章の表に入る。判断の係（real_ur/d1_server.py）を先に動かしておく。
使い方:  MAT_V_MIN=90 MAT_S_MAX=60 python robot/eval_d1_fast.py
"""
import contextlib
import io
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scan_archive as SA  # noqa: E402
import kind_ai as KA  # noqa: E402
import pick_any as PA  # noqa: E402
import eval_plan_ai as EP  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import results_table  # noqa: E402


def main():
    if not KA.ready():
        sys.exit("判断の係（real_ur/d1_server.py）が動いていない")
    KA.RECORD = []
    folder = os.path.join(SA.REAL, "cam_shots", "archive", "20261016")
    with contextlib.redirect_stdout(io.StringIO()):            # 集める間の表示は出さない
        for ta, tb, tc in SA.triplets(folder):
            A, tA = SA.load(folder, ta, "A")
            found = PA.survey(A, tA, [SA.load(folder, tb, "B"), SA.load(folder, tc, "C")])
            KA.decide(A, found, KA.kind_of, want=lambda f: f["on_mat"] >= PA.MAT_RING and f["h"] >= PA.MIN_H, log=lambda s: None)
        n_kind = len(KA.RECORD)
        EP.main()
    bodies, KA.RECORD = KA.RECORD, None
    print(f"集めた問い：種類 {n_kind}、4 つの判断 {len(bodies) - n_kind}")
    same, dmax, ms = 0, 0.0, {"ref": [], "fast": []}
    for i, b in enumerate(bodies):
        r = {p: KA.post(dict(b, path=p)) for p in ("ref", "fast")}
        for p in r:
            ms[p].append(r[p]["ms"])
        ok = r["ref"]["choice"] == r["fast"]["choice"]
        same += ok
        d = max(abs(r["ref"]["probs"][k] - r["fast"]["probs"][k]) for k in r["ref"]["probs"])
        dmax = max(dmax, d)
        if not ok:
            print(f"  ← 違う：問い {i}（{'種類' if i < n_kind else '判断'}） 元 {r['ref']['choice']} {r['ref']['probs']} ／ 速い {r['fast']['choice']} {r['fast']['probs']}")
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "n": len(bodies), "same": same, "dmax": dmax,
           "ref_img": float(np.median([ms["ref"][i] for i in range(n_kind)])), "fast_img": float(np.median([ms["fast"][i] for i in range(n_kind)])),
           "ref_txt": float(np.median(ms["ref"][n_kind:])), "fast_txt": float(np.median(ms["fast"][n_kind:]))}
    results_table.save("d1_fast", out)
    print(f"==== 同じ答え {same}/{len(bodies)}、確率の差 最大 {dmax:.4f}、"
          f"画像つき 元 {out['ref_img']:.0f}ms → 速い {out['fast_img']:.0f}ms、文章だけ 元 {out['ref_txt']:.0f}ms → 速い {out['fast_txt']:.0f}ms（中央値）")


if __name__ == "__main__":
    main()
