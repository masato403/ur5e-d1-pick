"""カメラで読んだ位置と、ゲームの本当の位置（--dump）を比べる。  python robot/check_truth.py logs/cam_xxx.jsonl logs/truth.jsonl"""
import bisect
import json
import math
import sys

import numpy as np

cam = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if '"summary"' not in l]
tr = [json.loads(l) for l in open(sys.argv[2])]
ts = [r["t"] for r in tr]


def at(t):
    i = bisect.bisect_left(ts, t)
    return tr[min(max(i, 0), len(tr) - 1)]


best = None
for off in np.arange(0, 0.25, 0.004):
    e = [math.hypot(c["ship"][0] - at(c["wall"] - off)["ship"][0], c["ship"][1] - at(c["wall"] - off)["ship"][1]) for c in cam if c["ship"]]
    m = float(np.median(e))
    if best is None or m < best[1]:
        best = (off, m)
off = best[0]
found = miss = 0
perr, verr, dy = [], [], []
for c in cam:
    g = at(c["wall"] - off)
    for b in g["bullets"]:
        if not (5 < b[0] < 475 and 35 < b[1] < 590):
            continue
        d = [(math.hypot(b[0] - x, b[1] - y), vx, vy, y - b[1]) for x, y, vx, vy in c["bullets"]]
        m = min(d) if d else None
        if m and m[0] < 15:
            found += 1
            perr.append(m[0])
            verr.append(math.hypot(m[1] - b[2], m[2] - b[3]))
            dy.append(m[3])
        else:
            miss += 1
print(json.dumps({"latency_ms": round(off * 1000), "ship_err_px": round(best[1], 1), "bullets_found": found, "bullets_missed": miss,
                  "pos_err_median": round(float(np.median(perr)), 1), "pos_err_p90": round(float(np.percentile(perr, 90)), 1),
                  "y_bias": round(float(np.median(dy)), 1), "vel_err_median": round(float(np.median(verr)), 2)}))
