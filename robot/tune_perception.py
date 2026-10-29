"""記録した画像（robot/rec7）とゲームの本当の位置を比べて、見分け方の正確さを測る。"""
import bisect
import json
import math
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from cam_player import Perception  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "robot/rec7"
tr = [json.loads(l) for l in open(f"{D}/truth.jsonl")]
ts = [r["t"] for r in tr]
frames = [json.loads(l) for l in open(f"{D}/frames.jsonl")]
imgs = {f["i"]: cv2.imread(f"{D}/{f['i']:04d}.png") for f in frames}


def at(t):
    return tr[min(max(bisect.bisect_left(ts, t), 0), len(tr) - 1)]


def evaluate(per, verbose=False):
    seen = [(f, per.find(imgs[f["i"]])) for f in frames]
    best = None
    for off in np.arange(0.0, 0.2, 0.004):
        e = [math.hypot(s["ship"][0] - at(f["wall"] - off)["ship"][0], s["ship"][1] - at(f["wall"] - off)["ship"][1])
             for f, s in seen if s["ship"]]
        m = float(np.median(e)) if e else 999
        if best is None or m < best[1]:
            best = (off, m)
    off = best[0]
    tp = fn = fp = 0
    perr, sdx, sdy, ship_miss = [], [], [], 0
    boss_n = boss_ok = en_n = en_ok = en_fp = 0
    for f, s in seen:
        g = at(f["wall"] - off)
        if s["ship"]:
            sdx.append(s["ship"][0] - g["ship"][0])
            sdy.append(s["ship"][1] - g["ship"][1])
        else:
            ship_miss += 1
        if g["boss"] and g["boss"][1] > 40:
            boss_n += 1
            boss_ok += bool(s["boss"]) and math.hypot(s["boss"][0] - g["boss"][0], s["boss"][1] - g["boss"][1]) < 15
        for e in g["enemies"]:
            if 30 < e[1] < 585:
                en_n += 1
                en_ok += any(math.hypot(x - e[0], y - e[1]) < 12 for x, y in s["enemies"])
        en_fp += sum(1 for x, y in s["enemies"] if not any(math.hypot(x - e[0], y - e[1]) < 12 for e in g["enemies"]))
        tb = [b for b in g["bullets"] if 4 < b[0] < 476 and 32 < b[1] < 592]
        used = set()
        for b in tb:
            d = [(math.hypot(b[0] - x, b[1] - y), j) for j, (x, y, r) in enumerate(s["bullets"]) if j not in used]
            if d and min(d)[0] < 12:
                tp += 1
                used.add(min(d)[1])
                perr.append(min(d)[0])
            else:
                fn += 1
        fp += len(s["bullets"]) - len(used)
    r = {"latency_ms": round(off * 1000), "ship_miss": ship_miss, "ship_dx": round(float(np.median(sdx)), 1),
         "ship_dy": round(float(np.median(sdy)), 1), "ship_err_p90": round(float(np.percentile(np.hypot(sdx, sdy), 90)), 1),
         "bullet_recall": round(tp / max(1, tp + fn), 3), "bullet_precision": round(tp / max(1, tp + fp), 3),
         "bullet_err": round(float(np.median(perr)), 1) if perr else None,
         "boss_rate": round(boss_ok / max(1, boss_n), 3), "enemy_recall": round(en_ok / max(1, en_n), 3), "enemy_fp": en_fp, "n_frames": len(frames)}
    return r


if __name__ == "__main__":
    print(json.dumps(evaluate(Perception())))
