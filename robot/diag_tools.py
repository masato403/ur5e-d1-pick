"""どこで間違えているかを分けて調べる（ロボットは動かさない、止まった場面だけ）。

  ① 見分け：番号つき画像で「〇番は何？」（ドライバー・ネジ・ペン・その他）の正解率
     番号の付け方を変えて比べる：corner（物の右上に小さく）／on（物の真ん中に大きく）／crop（その物だけを切り出して見せる）
  ② 指示の読み取り：文章だけで「この指示は〇〇を箱に入れる指示？」の正解率（言い方を変えた指示で）
  ③ 1 つずつ聞く方式：「この物は指示どおりなら箱に入れるべき？」の確率（正解の物とそうでない物で分けて）
  python robot/diag_tools.py [場面の数=40]   → robot/logs/diag_tools.jsonl
"""
import json
import os
import statistics
import sys
import time

import cv2
import mujoco
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
os.chdir(os.path.join(HERE, ".."))

from run_tools import PARK_Q, PART_DESC, PARTS, STATE, SegPerception  # noqa: E402
from sim_tools import KIND, OBJECTS, ToolEnv  # noqa: E402

INSTRS = [  # （指示, 箱に入れるべき部品）
    ("Put all the screws into the box.", {"screw"}),
    ("Put the screwdriver into the box.", {"screwdriver"}),
    ("Put everything except the pen into the box.", {"screwdriver", "screw"}),
    ("Put the pen and the screwdriver into the box.", {"pen", "screwdriver"}),
    ("Only the pen goes in the box.", {"pen"}),
    ("Clear the table: put every object into the box.", {"pen", "screwdriver", "screw"}),
    ("Move the screws and the pen to the box, but leave the screwdriver.", {"screw", "pen"}),
    ("I need the screwdriver and the screws in the box.", {"screwdriver", "screw"}),
]


def marks(rgb, objs, style):
    im = rgb.copy()
    for k, o in objs.items():
        u, v, w, h = o["uv"]
        if style == "corner":
            x, y, r, fs = int(u + w / 2 + 12), int(v - h / 2 - 10), 13, 0.65
        else:                                   # on：物の真ん中に大きく（少し透かす）
            x, y, r, fs = int(u), int(v), 16, 0.8
        cv2.circle(im, (x, y), r, (255, 255, 255), -1)
        cv2.circle(im, (x, y), r, (0, 0, 0), 2)
        cv2.putText(im, str(k), (x - 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2)
    return im


def crop(rgb, o, size=128):
    u, v, w, h = o["uv"]
    half = int(max(w, h) / 2 + 20)
    x1, y1 = max(0, int(u) - half), max(0, int(v) - half)
    c = rgb[y1:int(v) + half, x1:int(u) + half]
    return cv2.resize(c, (size, size))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    from d1_fast import D1Fast
    m = D1Fast(quant="fp8").model
    env = ToolEnv()
    rng = np.random.default_rng(21)
    crit = {p: PART_DESC[p] for p in PARTS + ["other"]}
    ident = {s: {"ok": 0, "n": 0, "ms": [], "conf": {}} for s in ("corner", "on", "crop")}
    one = {"pos": [], "neg": []}
    for i in range(n):
        env.reset(rng, n_in_bin=int(rng.integers(0, 2)))
        per = SegPerception(env.empty_table())
        env.d.qpos[:6] = PARK_Q
        mujoco.mj_forward(env.m, env.d)
        rgb = env.render()
        per.update(rgb, 0)
        objs = per.visible(0)
        truth = {k: KIND[min(OBJECTS, key=lambda ob: np.linalg.norm(env.obj_pos(ob)[:2] - o["xy"]))] for k, o in objs.items()}
        for style in ("corner", "on", "crop"):
            img = None if style == "crop" else Image.fromarray(marks(rgb, objs, style))
            for k, o in objs.items():
                t = time.perf_counter()
                if style == "crop":
                    r = m.system_one("Close-up camera image of one object on a work table.",
                                     {"q": {"type": "choice", "instructions": "What is this object?", "criteria": crit}},
                                     images=[Image.fromarray(crop(rgb, o))])
                else:
                    r = m.system_one(STATE, {"q": {"type": "choice", "instructions": f"What is object number {k}?",
                                                   "criteria": crit}}, images=[img])
                s = ident[style]
                s["ms"].append((time.perf_counter() - t) * 1000)
                a = r["answers"]["q"]["choice"]
                s["n"] += 1
                s["ok"] += a == truth[k]
                key = f"{truth[k]}->{a}"
                s["conf"][key] = s["conf"].get(key, 0) + 1
        if i < 8:          # ③ 1 つずつ聞く方式の確率（時間がかかるので最初の場面だけ）
            img = Image.fromarray(marks(rgb, objs, "corner"))
            for instr, want in INSTRS[:4]:
                for k in objs:
                    r = m.system_one(STATE, {"q": {"type": "noul", "instructions": f"Instruction: {instr} Is object number {k} "
                                                   "one of the objects this instruction asks to put into the box?"}}, images=[img])
                    one["pos" if truth[k] in want else "neg"].append(round(r["answers"]["q"]["noul"], 3))
        if i == 0:
            os.makedirs("robot/som", exist_ok=True)
            cv2.imwrite("robot/som/tools_corner.png", marks(rgb, objs, "corner")[:, :, ::-1])
            cv2.imwrite("robot/som/tools_on.png", marks(rgb, objs, "on")[:, :, ::-1])
        print(i + 1, {s: round(v["ok"] / max(1, v["n"]), 3) for s, v in ident.items()}, flush=True)
    # ② 指示の読み取り（文章だけ）
    parse = {"ok": 0, "n": 0, "ms": [], "wrong": []}
    for instr, want in INSTRS:
        for p in PARTS:
            t = time.perf_counter()
            r = m.system_one(f"Instruction to the robot: {instr}",
                             {"q": {"type": "noul", "instructions": f"Does this instruction ask to put {PART_DESC[p]} (or all of them) into the box?"}})
            parse["ms"].append((time.perf_counter() - t) * 1000)
            y = r["answers"]["q"]["noul"]
            parse["n"] += 1
            ok = (y >= 0.5) == (p in want)
            parse["ok"] += ok
            if not ok:
                parse["wrong"].append({"instr": instr, "part": p, "yes": round(y, 3)})
    out = {"scenes": n,
           "identify": {s: {"acc": round(v["ok"] / v["n"], 3), "n": v["n"], "ms_median": round(statistics.median(v["ms"]), 1),
                            "confusion": v["conf"]} for s, v in ident.items()},
           "parse": {"acc": round(parse["ok"] / parse["n"], 3), "n": parse["n"], "ms_median": round(statistics.median(parse["ms"]), 1),
                     "wrong": parse["wrong"]},
           "onestep_yes": {"pos_median": statistics.median(one["pos"]) if one["pos"] else None, "pos_min": min(one["pos"], default=None),
                           "neg_median": statistics.median(one["neg"]) if one["neg"] else None, "neg_max": max(one["neg"], default=None),
                           "n_pos": len(one["pos"]), "n_neg": len(one["neg"]),
                           "pos_below_0.5": sum(p < 0.5 for p in one["pos"]), "neg_above_0.5": sum(p >= 0.5 for p in one["neg"])}}
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    with open("robot/logs/diag_tools.jsonl", "a", encoding="utf-8") as fp:
        fp.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
