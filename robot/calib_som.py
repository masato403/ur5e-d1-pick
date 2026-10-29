"""「この物は指示どおりなら箱に入れるべき？」（はい・いいえ）の答えの確率を、止まった場面で集めて、線引きを決める。
  python robot/calib_som.py [場面の数=20]
"""
import json
import time
import os
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
os.chdir(os.path.join(HERE, ".."))

from run_pick_som import SegPerception, SomBrain, draw_marks, empty_table, in_box  # noqa: E402
from sim_env import BLOCKS, Env  # noqa: E402

TASKS = [("Put all blocks into the box.", {"red", "blue", "green"}),
         ("Put only the blue block into the box.", {"blue"}),
         ("Put the red and green blocks into the box. Leave the blue block on the table.", {"red", "green"}),
         ("Put every block that is not green into the box.", {"red", "blue"}),
         ("Put only the red block into the box.", {"red"}),
         ("Put the green block into the box.", {"green"})]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    env = Env()
    rng = np.random.default_rng(11)
    brain = SomBrain("")
    rows = []
    for i in range(n):
        env.reset(rng, n_in_bin=int(rng.integers(0, 2)))
        keep = {b: env.block_pos(b) for b in BLOCKS}
        per = SegPerception(empty_table(env))
        for b in BLOCKS:
            env.set_block(b, keep[b][:2], keep[b][2])
        env.d.qpos[:6] = env.d.qpos[:6]
        from sim_env import HOME
        env.d.qpos[:6] = HOME
        mujoco.mj_forward(env.m, env.d)
        rgb = env.render()
        per.update(rgb, 0)
        objs = per.visible(0)
        # 番号 → 本当の色（答え合わせ用：いちばん近い積み木）
        truth = {k: min(BLOCKS, key=lambda b: np.linalg.norm(env.block_pos(b)[:2] - o["xy"])) for k, o in objs.items()}
        for instr, want in TASKS:
            brain.instr = instr
            brain.ask(rgb, objs)
            while brain.busy:
                time.sleep(0.002)         # 休まず確かめ続けると、判断のスレッドが動けず遅くなる
            for k, p in brain.last["yes"].items():
                kk = int(k[3:])
                rows.append({"scene": i, "instr": instr, "color": truth[kk], "yes": p, "label": truth[kk] in want})
        if i == 0:
            cv2.imwrite("robot/som/calib0.png", draw_marks(rgb, objs)[:, :, ::-1])
    pos = sorted(r["yes"] for r in rows if r["label"])
    neg = sorted(r["yes"] for r in rows if not r["label"])
    print("positives", len(pos), "min/median", pos[0], pos[len(pos) // 2], "| negatives", len(neg), "median/max", neg[len(neg) // 2], neg[-1])
    for th in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        acc = (sum(p >= th for p in pos) + sum(p < th for p in neg)) / (len(pos) + len(neg))
        print(f"threshold {th}: accuracy {acc:.3f}  missed {sum(p < th for p in pos)}  wrong {sum(p >= th for p in neg)}")
    # 並び順だけで正しく選べる率（正しい物が 1 つ以上あるとき、いちばん高いのが正しい物か）
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[(r["scene"], r["instr"])].append(r)
    top = [max(v, key=lambda r: r["yes"])["label"] for v in g.values() if any(r["label"] for r in v)]
    print("top choice correct", round(sum(top) / len(top), 3), "of", len(top))
    with open("robot/logs/calib_som.jsonl", "w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
