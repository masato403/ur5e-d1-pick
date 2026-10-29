"""d1-3B が「次に運ぶ積み木」を正しく選べるかを、シミュレーターの画像で確かめる。

指示ごとに、箱に入っている積み木の組み合わせ（8通り）×置き方（数通り）の場面を作り、
正解（指示から決まる）と d1-3B の答えを比べる。
  python robot/test_d1_brain.py
"""
import json
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
os.chdir(os.path.join(HERE, ".."))

from run_pick import SLOTS, D1Brain  # noqa: E402
from sim_env import BIN, BLOCKS, HALF, Env  # noqa: E402

# 指示と、正しい答え（箱に入っているもの → 次に運ぶもの）
TASKS = [
    ("Put all blocks into the yellow box.", None),               # 残っていればどれでもよい
    ("Put only the red block into the yellow box.", ["red"]),
    ("Put the blue and green blocks into the yellow box. Leave the red block.", ["blue", "green"]),
    ("Put all blocks into the yellow box, the green one first, then red, then blue.", ["green", "red", "blue"]),
]


def answer_ok(task, done, choice):
    instr, want = task
    if want is None:
        left = [b for b in BLOCKS if b not in done]
        return choice in left if left else choice == "finished"
    left = [b for b in want if b not in done]
    if not left:
        return choice == "finished"
    if "first" in instr:                       # 順番の指定あり
        return choice == left[0]
    return choice in left


def main():
    env = Env()
    rng = np.random.default_rng(7)
    out = []
    d1 = None
    for task in TASKS:
        brain = D1Brain(task[0], d1=d1)
        d1 = brain.d1
        ok, n, ms = 0, 0, []
        for k in range(8):
            done = {b for i, b in enumerate(BLOCKS) if k >> i & 1}
            for _ in range(3):
                env.reset(rng)
                for j, b in enumerate(sorted(done)):
                    env.set_block(b, BIN + SLOTS[j], HALF)
                import mujoco
                mujoco.mj_forward(env.m, env.d)
                frame = env.render()
                brain.ask(frame, done)
                while brain.busy:
                    pass
                choice = brain.result[0]
                good = answer_ok(task, done, choice)
                ok += good
                n += 1
                if not good:
                    print("NG", task[0][:40], sorted(done), "->", choice, flush=True)
        r = {"instr": task[0], "acc": round(100 * ok / n), "n": n,
             "ms_avg": round(statistics.mean(brain.ms[3:] or brain.ms), 1)}
        print(json.dumps(r), flush=True)
        out.append(r)
        del brain
        import torch
        torch.cuda.empty_cache()
    with open("robot/logs/d1_brain.jsonl", "a", encoding="utf-8") as fp:
        for r in out:
            fp.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
