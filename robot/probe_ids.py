"""番号が増える原因を調べる：番号が多かった 5 つの場面を流し、新しい番号がついた瞬間を記録する。
  python robot/probe_ids.py   → robot/logs/probe_ids.jsonl と robot/births/*.png
"""
import gc
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
os.chdir(os.path.join(HERE, ".."))
os.environ.setdefault("BIRTHS", "robot/births")

import torch  # noqa: E402
from run_tools import main  # noqa: E402

CASES = [("Put all the screws into the box.", "screw", 31),
         ("Put all the screws into the box.", "screw", 32),
         ("Put everything except the pen into the box.", "screwdriver,screw", 32),
         ("Put the pen and the screwdriver into the box.", "pen,screwdriver", 31),
         ("Put the pen and the screwdriver into the box.", "pen,screwdriver", 32)]

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    show = ["--show"] if "--show" in sys.argv else []          # --show：ロボットの動きと AI の見ている画像を窓に出す
    done = set()                                               # 終わった場面は飛ばす
    if os.path.exists("robot/logs/probe_ids.jsonl"):
        import json
        done = {(r["instr"], r["seed"]) for r in map(json.loads, open("robot/logs/probe_ids.jsonl", encoding="utf-8"))}
    for instr, want, seed in CASES:
        if (instr, seed) in done:
            continue
        main(["--brain", "twostep", "--instr", instr, "--want", want, "--seed", str(seed), "--log", "robot/logs/probe_ids.jsonl"] + show)
        gc.collect()
