"""指示の読み取り方を比べる（文章だけ、d1-3B は追加学習なし）。
  A：「〇〇を箱に入れる指示？」（はい・いいえ）
  B：「〇〇はどうする？」→ 箱に入れる／作業台に残す（選択肢）
  C：A と「〇〇を作業台に残す指示？」の両方を聞き、入れる＞残す なら入れる
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
os.chdir(os.path.join(HERE, ".."))

PARTS = {"screwdriver": "the screwdriver", "screw": "the screws", "pen": "the pen"}
INSTRS = [
    ("Put all the screws into the box.", {"screw"}),
    ("Put the screwdriver into the box.", {"screwdriver"}),
    ("Put everything except the pen into the box.", {"screwdriver", "screw"}),
    ("Put the pen and the screwdriver into the box.", {"pen", "screwdriver"}),
    ("Only the pen goes in the box.", {"pen"}),
    ("Clear the table: put every object into the box.", {"pen", "screwdriver", "screw"}),
    ("Move the screws and the pen to the box, but leave the screwdriver.", {"screw", "pen"}),
    ("I need the screwdriver and the screws in the box.", {"screwdriver", "screw"}),
    # 言い方を変えたもの（上の 8 つに合わせすぎないように）
    ("Put everything into the box except the screws.", {"screwdriver", "pen"}),
    ("Leave the screwdriver on the table and box everything else.", {"screw", "pen"}),
    ("All items but the pen should go into the box.", {"screwdriver", "screw"}),
    ("Pick up the screws only.", {"screw"}),
    ("Put the tools into the box, not the pen.", {"screwdriver", "screw"}),
    ("Store the pen in the box.", {"pen"}),
]


def main():
    from d1_fast import D1Fast
    m = D1Fast(quant="fp8").model
    score = {"A": 0, "B": 0, "C": 0}
    inst_ok = {"A": 0, "B": 0, "C": 0}
    wrong = {"A": [], "B": [], "C": []}
    ms = []
    for instr, want in INSTRS:
        st = f"Instruction to the robot: {instr}"
        got = {"A": set(), "B": set(), "C": set()}
        for p, name in PARTS.items():
            t = time.perf_counter()
            put = m.system_one(st, {"q": {"type": "noul", "instructions": f"Does this instruction ask to put {name} into the box?"}})["answers"]["q"]["noul"]
            leave = m.system_one(st, {"q": {"type": "noul", "instructions": f"Does this instruction ask to leave {name} on the table "
                                                                             "(not put it into the box)?"}})["answers"]["q"]["noul"]
            ch = m.system_one(st, {"q": {"type": "choice", "instructions": f"Following this instruction, what should the robot do with {name}?",
                                         "criteria": {"put": "put it into the box", "leave": "leave it on the table"}}})["answers"]["q"]["choice"]
            ms.append((time.perf_counter() - t) * 1000 / 3)
            if put >= 0.5:
                got["A"].add(p)
            if ch == "put":
                got["B"].add(p)
            if put > leave:
                got["C"].add(p)
            for k in "ABC":
                ok = (p in got[k]) == (p in want)
                score[k] += ok
                if not ok:
                    wrong[k].append(f"{instr} / {p} (put {put:.2f}, leave {leave:.2f}, choice {ch})")
        for k in "ABC":
            inst_ok[k] += got[k] == want
    n = len(INSTRS) * len(PARTS)
    out = {k: {"part_acc": round(score[k] / n, 3), "instr_all_correct": f"{inst_ok[k]}/{len(INSTRS)}", "wrong": wrong[k]} for k in "ABC"}
    out["ms_per_question_median"] = round(sorted(ms)[len(ms) // 2], 1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open("robot/logs/test_parse.jsonl", "a", encoding="utf-8") as fp:
        fp.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
