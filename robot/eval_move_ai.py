r"""移動の先（見張りの位置・箱への道筋・箱の中の置き場所）を d1-3B に選ばせたらどうなるか、シミュレーターを回さずに確かめる（10/23）。
本番では答えが 1 つに決まる所なのでプログラムのままにするが、試しに候補を作って選ばせる。候補の結果はプログラムが数字で書く。
使い方:  .venv\Scripts\python.exe robot\eval_move_ai.py   （判断の係 real_ur/d1_server.py を先に動かす）"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "real_ur"))
import kind_ai as KA  # noqa: E402

N = 20
rng = np.random.default_rng(0)


def ask(state, question, choices):
    keys = list(choices)
    order = rng.permutation(len(keys))                           # 選択肢の並びで答えが変わらないかも見るため、毎回並べ替える
    ch = {keys[i]: choices[keys[i]] for i in order}
    return KA.post({"state": state, "question": question, "choices": ch})["choice"]


def look_pos():
    """見張りの位置。正しい答え：三脚と手首から物が全部見えて、腕が何も隠さない所。"""
    ok = 0
    for _ in range(N):
        n_obj = int(rng.integers(1, 4))
        hide = int(rng.integers(1, n_obj + 1))
        part = int(rng.integers(20, 45))
        state = (f"A robot arm is about to look for {n_obj} objects on a cutting mat with a tripod camera and a wrist camera. "
                 "It must first move to a pose to look from.")
        ch = {"mid": "high above the middle of the mat: the tripod and wrist cameras see the whole mat, and the arm hides no object",
              "box": f"above the box at the side: the arm hides {hide} of the {n_obj} objects from the tripod camera",
              "low": f"low near the robot base: the wrist camera sees only {part}% of the mat"}
        ok += ask(state, "Where should the arm look from?", ch) == "mid"
    return ok


def path_to_box():
    """箱へ運ぶ道筋。正しい答え：縁に当たらず、いちばん早い道。"""
    ok = 0
    for _ in range(N):
        rim = float(rng.uniform(15, 40))
        h0 = float(rng.uniform(rim - 15, rim - 3))              # まっすぐ運ぶと、持った物の下の端が縁より下を通る
        up = rim + float(rng.uniform(20, 50))
        extra = float(rng.uniform(0.8, 1.6))
        state = (f"A robot hand holds an object and must carry it into a box. The box rim is {rim:.0f} mm above the table. "
                 f"The bottom of the held object is now {h0:.0f} mm above the table.")
        ch = {"straight": f"move straight to the box at this height: the object passes {rim - h0:.0f} mm below the rim, hits the rim and drops (fails)",
              "up": f"go up to {up:.0f} mm, move over the box, then go down: clears the rim by {up - rim:.0f} mm",
              "high": f"go up to 300 mm, move over the box, then go down: clears the rim by {300 - rim:.0f} mm, but takes {extra:.1f} s longer"}
        ok += ask(state, "How should the hand carry the object into the box?", ch) == "up"
    return ok


def place_spot():
    """箱の中の置き場所。正しい答え：先に入れた物からいちばん離れる所。"""
    ok = 0
    for _ in range(N):
        d = rng.uniform(5, 80, 3)
        names = ["left", "middle", "right"]
        state = "A robot hand holds an object above a box and will drop it inside. Some objects are already in the box."
        ch = {k: f"drop it at the {k} of the box: {v:.0f} mm away from the nearest object already in the box"
                 + (" (it lands on top of that object)" if v < 15 else "") for k, v in zip(names, d)}
        ok += ask(state, "Where in the box should it drop the object?", ch) == names[int(np.argmax(d))]
    return ok


if __name__ == "__main__":
    for name, fn in (("見張りの位置", look_pos), ("箱への道筋", path_to_box), ("箱の中の置き場所", place_spot)):
        print(f"{name}: 正しい {fn()}/{N}", flush=True)
