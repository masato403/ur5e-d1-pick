"""種類のほかに、ロボットが迷う所を、ローカルの AI「d1-3B」に Jev 方式で選ばせる（10/19。d1-3B を使うのがこの課題の目的）。
  ・指示を読む（instruction）：指示文を読み、種類ごとに「箱に入れる／机に残す」の 2 択
  ・どれから運ぶか（order）：プログラムが候補ごとに、今の手先からの距離・手首を回す角度・隣の物までの距離を計算し、選択肢の説明に書く
  ・つかみ方（grip）：プログラムが、指先を下ろす高さの候補と、その高さで起きること（試験で分かったこと）を説明に書く
  ・持てなかった時（retry）：同じ高さ／低くする／やめる の中から選ぶ。できない選択肢（もう下げられない、など）は先に外す
数字の比べや安全に関わることはプログラムがして、d1-3B には選ぶことだけをさせる。判断の係は kind_ai と同じ（d1_server.py）。
どれも PLAN_AI=0 で、前の決まり（プログラムだけで決める）に戻せる。
"""
import math
import os
import time

import numpy as np

import kind_ai as KA
import pick_any as PA
import pick_by_camera as PB

USE_AI = os.environ.get("PLAN_AI", "1") == "1"        # 0 の時は、前の決まりで決める（前の決め方に戻す設定）
MIN_GAP = float(os.environ.get("PLAN_MIN_GAP", "8"))   # 隣の物までの空き（mm）がこれより狭い物は、どれから運ぶかの選択肢から外す
ENG = {"消しゴム": "the eraser", "ペン": "the pen", "銀の棒": "the silver rod"}
LOG = print
PLAN_TEXT = int(os.environ.get("PLAN_TEXT", "0"))      # 問いの書き方。0：前の文。1：選んだ結果を数字で書き、問いは中立（10/22。追い方の問いで 60/90 → 90/90 だった書き方）


def choose(state, question, choices, timeout=10.0):
    """判断の係に 1 問聞く。戻り値：(選んだ名前, 確率の辞書, ms)。選択肢が 1 つなら聞かずにそれを返す。"""
    if len(choices) == 1:
        return next(iter(choices)), {next(iter(choices)): 1.0}, 0.0
    body = {"state": state, "question": question, "choices": choices}
    t0 = time.perf_counter()
    r = KA.post(body, timeout)
    return r["choice"], r["probs"], (time.perf_counter() - t0) * 1000


def _show(what, name, probs, ms, label=lambda k: k):
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
    LOG(f"   d1-3B（{what}）：{label(name)}（" + "、".join(f"{label(k)} {p:.2f}" for k, p in top) + f"、{ms:.0f}ms）")


# ------------------------------------------------------------------ 指示を読む
def instruction(text, kinds=tuple(ENG)):
    """指示文から、種類ごとに箱に入れるか（True）残すか（False）を決める。はい・いいえで聞くより 2 択の方が「〜以外」に強かった
    （10/14 のシミュレーター：2 択 14/14、はい・いいえ 8/14）。"""
    out = {}
    for k in kinds:
        if not USE_AI:
            out[k] = True                                              # 前の決め方：全部運ぶ
            continue
        name, probs, ms = choose(f"Instruction to the robot: {text}",
                                 f"Following this instruction, what should the robot do with {ENG[k]}?",
                                 {"put": "put it into the box", "leave": "leave it on the table"})
        out[k] = name == "put"
        _show(f"指示「{text}」の {k}", name, probs, ms, {"put": "箱に入れる", "leave": "残す"}.get)
    return out


# ------------------------------------------------------------------ どれから運ぶか
def turn_deg(f, tcp):
    """つかむ時に手首の先を回す角度（度、絶対値）。pick_by_camera.grasp と同じ計算（指を結ぶ向きを長い辺と直角にする）。"""
    Rt = PB.rotvec_to_R(tcp[3:])
    cur_y = Rt @ np.array(PB.GEOM["close_dir_tool"])
    want = np.array([-f["longv"][1], f["longv"][0], 0.0])
    if np.dot(want, cur_y) < 0:
        want = -want
    return abs(math.degrees(math.atan2(np.cross(cur_y, want)[2], np.dot(cur_y, want))))


def facts_of(f, tcp, others):
    """候補 1 つの、プログラムが計算した値。"""
    far = math.hypot(f["center"][0] - tcp[0] * 1000, f["center"][1] - tcp[1] * 1000)
    gap = min((float(np.linalg.norm(np.asarray(g["center"][:2]) - np.asarray(f["center"][:2])))
               - (g["size"][0] + f["size"][0]) / 2 for g in others if g is not f), default=999.0)
    return far, turn_deg(f, tcp), max(0.0, gap)


def order(cands, tcp, everything):
    """候補（つかめると決まった物）から、次に運ぶ 1 つを選ぶ。everything：見つけた物すべて（隣の物までの距離に使う）。"""
    if len(cands) == 1 or not USE_AI:
        return cands[0]
    # 指が入らないほど隣の物に近い物は、聞く前に外す（説明に「3mm」と書いても d1-3B はそれを選んだ。距離の比べは苦手。10/19）。
    # 全部が狭い時は外さない（どれかは運ぶ）
    roomy = [f for f in cands if facts_of(f, tcp, everything)[2] >= MIN_GAP]
    if roomy and len(roomy) < len(cands):
        LOG(f"   隣の物まで {MIN_GAP:.0f}mm より狭い物は、先に運ぶ候補から外す："
            + "・".join(str(f.get("kind")) for f in cands if f not in roomy))
        cands = roomy
        if len(cands) == 1:
            return cands[0]
    choices, by = {}, {}
    for i, f in enumerate(cands):
        far, turn, gap = facts_of(f, tcp, everything)
        key = f"object_{i + 1}"
        by[key] = f
        choices[key] = (f"{ENG.get(f.get('kind'), 'an object')}: {far:.0f} mm from the gripper, the wrist must turn {turn:.0f} degrees, "
                        f"the nearest other object is {gap:.0f} mm away from its side")
    if PLAN_TEXT >= 1:                                           # 選んだ結果（着くまでの秒・指が隣に当たるか）を書き、問いは中立に
        for key, f in by.items():
            far, turn, gap = facts_of(f, tcp, everything)
            t_go = far / (PB.SPEED * 1000) + turn / 90.0           # 着くまでの目安（腕の速さ ＋ 手首を 90°/秒 で回す）
            choices[key] = (f"Pick up {ENG.get(f.get('kind'), 'an object')} first: the gripper gets there in about {t_go:.1f} s "
                            f"({far:.0f} mm away, wrist turns {turn:.0f} degrees). "
                            + ("The fingers have room around it." if gap >= 2 * MIN_GAP else
                               f"Another object is only {gap:.0f} mm from its side, so a finger may hit it and the grasp may fail."))
        name, probs, ms = choose("A robot arm with a two-finger gripper must pick up these objects one at a time and carry them away.",
                                 "Which object should the robot pick up first?", choices)
        _show("どれから運ぶか", name, probs, ms, lambda k: by[k].get("kind") or k)
        return by[name]
    name, probs, ms = choose("A robot arm with a two-finger gripper must pick up these objects one at a time and carry them away. "
                             "All of them can be picked up safely.",
                             "Which object should the robot pick up first? Prefer one that is close to the gripper, needs a small "
                             "wrist turn, and has free space around it so the fingers do not hit another object.", choices)
    _show("どれから運ぶか", name, probs, ms, lambda k: by[k].get("kind") or k)
    return by[name]


# ------------------------------------------------------------------ つかみ方
def grip_options(f, empty):
    """指先を下ろす高さ（机から mm）の候補と、その高さで起きることの説明。説明は、試験で分かったことをプログラムが書く。"""
    h = min(f["h"], f.get("h_whole", f["h"])) if empty else f["h"]
    rnd = bool(f.get("round"))
    lo = PA.ROUND_MIN if (rnd and f.get("grip_on") != "thin") else PA.GRIP_MIN
    opts = {}

    def add(key, mm, what):
        mm = max(lo, min(PA.GRIP_MAX, mm))
        if all(abs(mm - v[0]) > 0.9 for v in opts.values()):        # 同じ高さになる候補は 1 つにする
            opts[key] = (mm, what)
    if f.get("grip_on") == "thin":
        add("below_shaft", h / 2 - f["size"][0] / 2 - PA.THIN_BELOW,
            "fingertips just below the bottom of the thin shaft, so the finger faces close around the shaft")
    add("half", h / 2, "fingertips at half the object's height" +
        (": on a round object this is exactly its widest point, where the fingertips only touch and can slip off"
         + (", so the object often drops while lifting" if PLAN_TEXT >= 1 else "") if rnd else
         ": the finger faces press on the flat sides"))
    if rnd:
        add("low", h * PA.ROUND_GRIP, "fingertips low, below the widest point of the round object, so the finger faces wrap around it")
    add("deep", lo, "fingertips as low as allowed, just above the table")
    return opts


def grip(f, drop=0.0, empty=False):
    """指先を下ろす高さ（机から mm）。drop：やり直しで下げる量（mm）。"""
    if not USE_AI:
        return PA.retry_grip(f, int(round(drop / PA.GRIP_STEP)), empty)
    opts = grip_options(f, empty)
    w, l = f["size"]
    state = (f"A two-finger gripper will grasp {ENG.get(f.get('kind'), 'an object')} lying on a table, from above. "
             f"Measured: {l:.0f} mm long, {w:.0f} mm wide, top {f['h']:.0f} mm above the table, "
             + ("round cross-section (a cylinder lying on its side)." if f.get("round") else "flat sides (a box shape)."))
    if f.get("grip_on") == "thin":
        state += " The gripper will hold its thin shaft."
    q = ("How deep should the fingertips go down before closing?" if PLAN_TEXT >= 1 else
         "How deep should the fingertips go down before closing, so the object is held firmly and does not slip?")
    name, probs, ms = choose(state, q,
                             {k: f"{v[0]:.1f} mm above the table: {v[1]}" for k, v in opts.items()})
    _show("つかむ高さ", name, probs, ms, lambda k: f"{k} {opts[k][0]:.1f}mm")
    lo = PA.ROUND_MIN if (f.get("round") and f.get("grip_on") != "thin") else PA.GRIP_MIN
    return max(lo, opts[name][0] - drop)


# ------------------------------------------------------------------ 持てなかった時
def retry(f, attempt, grip_mm, empty):
    """持てなかった時のやり直し方："same"（撮り直して同じ高さ）・"lower"（撮り直して GRIP_STEP mm 低く）・"skip"（その物はやめる）・
    "slow"（滑った時だけ：撮り直して同じ高さ、持ち上げる速さを半分に）。
    やり直しの回数が上限なら聞かずに skip。もう下げられないなら lower を外す。"""
    if attempt >= PA.RETRY:
        return "skip"
    lo = PA.ROUND_MIN if (f.get("round") and f.get("grip_on") != "thin") else PA.GRIP_MIN
    if not USE_AI:
        return "lower"                                                  # 前の決まり：撮り直して低くする
    # 説明には、試験で分かったことを書く（空振りは、物の上で閉じていて、低くすると持てた。10/16）。
    # 書かないと、空振りの後に「やめる」を選んだ（10/19、ロボット無しの確かめ）
    choices = {"same": "measure the object again and try again at the same planned height"
                       + (": the fingers already reached the object, so it usually works after re-measuring" if not empty else
                          ": after a miss the height is re-measured over the whole object, which is often lower, so this can work"),
               # 滑って落とした後（指が物に届いていた）に、まだやり直せるのに「やめる」を 0.67〜0.72 で選んだ（10/19 夜の実機 2 回）。
               # やめると指示が終わらないこと・やり直しの回数が残っていることを、説明に書く
               "skip": (f"give up on this object: it stays on the table, the instruction is not completed, "
                        f"and the {PA.RETRY - attempt} remaining tries are wasted")}
    if not empty:
        # 滑って落とした時の選択肢（10/19 夜）：持てていたのに持ち上げる途中で抜けたので、持ち上げる速さを半分にする
        choices["slow"] = ("measure the object again, grip it at the same height and lift it at half speed, "
                           "so it does not slip out of the fingers while lifting")
    if grip_mm - PA.GRIP_STEP >= lo:
        choices["lower"] = (f"measure the object again and try again {PA.GRIP_STEP:.0f} mm lower"
                            + (": when the fingers closed completely they were above the object, and going lower usually works" if empty else
                               ": the fingertips reach further under the widest part"))
    state = (f"The gripper tried to pick up {ENG.get(f.get('kind'), 'an object')} with the fingertips {grip_mm:.1f} mm above the table "
             f"(its top is {f['h']:.0f} mm above the table), and it was not holding it after lifting. "
             + ("The fingers closed completely, so they missed the object (probably closed above it). " if empty else
                "The fingers did not close completely, so they held the object at first and it slipped out while lifting; "
                "the object is still on the table and can be picked again. ")
             + f"This was try {attempt + 1}; {PA.RETRY - attempt} more tries are allowed.")
    name, probs, ms = choose(state, "What should the robot do now?", choices)
    _show("やり直し方", name, probs, ms, {"same": "同じ高さ", "lower": "低く", "skip": "やめる", "slow": "ゆっくり持ち上げる"}.get)
    return name
