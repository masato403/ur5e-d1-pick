"""d1-3B に任せた 4 つの判断（plan_ai）を、ロボット無しでまとめて確かめる（10/19）。
  ・指示：日本語と英語の指示文で、種類ごとの 箱に入れる／残す が正しい答えと合うか
  ・つかむ高さ：10/16 の実機の写真で測った物（消しゴム・ペン・銀の棒）で、d1-3B の選んだ高さと前の決まりの高さを比べる
  ・どれから運ぶか：近さ・手首を回す角度・まわりの空きで、答えがはっきりしている場面で選べるか
  ・やり直し方：空振り（指が閉じきった）と、すべって抜けた時に、何を選ぶか（もう下げられない時に lower が出ないことも）
結果を robot/logs/results/plan_eval.json に書き、仕様書 19 章の表に入る。判断の係（real_ur/d1_server.py）を先に動かしておく。
使い方:  MAT_V_MIN=90 MAT_S_MAX=60 python robot/eval_plan_ai.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scan_archive as SA  # noqa: E402（実機のプログラムの読み込みと、3 枚の組の探し方を使い回す）
import kind_ai as KA  # noqa: E402
import pick_any as PA  # noqa: E402
import plan_ai as PL  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import results_table  # noqa: E402

E, P, R = "消しゴム", "ペン", "銀の棒"
SAYS = [   # 指示と、箱に入れる種類（正しい答え）
    ("全部の物を箱に入れて", {E, P, R}),
    ("ペン以外を箱に入れて", {E, R}),
    ("消しゴムだけ箱に入れて", {E}),
    ("銀の棒とペンを箱に入れて", {P, R}),
    ("消しゴムは机に残して、ほかは箱に入れて", {P, R}),
    ("ペンを片付けて", {P}),
    ("Put everything except the pen into the box.", {E, R}),
    ("Put only the eraser into the box.", {E}),
    ("Leave the silver rod on the table and box everything else.", {E, P}),
    ("Put the pen and the silver rod into the box.", {P, R}),
]
TCP = [0.0, 0.0, 0.3, np.pi, 0.0, 0.0]   # 手先：根元の真上 300mm、真下向き（指を結ぶ向きは grip_geom.json のまま）


def fake(kind, x, y, ang, w=11.0, l=80.0):
    a = np.radians(ang)
    return {"kind": kind, "center": np.array([x, y, 0.0]), "longv": np.array([np.cos(a), np.sin(a)]), "size": (w, l), "h": w}


def main():
    if not KA.ready():
        sys.exit("判断の係（real_ur/d1_server.py）が動いていない")
    out = {"when": time.strftime("%Y-%m-%d %H:%M")}
    # ---- 指示
    ok = 0
    for text, want in SAYS:
        got = {k for k, v in PL.instruction(text).items() if v}
        ok += got == want
        print(f"指示「{text}」→ 箱に入れる {'・'.join(sorted(got)) or 'なし'}" + ("" if got == want else f"  ← 違う（正しくは {'・'.join(sorted(want))}）"))
    out["say_ok"], out["say_n"] = ok, len(SAYS)
    # ---- つかむ高さ（実機の写真の、最後の 3 組）
    folder = os.path.join(SA.REAL, "cam_shots", "archive", "20261016")
    rows = []
    for ta, tb, tc in list(SA.triplets(folder))[-3:]:
        A, tcpA = SA.load(folder, ta, "A")
        found = PA.survey(A, tcpA, [SA.load(folder, tb, "B"), SA.load(folder, tc, "C")])
        KA.decide(A, found, KA.kind_of, want=lambda f: f["on_mat"] >= PA.MAT_RING and f["h"] >= PA.MIN_H, log=lambda s: None)
        for f in found:
            if f.get("kind"):
                g_ai, g_rule = PL.grip(f), PA.retry_grip(f, 0, False)
                rows.append((f["kind"], g_ai, g_rule))
                print(f"つかむ高さ {f['kind']}：d1-3B {g_ai:.1f}mm ／ 前の決まり {g_rule:.1f}mm" + ("" if abs(g_ai - g_rule) < 0.05 else "  ← 違う"))
    out["grip_same"], out["grip_n"] = sum(abs(a - b) < 0.05 for _, a, b in rows), len(rows)
    out["grip_rows"] = [[k, round(a, 1), round(b, 1)] for k, a, b in rows]
    # ---- どれから運ぶか（正しい答えは、近く・回さず・まわりが空いている物）
    cases = [
        ([fake(E, 20, 10, 90), fake(P, 160, 120, 45), fake(R, -150, 130, 30)], E, "近くて手首を回さない消しゴム"),
        # ペンと銀の棒の間は 3mm（中心の間 14mm − 太さ 11mm）。指が入らないので、離れた消しゴムが先（前は間 19mm で作っていて、
        # ペンを先にしてもよい場面だった。10/19）
        ([fake(P, 30, -20, 90), fake(R, 44, -20, 90), fake(E, -120, 60, 60)], E, "ペンと銀の棒が 3mm でくっついている"),
        ([fake(R, -150, 100, 45), fake(E, 140, -120, 40), fake(P, 15, 0, 90)], P, "近くて回さないペン"),
    ]
    ok = 0
    for cands, want, why in cases:
        got = PL.order(cands, TCP, cands)["kind"]
        ok += got == want
        print(f"どれから運ぶか（{why}）→ {got}" + ("" if got == want else f"  ← 違う（正しくは {want}）"))
    out["order_ok"], out["order_n"] = ok, len(cases)
    # ---- やり直し方
    pen = fake(P, 0, 0, 0)
    pen["round"] = True
    # 1 回目の高さは 6mm（2mm 下げても丸い物の下限 2mm より上）。前は 3.3mm で作っていて、「低く」が選択肢に入っていなかった（10/19）
    tries = [("空振り・1 回目", pen, 0, 6.0, True), ("すべって抜けた・1 回目", pen, 0, 6.0, False),
             ("空振り・もう下げられない", pen, 0, PA.ROUND_MIN + 1, True), ("やり直しの上限", pen, PA.RETRY, 3.3, True)]
    out["retry_rows"] = []
    for name, f, att, g, empty in tries:
        how = PL.retry(f, att, g, empty)
        out["retry_rows"].append([name, how])
        print(f"やり直し方（{name}）→ {how}")
    out["retry_lower_blocked"] = out["retry_rows"][2][1] != "lower"
    results_table.save("plan_eval", out)
    print(f"==== 指示 {out['say_ok']}/{out['say_n']}、つかむ高さ 前と同じ {out['grip_same']}/{out['grip_n']}、"
          f"どれから {out['order_ok']}/{out['order_n']}、やり直し {out['retry_rows']}")


if __name__ == "__main__":
    main()
