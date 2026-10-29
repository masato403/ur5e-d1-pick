"""箱の所を探す範囲から外す直し（teach_box.mask_px・pick_any.BOX。10/19 夜）を、実機の写真（箱は写っていない）で確かめる。
組ごとに 2 通りの仮の箱を置く：
  1) 物の無い所（見つけた物から最も離れたマットの上）→ 見つかる物が、箱なしと全く同じになるか
  2) 1 つの物の真上 → その物だけが消え、他の物は同じか
使い方:  MAT_V_MIN=90 MAT_S_MAX=60 python robot/eval_box_mask.py [日付 20261016] [組の数 12]
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scan_archive as SA  # noqa: E402
import pick_any as PA  # noqa: E402
import measure as M  # noqa: E402
import teach_box as TB  # noqa: E402


def key(found):
    return [(round(float(f["center"][0]), 2), round(float(f["center"][1]), 2), round(f["h"], 2), round(f["size"][0], 2)) for f in found]


def fake_box(xy, axis=(0.0, 1.0)):
    return {"center": [float(xy[0]), float(xy[1])], "axis": list(axis), "length": 170.0, "rim": 50.0, "when": "仮"}


def obj_masks(A, found):
    ms = []
    for f in found:
        m = np.zeros(A.shape[:2], np.uint8)
        cv2.drawContours(m, [f["contour"]], -1, 255, -1)
        ms.append(cv2.dilate(m, np.ones((15, 15), np.uint8)))
    return ms


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else "20261016"
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    folder = os.path.join(SA.REAL, "cam_shots", "archive", day)
    ok_empty = ok_on = n = n_empty = 0
    for ta, tb, tc in list(SA.triplets(folder))[:n_max]:
        A, tcpA = SA.load(folder, ta, "A")
        others = [SA.load(folder, tb, "B"), SA.load(folder, tc, "C")]
        PA.BOX = None
        base = PA.survey(A, tcpA, others)
        if not base:
            continue
        n += 1
        oms = obj_masks(A, base)
        _, mat = M.find_objects(A)                               # マットの外形の中だけに置く
        filled = np.zeros_like(mat)
        cv2.drawContours(filled, cv2.findContours(mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, 255, -1)
        c0 = np.array(tcpA[:2]) * 1000
        # 1) マットの上で、どの物にもかからない所（見張りの位置に近い順に、長い向き x・y の両方を試す）
        spot = None
        grid = sorted([c0 + [dx, dy] for dx in range(-200, 201, 10) for dy in range(-200, 201, 10)], key=lambda g: np.linalg.norm(g - c0))
        for g in grid:
            for ax in ((0.0, 1.0), (1.0, 0.0)):
                bm = TB.mask_px(fake_box(g, ax), tcpA, A.shape)
                if (bm > 0).sum() and ((filled[bm > 0] > 0).mean() > 0.97) and not any((bm & om).any() for om in oms):
                    spot = (g, ax)
                    break
            if spot:
                break
        msg1 = "置ける所が無い"
        if spot:
            n_empty += 1
            PA.BOX = fake_box(*spot)
            e = PA.survey(A, tcpA, others)
            same = key(e) == key(base)
            ok_empty += same
            msg1 = f"x {spot[0][0]:.0f} y {spot[0][1]:.0f} → {len(e)} 個、{'同じ' if same else '違う'}"
        # 2) いちばん長い物の真上。箱にかからない物は同じ、中心が箱の中の物は消える
        t = max(range(len(base)), key=lambda i: base[i]["size"][1])
        PA.BOX = fake_box(base[t]["center"][:2])
        bm = TB.mask_px(PA.BOX, tcpA, A.shape)
        o = [g for g in PA.survey(A, tcpA, others) if not g["by_box"]]    # 箱の縁にかかった物は why_not で外す
        free = [f for f, om in zip(base, oms) if not (bm & om).any()]
        inside = [f for f in base if bm[int(np.mean(f["contour"][:, 0, 1])), int(np.mean(f["contour"][:, 0, 0]))]]
        keep_ok = all(k in key(o) for k in key(free))
        gone_ok = all(all(np.linalg.norm(g["center"][:2] - f["center"][:2]) > 30 for g in o) for f in inside)
        ok_on += keep_ok and gone_ok
        print(f"{ta}: 物 {len(base)} 個 ｜ 1) {msg1} ｜ 2) 箱の中 {len(inside)} 個 → {'消えた' if gone_ok else '残った'}、"
              f"かからない {len(free)} 個 → {'同じ' if keep_ok else '違う'}（残り {len(o)} 個）")
    print(f"---- 1) 物にかからない所に置いて同じ {ok_empty}/{n_empty}（置けた組 {n_empty}/{n}）、2) 物の上に置いて正しい {ok_on}/{n}")


if __name__ == "__main__":
    main()
