"""暗い所の直しの候補（10/19 夜。デモが終わるまで実機用のプログラムは変えないので、ここで pick_any.survey を差し替えて試す）。
物ごとに境目を決め直す：今の決まり（明るさ DARK_V_MIN 以上の暗い所だけマット）で測り、つかめる物はそのまま使う。
つかめない物（まわりがマットでない・幅が指に入らない など）と、今の決まりで見つからなかった所だけ、暗い所（明るさ TWO_V_MIN 以上、
マットの色合い）もマットとみなす決まりで測り直した物を使う。
暗い所を全部マットにすると銀の棒の暗い面までマットになり棒が減った（sweep_mat.py の DARK_V_MIN=75）ので、全体は変えない。
使い方:  import dark_two; dark_two.install()   （試験の台本は SURVEY_TWO=1 で入れる）
"""
import os

import cv2
import numpy as np

import measure as M
import pick_any as PA

TWO_V_MIN = int(os.environ.get("TWO_V_MIN", 35))   # 測り直す時の、暗い所の明るさの下限
ORIG = PA.survey
TWIN_ALL = os.environ.get("TWO_TWIN", "1") == "1"   # 今の物と測り直した物を合わせて二重を 1 つにする（0 で前と同じ）


def ok(f):
    return not [w for w in PA.why_not(f, 0.0)]


def overlap(a, b):
    ra, rb = cv2.boundingRect(a["contour"]), cv2.boundingRect(b["contour"])
    x0, y0 = min(ra[0], rb[0]), min(ra[1], rb[1])
    x1, y1 = max(ra[0] + ra[2], rb[0] + rb[2]), max(ra[1] + ra[3], rb[1] + rb[3])
    ma = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    mb = ma.copy()
    cv2.drawContours(ma, [a["contour"] - [x0, y0]], -1, 1, -1)
    cv2.drawContours(mb, [b["contour"] - [x0, y0]], -1, 1, -1)
    return bool((ma & mb).any())


def survey(imgA, tcpA, others):
    base = ORIG(imgA, tcpA, others)
    keep = [f for f in base if ok(f)]
    if len(keep) == len(base) and os.environ.get("TWO_ALWAYS", "1") != "1":
        return base
    v0 = M.DARK_V_MIN
    M.DARK_V_MIN = TWO_V_MIN
    try:
        again = ORIG(imgA, tcpA, others)
    finally:
        M.DARK_V_MIN = v0
    add = [g for g in again if ok(g) and not any(overlap(g, f) for f in keep)]
    if TWIN_ALL:
        # 銀の棒が二重になった（10/16 の写真で 2 つ以上 13 組）：今の決まりでは二重を 1 つにして暗い筋が残り、測り直すと暗い面がマットになって
        # 白く光る筋だけが残る。2 つは平行に並ぶだけで重ならないので、足されていた。今の物と足す物を合わせて、二重を 1 つにする（10/20）
        both = PA.drop_twins(keep + add)
        keep, add = [f for f in keep if any(f is x for x in both)], [g for g in add if any(g is x for x in both)]
    # 今の決まりで外れた物は、測り直してつかめる物が重なれば、そちらに替える（重ならなければ、外れた物のまま残す）
    bad = [f for f in base if not ok(f) and not any(overlap(f, g) for g in add)]
    for g in add:
        g["dark_two"] = True
    return keep + add + bad


def install():
    PA.survey = survey
