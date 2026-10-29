r"""マットの上の物を色に関係なく見つけ、まわりの方眼（1 マス 10mm）をものさしにして大きさを測る（ロボットは動かさない）。

  1) マット（暗い紫）を見分け、マットに囲まれた「マットの色でない塊」を物とする（白・青だけでなく何色でも）
  2) 物のまわりの方眼の交点を拾って番号 (i, j) を付け、レンズの曲がりを直してから、画像 → マットの上の mm の対応（射影変換）を
     その場で求める。物の輪郭をこれで mm に直し、外接する長方形の辺を大きさとする（校正の結果に頼らない）
  3) 物の上の面は机より高いので、少し大きく写る。高さ h が分かっていれば、カメラまでの距離 D から (D - h) / D 倍して直す
使い方:  .venv_ur\Scripts\python.exe real_ur\measure.py cam_shots\size_test.jpg cam_shots\size_test_tcp.json
"""
import json
import os
import sys

import cv2
import numpy as np

from locate import G0, RC, TC, pixel_to_table

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = json.load(open(os.path.join(HERE, "calib2", "calib_result.json")))
K, DIST = np.array(CAL["K"]), np.array(CAL["dist"])
GR = np.array(CAL["grid_R_base"])          # 方眼の向き（列 0：i の向き、列 1：j の向き、根元の座標）
SEARCH = 9                                 # 予想した交点のまわりで、本物の交点を探す範囲（画素）
MAT_V_MAX = 205
MAT_S_MAX = int(os.environ.get("MAT_S_MAX", 45))   # マットの色の濃さの上限。10/16 夕方の実機：マットの右側が暗く色付き（明るさ 90〜95・濃さ 45〜57）、ペンを見落とした → MAT_V_MIN=90・MAT_S_MAX=60 で見つかった
MAT_V_MIN = int(os.environ.get("MAT_V_MIN", 110))   # マットの明るさの下限（マットは 140〜175、10/15）
MAT_BLUR = float(os.environ.get("MAT_BLUR", 2))   # マットを見分ける前のぼかし（画素）。5 だと物の幅が 5〜7mm 太った（シミュレーションで確認、10/15）
MAT_EDGE_PX = int(os.environ.get("MAT_EDGE_PX", 80))   # マットの縁から内側へこの画素数（見張りの高さで約 25mm）より外の物は外す
MIN_AREA = 3000            # 見張りの高さ（机から 150mm）で 1 cm 角 ≈ 1100 画素。それより大きい物を探す
RING = 260                 # 物のまわりの方眼を拾う範囲（画素、最高解像度で）


def mat_area(img):
    """マット（暗い紫）の部分。方眼の線もマットに含める（ぼかしてから判定）。"""
    hsv = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (0, 0), MAT_BLUR)
    if MAT_NORM:
        # 照明のむらをならす：広い範囲（NORM_WIN 画素）の明るさの中央値で割り、明るさ NORM_TARGET にそろえる（10/19 に試した）
        v = hsv[..., 2]
        small = cv2.resize(v, (v.shape[1] // 8, v.shape[0] // 8), interpolation=cv2.INTER_AREA)
        bg = cv2.resize(cv2.medianBlur(small, NORM_WIN // 8 | 1), (v.shape[1], v.shape[0]), interpolation=cv2.INTER_LINEAR)
        hsv = hsv.copy()
        hsv[..., 2] = np.clip(v.astype(np.float32) * NORM_TARGET / np.maximum(bg, 20), 0, 255).astype(np.uint8)
    m = cv2.inRange(hsv, (0, 0, MAT_V_MIN), (180, MAT_S_MAX, MAT_V_MAX))   # 下限：暗い灰色の床（明るさ 40〜83）を外す
    if MAT_DARK:
        # 照明が当たりにくいマットの右側は、明るさ 67〜80 で下限より暗い（10/16 夕方の実機）。明るさの下限を下げると、銀の棒の
        # 暗い面や影までマットになり、棒を見失った。暗い所は「マットの色合い（赤紫〜茶色）で、色がある程度付いている」所だけ足す。
        # 銀の棒の暗い面・灰色の床は色がほとんど無い（色の濃さが低い）ので入らない。暗い所の方眼の線（色合い 12〜20・
        # 色の濃さ 64〜74・明るさ 94〜128）も、色の濃さの上限 60 を超えてマットから外れ、マットが網目に切れていたので、
        # この色合いの所は明るさの下限より明るくても、色の濃さ DARK_S_MAX まで入れる（10/19）
        h, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        dark = ((v >= DARK_V_MIN) & (v <= MAT_V_MAX) & (sat >= DARK_S_MIN) & (sat <= DARK_S_MAX)
                & ((h <= DARK_H_LO) | (h >= DARK_H_HI)))
        m[dark] = 255
    m[int(img.shape[0] * 0.72):] = 0                              # 下の方はハンドの本体
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


MAT_NORM = os.environ.get("MAT_NORM", "0") == "1"   # 照明のむらをならしてからマットを見分ける
NORM_WIN = int(os.environ.get("NORM_WIN", 301))
NORM_TARGET = float(os.environ.get("NORM_TARGET", 125))
# 暗い所も、マットの色合いならマットとする。明るさ 90 以上だけ足す形（暗い所の方眼の線）を 10/19 から使う：10/16 夕方の写真 49 組で、
# ペンが 22 → 24 組、消しゴム 49・銀の棒 40・二重 0・どれでもない候補 3 は変わらない（robot/sweep_mat.py）。0 で前と同じ
MAT_DARK = os.environ.get("MAT_DARK", "1") == "1"
DARK_V_MIN = int(os.environ.get("DARK_V_MIN", 90))   # 55 まで下げると、銀の棒の暗い面までマットになった
DARK_S_MIN, DARK_S_MAX = int(os.environ.get("DARK_S_MIN", 25)), int(os.environ.get("DARK_S_MAX", 80))
DARK_H_LO, DARK_H_HI = int(os.environ.get("DARK_H_LO", 25)), int(os.environ.get("DARK_H_HI", 135))   # 色合い（0〜180）：赤〜茶 と 赤紫


# マットの外形を凸の形で囲む（暗い角で外形が欠けない）。ペンは 22 → 27 組に増えたが、暗い所が物として見つかり（どれでもない候補 3 → 70）、
# 消しゴムが二重に見つかる組（0 → 4）も出たので使わない（10/19）
MAT_HULL = os.environ.get("MAT_HULL", "0") == "1"
OBJ_OPEN = int(os.environ.get("OBJ_OPEN", 7))    # 物の塊の細い出っ張りを削る大きさ（画素）


def find_objects(img, exclude=None):
    """マットに囲まれた、マットの色でない塊（物）の輪郭の一覧（大きい順）。
    exclude：物を探さない所の画素（マットの上に置いた箱。teach_box.mask_px）。そこはマットとみなす（10/19）"""
    mat = mat_area(img)
    if exclude is not None:
        mat = mat | exclude
    if FINGER:
        mat = mat | finger_mask(img)
    if BORDER:
        mat = border_as_mat(mat)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mat)
    if n <= 1:
        return [], mat
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    mat_big = (lab == big).astype(np.uint8) * 255
    # マットの外形（穴を埋めた形）の中で、マットでない所 = マットの上の物
    cnts, _ = cv2.findContours(mat_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if MAT_HULL:
        # マットは長方形なので、外形は凸。照明が当たりにくい角が暗くてマットに入らないと、外形がそこで欠け、欠けた所に触れた物
        # （ペン）が「マットの外」とつながって見つからなかった（10/16 夕方の写真 49 組でペン 22 組だけ）。外形を凸の形で囲む（10/19）
        cnts = [cv2.convexHull(np.vstack(cnts))]
    filled = np.zeros_like(mat_big)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    holes = filled & ~mat_big
    # 立てた筒（カバーなど）は中がマットと同じ暗い色に見えて「輪」になる。輪の内側も塗りつぶして 1 つの物にする（10/15）
    hc, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(holes, hc, -1, 255, -1)
    # 方眼の線は明るくマットの色に入らないので、物に当たる所で輪郭が線の太さの分だけ外へ出っ張る。輪郭を囲む凸の形で測ると、
    # その分だけ幅が大きく出た（消しゴム 本当 20mm → 22〜24mm、シミュレーション・実機とも。10/16）。細い出っ張りを削る
    k = OBJ_OPEN
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    objs, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if JOIN:
        objs = join_pieces(objs)
    objs = [c for c in objs if cv2.contourArea(c) >= MIN_AREA]
    # 物はマットの中にしか置かない（10/15）。マットの外形の縁から MAT_EDGE_PX より内側に中心がある物だけにする
    # （マットの縁に印刷された白い帯・ラベル、マットの外の棚や枠を物と間違えない）
    inner = cv2.distanceTransform((filled > 0).astype(np.uint8), cv2.DIST_L2, 5) > MAT_EDGE_PX
    keep = []
    for c in objs:
        (cx, cy), _, _ = cv2.minAreaRect(c)
        if inner[int(min(max(cy, 0), inner.shape[0] - 1)), int(min(max(cx, 0), inner.shape[1] - 1))]:
            keep.append(c)
    objs = keep
    if REFINE:
        objs = [refine(img, c) for c in objs]
    return sorted(objs, key=cv2.contourArea, reverse=True), mat_big


# 写真の中で赤い指に重なった物（ペンの端が左の指に隠れた）は、マットの外（指）とつながって物にならず、指から離れた切れ端だけが残った
# （10/19 夜の実機、4 回の通しでペンが 1 回も見つからなかった）。指の所はマットとみなす（物の輪郭は指の縁で切れる）。0 で前と同じ
FINGER = os.environ.get("OBJ_FINGER", "1") == "1"
HAND_ROW = 0.72                                          # これより下は、ハンドの本体（mat_area と同じ）


def finger_mask(img):
    """赤い指が写っている所。赤い塊のうち、ハンドの本体（写真の下の方）とつながっている物だけ（マットの上の赤い物は入れない）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    row = int(img.shape[0] * HAND_ROW)
    n, lab = cv2.connectedComponents(red)
    keep = np.unique(lab[row - 3:row + 3][lab[row - 3:row + 3] > 0])
    m = np.isin(lab, keep).astype(np.uint8) * 255
    m[row:] = 0
    return cv2.dilate(m, np.ones((7, 7), np.uint8))


# 写真の端に触れた物（消しゴム・銀の棒を写真の上の端まで置いた）は、写真の外とつながって物にならなかった（10/19 夜の実機、6・7 回目）。
# マットが写真の上の端まで続いている所（上の端から BORDER_PX 画素の帯にマットがある所の、左端から右端まで）は、上の端の 2 行もマットとみなす。
# 左右の端は、見張りの位置からは枠・床・箱が写り、マットとみなすと床の布が物になった（10/19）ので、上の端だけにする。
# 端からはみ出した物は一部しか写らないので、pick_any が「写真の端に触れている」印（by_edge）を付ける。0 で前と同じ
BORDER = os.environ.get("OBJ_BORDER", "1") == "1"
BORDER_PX = 40
BORDER_GAP = 200


def border_as_mat(mat):
    # 端から端まで全部埋めると、10/16 の写真で上の端に写った箱が物になり、消しゴムが二重に見つかった（4 組）。
    # 上の帯にマットがある列どうしの間の、BORDER_GAP 画素以下の狭いすき間（物 1 つ分）だけを埋める
    m = mat.copy()
    idx = np.flatnonzero(m[:BORDER_PX, :].any(axis=0))
    m[:2, idx] = 255
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= BORDER_GAP:
            m[:2, a:b] = 255
    return m


# 1 本の細長い物が、途中の明るく色の薄い所（ペンの銀色の輪）でマットと見分けられ、端と端で並んだ 2 つの塊に切れた
# （10/19 夜の実機：ペン 145mm が 87mm と 42mm に切れ、d1-3B が撮るたびに ペン／どれでもない と答えを変え、つかめなかった）。
# 向きがそろい（JOIN_ANGLE° 以内）、同じ線の上にあり（横のずれが太い方の幅の半分以内）、端と端のすき間が JOIN_GAP_PX 画素以内で、
# 太さが近い（JOIN_WRATIO 倍以内）細長い 2 つの塊は、1 つの物としてつなぐ（2 つを囲む凸の形）。0 で前と同じ
JOIN = os.environ.get("OBJ_JOIN", "1") == "1"
JOIN_GAP_PX = float(os.environ.get("JOIN_GAP_PX", 40))   # 見張りの高さで約 12mm
JOIN_ANGLE, JOIN_WRATIO, JOIN_ASPECT = 12.0, 2.0, 1.8   # 太さの比 1.6 では、クリップ付きの太い方（21mm）と軸（13.7mm）で 1.62 となり、つながらなかった（10/19 夜、箱の通し）


def _axis(c):
    """輪郭の (中心, 長さ, 幅, 長い向きの単位ベクトル)（画素）。"""
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    if w < h:
        w, h, ang = h, w, ang + 90.0
    a = np.radians(ang)
    return np.array([cx, cy]), w, h, np.array([np.cos(a), np.sin(a)])


def end_to_end(ca, cb):
    pa, la, wa, ua = _axis(ca)
    pb, lb, wb, ub = _axis(cb)
    if min(wa, wb) < 1 or la < JOIN_ASPECT * wa or lb < JOIN_ASPECT * wb or max(wa, wb) > JOIN_WRATIO * min(wa, wb):
        return False
    if np.degrees(np.arccos(min(1.0, abs(float(ua @ ub))))) > JOIN_ANGLE:
        return False
    u = ua if la >= lb else ub
    d = pb - pa
    side = abs(float(d @ np.array([-u[1], u[0]])))
    gap = abs(float(d @ u)) - (la + lb) / 2
    return side <= max(wa, wb) / 2 and -0.2 * min(la, lb) <= gap <= JOIN_GAP_PX


def bridge(ca, cb):
    """2 つの塊を、向かい合う端どうしを細い帯（細い方の幅）でつないで 1 つの輪郭にする。2 つを囲む凸の形にすると、
    横に少しずれて並んだ時に幅が太く出た（ペン 本当 11mm → 24mm で「指に入らない」と外した。10/19 夜）"""
    pa, la, wa, ua = _axis(ca)
    pb, lb, wb, ub = _axis(cb)
    u = ua if la >= lb else ub
    ea = pa + np.sign(float((pb - pa) @ u)) * u * la / 2           # a の、b に向いた端
    eb = pb - np.sign(float((pb - pa) @ u)) * u * lb / 2
    pts = np.vstack([ca, cb]).reshape(-1, 2)
    x0, y0 = pts.min(0) - 5
    x1, y1 = pts.max(0) + 5
    m = np.zeros((int(y1 - y0) + 1, int(x1 - x0) + 1), np.uint8)
    off = np.array([x0, y0])
    cv2.drawContours(m, [ca - off, cb - off], -1, 255, -1)
    cv2.line(m, tuple(int(v) for v in ea - off), tuple(int(v) for v in eb - off), 255, max(3, int(min(wa, wb) * 0.8)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cs, key=cv2.contourArea) + off.astype(np.int32)


def join_pieces(objs):
    objs = list(objs)
    changed = True
    while changed:
        changed = False
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                if cv2.contourArea(objs[i]) > 200 and cv2.contourArea(objs[j]) > 200 and end_to_end(objs[i], objs[j]):
                    objs[i] = bridge(objs[i], objs[j])
                    del objs[j]
                    changed = True
                    break
            if changed:
                break
    return objs


REFINE = os.environ.get("OBJ_REFINE", "1") == "1"
REFINE_OPEN = int(os.environ.get("OBJ_REFINE_OPEN", 13))    # 決め直した後に、細い出っ張りを削る大きさ（画素）。9 で 21.4mm、13 で 20.8mm（消しゴム 本当 20mm）
REFINE_THR = float(os.environ.get("OBJ_REFINE_THR", 18.0))   # 背景の色（マットの地・方眼の線）からの色の差（Lab）がこれより大きい所を物とする


REFINE_SEED = int(os.environ.get("MEAS_SEED", "12345"))   # refine の kmeans の乱数の種（-1 で決めない＝前と同じく毎回変わる）


def refine(img, cnt):
    """物の輪郭を決め直す。マットの色の範囲で見分けた輪郭は、物のまわりで 1〜2 画素ふくらみ、方眼の線に当たる所で外へ出っ張った
    （消しゴムの幅 本当 20mm → 22〜24mm。シミュレーターの本当の物の画素と比べて分かった、10/16）。
    まわりの輪の画素から背景の色を 2 つ（マットの地と方眼の線）取り、どちらからも色が離れている画素だけを物とする。"""
    x, y, w, h = cv2.boundingRect(cnt)
    pad = 40
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    roi = img[y0:y1, x0:x1]
    m = np.zeros(roi.shape[:2], np.uint8)
    cv2.drawContours(m, [cnt - [x0, y0]], -1, 255, -1)
    ring = cv2.dilate(m, np.ones((31, 31), np.uint8)) & ~cv2.dilate(m, np.ones((11, 11), np.uint8))
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = lab[ring > 0]
    if len(bg) < 200:
        return cnt
    # kmeans は乱数で始まりの点を選び、OpenCV の乱数は呼ぶたびに進む。そのため同じ写真でも計算するたびに輪郭が少し変わり、
    # 高さの合い具合が最大 0.0012・中心が 0.26mm 揺れた（10/19。先の計算 BH_OVERLAP で答えが変わったのも、乱数が余分に進んだため）。
    # 呼ぶ直前に種を決めて、毎回同じにする（OpenCV の乱数は糸ごとなので、裏の糸で呼んでも同じ）
    if REFINE_SEED >= 0:
        cv2.setRNGSeed(REFINE_SEED)
    _, _, centers = cv2.kmeans(bg, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3, cv2.KMEANS_PP_CENTERS)
    d = np.min(np.stack([np.linalg.norm(lab - c_, axis=2) for c_ in centers]), axis=0)
    obj = ((d > REFINE_THR) & (cv2.dilate(m, np.ones((9, 9), np.uint8)) > 0)).astype(np.uint8) * 255
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # 方眼の線と物の境目の画素は、線と物の色が混ざって背景の 2 色のどちらにも似ず、小さな出っ張りとして残る → 削る
    k = REFINE_OPEN
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cs, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return cnt
    best = max(cs, key=cv2.contourArea)
    if cv2.contourArea(best) < 0.5 * cv2.contourArea(cnt):      # 決め直しで大きく減ったら（物の色が背景に近い）、元のままにする
        return cnt
    return best + [x0, y0]


def undist(p):
    return cv2.undistortPoints(np.asarray(p, np.float64).reshape(-1, 1, 2), K, DIST, P=K).reshape(-1, 2)


def project_table(pts_mm, tcp):
    """机の上の点（根元の座標 mm）を、そのときの腕の姿勢での画像の画素へ写す（レンズの曲がりも入れる）。"""
    from plan_hover import rotvec_to_R
    RB, tB = rotvec_to_R(tcp[3:]), np.array(tcp[:3]) * 1000.0
    Rcb, o = RB @ RC, RB @ TC + tB                      # カメラの向きと位置（根元の座標）
    rv, _ = cv2.Rodrigues(Rcb.T)
    px, _ = cv2.projectPoints(np.asarray(pts_mm, np.float64).reshape(-1, 1, 3), rv, -Rcb.T @ o, K, DIST)
    return px.reshape(-1, 2)


def grid_homography(img, center_px, tcp, obj_mask):
    """物のまわりの方眼の交点を使い、（曲がりを直した）画像の点 → マットの上の mm への射影変換を求める。
    交点の位置は、校正から予想した所の近く（±SEARCH 画素）で、線の明るさがいちばん集まる所に合わせ直す（実際の方眼で決める）。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lines = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    resp = cv2.GaussianBlur(lines, (0, 0), 3)               # 2 本の線が交わる所が最も明るくなる
    # 物の中心のあたりの、方眼の番号 (i, j) を校正から求め、まわり ±N マスの交点を予想する
    c_mm = pixel_to_table(center_px, tcp)
    ij0 = np.round(GR[:, :2].T @ (c_mm - G0) / 10.0).astype(int)
    N = 6
    ij = np.array([(ij0[0] + a, ij0[1] + b) for a in range(-N, N + 1) for b in range(-N, N + 1)])
    pred = project_table(G0 + (ij * 10.0) @ GR[:, :2].T, tcp)
    keep_mask = cv2.dilate(obj_mask, np.ones((25, 25), np.uint8))
    src, dst = [], []
    for (i, j), (u, v) in zip(ij, pred):
        u, v = int(round(u)), int(round(v))
        if not (SEARCH <= u < img.shape[1] - SEARCH and SEARCH <= v < img.shape[0] * 0.72):
            continue
        if keep_mask[v, u] or np.hypot(u - center_px[0], v - center_px[1]) > RING:
            continue
        win = resp[v - SEARCH:v + SEARCH + 1, u - SEARCH:u + SEARCH + 1]
        dv, du = np.unravel_index(int(np.argmax(win)), win.shape)
        if win[dv, du] < 8 or not (2 <= du <= 2 * SEARCH - 2 and 2 <= dv <= 2 * SEARCH - 2):
            continue                                          # 線が弱い・窓の端（本当の交点は窓の外かもしれない）は使わない
        src.append((u - SEARCH + du, v - SEARCH + dv))
        dst.append((i * 10.0, j * 10.0))
    if len(src) < 12:
        return None, len(src), None
    src = undist(src).astype(np.float32)
    dst = np.array(dst, np.float32)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 1.5)
    if H is None:
        return None, len(src), None
    m = inl.ravel() > 0
    proj = cv2.perspectiveTransform(src[m].reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.sqrt(np.mean(np.sum((proj - dst[m]) ** 2, axis=1))))
    return H, int(m.sum()), err


def measure(img, contour, tcp):
    """物の輪郭の大きさ（mm、短い辺・長い辺）。方眼で直した値と、校正だけで直した値の両方を返す（比べる用）。"""
    (cx, cy), _, _ = cv2.minAreaRect(contour)
    obj = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(obj, [contour], -1, 255, -1)
    H, n, err = grid_homography(img, np.array([cx, cy]), tcp, obj)
    pts = contour.reshape(-1, 2).astype(np.float64)
    cal = np.array([pixel_to_table(p, tcp)[:2] for p in pts], np.float32)
    out = {"size_cal_mm": sorted(cv2.minAreaRect(cal)[1]), "grid_points": n, "fit_err_mm": err}
    if H is not None:
        mm = cv2.perspectiveTransform(undist(pts).astype(np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
        out["size_mm"] = sorted(cv2.minAreaRect(mm.astype(np.float32))[1])
    return out


def main():
    path = sys.argv[1]
    tcp = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    img = cv2.imread(os.path.join(HERE, path) if not os.path.isabs(path) else path)
    if img.shape[1] != 1920:
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    objs, mat = find_objects(img)
    dbg = img.copy()
    for i, c in enumerate(objs):
        r = measure(img, c, tcp)
        box = cv2.boxPoints(cv2.minAreaRect(c)).astype(int)
        cv2.drawContours(dbg, [box], -1, (0, 255, 0), 3)
        txt = f"{r['size_mm'][0]:.0f}x{r['size_mm'][1]:.0f}mm" if "size_mm" in r else "?"
        cv2.putText(dbg, f"{i}: {txt}", tuple(box.min(0)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cal = f"校正だけ {r['size_cal_mm'][0]:.1f} × {r['size_cal_mm'][1]:.1f} mm"
        if "size_mm" not in r:
            print(f"{i}: 方眼が足りず測れない（交点 {r['grid_points']} 個）、{cal}")
        else:
            print(f"{i}: 方眼で {r['size_mm'][0]:.1f} × {r['size_mm'][1]:.1f} mm（交点 {r['grid_points']} 個、当てはまりの誤差 {r['fit_err_mm']:.2f}mm）、{cal}")
    cv2.imwrite(os.path.join(HERE, "cam_shots", "measure_dbg.jpg"), cv2.resize(dbg, (960, 540)))


if __name__ == "__main__":
    main()
