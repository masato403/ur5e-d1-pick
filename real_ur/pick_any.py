r"""マットの上の物を（色・形に関係なく）見つけ、2 か所から撮って高さと上の面を測り、つかむ（実機）。

  1) 見張りの高さで撮る（A）→ 横へ 35mm ずらして撮る（B）→ 戻る（height_by_motion.py と同じ）
  2) マットの色でない塊を物とし、物ごとに高さ・上の面の中心・長い辺の向き・大きさ（短い辺＝指で挟む幅）を出す
  3) つかめる物だけを候補にする：高さ MIN_H mm 以上（紙などの平らな物は外す）、挟む幅 MAX_W mm 以下（指の開き 24.8mm より余裕を持つ）、
     今の位置から MAX_FAR mm 以内。--pick で番号を選ぶ（無ければ候補の中でいちばん近い物）
  4) 指先を下ろす高さ = 高さの半分（GRIP_MIN〜GRIP_MAX mm に収める）。あとは消しゴム用と同じ（pick_by_camera.grasp）
  5) 持てなかったら、見張りの位置へ上がって撮り直し、測り直して、指先を GRIP_STEP mm ずつ下げて最大 RETRY 回やり直す（10/16）
使い方:  .venv_ur\Scripts\python.exe real_ur\pick_any.py --ip 192.0.2.10 --look --go     （測って候補を出すだけ）
        .venv_ur\Scripts\python.exe real_ur\pick_any.py --ip 192.0.2.10 --pick 0 --place 129.9 -433.9 --go
"""
import argparse
import math
import os

import cv2
import numpy as np

import height_by_motion as HM
import measure as M
import pick_by_camera as PB
import teach_box as TB
from pick_by_camera import HOLD_MM, grasp

HERE = os.path.dirname(os.path.abspath(__file__))
# MAX_W：指で挟める幅の上限。指は 24.8mm まで開く。10/16 に 21 → 23mm（消しゴム 本当 20mm を 21〜22mm と測って外していた。
# 23mm の物でも左右に約 0.9mm ずつ余裕があり、当たっても保護停止とやり直しがある）
MIN_H, MAX_H, MAX_W, MAX_FAR = 3.0, float(os.environ.get("PICK_MAX_H", 50)), 23.0, 200.0
MAT_RING = 0.6             # 物のまわり（輪）のうち、マットである割合の下限（マットの端・枠・床を外す）
GRIP_MIN, GRIP_MAX = 4.0, 20.0
DEBUG = os.environ.get("PA_DEBUG", "0") == "1"   # 1 なら、物ごとの区切りの高さと、挟む所の決め方の理由を出す
GRIP_RULE = os.environ.get("PA_GRIP_RULE", "shape")   # 下ろす高さの決め方（shape：形で変える。half：前と同じく高さの半分）
ROUND_GRIP, THIN_BELOW = 0.3, 1.5
# 丸い物の指先の下限（mm）。4mm では、太さ 9mm の銀の棒の中心（4.5mm）とほぼ同じ高さで、指の面が一番太い所の下に回り込めず、
# 持ち上げる途中で抜けた（10/16 実機）。机の高さは触って測った値（傾き 0.29°、マットの端から端で約 0.5mm）なので 2mm にする
ROUND_MIN = float(os.environ.get("PA_ROUND_MIN", 2.0))                   # 丸い物は高さの 0.3 倍、軸は軸の下の面より 1.5mm 下まで下ろす
# 高さを、輪郭の内側からの距離がいちばん奥の BAND 倍より大きい所（真ん中の帯）だけで測る。0 なら全体。
# 帯は模様が少なく（消しゴムの青いカバーなど）高さが決まらないことがあり、物を 1 つずつ置いた比べでは悪くなった
# （消しゴムの誤差 +0.1 → −4.8mm、ドライバー −3.9 → −5.8mm、10/16）ので、使わない
BAND = float(os.environ.get("PICK_BAND", 0.0))


def band(mask, frac=BAND):
    """細長い物の、長い向きに沿った真ん中の帯。寝かせた円柱（ドライバーの持ち手）は、横の丸みが上から写って
    いちばん高い所より低く出た（本当 25mm → 13〜21mm）。帯だけで測ると 20〜24mm（シミュレーション、10/16）。"""
    if frac <= 0:
        return mask
    d = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return ((d > d.max() * frac) * 255).astype(np.uint8)


# 高さがこの倍の幅以上なら寝かせた円柱とみなす（消しゴムは 12 ÷ 21 ＝ 0.57〜0.6、ペン・持ち手は 約 1.1。
# 持ち手の高さを低く測った時（15.5 ÷ 21 ＝ 0.74）も拾えるよう 0.8 → 0.68、10/16）
ROUND_RATIO = 0.68


def round_height(pts, tcp, h0, axis):
    """寝かせた円柱の高さ（＝太さ）。輪郭を高さ h の半分の面に写した時の幅から h を求める（二分法）。
    物全体の明るさの比べでは、横の丸みが混ざって低く出た（持ち手 本当 25mm → 15.5〜24mm、10/16）。
    斜めから見ると、輪郭のいちばん外側（視線が円に接する所）は高さの半分より少し上にあり、半分の面に写した幅は
    太さ ÷ cos φ になる（φ：円柱の軸と直角な面の中での、視線の傾き）。その分を直す（直さないと +2mm ほど大きく出た）。"""
    o, d = HM.rays(np.array([pts.mean(0)]), tcp)
    d = d[0] / np.linalg.norm(d[0])
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    dp = d - (d @ a) * a                                  # 視線の、円柱の軸と直角な面への成分
    cos_phi = abs(dp @ HM.GN) / max(np.linalg.norm(dp), 1e-9)

    def gap(h):
        hull = cv2.convexHull(HM.on_plane(pts, tcp, h / 2)[:, :2].astype(np.float32))
        return sorted(cv2.minAreaRect(hull)[1])[0] * cos_phi - h
    lo, hi = h0, MAX_H
    if gap(lo) <= 0:
        return h0
    if gap(hi) > 0:
        return hi
    for _ in range(20):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if gap(m) > 0 else (lo, m)
    return (lo + hi) / 2
# 物ごとの挟む力（%）。力の % は電流ではなく、指を押し込むずれの上限（3〜11）。銀の棒は 60% では運ぶ途中で落ち、100% で運べた（10/16）。
# ペンも 60% では持ち上げる途中で抜けた（10/16）。実機の通し（pick_pass）とシミュレーターの試験で同じ値を使う（10/19 に pick_pass から移した）
FORCE = {"消しゴム": 60, "ペン": 100, "銀の棒": 100}
RETRY, GRIP_STEP, REFIND_MM = 2, 2.0, 40.0   # 持てなかった時のやり直しの回数、やり直すたびに指先を下げる量（mm）、同じ物とみなす中心の距離（mm）
PARTS, PART_MATCH, PART_TIE, PART_STEP = 4, 0.6, 2.0, 5.0
STEP_RULE = os.environ.get("PA_STEP_RULE", "width")   # 太さが変わる物の見分け方（width：区切りの幅の差。height：前と同じく高さの差）
COMBINE = os.environ.get("PA_COMBINE", "mean")   # 高さを決める写真の使い方（mean：右と左の平均。best：前と同じく良い方の 1 枚）
SHINY = os.environ.get("PA_SHINY", "1") == "1"   # 高さが測れない細長い物を丸い棒とみなす（0 で前と同じ）
SHINY_ASPECT = 4.0
ELONG = os.environ.get("PA_ELONG", "1") == "1"   # 細長い物は丸い物として指先を決める（0 で前と同じ）
ELONG_ASPECT = 5.0
UNIFORM_H = os.environ.get("PA_UNIFORM_H", "median")   # 太さが変わらない物の高さ（median：中央値。max：前と同じく区切りの最高）
# 区切りの幅の差がこれ（mm）より大きければ、太さが変わる物。5 では、ペンのクリップの所（幅 19〜22mm、軸 14〜16mm）で差が 4〜7mm と
# 境目をまたぎ、撮るたびに 真ん中／クリップ側 と挟む所が 15〜20mm 入れ替わった（10/19 夜の実機）。ドライバー（持ち手 25・軸 5mm）は差 20mm
PART_WSTEP = float(os.environ.get("PA_PART_WSTEP", 8.0))   # 高さを区切りごとに測る数、その区切りを使う合い具合の下限、同じ高さとみなす差（mm）


def retry_grip(f, attempt, empty):
    """指先を下ろす高さ（机から mm）。やり直すたびに GRIP_STEP mm 下げる。空振りした後は、区切りの最高ではなく、
    物全体で測った高さと比べて低い方を使う（模様の無い区切りで高さを高く見すぎると、物より上で閉じて空振りする。10/16）。"""
    h = min(f["h"], f.get("h_whole", f["h"])) if empty else f["h"]
    base = h / 2
    if GRIP_RULE == "shape":
        # 高さの半分は、丸い物では指先がちょうど一番太い所に来て、指の先端だけで挟んで滑って抜けた。
        # ドライバーの軸（太さ 5mm、中心が高さの半分）では、指の一番下の端しか軸にかからず空振りした（シミュレーター 10/16。
        # 前は腕が 5〜10mm 垂れていて、たまたま深く下りていた）。一番太い所より下まで下ろし、指の面で挟む
        if f.get("grip_on") == "thin":
            base = h / 2 - f["size"][0] / 2 - THIN_BELOW         # 軸の下の面より少し下
        elif f.get("round"):
            base = h * ROUND_GRIP
    lo = ROUND_MIN if (GRIP_RULE == "shape" and f.get("round") and f.get("grip_on") != "thin") else GRIP_MIN
    return max(lo, min(GRIP_MAX, base) - GRIP_STEP * attempt)


def footprint(pts, tcp, h):
    """上から見た物の形（mm、根元の座標の x, y）。写った輪郭を「上の面の高さ h」と「机の高さ 0」の 2 通りで机に写すと、
    前者はカメラ側の側面の分だけカメラの方へ、後者は上の面の分だけ反対側へはみ出す。はみ出す向きが逆なので、
    2 つの重なり（凸包どうしの共通部分）が物の底の形になる（側面を明るさで見分けるより、照明や色に左右されない。10/15）。"""
    up = cv2.convexHull(HM.on_plane(pts, tcp, h)[:, :2].astype(np.float32))
    if h < MIN_H:
        return up.reshape(-1, 2)
    lo = cv2.convexHull(HM.on_plane(pts, tcp, 0.0)[:, :2].astype(np.float32))
    area, inter = cv2.intersectConvexConvex(up, lo)
    return inter.reshape(-1, 2) if inter is not None and area > 0 else up.reshape(-1, 2)


# 区切りの高さを使う合い具合の下限。正しい区切りは 0.98〜1.00、模様の無いカバーの所などの間違った区切りは 0.77〜0.92 だった
# （消しゴム 本当 12mm → 3・15mm、ペン 本当 11mm → 23.5mm。シミュレーション 10/16）。足りなければ物全体の高さだけを使う
PART_GOOD = float(os.environ.get("PART_GOOD", 0.95))
PART_PEAK = float(os.environ.get("PART_PEAK", 0.05))  # ドライバーの持ち手は 0.08〜0.14、ペンの一色の所は 0.00（10/16）   # 区切りの合い具合の山の鋭さの下限


def peak(sc):
    """合い具合の山の鋭さ：いちばん良い値と、比べた高さ全体の真ん中の値の差。模様の無い一色の所（ペンの軸）は、
    どの高さでもよく合って山が平ら（差がほぼ 0）になり、高さが決まらない（ペン 本当 11mm の区切りが 49mm になった。10/16）。"""
    v = sc[sc > -0.99]
    return float(v.max() - np.median(v)) if len(v) else 0.0


def part_width(mask, tcp, z):
    """区切り（画素の塊 mask）の幅（mm）：画素を机から z mm の面に写した形の短い辺。"""
    yy, xx = HM.nonzero_inner(mask)
    P = HM.on_plane(np.column_stack([xx, yy]).astype(np.float64), tcp, z)[:, :2].astype(np.float32)
    return sorted(cv2.minAreaRect(P)[1])[0]


def part_heights(imgA, tcpA, imgB, tcpB, mask, cnt, n=PARTS, others=None, coarse_only=False):
    """物を長い向きに n 個に区切り、区切りごとの高さ（合い具合が十分な所だけ）。
    1 つの物として高さを 1 つにまとめると、太い所と細い所が混ざって低めに出た（ドライバー：本当 25mm → 19〜23mm。10/15）。"""
    ys, xs = HM.nonzero_inner(mask)                             # np.nonzero(mask) と同じ（物のまわりだけ調べて速くした）
    (cx, cy), (w, l), ang = cv2.minAreaRect(cnt)
    a = math.radians(ang if w >= l else ang + 90)
    t = (xs - cx) * math.cos(a) + (ys - cy) * math.sin(a)       # 画像の上で、長い向きに沿った位置
    edges = np.linspace(t.min(), t.max(), n + 1)
    out = []
    for i in range(n):
        sel = (t >= edges[i]) & (t <= edges[i + 1])
        if sel.sum() < 30:
            continue
        m = np.zeros_like(mask)
        m[ys[sel], xs[sel]] = 255
        xy = (float(xs[sel].mean()), float(ys[sel].mean()))
        if sel.sum() < 400:
            out.append((None, xy, m, "thin"))                   # 細すぎて測れない区切り（ドライバーの軸など）：画素の塊も返す
            continue
        r_ = HM.best_height_multi(imgA, tcpA, others if others else [(imgB, tcpB)], m, coarse_only)   # others：何枚かの平均（survey の COMBINE）
        if coarse_only:
            continue
        h_, _, sc_ = r_
        if sc_.max() < 0:
            out.append((None, xy, m, "thin"))                   # 内側を削ると画素が残らない＝細い
            continue
        if sc_.max() >= PART_GOOD and peak(sc_) >= PART_PEAK:
            out.append((h_, xy, m, "ok"))
        elif sc_.max() >= PART_MATCH:
            # 合い具合が足りない、または山が平ら（模様が無い）：高さは使わない。細すぎて測れない区切りとは分ける
            out.append((None, xy, m, "flat"))
        else:
            out.append((None, xy, m, "bad"))
    return out


BOX = None   # マットの上に置いた箱（teach_box.load の形）。入れると、箱の所では物を探さない（pick_pass が入れる。10/19）


def box_mask(img, tcp):
    """箱の所の画素（箱が無い・BOX_MASK=0 なら None）。"""
    return TB.mask_px(BOX, tcp, img.shape) if BOX is not None and TB.MASK else None


def pre_scores(imgA, tcpA, imgB, tcpB):
    """survey と同じ物・同じ区切りについて、B の写真との粗い比べだけを先にしておく（腕が C へ動いている間に裏で。
    HM.capture_two の during）。結果は HM の入れ物に入り、survey が同じ計算をする時にそのまま使う。答えは変わらない（10/16）"""
    if COMBINE != "mean":
        return
    try:
        objs, _ = M.find_objects(imgA, box_mask(imgA, tcpA))
        for cnt in objs:
            mask = np.zeros(imgA.shape[:2], np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mid = band(mask)
            HM.best_height_multi(imgA, tcpA, [(imgB, tcpB)], mid, coarse_only=True)
            part_heights(imgA, tcpA, imgB, tcpB, mid, cnt, others=[(imgB, tcpB)], coarse_only=True)
    except Exception as e:                                      # 先の計算が失敗しても、survey が全部計算するので困らない
        print(f"   （先の計算を飛ばした：{e}）")


def survey(imgA, tcpA, others):
    """物ごとの高さ・上の面の中心（根元の座標 mm）・長い辺の向き・大きさ。
    others：横へずらして撮った (画像, 姿勢) の並び。物ごとに、指の陰にならず、いちばんよく合う 1 枚で高さを測る。"""
    ex = box_mask(imgA, tcpA)
    objs, _ = M.find_objects(imgA, ex)
    fingers = M.finger_mask(imgA) if M.FINGER else None
    out = []
    for cnt in objs:
        mask = np.zeros(imgA.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mid = band(mask)
        best = None
        for imgB, tcpB in others:
            if COMBINE == "mean":
                break
            h_, _, sc_ = HM.best_height(imgA, tcpA, imgB, tcpB, mid)
            if best is None or sc_.max() > best[1].max():
                best = (h_, sc_, imgB, tcpB)
        if COMBINE == "mean":
            # 右と左の合い具合の平均で高さを決める（HM.best_height_multi の説明）。区切りの高さも同じ
            h_, _, sc_ = HM.best_height_multi(imgA, tcpA, others, mid)
            best = (h_, sc_) + tuple(others[0])
        h, sc, imgB, tcpB = best
        h_whole = h
        if sc.max() < 0:
            continue                                            # どの写真でも指の陰で比べられない
        parts_all = part_heights(imgA, tcpA, imgB, tcpB, mid, cnt, others=others if COMBINE == "mean" else None)
        parts = [p for p in parts_all if p[3] == "ok"]
        grip_px = None
        if parts:
            h_max = max(h, max(p[0] for p in parts))            # 太さが変わる物（ドライバー）は、いちばん高い所をその物の高さにする
            # 挟む所：いちばん高い区切り（高さの差 PART_TIE mm 以内なら、物の真ん中に近い方）の真ん中。
            # 物全体の真ん中は、ドライバーでは持ち手と軸のつなぎ目に来て、挟むと抜けた（10/15）
            # ただし、本当に太さが変わる物だけ（区切りの高さの差が PART_STEP mm より大きい、または細すぎて測れない区切りがある）。
            # 太さが一定の物（ペン）でこれをすると、高さのぶれで端を挟み、運ぶ途中で落ちた（10/15）
            hs_ = [p[0] for p in parts]
            thin_any = any(p[3] == "thin" for p in parts_all)
            # 高さの差で決めると、模様の無い一色のペンは区切りの高さが ±3mm ばらつき（10・10・15.5mm など）、差が 5mm を超えて
            # 端をつかみに行った（真ん中から 50mm、空振りもした。10/16）。幅は輪郭から測るのでばらつかない（ドライバー 持ち手 25mm・軸 5mm、ペン 11mm）
            ws_ = [part_width(p_[2], tcpA, h_max / 2) for p_ in parts_all]
            step = (max(ws_) - min(ws_) > PART_WSTEP) if STEP_RULE == "width" else (max(hs_) - min(hs_) > PART_STEP)
            if DEBUG:
                print("   [区切り] " + "、".join(f"{k_}:{'-' if h_ is None else f'{h_:.1f}'}／幅{w_:.1f}" for (h_, _, _, k_), w_ in zip(parts_all, ws_))
                      + f" → 高さの差 {max(hs_) - min(hs_):.1f}、幅の差 {max(ws_) - min(ws_):.1f}、細い所 {thin_any}、数 {len(parts_all)}")
            stepped = step or thin_any or len(parts_all) < PARTS
            # 太さが変わらない物で区切りの最高を使うと、外れた 1 区切りで高さが跳ねた（ペン 本当 11mm、区切り 8・11・42.5・16mm → 42.5mm で
            # 「指に入らない」と外した。10/16）。物全体の高さと区切りの高さの中央値にする
            h = h_max if (stepped or UNIFORM_H != "median") else float(np.median([h_whole] + hs_))
            if stepped:
                (mx, my), _, _ = cv2.minAreaRect(cnt)
                tall = [p for p in parts if p[0] >= max(hs_) - PART_TIE]   # 物全体の高さ h ではなく区切りの最高と比べる（h の方が高いと空になって止まった。10/16）
                grip_px = min(tall, key=lambda p: math.hypot(p[1][0] - mx, p[1][1] - my))[1]
        pts = cnt.reshape(-1, 2).astype(np.float64)
        top = HM.on_plane(pts, tcpA, h)
        foot = footprint(pts, tcpA, h)
        (cx, cy), (w, l), ang = cv2.minAreaRect(foot)
        a = math.radians(ang)
        e1 = np.array([math.cos(a), math.sin(a), 0.0])            # minAreaRect の 1 辺目（長さ w）の向き
        longv = e1 if w >= l else np.array([-e1[1], e1[0], 0.0])
        inside = cv2.dilate(mask, np.ones((11, 11), np.uint8))
        ring = cv2.dilate(mask, np.ones((61, 61), np.uint8)) & ~inside
        mat = M.mat_area(imgA)
        if ex is not None:
            mat = mat | ex                                      # 箱のすぐ横の物を「まわりがマットでない」で外さない
        if M.FINGER:
            mat = mat | fingers                                 # 指に触れた物も同じ（find_objects と同じく、指の所はマットとみなす）
        on_mat = float((mat[ring > 0] > 0).mean()) if (ring > 0).any() else 0.0
        # 箱の縁にかかった物：箱の縁で切れた切れ端（長さ 54mm の物が 20mm と出た。10/19 夜、写真に仮の箱を置いて確かめた）か、
        # 箱に寄りかかった物。つかむと指が箱の壁に当たるおそれもあるので外す（why_not）
        by_box = ex is not None and bool((cv2.dilate(mask, np.ones((9, 9), np.uint8)) & ex).any())
        center = np.array([cx, cy, top[:, 2].mean()])
        if grip_px is not None:
            center = HM.on_plane(np.array([grip_px]), tcpA, h)[0]
        size = sorted([w, l])
        # 寝かせた円柱（ペン・ドライバーの持ち手）は、輪郭のいちばん外側が上の面ではなく横（高さの半分あたり）にあるので、
        # 上の面と机に写した重なりでは両側が内側にずれ、細く出た（ペン 本当 11mm → 9.5mm、持ち手 25mm → 21〜22mm。10/16）。
        # 寝かせた円柱は 太さ ＝ 高さ。高さが幅の ROUND_RATIO 倍以上なら円柱とみなし、幅は 今の幅 と 高さ の大きい方にする
        # 光る物（銀の棒）は、横へ動くと映り込みも動くので、見え方のずれで高さが測れず 1〜2mm と出て「平ら」で外した（10/16 実機）。
        # 高さが測れないのに細長い（長さが幅の SHINY_ASPECT 倍以上、幅が指に入る）物は、寝かせた丸い棒とみなし、太さ ＝ 見えている幅とする。
        # 外れても（本当に平らな定規など）、指先は下限の高さで物の上で閉じて空振りするだけ
        shiny = SHINY and h < MIN_H and size[1] >= SHINY_ASPECT * size[0] and size[0] <= MAX_W
        if shiny:
            print(f"   （高さが測れない細長い物（高さ {h:.1f}mm、{size[0]:.1f} × {size[1]:.1f} mm）は、太さ {size[0]:.1f}mm の丸い棒とみなす）")
            h = size[0]
        rnd = h >= ROUND_RATIO * size[0]
        if rnd:
            h = max(h, round_height(pts, tcpA, h, longv))
            size[0] = max(size[0], h)
        # 細長い物（ペン・棒）は寝かせた丸い物として指先を決める。一色のペンは高さを低く測り（本当 11mm → 6.5mm）丸い物と見なされず、
        # 指先が下限 4mm（ペンの中心 5.5mm の 1.5mm 下）で止まって、力 100% でも持ち上げる途中で抜けた（10/16 実機）
        if ELONG and size[1] >= ELONG_ASPECT * size[0] and size[0] <= MAX_W:
            rnd = True
        grip_on = "body"
        def part_w(p):
            """区切りの幅（mm）：区切りの画素を高さの半分の面に写した形の短い辺。"""
            yy, xx = np.nonzero(p[2])
            P = HM.on_plane(np.column_stack([xx, yy]).astype(np.float64), tcpA, h / 2)[:, :2].astype(np.float32)
            return sorted(cv2.minAreaRect(P)[1])[0]
        thin = [p for p in parts_all if p[3] != "ok" and part_w(p) <= MAX_W] if size[0] > MAX_W else []
        if size[0] > MAX_W and thin:
            # 太い所（ドライバーの持ち手 25mm）が指（最大 24.8mm）に入らない時は、細い所（軸）をつかむ。持ち手に近い方を選ぶ
            # （重心が近く落としにくい）。軸は持ち手と同じ中心線の上にあるので、指先は高さの半分まで下ろす（10/16）
            ref = grip_px if grip_px is not None else (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
            ws = [part_w(p) for p in thin]
            # いちばん細い区切りを選ぶ（持ち手の端の区切り 20mm を選んで失敗した、10/16）。同じくらい細ければ、太い所に近い方
            near = [p for p, w_ in zip(thin, ws) if w_ <= min(ws) + 3.0]
            tp = min(near, key=lambda p: math.hypot(p[1][0] - ref[0], p[1][1] - ref[1]))
            tw = part_w(tp)
            center = HM.on_plane(np.array([tp[1]]), tcpA, h / 2)[0]
            size = [tw, size[1]]
            grip_on = "thin"
        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        by_edge = x_ <= 2 or y_ <= 2 or x_ + w_ >= imgA.shape[1] - 2   # 写真の端に触れている（一部しか写っていないかもしれない）
        # 赤い指に触れている（一部が指に隠れて、短く測られたかもしれない。ペンが 80mm と出て「銀の棒」になった。10/20）
        by_finger = fingers is not None and cv2.countNonZero(cv2.dilate(mask, np.ones((9, 9), np.uint8)) & fingers) > 0
        out.append({"on_mat": on_mat, "by_box": by_box, "by_edge": by_edge, "by_finger": by_finger, "h": h, "h_whole": h_whole, "match": float(sc.max()), "center": center,
                    "longv": longv, "size": size, "round": rnd, "grip_on": grip_on, "shiny": shiny, "contour": cnt})
    return drop_twins(out) if TWIN else out


TWIN = os.environ.get("PA_TWIN", "1") == "1"   # 平行にくっついた 2 つの塊を、同じ物の二重とみなして 1 つにする（0 で前と同じ）
TWIN_GAP_PX, TWIN_ANGLE, TWIN_OVERLAP = 6.0, 10.0, 0.6


def _rect_px(c):
    """写真の上の、細長い向きをそろえた (中心, 長さ, 幅, 長い向きの単位ベクトル)（画素）。"""
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    if w < h:
        w, h, ang = h, w, ang + 90.0
    a = math.radians(ang)
    return np.array([cx, cy]), w, h, np.array([math.cos(a), math.sin(a)])


def is_twin(ca, cb):
    """2 つの輪郭が、平行（向きの差 TWIN_ANGLE° 以内）で、すき間 TWIN_GAP_PX 画素以内で横に並び、
    長い向きに短い方の TWIN_OVERLAP 以上重なる細長い塊か。"""
    pa, la, wa, ua = _rect_px(ca)
    pb, lb, wb, ub = _rect_px(cb)
    if la < 2.5 * wa or lb < 2.5 * wb:
        return False
    if math.degrees(math.acos(min(1.0, abs(float(ua @ ub))))) > TWIN_ANGLE:
        return False
    d = pb - pa
    along, across = abs(float(d @ ua)), abs(float(d @ np.array([-ua[1], ua[0]])))
    return (la + lb) / 2 - along >= TWIN_OVERLAP * min(la, lb) and across - (wa + wb) / 2 <= TWIN_GAP_PX


def drop_twins(found):
    """光る丸い棒は、白く光る筋と、暗い面＋影の筋に分かれ、平行にくっついた 2 つの塊として見つかった
    （10/16 の実機の写真 49 組を 10/19 に測り直した：棒が二重になった組・片方を消しゴムと見間違えた組が多数）。
    2 つを囲む形にまとめると影まで入り、太さ 15mm（本当は約 11mm）・高さ 5mm と測り間違えた。そこで、まとめずに、
    高さの合い具合（match）が良い方を残し、もう片方は捨てる。"""
    keep = list(found)
    changed = True
    while changed:
        changed = False
        for i in range(len(keep)):
            for j in range(i + 1, len(keep)):
                if is_twin(keep[i]["contour"], keep[j]["contour"]):
                    worse = i if keep[i]["match"] < keep[j]["match"] else j
                    if DEBUG:
                        a, b = keep[i], keep[j]
                        print(f"  [二重] 高さ {a['h']:.1f}（合い {a['match']:.2f}）と 高さ {b['h']:.1f}（合い {b['match']:.2f}）→ 合い具合の良い方を残す")
                    del keep[worse]
                    changed = True
                    break
            if changed:
                break
    return keep


def why_not(f, far):
    """つかむ候補から外す理由の並び（空なら つかめる）。far：見張りの位置からの距離（mm）"""
    why = []
    if f["h"] < MIN_H:
        why.append("平ら")
    if f["h"] > MAX_H:
        why.append(f"高さ {f['h']:.0f}mm は測り間違いとみなす")
    if f["on_mat"] < MAT_RING:
        why.append(f"まわりがマットでない（{f['on_mat']:.0%}）")
    if f["size"][0] > MAX_W:
        why.append(f"幅 {f['size'][0]:.0f}mm は指に入らない")
    if far > MAX_FAR:
        why.append("遠い")
    if f.get("by_box"):
        why.append("箱の縁にかかっている")
    return why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--pick", type=int, help="つかむ物の番号（--look で出る）。無ければ候補の中でいちばん近い物")
    ap.add_argument("--force", type=int, default=60)
    ap.add_argument("--place", type=float, nargs=2, help="置く場所 x y（mm）")
    ap.add_argument("--place-h", type=float, help="箱に入れる時：箱の高さ ＋ すき間（mm）。指先は これ ＋ 指先を下ろした高さ で離す（落として入れる）")
    ap.add_argument("--near", type=float, nargs=2, help="この位置 x y（mm）にいちばん近い候補をつかむ（40mm 以内。続けて動かす時に、番号の代わりに使う）")
    ap.add_argument("--place-back", action="store_true", help="つかんだ物を、測った中心の位置に戻して離す（箱が無い時の試し用）")
    ap.add_argument("--look", action="store_true", help="測って候補を出すだけ（つかまない）")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    import web_page
    web_page.tee_stdout()                                         # print の文字を、cam_view が出す Chrome のタブにも出す（10/19）
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    if not a.go:
        print("--go が無いので、ここまで（何も送っていません）")
        return
    from hand import HandClient
    h = HandClient()                                         # 撮る前に全開にする（閉じた指が画面の真ん中をふさぐ。10/15）
    print("0) ハンドを全開に:", round(h.open(), 1), "mm")
    h.close_port()
    HM.clear_memo()
    imgA, tcpA, others = HM.capture_two(a.ip, during=pre_scores)
    found = survey(imgA, tcpA, others)
    dbg = imgA.copy()
    cands = []
    for i, f in enumerate(found):
        far = math.hypot(f["center"][0] - tcpA[0] * 1000, f["center"][1] - tcpA[1] * 1000)
        why = why_not(f, far)
        ok = not why
        if ok:
            cands.append((far, i))
        box = cv2.boxPoints(cv2.minAreaRect(f["contour"])).astype(int)
        cv2.drawContours(dbg, [box], -1, (0, 255, 0) if ok else (0, 0, 255), 3)
        cv2.putText(dbg, f"{i}: h{f['h']:.0f} {f['size'][0]:.0f}x{f['size'][1]:.0f}", tuple(box.min(0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0) if ok else (0, 0, 255), 3)
        print(f"{i}: 高さ {f['h']:.1f}mm、上の面 {f['size'][0]:.1f} × {f['size'][1]:.1f} mm、中心 x {f['center'][0]:.1f}  y {f['center'][1]:.1f} → "
              + ("つかめる" if ok else "外す（" + "、".join(why) + "）"))
    cv2.imwrite(os.path.join(HERE, "cam_shots", "pick_any_found.jpg"), cv2.resize(dbg, (960, 540)))
    web_page.push_image("見つけた物（真ん中の写真）", dbg)
    HM.L.archive(cv2.resize(dbg, (960, 540)), "pick_any_found",
                 [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in f.items() if k != "contour"} for f in found])
    if a.near is not None:
        near = [(math.hypot(found[i]["center"][0] - a.near[0], found[i]["center"][1] - a.near[1]), i) for _, i in cands]
        near = [d for d in near if d[0] < 40.0]
        if not near:
            raise SystemExit(f"x {a.near[0]:.0f}  y {a.near[1]:.0f} の近く（40mm 以内）に、つかめる候補が無いので止めます")
        k = min(near)[1]
    elif a.pick is not None:
        if a.pick not in [i for _, i in cands]:
            raise SystemExit(f"{a.pick} 番はつかめる候補ではないので止めます")
        k = a.pick
    elif len(cands) == 1:
        k = cands[0][1]
    elif cands:
        raise SystemExit("つかめる候補が 2 つ以上あるので、--pick で番号を選んでください（画像 cam_shots/pick_any_found.jpg）")
    else:
        raise SystemExit("つかめる物が無い")
    f = found[k]
    if a.place_back:
        a.place = [float(f["center"][0]), float(f["center"][1])]
        print(f"   （つかんだ物は、元の位置 x {a.place[0]:.1f}  y {a.place[1]:.1f} に戻して離す）")
    r = RTDEReceiveInterface(a.ip)
    c = RTDEControlInterface(a.ip)
    try:
        empty = False
        for attempt in range(1 + RETRY):
            grip = retry_grip(f, attempt, empty)
            print(f"→ {k} 番をつかむ{'' if attempt == 0 else f'（やり直し {attempt} 回目）'}：指先を机から {grip:.1f}mm まで下ろす"
                  f"（高さ {f['h']:.1f}mm の半分{'' if attempt == 0 else f' − {GRIP_STEP * attempt:.0f}mm'}）")
            place_h = None if a.place_h is None else a.place_h + grip
            if grasp(c, r, r.getActualTCPPose(), f["center"], f["longv"], grip, a.force, a.place, a.look, place_h=place_h,
                     miss="retry" if attempt < RETRY else "stop") is not False:
                break
            # 持てなかった：見張りの位置へ上がり、撮り直して同じ物を測り直す（向き・高さ・中心を計算し直す。10/16）
            empty = PB.LAST["closed"] is not None and PB.LAST["closed"] <= HOLD_MM
            print("   見張りの位置へ上がって撮り直す" + ("（空振り：次は低い方の高さを使う）" if empty else ""))
            HM.clear_memo()
            imgA, tcpA2, others = HM.capture_two(a.ip, c=c, r=r, at=tcpA[:2], during=pre_scores)
            again = [g for g in survey(imgA, tcpA2, others)
                     if np.linalg.norm(g["center"][:2] - f["center"][:2]) < REFIND_MM and MIN_H <= g["h"] <= MAX_H
                     and g["size"][0] <= MAX_W and g["on_mat"] >= MAT_RING]
            if not again:
                raise SystemExit("撮り直したら、つかめる物として見つからないので止めます")
            f = min(again, key=lambda g: np.linalg.norm(g["center"][:2] - f["center"][:2]))
            print(f"   測り直した：高さ {f['h']:.1f}mm、上の面 {f['size'][0]:.1f} × {f['size'][1]:.1f} mm、中心 x {f['center'][0]:.1f}  y {f['center'][1]:.1f}")
    finally:
        c.stopScript()
        c.disconnect()
        r.disconnect()


if __name__ == "__main__":
    main()
