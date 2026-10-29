r"""手首カメラ 1 つで、腕を横にずらして 2 か所から撮り、マットの上の物の高さと上の面の大きさを測る（実機を動かす）。

  1) 見張りの高さ（机から OBS mm、真下向き）で 1 枚撮る（A）
  2) 腕を横に SHIFT mm ずらして（下には下ろさない）もう 1 枚撮る（B）、元の位置へ戻る
  3) 高さ h を 0〜HMAX mm で変えながら、「A の物の画素が、高さ h の面の上にある」としたときに B のどこに写るかを計算し、
     A と B の明るさがいちばんよく合う h を物の上の面の高さとする（近い物ほど、腕を動かすと大きくずれて見える）
  4) 物の輪郭を高さ h の面で mm に直し、上の面の大きさとする（机の面で直した大きさ＝方眼で測った値も並べて出す）
使い方:  .venv_ur\Scripts\python.exe real_ur\height_by_motion.py --ip 192.0.2.10 --go
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

import locate as L
import measure as M
from plan_hover import R_to_rotvec, TABLE_Z, rotvec_to_R

HERE = os.path.dirname(os.path.abspath(__file__))
OBS, SHIFT, HMAX = 150.0, 35.0, 80.0
SPEED, ACCEL = float(os.environ.get("BH_SPEED", 0.25)), float(os.environ.get("BH_ACCEL", 0.8))  # 見張りの位置へ行く速さ。10/16 夕方に上げた（前は 0.10・0.5）                                # 10/16 に上げた（前は 0.05・0.2）。撮る前に揺れが収まるのを待つ（grab_fresh）
GN = np.array(L.CAL["grid_R_base"])[:, 2]                # 机の面の向き（根元の座標）
TOP_DIFF = 18.0                                          # 上の面とみなす明るさの差（0〜255）


def rays(px, tcp):
    """画素（最高解像度）の並び → カメラの位置と、光線の向き（根元の座標）。"""
    n = cv2.undistortPoints(np.asarray(px, np.float64).reshape(-1, 1, 2), L.K, L.DIST).reshape(-1, 2)
    d_cam = np.column_stack([n, np.ones(len(n))])
    RB, tB = rotvec_to_R(tcp[3:]), np.array(tcp[:3]) * 1000.0
    return RB @ L.TC + tB, d_cam @ (RB @ L.RC).T


def on_plane(px, tcp, h):
    """画素の並びを、机から高さ h mm の面の上の点（根元の座標 mm）に直す。"""
    o, d = rays(px, tcp)
    p0 = L.G0 + GN * h
    s = ((p0 - o) @ GN) / (d @ GN)
    return o + s[:, None] * d


def project_fast(P, tcp):
    """measure.project_table と同じ計算（机の上の点 mm → 画素、レンズの曲がりも入れる）を numpy で直接する。
    cv2.projectPoints は 1 回の探す計算の時間の約 8 割を使っていた（10/16）。係数が 8 個より後ろ（薄いプリズム・傾き）は 0 の時だけ使える"""
    Rcb, o = cam_pose(tcp)
    return cam_to_px((np.asarray(P, np.float64) - o) @ Rcb)  # カメラの座標 → 画素


def cam_pose(tcp):
    """手先の姿勢 tcp のときの、カメラの向き Rcb と位置 o（根元の座標 mm）。"""
    from plan_hover import rotvec_to_R as _r
    RB, tB = _r(tcp[3:]), np.array(tcp[:3]) * 1000.0
    return RB @ L.RC, RB @ L.TC + tB


def cam_to_px(X):
    """カメラの座標の点 X → 画素（レンズの曲がりも入れる）。"""
    x, y = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
    k1, k2, p1, p2, k3, k4, k5, k6 = _D8
    r2 = x * x + y * y
    rad = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    K = L.K
    return np.column_stack([K[0, 0] * xd + K[0, 1] * yd + K[0, 2], K[1, 1] * yd + K[1, 2]])


_D = np.asarray(L.DIST, np.float64).ravel()
_D8 = np.concatenate([_D, np.zeros(8)])[:8]
FAST_PROJ = os.environ.get("BH_FAST_PROJ", "1") == "1" and not np.any(_D[8:])


def hand_mask(img):
    """ロボット自身（赤い指と、画面の下の方のハンドの本体）が写っている所。少し広げる。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    red[int(img.shape[0] * 0.72):] = 255
    return cv2.dilate(red, np.ones((21, 21), np.uint8))


STILL = os.environ.get("BH_STILL", "1") == "1"        # 止まったら撮る（関節の速さが STILL_QD より小さくなったらすぐ撮る）。0 で前と同じく決めた時間待つ
STILL_QD = float(os.environ.get("BH_STILL_QD", "0.003"))   # 止まったとみなす関節の速さ（rad/秒）。手首のカメラで約 1mm/秒
STILL_MAX = None                                      # 待つ時間の上限（秒）。None なら locate.SETTLE（前の決まった待ち。これより遅くはならない）
now, tick = time.perf_counter, (lambda: time.sleep(0.008))   # 時計と、1 回待つ動き（シミュレーターはシミュレーションの時計と 1 コマに差し替える）


def grab_fresh(r=None, wait=6.0):
    """腕を止めた後の、新しい画像。カメラの窓に撮る合図を出してすぐ撮る（窓が古い版なら、今までどおり wait 秒待つ。10/16）。
    r（RTDE の受け側）があれば、関節の速さを見て、止まったらすぐ撮る（決めた時間は待たない。10/19）。"""
    if not STILL or r is None or not hasattr(r, "getActualQd"):
        return L.grab_now()
    limit = L.SETTLE if STILL_MAX is None else STILL_MAX
    t0, calm = now(), 0
    while now() - t0 < limit:
        calm = calm + 1 if max(abs(v) for v in r.getActualQd()) < STILL_QD else 0
        if calm >= 3:                                  # 3 回続けて止まっていたら
            break
        tick()
    LAST_WAIT.append(now() - t0)
    return L.grab_now(settle=0.0)


LAST_WAIT = []                                        # 止まるまで待った時間（秒）。確かめ用


def sample(g, pts, border=-1.0):
    """画像 g の、点の並び pts（画素、小数）の明るさ。remap は 1 回に 32767 行までなので、1024 個ずつの行に並べて調べる。"""
    n = len(pts)
    w = 1024
    rows = (n + w - 1) // w
    buf = np.full((rows * w, 2), -10.0, np.float32)
    buf[:n] = pts
    mx, my = buf[:, 0].reshape(rows, w), buf[:, 1].reshape(rows, w)
    out = cv2.remap(g, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return out.reshape(-1)[:n]


# 横へずらす短い動き（35・70mm）の速さ・加速度。短いのでほぼ加速と減速だけで終わる。撮る前に揺れが収まるのを待つ（grab_now の settle）
# 探す時間（シミュレーション、10/16）：8.3 秒 → 戻らない 7.4 → 速く 6.9 → 間引く 6.3 秒。間引いても高さの誤差は同じだった
SHIFT_SPEED = float(os.environ.get("BH_SHIFT_SPEED", 0.20))
SHIFT_ACCEL = float(os.environ.get("BH_SHIFT_ACCEL", 1.0))
BACK = os.environ.get("BH_BACK", "0") == "1"
ORDER_BAC = os.environ.get("BH_ORDER", "BAC") == "BAC"   # 撮る順番（BAC：右・真ん中・左。ABC：前の順番）
MAX_PX = int(os.environ.get("BH_MAX_PX", 2500))   # 1 つの物（区切り）の高さを比べる画素の数の上限。多いと計算が遅い（前は 6000）
ERODE = int(os.environ.get("BH_ERODE", 15))   # 比べる前に、物の輪郭を内側へ削る量（画素）
# この差より小さい合い具合の違いは同じとみなし、低い方を取る。0.003 にするとドライバーの持ち手を 15mm（本当 25mm）と
# 低く出す回があり、円柱と判断されず危ないので 0（今までと同じ、10/16）
TIE = float(os.environ.get("BH_TIE", 0.0))
COARSE, FINE, FINE_SPAN = 2.0, 0.5, 2.5     # 高さは、まず COARSE mm おきに比べ、いちばん合った所の ±FINE_SPAN だけ FINE mm おきに比べ直す


def best_height(imgA, tcpA, imgB, tcpB, mask):
    """A の物の画素（mask、少し内側）について、高さ h ごとに B の明るさと比べ、いちばん合う h（mm）と、h ごとの合い具合を返す。
    0.5mm おきに 0〜80mm の 161 通りを全部比べると、1 回の探す計算が 約 3.4 秒かかった（ほぼこの関数、シミュレーション 10/16）。
    粗く比べてから近くだけ細かく比べる（約 50 通り）。比べなかった高さの合い具合は -1 にしておく。"""
    return best_height_multi(imgA, tcpA, [(imgB, tcpB)], mask)


def best_height_multi(imgA, tcpA, others, mask, coarse_only=False):
    """best_height を、横へずらした何枚かの写真 others = [(画像, 姿勢), …] で同時にする。高さ h ごとの合い具合は、
    指の陰で比べられない写真を除いた平均。模様の無い面では、右の写真は高めに、左は低めにずれることが多く
    （消しゴム 本当 12mm → 右 18mm・左 1mm）、良い方の 1 枚を選ぶより平均の方が本当に近かった
    （10 場面 20 個：本当との差 平均 2.1 → 1.7mm、最大 6.5 → 3.0mm。robot/diag_combine.py、10/16）。写真が 1 枚なら best_height と同じ"""
    hs = np.arange(0.0, HMAX + FINE / 2, FINE)
    scores = np.full(len(hs), -1.0)
    ys, xs = nonzero_inner(mask, ERODE)
    if len(xs) < 50:                                      # 細すぎて、内側に画素が残らない（ドライバーの軸など）：測れない
        return 0.0, hs, scores
    step = max(1, int(math.sqrt(len(xs) / MAX_PX)))       # 画素が多すぎるときは間引く
    xs, ys = xs[::step], ys[::step]
    gA = gray_cached(imgA)
    a = gA[ys, xs]
    a = (a - a.mean()) / (a.std() + 1e-6)
    pxA = np.column_stack([xs, ys]).astype(np.float64)
    # 光の線は高さによらないので 1 回だけ求める。高さ h の面との交点は h に比例して動く：P(h) = P0 + h * dP（10/16 速くした）
    o, d = rays(pxA, tcpA)
    dn = d @ GN
    P0 = o + (((L.G0 - o) @ GN) / dn)[:, None] * d
    dP = d / dn[:, None]

    px_key = (xs.tobytes(), ys.tobytes())

    def scorer(imgB, tcpB):
        memo = memo_for(imgB, px_key)                     # 先に計算しておいた合い具合（腕が動いている間に。pre_scores）
        f = _scorer(imgB, tcpB)

        def cached(hl):
            hl = np.asarray(hl, np.float64)
            ks = [int(round(h / FINE)) for h in hl]
            todo = [i for i, k in enumerate(ks) if k not in memo]
            if todo:
                for i, v in zip(todo, f(hl[todo])):
                    memo[ks[i]] = v
            return [memo[k] for k in ks]
        return cached

    def _scorer(imgB, tcpB):
        gB = gray_cached(imgB)
        handB = hand_mask_cached(imgB)
        if FAST_PROJ:
            # カメラの座標も高さ h に比例して動く：X(h) = X0 + h * dX。点ごとの向きの直しを、高さごとにしなくてよい（10/16）
            Rcb, o = cam_pose(tcpB)
            X0, dX = (P0 - o) @ Rcb, dP @ Rcb

        def score_many(hl):
            """高さの並び hl ごとの合い具合（前の score を、まとめて計算するようにしたもの。答えは同じ）"""
            hl = np.asarray(hl, np.float64)
            if FAST_PROJ:
                pB = cam_to_px((X0[None] + hl[:, None, None] * dX[None]).reshape(-1, 3))
            else:
                pB = M.project_table((P0[None] + hl[:, None, None] * dP[None]).reshape(-1, 3), tcpB)
            pB = np.nan_to_num(pB, nan=-10.0, posinf=-10.0, neginf=-10.0)
            pB = np.clip(pB, -10, 30000).astype(np.float32)
            b = sample(gB, pB).reshape(len(hl), -1)
            u = np.clip(pB[:, 0].astype(int), 0, gB.shape[1] - 1)
            v = np.clip(pB[:, 1].astype(int), 0, gB.shape[0] - 1)
            ok = (b >= 0) & (handB[v, u] == 0).reshape(len(hl), -1)   # B で指の陰になる所は比べない（10/15：カバーが指に隠れて高さを誤った）
            out = []
            for bi, oi in zip(b, ok):
                if oi.mean() < 0.6:
                    out.append(-1.0)
                    continue
                bb = bi[oi]
                bb = (bb - bb.mean()) / (bb.std() + 1e-6)
                out.append(float(np.mean(a[oi] * bb)))    # 正規化した相関（1 に近いほど合う）
            return out
        return score_many

    scorers = [scorer(im, t) for im, t in others]

    def score_many(hl):
        if PARALLEL and len(scorers) > 1:
            # 右と左の計算は互いに関係ないので、同時に走らせる（計算の中身は同じなので答えも同じ。10/16）
            per = np.array(list(_POOL.map(lambda f: f(hl), scorers)))
        else:
            per = np.array([f(hl) for f in scorers])      # 写真ごと × 高さごと
        valid = per != -1.0
        n = valid.sum(0)
        return np.where(n > 0, np.where(valid, per, 0.0).sum(0) / np.maximum(n, 1), -1.0)

    def at_many(hl):
        ks = [int(round(h / FINE)) for h in hl]
        todo = [k for k in ks if scores[k] == -1.0]
        if todo:
            scores[todo] = score_many(hs[todo])

    def lowest_best():
        """いちばん良い合い具合と TIE 以内の差しかない高さのうち、いちばん低いもの。模様の無い面（消しゴムの青いカバーなど）では
        合い具合の山が平らで、11〜16mm のどれでも同じくらい合い、少しの違いで答えが動いた（本当 12mm → 16mm、10/16）。
        低めに取ると指先が物の低い所まで下りるので、つかみ損ねにくい"""
        best = scores.max()
        return float(hs[int(np.flatnonzero(scores >= best - TIE)[0])])

    at_many(np.arange(0.0, HMAX + COARSE / 2, COARSE))
    if coarse_only:
        return None
    h0 = lowest_best()
    at_many(np.arange(max(0.0, h0 - FINE_SPAN), min(HMAX, h0 + FINE_SPAN) + FINE / 2, FINE))
    return lowest_best(), hs, scores


_MEMO = []


def memo_for(img, px_key):
    """写真 img と、比べる画素の並び px_key ごとの、高さの番号 → 合い具合 の入れ物。同じ写真（同じもの）で同じ画素の時だけ同じ入れ物を返す。
    写真を持ったままにするので、古い写真の id が別の写真に使い回されて取り違えることはない"""
    for im, k, d in _MEMO:
        if im is img and k == px_key:
            return d
    d = {}
    _MEMO.append((img, px_key, d))
    return d


def clear_memo():
    _MEMO.clear()



def nonzero_inner(mask, erode=0):
    """mask を erode 画素ぶん内側へ削った所の画素 (ys, xs)。np.nonzero(cv2.erode(mask)) と同じ答え（並び順も同じ）を、
    写真全体（1920×1080）ではなく物のまわりだけで調べて速くする（1 回の探す計算で約 0.3 秒使っていた、10/16）"""
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pad = erode + 2
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(mask.shape[1], x + w + pad), min(mask.shape[0], y + h + pad)
    sub = mask[y0:y1, x0:x1]
    if erode > 1:
        sub = cv2.erode(sub, np.ones((erode, erode), np.uint8))
    ys, xs = np.nonzero(sub)
    return ys + y0, xs + x0


SHOT_DIR = os.path.join(HERE, "cam_shots")   # 撮った 3 枚を上書きで残す所。シミュレーターは robot/logs/sim_shots に変える（10/16：実機の写真を上書きして消した）
# 腕が動いている間に、先にできる計算をする。10/16 にシミュレーターの 31 個中 1 個で合い具合が 0.0002 違い、使っていなかった。
# 原因は先の計算ではなく、measure.refine の kmeans の乱数（呼ぶたびに進む）だった。種を決めた後は、実機の写真 120 個で全部同じ（10/19）
OVERLAP = os.environ.get("BH_OVERLAP", "1") == "1"
PARALLEL = os.environ.get("BH_PARALLEL", "1") == "1"   # 右と左の写真の計算を同時に走らせる（0 で前と同じく順に）
_POOL = None
if PARALLEL:
    from concurrent.futures import ThreadPoolExecutor
    _POOL = ThreadPoolExecutor(max_workers=2)


_GRAY = []


def gray_cached(img):
    """同じ画像の白黒（float32）を何度も作らない。"""
    for im, g in _GRAY:
        if im is img:
            return g
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    _GRAY[:] = (_GRAY + [(img, g)])[-4:]
    return g


_HAND = []


def hand_mask_cached(img):
    """同じ画像の hand_mask を何度も求めない（物ごと・区切りごとに呼ばれて、1 回の探す計算で約 0.2 秒使っていた）"""
    for im, m in _HAND:
        if im is img:
            return m
    m = hand_mask(img)
    _HAND[:] = (_HAND + [(img, m)])[-4:]
    return m


def capture_two(ip, go=True, c=None, r=None, at=None, back=None, during=None):
    """見張りの高さで撮り（A）、横へ ±SHIFT mm ずらして撮り（B・C）、A へ戻る。(画像 A, 姿勢 A, [(画像 B, 姿勢 B), (画像 C, 姿勢 C)]) を返す。
    c, r：つながっている RTDE を使う時（やり直しの時。無ければここでつなぎ、終わったら切る）。at：撮る位置 x y（m）。無ければ今の真上
    back：撮り終わったら A へ戻るか（無ければ BACK）。次につかみに行くなら戻らなくてよい（約 0.8 秒縮む、10/16）
    during(imgA, tcpA, imgB, tcpB)：A と B を撮った後、C へ動いて撮る間に、裏で走らせる計算（pick_any.pre_scores）。C を撮った後に終わるのを待つ"""
    back = BACK if back is None else back
    own = c is None
    if own:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
        r = RTDEReceiveInterface(ip)
        c = RTDEControlInterface(ip)
    try:
        p0 = r.getActualTCPPose()
        if at is not None:
            p0 = [at[0], at[1]] + list(p0[2:])
        R0 = rotvec_to_R(p0[3:])
        x = R0[:, 0].copy(); x[2] = 0; x /= np.linalg.norm(x)
        Rdown = np.column_stack([x, np.cross([0, 0, -1.0], x), [0, 0, -1.0]])
        poseA = [p0[0], p0[1], (TABLE_Z + OBS) / 1000] + list(R_to_rotvec(Rdown))
        side = Rdown[:, 1] * SHIFT / 1000                    # 道具の y の向きへずらす（指を結ぶ向き）
        poseB = [poseA[0] + side[0], poseA[1] + side[1], poseA[2]] + poseA[3:]
        poseC = [poseA[0] - side[0], poseA[1] - side[1], poseA[2]] + poseA[3:]
        for p in (poseA, poseB, poseC):
            if not c.isPoseWithinSafetyLimits(p):
                raise SystemExit("撮る位置が安全設定の範囲の外なので止めます")
        print(f"1) 机から {OBS:.0f}mm で、" + ("右・真ん中・左" if ORDER_BAC else "真ん中・右・左") + f"（横に {SHIFT:.0f}mm ずつ）の 3 枚を撮る（下には下ろさない）")
        if not go:
            print("--go が無いので、ここまで（何も送っていません）")
            return None
        if ORDER_BAC:
            # 右（B）→ 真ん中（A）→ 左（C）の順に撮る。横の動きが 35 + 70mm から 35 + 35mm に減る（10/16）
            c.moveL(poseB, SPEED, ACCEL)
            imgB, tcpB = grab_fresh(r), r.getActualTCPPose()
            c.moveL(poseA, SHIFT_SPEED, SHIFT_ACCEL)
            imgA, tcpA = grab_fresh(r), r.getActualTCPPose()
        else:
            c.moveL(poseA, SPEED, ACCEL)
            imgA, tcpA = grab_fresh(r), r.getActualTCPPose()
            c.moveL(poseB, SHIFT_SPEED, SHIFT_ACCEL)
            imgB, tcpB = grab_fresh(r), r.getActualTCPPose()
        th = None
        if during is not None and OVERLAP:
            # 腕が C へ動いて揺れが収まるのを待つ間に、A と B だけでできる計算を先にする（答えは同じ。10/16）
            import threading
            th = threading.Thread(target=during, args=(imgA, tcpA, imgB, tcpB), daemon=True)
            th.start()
        c.moveL(poseC, SHIFT_SPEED, SHIFT_ACCEL)
        imgC, tcpC = grab_fresh(r), r.getActualTCPPose()
        if th is not None:
            th.join()
        if back:
            c.moveL(poseA, SHIFT_SPEED, SHIFT_ACCEL)
    finally:
        if own:
            c.stopScript()
            c.disconnect()
            r.disconnect()
    shot = SHOT_DIR
    cv2.imwrite(os.path.join(shot, "height_A.jpg"), imgA)
    cv2.imwrite(os.path.join(shot, "height_B.jpg"), imgB)
    cv2.imwrite(os.path.join(shot, "height_C.jpg"), imgC)
    json.dump({"A": list(tcpA), "B": list(tcpB), "C": list(tcpC)}, open(os.path.join(shot, "height_tcp.json"), "w"))
    for k, (im, t) in {"A": (imgA, tcpA), "B": (imgB, tcpB), "C": (imgC, tcpC)}.items():
        L.archive(im, f"height_{k}", {"tcp": list(t)})   # 上書きせずにも残す（後でシミュレーターとの合わせ込みに使う）
    return imgA, tcpA, [(imgB, tcpB), (imgC, tcpC)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    got = capture_two(a.ip, a.go)
    if got:
        report(got[0], got[1], *got[2][0])


def report(imgA, tcpA, imgB, tcpB):
    objs, _ = M.find_objects(imgA)
    if not objs:
        raise SystemExit("物が見つからない")
    for i, cnt in enumerate(objs):
        mask = np.zeros(imgA.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        h, hs, sc = best_height(imgA, tcpA, imgB, tcpB, mask)
        tf = top_face(imgA, tcpA, imgB, tcpB, mask, h) if h >= 3 else None   # 高さのある物は、側面を外して上の面だけにする
        pts = (cnt if tf is None else tf).reshape(-1, 2).astype(np.float64)
        top = on_plane(pts, tcpA, h)[:, :2].astype(np.float32)
        size_top = sorted(cv2.minAreaRect(top)[1])
        flat = M.measure(imgA, cnt, tcpA)
        grid = flat.get("size_mm", flat["size_cal_mm"])
        print(f"{i}: 高さ {h:.1f}mm（合い具合 {sc.max():.2f}、h=0 では {sc[0]:.2f}）、"
              f"上の面の大きさ {size_top[0]:.1f} × {size_top[1]:.1f} mm（机の面で測ると {grid[0]:.1f} × {grid[1]:.1f} mm）")


if __name__ == "__main__":
    main()


def top_face(imgA, tcpA, imgB, tcpB, mask, h):
    """高さ h で A と B の明るさが合う画素だけを、上の面とみなす（側面は 2 か所で見え方が変わるので合わない）。上の面の輪郭（A の画素）を返す。"""
    ys, xs = np.nonzero(mask)
    gA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY).astype(np.float32)
    P = on_plane(np.column_stack([xs, ys]).astype(np.float64), tcpA, h)
    pB = np.nan_to_num(M.project_table(P, tcpB), nan=-10.0, posinf=-10.0, neginf=-10.0)
    pB = np.clip(pB, -10, 30000).astype(np.float32)            # 画面のはるか外の点で remap が止まらないように
    b = sample(gB, pB, border=0.0)
    diff = np.zeros(mask.shape, np.float32)
    diff[ys, xs] = np.abs(gA[ys, xs] - b)
    diff = cv2.GaussianBlur(diff, (0, 0), 3)
    good = ((diff < TOP_DIFF) & (mask > 0)).astype(np.uint8) * 255
    good = cv2.morphologyEx(good, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(good, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None
