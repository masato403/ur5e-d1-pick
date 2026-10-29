r"""三脚のカメラ（マットの手前）で見たマットの上の物の位置（根元の座標 mm）。通し（pick_pass.py）で手首の結果と突き合わせる（10/20）。
写真は cam_view.py --cam2 が 0.5 秒ごとに残す cam_shots/tripod_latest.jpg（960×540）。校正は tripod_calib.json。
位置のずれ：10/20 の実機で、手首の位置より y が平均 +11.5mm（x -1.4mm）ずれていた（logs_tripod_calib/vs_wrist_20261020_1623.json）ので引く。"""
import json
import os
import time

import cv2
import numpy as np

from plan_hover import TABLE_Z, rotvec_to_R

HERE = os.path.dirname(os.path.abspath(__file__))
LATEST = os.path.join(HERE, "cam_shots", "tripod_latest.jpg")
OBJ_H = 6.0                                                    # 光線を交える高さ（机から mm）
FIX = np.array([float(os.environ.get("TRI_DX", 1.4)), float(os.environ.get("TRI_DY", -11.5))])
MATCH_MM = float(os.environ.get("TRI_MATCH", 30))              # 手首の物と同じ物とみなす近さ
_cal = None


def cal():
    global _cal
    if _cal is None:
        c = json.load(open(os.path.join(HERE, "tripod_calib.json"), encoding="utf-8"))
        _cal = {"K": np.array(c["K"]), "dist": np.array(c["dist"]), "rvec": np.array(c["rvec"]), "tvec": np.array(c["tvec"])}
    return _cal


def grab(after, timeout=3.0):
    """after（時刻）より後に書かれた三脚の写真。来なければ None。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if os.path.getmtime(LATEST) > after:
                img = cv2.imread(LATEST)
                if img is not None:
                    return img
        except OSError:
            pass
        time.sleep(0.05)
    return None


def to_table(uv, h=OBJ_H):
    c = cal()
    R = cv2.Rodrigues(c["rvec"])[0]
    o = -R.T @ c["tvec"].ravel()
    n = cv2.undistortPoints(np.array([[uv]], np.float64), c["K"], c["dist"]).reshape(2)
    d = R.T @ np.array([n[0], n[1], 1.0])
    s = (TABLE_Z + h - o[2]) / d[2]
    return (o + s * d)[:2] + FIX


def to_px(P, lens=True):
    """根元の座標 mm の点 → 三脚の写真の画素。lens=False でレンズの曲がりを入れない（写真の外の遠い点で曲がりの式が暴れるので、腕の所を描く時に使う）。"""
    c = cal()
    return cv2.projectPoints(np.asarray(P, np.float32).reshape(-1, 3), c["rvec"], c["tvec"], c["K"], c["dist"] if lens else np.zeros(5))[0].reshape(-1, 2)


MASK_BOXES = [((-30, 30), (-50, 50), (-10, 150), 25)]           # 腕の先を囲む箱（道具の x＝指の厚み・y＝指が開く向き、上へ z mm）と太らせる画素
# 下りながら追う時（cut）は、指（細い）とハンドの本体（太い）に分けて囲む。1 つの大きな箱では手前の下の角が指先よりずっと下に写り、
# 見えている消しゴムまで消していた（10/22、シミュレーター）
CUT_BOXES = [((-15, 15), (-25, 25), (-5, 45), 12), ((-30, 30), (-50, 50), (45, 150), 12)]


def robot_mask(shape, tcp, boxes=None):
    """腕の先（ハンドと手首のカメラ、TCP のまわり ±30×±50mm・上へ 150mm。10/20 の記録 9 枚で確かめた）が三脚の写真に写る所。"""
    R, t = rotvec_to_R(tcp[3:]), np.array(tcp[:3]) * 1000.0
    m = np.zeros(shape[:2], np.uint8)
    for (x0, x1), (y0, y1), (z0, z1), grow in boxes or MASK_BOXES:
        pts = [t + R @ np.array([dx, dy, 0.0]) + np.array([0, 0, dz]) for dx in (x0, x1) for dy in (y0, y1) for dz in (z0, z1)]
        one = np.zeros(shape[:2], np.uint8)
        cv2.fillConvexPoly(one, cv2.convexHull(to_px(pts, lens=False).astype(np.int32)), 255)
        m |= cv2.dilate(one, np.ones((grow, grow), np.uint8))
    return m


BG = None                                                       # 物の無いマットの三脚の写真（960×540）。無ければ「暗くない所」を物とする
BG_FILE = os.path.join(HERE, "tripod_bg.jpg")                   # 実機の背景の写真（物をどけて、腕を LOOK に置いて --save-bg で撮る）。TRI_BG=1 の時だけ使う（10/22）
if os.environ.get("TRI_BG", "0") == "1" and os.path.exists(BG_FILE):
    BG = cv2.imread(BG_FILE)
MAT_PX = np.array([[262, 200], [680, 200], [930, 502], [148, 502]], float)   # マットの四隅（960×540 の画素、10/20 の三脚の写真で読んだ）。
try:                                                           # 三脚を置き直して calib_tripod_mat.py --write で校正した時は、その時に読んだ四隅を使う（10/23）
    MAT_PX = np.array(json.load(open(os.path.join(HERE, "tripod_calib.json"), encoding="utf-8"))["mat_px"], float)
except (OSError, KeyError, ValueError):
    pass
# 前は暗い所のいちばん大きい塊の外形をマットとしていたが、腕が手前に来ると床とつながったり切れたりして、物が外に出た（記録 9 枚のうち 2 枚）


MIN_MM2 = float(os.environ.get("TRI_MIN_MM2", 0))                # >0：塊の小ささの下限を画素（300）でなく机の上の面積 mm² で決める。
                                                                # 大きいマットの奥の銀の棒は 1 画素≒1.5mm と小さく写り、300 画素に届かず捨てられた（10/22）


def px_mm(uv):
    """写真のこの画素の所で、1 画素が机の上で何 mm か（横に 1 画素ずらして測る）。"""
    a = to_table((float(uv[0]), float(uv[1])))
    b = to_table((float(uv[0]) + 1, float(uv[1])))
    return float(np.hypot(*(b - a)))


def blobs(img, cut=None):
    """マット（暗い所）に囲まれた明るい・色のある塊。(中心の画素, 輪郭, 長さ 画素)
    cut：この画素（腕の所）を物の候補から消してから探す（下りながら追う時。10/22）"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark = (hsv[..., 2] < 90).astype(np.uint8) * 255
    inside = np.zeros_like(dark)
    cv2.fillConvexPoly(inside, (MAT_PX * np.array([img.shape[1] / 960, img.shape[0] / 540])).astype(np.int32), 255)
    inside = cv2.erode(inside, np.ones((15, 15), np.uint8))
    if BG is not None and BG.shape == img.shape:                 # 物の無いマットの写真があれば、それと違う所を物とする（マットに明るい所があっても使える）
        fg = cv2.absdiff(img, BG).max(axis=2) > 40
    else:
        fg = dark == 0
    obj = ((inside > 0) & fg).astype(np.uint8) * 255
    if cut is not None:
        obj[cut > 0] = 0
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(obj)
    out = []
    for i in range(1, n):
        if MIN_MM2 > 0:                                          # 小さすぎる塊を mm² で捨てる（奥は小さく写る。10/22）
            if st[i, 4] < 30 or st[i, 4] > 8000 or st[i, 4] * px_mm(cen[i]) ** 2 < MIN_MM2:
                continue
        elif not (300 <= st[i, 4] <= 8000 and min(st[i, 2], st[i, 3]) >= 8):
            continue
        if True:
            for part in split((lab == i).astype(np.uint8), img):
                cnt = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
                m = cv2.moments(cnt)
                c = np.array([m["m10"] / max(m["m00"], 1e-9), m["m01"] / max(m["m00"], 1e-9)])
                out.append((c, cnt, max(cv2.minAreaRect(cnt)[1])))
    return out


SPLIT_W = float(os.environ.get("TRI_SPLIT_W", 40))              # 上から見た幅がこれ（mm）より広い塊は、くっついた物とみて分けてみる（10/20）


def split(mask, img):
    """くっついた物の塊を分ける。幅が SPLIT_W mm 以下ならそのまま。広ければ、縁からの距離の山を種にして分水嶺で分ける
    （山が 1 つしか無ければそのまま）。シミュレーターで、消しゴムと銀の棒が触れて 128×49mm の 1 つの塊になり、どちらも見つからなかった。"""
    cnt = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    if size_mm(cnt)[0] <= SPLIT_W:
        return [mask]
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    for frac in (0.5, 0.6, 0.7, 0.4):
        k, seeds = cv2.connectedComponents((dist > frac * dist.max()).astype(np.uint8))
        big = [j for j in range(1, k) if (seeds == j).sum() >= 15]
        if len(big) >= 2:
            break
    else:
        return [mask]
    markers = np.zeros(mask.shape, np.int32)
    for n_, j in enumerate(big, 1):
        markers[seeds == j] = n_
    markers[cv2.dilate(mask, np.ones((5, 5), np.uint8)) == 0] = len(big) + 1      # 外（背景）
    cv2.watershed(img.copy(), markers)
    parts = [((markers == n_) & (mask > 0)).astype(np.uint8) for n_ in range(1, len(big) + 1)]
    return [p_ for p_ in parts if p_.sum() >= 150] or [mask]


def objects(tcp, after=None, cut=False):
    """今の三脚の写真で見えるマットの上の物 [{"xy": 根元の座標 mm, "px": 画素, "len_px": 長さ, "contour": 輪郭}]（腕の先の所は除く）と写真。写真が無ければ (None, None)。
    cut：腕の先の所の画素を消してから探す（真下の物も残るが、腕に一部が隠れた物は小さく出る。下りながら追う時に大きさで確かめて使う。10/22）。
    無ければ前と同じく、真ん中が腕の先の所に入る塊を捨てる。"""
    img = grab(time.time() if after is None else after)
    if img is None:
        return None, None
    rm = robot_mask(img.shape, tcp, CUT_BOXES if cut else None)
    out = []
    for c, cnt, L in blobs(img, rm if cut else None):
        if not cut and rm[int(c[1]), int(c[0])]:
            continue
        out.append({"xy": table_center(cnt), "px": c, "len_px": float(L), "contour": cnt})
    return out, img


def match(found, tri, on_mat):
    """手首で見つけた物（found、on_mat(f) が真の物）と三脚の物を近さで組にする。三脚だけに見える物の並びを返し、
    組になった手首の物には f["tri_mm"]（三脚との位置の差）を入れる。"""
    rest = list(tri)
    for f in found:
        if not on_mat(f) or not rest:
            continue
        d = [np.linalg.norm(t["xy"] - np.asarray(f["center"][:2], float)) for t in rest]
        j = int(np.argmin(d))
        if d[j] < MATCH_MM:
            f["tri_mm"] = float(d[j])
            rest.pop(j)
    return rest


def table_center(cnt, h=OBJ_H):
    """輪郭を机から h の面へ写し、それを囲む長方形の真ん中（根元の座標 mm）。写真の上の重心は、斜めから見ると手前（大きく写る側）へ寄る
    （シミュレーターで、カメラの方を向いたペンが 20mm 手前に出た。10/20）。"""
    pts = np.array([to_table(p, h) for p in cnt.reshape(-1, 2).astype(float)], np.float32)
    return np.array(cv2.minAreaRect(pts)[0], float)


def size_mm(cnt, h=OBJ_H):
    """輪郭を机から h の面へ写した、上から見た (幅, 長さ) mm。"""
    pts = np.array([to_table(p) for p in cnt.reshape(-1, 2).astype(float)], np.float32)
    return tuple(sorted(cv2.minAreaRect(pts)[1]))


def kinds(img, objs, log=print):
    """三脚の写真の物それぞれの種類を d1-3B に決めさせ、o["kind"]・o["kind_p"]・o["size"] を入れる（10/20）。
    手首の時（kind_ai.ask）と同じ選択肢・切り抜き方で、状態の文だけ「三脚のカメラ、斜め上から。高さは測っていない」に替える。"""
    import kind_ai as KA
    for o in objs:
        o["size"] = size_mm(o["contour"])
        w, l = o["size"]
        o["kind"], o["kind_p"] = None, None
        choices = KA.options({"size": o["size"]})
        if len(choices) == 1:
            continue
        crop = KA.crop_upright(img, o["contour"], pad_short=4)   # まわりは 4 画素（10/20 の記録 9 枚：8 画素で 21/24・銀の棒が「それ以外」、4 画素で 24/24）
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        body = {"state": "Image from a fixed camera in front of the table, looking down at an angle, of one object lying on a dark cutting mat. "
                         f"Measured by the camera: seen from above it is {l:.0f} mm long and {w:.0f} mm wide (height not measured).",
                "question": "What is this object?", "choices": choices, "image": __import__("base64").b64encode(buf.tobytes()).decode("ascii")}
        t0 = time.perf_counter()
        r = KA.post(body)
        o["kind"] = KA.NAME[r["choice"]] if r["probs"][r["choice"]] >= KA.P_MIN else None
        o["kind_p"] = r["probs"]
        top = sorted(r["probs"].items(), key=lambda kv: -kv[1])[:2]
        log(f"   d1-3B（三脚）：{o['kind'] or 'それ以外'}（" + "、".join(f"{KA.NAME.get(k) or 'それ以外'} {p:.2f}" for k, p in top)
            + f"、{(time.perf_counter() - t0) * 1000:.0f}ms、長さ {l:.0f}・幅 {w:.0f}mm）")
    return objs


REF = os.path.join(HERE, "tripod_ref.jpg")                     # 三脚を置いて校正した時の写真（三脚が動いたかを比べる元。--save-ref で置く）
DRIFT_PX = float(os.environ.get("TRI_DRIFT_PX", 3.0))           # これより大きくずれていたら、三脚が動いたとみなす（1 画素 ≒ 0.78mm）
DRIFT_RESP = 0.3                                                # 確からしさがこれより低い時も動いたとみなす（10/20 の写真：動かさない時 0.62〜0.86、1° 回すと 0.08）


def drift(img, ref=None):
    """三脚の写真 img が、元の写真 ref（無ければ REF）から何画素ずれたか (dx, dy, 確からしさ 0〜1)。元が無ければ None（10/22）。
    物や腕が写る所（マットの中と、その上のあたり）は比べず、動かない床・机の縁で比べる（位相相関）。"""
    if ref is None:
        if not os.path.exists(REF):
            return None
        ref = cv2.imread(REF)
    if ref is None or ref.shape != img.shape:
        return None
    keep = np.full(img.shape[:2], 255, np.uint8)
    top = MAT_PX.copy()
    top[:2, 1] = 0                                               # マットの奥の縁から上（腕が写る所）も除く
    cv2.fillConvexPoly(keep, (top * np.array([img.shape[1] / 960, img.shape[0] / 540])).astype(np.int32), 0)
    keep = cv2.erode(keep, np.ones((31, 31), np.uint8))
    g = [np.where(keep > 0, cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), 0).astype(np.float32) for x in (ref, img)]
    win = cv2.createHanningWindow(g[0].shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(g[0], g[1], win)
    return float(dx), float(dy), float(resp)


def check_still(img, log=print):
    """三脚が校正した時から動いていないか。動いていたら SystemExit（校正し直す）。元の写真が無ければ知らせるだけ。"""
    d = drift(img)
    if d is None:
        log(f"   三脚が動いたかの確かめ：元の写真 {REF} が無いので、確かめていない（tripod_view.py --save-ref で置く）")
        return
    dx, dy, resp = d
    log(f"   三脚が動いたかの確かめ：元の写真から dx {dx:+.1f} dy {dy:+.1f} 画素（確からしさ {resp:.2f}）")
    if resp < DRIFT_RESP:
        raise SystemExit(f"三脚の写真が校正した時と大きく違います（確からしさ {resp:.2f}。回った・明かりが変わったなど）。"
                         "calib_tripod.py で校正し直してから --save-ref で元の写真を置き直してください")
    if np.hypot(dx, dy) > DRIFT_PX:
        raise SystemExit(f"三脚が校正した時から {np.hypot(dx, dy):.1f} 画素（約 {np.hypot(dx, dy) * 0.78:.0f}mm）動いています。"
                         "calib_tripod.py で校正し直してから --save-ref で元の写真を置き直してください")


if __name__ == "__main__":
    import sys
    if "--save-ref" in sys.argv:                                 # 今の三脚の写真を、比べる元として置く（校正した直後に）
        im = grab(0)
        if im is None:
            raise SystemExit("三脚の写真が来ない（cam_view.py --cam2 が動いているか）")
        cv2.imwrite(REF, im)
        print("元の写真を置いた:", REF)
    elif "--save-bg" in sys.argv:                               # 物をどけて、腕を見る位置（track_real.LOOK）に置いてから
        im = grab(0)
        if im is None:
            raise SystemExit("三脚の写真が来ない（cam_view.py --cam2 が動いているか）")
        cv2.imwrite(BG_FILE, im)
        print("背景の写真を置いた:", BG_FILE, "（使う時は TRI_BG=1）")
    elif "--check" in sys.argv:
        im = grab(0)
        if im is None:
            raise SystemExit("三脚の写真が来ない（cam_view.py --cam2 が動いているか）")
        check_still(im)
