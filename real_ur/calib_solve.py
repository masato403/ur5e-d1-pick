"""手首カメラの校正（ロボットは動かさない）：calib_capture.py で撮ったマットの方眼から、
  ① レンズの曲がり（カメラ行列と歪み係数）  ② カメラの取り付け（ツール TCP_2 から見たカメラの位置と向き）を求める。

  1) 方眼の線を取り出し（暗いマットの上の明るい細い線）、交点を見つける
  2) 交点に「何マス目か」の番号を付ける（隣どうしをたどる。曲がりがあっても局所の向きで追う）
     番号の 0 点は、消しゴム（目印）にいちばん近い交点。向きは 1 枚目にそろえる → 全部の写真で同じ番号になる
  3) 番号 × 10mm を机の上の座標として、cv2.calibrateCamera で ①、cv2.calibrateHandEye で ② を求める
  4) 確かめ：消しゴムの位置を、各写真から机の上の座標として計算し、ばらつきを見る
使い方:  .venv_ur\\Scripts\\python.exe real_ur\\calib_solve.py
"""
import json
import math
import os

import cv2
import numpy as np

from plan_hover import rotvec_to_R

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, os.environ.get("CALIB_DIR", "calib2"))
PITCH = 10.0           # mm：方眼 1 マス


def mat_mask(img):
    """マット（暗い紫）の部分。指（赤）・ハンド・机の枠・消しゴム（白）を外す。"""
    hsv = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (0, 0), 9)
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 35, 178))        # マット：色が薄く（彩度 < 35）、枠（明るさ 210 前後）より暗い
    dark[int(img.shape[0] * 0.72):] = 0                       # 下の方はハンドの本体
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
    if n <= 1:
        return dark
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8) * 255
    return cv2.erode(m, np.ones((15, 15), np.uint8))


def eraser_center(img):
    """目印（マットの上の白い物：平らな紙や消しゴム）の中心。向きに関係なく、回した長方形によく埋まる白い塊で、
    周りがマットに囲まれているものを選ぶ。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mat = cv2.inRange(cv2.GaussianBlur(hsv, (0, 0), 5), (0, 0, 0), (180, 40, 185))
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if not 400 < area < 60000:
            continue
        (cx, cy), (w, h), _ = cv2.minAreaRect(c)
        if area / max(1.0, w * h) < 0.75 or max(w, h) / max(1.0, min(w, h)) > 5 or cy > img.shape[0] * 0.75:
            continue
        blob = np.zeros(white.shape, np.uint8)
        cv2.drawContours(blob, [c], -1, 255, -1)
        ring = cv2.dilate(blob, np.ones((31, 31), np.uint8)) & ~cv2.dilate(blob, np.ones((11, 11), np.uint8))
        around = (mat[ring > 0] > 0).mean() if (ring > 0).any() else 0
        if around < 0.6:
            continue
        if best is None or area > best[1]:
            best = ((cx, cy), area)
    return None if best is None else np.array(best[0])


def grid_points(img, mask):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    lines = (top > 18).astype(np.uint8) * 255
    lines[mask == 0] = 0
    lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    blur = cv2.GaussianBlur(lines.astype(np.float32), (0, 0), 2.0)
    har = cv2.cornerHarris(blur, 7, 5, 0.04)
    har[mask == 0] = 0
    mx = cv2.dilate(har, np.ones((15, 15), np.uint8))
    pk = (har == mx) & (har > 0.02 * har.max())
    ys, xs = np.nonzero(pk)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    # 細かく：応答の重心
    out = []
    for x, y in pts:
        x0, y0 = int(x), int(y)
        ya, yb, xa, xb = max(0, y0 - 4), min(har.shape[0], y0 + 5), max(0, x0 - 4), min(har.shape[1], x0 + 5)
        win = har[ya:yb, xa:xb].clip(0)
        if win.sum() <= 0:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb]
        out.append([(xx * win).sum() / win.sum(), (yy * win).sum() / win.sum()])
    return np.array(out, np.float32), lines


def index_lattice(pts, start_xy, ref_axes=None):
    """交点に (i, j) の番号を付ける。start_xy に一番近い交点から、隣をたどる。"""
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    nn = tree.query(pts, k=2)[0][:, 1]
    spacing = float(np.median(nn[nn > 20]))                     # 方眼 1 マスの画素数（偽の点どうしの近い組は除く）

    def cross(k):
        """上下左右に 1 マス先の隣が 4 つ（2 組の向かい合わせ）ある、きれいな交点なら (v1, v2) を返す。"""
        d, nb = tree.query(pts[k], k=7)
        vs = [pts[n] - pts[k] for dd, n in zip(d[1:], nb[1:]) if 0.75 * spacing < dd < 1.25 * spacing]
        if len(vs) != 4:
            return None
        v1 = vs[0]
        opp = min(vs[1:], key=lambda v: np.linalg.norm(v + v1))
        rest = [v for v in vs[1:] if v is not opp]
        if np.linalg.norm(opp + v1) > 0.25 * spacing or np.linalg.norm(rest[0] + rest[1]) > 0.25 * spacing:
            return None
        return v1, rest[0]

    order = np.argsort(np.linalg.norm(pts - start_xy, axis=1))
    s, v1, v2 = None, None, None
    for k in order:
        c = cross(int(k))
        if c is not None:
            s, (v1, v2) = int(k), c
            break
    if s is None:
        return {}, None
    if ref_axes is not None:                                     # 1 枚目の向きにそろえる
        best = None
        for a, b in ((v1, v2), (v2, v1)):
            for sa in (1, -1):
                for sb in (1, -1):
                    sc = np.dot(sa * a / np.linalg.norm(a), ref_axes[0]) + np.dot(sb * b / np.linalg.norm(b), ref_axes[1])
                    if best is None or sc > best[0]:
                        best = (sc, sa * a, sb * b)
        v1, v2 = best[1], best[2]
    idx = {s: (0, 0)}
    loc = {s: (v1, v2)}
    queue = [s]
    taken = {(0, 0): s}
    while queue:
        k = queue.pop(0)
        i, j = idx[k]
        a, b = loc[k]
        for di, dj, step in ((1, 0, a), (-1, 0, -a), (0, 1, b), (0, -1, -b)):
            key = (i + di, j + dj)
            if key in taken:
                continue
            pred = pts[k] + step
            dd, n = tree.query(pred)
            if dd > 0.25 * np.linalg.norm(step) or n in idx:
                continue
            ok = True                                            # ほかの番号付きの隣とも食い違わないか
            for ei, ej in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                m = taken.get((key[0] + ei, key[1] + ej))
                if m is None or m == k:
                    continue
                ma, mb = loc[m]
                exp = pts[m] - ei * ma - ej * mb
                if np.linalg.norm(exp - pts[n]) > 0.3 * spacing:
                    ok = False
                    break
            if not ok:
                continue
            idx[n] = key
            taken[key] = n
            obs = pts[n] - pts[k]
            na, nb_ = (obs * (1 if di + dj > 0 else -1), b) if di != 0 else (a, obs * (1 if di + dj > 0 else -1))
            loc[n] = (0.5 * (na + a), 0.5 * (nb_ + b))                # 局所の向きを少しずつ更新（曲がりに追従）
            queue.append(n)
    return idx, (v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))


def to_lattice_coords(img_pt, pts, idx):
    """画像の点を、近くの交点 12 個から作った局所の射影変換で (i, j) に直す。"""
    ks = list(idx)
    P = pts[ks]
    near = np.argsort(np.linalg.norm(P - img_pt, axis=1))[:12]
    src = P[near]
    dst = np.array([idx[ks[n]] for n in near], np.float32)
    H, _ = cv2.findHomography(src, dst, 0)
    v = H @ np.array([img_pt[0], img_pt[1], 1.0])
    return v[:2] / v[2]


def main():
    meta = json.load(open(os.path.join(DIR, "poses.json")))
    obj_all, img_all, used, ref_axes = [], [], [], None
    anchor_frac = []
    for vi, v in enumerate(meta["views"]):
        img = cv2.imread(os.path.join(DIR, v["image"]))
        mask = mat_mask(img)
        pts, lines = grid_points(img, mask)
        e = eraser_center(img)
        if e is None or len(pts) < 40:
            print(f"{v['image']}: 目印か交点が足りない（交点 {len(pts)}）→ 使わない")
            continue
        idx, axes = index_lattice(pts, e, ref_axes)
        if len(idx) < 40:
            print(f"{v['image']}: 番号を付けられた交点が少ない（{len(idx)}）→ 使わない")
            continue
        if ref_axes is None:
            ref_axes = axes
        ec = to_lattice_coords(e, pts, idx)                      # 消しゴムの中心の (i, j)
        base = np.floor(ec)                                      # 消しゴムの左下の交点を 0 点に
        anchor_frac.append(ec - base)
        ks = list(idx)
        obj = np.array([[(idx[k][0] - base[0]) * PITCH, (idx[k][1] - base[1]) * PITCH, 0] for k in ks], np.float32)
        imgp = pts[ks].reshape(-1, 1, 2)
        obj_all.append(obj)
        img_all.append(imgp)
        used.append(vi)
        dbg = img.copy()
        for k in ks:
            cv2.circle(dbg, tuple(int(t) for t in pts[k]), 4, (0, 255, 0), -1)
            if idx[k] == (0, 0) or (idx[k][0] - base[0], idx[k][1] - base[1]) == (0, 0):
                cv2.circle(dbg, tuple(int(t) for t in pts[k]), 12, (0, 0, 255), 3)
        cv2.circle(dbg, tuple(int(t) for t in e), 10, (255, 0, 0), 3)
        cv2.imwrite(os.path.join(DIR, f"dbg_{v['image'][:-4]}.jpg"), cv2.resize(dbg, (960, 540)))
        print(f"{v['image']}: 交点 {len(pts)} 個、番号を付けた {len(ks)} 個、消しゴムの位置 (i, j) = ({ec[0]:.2f}, {ec[1]:.2f})")
    fr = np.array(anchor_frac)
    print("消しゴムの位置の端数（全部の写真でそろっていれば番号が一致している）:", np.round(fr.mean(0), 2), "ばらつき", np.round(fr.std(0), 3))

    h, w = cv2.imread(os.path.join(DIR, meta["views"][0]["image"])).shape[:2]
    flags = cv2.CALIB_RATIONAL_MODEL
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_all, img_all, (w, h), None, None, flags=flags)
    for rnd in range(4):                                         # 計算と合わない交点・写真を外して解き直す
        keep_o, keep_i, keep_u = [], [], []
        errs = []
        for n in range(len(obj_all)):
            proj, _ = cv2.projectPoints(obj_all[n], rvecs[n], tvecs[n], K, dist)
            e = np.linalg.norm(proj.reshape(-1, 2) - img_all[n].reshape(-1, 2), axis=1)
            errs.append(float(np.median(e)))
            good = e < max(3.0, 2.5 * np.median(e))
            if good.sum() >= 40 and np.median(e) < 6.0:
                keep_o.append(obj_all[n][good])
                keep_i.append(img_all[n][good])
                keep_u.append(used[n])
        print(f"   外して解き直し {rnd + 1} 回目：使う写真 {len(keep_u)} 枚（写真ごとの誤差の中央値 {np.round(errs, 1).tolist()}）")
        obj_all, img_all, used = keep_o, keep_i, keep_u
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_all, img_all, (w, h), None, None, flags=flags)
    print(f"① レンズ：再投影の誤差 {rms:.2f} 画素、焦点距離 {K[0,0]:.0f}／{K[1,1]:.0f} 画素、中心 ({K[0,2]:.0f}, {K[1,2]:.0f})")
    print("   歪み係数:", np.round(dist.ravel()[:8], 4).tolist())

    Rg, tg, Rt, tt = [], [], [], []
    for n, vi in enumerate(used):
        tcp = meta["views"][vi]["tcp"]
        Rg.append(rotvec_to_R(tcp[3:]))
        tg.append(np.array(tcp[:3]) * 1000.0)
        Rt.append(cv2.Rodrigues(rvecs[n])[0])
        tt.append(tvecs[n].ravel())
    # 向き：腕と方眼の回転だけから解く（方眼の番号のずれに影響されない）
    Rc, _ = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=cv2.CALIB_HAND_EYE_PARK)
    RG = [Rg[n] @ Rc @ Rt[n] for n in range(len(used))]
    rg = np.array([cv2.Rodrigues(R)[0].ravel() for R in RG])
    R_G = cv2.Rodrigues(np.median(rg, 0))[0]                     # 机の上の方眼の向き（根元の座標で）
    print("   方眼の向きのばらつき（度）:", round(float(np.degrees(np.std([np.linalg.norm(cv2.Rodrigues(R_G.T @ R)[0]) for R in RG]))), 2))
    # 位置：t_X（カメラの位置）・t_G（方眼の 0 点）・各写真の番号のずれ n_i（整数マス）を交互に解く
    nshift = [np.zeros(2) for _ in used]
    for it in range(10):
        A, y = [], []
        for n in range(len(used)):
            row = np.zeros((3, 6))
            row[:, :3] = Rg[n]
            row[:, 3:] = -np.eye(3)
            A.append(row)
            y.append(R_G @ np.array([nshift[n][0] * PITCH, nshift[n][1] * PITCH, 0.0]) - Rg[n] @ Rc @ tt[n] - tg[n])
        sol = np.linalg.lstsq(np.vstack(A), np.concatenate(y), rcond=None)[0]
        tc, tG = sol[:3], sol[3:]
        new_shift = []
        for n in range(len(used)):
            p = Rg[n] @ (Rc @ tt[n] + tc) + tg[n]                  # この写真の方眼の 0 点（根元の座標）
            d = R_G.T @ (p - tG)
            new_shift.append(nshift[n] - np.round(d[:2] / PITCH))
        changed = any(np.any(a_ != b_) for a_, b_ in zip(new_shift, nshift))
        nshift = new_shift
        if not changed:
            break
    print(f"   位置の解き直し {it + 1} 回、写真ごとの番号のずれ（マス）:", [tuple(int(v) for v in s_) for s_ in nshift])
    res = []
    for n in range(len(used)):
        p = Rg[n] @ (Rc @ tt[n] + tc) + tg[n]
        res.append(R_G.T @ (p - tG) + np.array([-nshift[n][0] * PITCH, -nshift[n][1] * PITCH, 0.0]))
    res = np.array(res)
    print("   方眼の 0 点の残りのずれ（mm、方眼の向きで x, y, 面に垂直）: ばらつき", np.round(res.std(0), 2).tolist())
    good = [n for n in range(len(used)) if np.all(np.abs(nshift[n]) <= 1) and np.linalg.norm(res[n] - np.median(res, 0)) < 5]
    if len(good) < len(used):                                   # 外れた写真を除いて、位置をもう一度解く
        A, y = [], []
        for n in good:
            row = np.zeros((3, 6)); row[:, :3] = Rg[n]; row[:, 3:] = -np.eye(3)
            A.append(row)
            y.append(R_G @ np.array([nshift[n][0] * PITCH, nshift[n][1] * PITCH, 0.0]) - Rg[n] @ Rc @ tt[n] - tg[n])
        sol = np.linalg.lstsq(np.vstack(A), np.concatenate(y), rcond=None)[0]
        tc, tG = sol[:3], sol[3:]
        res = np.array([R_G.T @ (Rg[n] @ (Rc @ tt[n] + tc) + tg[n] - tG) - np.array([nshift[n][0] * PITCH, nshift[n][1] * PITCH, 0.0]) for n in good])
        print(f"   外れた写真 {len(used) - len(good)} 枚を除いて解き直し：残りのずれのばらつき", np.round(res.std(0), 2).tolist(), "mm")
    ang = np.degrees(np.linalg.norm(cv2.Rodrigues(Rc)[0]))
    print(f"② 取り付け（TCP_2 から見たカメラ）：位置 {np.round(tc, 1).tolist()} mm、回転 {ang:.1f}°、カメラの向く向き（TCP_2 の座標で）{np.round(Rc[:, 2], 3).tolist()}")
    o = tG[None, :]
    nz = R_G[:, 2][None, :]
    print("方眼の 0 点（根元の座標 mm）:", np.round(tG, 1).tolist(), " 方眼の面の向き（z 軸）:", np.round(R_G[:, 2], 3).tolist())
    json.dump({"K": K.tolist(), "dist": dist.ravel().tolist(), "size": [w, h], "rms": rms,
               "R_cam2tcp": Rc.tolist(), "t_cam2tcp_mm": tc.tolist(),
               "grid_origin_base_mm": o.mean(0).tolist(), "grid_origin_std_mm": res.std(0).tolist(),
               "grid_R_base": R_G.tolist(), "views_used": used},
              open(os.path.join(DIR, "calib_result.json"), "w"), indent=1)
    print("保存: calib/calib_result.json")


if __name__ == "__main__":
    main()
