r"""三脚のカメラを、マットの 1cm のマス目の交点（たくさん）と、腕の指先の点（tripod_calib.json の points）で校正し直す（10/23）。腕は動かさない。
指先の点だけ（机から 50mm と 110mm の 2 つの高さ）では焦点距離と奥行きの見分けがつかず、四隅を足しても合わなかった（calib_tripod_matgrid.py）。
マス目の交点は机の面の上に何百もあり、間が 10mm と分かっているので、焦点距離・ゆがみ・カメラとマットの位置関係が決まる。
マットが根元のどこにあるか（x・y・回り）は、指先の点で決める。
  1) 四隅の画素で大まかにマットを真上から見た絵に直し、縦横の線の並び（10mm おき）を見つける
  2) 交点ごとに、近くでいちばん線らしい所を拾い、元の写真の画素に戻す（2 回くり返して見当を良くする）
  3) 焦点距離・中心・ゆがみ k1・カメラの位置と向き・マットの置き場所を、交点と指先の点のずれが小さくなるように合わせる
使い方:
  .venv\Scripts\python.exe real_ur\calib_tripod_grid.py          （結果を tripod_calib_grid.json と、確かめの絵 tripod_calib_grid.jpg に出すだけ）
tripod_calib.json は書き換えない（確かめてから切り替える）。"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from plan_hover import TABLE_Z

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(HERE, "tripod_calib.json")
IMG = os.path.join(HERE, "tripod_ref.jpg")
OUT = os.path.join(HERE, "tripod_calib_grid.json")
OUT_IMG = os.path.join(HERE, "tripod_calib_grid.jpg")
MAT_W, MAT_D = 455.0, 320.0      # マットの外の寸法（mm、定規で測った）
MAT_T = 2.0                      # マットの厚さ（mm、見当）
PX_10_23 = [(272, 165), (650, 158), (741, 410), (180, 423)]   # 10/23 の tripod_ref.jpg の四隅（左奥・右奥・右手前・左手前）
S = 3.0                          # 真上から見た絵の細かさ（画素/mm）
PITCH = 10.0                     # マス目（mm）
MARGIN = 6.0                     # マットの縁から、この（mm）より内側の交点だけ使う


def rectify(img, H):
    """H：マットの mm（左手前が原点、右が +x、奥が +y）→ 写真の画素。真上から見た絵（S 画素/mm、上が奥）を返す。"""
    W, D = int(MAT_W * S), int(MAT_D * S)
    A = np.array([[1 / S, 0, 0], [0, -1 / S, MAT_D], [0, 0, 1.0]])   # 絵の画素 → マットの mm
    return cv2.warpPerspective(img, H @ A, (W, D), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP), A


def line_map(rect):
    """明るい細い線（マス目）を強める。"""
    g = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(4 * S) | 1, int(4 * S) | 1))
    return cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)


def comb_phase(profile):
    """10mm おきに並ぶ山の位置（mm、0〜10）を見つける。"""
    n = profile.size
    best = (-1, 0.0)
    for ph in np.arange(0, PITCH, 0.1):
        idx = np.round((np.arange(ph, n / S, PITCH)) * S).astype(int)
        idx = idx[(idx >= 0) & (idx < n)]
        sc = profile[idx].mean()
        if sc > best[0]:
            best = (sc, ph)
    return best[1]


def find_points(img, H):
    """マス目の交点を拾う。返り値：(マットの mm の点 N×2, 写真の画素 N×2, 交点の番号 N×2)。"""
    rect, A = rectify(img, H)
    lm = line_map(rect)
    m = int(MARGIN * S)
    inner = lm[m:-m, m:-m]
    ph_x = comb_phase(inner.mean(axis=0)) + MARGIN               # 縦の線（x）の位置
    ph_y_img = comb_phase(inner.mean(axis=1)) + MARGIN           # 横の線（絵の上から）
    ph_y = (MAT_D - ph_y_img) % PITCH                            # マットの mm（手前から）に直す
    xs = np.arange(ph_x, MAT_W - MARGIN, PITCH)
    ys = np.arange(ph_y + PITCH * np.ceil((MARGIN - ph_y) / PITCH), MAT_D - MARGIN, PITCH)
    # 十字の形で、近く（±2.5mm）でいちばん線らしい所を探す
    cross = np.zeros((int(5 * S) | 1,) * 2, np.float32)
    c = cross.shape[0] // 2
    cross[c, :] = 1
    cross[:, c] = 1
    resp = cv2.filter2D(lm, -1, cross / cross.sum())
    mm, px, ij = [], [], []
    r = int(2.5 * S)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            u, v = int(round(x * S)), int(round((MAT_D - y) * S))
            if not (r <= u < resp.shape[1] - r and r <= v < resp.shape[0] - r):
                continue
            win = resp[v - r:v + r + 1, u - r:u + r + 1]
            dv, du = np.unravel_index(int(np.argmax(win)), win.shape)
            if win[dv, du] < 0.5 * np.median(resp[m:-m, m:-m]) + 2:
                continue
            # 山の形から少数点以下まで（左右・上下に 2 次式）
            def sub(a, b, cc):
                den = a - 2 * b + cc
                return 0.0 if abs(den) < 1e-6 else 0.5 * (a - cc) / den
            uu, vv = u - r + du, v - r + dv
            if 1 <= du < win.shape[1] - 1 and 1 <= dv < win.shape[0] - 1:
                uu += sub(win[dv, du - 1], win[dv, du], win[dv, du + 1])
                vv += sub(win[dv - 1, du], win[dv, du], win[dv + 1, du])
            q = A @ np.array([uu, vv, 1.0])
            p = H @ np.array([q[0], q[1], 1.0])
            mm.append((x, y))
            px.append(p[:2] / p[2])
            ij.append((i, j))
    return np.array(mm), np.array(px), np.array(ij), (ph_x, ph_y)


def project(p, P3):
    f, cx, cy, k1 = p[:4]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    uv, _ = cv2.projectPoints(P3.reshape(-1, 1, 3), p[4:7], p[7:10], K, np.array([k1, 0, 0, 0, 0.0]))
    return uv.reshape(-1, 2)


def mat_to_base(p, mm):
    """マットの mm（左手前が原点）→ 根元の座標。p[10:12] が左手前の角、p[12] が回り。"""
    ox, oy, yaw = p[10:13]
    c, s = np.cos(yaw), np.sin(yaw)
    xy = np.array([ox, oy]) + np.asarray(mm, float) @ np.array([[c, s], [-s, c]])
    return np.c_[xy, np.full(len(xy), TABLE_Z + MAT_T)]


def residual(p, G_mm, G_px, P3, P2, w_arm):
    return np.r_[(project(p, mat_to_base(p, G_mm)) - G_px).ravel(), w_arm * (project(p, P3) - P2).ravel()]


def lm(p, args, free, iters=300):
    """Levenberg–Marquardt（数値微分。scipy が無いので手で書く）。free に無い値は動かさない。"""
    lam = 1e-3
    r = residual(p, *args)
    for _ in range(iters):
        J = np.empty((r.size, len(free)))
        for k, i in enumerate(free):
            d = np.zeros_like(p)
            d[i] = 1e-6 * max(1.0, abs(p[i]))
            J[:, k] = (residual(p + d, *args) - r) / d[i]
        A, g = J.T @ J, J.T @ r
        while True:
            step = np.zeros_like(p)
            step[free] = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            r2 = residual(p + step, *args)
            if r2 @ r2 < r @ r:
                p, r, lam = p + step, r2, lam * 0.3
                break
            lam *= 10
            if lam > 1e10:
                return p, r
        if np.linalg.norm(step) < 1e-9:
            break
    return p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=IMG)
    ap.add_argument("--w-arm", type=float, default=1.0)       # 指先の点 1 つの重み（交点 1 つを 1 とする）
    ap.add_argument("--corners", type=float, nargs=8)           # 四隅の画素（左奥・右奥・右手前・左手前の u v）。配置を変えた時（10/23 夕方）
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--mat-pts")                                # mat_grid_points.json：指先をマス目の交点の上に置いて読んだ点（10/23 夕方）。マットの置き場所をこれで決めて動かさない
    ap.add_argument("--tick0", type=float, nargs=2, default=[0.0, 0.0])   # 目盛り 0,0 の交点の、マットの左手前の角からの位置（mm）
    ap.add_argument("--save-rect")                              # 真上から見た絵を書き出す（目盛り 0 の位置を見るため）
    a = ap.parse_args()
    img = cv2.imread(a.img)
    cal = json.load(open(CAL, encoding="utf-8"))
    keep = [i for i in range(len(cal["points"])) if i not in cal.get("excluded", [])]
    P3 = np.array([cal["points"][i]["tcp_mm"] for i in keep], float)
    P2 = np.array([cal["points"][i]["uv"] for i in keep], float)

    # 1)・2) 四隅で大まかに → 交点を拾う → 交点で H を合わせ直す → もう一度拾う
    corners_mm = np.array([[0, MAT_D], [MAT_W, MAT_D], [MAT_W, 0], [0, 0]], np.float32)
    H = cv2.getPerspectiveTransform(corners_mm, np.array(np.reshape(a.corners, (4, 2)) if a.corners else PX_10_23, np.float32))
    for it in range(3):
        G_mm, G_px, ij, ph = find_points(img, H)
        H2, inl = cv2.findHomography(G_mm.astype(np.float32), G_px.astype(np.float32), cv2.RANSAC, 1.5)
        e = np.linalg.norm(cv2.perspectiveTransform(G_mm.reshape(-1, 1, 2).astype(np.float64), H2).reshape(-1, 2) - G_px, axis=1)
        print(f"  {it + 1} 回目：交点 {len(G_mm)} 個（線の位置 x {ph[0]:.1f}・y {ph[1]:.1f}mm）、平面に合わせたずれ 中央 {np.median(e):.2f}・"
              f"90% {np.percentile(e, 90):.2f} 画素、外れ {int((~inl.ravel().astype(bool)).sum())} 個")
        H = H2
    good = inl.ravel().astype(bool)
    G_mm, G_px = G_mm[good], G_px[good]
    if a.save_rect:
        cv2.imwrite(a.save_rect, rectify(img, H)[0])
    fixed = None
    if a.mat_pts:
        # 指先の傾きの分を直す（合わせた点は道具の軸に沿って L mm ずれる。L も一緒に求める）→ 目盛りの mm（＋tick0）と根元の xy の回り＋ずらし
        d_ = json.load(open(a.mat_pts, encoding="utf-8"))
        Gt = np.array([v["grid_cm"] for v in d_.values()], float) * 10 + np.array(a.tick0)
        Pt = np.array([v["tcp_mm"][:2] for v in d_.values()], float)
        Zt = np.array([cv2.Rodrigues(np.array(v["rv"]))[0][:2, 2] for v in d_.values()])
        A_, b_ = [], []
        for g, q, z in zip(Gt, Pt, Zt):
            A_ += [[g[0], -g[1], 1, 0, -z[0]], [g[1], g[0], 0, 1, -z[1]]]
            b_ += [q[0], q[1]]
        x_ = np.linalg.lstsq(np.array(A_), np.array(b_), rcond=None)[0]
        e_ = np.linalg.norm((np.array(A_) @ x_ - b_).reshape(-1, 2), axis=1)
        fixed = (x_[2], x_[3], float(np.arctan2(x_[1], x_[0])))
        print(f"  指先の点 {len(Gt)} 個：倍率 {np.hypot(x_[0], x_[1]):.4f}、回り {np.degrees(fixed[2]):.2f}°、L {x_[4]:.1f}mm、ずれ {e_.round(1).tolist()} mm")
        a.w_arm = 0.0

    # 3) 焦点距離など＋カメラの位置＋マットの置き場所
    K = np.array(cal["K"])
    best = None
    for f in ((350.0, 450.0, 550.0, 700.0) if fixed else (350.0, 500.0, 650.0, 800.0, 1000.0)):
        Kf = np.array([[f, 0, 480.0], [0, f, 270.0], [0, 0, 1.0]])
        # マットの置き場所の見当：指先の点の真下あたり（左手前の角は、指先の x 最小・y 最小の近く）
        for ox, oy in ((-30.0, -450.0), (-30.0, -520.0), (0.0, -430.0)):
            q = np.r_[f, 480.0, 270.0, 0.0, np.zeros(6), ox, oy, 0.0]
            if fixed:
                q[10:13] = fixed
                ok, rv, tv = cv2.solvePnP(mat_to_base(q, G_mm), G_px, Kf, None)
                q[4:7], q[7:10] = rv.ravel(), tv.ravel()
                q, r = lm(q, (G_mm, G_px, P3, P2, a.w_arm), free=list(range(10)))
                if best is None or r @ r < best[1] @ best[1]:
                    best = (q, r)
                break
            ok, rv, tv = cv2.solvePnP(P3, P2, Kf, None)
            q[4:7], q[7:10] = rv.ravel(), tv.ravel()
            q, _ = lm(q, (G_mm, G_px, P3, P2, a.w_arm), free=[10, 11, 12], iters=100)
            q, r = lm(q, (G_mm, G_px, P3, P2, a.w_arm), free=list(range(13)))
            if best is None or r @ r < best[1] @ best[1]:
                best = (q, r)
    p, r = best
    n = len(G_mm)
    e_g = np.linalg.norm(r[:2 * n].reshape(-1, 2), axis=1)
    e_a = np.linalg.norm((project(p, P3) - P2), axis=1)              # 前の指先の点（配置を変えた後は合わなくて当たり前）
    R = cv2.Rodrigues(p[4:7])[0]
    cam = -R.T @ p[7:10]
    mc = mat_to_base(p, [[0, MAT_D], [MAT_W, MAT_D], [MAT_W, 0], [0, 0]])
    print(f"前の校正：焦点距離 {K[0, 0]:.0f} 画素（横の画角 {np.degrees(2 * np.arctan(480 / K[0, 0])):.0f}°）、指先のずれ 平均 {cal['reproj_px_mean']:.1f} 画素")
    print(f"新しい校正：焦点距離 {p[0]:.0f} 画素（横の画角 {np.degrees(2 * np.arctan(480 / p[0])):.0f}°）、中心 ({p[1]:.0f}, {p[2]:.0f})、k1 {p[3]:.3f}")
    print(f"  交点のずれ 中央 {np.median(e_g):.2f}・90% {np.percentile(e_g, 90):.2f}・最大 {e_g.max():.2f} 画素（{n} 個）")
    print(f"  指先のずれ 平均 {e_a.mean():.1f}・最大 {e_a.max():.1f} 画素（{len(P3)} 個）：{e_a.round(1).tolist()}")
    print(f"  カメラの位置 x {cam[0]:.0f} y {cam[1]:.0f} 机から {cam[2] - TABLE_Z:.0f}mm")
    print(f"  マットの四隅（左奥・右奥・右手前・左手前）{mc[:, :2].round(0).tolist()}、回り {np.degrees(p[12]):.2f}°")
    # 確かめの絵：拾った交点（緑）、合わせた模型で描いた交点（赤）、指先の点（黄：測った、水色：模型）
    vis = img.copy()
    for (u, v) in G_px:
        cv2.circle(vis, (int(round(u)), int(round(v))), 2, (0, 255, 0), -1)
    for (u, v) in project(p, mat_to_base(p, G_mm)):
        cv2.circle(vis, (int(round(u)), int(round(v))), 1, (0, 0, 255), -1)
    for (u, v), (u2, v2) in zip(P2, project(p, P3)):
        cv2.circle(vis, (int(round(u)), int(round(v))), 5, (0, 255, 255), 1)
        cv2.circle(vis, (int(round(u2)), int(round(v2))), 3, (255, 255, 0), -1)
    edge = project(p, mc).astype(int)
    cv2.polylines(vis, [edge.reshape(-1, 1, 2)], True, (255, 0, 255), 1)
    cv2.imwrite(a.out.replace('.json', '.jpg'), vis)
    out = dict(cal)
    out.update(when=time.strftime("%Y-%m-%d %H:%M"), K=[[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1.0]], dist=[p[3], 0, 0, 0, 0],
               rvec=p[4:7].tolist(), tvec=p[7:10].tolist(), cam_pos_mm=cam.tolist(), reproj_px_mean=float(e_a.mean()),
               reproj_px_max=float(e_a.max()), grid_px_median=float(np.median(e_g)), grid_n=int(n),
               mat_corners_mm=mc.tolist(), mat_yaw_deg=float(np.degrees(p[12])),
               note=cal.get("note", "") + f" / 10/23 calib_tripod_grid.py：マス目の交点 {n} 個＋指先の点で合わせ直した")
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("書いた:", a.out)


if __name__ == "__main__":
    main()
