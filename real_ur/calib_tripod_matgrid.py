r"""三脚のカメラを、腕の指先の点（tripod_calib.json の points）と、マットの四隅（寸法が分かっている長方形）を一緒に使って校正し直す（10/23）。腕は動かさない。
指先の点は机から 50mm と 110mm の 2 つの高さだけなので、焦点距離とカメラまでの距離の見分けがつかず、机の面に写すとマットが台形になった
（奥 542mm・手前 380mm。本当は 455×320mm の長方形。焦点距離が 348 画素＝横の画角 108° と出ていた）。
机の面の上にある、形の分かったマットの四隅を足すと、焦点距離と奥行きが決まる。マットの置き場所（x・y・回り）も同時に求める。
使い方:
  .venv\Scripts\python.exe real_ur\calib_tripod_matgrid.py                       （結果を tripod_calib_matgrid.json に出すだけ）
  .venv\Scripts\python.exe real_ur\calib_tripod_matgrid.py --px 272,165 650,158 741,410 180,423   （四隅の画素：左奥・右奥・右手前・左手前）
  --write で tripod_calib.json を書き換える（前のものは tripod_calib_YYYYMMDD_HHMM.json に控える）。"""
import argparse
import json
import os
import shutil
import time

import cv2
import numpy as np

from plan_hover import TABLE_Z

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(HERE, "tripod_calib.json")
MAT_W, MAT_D = 455.0, 320.0      # マットの外の寸法（mm、定規で測った、10/23）。横（根元の x の向き）× 奥行き
MAT_T = 2.0                      # マットの厚さ（mm、見当）
PX_10_23 = ["272,165", "650,158", "741,410", "180,423"]   # 10/23 の tripod_ref.jpg で拡大して読んだ四隅


def project(p, P3):
    f, cx, cy, k1 = p[:4]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    uv, _ = cv2.projectPoints(P3.reshape(-1, 1, 3), p[4:7], p[7:10], K, np.array([k1, 0, 0, 0, 0.0]))
    return uv.reshape(-1, 2)


def mat_corners(p):
    """マットの四隅（根元の座標、mm）。p[10:12] が左奥の角、p[12] が回り（rad）。左奥→右奥は +x、奥→手前は -y。"""
    ox, oy, yaw = p[10:13]
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array([[0, 0], [MAT_W, 0], [MAT_W, -MAT_D], [0, -MAT_D]])
    xy = np.array([ox, oy]) + loc @ np.array([[c, s], [-s, c]])
    return np.c_[xy, np.full(4, TABLE_Z + MAT_T)]


def residual(p, P3, P2, M2, w_mat):
    return np.r_[(project(p, P3) - P2).ravel(), w_mat * (project(p, mat_corners(p)) - M2).ravel()]


FREE = None                      # 動かす値の番号（None は全部）


def lm(p, args, iters=200):
    """Levenberg–Marquardt（数値微分）。scipy が無いので手で書く。FREE に無い値は動かさない。"""
    lam = 1e-3
    free = list(range(p.size)) if FREE is None else FREE
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
            if lam > 1e8:
                return p, r
        if np.linalg.norm(step) < 1e-8:
            break
    return p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", nargs=4, default=PX_10_23)
    ap.add_argument("--w-mat", type=float, default=1.0)       # 四隅の重み（指先の点 1 つと同じ）
    ap.add_argument("--fix-center", action="store_true")     # 画像の中心を写真の真ん中（480, 270）に決めて動かさない
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    global FREE
    if a.fix_center:
        FREE = [0, 3] + list(range(4, 13))
    cal = json.load(open(CAL, encoding="utf-8"))
    keep = [i for i in range(len(cal["points"])) if i not in cal.get("excluded", [])]
    P3 = np.array([cal["points"][i]["tcp_mm"] for i in keep], float)
    P2 = np.array([cal["points"][i]["uv"] for i in keep], float)
    M2 = np.array([[float(v) for v in s.split(",")] for s in a.px])
    K = np.array(cal["K"])
    p0 = np.r_[K[0, 0], 480.0 if a.fix_center else K[0, 2], 270.0 if a.fix_center else K[1, 2], cal["dist"][0], np.ravel(cal["rvec"]), np.ravel(cal["tvec"]), -76.0, -171.0, 0.0]
    args = (P3, P2, M2, a.w_mat)
    r0 = residual(p0, *args)
    # 初めの見当：前の校正のまま、マットの置き場所だけ合わせる（焦点距離が違いすぎると落ち込むので、焦点距離も何通りか試す）
    best = None
    for f in (348.0, 500.0, 650.0, 800.0):
        q = p0.copy()
        q[0] = f
        ok, rv, tv = cv2.solvePnP(P3, P2, np.array([[f, 0, q[1]], [0, f, q[2]], [0, 0, 1.0]]), np.array([q[3], 0, 0, 0, 0.0]))
        q[4:7], q[7:10] = rv.ravel(), tv.ravel()
        q, r = lm(q, args)
        print(f"  見当 f={f:.0f} → f={q[0]:.0f}、ずれの二乗和 {r @ r:.0f}")
        if best is None or r @ r < best[1] @ best[1]:
            best = (q, r)
    p, r = best
    n = len(P3)
    e_arm = np.linalg.norm(r[:2 * n].reshape(-1, 2), axis=1)
    e_mat = np.linalg.norm(r[2 * n:].reshape(-1, 2), axis=1) / a.w_mat
    e0_mat = np.linalg.norm(r0[2 * n:].reshape(-1, 2), axis=1) / a.w_mat
    R = cv2.Rodrigues(p[4:7])[0]
    cam = -R.T @ p[7:10]
    hfov = np.degrees(2 * np.arctan(480 / p[0]))
    print(f"前の校正：焦点距離 {K[0, 0]:.0f} 画素（横の画角 {np.degrees(2 * np.arctan(480 / K[0, 0])):.0f}°）、"
          f"指先のずれ 平均 {cal['reproj_px_mean']:.1f} 画素、四隅のずれ（マットの置き場所だけ合わせても）{e0_mat.round(1).tolist()}")
    print(f"新しい校正：焦点距離 {p[0]:.0f} 画素（横の画角 {hfov:.0f}°）、中心 ({p[1]:.0f}, {p[2]:.0f})、k1 {p[3]:.3f}")
    print(f"  指先のずれ 平均 {e_arm.mean():.1f}・最大 {e_arm.max():.1f} 画素、四隅のずれ {e_mat.round(1).tolist()} 画素")
    print(f"  カメラの位置 x {cam[0]:.0f} y {cam[1]:.0f} 机から {cam[2] - TABLE_Z:.0f}mm")
    mc = mat_corners(p)
    print(f"  マットの四隅（左奥・右奥・右手前・左手前）{mc[:, :2].round(0).tolist()}、回り {np.degrees(p[12]):.1f}°")
    out = dict(cal)
    out.update(when=time.strftime("%Y-%m-%d %H:%M"), K=[[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1.0]], dist=[p[3], 0, 0, 0, 0],
               rvec=p[4:7].tolist(), tvec=p[7:10].tolist(), cam_pos_mm=cam.tolist(), reproj_px_mean=float(e_arm.mean()),
               reproj_px_max=float(e_arm.max()), mat_px=M2.tolist(), mat_corners_mm=mc.tolist(), mat_err_px=e_mat.tolist(),
               note=cal.get("note", "") + f" / 10/23 calib_tripod_matgrid.py：指先の点＋マットの四隅（{MAT_W:.0f}×{MAT_D:.0f}mm）で合わせ直した")
    dst = CAL if a.write else os.path.join(HERE, "tripod_calib_matgrid.json")
    if a.write:
        shutil.copy(CAL, os.path.join(HERE, time.strftime("tripod_calib_%Y%m%d_%H%M.json")))
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("書いた:", dst)


if __name__ == "__main__":
    main()
