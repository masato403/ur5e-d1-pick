"""実機の写真（cam_shots/archive/日付）から、マットの場所ごとの明るさ・色の濃さ・色合いの地図を作る（10/19 夜）。
シミュレーターのマットを実機に合わせる（暗い角でペンを見落とすのを再現する）ための下調べ。
マットの外形（凸の形）を写真ごとに求め、マットを基準にした位置（左上から mm）で 10mm ごとに、物でない所の HSV の中央値を集める。
結果：robot/logs/mat_light_<日付>.npz（V・S・H の 32×45 の地図）と、見るための絵 .png
使い方:  python robot/map_mat_light.py [日付 20261016]
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scan_archive as SA  # noqa: E402
import measure as M  # noqa: E402
import height_by_motion as HM  # noqa: E402

CELL = 10.0
SPAN = (-260.0, 260.0, -200.0, 200.0)   # 手先の真下から、手先の x・y の向きの範囲（mm）
BG_Q, LINE_Q = 30, 97                   # マスの中の画素の、地の色とみなす下からの割合（%）・方眼の線とみなす割合


def grid_shape():
    return int((SPAN[3] - SPAN[2]) / CELL), int((SPAN[1] - SPAN[0]) / CELL)


def tool_xy(P, tcp):
    """机の上の点（根元の座標 mm）→ 手先の真下からの位置を、手先の x・y の向き（写真の向きと同じ）で測った値（mm）。
    実機とシミュレーターは根元に対するマットの向きが逆（実機の y はマイナス、シミュレーターはプラス）なので、手先の向きで比べる"""
    from plan_hover import rotvec_to_R
    R = rotvec_to_R(tcp[3:])
    ex, ey = R[:2, 0] / np.linalg.norm(R[:2, 0]), R[:2, 1] / np.linalg.norm(R[:2, 1])
    d = np.asarray(P, np.float64)[:, :2] - np.array(tcp[:2]) * 1000
    return np.column_stack([d @ ex, d @ ey])


def cell_index(xy):
    """手先の真下からの x, y（mm）→ マスの (行, 列)。"""
    return ((xy[:, 1] - SPAN[2]) / CELL).astype(int), ((xy[:, 0] - SPAN[0]) / CELL).astype(int)


def mat_pixels(img):
    """マットの地か方眼の線らしい画素：物・ハンド・光る箱（明るく色の無い所）・床の青を除く。"""
    objs, mat = M.find_objects(img)
    filled = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(filled, cv2.findContours(mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, 255, -1)
    filled = cv2.fillConvexPoly(np.zeros_like(filled), cv2.convexHull(cv2.findNonZero(filled)), 255)
    bad = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(bad, objs, -1, 255, -1)
    bad = cv2.dilate(bad, np.ones((41, 41), np.uint8))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    shiny = ((hsv[..., 2] > 150) & (hsv[..., 1] < 35)).astype(np.uint8) * 255   # 箱・アルミの枠（方眼の線も入るので、大きな塊だけ）
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(shiny, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)))
    big = np.isin(lab, [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] > 8000])
    bad |= cv2.dilate(big.astype(np.uint8) * 255, np.ones((31, 31), np.uint8))
    h, sat = hsv[..., 0], hsv[..., 1]
    colored = (sat > 120) & (((h >= 95) & (h <= 130)) | (h <= 8) | (h >= 172))   # 見落としたペンの青・指の赤・床の青
    bad |= cv2.dilate(colored.astype(np.uint8) * 255, np.ones((31, 31), np.uint8))
    bad[int(img.shape[0] * 0.62):] = 255                      # ハンドの本体と指
    return (filled > 0) & (bad == 0)


def collect(pairs, pick=None):
    """(画像, 手先の姿勢) の並び → マスごとの BGR の画素の並び。pick(img, tcp)：数える画素（無ければ mat_pixels）"""
    ny, nx = grid_shape()
    acc = [[[] for _ in range(nx)] for _ in range(ny)]
    for img, tcp in pairs:
        ok_px = pick(img, tcp) if pick else mat_pixels(img)
        ys, xs = np.mgrid[0:img.shape[0]:4, 0:img.shape[1]:4]
        ys, xs = ys.ravel(), xs.ravel()
        k = ok_px[ys, xs]
        ys, xs = ys[k], xs[k]
        P = tool_xy(HM.on_plane(np.column_stack([xs, ys]).astype(np.float64), tcp, 0.0)[:, :2], tcp)
        i, j = cell_index(P)
        ok = (i >= 0) & (i < ny) & (j >= 0) & (j < nx)
        vals = img[ys[ok], xs[ok]]
        for ii, jj, val in zip(i[ok], j[ok], vals):
            acc[ii][jj].append(val)
    return acc


def summarize(acc, min_n=40):
    """マスごとの 地の色・線の色（BGR）と、その明るさ・色の濃さ。"""
    ny, nx = grid_shape()
    bg, line = np.full((ny, nx, 3), np.nan), np.full((ny, nx, 3), np.nan)
    for ii in range(ny):
        for jj in range(nx):
            if len(acc[ii][jj]) >= min_n:
                a_ = np.array(acc[ii][jj], np.float64)
                v = a_.max(1)
                lo, hi = np.percentile(v, BG_Q), np.percentile(v, LINE_Q)
                bg[ii, jj] = np.median(a_[v <= lo], axis=0)
                line[ii, jj] = np.median(a_[v >= hi], axis=0)
    return bg, line


def hsv_of(bgr):
    ok = np.isfinite(bgr[..., 0])
    out = np.full(bgr.shape, np.nan)
    out[ok] = cv2.cvtColor(np.clip(bgr[ok], 0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return out


def table(name, g):
    ny, nx = g.shape
    print(name + "（縦 5 × 横 6 にまとめた中央値。行は手先の y が小さい順、列は手先の x が小さい順）")
    for bi in np.array_split(np.arange(ny), 5):
        print("  " + "  ".join(f"{np.nanmedian(g[np.ix_(bi, bj)]):5.0f}" if np.isfinite(g[np.ix_(bi, bj)]).any() else "    -"
                               for bj in np.array_split(np.arange(nx), 6)))


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else "20261016"
    folder = os.path.join(SA.REAL, "cam_shots", "archive", day)
    pairs = [SA.load(folder, ta, "A") for ta, _, _ in SA.triplets(folder)]
    bg, line = summarize(collect(pairs))
    out = os.path.join(HERE, "logs", f"mat_light_{day}")
    np.savez(out + ".npz", bg=bg, line=line, cell=CELL, span=np.array(SPAN))
    hb, hl = hsv_of(bg), hsv_of(line)
    print(f"写真 {len(pairs)} 枚、測れたマス {np.isfinite(bg[..., 0]).sum()}/{bg.shape[0] * bg.shape[1]}")
    table("地の明るさ V", hb[..., 2])
    table("地の色の濃さ S", hb[..., 1])
    table("地の色合い H", hb[..., 0])
    table("線の明るさ V", hl[..., 2])
    table("線の色の濃さ S", hl[..., 1])


if __name__ == "__main__":
    main()
