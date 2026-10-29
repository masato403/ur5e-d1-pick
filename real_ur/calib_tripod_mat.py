r"""腕を動かさずに三脚のカメラを校正し直す（10/23）。三脚を置き直した時、または他の人が腕を使っていて calib_tripod.py ができない時に使う。
マットの四隅の画素を新しい写真で読み、根元の座標での四隅（10/20 の校正 tripod_calib.json と tripod_view.MAT_PX から出す）と合わせて、
solvePnP でカメラの位置と向きだけを求め直す。焦点距離・中心・レンズの曲がり（K・dist）は 10/20 のものを使い回す。
前提：マットとロボットの根元は 10/20 から動いていない。写真は、腕が四隅を隠していない時のものを選ぶ（隠れた角が 1 つなら - で外せる）。
粗さは数 mm の見込み。本番の精度が要る時は、腕を使う calib_tripod.py をやり直す。
使い方:
  .venv\Scripts\python.exe real_ur\calib_tripod_mat.py                 （tripod_latest.jpg に 50 画素ごとの目盛りを描いて cam_shots/tripod_mat_grid.jpg に出す）
  .venv\Scripts\python.exe real_ur\calib_tripod_mat.py --px 262,200 680,200 - 148,502   （読んだ四隅：左奥・右奥・右手前・左手前。隠れた角は -、3 つ以上）
  自動で四隅を読むのはやめた（10/23：マットは黒く、床の青い敷物・手前の黒い縁とつながる）。
  --write で tripod_calib.json を書き換える（前のものは tripod_calib_YYYYMMDD_HHMM.json に控える）。無ければ tripod_calib_mat.json に出すだけ。"""
import argparse
import json
import os
import shutil
import time

import cv2
import numpy as np

import tripod_view as TV
from plan_hover import TABLE_Z

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(HERE, "tripod_calib.json")


def mat_corners_mm():
    """根元の座標での四隅（机の面、mm）。to_table は手首とのずれ FIX を足すので引く（FIX は tripod_view が後で足す）。"""
    return np.array([np.r_[TV.to_table(p, 0.0) - TV.FIX, TABLE_Z] for p in TV.MAT_PX], np.float64)


def grid_image(img):
    """四隅の画素を読みやすいように、50 画素ごとの目盛りと前の四隅（黄）を描いた写真。"""
    g = img.copy()
    for x in range(0, g.shape[1], 50):
        cv2.line(g, (x, 0), (x, g.shape[0]), (0, 200, 255) if x % 100 == 0 else (120, 120, 120), 1)
        if x % 100 == 0:
            cv2.putText(g, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    for y in range(0, g.shape[0], 50):
        cv2.line(g, (0, y), (g.shape[1], y), (0, 200, 255) if y % 100 == 0 else (120, 120, 120), 1)
        if y % 100 == 0:
            cv2.putText(g, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    for p in TV.MAT_PX:
        cv2.circle(g, tuple(int(v) for v in p), 7, (0, 255, 255), 2)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", nargs="*")
    ap.add_argument("--px", nargs=4)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    old = json.load(open(CAL, encoding="utf-8"))
    W, H = old["image_size"]
    img = None
    if not a.px:                                                 # 画素を読む前：写真に目盛りを描いて出す
        img = cv2.imread(a.img[0]) if a.img else TV.grab(0.0)
        if img is None:
            print("三脚の写真が無い（cam_view.py --cam2 を動かすか --img で渡す）")
            return
        img = cv2.resize(img, (W, H)) if img.shape[1] != W else img
        out = os.path.join(HERE, "cam_shots", "tripod_mat_grid.jpg")
        cv2.imwrite(out, grid_image(img))
        print(f"{out} に目盛りを描いた（黄の丸は前の四隅）。マットの四隅の画素を読んで --px で渡す（左奥・右奥・右手前・左手前、隠れた角は -）")
        return
    if a.img:
        img = cv2.imread(a.img[0])
    seen = [s_ != "-" for s_ in a.px]
    if sum(seen) < 3:
        print("見える角が 3 つ以上要る")
        return
    px = np.array([[float(v) for v in s_.split(",")] if s_ != "-" else [np.nan, np.nan] for s_ in a.px])
    P = np.array(old["mat_mm"]) if "mat_mm" in old else mat_corners_mm()   # 2 回目からは、1 回目に出した四隅の位置をそのまま使う（ずれが積もらない）
    K, dist = np.array(old["K"]), np.array(old["dist"])
    sv = np.array(seen)
    if sv.all():                                                 # 4 つ見える：平らな 4 点の解き方
        ok, rvec, tvec = cv2.solvePnP(P, px.reshape(-1, 1, 2), K, dist, flags=cv2.SOLVEPNP_IPPE)
    else:                                                        # 3 つ：前の位置から解き始める（三脚は少ししか動いていない前提）
        ok, rvec, tvec = cv2.solvePnP(P[sv], px[sv].reshape(-1, 1, 2), K, dist, np.array(old["rvec"]).reshape(3, 1),
                                      np.array(old["tvec"]).reshape(3, 1), True, cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print("solvePnP が解けなかった")
        return
    rvec, tvec = cv2.solvePnPRefineLM(P[sv], px[sv].reshape(-1, 1, 2), K, dist, rvec, tvec)
    pr = cv2.projectPoints(P, rvec, tvec, K, dist)[0].reshape(-1, 2)
    err = np.linalg.norm(pr - px, axis=1)[sv]
    px = np.where(sv[:, None], px, pr)                          # 隠れた角は、解いた位置で埋める（mat_px 用）
    R = cv2.Rodrigues(rvec)[0]
    pos = (-R.T @ tvec).ravel()
    old_pos = np.array(old["cam_pos_mm"])
    print("四隅（画素） 前 → 今:", "  ".join(f"({o[0]:.0f},{o[1]:.0f})→({n[0]:.0f},{n[1]:.0f})" for o, n in zip(TV.MAT_PX, px)))
    print(f"合わせの誤差 平均 {err.mean():.1f}px・最大 {err.max():.1f}px")
    if not sv.all():
        print("注意：角 3 つでは解がちょうど決まるだけで、誤差 0px でも確かめになっていない（画素 10px のずれで位置が 20mm 動いた。10/23）。できれば 4 つ見える写真で")
    print(f"カメラの位置（根元の座標 mm）前 {np.round(old_pos, 1).tolist()} → 今 {np.round(pos, 1).tolist()}（{np.linalg.norm(pos - old_pos):.1f}mm 動いた）")
    print(f"机からの高さ {pos[2] - TABLE_Z:.0f}mm")
    new = dict(old)
    new.update({"when": time.strftime("%Y-%m-%d %H:%M"), "how": "calib_tripod_mat.py（マットの四隅、腕は動かさない。K・dist は前のまま）",
                "rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(), "cam_pos_mm": pos.tolist(),
                "reproj_px_mean": float(err.mean()), "reproj_px_max": float(err.max()), "mat_px": px.tolist(),
                "mat_mm": P.tolist(), "points": [], "excluded": []})
    if img is not None:                                          # 確かめ用：読んだ四隅（緑）と合わせた後の四隅（赤）
        dbg = img.copy()
        cv2.polylines(dbg, [px.astype(np.int32)], True, (0, 255, 0), 2)
        for p in pr:
            cv2.circle(dbg, tuple(int(v) for v in p), 6, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(HERE, "cam_shots", "tripod_mat_calib.jpg"), dbg)
    if a.write:
        bak = os.path.join(HERE, f"tripod_calib_{time.strftime('%Y%m%d_%H%M')}.json")
        shutil.copy(CAL, bak)
        json.dump(new, open(CAL, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"tripod_calib.json を書き換えた（前のものは {os.path.basename(bak)}）")
    else:
        out = os.path.join(HERE, "tripod_calib_mat.json")
        json.dump(new, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{os.path.basename(out)} に出した（使うなら --write）")


if __name__ == "__main__":
    main()
