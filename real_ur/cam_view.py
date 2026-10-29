"""手首カメラの映像を窓に出し続ける（ロボットは動かさない）。

  画面の左上に、ピントの良さ（大きいほどくっきり）と明るさ（0〜255、120 前後がちょうど良い）を出す。
  レンズの曲がりを直した映像を出す（校正 calib2/calib_result.json）。キー：u = 直す／直さないの切り替え。
  キー：s = 今の画像を保存、q = 閉じる。5 秒ごとに最新の画像を latest.jpg にも保存する（ほかのプログラムから読む用）。
  latest.jpg は直す前の画像のまま（locate.py が画素の位置を、直す前の画像として使うため）。
  撮る合図（cam_shots/grab_req.txt）があれば、カメラにたまった古い画像を FLUSH 枚読み捨て、次の画像を元の大きさ（直す前）で
  grab_full.jpg に保存し、grab_done.txt に合図の番号を書く（locate.grab_now が使う。1 枚 6 秒待っていたのを 1 秒以下に、10/16）。
10/19 から、窓ではなく Chrome の普通のタブ（http://127.0.0.1:8766/）に出す（CAM_WEB=0 で前の窓）。
ページには、pick_pass.py などが撮った写真と進み具合の文字も並ぶ（web_page.py）。ページのボタンで u・s と同じことができる。
使い方:  .venv\\Scripts\\python.exe real_ur\\cam_view.py --cam 1
"""
import argparse
import json
import os
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("CAM_OUT") or os.path.join(HERE, "cam_shots")   # 三脚のカメラを同時に映す時は別の所へ（10/20）
REQ, DONE, FULL = (os.path.join(OUT, n) for n in ("grab_req.txt", "grab_done.txt", "grab_full.jpg"))
FLUSH = 2                    # 合図の後に読み捨てる画像の数（カメラの中にたまった、合図より前の画像）
CAL = os.path.join(HERE, "calib2", "calib_result.json")
MODE = os.environ.get("CAM_WEB", "1")   # 1：ページ、0：窓、2：ページと窓の両方（10/19 夜）
WEB, WIN = MODE in ("1", "2"), MODE in ("0", "2")


def undistort_maps(w, h):
    """校正の K（1920×1080）を今の画像の大きさに合わせ、曲がりを直す表を作る。黒い縁が出ない範囲で切り取る。"""
    cal = json.load(open(CAL))
    K, D = np.array(cal["K"], float), np.array(cal["dist"], float)
    K = K * np.array([[w / 1920.0], [h / 1080.0], [1.0]])
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0.0, (w, h))
    return cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_16SC2)


WRIST_EVERY = float(os.environ.get("CAM_WRIST_EVERY", 0))   # >0 なら、手首の写真（元の大きさ）をこの間（秒）ごとに wrist_live.jpg に置く（track_real が常に使う。10/22）
TRI_EVERY = float(os.environ.get("CAM_TRI_EVERY", 0.5))   # 三脚の写真を残す間（秒）。前は 5 秒で、通しで「今の写真」が取れなかった（10/20）


def tripod_loop(cam, page):
    """三脚のカメラを読み続け、ページの右の枠に出す。TRI_EVERY 秒ごとに tripod_latest.jpg（960×540）にも残す。"""
    c = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    last = 0.0
    while True:
        ok, f = c.read()
        if not ok:
            time.sleep(0.05)
            continue
        if page is not None:
            page.live2(f)
        if time.time() - last > TRI_EVERY:                      # 書き終わってから名前を替える（読む側が書きかけを読まない）
            tmp = os.path.join(OUT, "tripod_latest_tmp.jpg")
            cv2.imwrite(tmp, cv2.resize(f, (960, 540)))
            try:
                os.replace(tmp, os.path.join(OUT, "tripod_latest.jpg"))
            except PermissionError:                              # 読む側が開いている間は替えられない。次のコマで替える（10/23、ここで止まり校正が写真を待ち続けた）
                continue
            last = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--cam2", type=int, default=-1)              # 三脚のカメラの番号（同じページに並べる。-1 で使わない。10/20）
    ap.add_argument("--seconds", type=float, default=12 * 3600)   # 前は 1800（30 分）で、実機の通しの途中でカメラが止まり、古い写真で測った（10/19 夜）
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    c = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW)
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    win = "wrist camera (real)"
    page = None
    if WEB:
        from web_page import Page
        page = Page()
        page.log(time.strftime("---- %H:%M:%S カメラの係を始めた\n"))
    if a.cam2 >= 0:
        threading.Thread(target=tripod_loop, args=(a.cam2, page), daemon=True).start()
    if WIN:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
    t0 = last = time.time()
    raised = False
    maps, fix = None, os.path.exists(CAL)
    req, skip = None, 0
    last_w = 0.0
    while time.time() - t0 < a.seconds:
        ok, f = c.read()
        if not ok:
            time.sleep(0.05)
            continue
        if req is None and os.path.exists(REQ):
            try:
                req = open(REQ).read().strip()
            except OSError:
                req = None
            skip = FLUSH
        elif req is not None:
            if skip > 0:
                skip -= 1
            else:
                tmp = FULL + ".tmp.jpg"
                cv2.imwrite(tmp, f, [cv2.IMWRITE_JPEG_QUALITY, 95])
                os.replace(tmp, FULL)                    # 書きかけを読まれないように、書き終えてから名前を変える
                with open(DONE, "w") as fp:
                    fp.write(req)
                try:
                    os.remove(REQ)
                except OSError:
                    pass
                req = None
        if WRIST_EVERY > 0 and time.time() - last_w >= WRIST_EVERY:
            last_w = time.time()
            tmp = os.path.join(OUT, "wrist_live_tmp.jpg")
            cv2.imwrite(tmp, f, [cv2.IMWRITE_JPEG_QUALITY, 90])
            os.replace(tmp, os.path.join(OUT, "wrist_live.jpg"))  # 書き終えてから名前を変える。撮った時刻＝読んだ時刻（last_w）を名前の時刻にする
            os.utime(os.path.join(OUT, "wrist_live.jpg"), (last_w, last_w))
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(cv2.resize(g, (960, 540)), cv2.CV_64F).var()
        bright = g.mean()
        if fix and maps is None:
            maps = undistort_maps(f.shape[1], f.shape[0])
        view = cv2.remap(f, maps[0], maps[1], cv2.INTER_LINEAR) if fix else f.copy()
        label, col = ("undistorted (u)", (0, 255, 0)) if fix else ("raw (u)", (0, 160, 255))
        cv2.putText(view, label, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 7)
        cv2.putText(view, label, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 2)
        cv2.putText(view, f"sharp {sharp:6.0f}   bright {bright:5.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8)
        cv2.putText(view, f"sharp {sharp:6.0f}   bright {bright:5.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3)
        k = 0xFF
        if page is not None:
            page.live(view)
            keys = page.commands()
            k = ord(keys[0]) if keys and keys[0] else 0xFF
        if WIN:
            cv2.imshow(win, view)
        if WIN and not raised:                                   # 初めて出た時だけ手前に出す
            cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(1)
            cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 0)
            raised = True
        if time.time() - last > 5:
            cv2.imwrite(os.path.join(OUT, "latest.jpg"), cv2.resize(f, (960, 540)))
            with open(os.path.join(OUT, "latest.txt"), "w") as fp:
                fp.write(f"{sharp:.0f} {bright:.1f}\n")
            last = time.time()
        if WIN:
            kw = cv2.waitKey(1) & 0xFF
            k = kw if kw != 0xFF else k
        if k == ord("u") and os.path.exists(CAL):
            fix = not fix
            print("曲がりを直す:", "あり" if fix else "なし", flush=True)
            if page is not None:
                page.log(f"曲がりを直す: {'あり' if fix else 'なし'}\n")
        if k == ord("s"):
            p = os.path.join(OUT, time.strftime("shot_%H%M%S.jpg"))
            cv2.imwrite(p, f)
            print("保存:", p, flush=True)
            if page is not None:
                page.log(f"保存: {p}\n")
        if WIN and (k == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1):
            break
    c.release()
    if WIN:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
