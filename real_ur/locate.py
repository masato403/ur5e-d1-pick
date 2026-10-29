"""手首カメラの映像の点を、机の上の位置（根元の座標 mm）に直す（校正の結果 calib2/calib_result.json を使う）。

  画素 → レンズの曲がりを直した光線 → カメラの位置と向き（今の腕の姿勢 × 取り付け）→ 机の面と交わる点。
使い方:  .venv_ur\\Scripts\\python.exe real_ur\\locate.py --ip 192.0.2.10      （白い目印を探して位置を出す）
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from calib_solve import eraser_center
from plan_hover import rotvec_to_R

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = json.load(open(os.path.join(HERE, "calib2", "calib_result.json")))
K, DIST = np.array(CAL["K"]), np.array(CAL["dist"])
RC, TC = np.array(CAL["R_cam2tcp"]), np.array(CAL["t_cam2tcp_mm"])
G0, GN = np.array(CAL["grid_origin_base_mm"]), np.array(CAL["grid_R_base"])[:, 2]


def pixel_to_table(px, tcp):
    """px：最高解像度（1920×1080）の画素。tcp：そのときの腕の姿勢（ur_rtde の getActualTCPPose、m と回転ベクトル）。"""
    n = cv2.undistortPoints(np.array([[px]], np.float64), K, DIST).reshape(2)
    d_cam = np.array([n[0], n[1], 1.0])
    RB, tB = rotvec_to_R(tcp[3:]), np.array(tcp[:3]) * 1000.0
    o = RB @ TC + tB                                   # カメラの位置（根元の座標 mm）
    d = RB @ RC @ d_cam                                # 光線の向き
    s = np.dot(G0 - o, GN) / np.dot(d, GN)
    return o + s * d


SETTLE = float(os.environ.get("LOC_SETTLE", 0.15))   # 撮る前の待ち（秒）。10/16 実機：0・0.15・0.3・0.5 秒で写真のぶれも消しゴムの高さ・中心も変わらなかった（前は 0.5）


def grab_now(timeout=3.0, settle=None):
    """今の画像（元の大きさ 1920×1080、直す前）。カメラの窓（cam_view.py）に撮る合図を出し、返事を待つ。
    settle 秒待ってから合図を出す（腕を止めた直後の揺れが収まるまで）。timeout 秒で返事が無ければ（窓が古い版など）、
    今までのやり方（6 秒待って latest.jpg）に戻す。"""
    shot = os.path.join(HERE, "cam_shots")
    req, done, full = (os.path.join(shot, n) for n in ("grab_req.txt", "grab_done.txt", "grab_full.jpg"))
    settle = SETTLE if settle is None else settle
    time.sleep(settle)
    rid = f"{time.time():.3f}"
    with open(req, "w") as fp:
        fp.write(rid)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if open(done).read().strip() == rid:
                img = cv2.imread(full)
                if img is not None:
                    return img
        except OSError:
            pass
        time.sleep(0.02)
    try:
        os.remove(req)
    except OSError:
        pass
    print("   （カメラの窓から返事が無いので、6 秒待って latest.jpg を使う。cam_view.py を新しい版で開き直すと速くなる）")
    time.sleep(max(0.0, 6.0 - settle - timeout))
    return grab()


def archive(img, name, data=None):
    """撮った画像（と腕の姿勢などの data）を、上書きせずに cam_shots/archive/日付/時刻_name.jpg（.json）に残す（後で使えるように、10/16）。"""
    d = os.path.join(HERE, "cam_shots", "archive", time.strftime("%Y%m%d"))
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, time.strftime("%H%M%S_") + name)
    cv2.imwrite(base + ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if data is not None:
        with open(base + ".json", "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    return base + ".jpg"


def grab(cam=1):
    """手首カメラの窓（cam_view.py）が動いていれば、その最新の画像（960×540）を使う。"""
    p = os.path.join(HERE, "cam_shots", "latest.jpg")
    img = cv2.imread(p)
    return cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    a = ap.parse_args()
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(a.ip)
    tcp = r.getActualTCPPose()
    r.disconnect()
    img = grab()
    e = eraser_center(img)
    if e is None:
        raise SystemExit("目印が見つからない")
    p = pixel_to_table(e, tcp)
    print(f"目印の画素 ({e[0]:.0f}, {e[1]:.0f}) → 机の上 x {p[0]:.1f}  y {p[1]:.1f}  z {p[2]:.1f} mm")
    print(f"今の指先（TCP_2）の真下: x {tcp[0]*1000:.1f}  y {tcp[1]*1000:.1f} mm → 目印までのずれ "
          f"({p[0] - tcp[0]*1000:+.1f}, {p[1] - tcp[1]*1000:+.1f}) mm")
    json.dump({"mark_mm": p.tolist(), "pixel": e.tolist(), "tcp": list(tcp)}, open(os.path.join(HERE, "mark.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
