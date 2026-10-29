"""手首カメラの校正用に、腕の高さと傾きを変えながらマットの方眼を撮る（実機を動かす）。

  基準：今の位置（消しゴムの真上、指先は真下向き）。指先の点を中心に、高さと傾きを変える。
    高さ（机から）: HEIGHTS mm、傾き：まっすぐ・道具の x 軸／y 軸まわりに ±TILT°
  各姿勢で止まってから、カメラの最高解像度で 1 枚撮り、そのときの腕の姿勢（ツール先端・関節）を記録する。
  保存先：real_ur/calib/（画像 viewNN.png と poses.json）。最後に、出発した姿勢へ戻る。
使い方:  .venv_ur\\Scripts\\python.exe real_ur\\calib_capture.py --ip 192.0.2.10 --go
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

from plan_hover import R_to_rotvec, TABLE_Z, rotvec_to_R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.environ.get("CALIB_DIR", "calib2"))
HEIGHTS = [100, 150, 200]
TILT = 12.0
SPEED, ACCEL = 0.05, 0.2


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()

    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(a.ip)
    p0 = r.getActualTCPPose()
    R0 = rotvec_to_R(p0[3:])
    poses = []
    for h in HEIGHTS:
        for axis, deg in (("x", 0), ("x", TILT), ("x", -TILT), ("y", TILT), ("y", -TILT)):
            R = R0 @ rot(axis, deg)                      # 道具の向きを、道具自身の軸のまわりに傾ける
            poses.append([p0[0], p0[1], (TABLE_Z + h) / 1000] + list(R_to_rotvec(R)))
    for dx, dy in ((23, 0), (-17, 0), (0, 23), (0, -17), (16, 13)):  # 横にずらす（10mm の倍数を避ける）
        poses.append([p0[0] + dx / 1000, p0[1] + dy / 1000, (TABLE_Z + 150) / 1000] + list(p0[3:]))
    for h in (100, 150):                                # 手首の先を回す（カメラの位置を決めるのに効く）
        for deg in (20, -20):
            poses.append([p0[0], p0[1], (TABLE_Z + h) / 1000] + list(R_to_rotvec(R0 @ rot("z", deg))))
    print(f"撮る姿勢: {len(poses)} 個（高さ {HEIGHTS} mm × 傾き 0／±{TILT}°）")
    if not a.go:
        print("--go が無いので、ここまで（何も送っていません）")
        return

    from rtde_control import RTDEControlInterface
    os.makedirs(OUT, exist_ok=True)
    cap = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    for _ in range(10):
        cap.read()
    c = RTDEControlInterface(a.ip)
    rec = []
    try:
        for i, pose in enumerate(poses):
            if not c.isPoseWithinSafetyLimits(pose):
                print(f"  {i:02d}: 安全設定の範囲の外なので飛ばす")
                continue
            c.moveL(pose, SPEED, ACCEL)
            time.sleep(0.6)                                # 揺れが収まるまで
            for _ in range(5):                              # 古い画像を捨てる
                cap.read()
            ok, img = cap.read()
            if not ok:
                print(f"  {i:02d}: 撮れなかった")
                continue
            name = f"view{i:02d}.png"
            cv2.imwrite(os.path.join(OUT, name), img)
            tcp, q = r.getActualTCPPose(), r.getActualQ()
            rec.append({"image": name, "tcp": list(tcp), "q": list(q)})
            print(f"  {i:02d}: 机から {tcp[2]*1000 - TABLE_Z:.0f}mm で撮った")
        c.moveL(p0, SPEED, ACCEL)
        print("出発した姿勢へ戻った")
    finally:
        c.stopScript()
        c.disconnect()
        r.disconnect()
        cap.release()
        json.dump({"start_tcp": list(p0), "views": rec}, open(os.path.join(OUT, "poses.json"), "w"), indent=1)
    print(f"{len(rec)} 枚を {OUT} に保存")


if __name__ == "__main__":
    main()
