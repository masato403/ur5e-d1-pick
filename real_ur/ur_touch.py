"""机の面を、道具の先で触った点から測る（読むだけ。ロボットには何も送らない）。

使い方:
  .venv_ur\\Scripts\\python.exe real_ur\\ur_touch.py --ip 192.0.2.10 --add     いまの先端の位置を 1 点記録する
  .venv_ur\\Scripts\\python.exe real_ur\\ur_touch.py --show                        記録した点から机の高さ・傾き・向きを計算する
  .venv_ur\\Scripts\\python.exe real_ur\\ur_touch.py --clear                       記録を消す
座標はロボットの根元の座標（PolyScope の「ベース」と同じ。単位 mm）。
"""
import argparse
import json
import math
import os

import numpy as np

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_points.json")


def load():
    try:
        return json.load(open(FILE, encoding="utf-8"))
    except (OSError, ValueError):
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip")
    ap.add_argument("--add", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--clear", action="store_true")
    a = ap.parse_args()
    pts = load()
    if a.clear:
        pts = []
    if a.add:
        from rtde_receive import RTDEReceiveInterface
        r = RTDEReceiveInterface(a.ip)
        tcp, q = r.getActualTCPPose(), r.getActualQ()
        r.disconnect()
        pts.append({"xyz_mm": [round(v * 1000, 1) for v in tcp[:3]], "rot": [round(v, 4) for v in tcp[3:]],
                    "q_deg": [round(math.degrees(v), 2) for v in q]})
        p = pts[-1]["xyz_mm"]
        print(f"{len(pts)} 点目: x {p[0]:.1f}  y {p[1]:.1f}  z {p[2]:.1f} mm ／ 関節 {pts[-1]['q_deg']}")
    json.dump(pts, open(FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    if a.show or a.add:
        P = np.array([p["xyz_mm"] for p in pts])
        if len(P):
            print("記録した点 [mm]:")
            for i, p in enumerate(P, 1):
                print(f"  {i}: x {p[0]:7.1f}  y {p[1]:7.1f}  z {p[2]:7.1f}  根元からの水平距離 {math.hypot(p[0], p[1]):5.0f}  向き {math.degrees(math.atan2(p[1], p[0])):6.1f}°")
            print(f"机の高さ（根元の座標の z）: 平均 {P[:, 2].mean():.1f} mm（ばらつき {P[:, 2].max() - P[:, 2].min():.1f} mm）")
        if len(P) >= 3:
            c = P.mean(axis=0)
            n = np.linalg.svd(P - c)[2][-1]
            n = n if n[2] > 0 else -n
            print(f"机の傾き: {math.degrees(math.acos(min(1.0, n[2]))):.2f}°（0° なら根元と平行）")


if __name__ == "__main__":
    main()
