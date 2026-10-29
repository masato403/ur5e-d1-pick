r"""指先を、決めた位置・高さへまっすぐ動かす（実機）。手首の先を真上から見て回す量も指定できる。
使い方:  .venv_ur\Scripts\python.exe real_ur\move_to.py --ip 192.0.2.10 --x 129.9 --y -433.9 --h 100 --turn -66 --go
"""
import argparse
import math

import numpy as np

from plan_hover import R_to_rotvec, TABLE_Z, rotvec_to_R

SPEED, ACCEL = 0.05, 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--h", type=float, required=True, help="机から指先までの高さ mm（50 以上）")
    ap.add_argument("--turn", type=float, default=0.0, help="手首の先を回す角度（度、±90 まで）")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    if a.h < 50 or abs(a.turn) > 90:
        raise SystemExit("高さは 50mm 以上、回す角度は ±90° まで")
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(a.ip)
    p0 = r.getActualTCPPose()
    t = math.radians(a.turn)
    Rz = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    goal = [a.x / 1000, a.y / 1000, (TABLE_Z + a.h) / 1000] + list(R_to_rotvec(Rz @ rotvec_to_R(p0[3:])))
    d = math.dist(goal[:3], p0[:3]) * 1000
    print(f"今: x {p0[0]*1000:.1f}  y {p0[1]*1000:.1f}  机から {p0[2]*1000 - TABLE_Z:.0f}mm → "
          f"x {a.x:.1f}  y {a.y:.1f}  机から {a.h:.0f}mm、手首の先 {a.turn:+.0f}°（{d:.0f}mm 動く）")
    if d > 400:
        raise SystemExit("40cm 以上動くので止めます")
    if not a.go:
        print("--go が無いので、ここまで（何も送っていません）")
        return
    c = RTDEControlInterface(a.ip)
    try:
        if not c.isPoseWithinSafetyLimits(goal):
            raise SystemExit("行き先が安全設定の範囲の外なので止めます")
        c.moveL(goal, SPEED, ACCEL)
        p = r.getActualTCPPose()
        print(f"着いた: x {p[0]*1000:.1f}  y {p[1]*1000:.1f}  机から {p[2]*1000 - TABLE_Z:.0f}mm")
    finally:
        c.stopScript()
        c.disconnect()
        r.disconnect()


if __name__ == "__main__":
    main()
