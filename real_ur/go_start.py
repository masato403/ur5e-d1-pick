r"""腕を、見張りの最初の位置（10/16 夕方に撮った位置：机から約 148mm、マット全体が写る）へ、関節の動きでゆっくり戻す。
撮る位置は「今いる所」から決まるので、つかんだ後のまま撮ると、写る範囲と向きがずれた（10/16）。
使い方:  .venv_ur\Scripts\python.exe real_ur\go_start.py --ip 192.0.2.10
"""
import argparse
import math

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

START_DEG = [83.5, -97.3, 125.9, -118.5, -89.7, 83.4]
SPEED, ACCEL = 1.0, 1.5                                  # 関節の速さ（rad/秒）・加速度（UR のふだんの速さ 約 1.05 rad/秒）


def go_start(c, r):
    q0 = r.getActualQ()
    qt = [math.radians(v) for v in START_DEG]
    big = max(abs(math.degrees(a - b)) for a, b in zip(qt, q0))
    if big > 120:
        raise SystemExit(f"最初の位置まで関節が {big:.0f}° も動くので止めます（手で近くまで動かしてください）")
    c.moveJ(qt, SPEED, ACCEL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    a = ap.parse_args()
    r = RTDEReceiveInterface(a.ip)
    c = RTDEControlInterface(a.ip)
    try:
        go_start(c, r)
        p = r.getActualTCPPose()
        print("最初の位置へ戻った：手先", [round(v * 1000, 1) for v in p[:3]], "mm")
    finally:
        c.stopScript()
        c.disconnect()
        r.disconnect()


if __name__ == "__main__":
    main()
