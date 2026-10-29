"""今の位置で、持っている物を離して落とす（腕は動かさない）。ハンドは hand_server.py を通す。
使い方:  .venv_ur\Scripts\python.exe real_ur\drop_here.py --ip 192.0.2.10 --go
"""
import argparse

from hand import HandClient
from plan_hover import TABLE_Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(a.ip)
    p = r.getActualTCPPose()
    r.disconnect()
    h = HandClient()
    print(f"今: 指先は机から {p[2]*1000 - TABLE_Z:.0f}mm、ハンドの開き量 {h.pos_mm():.1f}mm")
    if not a.go:
        print("--go が無いので、ここまで（何も送っていません）")
        return
    print("離した: 開き量", round(h.open(), 1), "mm")
    h.close_port()


if __name__ == "__main__":
    main()
