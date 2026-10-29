"""持っている物を、横へ動かしてから机に置く（実機）。

  ハンドは hand_server.py（USB をつなぎっぱなしにする係）を通して動かす。
  以前は持ったまま USB を開け直してリセットがかかり、0 点がずれて全開で突き当たった（10/15、ガリガリ鳴った）。
使い方:  .venv_ur\\Scripts\\python.exe real_ur\\place_here.py --ip 192.0.2.10 --dx 50 --dy 0 --hold 6 --go
"""
import argparse
from hand import HandClient
from plan_hover import TABLE_Z

SPEED, DOWN_SPEED, ACCEL = 0.03, 0.02, 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--dx", type=float, default=50.0, help="横へ動かす量 mm（根元の座標）")
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--hold", type=float, default=6.0, help="つかんだ時の指先の高さ（机から mm）")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    if abs(a.dx) > 100 or abs(a.dy) > 100:
        raise SystemExit("横へ動かすのは 10cm まで")

    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(a.ip)
    p0 = r.getActualTCPPose()
    side = [p0[0] + a.dx / 1000, p0[1] + a.dy / 1000, p0[2]] + list(p0[3:])
    low = side[:2] + [(TABLE_Z + a.hold + 1.0) / 1000] + list(p0[3:])
    print(f"今: 机から {p0[2]*1000 - TABLE_Z:.0f}mm → 横へ ({a.dx:+.0f}, {a.dy:+.0f})mm → 指先を {a.hold + 1:.0f}mm まで下ろして離す → 戻る")
    if not a.go:
        print("--go が無いので、ここまで（何も送っていません）")
        return
    c = RTDEControlInterface(a.ip)
    try:
        c.moveL(side, SPEED, ACCEL)
        c.moveL(low, DOWN_SPEED, ACCEL)
        print(f"1) 置く高さ: 机から {r.getActualTCPPose()[2]*1000 - TABLE_Z:.1f}mm")
        h = HandClient()                  # hand_server.py を通す（持ったままリセットされない）
        print("2) 開いた: 開き量", round(h.open(), 1), "mm")
        h.close_port()
        c.moveL(side, SPEED, ACCEL)
        print(f"3) 上げた: 机から {r.getActualTCPPose()[2]*1000 - TABLE_Z:.1f}mm")
    finally:
        c.stopScript()
        c.disconnect()
        r.disconnect()


if __name__ == "__main__":
    main()
