"""今いる場所の真下にある物を、つかんで持ち上げる（実機の通しの試験）。

  1) ハンドを開く  2) 向きはそのままで、指先が机から GRIP mm の高さまでまっすぐ下ろす（毎秒 2cm）
  3) 力 FORCE% で閉じる → 止まった開き量でつかめたかを判断  4) 机から 10cm の高さまで持ち上げる
  机の面は ur_touch.py で測った高さ（TCP_2 の z = -62.5mm）。
使い方:  .venv_ur\\Scripts\\python.exe real_ur\\pick_here.py --ip 192.0.2.10 --grip 4 --force 30 --go
"""
import argparse
import math

from hand import HandClient
from plan_hover import TABLE_Z

DOWN_SPEED, UP_SPEED, ACCEL = 0.02, 0.03, 0.2
HOLD_MM = 2.0            # 閉じた後の開き量がこれより広ければ、物を挟んでいる


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--grip", type=float, default=4.0, help="つかむ時の指先の高さ（机の面から mm、3 以上）")
    ap.add_argument("--force", type=int, default=30)
    ap.add_argument("--lift", type=float, default=100.0, help="持ち上げた後の高さ（机の面から mm）")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    if a.grip < 3:
        raise SystemExit("机に近すぎるので、3mm 以上にしてください")

    from dashboard_client import DashboardClient
    from rtde_receive import RTDEReceiveInterface
    db = DashboardClient(a.ip)
    db.connect()
    mode, safety, remote = db.robotmode(), db.safetystatus(), db.isInRemoteControl()
    db.disconnect()
    r = RTDEReceiveInterface(a.ip)
    p0 = r.getActualTCPPose()
    down = [p0[0], p0[1], (TABLE_Z + a.grip) / 1000] + list(p0[3:])
    up = [p0[0], p0[1], (TABLE_Z + a.lift) / 1000] + list(p0[3:])
    print("ロボット:", mode, "／", safety, "／ 遠隔操作:", remote)
    print(f"今: 机から {p0[2]*1000 - TABLE_Z:.0f}mm → 下ろす: {a.grip:.0f}mm → 力 {a.force}% で閉じる → 持ち上げる: {a.lift:.0f}mm")
    if "RUNNING" not in mode.upper() or "NORMAL" not in safety.upper() or not remote:
        raise SystemExit("動ける状態ではないので止めます")
    if p0[2] * 1000 - TABLE_Z < a.grip:
        raise SystemExit("今の位置が、つかむ高さより低いので止めます")
    if not a.go:
        print("--go が無いので、ここまで（何も送っていません）")
        return

    from rtde_control import RTDEControlInterface
    h = HandClient()                      # hand_server.py を通す（USB を開け直さないのでリセットされない）
    c = RTDEControlInterface(a.ip)
    try:
        print("1) 開く:", round(h.open(), 1), "mm")
        c.moveL(down, DOWN_SPEED, ACCEL)
        print(f"2) 下ろした: 机から {r.getActualTCPPose()[2]*1000 - TABLE_Z:.1f}mm")
        w = h.grip(a.force)
        held = w is not None and w > HOLD_MM
        print(f"3) 閉じた: 開き量 {w:.1f}mm →", "つかめた" if held else "つかめていない（空振り）")
        c.moveL(up, UP_SPEED, ACCEL)
        w2 = h.pos_mm()
        print(f"4) 持ち上げた: 机から {r.getActualTCPPose()[2]*1000 - TABLE_Z:.1f}mm、開き量 {w2:.1f}mm →",
              "持てている" if w2 and w2 > HOLD_MM else "持てていない")
    finally:
        c.stopScript()
        c.disconnect()
        r.disconnect()
        h.close_port()


if __name__ == "__main__":
    main()
