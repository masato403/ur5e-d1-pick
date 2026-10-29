"""UR5e 実機を、1 つの関節だけ少し回して元に戻す（初めて動かす時の確認用）。

使い方:  .venv_ur\\Scripts\\python.exe real_ur\\ur_jog.py --ip 192.0.2.10 --joint 5 --deg 5 --go
  --go を付けない時は、動かす予定を表示するだけで、ロボットには何も送らない。
  安全のため：回す量は最大 10°、速さはとても遅く（0.1 rad/s ≒ 6°/秒）。
  動かす前に、ロボットが「動ける状態」で安全状態が「正常」かを確かめ、違えば止める。
  動かす間は、必ず人が非常停止ボタンのそばにいること。
"""
import argparse
import math
import time

MAX_DEG = 10.0
SPEED, ACCEL = 0.1, 0.2           # rad/s, rad/s^2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--joint", type=int, default=5, help="0=根元 … 5=手首の先（いちばん安全なのは 5）")
    ap.add_argument("--deg", type=float, default=5.0)
    ap.add_argument("--go", action="store_true", help="付けた時だけ本当に動かす")
    a = ap.parse_args()
    if abs(a.deg) > MAX_DEG or not 0 <= a.joint <= 5:
        raise SystemExit(f"回す量は ±{MAX_DEG}° まで、関節は 0〜5")

    from dashboard_client import DashboardClient
    from rtde_receive import RTDEReceiveInterface
    db = DashboardClient(a.ip)
    db.connect()
    mode, safety = db.robotmode(), db.safetystatus()
    print("ロボットの状態:", mode, "／ 安全状態:", safety, "／ 遠隔操作モード:", db.isInRemoteControl())
    db.disconnect()
    r = RTDEReceiveInterface(a.ip)
    q0 = r.getActualQ()
    q1 = list(q0)
    q1[a.joint] += math.radians(a.deg)
    print("いまの角度 [度]:", [round(math.degrees(v), 1) for v in q0])
    print(f"予定：関節 {a.joint} を {a.deg:+.1f}° 回して、元に戻す（速さ {math.degrees(SPEED):.0f}°/秒）")
    if "RUNNING" not in mode.upper() or "NORMAL" not in safety.upper():
        raise SystemExit("動ける状態ではないので止めます（ペンダントで電源オン・ブレーキ解除、保護停止の解除を）")
    if not a.go:
        print("--go が無いので、ここまで（何も送っていません）")
        return

    from rtde_control import RTDEControlInterface
    c = RTDEControlInterface(a.ip)
    try:
        c.moveJ(q1, SPEED, ACCEL)
        time.sleep(0.5)
        print("回した後 [度]:", [round(math.degrees(v), 1) for v in r.getActualQ()])
        c.moveJ(q0, SPEED, ACCEL)
        print("戻した後 [度]:", [round(math.degrees(v), 1) for v in r.getActualQ()])
    finally:
        c.stopScript()
        c.disconnect()
        r.disconnect()


if __name__ == "__main__":
    main()
