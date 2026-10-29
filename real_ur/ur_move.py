"""UR5e 実機を、決めた関節の角度へゆっくり動かす（moveJ）。

使い方:  .venv_ur\\Scripts\\python.exe real_ur\\ur_move.py --ip 192.0.2.10 --expect 0,-90,0,-90,0,0 --to 0,-90,90,-90,-90,0 --go
  --go を付けない時は、確認と予定の表示だけで、ロボットには何も送らない。
  安全のため：
    ・今の角度が --expect から 2° 以上ずれていたら止める（窓で見せた動きと違う動きになるため）
    ・1 回で動かせるのは、どの関節も 120° まで
    ・速さ 0.2 rad/s（約 11°/秒）、加速度 0.3 rad/s^2
    ・ロボットが「動ける状態」・安全状態「正常」・遠隔操作モードでなければ止める
"""
import argparse
import math

SPEED, ACCEL = 0.2, 0.3
MAX_STEP_DEG = 120.0
EXPECT_TOL_DEG = 2.0


def degs(s):
    v = [float(x) for x in s.split(",")]
    if len(v) != 6:
        raise SystemExit("角度は 6 個（度、カンマ区切り）")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--expect", required=True, help="今の角度（度）。窓で見せた出発点")
    ap.add_argument("--to", required=True, help="行き先の角度（度）")
    ap.add_argument("--go", action="store_true", help="付けた時だけ本当に動かす")
    a = ap.parse_args()
    exp, to = degs(a.expect), degs(a.to)

    from dashboard_client import DashboardClient
    from rtde_receive import RTDEReceiveInterface
    db = DashboardClient(a.ip)
    db.connect()
    mode, safety, remote = db.robotmode(), db.safetystatus(), db.isInRemoteControl()
    db.disconnect()
    print("ロボットの状態:", mode, "／ 安全状態:", safety, "／ 遠隔操作モード:", remote)
    r = RTDEReceiveInterface(a.ip)
    now = [math.degrees(v) for v in r.getActualQ()]
    print("いまの角度 [度]:", [round(v, 1) for v in now])
    print("行き先     [度]:", to)
    off = max(abs(n - e) for n, e in zip(now, exp))
    step = max(abs(t - n) for t, n in zip(to, now))
    print(f"出発点とのずれ {off:.1f}°、いちばん大きく動く関節 {step:.1f}°（約 {step / math.degrees(SPEED):.0f} 秒）")
    if off > EXPECT_TOL_DEG:
        raise SystemExit("今の角度が、窓で見せた出発点と違うので止めます")
    if step > MAX_STEP_DEG:
        raise SystemExit(f"1 回の動きが {MAX_STEP_DEG}° を超えるので止めます")
    if "RUNNING" not in mode.upper() or "NORMAL" not in safety.upper() or not remote:
        raise SystemExit("動ける状態ではないので止めます")
    if not a.go:
        print("--go が無いので、ここまで（何も送っていません）")
        return

    from rtde_control import RTDEControlInterface
    c = RTDEControlInterface(a.ip)
    try:
        c.moveJ([math.radians(v) for v in to], SPEED, ACCEL)
        print("着いた後 [度]:", [round(math.degrees(v), 1) for v in r.getActualQ()])
    finally:
        c.stopScript()
        c.disconnect()
        r.disconnect()


if __name__ == "__main__":
    main()
