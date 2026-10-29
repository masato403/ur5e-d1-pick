"""plan_hover.json の計画を、実機でゆっくり実行する（moveL：道具の先がまっすぐ動く）。

使い方:  .venv_ur\\Scripts\\python.exe real_ur\\ur_run_plan.py --ip 192.0.2.10 --go
  --go を付けない時は、確認だけで何も送らない。
  安全のため：
    ・今の位置が計画の出発点から 3mm 以上（関節 2° 以上）ずれていたら止める
    ・道具の先の速さ 50mm/s、加速度 200mm/s^2
    ・ロボットが「動ける状態」・安全状態「正常」・遠隔操作モードでなければ止める
"""
import argparse
import json
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPEED, ACCEL = 0.05, 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--plan", default="plan_hover.json")
    ap.add_argument("--pause", type=float, default=0.0, help="1) に着いた後に止まる秒数")
    ap.add_argument("--speed", type=float, default=SPEED * 1000, help="道具の先の速さ mm/s（50 まで）")
    a = ap.parse_args()
    plan = json.load(open(os.path.join(HERE, a.plan)))
    speed = min(a.speed, 50.0) / 1000

    from dashboard_client import DashboardClient
    from rtde_receive import RTDEReceiveInterface
    db = DashboardClient(a.ip)
    db.connect()
    mode, safety, remote = db.robotmode(), db.safetystatus(), db.isInRemoteControl()
    db.disconnect()
    print("ロボットの状態:", mode, "／ 安全状態:", safety, "／ 遠隔操作モード:", remote)
    r = RTDEReceiveInterface(a.ip)
    p, q = r.getActualTCPPose(), r.getActualQ()
    dpos = math.dist(p[:3], plan["start_pose"][:3]) * 1000
    dq = max(abs(math.degrees(x - y)) for x, y in zip(q, plan["start_q"]))
    print(f"計画の出発点とのずれ：位置 {dpos:.1f} mm、関節 {dq:.1f}°")
    for name, pp in (("1)", plan["pA"]), ("2)", plan["pB"])):
        print(f"  {name}: x {pp[0]*1000:.1f}  y {pp[1]*1000:.1f}  z {pp[2]*1000:.1f} mm")
    if dpos > 3 or dq > 2:
        raise SystemExit("今の位置が計画の出発点と違うので止めます（計画を作り直してください）")
    if "RUNNING" not in mode.upper() or "NORMAL" not in safety.upper() or not remote:
        raise SystemExit("動ける状態ではないので止めます")
    if not a.go:
        print("--go が無いので、ここまで（何も送っていません）")
        return

    from rtde_control import RTDEControlInterface
    c = RTDEControlInterface(a.ip)
    try:
        for name, pp in (("1)", plan["pA"]), ("2)", plan["pB"])):
            c.moveL(pp, speed, ACCEL)
            now = r.getActualTCPPose()
            print(f"  {name} 着いた: x {now[0]*1000:.1f}  y {now[1]*1000:.1f}  z {now[2]*1000:.1f} mm ／ 関節",
                  [round(math.degrees(v), 1) for v in r.getActualQ()])
            if name == "1)" and a.pause > 0:
                time.sleep(a.pause)                    # 1) の位置で止まって、目で確かめる時間
    finally:
        c.stopScript()
        c.disconnect()
        r.disconnect()


if __name__ == "__main__":
    main()
