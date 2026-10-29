"""UR5e 実機との接続確認（読むだけ。ロボットには動きを一切送らない）。

使い方:  .venv_ur\\Scripts\\python.exe real_ur\\ur_check.py --ip 192.0.2.10
  1) 主なポートにつながるか（29999 ダッシュボード、30004 RTDE、30002 スクリプト受付）
  2) ダッシュボードから、ロボットの状態・安全状態・PolyScope の版・遠隔操作モードかを読む
  3) RTDE で、いまの関節の角度とツール先端の位置を読む
"""
import argparse
import math
import socket


def port_open(ip, port, timeout=2.0):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    a = ap.parse_args()

    print(f"== 1) ポートの確認 ({a.ip})")
    ports = {29999: "ダッシュボード", 30004: "RTDE", 30002: "スクリプト受付"}
    ok = {p: port_open(a.ip, p) for p in ports}
    for p, name in ports.items():
        print(f"  {p} {name}: {'つながる' if ok[p] else 'つながらない'}")
    if not any(ok.values()):
        print("  → どれにもつながらない：LAN ケーブル、PC とロボットの IP の組（同じ 192.168.x.* か）、ロボットの電源を確認")
        return

    if ok[29999]:
        from dashboard_client import DashboardClient
        print("== 2) ダッシュボード")
        db = DashboardClient(a.ip)
        db.connect()
        for label, f in (("PolyScope の版", db.polyscopeVersion), ("ロボットの状態", db.robotmode),
                         ("安全状態", db.safetystatus), ("遠隔操作モード", db.isInRemoteControl),
                         ("読み込み中のプログラム", db.getLoadedProgram)):
            try:
                print(f"  {label}: {f()}")
            except Exception as e:
                print(f"  {label}: 読めない ({e})")
        db.disconnect()

    if ok[30004]:
        from rtde_receive import RTDEReceiveInterface
        print("== 3) RTDE で読む")
        r = RTDEReceiveInterface(a.ip)
        q = r.getActualQ()
        tcp = r.getActualTCPPose()
        print("  関節の角度 [度]:", [round(math.degrees(v), 1) for v in q])
        print("  ツール先端 [mm, 度]:", [round(v * 1000, 1) for v in tcp[:3]], [round(math.degrees(v), 1) for v in tcp[3:]])
        print("  速度の倍率（ペンダントのスライダー）:", r.getSpeedScaling())
        r.disconnect()


if __name__ == "__main__":
    main()
