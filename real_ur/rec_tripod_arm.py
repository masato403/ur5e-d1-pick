r"""腕をマットの上の 3×3 か所（指先は机から 150mm、手首は回さない、0.05m/秒）へ動かし、三脚と手首の写真・腕の姿勢・関節角を残す（10/20）。
後で、三脚の写真の中のロボットの所（シミュレーターで描く腕の形）や手首の位置の比べに使う。ハンドは動かさない。
使い方:  .venv_ur\Scripts\python.exe real_ur\rec_tripod_arm.py --ip 192.0.2.10"""
import argparse, json, os, time
import cv2
from plan_hover import TABLE_Z
from calib_tripod import fresh_image
from locate import grab_now

HERE = os.path.dirname(os.path.abspath(__file__))
XS, YS, H = (20.0, 120.0, 220.0), (-350.0, -430.0, -510.0), 150.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ip", required=True); a = ap.parse_args()
    from rtde_receive import RTDEReceiveInterface
    from rtde_control import RTDEControlInterface
    r = RTDEReceiveInterface(a.ip)
    p0 = r.getActualTCPPose()
    out = os.path.join(HERE, "logs_tripod_calib", time.strftime("arm_%Y%m%d_%H%M"))
    os.makedirs(out, exist_ok=True)
    c = RTDEControlInterface(a.ip)
    rows = []
    try:
        for k, (x, y) in enumerate([(x, y) for y in YS for x in XS]):
            goal = [x / 1000, y / 1000, (TABLE_Z + H) / 1000] + list(p0[3:])
            if not c.isPoseWithinSafetyLimits(goal):
                print(f"  {k + 1}: 安全設定の範囲の外なので飛ばす", flush=True); continue
            c.moveL(goal, 0.05, 0.2)
            if r.getSafetyMode() != 1:
                print("  安全状態が普通でないので止めます", flush=True); break
            w = grab_now(settle=0.3); t = fresh_image(time.time())
            if w is not None: cv2.imwrite(os.path.join(out, f"w{k + 1:02d}.jpg"), w)
            if t is not None: cv2.imwrite(os.path.join(out, f"t{k + 1:02d}.jpg"), t)
            rows.append({"k": k + 1, "tcp": list(r.getActualTCPPose()), "q": list(r.getActualQ())})
            print(f"  {k + 1}/9: x {x:.0f} y {y:.0f}", flush=True)
        c.moveL(list(p0), 0.05, 0.2)
    finally:
        c.stopScript(); c.disconnect()
    json.dump({"table_z": TABLE_Z, "rows": rows}, open(os.path.join(out, "poses.json"), "w"), indent=1)
    print("記録:", out)


if __name__ == "__main__":
    main()
