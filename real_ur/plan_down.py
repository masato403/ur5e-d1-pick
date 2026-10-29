"""今の位置から、向きはそのままで机の CLEAR mm 上までまっすぐ下ろし、また今の高さへ戻す計画を作る（動かさない）。
計画の形は plan_hover.json と同じ（pA = 下ろした所、pB = 戻った所）なので、show_plan.py と ur_run_plan.py でそのまま使える。
使い方:  .venv_ur\\Scripts\\python.exe real_ur\\plan_down.py --ip 192.0.2.10 --clear 20
"""
import argparse
import json
import math
import os

import numpy as np

from plan_hover import HERE, TABLE_Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--clear", type=float, default=20.0, help="下ろした時の、机の面からの高さ mm（10 未満は受け付けない）")
    ap.add_argument("--out", default="plan_down.json")
    ap.add_argument("--stay", action="store_true", help="下ろしたまま止める（戻さない）")
    a = ap.parse_args()
    if a.clear < 10:
        raise SystemExit("机に近すぎるので、10mm 以上にしてください")

    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(a.ip)
    p0, q0 = r.getActualTCPPose(), r.getActualQ()
    pA = [p0[0], p0[1], (TABLE_Z + a.clear) / 1000] + list(p0[3:])
    pB = list(pA) if a.stay else list(p0)
    c = RTDEControlInterface(a.ip)          # 関節の角度の計算だけ頼む（動きは送らない）
    try:
        steps, q_prev = [], q0
        for seg, (ps, pe) in enumerate(((p0, pA), (pA, pB))):
            for t in np.linspace(0, 1, 11)[1:]:
                pose = list((1 - t) * np.array(ps[:3]) + t * np.array(pe[:3])) + list(p0[3:])
                if not c.isPoseWithinSafetyLimits(pose):
                    raise SystemExit(f"区間 {seg + 1} の途中が、安全設定の範囲の外です")
                q = c.getInverseKinematics(pose, q_prev)
                if max(abs(x - y) for x, y in zip(q, q_prev)) > math.radians(30):
                    raise SystemExit(f"区間 {seg + 1} の途中で、関節が急に大きく回るので止めます")
                steps.append({"seg": seg + 1, "pose": [float(v) for v in pose], "q": [float(v) for v in q]})
                q_prev = q
    finally:
        c.stopScript()
        c.disconnect()
        r.disconnect()
    json.dump({"start_pose": list(p0), "start_q": list(q0), "pA": pA, "pB": pB, "steps": steps},
              open(os.path.join(HERE, a.out), "w"), indent=1)
    f = lambda p: f"x {p[0]*1000:.1f}  y {p[1]*1000:.1f}  z {p[2]*1000:.1f} mm（机から {p[2]*1000 - TABLE_Z:.0f} mm）"
    print("今:          ", f(p0))
    print("1) 下ろす:   ", f(pA), " 関節", [round(math.degrees(v), 1) for v in steps[9]["q"]])
    print("2) 戻す:     ", f(pB))


if __name__ == "__main__":
    main()
