"""机の上の点の真上へ動く計画を作る（ロボットは動かさない。関節の角度の計算だけロボットに頼む）。

  1) 今の位置から、向きはそのままで真上へ LIFT mm
  2) 道具を真下に向け直しながら、(x, y) の真上 HOVER mm へ横に動く
  机の面は ur_touch.py で測った高さ（TCP_2 の z）。
  計画は plan_hover.json に保存し、show_plan.py（窓で見せる）と ur_run_plan.py（実機に送る）が読む。
使い方:  .venv_ur\\Scripts\\python.exe real_ur\\plan_hover.py --ip 192.0.2.10 --x 140 --y -430
"""
import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_Z = -62.5          # mm：TCP_2 の z がこの値で指先が机に触れる（3 点の平均、傾き 0.29°）


def rotvec_to_R(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def R_to_rotvec(R):
    """回転行列 → 回転ベクトル。四元数を経由する（180° 近くでも正しく出る）。"""
    t = np.trace(R)
    if t > max(R[0, 0], R[1, 1], R[2, 2]):
        w = math.sqrt(1 + t) / 2
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (4 * w)
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
        v = np.zeros(3)
        v[i] = s / 4
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
        w = (R[k, j] - R[j, k]) / s
    if w < 0:
        w, v = -w, -v
    n = np.linalg.norm(v)
    if n < 1e-12:
        return [0.0, 0.0, 0.0]
    return list(v / n * 2 * math.atan2(n, w))


def down_keep_yaw(R):
    """道具の z 軸を真下に向け、道具の x 軸の水平の向きは今のまま残す。"""
    x = R[:, 0].copy()
    x[2] = 0
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--x", type=float, required=True, help="mm（根元の座標）")
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--lift", type=float, default=100.0, help="今の位置から真上へ上げる量 mm")
    ap.add_argument("--hover", type=float, default=100.0, help="机の面からの高さ mm")
    a = ap.parse_args()

    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(a.ip)
    p0, q0 = r.getActualTCPPose(), r.getActualQ()
    R0 = rotvec_to_R(p0[3:])
    pA = [p0[0], p0[1], p0[2] + a.lift / 1000] + list(p0[3:])
    pB = [a.x / 1000, a.y / 1000, (TABLE_Z + a.hover) / 1000] + R_to_rotvec(down_keep_yaw(R0))

    c = RTDEControlInterface(a.ip)          # 計算だけ頼む（動きは送らない）
    try:
        steps, q_prev = [], q0
        for seg, (ps, pe) in enumerate(((p0, pA), (pA, pB))):
            Rs, Re = rotvec_to_R(ps[3:]), rotvec_to_R(pe[3:])
            for t in np.linspace(0, 1, 11)[1:]:
                pos = (1 - t) * np.array(ps[:3]) + t * np.array(pe[:3])
                dR = Rs.T @ Re                                 # 向きも少しずつ変える
                Rt = Rs @ rotvec_to_R(np.array(R_to_rotvec(dR)) * t)
                pose = list(pos) + R_to_rotvec(Rt)
                if not c.isPoseWithinSafetyLimits(pose):
                    raise SystemExit(f"区間 {seg + 1} の途中が、安全設定の範囲の外です")
                q = c.getInverseKinematics(pose, q_prev)
                if max(abs(a_ - b) for a_, b in zip(q, q_prev)) > math.radians(30):
                    raise SystemExit(f"区間 {seg + 1} の途中で、関節が急に大きく回る（特異姿勢の近く）ので止めます")
                steps.append({"seg": seg + 1, "pose": [float(v) for v in pose], "q": [float(v) for v in q]})
                q_prev = q
    finally:
        c.stopScript()
        c.disconnect()
        r.disconnect()

    plan = {"start_pose": list(p0), "start_q": list(q0), "pA": pA, "pB": pB, "steps": steps}
    json.dump(plan, open(os.path.join(HERE, "plan_hover.json"), "w"), indent=1)
    f = lambda p: f"x {p[0]*1000:.1f}  y {p[1]*1000:.1f}  z {p[2]*1000:.1f} mm（机から {p[2]*1000 - TABLE_Z:.0f} mm）"
    print("今:        ", f(p0), " 関節", [round(math.degrees(v), 1) for v in q0])
    print("1) 真上へ: ", f(pA), " 関節", [round(math.degrees(v), 1) for v in steps[9]["q"]])
    print("2) 行き先: ", f(pB), " 関節", [round(math.degrees(v), 1) for v in steps[-1]["q"]])
    print("関節がいちばん大きく回る量:", round(max(abs(math.degrees(b - a_)) for a_, b in zip(q0, steps[-1]["q"])), 1), "°")


if __name__ == "__main__":
    main()
