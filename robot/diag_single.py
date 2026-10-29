"""道具を 1 つずつ置いて、実機用のプログラム（pick_any.survey）で測った高さ・幅の誤差を出す（窓は出さない）。
ほかの物と重なる影響を除いて、測り方そのものの誤差を見る用。
使い方:  python robot/diag_single.py [道具ごとの回数]
"""
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_real as T  # noqa: E402
import sim_real as S  # noqa: E402
from run_pick import IK  # noqa: E402

import pick_any as PA  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    T.patch_real_code()
    env = S.RealLikeEnv()
    ik = IK(env)
    rng = np.random.default_rng(21)
    obs = np.array(S.OBS_XY)
    for i, o in enumerate(env.objects):
        env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
    T.go_to(env, ik, obs)
    for t in S.TOOLS:
        errs, miss = [], 0
        for k in range(n):
            for j, u in enumerate(S.TOOLS):
                env.set_tool(u, np.array([1.5, 0.5 + 0.1 * j]), 0.0)
            for _ in range(2000):
                xy = T.view_center(env) + rng.uniform([-0.09, -0.07], [0.09, 0.07])
                yaw = rng.uniform(0, math.pi)
                if T.in_view(env, t, xy, yaw) and T.away_from_box(t, xy, yaw):
                    break
            env.set_tool(t, xy, yaw)
            mujoco.mj_forward(env.m, env.d)
            imgA, tcpA, others = T.shots(env, ik, obs)
            found = [f for f in PA.survey(imgA, tcpA, others) if np.linalg.norm(f["center"][:2] - xy * 1000) < 40]
            if not found:
                miss += 1
                continue
            f = found[0]
            th, tw, _ = T.TRUE[t]
            errs.append((f["h"] - th, f["size"][0] - tw))
        e = np.array(errs)
        print(f"{T.NAME[t]}：高さの誤差 平均 {e[:, 0].mean():+.1f}mm（{' '.join(f'{v:+.1f}' for v in e[:, 0])}）、"
              f"幅の誤差 平均 {e[:, 1].mean():+.1f}mm、見つからない {miss} 回")


if __name__ == "__main__":
    main()
