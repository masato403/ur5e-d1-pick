"""高さの測り方の比べ（シミュレーション、窓は出さない）。道具を 1 つずつ画面の真ん中あたりに置き、
物全体・区切りごと・輪郭の真ん中の帯（いちばん高い所の近く）で測った高さを、本当の高さと並べる。
使い方:  python robot/diag_height.py [回数]
"""
import math
import os
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_real as T  # noqa: E402
import sim_real as S  # noqa: E402
from run_pick import IK  # noqa: E402

import height_by_motion as HM  # noqa: E402
import measure as M  # noqa: E402
import pick_any as PA  # noqa: E402


def band(mask, frac):
    """輪郭の内側からの距離が、いちばん奥の frac 倍より大きい所（細長い物では、長い向きに沿った真ん中の帯）。"""
    d = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return ((d > d.max() * frac) * 255).astype(np.uint8)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    T.patch_real_code()
    env = S.RealLikeEnv()
    ik = IK(env)
    rng = np.random.default_rng(5)
    for k in range(n):
        env.reset(rng)
        for i, o in enumerate(env.objects):
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        obs = np.array(S.OBS_XY)
        T.go_to(env, ik, obs)
        vc = T.view_center(env)
        for t in S.TOOLS:
            for j, u in enumerate(S.TOOLS):
                env.set_tool(u, np.array([1.5, 0.5 + 0.1 * j]), 0.0)
            xy = vc + rng.uniform([-0.04, -0.03], [0.04, 0.03])
            env.set_tool(t, xy, rng.uniform(0, math.pi))
            mujoco.mj_forward(env.m, env.d)
            imgA, tcpA, others = T.shots(env, ik, obs)
            objs, _ = M.find_objects(imgA)
            if not objs:
                print(f"{k}: {T.NAME[t]} 見つからない")
                continue
            cnt = max(objs, key=cv2.contourArea)
            mask = np.zeros(imgA.shape[:2], np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            row = []
            for name, m in [("全体", mask), ("帯0.4", band(mask, 0.4)), ("帯0.6", band(mask, 0.6)), ("帯0.8", band(mask, 0.8))]:
                best = max((HM.best_height(imgA, tcpA, b, tb, m) for b, tb in others), key=lambda r: r[2].max())
                row.append(f"{name} {best[0]:.1f}({best[2].max():.2f})")
            b, tb = max(others, key=lambda o: HM.best_height(imgA, tcpA, o[0], o[1], mask)[2].max())
            parts = PA.part_heights(imgA, tcpA, b, tb, mask, cnt)
            print(f"{k}: {T.NAME[t]}（本当 {T.TRUE[t][0]:.0f}）  " + "  ".join(row)
                  + "  区切り " + " ".join(p[3] if p[0] is None else f"{p[0]:.1f}" for p in parts))


if __name__ == "__main__":
    main()
