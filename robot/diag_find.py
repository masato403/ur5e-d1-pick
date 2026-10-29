"""test_sim_grasp.py と同じ場面で、見張りの位置の画像に、見つけた物の輪郭を描いて保存する（窓は出さない）。
使い方:  python robot/diag_find.py [回数]   → robot/logs/find_<番号>.jpg
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
from sim_rtde import SimRobot  # noqa: E402

import measure as M  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    T.patch_real_code()
    env = S.RealLikeEnv()
    robot = SimRobot(env)
    rng = np.random.default_rng(2)
    for k in range(n):
        env.reset(rng)
        for i, o in enumerate(env.objects):
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        obs = np.array(S.OBS_XY)
        T.go_to(env, robot.ik, obs)
        placed = T.place_tools(env, rng)
        mujoco.mj_forward(env.m, env.d)
        for _ in range(200):
            mujoco.mj_step(env.m, env.d)
        T.go_to(env, robot.ik, obs)
        img = env.render_real()
        objs, _ = M.find_objects(img)
        dbg = img.copy()
        cv2.drawContours(dbg, objs, -1, (0, 255, 0), 3)
        for t, (xy, _) in placed.items():
            p = M.project_table(np.array([[xy[0] * 1000, xy[1] * 1000, S.MAT_Z * 1000]]), env.tcp_like())[0]
            cv2.putText(dbg, T.NAME[t] and t, (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.imwrite(os.path.join(HERE, "logs", f"find_{k}.jpg"), cv2.resize(dbg, (960, 540)))
        print(f"{k}: 見つけた塊 {len(objs)} 個")


if __name__ == "__main__":
    main()
