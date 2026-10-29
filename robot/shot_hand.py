"""実機に近いシミュレーターのハンドとカメラ台を、横から撮って画像にする（窓は出さない）。手首カメラの画像も保存する。
使い方:  python robot/shot_hand.py   → robot/logs/hand_side.png、hand_wrist.jpg
"""
import os
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_real as S  # noqa: E402
import test_sim_real as T  # noqa: E402
from run_pick import IK  # noqa: E402


def main():
    T.patch_real_code()
    env = S.RealLikeEnv()
    ik = IK(env)
    obs = np.array(S.OBS_XY)
    T.go_to(env, ik, obs)
    vc = T.view_center(env)
    for i, t in enumerate(S.TOOLS):
        env.set_tool(t, vc + np.array([-0.05 + 0.05 * i, 0.0]), 0.3 * i)
    mujoco.mj_forward(env.m, env.d)
    r = mujoco.Renderer(env.m, 720, 960)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = env.d.site_xpos[env.site] + np.array([0, 0, 0.07])
    out = []
    for az in (0, 90):
        cam.distance, cam.elevation, cam.azimuth = 0.40, -10, az
        r.update_scene(env.d, cam)
        out.append(cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(HERE, "logs", "hand_side.png"), np.hstack(out))
    cv2.imwrite(os.path.join(HERE, "logs", "hand_wrist.jpg"), cv2.resize(env.render_real(), (960, 540)))
    print("gap", round(env.pad_gap() * 1000, 1), "mm")


if __name__ == "__main__":
    main()
