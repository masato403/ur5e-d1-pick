"""指で挟むハンドの物理だけを試す（d1-3B は使わない、物の本当の位置と向きを使う）。
  python robot/test_grip_phys.py [つかむ高さ]  → 物ごとに：閉じた指の角度、持ち上げた後の物の高さ
"""
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
os.chdir(os.path.join(HERE, ".."))

from run_grip import rot_yaw, nearest_yaw  # noqa: E402
from run_pick import IK  # noqa: E402
from sim_grip import GripEnv  # noqa: E402
from sim_tools import OBJECTS  # noqa: E402


def obj_yaw(env, o):
    a = env.qadr[o]
    w, x, y, z = env.d.qpos[a + 3:a + 7]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.array([w, x, y, z]))
    ax = R.reshape(3, 3)[:, 0]                    # 物はどれも x 軸が長い向き
    return math.atan2(ax[1], ax[0])


def run(env, ik, o, grip_z, log=False):
    xy = env.obj_pos(o)[:2].copy()
    yaw = obj_yaw(env, o)
    q = env.d.qpos[:6].copy()
    yaw = nearest_yaw(yaw, 0.0)

    def go(pos, n, close=None):
        nonlocal q
        if close is not None:
            env.set_finger(close)
        for _ in range(n):
            q = ik.solve(q, pos, rot_yaw(yaw))
            env.d.ctrl[:6] = q
            for _ in range(17):
                env.d.qfrc_applied[:6] = env.d.qfrc_bias[:6]
                mujoco.mj_step(env.m, env.d)
    for z in np.linspace(0.2, grip_z, 1):
        pass
    # 真上へ瞬間移動（腕の角度を直接置く）してから下ろす
    for _ in range(60):
        q = ik.solve(q, np.array([xy[0], xy[1], 0.12]), rot_yaw(yaw))
    env.d.qpos[:6] = q
    env.d.qvel[:] = 0
    env.set_finger(False)
    mujoco.mj_forward(env.m, env.d)
    for z in np.linspace(0.12, grip_z, 30):
        go(np.array([xy[0], xy[1], z]), 1)
    go(np.array([xy[0], xy[1], grip_z]), 30)
    pin = env.tool_pos()[2]
    go(np.array([xy[0], xy[1], grip_z]), 18, close=True)
    f_closed = env.pad_gap()
    for z in np.linspace(grip_z, 0.15, 30):
        go(np.array([xy[0], xy[1], z]), 1)
    go(np.array([xy[0], xy[1], 0.15]), 10)
    return f_closed, env.pad_gap(), env.obj_pos(o)[2], pin


if __name__ == "__main__":
    gz = float(sys.argv[1]) if len(sys.argv) > 1 else 0.009
    env = GripEnv()
    ik = IK(env)
    ok = {o: 0 for o in OBJECTS}
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    for seed in range(N):
        for o in OBJECTS:
            env.reset(np.random.default_rng(seed))
            mujoco.mj_forward(env.m, env.d)
            fc, fl, oz, pin = run(env, ik, o, gz)
            held = oz > 0.08
            ok[o] += held
            print(f"seed {seed} {o:12s} すき間 閉じた時 {fc*1000:.1f}mm 持ち上げ後 {fl*1000:.1f}mm 物の高さ {oz:.3f} {'○' if held else '×'}  (つかむ点 {pin:.4f})")
    print({o: f"{v}/{N}" for o, v in ok.items()})
