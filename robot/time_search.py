"""物を探す時間（見張りの位置へ行く → 3 枚撮る → 測る）を、シミュレーターで段階ごとに測る（窓は出さない）。
実機用の height_by_motion.capture_two と pick_any.survey をそのまま使う。腕は実機と同じ速さ・加速度で動かし、
その時間はシミュレーションの中の時間で数える。撮る前の待ち（locate.grab_now の settle）もシミュレーションの時間で数える。
使い方:  python robot/time_search.py [回数]
"""
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_real as T  # noqa: E402
import sim_real as S  # noqa: E402
import sim_rtde  # noqa: E402
from sim_rtde import SimRobot  # noqa: E402

import height_by_motion as HM  # noqa: E402
import pick_any as PA  # noqa: E402

SETTLE = 0.5                 # 撮る前に揺れが収まるのを待つ時間（locate.grab_now と同じ）


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    T.patch_real_code()
    env = S.RealLikeEnv()
    robot = SimRobot(env)
    HM.TABLE_Z = S.MAT_Z * 1000.0
    shots = {"n": 0, "wait": 0.0}

    def grab_now(**_):
        t0 = env.d.time
        while env.d.time - t0 < SETTLE:
            sim_rtde.frame(env)
        shots["n"] += 1
        shots["wait"] += env.d.time - t0
        return env.render_real()
    HM.L.grab_now = grab_now
    HM.L.archive = lambda *a, **k: None
    rng = np.random.default_rng(4)
    obs = np.array(S.OBS_XY)
    rows = []
    for k in range(n):
        env.reset(rng)
        for i, o in enumerate(env.objects):
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        T.go_to(env, robot.ik, obs)
        T.place_tools(env, rng)
        # つかんだ後の位置（物の真上 100mm）から探し始める、とする
        p = robot.getActualTCPPose()
        T.go_to(env, robot.ik, obs + np.array([0.05, 0.03]))
        env.d.qpos[:6] = robot.ik.solve(env.d.qpos[:6].copy(), np.array([obs[0] + 0.05, obs[1] + 0.03, S.MAT_Z + 0.10]),
                                        sim_rtde.rotvec_to_R(p[3:]), iters=60)
        env.d.ctrl[:6] = env.d.qpos[:6]
        mujoco.mj_forward(env.m, env.d)
        shots.update(n=0, wait=0.0)
        t_sim0, t_wall0 = env.d.time, time.perf_counter()
        robot.moveL([obs[0], obs[1], S.MAT_Z + HM.OBS / 1000] + list(p[3:]), HM.SPEED, HM.ACCEL)   # 見張りの位置へ
        t_go = env.d.time - t_sim0
        t1 = env.d.time
        imgA, tcpA, others = HM.capture_two(None, c=robot, r=robot)
        t_cap = env.d.time - t1
        w0 = time.perf_counter()
        found = PA.survey(imgA, tcpA, others)
        t_calc = time.perf_counter() - w0
        rows.append((t_go, t_cap - shots["wait"], shots["wait"], t_calc))
        print(f"{k}: 見張りへ {t_go:.1f} 秒、撮る時の動き {t_cap - shots['wait']:.1f} 秒、撮る前の待ち {shots['wait']:.1f} 秒（{shots['n']} 枚）、"
              f"計算 {t_calc:.1f} 秒 → 合計 {t_go + t_cap + t_calc:.1f} 秒（物 {len(found)} 個）")
    a = np.array(rows).mean(0)
    print(f"---- 平均：見張りへ {a[0]:.1f}、撮る時の動き {a[1]:.1f}、撮る前の待ち {a[2]:.1f}、計算 {a[3]:.1f} → 合計 {a.sum():.1f} 秒")


if __name__ == "__main__":
    main()
