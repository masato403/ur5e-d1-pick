"""物を探して測る計算（pick_any.survey）に、どれだけ時間がかかるかを測る（シミュレーション、窓は出さない）。
使い方:  python robot/time_survey.py [回数]
"""
import math
import os
import sys
import time

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


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    T.patch_real_code()
    env = S.RealLikeEnv()
    ik = IK(env)
    rng = np.random.default_rng(3)
    # 中の部品ごとの時間も測る
    tt = {"find_objects": 0.0, "best_height": 0.0, "part_heights": 0.0}
    for name, mod in (("find_objects", M), ("best_height", HM), ("part_heights", PA)):
        f = getattr(mod, name)

        def wrap(*a, _f=f, _n=name, **k):
            t0 = time.perf_counter()
            r = _f(*a, **k)
            tt[_n] += time.perf_counter() - t0
            return r
        setattr(mod, name, wrap)
    total = 0.0
    for k in range(n):
        env.reset(rng)
        for i, o in enumerate(env.objects):
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        obs = np.array(S.OBS_XY)
        T.go_to(env, ik, obs)
        placed = T.place_tools(env, rng)
        mujoco.mj_forward(env.m, env.d)
        imgA, tcpA, others = T.shots(env, ik, obs)
        t0 = time.perf_counter()
        found = PA.survey(imgA, tcpA, others)
        dt = time.perf_counter() - t0
        total += dt
        print(f"{k}: 物 {len(found)} 個、計算 {dt:.1f} 秒")
    print(f"---- 1 回あたり {total / n:.1f} 秒（中の内訳の合計：" + "、".join(f"{a} {b / n:.1f} 秒" for a, b in tt.items()) + "）")


if __name__ == "__main__":
    main()
