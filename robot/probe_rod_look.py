"""シミュレーターの銀の棒の色と光り方の候補を、1 回動かすだけでまとめて比べる（10/19）。
材質の色は、模型を作り直さずに動かしている途中で変えられる（env.m.mat_rgba など）。候補ごとに同じ場面を撮って測り、
「物として見つかるか」と「d1-3B が銀の棒と答えるか」を数える。判断の係（real_ur/d1_server.py）を先に動かしておく。
使い方:  python robot/probe_rod_look.py [場面の数 3]
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_grasp as G  # noqa: E402（場面の作り方・撮って測る所・種類を決める所を使い回す）

# 候補：(灰色の明るさ, specular, shininess, reflectance)。最初が今の色
LOOKS = [(0.74, 1.0, 0.95, 0.2), (0.62, 1.0, 0.95, 0.2), (0.55, 1.0, 0.95, 0.15),
         (0.62, 0.6, 0.6, 0.3), (0.55, 0.8, 0.8, 0.3)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    G.T.patch_real_code()
    env = G.S.RealLikeEnv()
    G.patch_grasp(env)
    G.patch_capture(env)
    robot = G.SimRobot(env)
    mid = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_MATERIAL, "steel")
    score = {lk: [0, 0, 0] for lk in LOOKS}                      # 見つかった・銀の棒と答えた・場面
    rng = np.random.default_rng(2)
    for k in range(n):
        env.reset(rng)
        for i, o in enumerate(env.objects):
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        obs = np.array(G.S.OBS_XY)
        G.go_to(env, robot.ik, obs)
        G.T.place_tools(env, rng)
        mujoco.mj_forward(env.m, env.d)
        for _ in range(20):
            G.sim_rtde.frame(env)
        for lk in LOOKS:
            g, sp, sh, rf = lk
            env.m.mat_rgba[mid] = [g, g + 0.01, g + 0.03, 1]
            env.m.mat_specular[mid], env.m.mat_shininess[mid], env.m.mat_reflectance[mid] = sp, sh, rf
            _, _, _, found = G.search(env, robot, obs)
            rods = [f for f in found if G.nearest_tool(env, f) == "r_rod"]
            ans = [f.get("kind") for f in rods]
            score[lk][0] += bool(rods)
            score[lk][1] += "銀の棒" in ans
            score[lk][2] += 1
            print(f"場面 {k} 色 {lk}：見つかった {len(rods)} 個、d1-3B の答え {ans}", flush=True)
    print("==== 候補ごと（見つかった／銀の棒と答えた／場面）")
    for lk, (a, b, c) in score.items():
        print(f"   {lk}：{a}/{c}、{b}/{c}")


if __name__ == "__main__":
    main()
