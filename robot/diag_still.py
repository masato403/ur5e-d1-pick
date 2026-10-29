"""「止まったら撮る」（HM.STILL）と、前の決まった待ち（0.5 秒）で、測った高さ・中心が本当とどれだけ違うかを比べる（10/19）。
同じ置き方で、両方のやり方で撮って測る。速くするために答えが悪くならないかを確かめる。
使い方:  python robot/diag_still.py [場面の数 5]
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_grasp as G  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import results_table  # noqa: E402
import time  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    G.T.patch_real_code()
    env = G.S.RealLikeEnv()
    G.patch_grasp(env)
    G.patch_capture(env)
    robot = G.SimRobot(env)
    rng = np.random.default_rng(7)
    MODES = [("前の決まった待ち", None), ("止まったら撮る", 0.003)]
    err = {m: {"h": [], "c": []} for m, _ in MODES}
    t_search = {m: [] for m, _ in MODES}
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
        for still, qd in (MODES if k % 2 == 0 else MODES[::-1]):    # 測る順番を場面ごとに入れ替える（順番で差が出ないように）
            G.HM.STILL = qd is not None
            G.HM.STILL_QD = qd or 0.0
            G.SEARCH_S.clear()
            found = G.search(env, robot, obs)[3]
            t_search[still].append(G.SEARCH_S[-1])
            for f in found:
                t = G.nearest_tool(env, f)
                q = env.d.qpos[env.tool_q[t]:env.tool_q[t] + 3]
                err[still]["h"].append(abs(f["h"] - G.T.TRUE[t][0]))
                err[still]["c"].append(float(np.linalg.norm(f["center"][:2] - q[:2] * 1000)))
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "n": len(err[MODES[0][0]]["h"]), "rows": []}
    for name, _ in MODES:
        e = err[name]
        row = [name, float(np.mean(e["h"])), float(np.max(e["h"])), float(np.mean(e["c"])), float(np.max(e["c"])), float(np.mean(t_search[name]))]
        out["rows"].append(row)
        print(f"{name}：高さの差 平均 {row[1]:.2f}・最大 {row[2]:.2f}mm、中心の差 平均 {row[3]:.2f}・最大 {row[4]:.2f}mm、探す時間 平均 {row[5]:.1f} 秒（{len(e['h'])} 個）")
    results_table.save("still", out)


if __name__ == "__main__":
    main()
