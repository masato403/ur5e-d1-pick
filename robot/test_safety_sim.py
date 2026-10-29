r"""安全の決まり（10N で止まって上がる。PB_SAFE_N・TR_SAFE2）を、判断の係を使わずにシミュレーターで確かめる（10/23）。
場面 A：10/23 実機 7 回目のまね。ペンの横 OFF_MM の所を「物の真ん中」としてつかみに行き、指がペンの上に下りる。
場面 B：ペン・消しゴムを真ん中でつかんで箱に入れる（ふつうの時に、安全の決まりが勝手に止めないか）。
ハンドが受ける力（向きを入れて足したもの）の最大を数える。
使い方:  set SIM_FORCE=net & set PB_SAFE_N=10 & .venv\Scripts\python.exe robot\test_safety_sim.py   （PB_SAFE_N=0 で前と比べる）"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_grasp as G  # noqa: E402
import test_sim_real as T  # noqa: E402
import sim_real as S  # noqa: E402
import sim_rtde  # noqa: E402
from sim_rtde import SimHand, SimRobot  # noqa: E402
import pick_by_camera as PB  # noqa: E402

OFF_MM = float(os.environ.get("SAFE_OFF_MM", 9.0))
ONLY = tuple(os.environ.get("SAFE_ONLY", "").split(",")) if os.environ.get("SAFE_ONLY") else None   # 流す場面の頭の文字（例 A2,B）


def main():
    T.patch_real_code()
    env = S.RealLikeEnv()
    G.patch_grasp(env)
    robot = SimRobot(env)
    peak = [0.0]
    where = [""]
    base_frame = sim_rtde.frame

    def frame(env_):
        base_frame(env_)
        fn = float(np.linalg.norm(robot.net_force()))
        if fn > peak[0]:
            peak[0] = fn
            tcp = robot.getActualTCPPose()
            what = "、".join(f"{b_} {f_:.0f}" for f_, a_, b_ in robot.hand_force(detail=True)[1][:2])
            where[0] = f"指先は机から {(tcp[2] - S.MAT_Z) * 1000:.1f}mm、開き量 {SimHand(env).pos_mm():.1f}mm、当たった物 {what}"
    sim_rtde.frame = frame
    box = [S.BOX_XY[0] * 1000, S.BOX_XY[1] * 1000]
    rows = []
    for name, tool, off, grip, place in (("A ペンの横 9mm に下りる（7 回目のまね）", "r_pen", OFF_MM, 6.0, None),
                                         ("A2 消しゴムの横 14mm に下りる（指が消しゴムの上に下りる。転がらない）", "eraser", 14.0, 3.0, None),
                                         ("B ペンを真ん中でつかんで箱へ", "r_pen", 0.0, 4.0, box),
                                         ("B 消しゴムを真ん中でつかんで箱へ", "eraser", 0.0, 6.0, box)):
        if ONLY and not name.startswith(ONLY):
            continue
        env.reset(np.random.default_rng(0))
        for i, o in enumerate(env.objects):
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        for j, t in enumerate(S.TOOLS):
            env.set_tool(t, np.array([1.5 + 0.1 * j, -0.5]), 0.0)
        xy = np.array([S.MAT_CENTER0[0] + 0.05, S.MAT_CENTER0[1]])
        env.set_tool(tool, xy, 0.0)                              # 長い向きが x
        G.go_to(env, robot.ik, xy)
        mujoco.mj_forward(env.m, env.d)
        for _ in range(20):
            sim_rtde.frame(env)
        PB.LAST["safe"] = None
        peak[0] = 0.0
        where[0] = ""
        center = [xy[0] * 1000, xy[1] * 1000 + off, 10.0]
        q0 = env.d.qpos[env.tool_q[tool]:env.tool_q[tool] + 3].copy()
        try:
            ok = PB.grasp(robot, robot, robot.getActualTCPPose(), center, [1.0, 0.0], grip, 60, place, False, miss="retry",
                          place_far=PB.BOX_FAR if place else None, place_h=40.0 if place else None)
            how = "つかんだ" if ok else "やめた"
        except SystemExit as e:
            how = f"止まった（{e}）"
        moved = np.linalg.norm(env.d.qpos[env.tool_q[tool]:env.tool_q[tool] + 2] - q0[:2]) * 1000
        inside = G.in_box(env, tool)[0]
        rows.append((name, how, PB.LAST["safe"], peak[0], moved, inside))
        print(f"{name}：{how}、安全で止めた {PB.LAST['safe'] and round(PB.LAST['safe'], 1)}N、力の最大 {peak[0]:.1f}N、"
              f"物が動いた {moved:.0f}mm、箱の中 {'○' if inside else '×'}\n    いちばん強かった時：{where[0]}", flush=True)


if __name__ == "__main__":
    main()
