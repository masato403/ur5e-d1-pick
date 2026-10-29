"""test_track_real と同じ場面で、三脚で測った物の位置と本当の位置を比べる（10/20 夜の確かめ用）。写真 robot/logs/diag_track_real_<場面>.jpg"""
import os, sys
os.environ.setdefault("SIM_TRIPOD", "1"); os.environ.setdefault("SIM_MAT", "real")
import cv2, mujoco, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import test_track_real as R
import test_sim_grasp as G, test_sim_real as T, sim_real as S, sim_rtde
import tripod_view as TV
from sim_rtde import SimRobot
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
T.patch_real_code(); env = S.RealLikeEnv(); G.patch_grasp(env); G.patch_capture(env); R.patch_tripod(env)
robot = SimRobot(env); rng = np.random.default_rng(2)
for k in range(n):
    env.reset(rng)
    for i, o in enumerate(env.objects): env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
    G.go_to(env, robot.ik, np.array(S.OBS_XY)); mujoco.mj_forward(env.m, env.d); robot.moveL(R.look_pose(robot), 0.1, 0.4)
    [env.set_tool(t, np.array([1.5 + 0.1 * j, -0.5]), 0.0) for j, t in enumerate(S.TOOLS)]; mujoco.mj_forward(env.m, env.d); TV.BG = None; TV.BG = TV.grab(); G.go_to(env, robot.ik, np.array(S.OBS_XY)); T.place_tools(env, rng); mujoco.mj_forward(env.m, env.d)
    for _ in range(20): sim_rtde.frame(env)
    lx, ly = R.TR.LOOK[:2]
    robot.moveL([lx / 1000, ly / 1000, (R.TR.TABLE_Z + R.TR.LOOK[2]) / 1000] + list(robot.getActualTCPPose()[3:]), 0.1, 0.4)
    objs, img = TV.objects(robot.getActualTCPPose())
    truth = {t: env.d.qpos[env.tool_q[t]:env.tool_q[t] + 2] * 1000 for t in S.TOOLS}
    v = img.copy()
    for t, p in truth.items():
        u = TV.to_px([[p[0], p[1], S.MAT_Z * 1000]])[0]; cv2.drawMarker(v, (int(u[0]), int(u[1])), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
    for o in objs:
        t = min(truth, key=lambda t: np.linalg.norm(truth[t] - o["xy"])); d = o["xy"] - truth[t]
        cv2.drawContours(v, [o["contour"]], -1, (0, 255, 255), 1)
        print(k, t, "三脚", o["xy"].round(0), "本当", truth[t].round(0), "ずれ", d.round(1), "長さ画素", round(o["len_px"]))
    print(k, "見つけた数", len(objs), "塊", [(int(c[0]), int(c[1]), int(L)) for c, _, L in TV.blobs(img)])
    fg = (cv2.absdiff(img, TV.BG).max(axis=2) > 40).astype(np.uint8)
    nn, lab, st, cen = cv2.connectedComponentsWithStats(fg)
    print(k, "違う所（100 画素以上）", [(int(cen[i][0]), int(cen[i][1]), int(st[i, 4]), int(st[i, 2]), int(st[i, 3])) for i in range(1, nn) if st[i, 4] >= 100])
    cv2.imwrite(os.path.join(HERE, "logs", f"diag_track_real_{k}_fg.png"), fg * 255)
    cv2.imwrite(os.path.join(HERE, "logs", f"diag_track_real_{k}.jpg"), v)
