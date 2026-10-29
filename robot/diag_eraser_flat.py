"""test_sim_grasp と同じ並べ方の場面 k を作り直し、消しゴムの高さの合い具合（右・左の写真）を調べる（窓は出さない）。
使い方:  python robot/diag_eraser_flat.py [場面の番号]
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_real as T  # noqa: E402
import test_sim_grasp as G  # noqa: E402
import sim_real as S  # noqa: E402
import sim_rtde  # noqa: E402
from sim_rtde import SimRobot  # noqa: E402
import height_by_motion as HM  # noqa: E402
import pick_any as PA  # noqa: E402
import measure as M  # noqa: E402


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    T.patch_real_code()
    env = S.RealLikeEnv()
    G.patch_grasp(env)
    G.patch_capture(env)
    robot = SimRobot(env)
    rng = np.random.default_rng(2)
    obs = np.array(S.OBS_XY)
    for j in range(k + 1):
        env.reset(rng)
        for i, o in enumerate(env.objects):
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        G.go_to(env, robot.ik, obs)
        T.place_tools(env, rng)
    for _ in range(20):
        sim_rtde.frame(env)
    a = env.tool_q["eraser"]
    print("消しゴムの本当の位置 mm", (env.d.qpos[a:a + 3] * 1000).round(1), "向き(四元数)", env.d.qpos[a + 3:a + 7].round(3))
    robot.moveL([obs[0], obs[1], S.MAT_Z + HM.OBS / 1000] + list(robot.getActualTCPPose()[3:]), 0.15, 0.5)
    imgA, tcpA, others = HM.capture_two(None, c=robot, r=robot)
    objs, _ = M.find_objects(imgA)
    out = os.path.join(HERE, "logs")
    for j, cnt in enumerate(objs):
        mask = np.zeros(imgA.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        c = cnt.reshape(-1, 2).mean(0)
        P = HM.on_plane(np.array([c]), tcpA, 0)[0]
        d = np.linalg.norm(P[:2] - env.d.qpos[a:a + 2] * 1000)
        if d > 30:
            continue
        print(f"物 {j}：消しゴムから {d:.0f}mm、画素 {int(mask.sum() / 255)}")
        mid = PA.band(mask)
        for name, (imgB, tcpB) in zip(("右", "左"), others):
            h, hs, sc = HM.best_height(imgA, tcpA, imgB, tcpB, mid)
            v = sc > -0.99
            print(f"  {name}：高さ {h:.1f}mm 合い {sc.max():.3f}、山の鋭さ {PA.peak(sc):.3f}")
            print("    " + " ".join(f"{hh:.0f}:{s:+.2f}" for hh, s in zip(hs[v][::2], sc[v][::2])))
        vis = imgA.copy()
        cv2.drawContours(vis, [cnt], -1, (0, 255, 255), 3)
        cv2.imwrite(os.path.join(out, f"diag_eraser_flat_{k}.jpg"), vis)
        for name, (imgB, _) in zip(("B", "C"), others):
            cv2.imwrite(os.path.join(out, f"diag_eraser_flat_{k}_{name}.jpg"), imgB)


if __name__ == "__main__":
    main()
