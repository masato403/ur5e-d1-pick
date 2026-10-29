"""test_sim_grasp.py と同じ場面（乱数の種 2）を作り、指定した場面の、物ごとの高さの測り方の中身を出す（窓は出さない）。
使い方:  python robot/diag_scene.py 場面の番号   → robot/logs/scene_<番号>_A.jpg（見つけた物の輪郭と番号）
"""
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

import height_by_motion as HM  # noqa: E402
import measure as M  # noqa: E402
import pick_any as PA  # noqa: E402


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    T.patch_real_code()
    env = S.RealLikeEnv()
    robot = SimRobot(env)
    rng = np.random.default_rng(2)
    obs = np.array(S.OBS_XY)
    for k in range(want + 1):
        env.reset(rng)
        for i, o in enumerate(env.objects):
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        T.go_to(env, robot.ik, obs)
        placed = T.place_tools(env, rng)
    mujoco.mj_forward(env.m, env.d)
    for _ in range(200):
        mujoco.mj_step(env.m, env.d)
    imgA, tcpA, others = T.shots(env, robot.ik, obs)
    objs, _ = M.find_objects(imgA)
    dbg = imgA.copy()
    for i, cnt in enumerate(objs):
        mask = np.zeros(imgA.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        c = cnt.reshape(-1, 2).mean(0)
        P = HM.on_plane(np.array([c]), tcpA, 0.0)[0]
        truth = min(placed, key=lambda t: np.linalg.norm(placed[t][0] * 1000 - P[:2]))
        print(f"{i}: {T.NAME[truth]}（画素 {int(mask.sum() / 255)}）")
        for j, (b, tb) in enumerate(others):
            h, hs, sc = HM.best_height(imgA, tcpA, b, tb, mask)
            top = np.argsort(sc)[::-1][:4]
            print(f"   写真 {'BC'[j]}：高さ {h:.1f}mm（合い具合 {sc.max():.2f}）、上位 " + "、".join(f"{hs[t]:.1f}({sc[t]:.2f})" for t in top))
            parts = PA.part_heights(imgA, tcpA, b, tb, mask, cnt)
            print(f"      区切りごと：" + " ".join(p[3] if p[0] is None else f"{p[0]:.1f}" for p in parts))
        cv2.drawContours(dbg, [cnt], -1, (0, 255, 0), 3)
        cv2.putText(dbg, str(i), tuple(int(v) for v in c), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
    cv2.imwrite(os.path.join(HERE, "logs", f"scene_{want}_A.jpg"), cv2.resize(dbg, (960, 540)))
    for j, (b, _) in enumerate(others):
        cv2.imwrite(os.path.join(HERE, "logs", f"scene_{want}_{'BC'[j]}.jpg"), cv2.resize(b, (960, 540)))


if __name__ == "__main__":
    main()
