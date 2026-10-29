"""撮る枚数を 3 枚（真ん中・右・左）から 2 枚（真ん中・右）に減らせるかを調べる（窓は出さない）。
物ごとに、右だけで測った高さと、右・左の良い方で測った高さ、それぞれの合い具合を出す。
使い方:  python robot/diag_two_shots.py [場面の数]
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_real as T  # noqa: E402
import sim_real as S  # noqa: E402
import sim_rtde  # noqa: E402
from sim_rtde import SimRobot  # noqa: E402
import height_by_motion as HM  # noqa: E402
import pick_any as PA  # noqa: E402
import measure as M  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    T.patch_real_code()
    env = S.RealLikeEnv()
    robot = SimRobot(env)
    HM.TABLE_Z = S.MAT_Z * 1000.0

    def grab_now(**_):
        t0 = env.d.time
        while env.d.time - t0 < 0.5:
            sim_rtde.frame(env)
        return env.render_real()
    HM.L.grab_now = grab_now
    HM.L.archive = lambda *a, **k: None
    rng = np.random.default_rng(7)
    obs = np.array(S.OBS_XY)
    worse = 0
    total = 0
    for k in range(n):
        env.reset(rng)
        for i, o in enumerate(env.objects):
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        T.go_to(env, robot.ik, obs)
        T.place_tools(env, rng)
        imgA, tcpA, others = HM.capture_two(None, c=robot, r=robot)
        (imgB, tcpB), (imgC, tcpC) = others
        objs, _ = M.find_objects(imgA)
        for j, cnt in enumerate(objs):
            mask = np.zeros(imgA.shape[:2], np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mid = PA.band(mask)
            hB, _, sB = HM.best_height(imgA, tcpA, imgB, tcpB, mid)
            hC, _, sC = HM.best_height(imgA, tcpA, imgC, tcpC, mid)
            pick = "右" if sB.max() >= sC.max() else "左"
            total += 1
            bad = sB.max() < 0 or (pick == "左" and abs(hB - hC) > 1.0)
            worse += bad
            print(f"{k}-{j}: 右 高さ {hB:5.1f}mm 合い {sB.max():+.3f} ／ 左 高さ {hC:5.1f}mm 合い {sC.max():+.3f} → {pick}"
                  + ("  ←右だけだと違う" if bad else ""))
    print(f"---- 右だけだと答えが変わる物：{worse}/{total}")


if __name__ == "__main__":
    main()
