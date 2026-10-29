r"""手首のカメラで物が「大きさの合う物が無い」になる訳を確かめる（10/23）。判断の係（GPU）は使わない。
物を 1 つマットに置き、真上（追う高さ）へ行って手首で 1 枚撮り、写った物の大きさを全部出す。三脚の測った大きさ（--w0 --l0）で通るかも出す。
使い方:  .venv\Scripts\python.exe robot\diag_wrist_size.py [eraser|r_pen|r_rod] [--w0 40 --l0 62]"""
import argparse
import os
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_grasp as G  # noqa: E402
import test_sim_real as T  # noqa: E402
import sim_real as S  # noqa: E402
import sim_rtde  # noqa: E402
from sim_rtde import SimRobot  # noqa: E402
import track_real as TR  # noqa: E402

KIND = {"eraser": "消しゴム", "r_pen": "ペン", "r_rod": "銀の棒"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", nargs="?", default="eraser")
    ap.add_argument("--w0", type=float, default=40.0)
    ap.add_argument("--l0", type=float, default=62.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--up", type=float, default=0.0)          # 追う高さからさらに上がる（mm）
    a = ap.parse_args()
    T.patch_real_code()
    env = S.RealLikeEnv()
    G.patch_grasp(env)
    G.patch_capture(env)
    robot = SimRobot(env)
    env.reset(np.random.default_rng(0))
    for j, t in enumerate(S.TOOLS):
        env.set_tool(t, np.array([1.5 + 0.1 * j, -0.5]), 0.0)
    xy = np.array([S.MAT_CENTER0[0], S.MAT_CENTER0[1]])
    env.set_tool(a.tool, xy, np.radians(a.yaw))
    G.go_to(env, robot.ik, xy)
    mujoco.mj_forward(env.m, env.d)
    for _ in range(20):
        sim_rtde.frame(env)
    kind = KIND[a.tool]
    h = TR.H_NOM[kind]
    f = {"h": h, "size": [a.w0, a.l0], "longv": [np.cos(np.radians(a.yaw)), np.sin(np.radians(a.yaw))], "center": np.r_[xy * 1000, h], "kind": kind}
    p = list(robot.getActualTCPPose())
    p[2] = (TR.TABLE_Z + TR.STEPS[0] + a.up) / 1000                # 追う高さ（机から STEPS[0]）
    robot.moveL(p, 0.2, 0.5)
    tcp = robot.getActualTCPPose()
    img = TR.L.grab_now(settle=0.1)
    out = os.path.join(HERE, "logs", f"diag_wrist_size_{a.tool}.jpg")
    cv2.imwrite(out, img)
    print(f"指先は机から {(tcp[2] * 1000 - TR.TABLE_Z):.0f}mm、写真 {img.shape[1]}×{img.shape[0]}")
    for cnt in TR.M.find_objects(img)[0]:
        pts = np.array([TR.wrist_xy(q, tcp, h) for q in cnt.reshape(-1, 2)[::4].astype(float)], np.float32)
        if len(pts) < 5:
            continue
        (cx, cy), (u, v), _ = cv2.minAreaRect(pts)
        w, l = sorted((u, v))
        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        cut = x_ <= 3 or y_ <= 3 or x_ + w_ >= img.shape[1] - 3 or y_ + h_ >= img.shape[0] - 3
        d = float(np.hypot(cx - f["center"][0], cy - f["center"][1]))
        print(f"  写った物：幅 {w:.0f}・長さ {l:.0f}mm、見当から {d:.0f}mm、端で切れた {cut}")
    print("  三脚の大きさで選んだ結果:", TR.wrist_measure(img, tcp, f, f["center"][:2]))
    f2 = dict(f, size=[{"eraser": 20, "r_pen": 11, "r_rod": 11}[a.tool], {"eraser": 52, "r_pen": 145, "r_rod": 77}[a.tool]])
    print("  本当の大きさで選んだ結果:", TR.wrist_measure(img, tcp, f2, f["center"][:2]))


if __name__ == "__main__":
    main()
