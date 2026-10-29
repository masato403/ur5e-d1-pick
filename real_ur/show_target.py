"""実機に送る前に、今の姿勢 → 行き先 の動きをシミュレーターの窓で見せる（実機には何も送らない）。

使い方:  .venv\\Scripts\\python.exe real_ur\\show_target.py --from 0,-90,0,-90,0,0 --to 0,-90,90,-90,-90,0
  窓の中で、今の姿勢と行き先を数秒ずつ行き来する。道具の先の高さと、根元からの張り出しも表示する。
  シミュレーターの UR5e（mujoco_menagerie）は、関節の 0° の向きが実機と同じ。
"""
import argparse
import math
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "..", "robot", "ur5e_tools_2f85.xml")
TOOL_LEN = 0.211          # 実機の道具の長さ（ロボットから読んだ先端の位置から）


def deg_list(s):
    return np.radians([float(v) for v in s.split(",")])


def tool_tip(m, d):
    """手首の先の円盤（フランジ）から、道具の長さだけ先の点（実機の先端に当たる）。"""
    s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    R = d.site_xmat[s].reshape(3, 3)
    return d.site_xpos[s] + R[:, 2] * TOOL_LEN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="q0", default="0,-90,0,-90,0,0")
    ap.add_argument("--to", dest="q1", default="0,-90,90,-90,-90,0")
    ap.add_argument("--seconds", type=float, default=60)
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    q0, q1 = deg_list(a.q0), deg_list(a.q1)
    base = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")].copy() if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base") >= 0 else np.zeros(3)

    def pose(q):
        d.qpos[:6] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        base[:] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")]
        tip = tool_tip(m, d) - base
        return tip

    for name, q in (("今の姿勢", q0), ("行き先", q1)):
        tip = pose(q)
        print(f"{name}: 関節 {np.round(np.degrees(q), 1).tolist()} → 道具の先 根元から 高さ {tip[2]*100:.0f} cm、"
              f"水平の張り出し {math.hypot(tip[0], tip[1])*100:.0f} cm")
    lo = []
    for t in np.linspace(0, 1, 50):                               # 途中で一番低くなる所（ぶつかりやすさの目安）
        lo.append(pose(q0 + (q1 - q0) * t)[2])
    print(f"動く途中で、道具の先が一番低くなる高さ: {min(lo)*100:.0f} cm（根元の高さ基準）")

    m.opt.gravity[:] = 0
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = base + [0, 0, 0.4]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.0, 135, -15
        t0 = time.time()
        while v.is_running() and time.time() - t0 < a.seconds:
            ph = ((time.time() - t0) % 8.0) / 8.0                  # 8 秒で 1 往復（2 秒止まる → 2 秒で行く → 2 秒止まる → 2 秒で戻る）
            s = 0 if ph < .25 else (ph - .25) / .25 if ph < .5 else 1 if ph < .75 else 1 - (ph - .75) / .25
            s = 0.5 - 0.5 * math.cos(math.pi * s)
            d.qpos[:6] = q0 + (q1 - q0) * s
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
