"""plan_hover.json の動きを、シミュレーターの窓でくり返し見せる（実機には何も送らない）。
使い方:  .venv\\Scripts\\python.exe real_ur\\show_plan.py
"""
import json
import math
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "..", "robot", "ur5e_tools_2f85.xml")


def main(seconds=600, name="plan_hover.json"):
    plan = json.load(open(os.path.join(HERE, name)))
    qs = np.array([plan["start_q"]] + [s["q"] for s in plan["steps"]])
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    m.opt.gravity[:] = 0
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    mujoco.mj_forward(m, d)
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = d.xpos[base] + [0, 0, 0.3]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.6, 200, -25
        t0 = time.time()
        n = len(qs) - 1
        while v.is_running() and time.time() - t0 < seconds:
            ph = (time.time() - t0) % 10.0                 # 2 秒止まる → 6 秒で動く → 2 秒止まる
            s = 0.0 if ph < 2 else min(1.0, (ph - 2) / 6)
            f = s * n
            i = min(int(f), n - 1)
            d.qpos[:6] = qs[i] + (qs[i + 1] - qs[i]) * (f - i)
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    import sys
    main(name=sys.argv[1] if len(sys.argv) > 1 else "plan_hover.json")
