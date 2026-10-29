"""ROBOTIQ 版の 1 回の繰り返しの中で、どこに何ミリ秒かかっているかを測る。

  ① 描く（カメラ画像）  ② 腕の画素（色分けの画像を描く）  ③ 実環境ふうの加工  ④ FastSAM  ⑤ 番号をつける
  さらに、d1-3B が裏で動いている時の FastSAM の時間も測る（GPU の取り合い）。
  python robot/prof_grip.py   → robot/logs/prof_grip.json
"""
import json
import os
import sys
import threading
import time

import cv2
import mujoco
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
os.chdir(os.path.join(HERE, ".."))

from perception_fastsam import FastSamPerception  # noqa: E402
from run_tools import PARK_Q, crop  # noqa: E402
from sim_grip import GripEnv  # noqa: E402
from test_detect import grain, realistic_fast  # noqa: E402


def med(x):
    return round(float(np.median(x)), 1)


def main():
    env = GripEnv()
    env.reset(np.random.default_rng(32))
    empty = env.empty_table()
    table = np.all(np.abs(empty.astype(int) - empty[240, 600].astype(int)) < 6, axis=2)
    wood = grain(np.random.default_rng(1))
    rng = np.random.default_rng(0)
    env.d.qpos[:6] = PARK_Q + np.array([-0.8, 0, 0, 0, 0, 0])      # 腕が少し視野に入る姿勢
    mujoco.mj_forward(env.m, env.d)
    seg = mujoco.Renderer(env.m, 480, 640)
    seg.enable_segmentation_rendering()
    obj_geoms = set(env.object_geoms())
    robot_geoms = np.array([g for g in range(env.m.ngeom) if g not in obj_geoms and env.m.geom_bodyid[g] != 0])
    per = FastSamPerception()
    t = {k: [] for k in ("render", "robot_mask", "realistic", "fastsam", "update", "fastsam_with_d1")}
    for i in range(60):
        a = time.perf_counter()
        rgb = env.render()
        b = time.perf_counter()
        seg.update_scene(env.d, camera="top")
        mask = cv2.dilate(np.isin(seg.render()[..., 0], robot_geoms).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        c = time.perf_counter()
        img = realistic_fast(rgb, table, wood, (1.0, (1, 1, 1)))
        d = time.perf_counter()
        per.foreground(img)
        e = time.perf_counter()
        per.update(img, i, mask)
        f = time.perf_counter()
        if i >= 5:                                   # 最初の数回は準備の時間が入るので除く
            for k, v in zip(("render", "robot_mask", "realistic", "fastsam", "update"), (b - a, c - b, d - c, e - d, f - e)):
                t[k].append(v * 1000)
    # d1-3B を裏で動かし続けながら FastSAM を測る
    from d1_fast import D1Fast
    m = D1Fast(quant="fp8").model
    objs = per.visible(59)
    k0 = next(iter(objs))
    im = Image.fromarray(crop(img, objs[k0]))
    stop = [False]

    def busy():
        while not stop[0]:
            m.system_one("Close-up camera image of one object on a work table.",
                         {"q": {"type": "choice", "instructions": "What is this object?", "criteria": {"a": "a screw", "b": "a pen"}}}, images=[im])
    th = threading.Thread(target=busy, daemon=True)
    th.start()
    time.sleep(3)
    for i in range(40):
        img = realistic_fast(env.render(), table, wood, (1.0, (1, 1, 1)))
        a = time.perf_counter()
        per.foreground(img)
        t["fastsam_with_d1"].append((time.perf_counter() - a) * 1000)
    stop[0] = True
    th.join()
    out = {k: med(v) for k, v in t.items()}
    out["total_without_d1"] = round(sum(out[k] for k in ("render", "robot_mask", "realistic", "fastsam", "update")), 1)
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    with open("robot/logs/prof_grip.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
