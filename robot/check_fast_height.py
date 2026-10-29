"""高さの探し方（粗く→細かく）を変える前と後で、測った値と時間を比べる。
  ・実機の写真（real_ur/cam_shots/height_A/B/C.jpg と height_tcp.json、10/15）：実機の校正の値のまま
使い方:  python robot/check_fast_height.py
"""
import importlib.util
import json
import os
import sys
import time
import types

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "..", "real_ur")
sys.path.insert(0, REAL)
for name in ("serial", "serial.tools", "serial.tools.list_ports"):
    sys.modules.setdefault(name, types.ModuleType(name))
import height_by_motion as HM  # noqa: E402
import pick_any as PA  # noqa: E402

spec = importlib.util.spec_from_file_location("hm_old", os.path.join(HERE, "..", "backup", "height_by_motion_20261016_a.py"))
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)


def run(fn):
    HM.best_height = fn
    shot = os.path.join(REAL, "cam_shots")
    tcp = json.load(open(os.path.join(shot, "height_tcp.json")))
    A = cv2.imread(os.path.join(shot, "height_A.jpg"))
    others = [(cv2.imread(os.path.join(shot, f"height_{k}.jpg")), tcp[k]) for k in ("B", "C")]
    t0 = time.perf_counter()
    found = PA.survey(A, tcp["A"], others)
    return found, time.perf_counter() - t0


def main():
    new = HM.best_height
    f_old, t_old = run(OLD.best_height)
    f_new, t_new = run(new)
    print(f"実機の写真（10/15 カバー）：前 {t_old:.1f} 秒 → 後 {t_new:.1f} 秒")
    for a, b in zip(f_old, f_new):
        print(f"  高さ {a['h']:.1f} → {b['h']:.1f} mm、幅 {a['size'][0]:.1f} → {b['size'][0]:.1f}、"
              f"中心のずれ {np.linalg.norm(a['center'][:2] - b['center'][:2]):.1f}mm")


if __name__ == "__main__":
    main()
