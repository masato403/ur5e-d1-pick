"""実機に合わせたマット（SIM_MAT=real）のシミュレーターで残した写真（test_sim_grasp.py の SIM_SAVE）を、ロボット無しで測り直し、
暗い所の直し方の候補を比べる（10/19 夜。ペン「まわりがマットでない」、消しゴム「幅 24mm」で外れた件）。
物ごとに、本当の位置にいちばん近い候補の 幅・長さ・まわりのマットの割合 と、つかむ候補から外れる理由を出す（種類は聞かない）。
見つけた輪郭を描いた画像を <フォルダ>/<候補>_sceneNN.jpg に残す。
使い方:  python robot/diag_dark_refine.py robot/logs/sim_shots_realmat [候補 base ...]
"""
import glob
import math
import os
import pickle
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_real as T  # noqa: E402
import sim_real as S  # noqa: E402
import measure as M  # noqa: E402
import pick_any as PA  # noqa: E402

TRUE_W = {T.NAME[t]: T.TRUE[t][1] for t in S.TOOLS}
import dark_two  # noqa: E402
ORIG = {"refine": M.refine, "mat_area": M.mat_area, "survey": PA.survey, "v": M.DARK_V_MIN}


def variant(name):
    """候補の直し方を入れる（実機用のプログラムは書き換えず、ここで差し替える）。"""
    M.refine, M.mat_area, PA.survey, M.DARK_V_MIN = ORIG["refine"], ORIG["mat_area"], ORIG["survey"], ORIG["v"]
    if name == "base":
        return
    if name == "two":                                            # 物ごとに 2 つの決まりで測る（dark_two.py）
        dark_two.install()
        return
    if name.startswith("v"):                                     # 暗い所の明るさの下限を全体で変える（例 v40）
        M.DARK_V_MIN = int(name[1:])
        return
    raise SystemExit(f"知らない候補：{name}")


def run(folder, name):
    variant(name)
    stats = {k: {"ok": 0, "n": 0, "why": []} for k in TRUE_W}
    for p in sorted(glob.glob(os.path.join(folder, "scene*.pkl"))):
        d = pickle.load(open(p, "rb"))
        PA.BOX = d["box"]
        found = PA.survey(d["imgA"], d["tcpA"], d["others"])
        vis = d["imgA"].copy()
        for f in found:
            cv2.drawContours(vis, [f["contour"]], -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(folder, f"{name}_{os.path.basename(p)[:-4]}.jpg"), vis)
        for kind, xy in d["truth"].items():
            st = stats[kind]
            st["n"] += 1
            near = [f for f in found if math.hypot(f["center"][0] - xy[0], f["center"][1] - xy[1]) < 25]
            if not near:
                st["why"].append(f"{os.path.basename(p)[:-4]} 見つからない")
                continue
            f = min(near, key=lambda f: math.hypot(f["center"][0] - xy[0], f["center"][1] - xy[1]))
            far = math.hypot(f["center"][0] - d["tcpA"][0] * 1000, f["center"][1] - d["tcpA"][1] * 1000)
            why = PA.why_not(f, far)
            extra = f"（{len(near)} 個）" if len(near) > 1 else ""
            if why or len(near) > 1:
                st["why"].append(f"{os.path.basename(p)[:-4]} 幅 {f['size'][0]:.1f}（本当 {TRUE_W[kind]:.0f}）マット {f['on_mat']:.0%} "
                                 + "・".join(why) + extra)
            else:
                st["ok"] += 1
    print(f"== {name}：" + "、".join(f"{k} {s['ok']}/{s['n']}" for k, s in stats.items()), flush=True)
    for k, s in stats.items():
        for w in s["why"]:
            print(f"   {k}：{w}")


def main():
    T.patch_real_code()
    folder = sys.argv[1]
    for name in sys.argv[2:] or ["base"]:
        run(folder, name)


if __name__ == "__main__":
    main()
