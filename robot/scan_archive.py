"""実機で残した 3 枚の組（cam_shots/archive/日付/時刻_height_A/B/C.jpg ＋ .json）を、今の実機用のプログラムで全部測り直す（ロボットは使わない）。
見つけた物ごとに、種類（kind_ai.kind_of）・高さ・上の面の大きさ・中心を出す。見つけた物を描いた画像を robot/logs/scan_<日付>/ に残す。
銀の棒が二重に見つかる件など、実機の写真で起きたことを、ロボット無しで調べ直す用（10/19）。
使い方:  MAT_V_MIN=90 MAT_S_MAX=60 python robot/scan_archive.py [日付 20261016] [時刻の頭 16]
"""
import json
import os
import sys
import types

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "..", "real_ur")
sys.path.insert(0, REAL)
for name in ("serial", "serial.tools", "serial.tools.list_ports", "rtde_control", "rtde_receive"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["rtde_control"].RTDEControlInterface = None          # ロボットにはつながない（読み込みを通すだけ）
sys.modules["rtde_receive"].RTDEReceiveInterface = None
import pick_any as PA  # noqa: E402
if os.environ.get("SURVEY_TWO") == "1":                          # 暗い所の直しの候補（robot/dark_two.py。10/19 夜）
    sys.path.insert(0, HERE)
    import dark_two  # noqa: E402
    dark_two.install()
from kind_ai import kind_of  # noqa: E402


def secs(t):
    return int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


def triplets(folder):
    """A の時刻の前後 4 秒以内にある B と C を、同じ組とみなす（撮る順番は B → A → C）。"""
    shots = {}
    for f in os.listdir(folder):
        if f.endswith(".jpg") and "_height_" in f:
            t, k = f[:6], f[-5]
            shots.setdefault(k, []).append(t)
    out = []
    for ta in sorted(shots.get("A", [])):
        near = {}
        for k in "BC":
            c = [t for t in shots.get(k, []) if abs(secs(t) - secs(ta)) <= 4]
            if c:
                near[k] = min(c, key=lambda t: abs(secs(t) - secs(ta)))
        if len(near) == 2:
            out.append((ta, near["B"], near["C"]))
    return out


def load(folder, t, k):
    img = cv2.imread(os.path.join(folder, f"{t}_height_{k}.jpg"))
    tcp = json.load(open(os.path.join(folder, f"{t}_height_{k}.json")))["tcp"]
    return img, tcp


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else "20261016"
    pre = sys.argv[2] if len(sys.argv) > 2 else ""
    folder = os.path.join(REAL, "cam_shots", "archive", day)
    out = os.path.join(HERE, "logs", f"scan_{day}")
    os.makedirs(out, exist_ok=True)
    rows = []
    for ta, tb, tc in triplets(folder):
        if not ta.startswith(pre):
            continue
        A, tcpA = load(folder, ta, "A")
        others = [load(folder, tb, "B"), load(folder, tc, "C")]
        found = PA.survey(A, tcpA, others)
        dbg = A.copy()
        print(f"==== {ta}（物 {len(found)} 個）")
        for i, f in enumerate(found):
            k = kind_of(f) or "？"
            print(f"  {i}: {k:4s} 高さ {f['h']:5.1f}（全体 {f.get('h_whole', f['h']):5.1f}）上の面 {f['size'][0]:5.1f} × {f['size'][1]:5.1f}mm "
                  f"中心 {f['center'][0]:6.1f},{f['center'][1]:6.1f} 丸 {int(bool(f['round']))} 光る {int(bool(f.get('shiny')))} "
                  f"まわり {f['on_mat']:.0%} 合い {f['match']:.2f}")
            box = cv2.boxPoints(cv2.minAreaRect(f["contour"])).astype(int)
            cv2.drawContours(dbg, [box], -1, (0, 200, 0) if kind_of(f) else (0, 0, 255), 3)
            cv2.putText(dbg, f"{i}", tuple(box.min(0)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 200, 0) if kind_of(f) else (0, 0, 255), 4)
            rows.append({"t": ta, "i": i, "kind": kind_of(f), "h": f["h"], "size": list(map(float, f["size"])),
                         "center": list(map(float, f["center"][:2])), "on_mat": f["on_mat"], "match": f["match"]})
        cv2.imwrite(os.path.join(out, f"{ta}.jpg"), cv2.resize(dbg, (960, 540)))
    json.dump(rows, open(os.path.join(out, "scan.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
