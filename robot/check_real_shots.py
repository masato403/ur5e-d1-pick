"""昨日（10/15）の実機の写真を、今の実機用のプログラム（pick_any.survey）で測り直す。実機の校正の値のまま。
写真の組：height_A/B/C.jpg ＋ height_tcp.json。見つけた物を描いた画像を robot/logs/real_<名前>.jpg に保存する。
使い方:  python robot/check_real_shots.py [名前の頭（既定 height）]
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
for name in ("serial", "serial.tools", "serial.tools.list_ports"):
    sys.modules.setdefault(name, types.ModuleType(name))
import measure as M  # noqa: E402
import pick_any as PA  # noqa: E402


def main():
    pre = sys.argv[1] if len(sys.argv) > 1 else "height"
    shot = os.path.join(REAL, "cam_shots")
    tcp = json.load(open(os.path.join(shot, "height_tcp.json")))
    A = cv2.imread(os.path.join(shot, f"{pre}_A.jpg"))
    others = [(cv2.imread(os.path.join(shot, f"{pre}_{k}.jpg")), tcp[k]) for k in ("B", "C")]
    objs, _ = M.find_objects(A)
    print(f"マットの上の塊 {len(objs)} 個")
    found = PA.survey(A, tcp["A"], others)
    dbg = A.copy()
    cv2.drawContours(dbg, objs, -1, (255, 0, 255), 2)
    for i, f in enumerate(found):
        print(f"{i}: 高さ {f['h']:.1f}mm、上の面 {f['size'][0]:.1f} × {f['size'][1]:.1f}mm、まわりがマット {f['on_mat']:.0%}、合い具合 {f['match']:.2f}")
        cv2.drawContours(dbg, [f["contour"]], -1, (0, 255, 0), 3)
        x, y = f["contour"].reshape(-1, 2).min(0)
        cv2.putText(dbg, f"{i}: h{f['h']:.0f} {f['size'][0]:.0f}x{f['size'][1]:.0f}", (int(x), int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imwrite(os.path.join(HERE, "logs", f"real_{pre}_{os.environ.get('OBJ_REFINE', '1')}.jpg"), cv2.resize(dbg, (960, 540)))


if __name__ == "__main__":
    main()
