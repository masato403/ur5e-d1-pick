r"""今の指先の位置を読み、マス目の交点の印（目盛り）と一緒に mat_grid_points.json に足す（10/23、三脚の校正でマットの位置を決める用）。腕は動かさない。
使い方:  .venv_ur\Scripts\python.exe real_ur\read_mat_pt.py 0 0   （マス目の目盛り：横 i・縦 j（cm）。同じ目盛りは上書き）"""
import json
import os
import sys

from rtde_receive import RTDEReceiveInterface
from plan_hover import TABLE_Z

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mat_grid_points.json")
i, j = float(sys.argv[1]), float(sys.argv[2])
r = RTDEReceiveInterface("192.0.2.10")
p = r.getActualTCPPose()
r.disconnect()
d = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
d[f"{i:g},{j:g}"] = {"grid_cm": [i, j], "tcp_mm": [round(v * 1000, 2) for v in p[:3]], "rv": [round(v, 4) for v in p[3:]]}
json.dump(d, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"目盛り ({i:g},{j:g})：x {p[0] * 1000:.1f}  y {p[1] * 1000:.1f}  机から {p[2] * 1000 - TABLE_Z:.1f}mm（{len(d)} 点）")
