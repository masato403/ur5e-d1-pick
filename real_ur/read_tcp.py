r"""今の指先の位置を読み、box_corners.json に足す（10/23、箱の四隅を覚えさせる時用）。腕は動かさない。
使い方:  .venv_ur\Scripts\python.exe real_ur\read_tcp.py 1   （番号 1〜4。同じ番号は上書き）"""
import json
import os
import sys

from rtde_receive import RTDEReceiveInterface
from plan_hover import TABLE_Z

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "box_corners.json")
r = RTDEReceiveInterface("192.0.2.10")
p = r.getActualTCPPose()
r.disconnect()
d = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) and sys.argv[1] != "1" else {}
d[sys.argv[1]] = [p[0] * 1000, p[1] * 1000, p[2] * 1000 - TABLE_Z]
json.dump(d, open(OUT, "w", encoding="utf-8"), indent=1)
print(f"{sys.argv[1]}：x {d[sys.argv[1]][0]:.1f}  y {d[sys.argv[1]][1]:.1f}  机から {d[sys.argv[1]][2]:.1f} mm")
