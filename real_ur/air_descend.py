r"""直した descend_safe を、物の無い空中で確かめる（10/23）。ハンドは動かさない。物にも机にも触れない。
今の xy のまま、机から TOP mm → BOT mm へ力を見ながら下り、力の読み（基準との差）を 0.02 秒ごとに記録する。止まったら止まった所で終わる。
使い方:  .venv_ur\Scripts\python.exe real_ur\air_descend.py            （行き先を出すだけ）
         .venv_ur\Scripts\python.exe real_ur\air_descend.py --go       （動かす）"""
import argparse
import json
import os
import time

import numpy as np

import pick_by_camera as PB
from plan_hover import TABLE_Z

IP = "192.0.2.10"
ap = argparse.ArgumentParser()
ap.add_argument("--go", action="store_true")
ap.add_argument("--top", type=float, default=150.0)
ap.add_argument("--bot", type=float, default=60.0)
ap.add_argument("--n", type=float, default=10.0)
a = ap.parse_args()
assert a.bot >= 50 and a.top > a.bot
from rtde_receive import RTDEReceiveInterface
r = RTDEReceiveInterface(IP)
p0 = r.getActualTCPPose()
print(f"今の指先 x {p0[0] * 1000:.0f} y {p0[1] * 1000:.0f} 机から {p0[2] * 1000 - TABLE_Z:.0f}mm → 机から {a.top:.0f}mm へ → 力を見ながら {a.bot:.0f}mm まで下りる（止める力 {a.n}N、"
      f"速さ {PB.DOWN_FAST * 1000:.0f}mm/秒、物の近く {PB.DOWN_SPEED * 1000:.0f}mm/秒、加速 {PB.ACCEL} m/秒²）")
if not a.go:
    raise SystemExit("--go が無いので、ここまで（何も送っていません）")
from rtde_control import RTDEControlInterface
c = RTDEControlInterface(IP)
PB.SAFE_N = a.n
log = []
fv = PB.force_vec
PB.force_vec = lambda r_: (lambda f_: (log.append([time.time(), r_.getActualTCPPose()[2] * 1000 - TABLE_Z] + f_.tolist()), f_)[1])(fv(r_))
try:
    top = list(p0)
    top[2] = (TABLE_Z + a.top) / 1000
    c.moveL(top, 0.05, 0.2)
    bot = list(top)
    bot[2] = (TABLE_Z + a.bot) / 1000
    ok = PB.descend_safe(c, r, bot, PB.ACCEL)
    q = r.getActualTCPPose()
    print(f"{'下り切った' if ok else '安全で止まった'}：指先は机から {q[2] * 1000 - TABLE_Z:.1f}mm")
    c.moveL(top, 0.05, 0.2)
finally:
    c.stopScript()
    c.disconnect()
L = np.array(log)
if len(L) > 11:
    d = np.linalg.norm(L[11:, 2:5] - L[1:11, 2:5].mean(axis=0), axis=1)
    print(f"力の読みの差：最大 {d.max():.1f}N（机から {L[11 + d.argmax(), 1]:.0f}mm）、平均 {d.mean():.1f}N、{len(d)} 回")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"logs_air_descend_{time.strftime('%H%M%S')}.json")
json.dump(L.tolist(), open(out, "w"))
print("記録:", out)
