"""フリードライブ（手で腕を動かせる状態）にして、real_ur/stop_freedrive ができるか 10 分たつまで保つ（10/19 夜）。"""
import os
import time

from rtde_control import RTDEControlInterface

HERE = os.path.dirname(os.path.abspath(__file__))
STOP = os.path.join(HERE, "stop_freedrive")
if os.path.exists(STOP):
    os.remove(STOP)
c = RTDEControlInterface("192.0.2.10")
try:
    c.teachMode()
    print("フリードライブにした", flush=True)
    t0 = time.time()
    while not os.path.exists(STOP) and time.time() - t0 < 600:
        time.sleep(0.2)
finally:
    c.endTeachMode()
    c.stopScript()
    c.disconnect()
    print("フリードライブを終えた", flush=True)
