"""腕とハンドの今の状態を読むだけ（何も動かさない）。"""
import sys
from rtde_receive import RTDEReceiveInterface
from hand import HandClient
from plan_hover import TABLE_Z
r = RTDEReceiveInterface(sys.argv[1])
p = r.getActualTCPPose()
print(f"腕: x {p[0]*1000:.1f}  y {p[1]*1000:.1f}  机から {p[2]*1000 - TABLE_Z:.0f}mm、安全状態 {r.getSafetyMode()}、ロボットの状態 {r.getRobotMode()}")
r.disconnect()
h = HandClient(log=lambda *a: None)
print("ハンド:", h.cmd("status", quiet=True))
print(f"ハンドの開き量 {h.pos_mm():.1f}mm")
h.check("確認")
print("ハンドは動かせる状態")
h.close_port()
