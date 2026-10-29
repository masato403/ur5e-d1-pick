#!/bin/bash
# 止める値（PB_SAFE_N）と、物の近くで下りる速さ（PB_DOWN_SPEED）を組み合わせて、押し付ける場面とふつうの場面を流す（10/23）
cd "$(dirname "$0")/.."
for v in 0.05 0.025; do for n in 5 10 15 20; do
  echo "== 止める値 ${n}N・下りる速さ ${v}m/秒"
  SIM_FORCE=net PB_SAFE_N=$n PB_DOWN_SPEED=$v SAFE_ONLY=A2,B .venv/Scripts/python.exe -u robot/test_safety_sim.py 2>&1 | grep -aE "：|Traceback|Error"
done; done
