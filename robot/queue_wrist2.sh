#!/bin/bash
# 安全の通しの後に、手首の直し TR_WRIST2=1 を足して場面 3〜22 を流す（10/23、近づいて上がってしまうのを直す・手首のカメラをもっと使う）。safe2 と比べる
# 前の版は「powershell」が見つからず（127）、待たずに同時に流れたので powershell.exe にした。見張りが動かない時は流さずに止める。
cd "$(dirname "$0")/.."
n_run() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -like '*test_track_real.py*' }).Count" | tr -d '\r'; }
wait_idle() { while true; do c=$(n_run); case "$c" in 0) return 0;; [1-9]*) sleep 30;; *) echo "見張りが動かないので止める（$c）"; exit 1;; esac; done; }
X="TR_TO_GRIP=1 TR_D1_STEER=1 TR_CLEAR_MM=5 TR_PRED_LOW_S=2 TR_D1_CLOSE=1 TR_CLOSE_MOVING=1 TR_VEL_FIT=1.0 TR_PARTIAL=1 TR_D1_NOEXC=1 TR_D1_HES=1 TR_HES_TEXT=2 TR_D1_CLOSE_STILL=1"
S=$(seq -s, 3 22)
wait_idle
env TRR_SCENES=$S SIM_FORCE=net TR_SAFE2=1 TR_WRIST2=1 SIM_DEL_OLD=1 $X .venv/Scripts/python.exe -u robot/test_track_real.py 200 --show > robot/logs/track_mv_d1all_safe2_wrist2.log 2>&1
