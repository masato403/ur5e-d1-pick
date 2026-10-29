#!/bin/bash
# 手首の写真でだけ暗い所の下限を 40 にする（TR_WRIST_DARK_V=40）。wristdebug（9/15）と同じ場面・設定に 1 つだけ足して比べる（10/26）
cd "$(dirname "$0")/.."
n_run() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -like '*test_track_real.py*' }).Count" | tr -d '\r'; }
wait_idle() { while true; do c=$(n_run); case "$c" in 0) return 0;; [1-9]*) sleep 30;; *) echo "見張りが動かないので止める（$c）"; exit 1;; esac; done; }
X="TR_TO_GRIP=1 TR_D1_STEER=1 TR_CLEAR_MM=5 TR_PRED_LOW_S=2 TR_D1_CLOSE=1 TR_CLOSE_MOVING=1 TR_VEL_FIT=1.0 TR_PARTIAL=1 TR_D1_NOEXC=1 TR_D1_HES=1 TR_HES_TEXT=2 TR_D1_CLOSE_STILL=1"
wait_idle
until curl -s http://127.0.0.1:5006/ | grep -q ok; do sleep 10; done
env TRR_SCENES=3,5,6,7,9 SIM_FORCE=net TR_SAFE2=1 TR_WRIST2=1 TR_WRIST_DEBUG=1 TR_WRIST_DARK_V=40 SIM_DEL_OLD=1 $X .venv/Scripts/python.exe -u robot/test_track_real.py 200 --show > robot/logs/track_mv_wristdark40.log 2>&1
