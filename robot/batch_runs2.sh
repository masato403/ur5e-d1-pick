#!/bin/bash
# 席を外している間に回す検証（約 50 分）。結果：robot/logs/diag_tools.jsonl と robot/logs/run_tools.jsonl
cd "$(dirname "$0")/.."
# 今動いている運搬の試験が終わるまで待つ
while powershell -NoProfile -c "if (Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | ? {\$_.CommandLine -match 'run_tools|diag_tools'}) {exit 0} else {exit 1}"; do sleep 20; done


echo "=== runs $(date +%H:%M)"
while IFS='|' read -r instr want; do
  for b in twostep onestep; do
    for s in 20 21 22; do
      ./.venv/Scripts/python.exe robot/run_tools.py --brain $b --instr "$instr" --want "$want" --seed $s 2>&1 | grep '"success"' | cut -c1-200
    done
  done
done <<'LIST'
Put all the screws into the box.|screw
Put the screwdriver into the box.|screwdriver
Put everything except the pen into the box.|screwdriver,screw
Put the pen and the screwdriver into the box.|pen,screwdriver
LIST
echo "=== done $(date +%H:%M)"
