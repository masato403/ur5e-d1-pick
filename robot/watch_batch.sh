#!/bin/bash
# 24 回の試験が終わるまで見守り、途中で止まっていたら続きから自動でやり直す（最大 5 回）
cd "$(dirname "$0")/.."
count() { ./.venv/Scripts/python.exe -c "
import json
print(sum(1 for l in open('robot/logs/run_tools.jsonl',encoding='utf-8') if json.loads(l)['seed'] in (20,21,22)))"; }
for i in 1 2 3 4 5; do
  while powershell -NoProfile -c "if (Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | ? {\$_.CommandLine -match 'run_tools'}) {exit 0} else {exit 1}"; do sleep 20; done
  n=$(count); echo "$(date +%H:%M) runs done: $n"
  [ "$n" -ge 24 ] && break
  echo "$(date +%H:%M) restart $i"
  ./.venv/Scripts/python.exe robot/run_tools.py --batch 20 21 22 >> robot/logs/batch_runs2.txt 2>&1
done
echo "$(date +%H:%M) finished, runs: $(count)"
