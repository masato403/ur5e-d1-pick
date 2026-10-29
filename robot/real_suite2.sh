#!/bin/sh
# 追加の設定（置き場所を FastSAM で見つける）。real_suite.sh が終わってから、順番に流す
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
P="robot/real/photo01.png robot/real/photo02.png robot/real/photo03.png robot/real/photo04.png robot/real/photo05.png robot/real/photo06.png robot/real/photo07.png robot/real/photo08.png robot/real/photo09.png"
echo "=== place_sam $(date +%H:%M)"
env TAG=place_sam PLACE=sam $PY robot/test_real.py $P > robot/logs/real_suite_place_sam.txt 2>&1 || echo "失敗 place_sam"
echo "=== 終わり $(date +%H:%M)"
