#!/bin/sh
# 気になる設定を全部、順番に試す（GPU を使うので 1 つずつ。同時に動かすとメモリ不足で落ちる）
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
P="robot/real/photo01.png robot/real/photo02.png robot/real/photo03.png robot/real/photo04.png robot/real/photo05.png robot/real/photo06.png robot/real/photo07.png robot/real/photo08.png robot/real/photo09.png"
for cfg in "base" "imgsz640 IMGSZ=640" "imgsz320 IMGSZ=320" "imgsz256 IMGSZ=256" \
           "input640 INPUT=640 IMGSZ=640" "input256 INPUT=256 IMGSZ=256" \
           "crop128 CROP=128" "crop384 CROP=384" "conf010 CONF=0.10" "iou050 IOU=0.5" "iou090 IOU=0.9" "mobilesam SAM=mobilesam"; do
  set -- $cfg
  tag=$1; shift
  echo "=== $tag $(date +%H:%M)"
  env TAG=$tag "$@" $PY robot/test_real.py $P > robot/logs/real_suite_$tag.txt 2>&1 || echo "失敗 $tag"
done
echo "=== sim fastsam256 $(date +%H:%M)"
$PY robot/test_detect.py 30 fastsam,fastsam256 > robot/logs/test_detect_256.txt 2>&1 || echo "失敗 sim"
echo "=== 終わり $(date +%H:%M)"
