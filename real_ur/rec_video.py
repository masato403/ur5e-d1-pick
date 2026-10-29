r"""実機の通しの間、手首（latest.jpg）と三脚（tripod_latest.jpg）の写真を 0.1 秒ごとに読み、横に並べて mp4 に残す（10/23）。
cam_view.py が書く写真を読むだけで、カメラは開かない。
使い方:  .venv\Scripts\python.exe real_ur\rec_video.py 名前   （real_ur/logs_video/名前.mp4。real_ur/stop_rec ができるか 20 分で止まる）"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHOT = os.path.join(HERE, "cam_shots")
OUT = os.path.join(HERE, "logs_video")
STOP = os.path.join(HERE, "stop_rec")
os.makedirs(OUT, exist_ok=True)
if os.path.exists(STOP):
    os.remove(STOP)
path = os.path.join(OUT, (sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y%m%d_%H%M%S")) + ".mp4")
W, H, FPS = 640, 360, 10
vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W * 2, H))
last = [np.zeros((H, W, 3), np.uint8), np.zeros((H, W, 3), np.uint8)]
t_end, n = time.time() + 20 * 60, 0
while time.time() < t_end and not os.path.exists(STOP):
    t0 = time.time()
    for k, name in enumerate(("latest.jpg", "tripod_latest.jpg")):
        img = cv2.imread(os.path.join(SHOT, name))               # 書き換えの途中で読めない時は、前のコマを使う
        if img is not None:
            last[k] = cv2.resize(img, (W, H))
    frame = np.hstack(last)
    cv2.putText(frame, time.strftime("%H:%M:%S"), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    vw.write(frame)
    n += 1
    time.sleep(max(0.0, 1.0 / FPS - (time.time() - t0)))
vw.release()
print(f"{path} に {n} コマ（{n / FPS:.0f} 秒）残した")
