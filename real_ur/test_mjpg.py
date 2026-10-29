"""手首のカメラを YUY2（今）と MJPG で開き、1 秒に撮れる枚数と画質を比べる（10/20）。ロボットは要らない。
使い方:  python real_ur/test_mjpg.py [カメラ番号]   （cam_view を止めてから。写真は real_ur/logs_mjpg/）
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "logs_mjpg")
STD_Y = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                  14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                  49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)
ZIGZAG = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
          35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]


def jpeg_quality(buf):
    """JPEG の最初の量子化表（明るさ）から、品質（1〜100）を見積もる。"""
    b = bytes(buf)
    i = b.find(b"\xff\xdb")
    if i < 0:
        return None
    t = np.zeros(64)
    for k in range(64):
        t[ZIGZAG[k]] = b[i + 5 + k]
    s = float(np.mean(t / STD_Y) * 100)
    return round(100 - s / 2) if s <= 100 else round(5000 / s)


def open_cam(n, fourcc):
    c = cv2.VideoCapture(n, cv2.CAP_DSHOW)
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    c.set(cv2.CAP_PROP_FPS, 30)
    return c


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    imgs = {}
    for fc in ["YUY2", "MJPG"]:
        c = open_cam(n, fc)
        got = (int(c.get(cv2.CAP_PROP_FOURCC)) & 0xFFFFFFFF).to_bytes(4, "little").decode(errors="replace")
        for _ in range(15):
            c.read()
        t = time.perf_counter()
        k = 0
        while time.perf_counter() - t < 4.0:
            ok, img = c.read()
            k += ok
        fps = k / (time.perf_counter() - t)
        frames = [c.read()[1] for _ in range(8)]
        img = np.median(np.stack(frames), axis=0).astype(np.uint8)   # 8 枚の真ん中の値（ノイズを減らして比べる）
        imgs[fc] = img
        cv2.imwrite(os.path.join(OUT, f"{fc}.png"), img)
        q = None
        if fc == "MJPG":
            c.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            ok, raw = c.read()
            if ok and raw is not None and raw.size > 1000 and raw.reshape(-1)[0] == 0xFF:
                q = jpeg_quality(raw.reshape(-1))
                open(os.path.join(OUT, "MJPG_raw.jpg"), "wb").write(raw.tobytes())
        c.release()
        print(f"{fc}：実際の形式 {got}、大きさ {img.shape[1]}×{img.shape[0]}、1 秒に {fps:.1f} 枚" + (f"、JPEG の品質の見積もり {q}" if q else ""), flush=True)
    a, b = imgs["YUY2"].astype(float), imgs["MJPG"].astype(float)
    if a.shape == b.shape:
        mse = np.mean((a - b) ** 2)
        print(f"YUY2 と MJPG の差：PSNR {10 * np.log10(255 ** 2 / max(mse, 1e-9)):.1f}dB、"
              f"明るさの平均 {a.mean():.1f} → {b.mean():.1f}、色の差の平均 {np.abs(a - b).mean():.2f}")


if __name__ == "__main__":
    main()
