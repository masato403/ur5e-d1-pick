r"""実機で、撮る前の待ち時間（腕が止まってから撮るまで）を変えて 3 枚撮り、写真のぶれ・測った値・かかった時間を比べる（つかまない）。
待ち時間を縮めると探す時間が縮むが、ぶれで高さを誤ると意味がない（10/16）。精度が落ちない一番短い待ち時間を探す。
使い方:  .venv_ur\Scripts\python.exe real_ur\test_settle.py --ip 192.0.2.10 [--settle 0.5 0.3 0.15 0] [--rep 2]
"""
import argparse
import time

import cv2
import numpy as np

import height_by_motion as HM
import locate as L
import pick_any as PA
from go_start import go_start
from hand import HandClient
from kind_ai import kind_of


def sharp(img):
    """写真のくっきり具合（明るさの 2 回微分のばらつき。ぶれると小さくなる）。マットの真ん中あたりだけで測る。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    return float(cv2.Laplacian(g[int(h * 0.1):int(h * 0.6), int(w * 0.2):int(w * 0.8)], cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--settle", type=float, nargs="+", default=[0.5, 0.3, 0.15, 0.0])
    ap.add_argument("--rep", type=int, default=2)
    a = ap.parse_args()
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(a.ip)
    c = RTDEControlInterface(a.ip)
    rows = []
    try:
        go_start(c, r)
        h = HandClient(log=lambda s: None)
        h.open()
        h.close_port()
        for s in a.settle:
            for k in range(a.rep):
                grabs = []

                def grab(s=s):
                    t0 = time.perf_counter()
                    img = L.grab_now(settle=s)
                    grabs.append(time.perf_counter() - t0 - s)     # 待ちを除いた、受け渡しの時間
                    return img
                HM.grab_fresh = grab
                HM.clear_memo()
                t0 = time.perf_counter()
                imgA, tcpA, others = HM.capture_two(a.ip, c=c, r=r)
                t_cap = time.perf_counter() - t0
                found = PA.survey(imgA, tcpA, others)
                got = {}
                for f in found:
                    kd = kind_of(f)
                    if kd and f["on_mat"] >= PA.MAT_RING and kd not in got:
                        got[kd] = f
                sh = [sharp(imgA)] + [sharp(im) for im, _ in others]
                print(f"待ち {s:.2f} 秒 {k + 1} 回目：3 枚 {t_cap:.1f} 秒（受け渡し 1 枚 {np.mean(grabs):.2f} 秒）、くっきり "
                      + "・".join(f"{v:.0f}" for v in sh)
                      + "、" + "、".join(f"{kd} 高さ {f['h']:.1f} 中心 {f['center'][0]:.1f},{f['center'][1]:.1f} 幅 {f['size'][0]:.1f}"
                                        for kd, f in got.items()))
                rows.append((s, t_cap, np.mean(grabs), np.mean(sh), got))
                go_start(c, r)
    finally:
        c.stopScript()
        c.disconnect()
        r.disconnect()
    print("==== まとめ（待ち 0.5 秒の 1 回目と比べた差）")
    base = rows[0][4]
    for s, t_cap, g, sh, got in rows:
        diffs = []
        for kd, f in got.items():
            if kd in base:
                b = base[kd]
                diffs.append(f"{kd} 高さ {f['h'] - b['h']:+.1f}・中心 {np.linalg.norm(f['center'][:2] - b['center'][:2]):.1f}mm")
        print(f"待ち {s:.2f} 秒：3 枚 {t_cap:.1f} 秒、受け渡し {g:.2f} 秒、くっきり {sh:.0f}、" + "、".join(diffs))


if __name__ == "__main__":
    main()
