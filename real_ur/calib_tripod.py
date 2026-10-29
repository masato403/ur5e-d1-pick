r"""三脚のカメラの校正（10/20）。腕の先（赤い指）を目印に、マットの上の 3×3 か所・2 つの高さへ順に動かし、
三脚の写真の中の指先の位置と、腕の位置の組を集めて、三脚のカメラの位置・向き・焦点距離を求める。
ハンドは開け閉めしない。手首は回さない。cam_view.py --cam 2 --cam2 1 が動いていること（tripod_latest.jpg を使う）。
使い方:  .venv_ur\Scripts\python.exe real_ur\calib_tripod.py --ip 192.0.2.10 --go
結果: real_ur/tripod_calib.json、写真は real_ur/logs_tripod_calib/
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

from plan_hover import TABLE_Z
import web_page

HERE = os.path.dirname(os.path.abspath(__file__))
LATEST = os.path.join(HERE, "cam_shots", "tripod_latest.jpg")
OUT = os.path.join(HERE, "logs_tripod_calib")
SPEED, ACCEL = 0.05, 0.2
XS, YS = (20.0, 120.0, 220.0), (-350.0, -430.0, -510.0)   # 箱（x 300〜430）には近づかない
HS = (50.0, 110.0)                                         # 机から指先まで（move_to.py と同じく 50mm 以上）
H_HIGH = 200.0                                             # --high：高さの差が小さいと「広角で近い」と「普通で遠い」の区別がつかない（10/20）。三脚に近い手前の列には置かない


def fresh_image(after, timeout=12.0):
    """after（時刻）より後に書かれた三脚の写真を待って読む。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if os.path.getmtime(LATEST) > after:
                time.sleep(0.2)                                    # 書き終わるのを待つ
                img = cv2.imread(LATEST)
                if img is not None:
                    return img
        except OSError:
            pass
        time.sleep(0.2)
    return None


def finger_tip(img):
    """赤い指の先（写真の中でいちばん下の赤い所の真ん中）。いちばん大きい赤い塊の近くだけを見る。無ければ None。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (((hsv[..., 0] < 8) | (hsv[..., 0] > 172)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 70)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[big, cv2.CC_STAT_AREA] < 150:
        return None
    x, y, w, h = st[big, :4]
    pad = max(w, h)                                            # 2 本の指が別の塊になっても拾う
    keep = np.zeros_like(m)
    for i in range(1, n):
        cx, cy = st[i, 0] + st[i, 2] / 2, st[i, 1] + st[i, 3] / 2
        if st[i, cv2.CC_STAT_AREA] >= 30 and x - pad <= cx <= x + w + pad and y - pad <= cy <= y + h + pad:
            keep[lab == i] = 255
    ys, xs = np.nonzero(keep)
    low = ys >= np.percentile(ys, 97)                          # いちばん下の 3%
    return np.array([xs[low].mean(), ys[low].max()], float), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--high", action="store_true", help=f"奥の 2 列に、机から {H_HIGH:.0f}mm の点を足す")
    ap.add_argument("--closed", action="store_true", help="指を閉じて、指先を 1 点の目印にする（終わったら元の開き量に戻す）")
    a = ap.parse_args()
    web_page.tee_stdout()
    os.makedirs(OUT, exist_ok=True)
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(a.ip)
    p0 = r.getActualTCPPose()
    rot = list(p0[3:])                                         # 手首の向きは今のまま
    goals = [(x, y, h) for h in HS for y in YS for x in XS] + ([(x, y, H_HIGH) for y in YS[:2] for x in XS] if a.high else [])
    print(f"三脚の校正：{len(goals)} か所（x {XS}、y {YS}、机から {HS}mm）、速さ {SPEED}m/秒", flush=True)
    if not a.go:
        print("--go が無いので、ここまで（何も送っていません）")
        return
    from rtde_control import RTDEControlInterface
    hand, open0 = None, None
    if a.closed:                                               # ハンドを動かす前に、開き量と状態を確かめる（HandClient.check。異常なら止まる）
        from hand import HandClient
        hand = HandClient(log=lambda s: print(s, flush=True))
        open0 = hand.check("指を閉じる")
        print(f"  指を閉じた：開き量 {hand.move_mm(0.0):.1f}mm", flush=True)
    c = RTDEControlInterface(a.ip)
    pts3, pts2, rows = [], [], []
    try:
        for k, (x, y, h) in enumerate(goals):
            goal = [x / 1000, y / 1000, (TABLE_Z + h) / 1000] + rot
            if not c.isPoseWithinSafetyLimits(goal):
                print(f"  {k + 1}: 安全設定の範囲の外なので飛ばす", flush=True)
                continue
            c.moveL(goal, SPEED, ACCEL)
            p = r.getActualTCPPose()
            if r.getSafetyMode() != 1:
                print("  安全状態が普通でないので止めます", flush=True)
                break
            img = fresh_image(time.time() + 0.3)
            if img is None:
                print(f"  {k + 1}: 三脚の写真が来ない（cam_view が動いているか）", flush=True)
                continue
            f = finger_tip(img)
            xyz = [p[0] * 1000, p[1] * 1000, p[2] * 1000]
            if f is None:
                print(f"  {k + 1}: 赤い指が見つからない", flush=True)
                cv2.imwrite(os.path.join(OUT, f"p{k + 1:02d}_miss.jpg"), img)
                continue
            uv, keep = f
            pts3.append(xyz)
            pts2.append(uv)
            rows.append({"tcp_mm": xyz, "uv": uv.tolist()})
            v = img.copy()
            v[keep > 0] = (0.5 * v[keep > 0] + (0, 128, 0)).astype(np.uint8)
            cv2.circle(v, tuple(int(t) for t in uv), 6, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(OUT, f"p{k + 1:02d}.jpg"), v)
            web_page.push_image("三脚の校正（今の点）", v)
            print(f"  {k + 1}/{len(goals)}: 指先 x {xyz[0]:.1f} y {xyz[1]:.1f} 机から {xyz[2] - TABLE_Z:.0f}mm → 写真 ({uv[0]:.0f}, {uv[1]:.0f})", flush=True)
        c.moveL([p0[0], p0[1], p0[2]] + rot, SPEED, ACCEL)    # 始めの位置へ戻る
    finally:
        c.stopScript()
        c.disconnect()
        if hand is not None and open0 is not None:
            print(f"  指を元の開き量に戻した：{hand.move_mm(open0):.1f}mm", flush=True)
    if len(pts3) < 8:
        print(f"集まった組が {len(pts3)} しかないので、計算しない", flush=True)
        return
    P3 = np.array(pts3, np.float32)
    P2 = np.array(pts2, np.float32)
    W, H = 960, 540
    K0 = np.array([[500.0, 0, W / 2], [0, 500.0, H / 2], [0, 0, 1]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
             | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)                 # レンズの曲がり k1 と画像の中心も合わせる（開いた指で 12.1 → 7.5 画素）
    idx = np.arange(len(P3))
    for _ in range(len(P3) // 4):                              # いちばんずれた点が 8 画素より大きければ 1 つずつ外して計算し直す（指が隠れて台の角を拾った点など。10/20）
        err, K, dist, rv, tv = cv2.calibrateCamera([P3[idx]], [P2[idx]], (W, H), K0.copy(), np.zeros(5), flags=flags)
        e_all = np.linalg.norm(cv2.projectPoints(P3, rv[0], tv[0], K, dist)[0].reshape(-1, 2) - P2, axis=1)
        worst = idx[int(np.argmax(e_all[idx]))]
        if e_all[worst] <= 8.0:
            break
        idx = idx[idx != worst]
    err, K, dist, rv, tv = cv2.calibrateCamera([P3[idx]], [P2[idx]], (W, H), K0.copy(), np.zeros(5), flags=flags)
    proj = cv2.projectPoints(P3, rv[0], tv[0], K, dist)[0].reshape(-1, 2)
    e = np.linalg.norm(proj - P2, axis=1)[idx]
    excluded = [int(i) + 1 for i in sorted(set(range(len(P3))) - set(idx.tolist()))]
    print(f"外した点：{excluded}", flush=True)
    R = cv2.Rodrigues(rv[0])[0]
    cam = (-R.T @ tv[0]).ravel()
    # 写真の 1 画素が、マットの上で何 mm か（マットの真ん中あたり）
    mid = P3[:, :2].mean(axis=0)
    a_ = cv2.projectPoints(np.array([[mid[0], mid[1], TABLE_Z], [mid[0] + 10, mid[1], TABLE_Z]], np.float32), rv[0], tv[0], K, dist)[0].reshape(-1, 2)
    mm_px = 10.0 / max(np.linalg.norm(a_[1] - a_[0]), 1e-6)
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "image_size": [W, H], "K": K.tolist(), "dist": dist.ravel().tolist(),
           "rvec": rv[0].ravel().tolist(), "tvec": tv[0].ravel().tolist(), "cam_pos_mm": cam.tolist(),
           "reproj_px_mean": float(e.mean()), "reproj_px_max": float(e.max()), "mm_per_px_mid": float(mm_px), "points": rows, "excluded": excluded,
           "note": "画素は 960×540（tripod_latest.jpg）。目印は赤い指の先（いちばん下の赤い所）" + ("、指を閉じて" if a.closed else "") + "。根元の座標 mm。"}
    json.dump(out, open(os.path.join(HERE, "tripod_calib.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"結果：焦点距離 {K[0, 0]:.0f} 画素、カメラの位置 x {cam[0]:.0f} y {cam[1]:.0f} 机から {cam[2] - TABLE_Z:.0f}mm、"
          f"合わせのずれ 平均 {e.mean():.1f} 画素・最大 {e.max():.1f} 画素（マットの上で 1 画素 ≒ {mm_px:.1f}mm）", flush=True)
    print("書いた：real_ur/tripod_calib.json", flush=True)


if __name__ == "__main__":
    main()
