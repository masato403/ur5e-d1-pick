r"""三脚の校正で出した物の位置が、手首のカメラで出した位置と合うかを比べる（10/20）。腕は動かさない。
手首：今の写真の物 → locate.pixel_to_table（机の面）。三脚：三脚の写真の物 → tripod_calib.json の光線を机から OBJ_H の面と交える。
使い方:  .venv_ur\Scripts\python.exe real_ur\check_tripod_vs_wrist.py --ip 192.0.2.10   （cam_view.py --cam 2 --cam2 1 が動いていること）"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import measure as M
import tripod_view as TV
from locate import pixel_to_table, grab_now
from plan_hover import TABLE_Z
from calib_tripod import fresh_image

HERE = os.path.dirname(os.path.abspath(__file__))
OBJ_H = 6.0                                                    # 三脚の光線を交える高さ（机から mm。物の上の面の真ん中あたりの見当）


def tripod_to_table(uv, cal, h):
    K, dist = np.array(cal["K"]), np.array(cal["dist"])
    R = cv2.Rodrigues(np.array(cal["rvec"]))[0]
    o = -R.T @ np.array(cal["tvec"]).ravel()
    n = cv2.undistortPoints(np.array([[uv]], np.float64), K, dist).reshape(2)
    d = R.T @ np.array([n[0], n[1], 1.0])
    s = (TABLE_Z + h - o[2]) / d[2]
    return o + s * d


def tripod_blobs(img):
    """三脚の写真で、マット（暗い所）に囲まれた明るい・色のある塊。(中心の画素, 長さ)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark = (hsv[..., 2] < 90).astype(np.uint8) * 255
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
    big = 1 + int(np.argmax(st[1:, 4]))
    inside = np.zeros_like(dark)
    cv2.fillConvexPoly(inside, cv2.convexHull(cv2.findNonZero((lab == big).astype(np.uint8))), 255)
    inside = cv2.erode(inside, np.ones((15, 15), np.uint8))
    obj = ((inside > 0) & (dark == 0)).astype(np.uint8) * 255
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(obj)
    out = []
    for i in range(1, n):
        if 300 <= st[i, 4] <= 8000 and min(st[i, 2], st[i, 3]) >= 8:
            cnt = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
            out.append((cen[i], cnt, max(cv2.minAreaRect(cnt)[1])))
    return out


def blobs(img):
    out = []
    for cnt in M.find_objects(img)[0]:
        m = cv2.moments(cnt)
        if m["m00"] < 100:
            continue
        r = cv2.minAreaRect(cnt)
        out.append((np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]), cnt, max(r[1])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--move", action="store_true", help="腕を今の位置のまわり（前後左右 ±60/±40mm・上 +60mm）へ動かし、各所で手首と三脚を記録する")
    a = ap.parse_args()
    if a.move:
        return collect(a.ip)
    from rtde_receive import RTDEReceiveInterface
    tcp = RTDEReceiveInterface(a.ip).getActualTCPPose()
    cal = json.load(open(os.path.join(HERE, "tripod_calib.json"), encoding="utf-8"))
    wimg = grab_now()
    timg = fresh_image(time.time())
    W = [(pixel_to_table(c, tcp)[:2], L) for c, _, L in blobs(wimg)]
    T = [(tripod_to_table(c, cal, OBJ_H)[:2], c, L) for c, _, L in tripod_blobs(timg)]
    print(f"手首で {len(W)} 個、三脚で {len(T)} 個", flush=True)
    for i, (pw, L) in enumerate(W):
        print(f"  手首 {i + 1}: x {pw[0]:.1f} y {pw[1]:.1f}（長さ {L:.0f} 画素）", end="")
        if T:
            j = int(np.argmin([np.linalg.norm(pt - pw) for pt, _, _ in T]))
            pt = T[j][0]
            d = pt - pw
            print(f" ／ 三脚 x {pt[0]:.1f} y {pt[1]:.1f} → ずれ dx {d[0]:+.1f} dy {d[1]:+.1f}（{np.linalg.norm(d):.1f}mm）", flush=True)
        else:
            print()
    v = timg.copy()
    for pt, c, _ in T:
        cv2.circle(v, tuple(int(t) for t in c), 6, (0, 255, 255), 2)
    for pw, _ in W:                                            # 手首の位置を三脚の写真へ写す
        R = np.array(cal["rvec"]); t = np.array(cal["tvec"])
        uv = cv2.projectPoints(np.array([[pw[0], pw[1], TABLE_Z + OBJ_H]], np.float32), R, t, np.array(cal["K"]), np.array(cal["dist"]))[0].reshape(2)
        cv2.drawMarker(v, tuple(int(t_) for t_ in uv), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
    p = os.path.join(HERE, "logs_tripod_calib", "check_vs_wrist.jpg")
    cv2.imwrite(p, v)
    print("写真（黄の丸＝三脚で見つけた物、紫の十字＝手首の位置を写したもの）:", p)


OFFS = [(0, 0, 0), (60, 0, 0), (-60, 0, 0), (0, 40, 0), (0, -40, 0), (60, 40, 0), (-60, -40, 0), (0, 0, 60), (60, 0, 60), (-60, 0, 60), (0, 0, 0)]


def collect(ip):
    """腕を動かして、各所で手首の物の位置と三脚の物の位置の組を記録する（ハンドは動かさない、手首は回さない、0.05m/秒）。"""
    from rtde_receive import RTDEReceiveInterface
    from rtde_control import RTDEControlInterface
    r = RTDEReceiveInterface(ip)
    cal = json.load(open(os.path.join(HERE, "tripod_calib.json"), encoding="utf-8"))
    p0 = r.getActualTCPPose()
    c = RTDEControlInterface(ip)
    rows, diffs, diffs2 = [], [], []
    TV.FIX = np.zeros(2)                                         # track_real と同じ真ん中の取り方（TV.objects）で、直しを入れずに測る（10/22）
    try:
        for k, (dx, dy, dz) in enumerate(OFFS):
            goal = [p0[0] + dx / 1000, p0[1] + dy / 1000, p0[2] + dz / 1000] + list(p0[3:])
            if not c.isPoseWithinSafetyLimits(goal):
                print(f"  {k + 1}: 安全設定の範囲の外なので飛ばす", flush=True)
                continue
            c.moveL(goal, 0.05, 0.2)
            if r.getSafetyMode() != 1:
                print("  安全状態が普通でないので止めます", flush=True)
                break
            tcp = r.getActualTCPPose()
            wimg = grab_now(settle=0.3)
            timg = fresh_image(time.time())
            wb = [(c_.tolist(), float(L)) for c_, _, L in blobs(wimg)]
            tb = [(c_.tolist(), float(L)) for c_, _, L in tripod_blobs(timg)] if timg is not None else []
            tv = TV.objects(tcp, after=0, cut=True)[0] or []          # 腕の下の物も残す（腕の画素だけ消す）
            rows.append({"off_mm": [dx, dy, dz], "tcp": list(tcp), "wrist_px": wb, "tripod_px": tb, "tv_xy": [o["xy"].tolist() for o in tv]})
            T = [tripod_to_table(np.array(u), cal, OBJ_H)[:2] for u, _ in tb]
            line = []
            for u, L in wb:
                pw = pixel_to_table(np.array(u), tcp)[:2]
                if T:
                    d = min((pt - pw for pt in T), key=np.linalg.norm)
                    if np.linalg.norm(d) < 40:
                        diffs.append(d)
                        line.append(f"({d[0]:+.0f},{d[1]:+.0f})")
                if tv:
                    d2 = min((o["xy"] - pw for o in tv), key=np.linalg.norm)
                    if np.linalg.norm(d2) < 40:
                        diffs2.append(d2)
            print(f"  {k + 1}/{len(OFFS)}: ずらし {dx:+d},{dy:+d},{dz:+d}mm 手首 {len(wb)} 個・三脚 {len(tb)} 個、ずれ dx,dy mm {' '.join(line)}", flush=True)
        c.moveL(list(p0), 0.05, 0.2)
    finally:
        c.stopScript()
        c.disconnect()
    p = os.path.join(HERE, "logs_tripod_calib", time.strftime("vs_wrist_%Y%m%d_%H%M.json"))
    json.dump({"obj_h_mm": OBJ_H, "rows": rows}, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if diffs:
        D = np.array(diffs)
        print(f"組 {len(D)}：ずれの平均 dx {D[:, 0].mean():+.1f} dy {D[:, 1].mean():+.1f}mm、ばらつき（標準偏差） {D[:, 0].std():.1f}・{D[:, 1].std():.1f}mm", flush=True)
    if diffs2:
        D = np.array(diffs2)
        print(f"track_real の取り方（TV.objects）で組 {len(D)}：ずれの平均 dx {D[:, 0].mean():+.1f} dy {D[:, 1].mean():+.1f}mm、"
              f"ばらつき {D[:, 0].std():.1f}・{D[:, 1].std():.1f}mm → 直し TRI_DX={-D[:, 0].mean():.1f} TRI_DY={-D[:, 1].mean():.1f}", flush=True)
    print("記録:", p)


if __name__ == "__main__":
    main()
