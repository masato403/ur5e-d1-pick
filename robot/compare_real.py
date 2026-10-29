"""実機の手首カメラの写真（real_ur/cam_shots/height_A.jpg、10/15）と、シミュレーターの手首カメラの画像を重ねて比べる（窓は出さない）。
実機の写真はレンズの曲がりを校正の値で取り、シミュレーターと同じ焦点距離・真ん中の画像に直してから重ねる。
ハンド（本体・指）はカメラと一緒に動くので、腕の姿勢によらず同じ所に写る → 寸法の推定が合っているかを見られる。
使い方:  python robot/compare_real.py [写真]   → robot/logs/compare_real.jpg（左：実機、真ん中：シミュレーター、右：重ね）
"""
import os
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_real as S  # noqa: E402
import test_sim_real as T  # noqa: E402
from run_pick import IK  # noqa: E402

REAL = os.path.join(HERE, "..", "real_ur", "cam_shots", "height_A.jpg")


def undistort_real(img):
    K = np.array(S.CAL["K"], float)
    dist = np.array(S.CAL["dist"], float)
    sx, sy = img.shape[1] / S.IMG_W, img.shape[0] / S.IMG_H
    K = K.copy()
    K[0] *= sx
    K[1] *= sy
    Knew = S.K_SIM.copy()
    Knew[0] *= sx
    Knew[1] *= sy
    m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, Knew, (img.shape[1], img.shape[0]), cv2.CV_32FC1)
    return cv2.remap(img, m1, m2, cv2.INTER_LINEAR)


def hand_px(img):
    """手（赤い指）と、画面の下の本体（暗い所）の画素。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 120, 60), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 60), (180, 255, 255))
    return red


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else REAL
    real = cv2.resize(undistort_real(cv2.imread(path)), (S.IMG_W, S.IMG_H))
    T.patch_real_code()
    env = S.RealLikeEnv()
    ik = IK(env)
    for i, o in enumerate(env.objects):
        env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
    for j, t in enumerate(S.TOOLS):
        env.set_tool(t, np.array([1.5, 0.5 + 0.1 * j]), 0.0)
    T.go_to(env, ik, np.array(S.OBS_XY))
    mujoco.mj_forward(env.m, env.d)
    sim = env.render_real()
    rr, rs = hand_px(real), hand_px(sim)
    over = np.zeros_like(real)
    over[..., 2] = rr                     # 赤：実機の指
    over[..., 1] = rs                     # 緑：シミュレーションの指（重なると黄色）
    roi = np.zeros_like(rr)
    roi[350:, 300:1750] = 1                # 左の棚の赤い物を外し、指のあたりだけで比べる
    rr, rs = rr * roi, rs * roi
    both = (rr > 0) & (rs > 0)
    iou = both.sum() / max(1, ((rr > 0) | (rs > 0)).sum())
    for name, m in (("実機", rr), ("シミュレーション", rs)):
        ys, xs = np.nonzero(m)
        print(f"{name}の指：画素 {len(xs)}、上の端 y {ys.min() if len(ys) else -1}、左右の端 x {xs.min() if len(xs) else -1}〜{xs.max() if len(xs) else -1}")
    print(f"指の重なり（IoU）{iou:.2f}")
    out = np.hstack([cv2.resize(real, (640, 360)), cv2.resize(sim, (640, 360)), cv2.resize(cv2.addWeighted(real, 0.5, over, 0.8, 0), (640, 360))])
    cv2.imwrite(os.path.join(HERE, "logs", os.environ.get("CMP_OUT", "compare_real.jpg")), out)


if __name__ == "__main__":
    main()
