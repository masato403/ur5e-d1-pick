"""高さの決め方を比べる（窓は出さない）。消しゴム・ペンについて、本当の高さとの差を出す。
  best：右・左のうち合い具合の良い方（今のやり方）
  sum ：右と左の合い具合を足して、いちばん良い高さ
  grad：明るさの変わり目（輪郭の強さ）で比べ、右と左を足す
使い方:  python robot/diag_combine.py [場面の数]
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_real as T  # noqa: E402
import test_sim_grasp as G  # noqa: E402
import sim_real as S  # noqa: E402
import sim_rtde  # noqa: E402
from sim_rtde import SimRobot  # noqa: E402
import height_by_motion as HM  # noqa: E402
import pick_any as PA  # noqa: E402
import measure as M  # noqa: E402

TRUE_H = {"eraser": S.ERASER_MM[2], "r_pen": S.PEN_MM[1]}


def grad_img(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.5)
    m = np.clip(m / 4.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    T.patch_real_code()
    env = S.RealLikeEnv()
    G.patch_grasp(env)
    G.patch_capture(env)
    robot = SimRobot(env)
    HM.COARSE = HM.FINE                       # 比べるため、全部の高さを 0.5mm おきに比べる
    rng = np.random.default_rng(11)
    obs = np.array(S.OBS_XY)
    err = {"best": [], "sum": [], "grad": []}
    for k in range(n):
        env.reset(rng)
        for i, o in enumerate(env.objects):
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        G.go_to(env, robot.ik, obs)
        T.place_tools(env, rng)
        for _ in range(20):
            sim_rtde.frame(env)
        imgA, tcpA, others = HM.capture_two(None, c=robot, r=robot)
        gA = grad_img(imgA)
        gO = [(grad_img(im), t) for im, t in others]
        objs, _ = M.find_objects(imgA)
        for cnt in objs:
            mask = np.zeros(imgA.shape[:2], np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            c = HM.on_plane(np.array([cnt.reshape(-1, 2).mean(0)]), tcpA, 0)[0][:2]
            t = min(S.TOOLS, key=lambda u: np.linalg.norm(env.d.qpos[env.tool_q[u]:env.tool_q[u] + 2] * 1000 - c))
            if t not in TRUE_H or np.linalg.norm(env.d.qpos[env.tool_q[t]:env.tool_q[t] + 2] * 1000 - c) > 30:
                continue
            mid = PA.band(mask)
            sc = [HM.best_height(imgA, tcpA, im, tc, mid) for im, tc in others]
            hs = sc[0][1]
            hb = max(sc, key=lambda r: r[2].max())[0]
            ok = [r[2] for r in sc if r[2].max() > -0.99]
            ssum = np.sum([np.where(s > -0.99, s, np.nan) for s in ok], axis=0)
            hsum = float(hs[np.nanargmax(ssum)])
            gs = [HM.best_height(gA, tcpA, im, tc, mid)[2] for im, tc in gO]
            gsum = np.sum([np.where(s > -0.99, s, np.nan) for s in gs if s.max() > -0.99], axis=0)
            hgrad = float(hs[np.nanargmax(gsum)])
            tr = TRUE_H[t]
            for key, v in (("best", hb), ("sum", hsum), ("grad", hgrad)):
                err[key].append(v - tr)
            print(f"{k}: {T.NAME[t]} 本当 {tr:.0f}mm → 今 {hb:5.1f}、足す {hsum:5.1f}、輪郭 {hgrad:5.1f}")
    for key, e in err.items():
        e = np.abs(e)
        print(f"---- {key}: 平均の差 {e.mean():.1f}mm、最大 {e.max():.1f}mm、3mm 以内 {np.mean(e <= 3):.0%}（{len(e)} 個）")


if __name__ == "__main__":
    main()
