"""円柱（ドライバーの持ち手・ペン）の幅が小さく出る原因を調べる（窓は出さない）。物ごとに、輪郭をいろいろな高さの面に写した時の幅を並べる。
  ・今の方法：上の面の高さ h と机 0 に写した 2 つの凸包の重なり（箱の形なら正しい）
  ・h/2 に写した凸包（円柱がいちばん太い高さ ＝ 指で挟む高さ）
使い方:  python robot/diag_cyl.py [回数]
"""
import os
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_real as T  # noqa: E402
import sim_real as S  # noqa: E402
from run_pick import IK  # noqa: E402

import height_by_motion as HM  # noqa: E402
import pick_any as PA  # noqa: E402


def width(poly):
    return sorted(cv2.minAreaRect(np.asarray(poly, np.float32))[1])[0]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    T.patch_real_code()
    env = S.RealLikeEnv()
    ik = IK(env)
    rng = np.random.default_rng(11)
    obs = np.array(S.OBS_XY)
    rows = {t: [] for t in S.TOOLS}
    for k in range(n):
        env.reset(rng)
        for i, o in enumerate(env.objects):
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        T.go_to(env, ik, obs)
        placed = T.place_tools(env, rng)
        imgA, tcpA, others = T.shots(env, ik, obs)
        for f in PA.survey(imgA, tcpA, others):
            t = min(placed, key=lambda u: np.linalg.norm(placed[u][0] * 1000 - f["center"][:2]))
            pts = f["contour"].reshape(-1, 2).astype(np.float64)
            h = f["h"]
            th, tw, _ = T.TRUE[t]
            w_now = f["size"][0]
            w_mid = width(cv2.convexHull(HM.on_plane(pts, tcpA, h / 2)[:, :2].astype(np.float32)))
            w_true_h = width(PA.footprint(pts, tcpA, th))
            rows[t].append((w_now, w_mid, w_true_h, h))
            print(f"{k}: {T.NAME[t]}（本当 太さ {tw:.0f}・高さ {th:.0f}）測った高さ {h:.1f}：今の幅 {w_now:.1f}、h/2 に写す {w_mid:.1f}、"
                  f"本当の高さで今の方法 {w_true_h:.1f}")
    for t, r in rows.items():
        if r:
            a = np.array(r)
            print(f"---- {T.NAME[t]}：今の幅 {a[:, 0].mean():.1f}、h/2 に写す {a[:, 1].mean():.1f}、本当の高さで今の方法 {a[:, 2].mean():.1f}（本当 {T.TRUE[t][1]:.0f}）")


if __name__ == "__main__":
    main()
