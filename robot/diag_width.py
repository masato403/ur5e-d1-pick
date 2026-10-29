"""消しゴムの幅が大きく出る原因を調べる（窓は出さない）。シミュレーターが出す「本当の物の画素」（segmentation）と、
実機用のプログラムが切り取った輪郭（measure.find_objects）を比べ、どこが余分かを画像と数字で出す。
使い方:  python robot/diag_width.py [回数]   → robot/logs/width_<番号>.jpg（緑：本当の物、赤：プログラムの輪郭）
"""
import math
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

import measure as M  # noqa: E402
import pick_any as PA  # noqa: E402
import height_by_motion as HM  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    T.patch_real_code()
    env = S.RealLikeEnv()
    ik = IK(env)
    seg = mujoco.Renderer(env.m, S.IMG_H, S.IMG_W)
    seg.enable_segmentation_rendering()
    bid = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_BODY, "eraser")
    rng = np.random.default_rng(7)
    obs = np.array(S.OBS_XY)
    for k in range(n):
        env.reset(rng)
        for i, o in enumerate(env.objects):
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        T.go_to(env, ik, obs)
        for j, t in enumerate(S.TOOLS):
            env.set_tool(t, np.array([1.5, 0.5 + 0.1 * j]), 0.0)
        for _ in range(500):
            xy = T.view_center(env) + rng.uniform([-0.05, -0.04], [0.05, 0.04])
            yaw = rng.uniform(0, math.pi)
            if T.in_view(env, "eraser", xy, yaw) and T.away_from_box("eraser", xy, yaw):
                break
        env.set_tool("eraser", xy, yaw)
        mujoco.mj_forward(env.m, env.d)
        img = env.render_real()
        seg.update_scene(env.d, camera="wrist_real")
        s = seg.render()
        gids = s[..., 0]
        truth = np.isin(gids, [g for g in range(env.m.ngeom) if env.m.geom_bodyid[g] == bid]) & (s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        objs, _ = M.find_objects(img)
        if not objs:
            print(f"{k}: 見つからない")
            continue
        cnt = min(objs, key=lambda c: np.linalg.norm(c.reshape(-1, 2).mean(0) - np.argwhere(truth)[:, ::-1].mean(0)))
        mask = np.zeros(truth.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        t8 = truth.astype(np.uint8) * 255
        extra, miss = (mask > 0) & ~truth, truth & ~(mask > 0)
        tc = cv2.findContours(t8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        tcnt = max(tc, key=cv2.contourArea)
        _, (tw, tl), _ = cv2.minAreaRect(tcnt)
        _, (mw, ml), _ = cv2.minAreaRect(cnt)
        # 本当の輪郭で測ったら、幅はいくつになるか（輪郭の切り取りの誤差と、測り方の誤差を分ける）
        foot_t = PA.footprint(tcnt.reshape(-1, 2).astype(np.float64), env.tcp_like(), 12.0)
        foot_m = PA.footprint(cnt.reshape(-1, 2).astype(np.float64), env.tcp_like(), 12.0)
        wt = sorted(cv2.minAreaRect(foot_t.astype(np.float32))[1])
        wm = sorted(cv2.minAreaRect(foot_m.astype(np.float32))[1])
        print(f"{k}: 画素の数 本当 {truth.sum()}、輪郭 {int((mask > 0).sum())}（余分 {int(extra.sum())}、足りない {int(miss.sum())}）、"
              f"画像の上の幅 本当 {min(tw, tl):.0f}px・輪郭 {min(mw, ml):.0f}px")
        print(f"    机の上の大きさ（高さ 12mm として）：本当の輪郭から {wt[0]:.1f} × {wt[1]:.1f}mm、プログラムの輪郭から {wm[0]:.1f} × {wm[1]:.1f}mm（本当 20 × 52）")
        dbg = img.copy()
        dbg[truth] = (0.5 * dbg[truth] + [0, 120, 0]).astype(np.uint8)
        cv2.drawContours(dbg, [cnt], -1, (0, 0, 255), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        crop = dbg[max(0, y - 40):y + h + 40, max(0, x - 40):x + w + 40]
        cv2.imwrite(os.path.join(HERE, "logs", f"width_{k}.jpg"), cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    main()
