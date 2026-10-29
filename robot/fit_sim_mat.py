"""シミュレーターのマットの絵を、実機の写真のマットの明るさ・色に合わせる（10/19 夜）。
実機の 10/16 夕方の写真では、マットの地が暗いこげ茶紫（明るさ 61〜126、場所でむらがある）で、暗い角でペンを見落とした（49 組中 24 組）。
シミュレーターのマットは一様に明るい薄紫で、これを再現できなかった。
  1) 実機の地図（map_mat_light.py：手先から見た位置ごとの、地の色・方眼の線の色）を読む
  2) シミュレーターで見張りの位置から 3 枚撮り、同じ測り方で地図を作る
  3) マスごとに 実機 ÷ シミュレーター の比を、マットの絵の地と線の画素に掛ける。2)〜3) を数回くり返す
結果：robot/mat_texture_real.png（sim_real.py が SIM_MAT=real の時に使う）
使い方:  python robot/fit_sim_mat.py [くり返す回数 5]
"""
import os
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_real as TR  # noqa: E402（シミュレーターの校正への差し替え・見張りの位置へ動かす）
import sim_real as S  # noqa: E402
import map_mat_light as ML  # noqa: E402
import height_by_motion as HM  # noqa: E402
from run_pick import IK  # noqa: E402

OUT = os.path.join(HERE, "mat_texture_real.png")
SMOOTH = 2.0   # 明るさの比をなめらかにする幅（マス。1 マス 10mm）
BG_MAX = 165.0   # マットの地の画素の明るさの上限


def fill_nearest(g):
    """測れなかったマス（nan）を、いちばん近い測れたマスの値で埋める。"""
    ok = np.isfinite(g[..., 0])
    if ok.all():
        return g
    ys, xs = np.nonzero(ok)
    by, bx = np.nonzero(~ok)
    d = (by[:, None] - ys[None]) ** 2 + (bx[:, None] - xs[None]) ** 2
    k = d.argmin(1)
    out = g.copy()
    out[by, bx] = g[ys[k], xs[k]]
    return out


def real_target(day="20261016"):
    d = np.load(os.path.join(HERE, "logs", f"mat_light_{day}.npz"))
    bg, line = d["bg"].copy(), d["line"].copy()
    hb = ML.hsv_of(bg)
    mat_like = (hb[..., 0] >= 130) | (hb[..., 0] <= 30)          # マットの色合い（赤紫〜こげ茶）。床の青などは外す
    # マットの外の明るい枠（アルミ）は色が薄く、色合いの条件を通ってしまい、シミュレーターの指の近くが白くなった（10/19）。
    # 実機のマットの地は明るさ 46〜125 なので、150 より明るいマスは外す
    mat_like &= hb[..., 2] <= 150
    bg[~mat_like] = np.nan
    line[~mat_like] = np.nan
    return fill_nearest(bg), fill_nearest(line), np.isfinite(bg[..., 0])


def main():
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    if S.MAT_LOOK == "real" and n_iter > 1:
        sys.exit("SIM_MAT=real では、できた絵を確かめるだけ（回数 1）にする")
    TR.patch_real_code()
    env = S.RealLikeEnv()
    ik = IK(env)
    env.reset(np.random.default_rng(1))
    for i, o in enumerate(env.objects):
        env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
    for i, t in enumerate(S.TOOLS):                               # 道具もマットの外へ
        env.set_tool(t, np.array([1.0 + 0.1 * i, -0.8]), 0.0)
    mujoco.mj_forward(env.m, env.d)
    m = env.m
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "cutmat")
    H, W, C = int(m.tex_height[tid]), int(m.tex_width[tid]), int(m.tex_nchannel[tid])
    adr = int(m.tex_adr[tid])
    tex0 = m.tex_data[adr:adr + H * W * C].reshape(H, W, C).copy()       # RGB
    plain = cv2.cvtColor(cv2.imread(os.path.join(HERE, "mat_texture.png")), cv2.COLOR_BGR2RGB)
    if plain.shape[:2] != (H, W):
        plain = cv2.resize(plain, (W, H), interpolation=cv2.INTER_NEAREST)
    # 方眼の線の画素は、前の一様な絵（mat_texture.png）から決める（SIM_MAT=real で、合わせた絵を読んだ時も同じ所になる）
    flip = not np.array_equal(plain[0], tex0[0, :, :3]) and np.array_equal(plain[-1], tex0[0, :, :3])
    is_line = np.abs((plain[::-1] if flip else plain).astype(int) - [215, 185, 150]).sum(-1) < 30
    print(f"マットの絵 {W}×{H}（色 {C}）、線の画素 {is_line.mean():.0%}")
    mx0, my0 = (S.MAT_CENTER[0] - S.MAT_MM[0] / 2000) * 1000, (S.MAT_CENTER[1] - S.MAT_MM[1] / 2000) * 1000
    box = np.array(S.BOX_XY) * 1000
    half = (np.array(S.BOX_OUT) * 1000) / 2 + 15
    obs = np.array(S.OBS_XY)

    def shots():
        out = []
        for d in (0.0, 1.0, -1.0):
            TR.go_to(env, ik, obs)
            if d:
                R = env.d.site_xmat[env.site].reshape(3, 3)
                TR.go_to(env, ik, obs + d * R[:2, 1] * TR.SHIFT)
            out.append((env.render_real(), env.tcp_like()))
        TR.go_to(env, ik, obs)
        return out

    def pick(img, tcp):
        """シミュレーターで数える画素：マットの上で、箱とハンドを除く所（位置が分かっているので色で決めない）。"""
        ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        P = HM.on_plane(np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64), tcp, 0.0)[:, :2]
        on = (np.abs(P[:, 0] - S.MAT_CENTER[0] * 1000) < S.MAT_MM[0] / 2 - 3) & (np.abs(P[:, 1] - S.MAT_CENTER[1] * 1000) < S.MAT_MM[1] / 2 - 3)
        inbox = np.all(np.abs(P - box) < half, axis=1)
        ok = (on & ~inbox).reshape(img.shape[:2])
        ok[int(img.shape[0] * 0.62):] = False
        return ok

    # 絵の向きを決める：方眼は 10mm ごとにくり返すので、線の重なりでは反転を見分けられず、左右が逆になった（10/19）。
    # 絵の赤を「左から右へ」、緑を「上から下へ」明るくした傾きの絵に一時的に差し替えて撮り、4 通りの向きのどれで一番よく合うかで決める
    ramp = np.zeros((H, W, C), np.uint8)
    ramp[..., 0] = np.linspace(20, 235, W)[None, :]
    ramp[..., 1] = np.linspace(20, 235, H)[:, None]
    m.tex_data[adr:adr + H * W * C] = ramp.ravel()
    mujoco.mjr_uploadTexture(m, env.rr._mjr_context, tid)
    img, tcp = shots()[0]
    m.tex_data[adr:adr + H * W * C] = tex0.ravel()
    mujoco.mjr_uploadTexture(m, env.rr._mjr_context, tid)
    rr, cc = np.mgrid[0:H:8, 0:W:8]
    rr, cc = rr.ravel(), cc.ravel()
    best = None
    for fx in (False, True):
        for fy in (False, True):
            x = mx0 + (((W - 1 - cc) if fx else cc) + 0.5) / W * S.MAT_MM[0]
            y = my0 + (((H - 1 - rr) if fy else rr) + 0.5) / H * S.MAT_MM[1]
            px = HM.project_fast(np.column_stack([x, y, np.full(len(x), S.MAT_Z * 1000)]), tcp)
            ok = (px[:, 0] >= 0) & (px[:, 0] < img.shape[1] - 1) & (px[:, 1] >= 0) & (px[:, 1] < img.shape[0] * 0.6)
            got = img[px[ok, 1].astype(int), px[ok, 0].astype(int)].astype(float)    # BGR
            score = np.corrcoef(got[:, 2], cc[ok])[0, 1] + np.corrcoef(got[:, 1], rr[ok])[0, 1]
            if best is None or score > best[0]:
                best = (score, fx, fy)
    _, fx, fy = best
    print(f"絵の向き：左右の反転 {fx}、上下の反転 {fy}（傾きの合い方 {best[0]:.2f}、最大 2）")
    R_, C_ = np.mgrid[0:H, 0:W]
    X = mx0 + (((W - 1 - C_) if fx else C_) + 0.5) / W * S.MAT_MM[0]
    Y = my0 + (((H - 1 - R_) if fy else R_) + 0.5) / H * S.MAT_MM[1]
    T = ML.tool_xy(np.column_stack([X.ravel(), Y.ravel()]), tcp)   # 見張りの位置（A）の手先から見た位置
    ti, tj = ML.cell_index(T)
    ny, nx = ML.grid_shape()
    ti, tj = np.clip(ti, 0, ny - 1).reshape(H, W), np.clip(tj, 0, nx - 1).reshape(H, W)
    tgt_bg, tgt_line, measured = real_target()
    tex = tex0[..., :3].astype(np.float64)
    for it in range(n_iter):
        sim_bg, sim_line = ML.summarize(ML.collect(shots(), pick))
        both = np.isfinite(sim_bg[..., 0]) & measured
        e_bg = np.abs(sim_bg - tgt_bg)[both].mean(0)
        e_line = np.abs(sim_line - tgt_line)[both].mean(0)
        hs = ML.hsv_of(np.where(both[..., None], sim_bg, np.nan))
        hr = ML.hsv_of(np.where(both[..., None], tgt_bg, np.nan))
        print(f"{it}: 比べたマス {both.sum()}、地の色の差（B,G,R）{e_bg.round(1)}、線 {e_line.round(1)}、"
              f"地の色合い シミュ {np.nanmedian(hs[..., 0]):.0f}・濃さ {np.nanmedian(hs[..., 1]):.0f} 実機 {np.nanmedian(hr[..., 0]):.0f}・{np.nanmedian(hr[..., 1]):.0f}、"
              f"地の明るさ シミュ {np.nanmedian(hs[..., 2]):.0f}（{np.nanpercentile(hs[..., 2], 10):.0f}〜{np.nanpercentile(hs[..., 2], 90):.0f}）"
              f" 実機 {np.nanmedian(hr[..., 2]):.0f}（{np.nanpercentile(hr[..., 2], 10):.0f}〜{np.nanpercentile(hr[..., 2], 90):.0f}）", flush=True)
        if it == n_iter - 1:
            break
        # マスごとに B・G・R を別々に掛けると、マスごとに色合いが変わって、つぎはぎ（青・緑・黄色のマス）になった（10/19）。
        # 色合いは全体で 1 つ（地・線それぞれ、全部のマスの比の中央値を B・G・R ごとに）にし、場所ごとには明るさの比（1 つの数）だけを、
        # なめらかにして掛ける
        def gains(tgt, sim):
            ok = both & np.isfinite(sim[..., 0])
            col = np.median(tgt[ok] / np.maximum(sim[ok], 3), axis=0)              # B・G・R ごとの全体の比
            lum = np.where(ok, tgt.sum(-1) / np.maximum((sim * col).sum(-1), 3), np.nan)
            lum = fill_nearest(lum[..., None])[..., 0]
            lum = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), SMOOTH)
            # シミュレーターの腕・指の影の所は暗く写り、比が大きくなって絵が真っ白になった（10/19）。1 回に掛ける比を抑える
            lum = np.clip(lum, 0.6, 1.6)
            return col[::-1], lum                                                  # 色は RGB の順に
        c_bg, l_bg = gains(tgt_bg, sim_bg)
        c_ln, l_ln = gains(tgt_line, sim_line)
        ratio = np.where(is_line[..., None], c_ln * l_ln[ti, tj][..., None], c_bg * l_bg[ti, tj][..., None])
        tex = np.clip(tex * np.clip(ratio, 0.3, 3.0), 1, 255)
        # 地の画素は、実機のマットの地より明るくしない（実機の地の明るさの最大 125 の 1.3 倍）
        bright = tex.max(-1, keepdims=True)
        tex = np.where(~is_line[..., None] & (bright > BG_MAX), tex * BG_MAX / np.maximum(bright, 1), tex)
        full = tex if C == 3 else np.concatenate([tex, np.full((H, W, C - 3), 255.0)], -1)
        m.tex_data[adr:adr + H * W * C] = full.astype(np.uint8).ravel()
        mujoco.mjr_uploadTexture(m, env.rr._mjr_context, tid)
    cv2.imwrite(OUT, cv2.cvtColor(tex.astype(np.uint8), cv2.COLOR_RGB2BGR))   # 絵の画素の並びのまま（mat_texture と同じ向き）
    cv2.imwrite(os.path.join(HERE, "logs", "sim_mat_real_A.jpg"), cv2.resize(shots()[0][0], (960, 540)))
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
