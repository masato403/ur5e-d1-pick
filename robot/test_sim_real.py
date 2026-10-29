"""実機に近づけたシミュレーター（sim_real.py）で、実機用のプログラム（real_ur の measure / height_by_motion / pick_any）を
そのまま動かし、測った高さ・大きさ・位置を、シミュレーションの本当の値と比べる（窓は出さない）。

  実機用のプログラムが使う校正の値（カメラの中の値・取り付け・マットの位置）を、シミュレーションの本当の値に差し替える。
  写真（10/15）のペン・消しゴム・ドライバーを、カメラの画面の真ん中あたりに、重ならないように置く。
使い方:  python robot/test_sim_real.py [回数]
"""
import math
import os
import sys
import types

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "real_ur"))
for name in ("serial", "serial.tools", "serial.tools.list_ports"):     # シミュレーションではハンドの通信を使わない
    sys.modules.setdefault(name, types.ModuleType(name))
import sim_real as S  # noqa: E402
from run_grip import rot_yaw  # noqa: E402
from run_pick import IK  # noqa: E402

import locate as L  # noqa: E402
import measure as M  # noqa: E402
import height_by_motion as HM  # noqa: E402
import pick_any as PA  # noqa: E402

OBS_H, SHIFT = 0.150, 0.035
TRUE = {"eraser": (S.ERASER_MM[2], S.ERASER_MM[1], S.ERASER_MM[0]),                 # 高さ・幅・長さ（mm）
        "r_pen": (S.PEN_MM[1], S.PEN_MM[1], S.PEN_MM[0]),
        "r_driver": (S.DRIVER_MM[1], S.DRIVER_MM[1], S.DRIVER_MM[0] + S.DRIVER_MM[2]),
        "r_rod": (S.ROD_MM[1], S.ROD_MM[1], S.ROD_MM[0])}
NAME = {"eraser": "消しゴム", "r_pen": "ペン", "r_driver": "ドライバー", "r_rod": "銀の棒"}
GAP = 0.010                                  # 物と物の間をあける（m）


def footprint(t, xy, yaw):
    """道具の、上から見た形を、軸に沿った点の並びと半径で表す（重なりを調べる用）。"""
    _, w, l = TRUE[t]
    a = np.array([math.cos(yaw), math.sin(yaw)])
    pts = [xy + a * u for u in np.linspace(-l / 2000 + w / 2000, l / 2000 - w / 2000, 12)]
    return np.array(pts), w / 2000


def apart(f1, f2):
    d = np.min(np.linalg.norm(f1[0][:, None] - f2[0][None], axis=2))
    return d > f1[1] + f2[1] + GAP


def patch_real_code():
    """実機用のプログラムの校正の値を、シミュレーションの本当の値にする（レンズの曲がりは 0）。"""
    t, _, R_cv = S.cam_in_pinch()
    G0 = np.array([(S.MAT_CENTER[0] - S.MAT_MM[0] / 2000) * 1000, (S.MAT_CENTER[1] - S.MAT_MM[1] / 2000) * 1000, S.MAT_Z * 1000])
    GR = np.eye(3)
    for mod in (L, M):
        mod.K, mod.DIST = S.K_SIM.copy(), np.zeros(5)
        mod.RC, mod.TC, mod.G0 = R_cv, t * 1000.0, G0
    L.GN, HM.GN, M.GR = GR[:, 2], GR[:, 2], GR
    L.CAL = dict(L.CAL, grid_R_base=GR.tolist())
    # 撮った写真を実機の cam_shots に上書きしない（10/16：昨日の実機の height_A/B/C.jpg をシミュレーターの写真で上書きして消した）
    HM.SHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "sim_shots")
    os.makedirs(HM.SHOT_DIR, exist_ok=True)


def go_to(env, ik, xy, yaw=0.0):
    q = ik.solve(env.d.qpos[:6].copy(), np.array([xy[0], xy[1], S.MAT_Z + OBS_H]), rot_yaw(yaw), iters=60)
    env.d.qpos[:6] = q
    mujoco.mj_forward(env.m, env.d)


def view_center(env):
    """画面の真ん中に写る、机の上の点（m）。"""
    p = L.pixel_to_table(np.array([S.IMG_W / 2, S.IMG_H * 0.4]), env.tcp_like())
    return p[:2] / 1000.0


def shots(env, ik, obs):
    """実機と同じ撮り方：A（見張り）、B（横へ +SHIFT）、C（横へ -SHIFT）。"""
    go_to(env, ik, obs)
    imgA, tcpA = env.render_real(), env.tcp_like()
    R = env.d.site_xmat[env.site].reshape(3, 3)
    others = []
    for sgn in (1, -1):
        go_to(env, ik, obs + sgn * R[:2, 1] * SHIFT)
        others.append((env.render_real(), env.tcp_like()))
    go_to(env, ik, obs)
    return imgA, tcpA, others


VIEW_BOX = (0.06, 0.94, 0.04, 0.50)          # 物を置いてよい画面の範囲（横・縦の割合）。下の方は ESG1 の本体と指に隠れる（10/16）


def in_view(env, t, xy, yaw):
    """道具 t を xy・yaw に置いた時、上から見た形が全部、見張りの位置の画面の VIEW_BOX に入るか。"""
    pts, r = footprint(t, xy, yaw)
    ring = np.concatenate([pts + r * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)])
    P = np.column_stack([ring * 1000, np.full(len(ring), (S.MAT_Z * 1000) + TRUE[t][0])])
    uv = M.project_table(P, env.tcp_like())
    x0, x1, y0, y1 = VIEW_BOX
    return bool(np.all((uv[:, 0] > x0 * S.IMG_W) & (uv[:, 0] < x1 * S.IMG_W) & (uv[:, 1] > y0 * S.IMG_H) & (uv[:, 1] < y1 * S.IMG_H)))


BOX_GAP = 0.045               # 箱から道具を離す距離（m）。ハンドの本体（60×46mm）の中心から角まで 約 38mm。
# 15mm では、箱のそばの物をつかみに下りる時に、本体や指の付け根が箱の壁に当たった（10/16）。つかむ前の確かめは、まだ無い


def away_from_box(t, xy, yaw, gap=BOX_GAP):
    """道具が箱（とそのまわり gap m）に重ならないか。"""
    pts, r = footprint(t, xy, yaw)
    half = np.array(S.BOX_OUT) / 2 + gap + r                     # 箱の上から見た外形（トレーは縁の折り返しまで）から離す
    return bool(np.all(np.any(np.abs(pts - np.array(S.BOX_XY)) > half, axis=1)))


MAT_EDGE = 0.020              # 道具をマットの端からこれ（m）以上内側に置く。はみ出たペンを、マットの外の模様の無い所で高さを誤り空振りした（10/16）


def on_mat(t, xy, yaw):
    """道具 t を xy・yaw に置いた時、上から見た形が全部、マットの端から MAT_EDGE 以上内側に入るか。"""
    pts, r = footprint(t, xy, yaw)
    hx, hy = S.MAT_MM[0] / 2000 - MAT_EDGE - r, S.MAT_MM[1] / 2000 - MAT_EDGE - r
    return bool(np.all(np.abs(pts[:, 0] - S.MAT_CENTER[0]) < hx) and np.all(np.abs(pts[:, 1] - S.MAT_CENTER[1]) < hy))


def place_tools(env, rng):
    """道具を、画面のよく見える所に、重ならないように置く（腕は見張りの位置にある前提）。{道具: (xy, yaw)} を返す。"""
    vc = view_center(env)
    placed = {}
    for j, t in enumerate(S.TOOLS):
        for _ in range(2000):
            xy = vc + rng.uniform([-0.09, -0.07], [0.09, 0.07])
            yaw = rng.uniform(0, math.pi)
            if in_view(env, t, xy, yaw) and on_mat(t, xy, yaw) and away_from_box(t, xy, yaw) and all(apart(footprint(t, xy, yaw), footprint(u, p, y)) for u, (p, y) in placed.items()):
                env.set_tool(t, xy, yaw)
                placed[t] = (xy, yaw)
                break
        else:
            # 置ける所が無い：無理に置くと、物どうしや箱にめり込んで立ったり沈んだりした（10/16）ので、この場面には出さない
            env.set_tool(t, np.array([1.5, 0.5 + 0.1 * j]), 0.0)
            print(f"   （{NAME[t]} は置ける所が無いので、この場面には出さない）")
    mujoco.mj_forward(env.m, env.d)
    return placed


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    patch_real_code()
    env = S.RealLikeEnv()
    ik = IK(env)
    rng = np.random.default_rng(1)
    rows = []
    for k in range(n):
        env.reset(rng)
        for i, o in enumerate(env.objects):                      # 今までの場面の物（ねじなど）はマットの外へ
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        obs = np.array(S.OBS_XY)
        go_to(env, ik, obs)
        placed = place_tools(env, rng)
        mujoco.mj_forward(env.m, env.d)
        imgA, tcpA, others = shots(env, ik, obs)
        if k == 0:
            cv2.imwrite(os.path.join(HERE, "logs", "sim_real_A.jpg"), cv2.resize(imgA, (960, 540)))
        found = PA.survey(imgA, tcpA, others)
        for t, (xy, yaw) in placed.items():
            if not found:
                rows.append((k, t, None))
                continue
            f = min(found, key=lambda f: np.linalg.norm(f["center"][:2] - xy * 1000))
            dc = np.linalg.norm(f["center"][:2] - xy * 1000)
            if dc > 40:
                rows.append((k, t, None))
                print(f"{k}: {NAME[t]} 見つからない（いちばん近い物まで {dc:.0f}mm）")
                continue
            dang = math.degrees(math.acos(min(1.0, abs(float(f["longv"][:2] @ np.array([math.cos(yaw), math.sin(yaw)]))))))
            th, tw, tl = TRUE[t]
            rows.append((k, t, (f["h"] - th, f["size"][0] - tw, f["size"][1] - tl, dc, dang)))
            print(f"{k}: {NAME[t]} 高さ {f['h']:.1f}（本当 {th:.0f}）、上の面 {f['size'][0]:.1f} × {f['size'][1]:.1f}"
                  f"（本当 {tw:.0f} × {tl:.0f}）、中心のずれ {dc:.1f}mm、向きのずれ {dang:.1f}°")
    print("---- 道具ごとの平均の誤差（高さ・幅・長さ・中心・向き）と、見つけた回数")
    for t in S.TOOLS:
        e = np.array([r[2] for r in rows if r[1] == t and r[2] is not None])
        got = len(e)
        if got:
            print("{}: 高さ {:+.1f}mm、幅 {:+.1f}mm、長さ {:+.1f}mm、中心 {:.1f}mm、向き {:.1f}°（{}/{} 回）".format(NAME[t], *e.mean(0), got, n))
        else:
            print(f"{NAME[t]}: 一度も見つからない（0/{n} 回）")


if __name__ == "__main__":
    main()
