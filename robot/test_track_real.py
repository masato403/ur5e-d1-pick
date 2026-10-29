"""実機版の「三脚で見つけて追いながら下りる」台本（real_ur/track_real.py の run）を、書き換えずにシミュレーションの中で動かす（10/20 夜）。

  ・ロボットとハンドは、にせもの（sim_rtde.SimRobot / SimHand）。つかむ所の差し替えは test_sim_grasp.patch_grasp と同じ。
  ・三脚の写真は、シミュレーターの三脚（本物の校正の位置・焦点距離・画像の中心で置いた、sim_real.SIM_TRIPOD_POSE=real）を 960×540 にして渡す。
    画素と机の位置の変換は、シミュレーターの三脚のもの（tripod_on_plane・tripod_px）に替える。本物の位置のずれの直し（tripod_view.FIX）は 0。
  ・実機とシミュレーターは真上から見て 180° 回っているので、見る位置（track_real.LOOK）は箱の中心を合わせて回して置く。
使い方:  python robot/test_track_real.py [場面の数] [--show] [--say=全部の物を箱に入れて]
"""
import json
import os
import sys
import time

os.environ.setdefault("SIM_TRIPOD", "1")
os.environ.setdefault("SIM_MAT", "real")              # 実機と同じ暗いマット（三脚の物の見つけ方は「暗くない所」を物とみなす）
import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_grasp as G  # noqa: E402（つかむ所の差し替え・箱に入ったかの確かめ・場面の作り方を使い回す）
import test_sim_real as T  # noqa: E402
import sim_real as S  # noqa: E402
import sim_rtde  # noqa: E402
from sim_rtde import SimHand, SimRobot  # noqa: E402

import plan_ai as PL  # noqa: E402
import tripod_view as TV  # noqa: E402
import track_real as TR  # noqa: E402

VIEW_EVERY = int(os.environ.get("TRR_VIEW_EVERY", 10))          # ページの映像を送る間（コマ、1 コマ 20ms）。3 にすると試験の中でシミュレーターが本当の時間の約 45% に遅れた（10/23）ので 10 に戻した
VIEW_W = int(os.environ.get("TRR_VIEW_W", 480))                  # ページの映像の幅（高さは 9/16）。測る写真の大きさとは別
TW, TH = 960, 540                                                # 実機の三脚の写真の大きさ（tripod_view はこの大きさで考える）
SC = S.TRIPOD_W / TW


def real_to_sim(xy):
    """実機の根元の座標（mm）→ シミュレーターの座標（mm）。箱の中心を合わせ、180° 回す（sim_real.tripod_pose_real と同じ）。"""
    if S.LAYOUT_REAL:                                            # 実機の並び：根元を基準に 180° 回すだけ（10/23）
        return -np.asarray(xy[:2], float)
    box = json.load(open(S.TRIPOD_BOX_REAL, encoding="utf-8"))["center"]
    return np.array([S.BOX_XY[0] * 1000 - (xy[0] - box[0]), S.BOX_XY[1] * 1000 - (xy[1] - box[1])])


def patch_tripod(env):
    """tripod_view の写真の取り方と、画素と位置の変換を、シミュレーターの三脚にする。"""
    def grab(after=None, timeout=3.0):
        if after != 0:                                           # after=0 は「待たずに最新の写真」（下りながら追う時。腕は servoL で動いている）
            for _ in range(5):                                   # 0.1 秒進めてから撮る（実機の cam_view の間に近づける）
                sim_rtde.frame(env)
        return cv2.resize(env.render_tripod()[0], (TW, TH), interpolation=cv2.INTER_AREA)
    TV.grab = grab
    base_frame = sim_rtde.frame
    cnt = [0]

    view = {}

    def frame(env_):
        base_frame(env_)
        cnt[0] += 1
        if cnt[0] % VIEW_EVERY == 0 and sim_rtde.VIEWER is not None and hasattr(sim_rtde.VIEWER, "image"):   # VIEW_EVERY コマごとに、三脚と手首の映像をページへ（三脚が上、手首が下）
            if not view:                                         # 見せる映像だけ小さく描く（測る写真は今の大きさのまま。10/23、映像がカクカクしたため）
                view["r"] = mujoco.Renderer(env.m, VIEW_W * 9 // 16, VIEW_W)
                view["t0"], view["n"], view["sim0"] = time.perf_counter(), 0, env.d.time
            for name, cam in (("三脚のカメラ", "tripod"), ("手首のカメラ", "wrist_real")):
                view["r"].update_scene(env.d, camera=cam)
                sim_rtde.VIEWER.image(name, cv2.cvtColor(view["r"].render(), cv2.COLOR_RGB2BGR))
            view["n"] += 1
            now = time.perf_counter()
            if now - view["t0"] > 5.0 or env.d.time < view["sim0"]:   # 5 秒ごとに、その間の枚数とシミュレーターの速さを数える（場面が変わると時刻が戻るので数え直す）
                if env.d.time >= view["sim0"] and view["n"] > 1:
                    view.setdefault("log", []).append((view["n"] / (now - view["t0"]), (env.d.time - view["sim0"]) / (now - view["t0"])))
                    if len(view["log"]) % 6 == 0:
                        fps, sp = np.median([a_ for a_, _ in view["log"]]), np.median([b_ for _, b_ in view["log"]])
                        print(f"   （試験）ページの映像 {fps:.1f} 枚/秒、シミュレーターの速さ {sp * 100:.0f}%（本当の時間に対して、5 秒ごとの真ん中の値）", flush=True)
                view["t0"], view["n"], view["sim0"] = now, 0, env.d.time
    sim_rtde.frame = frame
    # 下りる間だけ、狙いの物を一定の速さで滑らせる（TRR_MOVE=mm/秒。手で動かす物のまね、10/22）
    v_mm = float(os.environ.get("TRR_MOVE", 0))
    MOVE_S = float(os.environ.get("TRR_MOVE_S", 2.0))
    MOVE_SIDE = os.environ.get("TRR_MOVE_SIDE", "0") == "1"   # 1：物を横（長い向きと直角）へだけ動かす
    if v_mm > 0 and TR.SERVO:
        tool_of = {"消しゴム": "eraser", "ペン": "r_pen", "銀の棒": "r_rod"}
        rng_m = np.random.default_rng(7)
        moving = {}
        base_follow = TR.follow_servo
        base_frame2 = sim_rtde.frame

        def frame_m(env_):
            if moving and (MOVE_S <= 0 or env.d.time - moving["t0"] < MOVE_S):   # 動かすのは MOVE_S 秒だけ（手で動かして離すまね）。0 ならつかむまでずっと（10/22）
                a = env.tool_q[moving["t"]]
                step = moving["v"] * env.m.opt.timestep * sim_rtde.STEPS_PER_FRAME
                nxt = env.d.qpos[a:a + 2] + step
                hx, hy = S.MAT_MM[0] / 2000 - 0.03, S.MAT_MM[1] / 2000 - 0.03
                for i_, h_ in ((0, hx), (1, hy)):                # マットの端の 30mm 手前で跳ね返る
                    if abs(nxt[i_] - S.MAT_CENTER[i_]) > h_ and np.sign(step[i_]) == np.sign(nxt[i_] - S.MAT_CENTER[i_]):
                        moving["v"][i_] *= -1
                        step[i_] *= -1
                bx, by = S.BOX_XY
                if abs(nxt[0] - bx) < S.BOX_RIM[0] / 2 + 0.04 and abs(nxt[1] - by) < S.BOX_RIM[1] / 2 + 0.04:   # 箱にも入らないよう跳ね返る
                    moving["v"] *= -1
                    step *= -1
                env.d.qpos[a:a + 2] += step
            base_frame2(env_)

        def follow_m(c, r, f, rec, steps, log=print, **kw):
            a = env.tool_q[tool_of[f["kind"]]]
            p0 = env.d.qpos[a:a + 2].copy()
            hx, hy = S.MAT_MM[0] / 2000 - 0.03, S.MAT_MM[1] / 2000 - 0.03       # 動かした後もマットの端から 30mm 内に収まる向きを選ぶ（前は外へ飛び出た）
            qw, qz = env.d.qpos[a + 3], env.d.qpos[a + 6]
            yaw = 2 * np.arctan2(qz, qw)                         # 物の長い向き（寝かせて置いた向き）
            for _ in range(50):
                ang = rng_m.uniform(0, 2 * np.pi)
                if MOVE_SIDE:                                    # 横（長い向きと直角＝指を閉じる向き）へ動かす。向きは左右どちらか（10/23、消しゴムの横の動きに弱かった）
                    ang = yaw + np.pi / 2 + (np.pi if rng_m.uniform() < 0.5 else 0.0)
                end = p0 + v_mm / 1000 * MOVE_S * np.array([np.cos(ang), np.sin(ang)])
                if abs(end[0] - S.MAT_CENTER[0]) < hx and abs(end[1] - S.MAT_CENTER[1]) < hy:
                    break
            moving.update(t0=env.d.time, t=tool_of[f["kind"]], v=v_mm / 1000 * np.array([np.cos(ang), np.sin(ang)]))
            try:
                return base_follow(c, r, f, rec, steps, log, **kw)
            finally:
                moved = (env.d.qpos[a:a + 2] - p0) * 1000
                if MOVE_S > 0:
                    moving.clear()
                true = env.d.qpos[a:a + 2] * 1000
                log(f"   （試験）物を {np.linalg.norm(moved):.0f}mm 動かした。下り終わりの見当と本当の位置の差 "
                    f"{np.linalg.norm(f['center'][:2] - true):.1f}mm")
        sim_rtde.frame = frame_m
        TR.follow_servo = follow_m
        base_grip = sim_rtde.SimHand.grip

        def grip_m(self, *a_, **k_):                             # つかむまでずっと動かす時は、閉じ始めで止める
            moving.clear()
            return base_grip(self, *a_, **k_)
        sim_rtde.SimHand.grip = grip_m
        base_start = sim_rtde.SimHand.grip_start

        def start_m(self, *a_, **k_):                            # 追いながら閉じる時も、閉じ始めで止める
            moving.clear()
            return base_start(self, *a_, **k_)
        sim_rtde.SimHand.grip_start = start_m
    TV.to_table = lambda uv, h=TV.OBJ_H: env.tripod_on_plane(np.asarray(uv, float) * SC, h)[0][:2]
    TV.to_px = lambda P, lens=True: env.tripod_px(P) / SC
    TV.FIX = np.zeros(2)
    TV.TABLE_Z = TR.TABLE_Z = S.MAT_Z * 1000.0
    hx, hy = S.MAT_MM[0] / 2, S.MAT_MM[1] / 2
    cx, cy = S.MAT_CENTER[0] * 1000, S.MAT_CENTER[1] * 1000
    TV.MAT_PX = TV.to_px([[cx - hx, cy - hy, S.MAT_Z * 1000], [cx + hx, cy - hy, S.MAT_Z * 1000],
                          [cx + hx, cy + hy, S.MAT_Z * 1000], [cx - hx, cy + hy, S.MAT_Z * 1000]])
    # 手首の最新の写真：シミュレーターでは今の姿で描き、姿勢も今のもの（実機は cam_view の wrist_live.jpg と、撮った時刻の姿勢）
    # TRR_WRIST_FPS>0 なら、手首のカメラはその枚数/秒でしか撮れない（実機の 4.5 枚/秒・MJPG の 30 枚/秒のまね。10/22）。
    # 渡すのは最後に撮った写真と、撮った時の姿勢（実機は写真の時刻の姿勢を記録から引く）
    fps = float(os.environ.get("TRR_WRIST_FPS", 0))
    last = {"t": -1.0}

    def wrist_live(r, hist, seen_t):
        if fps <= 0:
            return env.render_real(), list(r.getActualTCPPose()), seen_t + 1
        if env.d.time - last["t"] >= 1.0 / fps:
            last.update(t=env.d.time, img=env.render_real(), tcp=list(r.getActualTCPPose()))
        if last["t"] <= seen_t:
            return None, None, seen_t
        return last["img"], last["tcp"], last["t"]
    TR.wrist_live = wrist_live
    lx, ly = real_to_sim(TR.LOOK[:2])
    TR.LOOK = (float(lx), float(ly), TR.LOOK[2])


SPREAD_GAP = float(os.environ.get("TRR_GAP", 0.04))              # 物どうしのすき間の下限（m）


REACH = float(os.environ.get("TRR_REACH", 0.74))   # 腕の根元からこれ（m）より遠い所には置かない。マットの奥の角（根元から 777mm）の消しゴムは、
                                                    # 腕が届かず指が 36mm ずれて 2 回とも持てなかった（10/22。743mm は届いた）


REACH_MIN = float(os.environ.get("TRR_REACH_MIN", 0.26))   # 腕の根元からこれ（m）より近い所にも置かない（196mm でハンドが上腕に当たった。10/22）


def place_wide(env, rng):
    """道具をマット全体に散らして置く（10/22。T.place_tools は手首の写真に入るよう真ん中の 180×140mm に寄せて置き、物が近すぎた）。
    マットの端から T.MAT_EDGE 内・箱から離す・物どうし SPREAD_GAP 以上あける。置けない物はこの場面に出さない。"""
    placed = {}
    hx, hy = S.MAT_MM[0] / 2000, S.MAT_MM[1] / 2000
    for j, t in enumerate(S.TOOLS):
        for _ in range(3000):
            xy = np.array(S.MAT_CENTER) + rng.uniform([-hx, -hy], [hx, hy])
            yaw = rng.uniform(0, np.pi)
            fp = T.footprint(t, xy, yaw)
            if T.on_mat(t, xy, yaw) and T.away_from_box(t, xy, yaw) and np.max(np.linalg.norm(fp[0], axis=1)) + fp[1] <= REACH and np.min(np.linalg.norm(fp[0], axis=1)) - fp[1] >= REACH_MIN and all(
                    np.min(np.linalg.norm(fp[0][:, None] - q[0][None], axis=2)) > fp[1] + q[1] + SPREAD_GAP
                    for q in (T.footprint(u, p, y) for u, (p, y) in placed.items())):
                env.set_tool(t, xy, yaw)
                placed[t] = (xy, yaw)
                break
        else:
            env.set_tool(t, np.array([1.5, 0.5 + 0.1 * j]), 0.0)
            print(f"   （{t} は置ける所が無いので、この場面には出さない）")
    mujoco.mj_forward(env.m, env.d)
    return placed


def look_pose(robot):
    return [TR.LOOK[0] / 1000, TR.LOOK[1] / 1000, (TR.TABLE_Z + TR.LOOK[2]) / 1000] + list(robot.getActualTCPPose()[3:])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 3
    say = next((a[len("--say="):] for a in sys.argv[1:] if a.startswith("--say=")), "全部の物を箱に入れて")
    T.patch_real_code()
    env = S.RealLikeEnv()
    G.patch_grasp(env)
    G.patch_capture(env)
    patch_tripod(env)
    if "--show" in sys.argv:
        from web_view import WebView
        sim_rtde.VIEWER = WebView(env.m, env.d, title="三脚で見つけて追う（実機版の台本をシミュレーターで）")
        sim_rtde.VIEWER.cam.lookat[:] = [S.MAT_CENTER[0], S.MAT_CENTER[1] - 0.05, 0.05]
        sim_rtde.VIEWER.cam.distance, sim_rtde.VIEWER.cam.elevation, sim_rtde.VIEWER.cam.azimuth = 0.9, -35, 120
    robot = SimRobot(env)
    hand = SimHand(env)
    rng = np.random.default_rng(2)
    PICK = [int(x) for x in os.environ.get("TRR_SCENES", "").split(",") if x.strip()]
    n = max(n, max(PICK) + 1) if PICK else n
    to_box = PL.instruction(say, kinds=tuple(TR.H_NOM))
    carry = [k for k in TR.H_NOM if to_box.get(k)]
    long_x = S.BOX_BOTTOM[0] > S.BOX_BOTTOM[1]                   # 実機の並びでは箱の長い向きが x（10/23）
    box = {"center": [S.BOX_XY[0] * 1000, S.BOX_XY[1] * 1000], "axis": [1.0, 0.0] if long_x else [0.0, 1.0],
           "length": max(S.BOX_BOTTOM) * 1000, "rim": S.BOX_H * 1000}
    G.PA.BOX = box
    rot = None
    tot, ok = 0, 0
    for k in range(n):
        print(f"==== 場面 {k}：物を並べ直す", flush=True)
        env.reset(rng)
        for i, o in enumerate(env.objects):
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        if TV.BG is None:                                        # 背景は最初の場面で 1 回だけ撮る（マットと見る位置は場面が変わっても同じ。10/22、毎回挟まって見づらかった）
            G.go_to(env, robot.ik, np.array(S.OBS_XY))
            mujoco.mj_forward(env.m, env.d)
            robot.moveL(look_pose(robot), TR.SPEED, TR.ACCEL)          # 背景も、見る時と同じ腕の位置で撮る（腕の影が物に見えないように）
            for j, t in enumerate(S.TOOLS):                          # 消しゴム・ペン・銀の棒もマットの外へどけてから背景を撮る（前は写り込み、ペンとつながって消えた）
                env.set_tool(t, np.array([1.5 + 0.1 * j, -0.5]), 0.0)
            mujoco.mj_forward(env.m, env.d)
            TV.BG = None
            TV.BG = TV.grab()                                        # 物の無いマット（シミュレーターのマットは明るい所があり、「暗くない所」では物が分からない）
            G.go_to(env, robot.ik, np.array(S.OBS_XY))               # 物を並べる時は、腕を見張りの位置に戻す（見る位置のままだと腕の下に置けず、物が減った）
        else:
            G.go_to(env, robot.ik, np.array(S.OBS_XY))
        if os.environ.get("TRR_SPREAD", "1") == "1":
            place_wide(env, rng)                                 # マット全体に散らす（0 で前の、真ん中に寄せる置き方）
        else:
            T.place_tools(env, rng)
        if PICK and k not in PICK:                               # TRR_SCENES：選んだ場面だけ流す（並べ方は乱数を同じ順に使うので、場面の番号が同じなら同じ並び。10/23）
            continue
        mujoco.mj_forward(env.m, env.d)
        for _ in range(20):
            sim_rtde.frame(env)
        rot = list(robot.getActualTCPPose()[3:])
        rec = os.path.join(HERE, "logs", "track_real_sim", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(rec, exist_ok=True)
        cv2.imwrite(os.path.join(rec, "bg.jpg"), TV.BG)              # 背景の写真も残す（diag_cut.py で、後から物の見え方を調べる）
        t_sim = env.d.time
        res = TR.run(robot, robot, hand, carry, box, rot, rec)
        t_sim = env.d.time - t_sim                               # シミュレーターの中の時間（腕が動いた時間。実機の時間に近い）
        for t in S.TOOLS:
            inside = G.in_box(env, t)[0]
            tot += 1
            ok += inside
        print(f"   結果：{res}、本当に箱の中：" + "、".join(f"{t} {'○' if G.in_box(env, t)[0] else '×'}" for t in S.TOOLS) + f"、かかった時間 {t_sim:.1f} 秒（記録 {rec}）", flush=True)
    print(f"==== 箱に入った {ok}/{tot}", flush=True)
    if sim_rtde.VIEWER is not None:
        sim_rtde.VIEWER.close()


if __name__ == "__main__":
    main()
