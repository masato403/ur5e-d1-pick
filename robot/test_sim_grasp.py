"""実機用のつかむプログラム（real_ur の pick_any.survey と pick_by_camera.grasp）を、書き換えずにシミュレーションの中で動かす。

  ロボットとの通信とハンドは、にせもの（sim_rtde.SimRobot / SimHand）に差し替える。
  実機だけの値（机の高さ TABLE_Z、実機のハンドの横のずれ）は、シミュレーションの値に置き換える。
  流れ：指示を d1-3B が読む → 見張りの位置で 3 枚撮る → 物ごとに高さ・形を測る → 物の種類を d1-3B が決める（実機と同じ kind_ai.decide。10/19〜）
  → 指示で運ぶ物を、d1-3B が選んだ順に、d1-3B が選んだ高さでつかみ（持てなければ、やり直し方も d1-3B が選ぶ）、マットの上の箱へ運び、
  物の底が箱のふちより DROP_GAP mm 上に来る高さで離して落とし入れる → 物の本当の位置で、箱の中に入ったかを確かめる（10/16〜）。
使い方:  [SIM_SAVE=フォルダ] python robot/test_sim_grasp.py [回数] [--show] [--say=ペン以外を箱に入れて] [--want=消しゴム・銀の棒]     （--show：ロボットの 3D 表示の窓を出して、本当の時間の速さで見せる）
"""
import math
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_real as T  # noqa: E402（校正の値の差し替え・道具の置き方・撮り方を使い回す）
import sim_real as S  # noqa: E402
import sim_rtde  # noqa: E402
from sim_rtde import SimHand, SimRobot  # noqa: E402

import height_by_motion as HM  # noqa: E402
import kind_ai as KA  # noqa: E402
import plan_ai as PL  # noqa: E402
import teach_box as TB  # noqa: E402（箱の中で物ごとに位置をずらす・離す高さ。実機の pick_pass --box と同じ）（指示・どれから運ぶか・つかむ高さ・やり直し方を d1-3B が選ぶ。実機と同じ）
import pick_any as PA  # noqa: E402
if os.environ.get("SURVEY_TWO") == "1":                          # 暗い所の直しの候補（robot/dark_two.py。10/19 夜）
    import dark_two  # noqa: E402
    dark_two.install()
import pick_by_camera as PB  # noqa: E402
LIFT_BASE = PB.LIFT_SPEED   # 持ち上げる速さ（やり直し方「ゆっくり持ち上げる」で半分にする。10/19 夜）
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import results_table  # noqa: E402（結果を robot/logs/results/sim_grasp.json に残し、仕様書の表に入れる）
from kind_ai import kind_of  # noqa: E402（判断の係が動いていない時の予備。KIND_TABLE=1 の時だけ使う）

RETRY, REFIND_MM = PA.RETRY, PA.REFIND_MM   # 持てなかった時のやり直しの回数、同じ物とみなす中心の距離（mm）（実機と同じ）
WEB_MODE = os.environ.get("SIM_WEB", "1")      # --show の時の見せ方：1 Chrome の 1 枚のページ（10/19）、0 前と同じ窓、2 両方（10/20）
WEB = WEB_MODE in ("1", "2")


class _Both:
    """ページと窓の両方に映す（SIM_WEB=2）。視点は両方に同じ値を入れる。"""
    def __init__(self, page, win):
        self.page, self.win = page, win
        self.cam = self

    def __setattr__(self, k, v):
        if k in ("lookat", "distance", "elevation", "azimuth"):
            for c in (self.page.cam, self.win.cam):
                if k == "lookat":
                    c.lookat[:] = v
                else:
                    setattr(c, k, v)
        else:
            object.__setattr__(self, k, v)

    @property
    def lookat(self):
        return _Lookat(self)

    def sync(self):
        self.page.sync()
        self.win.sync()

    def image(self, name, bgr):
        self.page.image(name, bgr)

    def close(self):
        self.page.close()
        self.win.close()


class _Lookat:
    def __init__(self, both):
        self.both = both

    def __setitem__(self, i, v):
        self.both.lookat = v
SAY = "全部の物を箱に入れて"   # 指示（--say=… で変える）。種類ごとに 箱に入れる／残す を d1-3B が決める（10/19〜）
SAY_WANT = []                # 確かめ用の、指示の正しい答え（--want=消しゴム・銀の棒）
# 試験のためにわざと起こすこと（実機のプログラムには無い）。持てない場面と、種類を決められない場面は、ふだんのシミュレーターでは起きないので、
# やり直し方と撮り直しを d1-3B が選ぶ所を確かめるために使う（10/19）
MISS_KIND = os.environ.get("SIM_MISS", "")            # この種類の 1 回目は、高さを MISS_MM 高く測ったことにする（空振りする）
MISS_MM = 20.0
UNSURE_KIND = os.environ.get("SIM_UNSURE", "")        # 最初に撮った写真では、この種類を「決められなかった」ことにする
REAL_RULES = os.environ.get("SIM_REAL_RULES", "1") == "1"   # 実機のつかみ方の決まり（物ごとの挟む力）を使う。0 で前と同じ 60%
DROP_GAP = 10.0              # 離す時の、物の底と箱のふちのすき間（mm）。物の底は指先より grip mm 下にある


def patch_grasp(env):
    """実機だけの値を、シミュレーションの値にする。"""
    PB.TABLE_Z = S.MAT_Z * 1000.0                                # 道具の点（つかむ点）の高さ 0 が机の面
    PB.HandClient = lambda *a, **k: SimHand(env)
    if S.HAND != "esg1":                                         # 実機のハンド（ESG1）なら、横のずれ・開く幅も実機の値のまま
        PB.GEOM = {"close_dir_tool": [0.0, 1.0, 0.0], "bias_tool_mm": [0.0, 0.0, 0.0]}   # 2F-85 は横のずれなし
        PA.MAX_W = 40.0                                          # 2F-85 は 85mm まで開く


def nearest_tool(env, f):
    return min(S.TOOLS, key=lambda t: np.linalg.norm(env.d.qpos[env.tool_q[t]:env.tool_q[t] + 2] * 1000 - f["center"][:2]))


def reasons(f, far):
    """つかむ候補から外す理由（実機の pick_any.main と同じ）。"""
    return [w for w, bad in (("平ら", f["h"] < PA.MIN_H), ("高すぎ", f["h"] > PA.MAX_H),
                             (f"幅 {f['size'][0]:.0f}mm は指に入らない", f["size"][0] > PA.MAX_W), ("遠い", far > PA.MAX_FAR),
                             (f"まわりがマットでない（{f['on_mat']:.0%}）", f["on_mat"] < PA.MAT_RING),
                             ("箱の縁にかかっている", f.get("by_box", False))) if bad]   # 実機の pick_any.main にある（10/16 に足した）


def go_to_obs(robot, obs):
    """見張りの高さへ、腕を動かして上がる（見張りの位置へは、次の search が動かす）。"""
    pose = robot.getActualTCPPose()
    robot.moveL([pose[0], pose[1], S.MAT_Z + T.OBS_H] + list(pose[3:]), PB.SPEED, PB.ACCEL)


SETTLE = 0.5                 # 撮る前に揺れが収まるのを待つ時間（locate.grab_now と同じ）


def patch_capture(env):
    """実機の撮り方（height_by_motion.capture_two：腕を動かして 3 枚撮る）を、そのままシミュレーションで使えるようにする。"""
    HM.TABLE_Z = S.MAT_Z * 1000.0

    def grab_now(settle=None, **_):
        t0 = env.d.time
        while env.d.time - t0 < (SETTLE if settle is None else settle):
            sim_rtde.frame(env)
        return env.render_real()
    HM.L.grab_now = grab_now
    # 止まったら撮る（HM.grab_fresh）：時計はシミュレーションの時計、1 回待つのは 1 コマ。待つ上限は前の決まった待ち（SETTLE）
    HM.now, HM.tick, HM.STILL_MAX = (lambda: env.d.time), (lambda: sim_rtde.frame(env)), SETTLE
    HM.L.archive = lambda *a, **k: None


SEARCH_S = []                # 探す時間（秒）。結果のファイルに残す


def search(env, robot, obs):
    """見張りの位置へ行き、実機と同じに 3 枚撮って測る。かかった時間（シミュレーションの中の時間 ＋ 計算の時間）も返す。"""
    t0, w0 = env.d.time, time.perf_counter()
    pose = robot.getActualTCPPose()
    robot.moveL([obs[0], obs[1], S.MAT_Z + HM.OBS / 1000] + list(pose[3:]), PB.SPEED, PB.ACCEL)
    HM.clear_memo()
    imgA, tcpA, others = HM.capture_two(None, c=robot, r=robot, at=list(obs), during=PA.pre_scores)
    t_move = env.d.time - t0
    if hasattr(sim_rtde.VIEWER, "image"):                        # ページに、撮った 3 枚を出す
        for name, im in zip("ABC", [imgA] + [o[0] for o in others]):
            sim_rtde.VIEWER.image(f"撮った写真 {name}", im)
    w1 = time.perf_counter()
    found = PA.survey(imgA, tcpA, others)
    t_calc = time.perf_counter() - w1
    w2 = time.perf_counter()
    KA.decide(imgA, found, kind_of, want=lambda f: f["on_mat"] >= PA.MAT_RING and f["h"] >= PA.MIN_H)   # 種類は d1-3B が決める（実機と同じ）
    t_calc += time.perf_counter() - w2
    SEARCH_S.append(t_move + t_calc)
    print(f"   探す時間 {t_move + t_calc:.1f} 秒（腕の動きと撮る前の待ち {t_move:.1f} 秒、計算 {t_calc:.1f} 秒）")
    return imgA, tcpA, others, found


def in_box(env, truth):
    """道具の本当の位置で、箱の中に入ったか。（入ったか、箱の真ん中からの mm、マットからの高さ mm）"""
    a = env.tool_q[truth]
    p = env.d.qpos[a:a + 3]
    d = np.abs(p[:2] - np.array(S.BOX_XY))
    return bool(np.all(d < np.array(S.BOX_IN) / 2) and p[2] < S.MAT_Z + S.BOX_H), np.linalg.norm(d) * 1000, (p[2] - S.MAT_Z) * 1000


def go_to(env, ik, xy, yaw=0.0):
    T.go_to(env, ik, xy, yaw)
    env.d.ctrl[:6] = env.d.qpos[:6]                              # 腕の目標も今の姿勢にそろえる（急に動かないように）


def main():
    global SAY, SAY_WANT
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 3
    for a in sys.argv[1:]:
        if a.startswith("--say="):
            SAY = a[len("--say="):]
        if a.startswith("--want="):                              # 確かめ用：指示の正しい答え（箱に入れる種類、・で区切る）
            SAY_WANT = [w for w in a[len("--want="):].split("・") if w]
    T.patch_real_code()
    env = S.RealLikeEnv()
    patch_grasp(env)
    patch_capture(env)
    if "--show" in sys.argv:
        if WEB:
            from web_view import WebView
            sim_rtde.VIEWER = WebView(env.m, env.d, title="つかむ試験（シミュレーター）")
            if WEB_MODE == "2":
                import mujoco.viewer as mjv
                from run_grip import topmost_windows
                sim_rtde.VIEWER = _Both(sim_rtde.VIEWER, mjv.launch_passive(env.m, env.d))
        else:
            import mujoco.viewer as mjv
            from run_grip import topmost_windows
            sim_rtde.VIEWER = mjv.launch_passive(env.m, env.d)
        sim_rtde.VIEWER.cam.lookat[:] = [S.MAT_CENTER[0], S.MAT_CENTER[1] - 0.05, 0.05]
        sim_rtde.VIEWER.cam.distance, sim_rtde.VIEWER.cam.elevation, sim_rtde.VIEWER.cam.azimuth = 0.9, -35, 120
        if WEB_MODE != "1":
            time.sleep(1.5)
            topmost_windows("MuJoCo", set())
    robot = SimRobot(env)
    rng = np.random.default_rng(2)
    total, ok_n = 0, 0
    kind_ok, kind_n = 0, 0                                       # d1-3B が種類を正しく答えた数
    kind_by, kind_ms, who = {}, [], set()                       # 本当の種類ごとの数・d1-3B の時間・決めた係
    n_grip, same_grip, retries, left_ok, left_n = 0, 0, [], 0, 0  # つかむ高さ・やり直し方・指示で残した物の数
    n_retake, retake_ok, n_miss = 0, 0, 0                         # 撮り直して聞き直した回数・それで決まった数・わざと空振りさせた数
    print(f"==== 指示：{SAY}")
    to_box = PL.instruction(SAY)                                 # 種類ごとに 箱に入れる／残す を d1-3B が決める
    for k in range(n):
        if k > 0:
            # 場面の終わり：箱に入った様子が窓で見えるよう 2 秒止めてから並べ直す（前は一瞬で全部消えて、壊れたように見えた。10/16）
            for _ in range(100):
                sim_rtde.frame(env)
        print(f"==== 場面 {k}：物を並べ直す")
        env.reset(rng)
        for i, o in enumerate(env.objects):
            env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
        obs = np.array(S.OBS_XY)
        go_to(env, robot.ik, obs)
        placed = T.place_tools(env, rng)
        mujoco.mj_forward(env.m, env.d)
        for _ in range(20):                                       # 置いた物を落ち着かせる（0.4 秒。窓にも映す）
            sim_rtde.frame(env)
        # 置き場所：箱の長い向き（y）に、物ごとに位置をずらす（実機の pick_pass --box と同じ teach_box.spot）
        box = {"center": [S.BOX_XY[0] * 1000, S.BOX_XY[1] * 1000], "axis": [0.0, 1.0], "length": S.BOX_BOTTOM[1] * 1000, "rim": S.BOX_H * 1000}
        PA.BOX = box                                             # 箱の所では物を探さない（実機の pick_pass と同じ。BOX_MASK=0 で前と同じ）
        imgA, tcpA, others, found = search(env, robot, obs)
        if os.environ.get("SIM_SAVE"):                           # 撮った 3 枚と本当の位置を残す（ロボット無しで測り方を比べる用。robot/diag_dark_refine.py。10/19）
            import pickle
            os.makedirs(os.environ["SIM_SAVE"], exist_ok=True)
            truth_xy = {T.NAME[t]: (env.d.qpos[env.tool_q[t]:env.tool_q[t] + 2] * 1000).tolist() for t in S.TOOLS}
            with open(os.path.join(os.environ["SIM_SAVE"], f"scene{k:02d}.pkl"), "wb") as fp:
                pickle.dump({"imgA": imgA, "tcpA": tcpA, "others": others, "box": box, "truth": truth_xy}, fp)
        for f in found:
            if UNSURE_KIND and f.get("kind") == UNSURE_KIND:
                f.update(kind=None, kind_unsure=True)
                print(f"   （試験のため）{UNSURE_KIND} を、種類を決められなかったことにする")
        for _ in range(KA.RETAKE):                               # 決められなかった物があれば、撮り直して聞き直す（実機と同じ）
            if not KA.unsure(found):
                break
            print(f"   種類を決められなかった物が {len(KA.unsure(found))} 個 → 撮り直して聞き直す")
            n_retake += 1
            again = search(env, robot, obs)[3]
            KA.recheck(found, again)
            retake_ok += sum(1 for f in found if f.get("kind") == UNSURE_KIND and not f.get("kind_unsure"))
        if os.environ.get("GRASP_DEBUG"):
            import cv2
            cv2.imwrite(os.path.join(HERE, "logs", f"grasp_scene{k}_A.jpg"), imgA)
            for t in S.TOOLS:
                q = env.d.qpos[env.tool_q[t]:env.tool_q[t] + 7]
                Rm = np.zeros(9)
                mujoco.mju_quat2Mat(Rm, q[3:])
                tilt = math.degrees(math.acos(min(1.0, abs(Rm.reshape(3, 3)[2, 2]))))
                print(f"   [本当] {T.NAME[t]}：中心の高さ {(q[2] - S.MAT_Z) * 1000:.1f}mm、傾き {tilt:.0f}°")
            for f in found:
                print(f"   [測った] {T.NAME[nearest_tool(env, f)]}：高さ {f['h']:.1f}（全体 {f['h_whole']:.1f}）、"
                      f"幅 {f['size'][0]:.1f}、円柱 {f['round']}、合い具合 {f['match']:.2f}")
        pending, leave = [], []                                  # 運ぶ物（f, 本当の道具）・指示どおり残す物の本当の道具
        for f in found:
            far = math.hypot(f["center"][0] - tcpA[0] * 1000, f["center"][1] - tcpA[1] * 1000)
            truth = nearest_tool(env, f)
            why = reasons(f, far)
            if f.get("kind_by"):
                kind_n += 1
                kind_ok += f["kind"] == T.NAME[truth]
                kb = kind_by.setdefault(T.NAME[truth], {"ok": 0, "n": 0})
                kb["n"] += 1
                kb["ok"] += f["kind"] == T.NAME[truth]
                who.add(f["kind_by"])
                if f.get("kind_ms"):
                    kind_ms.append(f["kind_ms"])
                print(f"{k}: d1-3B の答え {f['kind'] or 'それ以外'}（本当は {T.NAME[truth]}）"
                      + ("" if f["kind"] == T.NAME[truth] else "  ← 違う"))
            if not why and f.get("kind") is None:
                why = ["種類が分からない（d1-3B が それ以外 と答えた）"]
            if why:
                print(f"{k}: {T.NAME[truth]}（らしい物）は外す：高さ {f['h']:.1f}mm、" + "、".join(why))
                continue
            if not to_box.get(f["kind"], False):
                print(f"{k}: {f['kind']} は指示どおり机に残す")
                leave.append(truth)
                continue
            pending.append((f, truth))
        n_carry, i_carry = len(pending), 0
        while pending:
            place_xy = TB.spot(box, i_carry, n_carry)
            i_carry += 1
            f = PL.order([p[0] for p in pending], robot.getActualTCPPose(), found)   # どれから運ぶかは d1-3B が選ぶ
            truth = next(t for g, t in pending if g is f)
            pending = [p for p in pending if p[0] is not f]
            far = math.hypot(f["center"][0] - tcpA[0] * 1000, f["center"][1] - tcpA[1] * 1000)
            held, empty, drop = False, False, 0.0
            PB.LIFT_SPEED = LIFT_BASE                                # 物ごとに、持ち上げる速さを元に戻す（実機の pick_pass と同じ）
            for attempt in range(1 + RETRY):
                if attempt == 0 and MISS_KIND and f["kind"] == MISS_KIND:
                    f = dict(f, h=f["h"] + MISS_MM, h_whole=f.get("h_whole", f["h"]) + MISS_MM)
                    n_miss += 1
                    print(f"   （試験のため）{MISS_KIND} の高さを {MISS_MM:.0f}mm 高く測ったことにする → 空振りするはず")
                grip = PL.grip(f, drop, empty)                       # 指先を下ろす高さは d1-3B が候補から選ぶ
                n_grip += 1
                same_grip += abs(grip - PA.retry_grip(f, int(round(drop / PA.GRIP_STEP)), empty)) < 0.05
                print(f"{k}: {T.NAME[truth]} をつかむ{'' if attempt == 0 else f'（やり直し {attempt} 回目）'}"
                      f"{'（細い所＝軸）' if f.get('grip_on') == 'thin' else ''}"
                      f"（高さ {f['h']:.1f}mm、幅 {f['size'][0]:.1f}mm、指先を {grip:.1f}mm まで下ろす）")
                try:
                    force = PA.FORCE.get(f["kind"], 60) if REAL_RULES else 60   # 挟む力は実機と同じ物ごとの値（10/19）
                    held = PB.grasp(robot, robot, robot.getActualTCPPose(), f["center"], f["longv"], grip, force, list(place_xy), False,
                                    place_h=TB.release_h(box, grip), miss="retry", place_far=PB.BOX_FAR)
                except SystemExit as e:
                    print(f"   止まった: {e}")
                    break
                if held:
                    break
                # 持てなかった：どうやり直すかは d1-3B が選ぶ（もう下げられない・回数の上限、はプログラムが先に外す）
                empty = PB.LAST["closed"] is not None and PB.LAST["closed"] <= PB.HOLD_MM
                how = PL.retry(f, attempt, grip, empty)
                retries.append(how)
                if how == "skip":
                    print("   この物はやめる")
                    break
                if how == "lower":
                    drop += PA.GRIP_STEP
                if how == "slow":                                      # 滑った時：次は持ち上げる速さを半分にする（10/19 夜）
                    PB.LIFT_SPEED = LIFT_BASE / 2
                imgA, tcpA, others, found2 = search(env, robot, obs)   # 見張りの高さへ上がり、撮り直して、同じ物を測り直す
                again = [g for g in found2
                         if np.linalg.norm(g["center"][:2] - f["center"][:2]) < REFIND_MM and not reasons(g, far)
                         and g.get("kind") == f["kind"]]
                if not again:
                    print("   撮り直したら、つかめる物として見つからない → やめる")
                    break
                f = min(again, key=lambda g: np.linalg.norm(g["center"][:2] - f["center"][:2]))
            for _ in range(50):                                      # 落とした物が落ち着くまで（1 秒）
                sim_rtde.frame(env)
            inside, dist, height = in_box(env, truth)
            good = held and inside                                   # 持てて、運んで、箱の中に入ったら成功
            total += 1
            ok_n += good
            print(f"   → {'箱の中' if inside else '箱の外'}（箱の真ん中から {dist:.0f}mm、高さ {height:.0f}mm）"
                  f" {'成功' if good else '失敗'}")
            # 箱に入れられなかった物は、どけずにその場に残す（前は机の外へ瞬間移動させていて、消えたように見えた。10/16）
            go_to_obs(robot, obs)
        for truth in leave:                                          # 指示で残した物が、箱に入っていないか
            left_n += 1
            left_ok += not in_box(env, truth)[0]
    put = [kk for kk, v in to_box.items() if v]
    print(f"---- 指示「{SAY}」：箱に入れる {'・'.join(put) or 'なし'}、"
          f"残す {'・'.join(kk for kk, v in to_box.items() if not v) or 'なし'}（正しい答え {'・'.join(SAY_WANT) if SAY_WANT else '―'}）")
    print(f"---- d1-3B が種類を正しく答えた数 {kind_ok}/{kind_n}")
    print(f"---- つかむ高さを d1-3B が選んだ回数 {n_grip}（前の決まりと同じ高さ {same_grip}）、やり直し方 {retries or 'なし'}")
    print(f"---- 指示どおり残した物 {left_ok}/{left_n}")
    print(f"---- 撮り直して聞き直した回数 {n_retake}（それで決まった {retake_ok}）、わざと空振りさせた数 {n_miss}")
    print(f"---- つかんで箱に入れられた回数 {ok_n}/{total}")
    if HM.LAST_WAIT:
        print(f"---- 撮る前に止まるまで待った時間 平均 {np.mean(HM.LAST_WAIT):.2f} 秒・最大 {np.max(HM.LAST_WAIT):.2f} 秒（前は決まって {SETTLE} 秒）")
    results_table.save("sim_grasp", {
        "when": time.strftime("%Y-%m-%d %H:%M"), "scenes": n, "tool3": "銀の棒" if S.TOOL3 == "rod" else "ドライバー",
        "grasp_ok": int(ok_n), "grasp_n": total, "kind_ok": int(kind_ok), "kind_n": kind_n, "kind_by": kind_by,
        "kind_by_who": "、".join(sorted("d1-3B" if w == "d1" else "長さと幅の表" for w in who)) or "―",
        "kind_ms": float(np.mean(kind_ms)) if kind_ms else 0.0,
        "search_s": [float(min(SEARCH_S)), float(max(SEARCH_S))] if SEARCH_S else [0.0, 0.0],
        "plan_ai": PL.USE_AI, "say": SAY, "to_box": to_box, "say_want": SAY_WANT,
        "say_ok": (set(put) == set(SAY_WANT)) if SAY_WANT else None,
        "left_ok": int(left_ok), "left_n": left_n, "n_grip": n_grip, "same_grip": int(same_grip), "retries": retries,
        "inject": "、".join(x for x in (f"{MISS_KIND} を空振りさせた" if MISS_KIND else "", f"{UNSURE_KIND} を決められなかったことにした" if UNSURE_KIND else "") if x),
        "n_retake": n_retake, "retake_ok": retake_ok, "n_miss": n_miss,
        "still_wait": [float(np.mean(HM.LAST_WAIT)), float(np.max(HM.LAST_WAIT))] if HM.LAST_WAIT else None, "still_n": len(HM.LAST_WAIT)})
    if sim_rtde.VIEWER is not None:
        sim_rtde.VIEWER.close()


if __name__ == "__main__":
    main()
