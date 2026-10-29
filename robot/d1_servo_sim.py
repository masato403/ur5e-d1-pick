"""d1-3B を毎コマの制御に使う案（10/20 にもらった実験案の 段階 1・2）を、シミュレーターで確かめる。
毎コマ：手首のカメラの写真に「今の指の真下の点」を緑の十字で描き（数はプログラムが計算する）、d1-3B に 1 回の計算でまとめて聞く
  dx（score）：物は十字の左か右か、どれだけ離れているか
  dy（score）：物は十字の上か下か（写真の上＝奥）
  on（noul）：十字が物の上にあるか
score の期待値（または一番確率の高い段階＝argmax）を、写真の向きの 1 コマの動き（mm）に直し、写真の右・上の向きを机の上の向きに直して腕を動かす。
十字が物に乗り、ずれが小さい時に下りて閉じる。物は止めたまま（段階 1）・動かす（追いかける目標）の両方を試す。
d1-3B が答えるまでの時間（本当にかかった時間）は、シミュレーションの時間も同じだけ進め、その間は前の目標へ動き続ける（実機と同じ遅れ）。
使い方:  python robot/d1_servo_sim.py [条件ごとの回数]   （判断の係 real_ur/d1_server.py が要る。結果は robot/logs/d1_servo_sim.json）
         D1S_GRID=quick で少しだけ
"""
import base64
import itertools
import json
import math
import os
import sys
import time
import urllib.request

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import track_sim as TS  # noqa: E402（物の動かし方・コマの進め方・向きの計算を使い回す）
import test_sim_grasp as TG  # noqa: E402
import test_sim_real as T  # noqa: E402
import sim_real as S  # noqa: E402
import sim_rtde  # noqa: E402
from sim_rtde import SimHand, SimRobot, rotvec_to_R, R_to_rotvec  # noqa: E402
import height_by_motion as HM  # noqa: E402
import pick_by_camera as PB  # noqa: E402

URL = "http://127.0.0.1:5006/ask"
DT = TS.DT
SEND_W = 768                  # d1-3B に送る写真の横幅（画素）


def levels_text(n, neg, pos):
    """n 段階の説明（低い順）。真ん中が「十字の上」。"""
    half = n // 2
    words = {1: "slightly", 2: "a little", 3: "clearly", 4: "far", 5: "very far"}
    out = []
    for i in range(n):
        k = i - half
        if k == 0:
            out.append("centered on the cross in this direction")
        else:
            w = words[min(5, max(1, round(abs(k) * 5 / half)))]
            out.append(f"{w} {neg if k < 0 else pos} of the cross")
    return out


def ask(img, n_lv):
    small = cv2.resize(img, (SEND_W, int(img.shape[0] * SEND_W / img.shape[1])), interpolation=cv2.INTER_AREA)
    b = base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode()
    state = ("A camera on a robot gripper looks down at a cutting mat. A green cross is drawn on the image: it marks the point on "
             "the mat right under the gripper. There is one object on the mat (an eraser, a pen or a metal rod). "
             "The robot moves the gripper so that the cross lands on the center of the object.")
    qs = {"dx": {"type": "score", "instructions": "In the image, where is the center of the object compared with the green cross, left or right?",
                 "criteria": levels_text(n_lv, "left", "right")},
          "dy": {"type": "score", "instructions": "In the image, where is the center of the object compared with the green cross, up or down?",
                 "criteria": levels_text(n_lv, "above", "below")},
          "on": {"type": "noul", "instructions": "Is the green cross on the center of the object?"}}
    body = json.dumps({"state": state, "questions": qs, "images": [b]}).encode()
    t = time.perf_counter()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"}), timeout=30).read())
    return r["answers"], (time.perf_counter() - t) * 1000


def mark(img, tcp, h_mm):
    """指の真下の点（机から h_mm/2 の面）を写真に緑の十字で描く。その画素と、写真の右・上の向きの机の上の向き（単位）を返す。"""
    P = np.array([[tcp[0] * 1000, tcp[1] * 1000, S.MAT_Z * 1000 + h_mm / 2]])
    px = HM.project_fast(P, tcp)[0] if hasattr(HM, "project_fast") else None
    u, v = int(round(px[0])), int(round(px[1]))
    out = img.copy()
    cv2.drawMarker(out, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 90, 6)
    w = HM.on_plane(np.array([[u, v], [u + 100, v], [u, v - 100]]), tcp, h_mm / 2)
    e_r = (w[1] - w[0])[:2]
    e_u = (w[2] - w[0])[:2]
    return out, (u, v), e_r / np.linalg.norm(e_r), e_u / np.linalg.norm(e_u)


def trial(env, robot, hand, rng, tool, mode, speed, n_lv, step_mm, use, alpha):
    TS.MOVER = None
    env.reset(rng)
    for i, o in enumerate(env.objects):
        env.set_obj(o, np.array([1.0 + 0.1 * i, -0.5]))
    TG.go_to(env, robot.ik, np.array(S.OBS_XY))
    T.place_tools(env, rng)
    for t in S.TOOLS:
        if t != tool:
            env.set_tool(t, (1.5, -0.5 + 0.1 * S.TOOLS.index(t)), 0.0)
    mujoco.mj_forward(env.m, env.d)
    hand.env.set_finger(False)
    for _ in range(40):
        TS.frame(env)
    z_work = S.MAT_Z + PB.HOVER / 1000
    robot.moveL(list(S.OBS_XY) + [z_work] + list(robot.getActualTCPPose()[3:]), PB.SPEED, PB.ACCEL)
    h_mm = S.TOOL_H[tool]
    TS.MOVER = TS.Mover(env, tool, mode, speed, rng)
    q = env.d.qpos[:6].copy()
    lead = (-env.m.actuator_biasprm[:6, 2] / env.m.actuator_gainprm[:6, 0]) if sim_rtde.FEEDFWD else np.zeros(6)
    p_cmd = np.array(env.d.site_xpos[env.site])
    R_cmd = env.d.site_xmat[env.site].reshape(3, 3).copy()
    goal = p_cmd[:2].copy()
    sm = np.zeros(2)
    t0 = env.d.time
    log, ms_all, flips, last_sign = [], [], 0, None
    why, reached_t = "時間切れ", None

    def run_frames(nf):
        nonlocal q
        for _ in range(nf):
            d = goal - p_cmd[:2]
            dist = np.linalg.norm(d)
            p_cmd[:2] += d / max(dist, 1e-9) * min(dist, TS.VMAX * DT)
            q_prev = q
            q = robot.ik.solve(q, p_cmd, R_cmd, iters=8)
            env.d.ctrl[:6] = q + lead * (q - q_prev) / DT
            TS.frame(env)

    while env.d.time - t0 < TS.TIMEOUT:
        tcp = env.tcp_like()
        img, px, e_r, e_u = mark(env.render_real(), tcp, h_mm)
        ans, ms = ask(img, n_lv)
        ms_all.append(ms)
        run_frames(max(1, int(math.ceil(ms / 1000 / DT))))        # 答えを待つ間も、前の目標へ動き続ける
        lv = np.arange(n_lv) - n_lv // 2
        def pick(a):
            p = np.array([a["probabilities"][str(i)] for i in range(n_lv)])
            return float(lv[int(np.argmax(p))]) if use == "argmax" else float(p @ lv)
        sx, sy = pick(ans["dx"]), -pick(ans["dy"])                 # 写真の下（below）は −上
        raw = np.array([sx, sy]) * step_mm / (n_lv // 2)
        sm = alpha * raw + (1 - alpha) * sm                        # ならし（alpha=1 でならさない）
        move = sm[0] * e_r + sm[1] * e_u                            # mm、机の上の向き
        goal = p_cmd[:2] + move / 1000
        tip = env.d.site_xpos[env.site][:2] * 1000
        true = TS.MOVER.true_xy()
        err = float(np.hypot(*(tip - true)))
        s_ = np.sign(sx) if abs(sx) > 0.3 else 0
        if last_sign is not None and s_ != 0 and s_ != last_sign:
            flips += 1
        if s_ != 0:
            last_sign = s_
        log.append({"t": round(env.d.time - t0, 2), "err": round(err, 1), "dx": round(sx, 2), "dy": round(sy, 2),
                    "on": round(ans["on"]["noul"], 3), "ms": round(ms)})
        if err < 10 and reached_t is None:
            reached_t = env.d.time - t0
        if ans["on"]["noul"] > 0.5 and np.linalg.norm(raw) < 0.5 * step_mm:
            why = "d1-3B が「十字が物の上」と答えた"
            break
    true = TS.MOVER.true_xy()
    tip = env.d.site_xpos[env.site][:2] * 1000
    TS.MOVER.on = False
    TS.MOVER = None
    return {"tool": tool, "mode": mode, "speed": speed, "levels": n_lv, "step": step_mm, "use": use, "alpha": alpha,
            "end_err": round(float(np.hypot(*(tip - true))), 1), "reached10_t": None if reached_t is None else round(reached_t, 2),
            "why": why, "t": round(env.d.time - t0, 2), "ms": round(float(np.median(ms_all)), 1), "flips": flips, "n": len(log), "log": log}


GRIDS = {
    "quick": dict(mode=["line"], speed=[0], levels=[5], step=[5], use=["exp"], alpha=[1.0]),
    "stage1": dict(mode=["line"], speed=[0], levels=[3, 5, 9], step=[2, 5], use=["exp", "argmax"], alpha=[1.0, 0.5]),
    "moving": dict(mode=["line", "stop"], speed=[20, 50], levels=[5, 9], step=[5], use=["exp"], alpha=[1.0, 0.5]),
}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    gname = os.environ.get("D1S_GRID", "quick")
    grid = GRIDS[gname]
    sim_rtde.frame = TS.frame
    T.patch_real_code()                                          # 実機の校正の値を、シミュレーターのカメラの値にする（test_sim_grasp と同じ）
    env = S.RealLikeEnv()
    TG.patch_grasp(env)
    TG.patch_capture(env)
    robot, hand = SimRobot(env), SimHand(env, log=lambda s: None)
    rng = np.random.default_rng(11)
    tools = ["eraser", "r_pen", "r_rod"]
    rows = []
    for i, (mode, speed, n_lv, step, use, alpha) in enumerate(itertools.product(*grid.values())):
        for j in range(n):
            tool = tools[(i + j) % 3]
            r = trial(env, robot, hand, rng, tool, mode, speed, n_lv, step, use, alpha)
            rows.append(r)
            print(f"{TS.NAMES[tool]} 動き {mode} {speed}mm/秒 段階 {n_lv} 1コマ最大 {step}mm {use} ならし {alpha}："
                  f"{r['why']}、最後のずれ {r['end_err']}mm、10mm 以内に入った {r['reached10_t']}秒、{r['t']}秒、"
                  f"聞いた {r['n']} 回（{r['ms']}ms）、左右の入れ替わり {r['flips']}", flush=True)
    out = os.path.join(HERE, "logs", f"d1_servo_sim_{gname}.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"結果：{out}")


if __name__ == "__main__":
    main()
