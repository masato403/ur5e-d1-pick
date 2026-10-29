"""実機用のプログラム（real_ur の pick_by_camera.grasp など）を、そのままシミュレーション（sim_real.RealLikeEnv）で動かすための
「にせのロボット」と「にせのハンド」。ur_rtde（RTDEControl / RTDEReceive）と hand.HandClient と同じ使い方。

  ・moveL：道具の点（つかむ点 g_pinch）を、今の位置から目標までまっすぐ、決めた速さで動かす（途中も物理を進める）
  ・getInverseKinematics：シミュレーションの逆運動学（run_pick.IK）
  ・ハンド：開く・閉じる（力は無視）・開き量（指先のすき間 mm）
"""
import math
import os
import time

import cv2
import mujoco
import numpy as np

from run_pick import IK

STEPS_PER_FRAME = 10          # 物理の 1 刻みは 2ms。20ms ごとに腕の目標を進める
VIEWER = None                 # 窓（mujoco.viewer）を出す時に入れる。入っていれば、本当の時間の速さで動きを見せる


def frame(env):
    """物理を 1 コマ（20ms）進める。窓があれば表示を更新し、本当の時間の速さに合わせる。"""
    t0 = time.perf_counter()
    for _ in range(STEPS_PER_FRAME):
        mujoco.mj_step(env.m, env.d)
    if VIEWER is not None:
        VIEWER.sync()
        wait = env.m.opt.timestep * STEPS_PER_FRAME - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)


def rotvec_to_R(rv):
    R, _ = cv2.Rodrigues(np.asarray(rv, np.float64).reshape(3, 1))
    return R


def R_to_rotvec(R):
    rv, _ = cv2.Rodrigues(np.asarray(R, np.float64))
    return rv.ravel()


FORCE_MODE = os.environ.get("SIM_FORCE", "sum")   # net：力センサーのまね（向きを入れて足す）。sum：前と同じ（大きさの合計）
STOP_FORCE = 60.0             # ハンドにかかる力がこれ（N）を超えたら、実機の保護停止のように止める


FEEDFWD = os.environ.get("SIM_FEEDFWD", "1") == "1"   # 腕の目標に速さの分を先回りして足す（0 で前と同じ）
TRACE = os.environ.get("SIM_TRACE", "0") == "1"   # 1 なら、動くたびに止まった時の目標とのずれを出す


class SimRobot:
    """RTDEControl と RTDEReceive をまとめて 1 つにした、にせのロボット。
    実機の UR5e は強く当たると自分で止まる（保護停止）。まねて、腕を動かしている間にハンドへ STOP_FORCE N を超える力がかかったら止める
    （止めないと、腕の強い力で指が物の中へめり込んだ。10/16）。"""

    def __init__(self, env):
        self.env, self.ik = env, IK(env)
        base = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_BODY, "g_base")
        self.hand_bodies = {b for b in range(env.m.nbody) if b == base or self._under(b, base)}

    def _under(self, b, root):
        while b > 0:
            b = int(self.env.m.body_parentid[b])
            if b == root:
                return True
        return False

    def hand_force(self, detail=False):
        """ハンド（指・本体）と、ほかの物との接触の力の合計（N）。detail なら（合計, [(力, ハンドの部品, 当たった物)]）。"""
        m, d = self.env.m, self.env.d
        f = np.zeros(6)
        total, parts = 0.0, []
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 in self.hand_bodies) != (b2 in self.hand_bodies):
                mujoco.mj_contactForce(m, d, i, f)
                total += abs(f[0])
                g_hand, g_other = (c.geom1, c.geom2) if b1 in self.hand_bodies else (c.geom2, c.geom1)
                name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or mujoco.mj_id2name(
                    m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or str(g)
                parts.append((abs(f[0]), name(g_hand), name(g_other)))
        return (total, sorted(parts, reverse=True)) if detail else total

    # ---- RTDEReceive と同じ
    def getActualTCPPose(self):
        return self.env.tcp_like()

    def net_force(self):
        """ハンドがほかの物から受ける力を、向きを入れて足したもの（根元の向き、N）。指が物をはさむ力は左右で打ち消し合う（実機の力センサーと同じ）。"""
        m, d = self.env.m, self.env.d
        f, tot = np.zeros(6), np.zeros(3)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 in self.hand_bodies) != (b2 in self.hand_bodies):
                mujoco.mj_contactForce(m, d, i, f)
                fw = c.frame.reshape(3, 3).T @ f[:3]                 # 接触の向き → 根元の向き（geom1 が geom2 を押す力）
                tot += -fw if b1 in self.hand_bodies else fw
        return tot

    def getActualTCPForce(self):
        """実機の getActualTCPForce のまね。SIM_FORCE=net：ハンドが受ける力を向きを入れて足す（はさむ力は打ち消し合う。10/23 安全の決まりの確かめ用）。
        標準（sum）：ハンドが物や机から受ける力の大きさを z に入れて返す（向きは見ない。はさむ力も入る）。"""
        if FORCE_MODE == "net":
            return list(self.net_force()) + [0.0, 0.0, 0.0]
        return [0.0, 0.0, float(self.hand_force()), 0.0, 0.0, 0.0]

    def getActualQ(self):
        return list(self.env.d.qpos[:6])

    def getActualQd(self):
        return list(self.env.d.qvel[:6])

    # ---- RTDEControl と同じ
    def getInverseKinematics(self, pose, qnear=None):
        q0 = np.array(qnear if qnear is not None else self.env.d.qpos[:6], float)
        return list(self.ik.solve(q0, np.array(pose[:3]), rotvec_to_R(pose[3:]), iters=60))

    def isPoseWithinSafetyLimits(self, pose):
        return True

    def moveL(self, pose, speed=0.05, accel=0.2):
        """実機の moveL と同じく、加速 → 等速 → 減速（台形）で、道具の点をまっすぐ動かす。止まったら返る（実機と同じ）。
        前は等速で動かした後、ぴったり止まるまで最大 2 秒待っていて、実機より遅く測っていた（10/16）。"""
        env = self.env
        p0 = np.array(env.d.site_xpos[env.site])
        R0 = env.d.site_xmat[env.site].reshape(3, 3).copy()
        p1, R1 = np.array(pose[:3]), rotvec_to_R(pose[3:])
        dR = R1 @ R0.T
        ang = np.linalg.norm(R_to_rotvec(dR))
        dist = max(np.linalg.norm(p1 - p0), ang * 0.1)            # 回るだけの時も、回る量に応じて時間をかける（0.1m ≒ 1rad）
        v, a = max(speed, 1e-3), max(accel, 1e-3)
        ta = v / a
        if a * ta * ta > dist:                                    # 最高速度に届かない（三角形）
            ta = (dist / a) ** 0.5
            T = 2 * ta
        else:
            T = 2 * ta + (dist - a * ta * ta) / v
        dt = env.m.opt.timestep * STEPS_PER_FRAME
        n = max(1, int(np.ceil(T / dt)))

        def frac(t):
            if t >= T:
                return 1.0
            if t < ta:
                d_ = 0.5 * a * t * t
            elif t > T - ta:
                d_ = dist - 0.5 * a * (T - t) ** 2
            else:
                d_ = 0.5 * a * ta * ta + v * (t - ta)
            return min(1.0, d_ / dist) if dist > 0 else 1.0
        def dev(p):
            """動き始めと目標を結ぶ真っすぐな線からの、ずれ（m）。"""
            u = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
            w = p - p0
            return float(np.linalg.norm(w - (w @ u) * u))
        q = env.d.qpos[:6].copy()
        # 関節は「目標との差 × kp − 速さ × kd」で動くので、目標を追う時に速さの分だけ遅れる（kd/kp = 0.2 秒。動いた後に 3mm 残っていた、10/16）。
        # 目標の角度に「目標の速さ × kd/kp」を足して、遅れを打ち消す
        lead = (-env.m.actuator_biasprm[:6, 2] / env.m.actuator_gainprm[:6, 0]) if FEEDFWD else np.zeros(6)
        for i in range(1, n + 1):
            s_ = frac(i * dt)
            R = rotvec_to_R(R_to_rotvec(dR) * s_) @ R0
            q_prev = q
            q = self.ik.solve(q, p0 + (p1 - p0) * s_, R, iters=8)
            env.d.ctrl[:6] = q + lead * (q - q_prev) / dt
            frame(env)
            if self.hand_force() > STOP_FORCE:
                env.d.ctrl[:6] = env.d.qpos[:6]                  # その場で止まる
                frame(env)
                tot, parts = self.hand_force(detail=True)
                what = "、".join(f"{a_}→{b_} {fn:.0f}N" for fn, a_, b_ in parts[:3])
                tip = env.d.site_xpos[env.site]
                raise SystemExit(f"保護停止：ハンドに {tot:.0f}N の力がかかった（{what}）。指先は机から {(tip[2]) * 1000:.1f}mm、"
                                 f"目標まで {np.linalg.norm(tip - p1) * 1000:.0f}mm、目標の高さ {p1[2] * 1000:.1f}mm、"
                                 f"動き始めから {np.linalg.norm(tip - p0) * 1000:.0f}mm、真っすぐな線から {dev(tip) * 1000:.1f}mm ずれ")
        env.d.ctrl[:6] = q
        for _ in range(15):                                      # 腕の位置の遅れが追いつくまで（最大 0.3 秒）
            if np.linalg.norm(env.d.site_xpos[env.site] - p1) < 0.0005:
                break
            frame(env)
        if TRACE:
            tip = env.d.site_xpos[env.site]
            Rn = env.d.site_xmat[env.site].reshape(3, 3)
            err_ang = math.degrees(np.linalg.norm(R_to_rotvec(R1 @ Rn.T)))
            print(f"  [動き] {np.linalg.norm(p1 - p0) * 1000:.0f}mm・{math.degrees(ang):.0f}° 動いた後：目標から 横 "
                  f"{np.linalg.norm((tip - p1)[:2]) * 1000:.2f}mm・縦 {(tip - p1)[2] * 1000:+.2f}mm、向き {err_ang:.2f}° ずれ")
        return True

    def servoL(self, pose, speed=0.0, accel=0.0, t=0.02, lookahead=0.1, gain=300):
        """実機の servoL のまね：道具の点を、t 秒かけて pose へ動かす（止まらずに次の servoL へつなぐ。10/22）。
        t 秒を 20ms のコマに分け、今の位置から pose までまっすぐ進める。ハンドに強い力がかかったら moveL と同じく止める。"""
        env = self.env
        p0 = np.array(env.d.site_xpos[env.site])
        R0 = env.d.site_xmat[env.site].reshape(3, 3).copy()
        p1, R1 = np.array(pose[:3]), rotvec_to_R(pose[3:])
        dR = R1 @ R0.T
        dt = env.m.opt.timestep * STEPS_PER_FRAME
        n = max(1, int(round(t / dt)))
        lead = (-env.m.actuator_biasprm[:6, 2] / env.m.actuator_gainprm[:6, 0]) if FEEDFWD else np.zeros(6)
        q = env.d.qpos[:6].copy()
        for i in range(1, n + 1):
            s_ = i / n
            q_prev = q
            q = self.ik.solve(q, p0 + (p1 - p0) * s_, rotvec_to_R(R_to_rotvec(dR) * s_) @ R0, iters=8)
            env.d.ctrl[:6] = q + lead * (q - q_prev) / dt
            frame(env)
            if self.hand_force() > STOP_FORCE:
                env.d.ctrl[:6] = env.d.qpos[:6]
                frame(env)
                tot, parts = self.hand_force(detail=True)
                raise SystemExit(f"保護停止（servoL）：ハンドに {tot:.0f}N の力がかかった（" + "、".join(f"{a_}→{b_} {fn:.0f}N" for fn, a_, b_ in parts[:3]) + "）")
        return True

    def servoStop(self, a=10.0):
        self.env.d.ctrl[:6] = self.env.d.qpos[:6]
        return True

    def initPeriod(self):
        return time.perf_counter()

    def waitPeriod(self, t0):
        pass                                                     # コマごとの時間合わせは frame がしている

    def stopScript(self):
        pass

    def disconnect(self):
        pass


class SimHand:
    """hand.HandClient と同じ使い方の、にせのハンド（シミュレーションの 2F-85）。"""

    def __init__(self, env, log=print):
        self.env, self.log = env, log

    def _settle(self, frames=150):
        """指が止まるまで待つ（最大 frames × 20ms）。"""
        hist = []
        for i in range(frames):
            frame(self.env)
            hist.append(self.pos_mm())
            if i > 25 and max(hist[-10:]) - min(hist[-10:]) < 0.1:     # 0.2 秒の間ほとんど動かなければ止まったとみなす
                break

    def pos_mm(self):
        return self.env.pad_gap() * 1000.0

    def check(self, why):
        self.log(f"  [確認] {why}の前: 開き量 {self.pos_mm():.1f}mm（シミュレーション）")
        return self.pos_mm()

    def open(self, wait=3.0):
        self.check("開く")
        self.env.set_finger(False)
        self._settle()
        return self.pos_mm()

    def move_mm(self, mm, wait=3.0):
        """開き量 mm まで開く（実機の HandClient.move_mm。離す時に全開にせず、物の幅＋少しだけ開くのに使う。10/19 に足した）。
        指 1 本の目標は開き量の半分。指の目標を決められないハンド（2F-85）は全開にする。"""
        if not hasattr(self.env, "jq"):
            return self.open(wait)
        import sim_esg1
        self.check(f"{mm:.1f}mm へ動かす")
        self.env.d.ctrl[self.env.act] = min(sim_esg1.STROKE, max(0.0, mm / 2000.0))
        self._settle()
        return self.pos_mm()

    def grip(self, force=50, wait=3.0):
        self.check(f"力 {force}% で閉じる")
        if hasattr(self.env, "set_force"):                   # 実機のハンド（ESG1）は力を % で決める
            self.env.set_force(force)
        self.env.set_finger(True)
        self._settle()
        self.report_offset()
        return self.pos_mm()

    def grip_start(self, force=50):
        """閉じ始めるだけ（指が止まるのは frame を進める側が待つ）。"""
        self.check(f"力 {force}% で閉じ始める")
        if hasattr(self.env, "set_force"):
            self.env.set_force(force)
        self.env.set_finger(True)

    def report_offset(self):
        """閉じた時の、左右の指の真ん中と、いちばん近い道具の中心のずれ（mm）を出す（指のずれの補正の確かめ用、10/16）。"""
        env = self.env
        if not hasattr(env, "pads") or not hasattr(env, "tool_q"):
            return
        mid = (env.d.geom_xpos[env.pads[0]] + env.d.geom_xpos[env.pads[1]]) / 2
        best = min(env.tool_q, key=lambda t: np.linalg.norm(env.d.qpos[env.tool_q[t]:env.tool_q[t] + 2] - mid[:2]))
        q = env.d.qpos[env.tool_q[best]:env.tool_q[best] + 7]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q[3:])
        ax = R.reshape(3, 3)[:, 0]                                   # 道具の長い向き（x）
        d = (mid - q[:3])[:2]
        along = float(d @ ax[:2] / max(np.linalg.norm(ax[:2]), 1e-9))
        self.log(f"  [指の真ん中] {best} の中心から {np.linalg.norm(d) * 1000:.1f}mm（長い向きに {along * 1000:+.1f}mm）")

    def close_port(self):
        pass
