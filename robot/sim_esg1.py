"""実機のハンド（TAIYO ESG1-SS-4225 ＋ 3D プリントの VLA フィンガ）を付けた UR5e の作業台シミュレーター。
sim_grip.GripEnv（ROBOTIQ 2F-85）と同じ使い方（set_finger・pad_gap・つかむ点 g_pinch・ハンドの根元 g_base）。

  寸法は実機のハンドの部品の寸法から（10/12 作成、10/16 に読み込み）：
    ・UR5e アダプタ：丸、直径 80、厚さ 12。UR の穴にはまる凸は長さに足さない
    ・三菱の丸フランジ：厚さは資料に無い。指先までの長さが実機（TCP_2 211mm − 56mm = 約 155mm）に合うよう 12 とした（推定）
    ・ESG1 本体：R60（開閉の向き）× Q46 × 高さ 83。爪の板厚 3、爪の幅 E24、爪の開閉の向きの幅 16.5
    ・指：ベースの板厚 10、指の長さ 45（爪の取付面から）、付け根の厚さ 6.5 → 先端 4.5、幅 24 → 12。
      内側の面は爪の位置決めピンから 5mm 内側 → 指先のすき間 = ピン間 G − 10（G 12〜35.5 → 2〜25.5mm）
  実機に合わせた所：
    ・開閉の向きは、道具の座標の y（実機の grip_geom.json：close_dir_tool ≈ −y）
    ・指の真ん中は、道具の点から x へ 約 5mm ずれている（grip_geom.json：bias_tool_mm。向きは実機の写真と重ねて決めた）。つかむ点 g_pinch は実機の道具の点と同じく
      フランジの真ん中の線の上に置き、ずれは実機用のプログラムの補正（bias）で直す
    ・開き量は実機のハンドが返す値と同じく、空で閉じて 0、全開で 24.8mm。力は 60% で 約 24N→ 100% で 40N とした
  当たり判定：本体・爪・指は、見た目と同じ形（指は 8 点の凸包）。指の内側の V 溝・横リブは入れていない（滑り止めは摩擦で代わりにする）
"""
import json
import os

import mujoco
import numpy as np

from sim_grip import tune
from sim_tools import HERE, ToolEnv

OUT_XML = os.path.join(HERE, "ur5e_tools_esg1.xml")
MM = 0.001
ADAPTER_T, ADAPTER_D = 12 * MM, 80 * MM
FLANGE_T, FLANGE_D = float(os.environ.get("ESG1_FLANGE_T", 12)) * MM, 75 * MM            # 三菱の丸フランジ（厚さは推定、上の説明）
BODY_X, BODY_Y, BODY_H = 46 * MM, 60 * MM, 83 * MM   # ESG1 本体（x：Q46、y：開閉の向き R60）
JAW_T, JAW_W, JAW_E = 3 * MM, 16.5 * MM, 24 * MM
BASE_T, BASE_HW = 10 * MM, 12 * MM
LEN, T0, T1, W1 = 45 * MM, 6.5 * MM, 4.5 * MM, 6 * MM
X_OUT = 11 * MM                                  # 爪のピンから外側の端まで（推定値）
# 指の真ん中の、道具の点からのずれ（grip_geom.json の bias_tool_mm：x 約 5.0mm、y 約 −0.8mm）。
# 向きは、実機の手首カメラの写真（10/15 height_A.jpg）と重ねて決めた：+bias で指の重なり 0.44、−bias で 0.14（compare_real.py、10/16）。
# 注意：実機用のプログラムは道具の点を「物 ＋ bias」へ動かす。指が「道具の点 ＋ bias」にあるなら、指は物から 2×bias（約 10mm）ずれる。
# 消しゴムでは、ずれが長い向き（閉じる向きと直角）なので、実機でもつかめていた可能性がある → シミュレーションで確かめる
_GEOM = json.load(open(os.path.join(HERE, "..", "real_ur", "grip_geom.json")))
_SIGN = float(os.environ.get("ESG1_BIAS_SIGN", 1))        # 合わせ込みの試し用（compare_real.py）
BIAS_X, BIAS_Y = (_SIGN * _GEOM["bias_tool_mm"][0] * MM, _SIGN * _GEOM["bias_tool_mm"][1] * MM)
# 実機のハンドが返す開き量は、何も挟まずに閉じると 0.0、全開で 24.8（10/15 の記録）。指の内側の面のすき間は閉じても 約 2mm 残る
# （推定）ので、返す値 = すき間 − GAP_MIN とする（すき間をそのまま返すと、空振りでも 2mm で「持てている」になった。10/16）
GAP_MIN, GAP_OPEN = 2 * MM, (2 + 24.8) * MM
FORCE_MAX = 40.0                                 # 力 100% の時の 1 本の指の押す力（N、推定）
Z_BODY = ADAPTER_T + FLANGE_T                    # 本体の上面（フランジから）
Z_JAW = Z_BODY + BODY_H                          # 爪の取付面（本体の下面）
Z_MOUNT = Z_JAW + JAW_T                          # 指のベースの取付面
PINCH_Z = Z_MOUNT + LEN                          # 指先（= つかむ点）。フランジから 0.155m
STROKE = (GAP_OPEN - GAP_MIN) / 2                # 1 本の指が動く量


def finger_mesh(spec, name):
    """指の茎（付け根 幅 24・厚さ 6.5 → 先端 幅 12・厚さ 4.5）。内側の面が y = 0、外側へ +y。z は取付面から指先へ。"""
    pts = []
    for y in (0.0, T0):
        for x in (-BASE_HW, BASE_HW):
            pts.append([x, y, 0.0])
    for y in (0.0, T1):
        for x in (-W1, W1):
            pts.append([x, y, LEN])
    spec.add_mesh(name=name, uservert=np.array(pts).flatten().tolist())


def add_hand(spec):
    vis = dict(contype=0, conaffinity=0, group=2)
    # 当たり判定を硬めにする（既定の solref 0.02 だと、腕の強い力で指が物の中へめり込み、離しても物がくっついたまま浮いた。10/16）
    hard = dict(condim=4, friction=[1.0, 0.005, 0.0001], solref=[0.005, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2])
    body = spec.site("attachment_site").parent.add_body(name="g_base", pos=spec.site("attachment_site").pos,
                                                        quat=spec.site("attachment_site").quat)
    body.add_geom(name="g_adapter", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[ADAPTER_D / 2, ADAPTER_T / 2, 0],
                  pos=[0, 0, ADAPTER_T / 2], rgba=[0.15, 0.15, 0.16, 1], mass=0.05, **vis)
    body.add_geom(name="g_flange", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[FLANGE_D / 2, FLANGE_T / 2, 0],
                  pos=[BIAS_X, BIAS_Y, ADAPTER_T + FLANGE_T / 2], rgba=[0.7, 0.7, 0.72, 1], mass=0.15, **vis)
    body.add_geom(name="g_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BODY_X / 2, BODY_Y / 2, BODY_H / 2],
                  pos=[BIAS_X, BIAS_Y, Z_BODY + BODY_H / 2], rgba=[0.62, 0.64, 0.66, 1], mass=0.6, **hard)
    body.add_site(name="g_pinch", pos=[0, 0, PINCH_Z], size=[0.003, 0, 0], rgba=[1, 0, 0, 0.5])
    for side, s in (("right", 1), ("left", -1)):
        finger_mesh(spec, f"g_{side}_stem")
        # 指の付け根の体：y = s × (GAP_MIN/2 + 動いた量)。指の内側の面が付け根の y にある
        f = body.add_body(name=f"g_{side}_finger", pos=[BIAS_X, BIAS_Y + s * GAP_MIN / 2, Z_JAW],
                          quat=[1, 0, 0, 0] if s > 0 else [0, 0, 0, 1])     # 左の指は z のまわりに 180° 回す（左右同じ部品）
        f.add_joint(name=f"g_{side}_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], range=[0, STROKE],
                    limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=20.0, armature=0.01)
        red = [0.8, 0.12, 0.1, 1]                                        # 実機の指は赤（hand_mask で赤を見る）
        f.add_geom(name=f"g_{side}_jaw", type=mujoco.mjtGeom.mjGEOM_BOX, size=[JAW_E / 2, JAW_W / 2, JAW_T / 2],
                   pos=[0, JAW_W / 2 - 0.5 * MM, JAW_T / 2], rgba=[0.55, 0.57, 0.6, 1], mass=0.03, **hard)
        f.add_geom(name=f"g_{side}_base", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BASE_HW, (X_OUT + 5 * MM) / 2, BASE_T / 2],
                   pos=[0, (X_OUT + 5 * MM) / 2, JAW_T + BASE_T / 2], rgba=red, mass=0.01, **hard)
        f.add_geom(name=f"g_{side}_pad1", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"g_{side}_stem",
                   pos=[0, 0, JAW_T], rgba=red, mass=0.01, **hard)
    spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="g_left_slide", name2="g_right_slide", data=[0, 1, 0, 0, 0] + [0] * 6)
    a = spec.add_actuator(name="g_fingers", target="g_right_slide", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a.set_to_position(kp=FORCE_MAX / (10 * MM))                          # 10mm 押し込むと全力（硬すぎると 2ms の刻みで暴れた）
    a.ctrlrange = [0, STROKE]
    a.forcerange = [-FORCE_MAX, FORCE_MAX]
    a.forcelimited = True
    a.ctrllimited = True
    return body


def build_spec():
    spec = mujoco.MjSpec.from_file(os.path.join(HERE, "ur5e_tools.xml"))
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    add_hand(spec)
    tune(spec)
    return spec


class Esg1Env(ToolEnv):
    SITE = "g_pinch"
    PINCH_Z = PINCH_Z

    def build_model(self):
        spec = build_spec()
        m = spec.compile()
        try:
            with open(OUT_XML, "w", encoding="utf-8") as fp:
                fp.write(spec.to_xml())
        except Exception:
            pass
        return m

    def __init__(self):
        self.force = 1.0
        super().__init__()
        self.jq = [int(self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"g_{s}_slide")])
                   for s in ("right", "left")]
        self.act = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, "g_fingers")
        self.pads = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"g_{s}_pad1") for s in ("right", "left")]
        self.reset()

    def reset(self, rng=None, n_in_bin=0):
        super().reset(rng, n_in_bin)
        if not hasattr(self, "jq"):                       # ToolEnv.__init__ の中から呼ばれた時（指の番号はまだ無い）
            return
        for q in self.jq:
            self.d.qpos[q] = STROKE
        self.d.ctrl[self.act] = STROKE
        mujoco.mj_forward(self.m, self.d)

    def finger(self):
        return float(1.0 - self.d.qpos[self.jq[0]] / STROKE)

    def pad_gap(self):
        """実機のハンドが返す開き量に当たる値（m）：指の内側の面のすき間 − GAP_MIN。何も挟まずに閉じると 0。"""
        return float(self.d.qpos[self.jq[0]] + self.d.qpos[self.jq[1]])

    def set_force(self, pct):
        """力（%）。実機の HandClient.grip(force) に合わせる。"""
        f = FORCE_MAX * max(1, min(100, pct)) / 100.0
        self.m.actuator_forcerange[self.act] = [-f, f]

    def set_finger(self, closed):
        self.d.ctrl[self.act] = 0.0 if closed else STROKE

    def tool_yaw(self):
        R = self.d.site_xmat[self.site].reshape(3, 3)
        return float(np.arctan2(R[0, 1], -R[1, 1]))
