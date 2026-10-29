"""ROBOTIQ 2F-85（指で挟むハンド）を付けた UR5e の作業台シミュレーター（sim_tools.ToolEnv と同じ使い方）。

  UR5e の手首の先（attachment_site）に、mujoco_menagerie の robotiq_2f85 を MjSpec で取り付ける。
  工具の先（逆運動学で動かす点）は、2 本の指の間のつかむ点（g_pinch）。
  組み立てたモデルは robot/ur5e_tools_2f85.xml にも書き出す（ビューアーなどで開く用）。
"""
import os

import mujoco
import numpy as np

from sim_tools import HERE, ToolEnv

G_XML = os.path.join(HERE, "..", "models", "menagerie", "m", "robotiq_2f85", "2f85.xml")
OUT_XML = os.path.join(HERE, "ur5e_tools_2f85.xml")
FINGER_OPEN, FINGER_CLOSE = 0.0, 255.0        # 指の制御の値（0 = 開く、255 = 閉じる）


def build_spec():
    spec = mujoco.MjSpec.from_file(os.path.join(HERE, "ur5e_tools.xml"))
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC      # ハンドの元の設定と同じ（摩擦でつかむときに滑りにくい）
    spec.option.impratio = 10
    gs = mujoco.MjSpec.from_file(os.path.abspath(G_XML))
    gdir = os.path.join(os.path.dirname(os.path.abspath(G_XML)), "assets")
    for m in gs.meshes:                                    # 読み込み先のフォルダが違うので、メッシュは絶対パスにする
        m.file = os.path.join(gdir, m.file)
    gs.meshdir = ""
    spec.site("attachment_site").attach_body(gs.body("base_mount"), "g_", "")
    # 手首のカメラ：ハンドの根元に、指の向き（下向き）を見るように付ける。指が視野の真ん中をふさがないよう 6cm 横にずらす。
    # MuJoCo のカメラは自分の -z の向きを見るので、x 軸のまわりに 180° 回して、ハンドの +z（指の向き）を見るようにする
    spec.body("g_base").add_camera(name="wrist", pos=[0.06, 0, 0.03], quat=[0, 1, 0, 0], fovy=60)
    add_wrist_cam_shape(spec.body("g_base"))
    tune(spec)
    return spec


def add_wrist_cam_shape(body):
    """手首カメラの形（Intel RealSense D405 相当 42×42×23mm）と取り付け金具。
    見た目だけ（当たり判定なし）。g_base は重さを直接決めてあるので、形を足しても重さは変わらない。
    レンズ面（z=0.03）より後ろ側に置くので、手首カメラの視界はふさがない。"""
    vis = dict(contype=0, conaffinity=0, group=2)
    body.add_geom(name="wrist_cam_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.021, 0.021, 0.0115],
                  pos=[0.06, 0, 0.0185], rgba=[0.75, 0.76, 0.78, 1], **vis)
    for side, y in (("l", -0.009), ("r", 0.009)):
        body.add_geom(name="wrist_cam_lens_" + side, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0045, 0.0005, 0],
                      pos=[0.06, y, 0.0295], rgba=[0.02, 0.02, 0.03, 1], **vis)
    body.add_geom(name="wrist_cam_bracket", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.016, 0.003],
                  pos=[0.033, 0, 0.010], rgba=[0.149, 0.149, 0.149, 1], **vis)


# 物理の調整（環境変数で試せるようにしておく。決めた値を既定値にする）
TUNE = {"spring": 1.0,   # 元のモデルは 0.05：弱いと指先が曲がって細い物を押し出してしまう（0/12 → 1.0 で 12/12）
        "obj_friction": 1.0,
        # 物の接触：6 ＝ すべり・ねじれ・転がりの摩擦をすべて計算する（3 だと転がりの摩擦がなく、寝かせたネジが 8cm 以上転がり続けた）
        "obj_condim": 6,
        "obj_roll": 0.003}       # 転がり摩擦：ネジを秒速 20cm で押すと 0.06 秒・0.5cm で止まる（本物のネジをつついた時に近い）


def tune(spec):
    t = {k: type(v)(os.environ.get("GRIP_" + k.upper(), v)) for k, v in TUNE.items()}
    for j in spec.joints:
        if j.name.endswith("spring_link_joint"):
            j.stiffness = [t["spring"], 0, 0]                     # 指先（パッド）を平行に保つばねの強さ
    for o in ("screwdriver", "pen", "screw1", "screw2"):
        for g in spec.body(o).geoms:
            g.friction = [t["obj_friction"], 0.005, t["obj_roll"]]
            g.condim = t["obj_condim"]


class GripEnv(ToolEnv):
    SITE = "g_pinch"

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
        super().__init__()
        self.finger_j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "g_right_driver_joint")
        self.finger_q = int(self.m.jnt_qposadr[self.finger_j])
        self.pads = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in
                     ("g_right_pad1", "g_right_pad2", "g_left_pad1", "g_left_pad2")]

    def finger(self):
        """指の閉じ具合（駆動の関節の角度：0 = 全開、約 0.8 = 全閉）。"""
        return float(self.d.qpos[self.finger_q])

    def pad_gap(self):
        """左右の指先（パッド）の内側のすき間（m）。何も挟んでいなければ 0 近くまで閉じる。
        2F-85 は指先が物に当たった後も指の付け根の角度が進む（物を包むように曲がる）ので、
        物を挟んでいるかは、関節の角度より、このすき間で見る方が確か（本物ではハンドが返す指の位置に当たる）。"""
        r, l = self.pads[0], self.pads[2]
        return float(np.linalg.norm(self.d.geom_xpos[r] - self.d.geom_xpos[l]) - 2 * self.m.geom_size[r][1])

    def set_finger(self, closed):
        self.d.ctrl[6] = FINGER_CLOSE if closed else FINGER_OPEN

    def tool_yaw(self):
        """指が閉じる向き（つかむ点の y 軸）から求めた、手首の向き（物の軸の向きと同じ角度で表す）。"""
        R = self.d.site_xmat[self.site].reshape(3, 3)
        return float(np.arctan2(R[0, 1], -R[1, 1]))
