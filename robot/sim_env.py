"""UR5e と積み木の机を MuJoCo で動かすシミュレーター（本物のロボットの代わり）。

  - 真上のカメラ（640x480）の画像を返す
  - 積み木の置き方・ロボットの姿勢・明るさ・色をランダムにできる（YOLO の学習画像づくり用）
  - 画像のどのピクセルが何の物かが分かるので、学習用の枠（正解）を自動で作れる
"""
import math
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "ur5e_table.xml")
W, H = 640, 480
BLOCKS = ["red", "blue", "green"]
CLASSES = ["red_block", "blue_block", "green_block", "box"]
HALF = 0.025                       # 積み木の半分の大きさ（5cm 角）
BIN = np.array([-0.30, 0.45])      # 箱（黄色い置き場）の中心
BIN_HALF = 0.07
AREA = ((-0.10, 0.30), (0.30, 0.65))   # 積み木を置く範囲（x, y）
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
CAM_POS = np.array([0.0, 0.5, 1.4])
CAM_FOVY = 45.0


class Env:
    def __init__(self, seg=False):
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, H, W)
        self.seg = None
        if seg:
            self.seg = mujoco.Renderer(self.m, H, W)
            self.seg.enable_segmentation_rendering()
        self.geom = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("bin", "floor")}
        for b in BLOCKS:
            body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
            self.geom[b] = int(self.m.body_geomadr[body])
        self.qadr = {b: int(self.m.jnt_qposadr[self.m.body_jntadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b)]])
                     for b in BLOCKS}
        self.vadr = {b: int(self.m.jnt_dofadr[self.m.body_jntadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b)]])
                     for b in BLOCKS}
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.base_rgba = {b: self.m.geom_rgba[self.geom[b]].copy() for b in BLOCKS}
        self.base_light = self.m.light_diffuse.copy()
        self.base_head = self.m.vis.headlight.diffuse.copy()
        self.reset()

    # ------------------------------------------------------------ 状態
    def reset(self, rng=None, n_in_bin=0):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:6] = HOME
        self.d.ctrl[:6] = HOME
        rng = rng or np.random.default_rng(0)
        self.place_blocks(rng, n_in_bin)
        mujoco.mj_forward(self.m, self.d)

    def place_blocks(self, rng, n_in_bin=0):
        names = list(BLOCKS)
        rng.shuffle(names)
        placed = []
        for i, b in enumerate(names):
            in_bin = i < n_in_bin
            for _ in range(200):
                if in_bin:
                    xy = BIN + rng.uniform(-BIN_HALF + HALF * 1.5, BIN_HALF - HALF * 1.5, 2)
                else:
                    xy = np.array([rng.uniform(*AREA[0]), rng.uniform(*AREA[1])])
                if all(np.linalg.norm(xy - p) > 0.08 for p in placed):
                    break
            placed.append(xy)
            self.set_block(b, xy, HALF, yaw=rng.uniform(0, math.pi / 2))

    def set_block(self, b, xy, z, yaw=0.0):
        a = self.qadr[b]
        self.d.qpos[a:a + 3] = [xy[0], xy[1], z]
        self.d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        v = self.vadr[b]
        self.d.qvel[v:v + 6] = 0

    def hold(self, b, on):
        """吸着中の積み木は、ロボットの手首とぶつからないようにする。"""
        g = self.geom[b]
        self.m.geom_contype[g] = 0 if on else 1
        self.m.geom_conaffinity[g] = 0 if on else 1

    def block_pos(self, b):
        a = self.qadr[b]
        return self.d.qpos[a:a + 3].copy()

    def tool_pos(self):
        return self.d.site_xpos[self.site].copy()

    def randomize_look(self, rng):
        """明るさ・机の色・積み木の色を少し変える（学習画像を本物っぽくばらつかせる）。"""
        self.m.light_diffuse[:] = self.base_light * rng.uniform(0.6, 1.5)
        self.m.vis.headlight.diffuse[:] = self.base_head * rng.uniform(0.6, 1.4)
        self.m.geom_rgba[self.geom["floor"]][:3] = rng.uniform(0.45, 0.9) * np.array([1, 0.97, 0.92]) + rng.uniform(-0.05, 0.05, 3)
        for b in BLOCKS:
            self.m.geom_rgba[self.geom[b]][:3] = np.clip(self.base_rgba[b][:3] + rng.uniform(-0.12, 0.12, 3), 0, 1)

    def random_arm(self, rng):
        q = HOME + rng.uniform(-1, 1, 6) * np.array([0.9, 0.5, 0.6, 0.6, 0.3, 3.0])
        self.d.qpos[:6] = q
        mujoco.mj_forward(self.m, self.d)

    # ------------------------------------------------------------ カメラ
    def render(self):
        self.r.update_scene(self.d, camera="top")
        return self.r.render()                      # H×W×3 の RGB（uint8）

    def labels(self, min_px=40):
        """各物の見えているピクセルから枠を作る（YOLO 形式：種類, 中心x, 中心y, 幅, 高さ を 0〜1 で）。"""
        self.seg.update_scene(self.d, camera="top")
        s = self.seg.render()
        ids, types = s[..., 0], s[..., 1]
        out = []
        for k, name in enumerate(["red", "blue", "green", "bin"]):
            ys, xs = np.nonzero((ids == self.geom[name]) & (types == int(mujoco.mjtObj.mjOBJ_GEOM)))
            if len(xs) < min_px:
                continue
            x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            out.append((k, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
        return out

    # ------------------------------------------------------------ 画像の位置 → 机の上の位置
    @staticmethod
    def pixel_to_table(u, v, z=2 * HALF):
        """真上のカメラの画像の位置 (u, v) を、高さ z の面の上の (x, y) に直す（本物ではカメラの位置合わせで求める値）。"""
        f = (H / 2) / math.tan(math.radians(CAM_FOVY) / 2)
        dist = CAM_POS[2] - z
        return np.array([CAM_POS[0] + (u - W / 2) * dist / f, CAM_POS[1] - (v - H / 2) * dist / f])
