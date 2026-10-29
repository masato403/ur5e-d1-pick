"""ドライバー・ペン・ネジを作業台の上にばらばらに置いた UR5e のシミュレーター（sim_env.Env と同じ使い方）。"""
import math
import os

import mujoco
import numpy as np

from sim_env import HOME, Env

HERE = os.path.dirname(os.path.abspath(__file__))
OBJECTS = ["screwdriver", "pen", "screw1", "screw2"]
KIND = {"screwdriver": "screwdriver", "pen": "pen", "screw1": "screw", "screw2": "screw"}
BIN = np.array([-0.32, 0.45])
BIN_HALF = 0.10
AREA = ((-0.10, 0.30), (0.30, 0.65))


CAM_Z = 0.9                 # カメラの高さ（物が小さいので、積み木のとき（1.4m）より近づける）


class ToolEnv(Env):
    XML = os.path.join(HERE, "ur5e_tools.xml")
    SITE = "attachment_site"            # 工具の先（逆運動学で動かす点）
    OBJ_NAMES = OBJECTS                 # 模型に入っている物（sim_real で前の試験の物を消した時は空にする。10/23）

    @staticmethod
    def pixel_to_table(u, v, z=0.01):
        """画像の位置 → 作業台の上の位置（高さ z の面）。本物ではカメラの位置合わせで求める。"""
        f = (480 / 2) / math.tan(math.radians(45.0) / 2)
        dist = CAM_Z - z
        return np.array([(u - 640 / 2) * dist / f, 0.5 - (v - 480 / 2) * dist / f])

    def __init__(self):
        self.m = self.build_model()
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, 480, 640)
        self.objects = list(self.OBJ_NAMES)
        self.bin, self.bin_half = BIN, BIN_HALF
        bid = {o: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, o) for o in self.objects}
        self.body = bid
        self.qadr = {o: int(self.m.jnt_qposadr[self.m.body_jntadr[b]]) for o, b in bid.items()}
        self.vadr = {o: int(self.m.jnt_dofadr[self.m.body_jntadr[b]]) for o, b in bid.items()}
        self.z0 = {o: float(self.m.body_pos[b][2]) for o, b in bid.items()}      # 寝かせたときの中心の高さ
        self.geoms = {o: [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b] for o, b in bid.items()}
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, self.SITE)
        self.reset()

    def build_model(self):
        """モデルを作る（指で挟むハンドを付けた版では、ここを差し替える）。"""
        return mujoco.MjModel.from_xml_path(self.XML)

    def reset(self, rng=None, n_in_bin=0):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:6] = HOME
        self.d.ctrl[:6] = HOME
        rng = rng or np.random.default_rng(0)
        names = list(self.objects)
        rng.shuffle(names)
        placed = []
        for i, o in enumerate(names):
            for _ in range(300):
                if i < n_in_bin:
                    xy = self.bin + rng.uniform(-0.05, 0.05, 2)
                else:
                    xy = np.array([rng.uniform(*AREA[0]), rng.uniform(*AREA[1])])
                if all(np.linalg.norm(xy - p) > 0.12 for p in placed):
                    break
            placed.append(xy)
            self.set_obj(o, xy, yaw=rng.uniform(0, 2 * math.pi))
        mujoco.mj_forward(self.m, self.d)

    def set_obj(self, o, xy, z=None, yaw=None):
        a = self.qadr[o]
        self.d.qpos[a:a + 3] = [xy[0], xy[1], self.z0[o] if z is None else z]
        if yaw is not None:
            self.d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        v = self.vadr[o]
        self.d.qvel[v:v + 6] = 0

    def obj_pos(self, o):
        a = self.qadr[o]
        return self.d.qpos[a:a + 3].copy()

    def hold(self, o, on):
        for g in self.geoms[o]:
            self.m.geom_contype[g] = 0 if on else 1
            self.m.geom_conaffinity[g] = 0 if on else 1

    def object_geoms(self):
        return [g for o in self.objects for g in self.geoms[o]]

    def empty_table(self):
        """物を遠くへどけ、腕をカメラに映らない向きへ回して、何もない作業台を撮る。"""
        keep = {o: self.d.qpos[self.qadr[o]:self.qadr[o] + 7].copy() for o in self.objects}
        q = self.d.qpos[:6].copy()
        for i, o in enumerate(self.objects):
            self.set_obj(o, np.array([5.0 + i, 5.0]))
        self.d.qpos[:6] = HOME + np.array([np.pi, 0, 0, 0, 0, 0])
        mujoco.mj_forward(self.m, self.d)
        img = self.render()
        for o in self.objects:
            self.d.qpos[self.qadr[o]:self.qadr[o] + 7] = keep[o]
        self.d.qpos[:6] = q
        mujoco.mj_forward(self.m, self.d)
        return img
