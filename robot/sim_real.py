"""実機に近づけた作業台シミュレーター（sim_grip.GripEnv に足す。今までの場面・結果には手を付けない）。

  ・机に、実機と同じ A3 のカッターマット（450×320mm、暗い紫、10mm の方眼）を敷く
  ・手首に「実物のカメラ」（wrist_real）を、校正（real_ur/calib2/calib_result.json）で求めた取り付け位置・向き・画角で付ける
      実機の道具の点（TCP_2）は指先より TCP2_TO_TIP mm 先にあるので、その分を引いてシミュレーションのつかむ点（g_pinch）に合わせる
      画像は実機と同じ 1920×1080。レンズの曲がりは無い（実機用のプログラムでは曲がりの係数を 0 にして使う）
  ・写真（10/15）のペン・消しゴム・ドライバー（スタビー型）を、ざっくりの寸法で置く（TOOLS）
使い方（確かめ）:  python robot/test_sim_real.py
"""
import json
import math
import os

import cv2
import mujoco
import numpy as np
SIM_JPEG = int(os.environ.get("SIM_JPEG", "0"))     # 0 より大きいと、実物のカメラの画像をこの品質の JPEG にして戻す（10/20、MJPG の確かめ）

import sim_esg1
import sim_grip
from sim_grip import OUT_XML  # noqa: F401

HAND = os.environ.get("SIM_HAND", "esg1")   # esg1：実機のハンド（ESG1 ＋ 3D プリントの指、10/16〜）、2f85：前のハンド
BaseEnv = sim_esg1.Esg1Env if HAND == "esg1" else sim_grip.GripEnv
build_spec = sim_esg1.build_spec if HAND == "esg1" else sim_grip.build_spec
PINCH_Z = sim_esg1.PINCH_Z if HAND == "esg1" else 0.145       # ハンドの根元 g_base から、つかむ点 g_pinch までの高さ（m）

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "..", "real_ur")
CAL = json.load(open(os.path.join(REAL, "calib2", "calib_result.json")))
MAT_MM = (450.0, 320.0)                 # A3 のカッターマット
MAT_CENTER = (0.10, 0.48)               # マットの中心（m、シミュレーションの根元の座標）
MAT_MM0, MAT_CENTER0 = MAT_MM, MAT_CENTER   # 実機と同じ大きさ（箱・見張りの位置はこれで決める）
# 実機のマットは A2 くらい（横 約 58〜60cm。三脚の写真の目盛りで測った、10/22）。前は A3（450×320）で、物が近すぎ・はみ出した。
# 箱の位置・見張りの位置は前の A3 の大きさ（MAT_MM0）で決めたままにして、箱の無い右へ 150mm、上下へ 130mm 広げて 600×450 にする（標準）。
MAT_WIDE = float(os.environ.get("SIM_MAT_WIDE", "150"))   # マットを右（+x）へこれ（mm）だけ広げる。箱のある左の端は動かさない（0 で前の A3）
MAT_TALL = float(os.environ.get("SIM_MAT_TALL", "130"))   # マットを上下（y）へこれ（mm）だけ広げる。三脚の側へ広げ、SIM_MAT_NEAR で腕の側へ寄せる（0 で前の A3）
if MAT_WIDE > 0 or MAT_TALL > 0:
    MAT_MM = (MAT_MM0[0] + MAT_WIDE, MAT_MM0[1] + MAT_TALL)
    # 上下は、ロボットの側（-y）の端を A3 の時のままにし、手前（三脚の側、+y）へ広げる。ロボットの側へ広げると、
    # 腕の根元に近すぎて腕を曲げきれず、ハンドが上腕に当たって保護停止した（10/22）
    # SIM_MAT_NEAR（mm）だけマット全体を腕の側へ寄せる（10/22。奥の角が腕から 777mm で届かなかった）
    MAT_NEAR = float(os.environ.get("SIM_MAT_NEAR", "80"))
    MAT_CENTER = (MAT_CENTER0[0] + MAT_WIDE / 2000, MAT_CENTER0[1] - MAT_MM0[1] / 2000 + MAT_MM[1] / 2000 - MAT_NEAR / 1000)
# 並び（10/23、実機にできるだけ近づける）。real：マット・箱・三脚を、実機のロボットの根元からの位置（sim_layout_real.json）に置く。
# シミュレーターの腕の根元は 180° 回っている（ur5e.xml の base の quat 0 0 0 -1）ので、実機の (x, y) はシミュレーターの (-x, -y)。
# old：前の並び（箱を基準に三脚を置き、マットは根元から横へ約 330mm ずれていた）
LAYOUT = os.environ.get("SIM_LAYOUT", "old")   # 標準は前の並び（10/23。real は三脚の校正を直してから）
LAYOUT_REAL = json.load(open(os.path.join(HERE, "sim_layout_real.json"), encoding="utf-8")) if LAYOUT == "real" else None
if LAYOUT_REAL:
    _mx, _my = LAYOUT_REAL["mat"]["x"], LAYOUT_REAL["mat"]["y"]
    MAT_MM = (float(_mx[1] - _mx[0]), float(_my[1] - _my[0]))
    MAT_CENTER = (-(_mx[0] + _mx[1]) / 2000, -(_my[0] + _my[1]) / 2000)
    MAT_MM0, MAT_CENTER0 = MAT_MM, MAT_CENTER
MAT_Z = 0.0003                          # 床から少し浮かせる（見た目だけ。重なってちらつかないように）
PX_PER_MM = 4
TCP2_TO_TIP = 56.0                      # 実機の TCP_2 は、指先より 56mm 先（10/15 に測った）
IMG_W, IMG_H = 1920, 1080
F = float(np.array(CAL["K"])[0, 0])     # 実物のカメラの焦点距離（画素、1920×1080）
ERASER_MM = (52.0, 20.0, 12.0)          # 消しゴム（長さ・幅・厚さ。実機で測った値、10/15）
PEN_MM = (145.0, 11.0)                  # ペン（長さ・太さ。写真からざっくり）
DRIVER_MM = (38.0, 25.0, 37.0, 5.0)     # スタビー型ドライバー（持ち手の長さ・太さ、軸の長さ・太さ）
ROD_MM = (77.0, 11.0, 18.0)             # 銀の棒（長さ・太さ・片方の端の白いラベルの長さ）。10/16 の実機の写真で測った（長さ 73.7〜79.6mm）
# 3 つ目の道具：rod＝銀の棒（10/19 から。実機ではドライバーの代わりに銀の棒を使っている）、driver＝前のスタビー型ドライバー
TOOL3 = os.environ.get("SIM_TOOL3", "rod")
TOOLS = ["eraser", "r_pen", "r_rod" if TOOL3 == "rod" else "r_driver"]   # 実機に近い場面の道具
TOOL_H = {"eraser": ERASER_MM[2], "r_pen": PEN_MM[1], "r_driver": DRIVER_MM[1], "r_rod": ROD_MM[1]}   # 寝かせた時の高さ（mm）
LYING = [0.70710678, 0, 0.70710678, 0]  # 円柱（軸が z）を寝かせる（y のまわりに 90°）
LIGHT = {"head": float(os.environ.get("SIMREAL_HEAD", 0.12)), "ambient": float(os.environ.get("SIMREAL_AMB", 0.30)),
         "top": float(os.environ.get("SIMREAL_TOP", 0.65))}
# 箱：実機と同じステンレスのトレー（10/19 に決まった）。縁の外側 横 150mm（x）× 縦 200mm（y）・高さ 50mm、壁は外へ広がる斜め。
# マットの上の左の端に置く（縁の折り返しの外側がマットの端から 10mm 内。10/19）。
# 前の仮の箱は、内側 90×170mm・高さ 50mm の段ボールで、マットの左の端に乗っていた（SIM_BOX=card で前の箱）
BOX_KIND = os.environ.get("SIM_BOX", "tray")
if BOX_KIND == "tray":
    BOX_RIM, BOX_H, BOX_T, BOX_SLOPE = (0.150, 0.200), 0.050, 0.0012, 0.015   # 縁の外側・高さ・板の厚さ・壁の上下の内外の差（底は縁より片側 15mm 内）
    BOX_XY = (MAT_CENTER0[0] - MAT_MM0[0] / 2000 + 0.010 + 0.010 + BOX_RIM[0] / 2, 0.450)   # 端から 10mm・縁の折り返し 10mm
    BOX_IN = (BOX_RIM[0] - 2 * BOX_T, BOX_RIM[1] - 2 * BOX_T)                  # 縁の高さでの内側（箱に入ったかの判定に使う）
    BOX_BOTTOM = (BOX_RIM[0] - 2 * BOX_SLOPE, BOX_RIM[1] - 2 * BOX_SLOPE)      # 平らな底
    BOX_OUT = (BOX_RIM[0] + 0.020, BOX_RIM[1] + 0.020)                         # 上から見た外形（縁の折り返し 10mm を含む）
else:
    BOX_XY = (-0.076, 0.450)
    BOX_IN, BOX_H, BOX_T = (0.090, 0.170), 0.050, 0.003
    BOX_BOTTOM = BOX_IN
    BOX_OUT = (BOX_IN[0] + 2 * BOX_T, BOX_IN[1] + 2 * BOX_T)
if LAYOUT_REAL and BOX_KIND == "tray":                                          # 実機の箱の位置・向き（長い向きが x なら縁の外側を 200×150 に）
    _b = LAYOUT_REAL["box"]
    BOX_RIM = tuple(v / 1000 for v in _b["rim_outer_mm"])
    BOX_XY = (-_b["center"][0] / 1000, -_b["center"][1] / 1000)
    BOX_IN = (BOX_RIM[0] - 2 * BOX_T, BOX_RIM[1] - 2 * BOX_T)
    BOX_BOTTOM = (BOX_RIM[0] - 2 * BOX_SLOPE, BOX_RIM[1] - 2 * BOX_SLOPE)
    BOX_OUT = (BOX_RIM[0] + 0.020, BOX_RIM[1] + 0.020)
# 見張りの位置（物を探す時に、腕が上から見る所。m）。トレーをマットの上の左の端に置くと、前の位置（マットの真ん中から左へ 20mm）では
# 画面の中で物を置ける所が足りず、ペンを 3 場面とも置けなかった（10/19）。トレーの時は右へ 60mm ずらし、箱から離れた所を見る
OBS_XY = (MAT_CENTER0[0] + (0.04 if BOX_KIND == "tray" else -0.02), MAT_CENTER0[1] - 0.08)
if LAYOUT_REAL:                                                                   # 実機の三脚で見る時の指先の位置（track_real.LOOK）
    OBS_XY = (-LAYOUT_REAL["look_mm"][0] / 1000, -LAYOUT_REAL["look_mm"][1] / 1000)
K_SIM = np.array([[F, 0, IMG_W / 2], [0, F, IMG_H / 2], [0, 0, 1.0]])


MAT_LOOK = os.environ.get("SIM_MAT", "plain")   # real：実機の 10/16 夕方の写真に合わせたマット（fit_sim_mat.py が作る）。plain：前の一様な薄紫


def mat_texture(path):
    """マットの絵：暗い紫に、10mm ごとの明るい線（5cm ごとは少し太く）。
    SIM_MAT=real なら、実機の写真の明るさ・色に合わせた絵（robot/mat_texture_real.png）を使う（10/19 夜）。
    実機は地が暗いこげ茶紫（明るさ 56〜126）で右側が暗く、暗い所でペンを見落とした。それをシミュレーターでも起こすため"""
    real = os.path.join(HERE, "mat_texture_real.png")
    if MAT_LOOK == "real" and os.path.exists(real):
        return real
    w, h = int(MAT_MM[0] * PX_PER_MM), int(MAT_MM[1] * PX_PER_MM)
    img = np.full((h, w, 3), (152, 146, 155), np.uint8)            # BGR：紫がかった灰色（実機の画像で測った値：明るさ約 155、10/15）
    line = (150, 185, 215)                                        # BGR：明るい肌色の線
    for mm in range(0, int(MAT_MM[0]) + 1, 10):
        x = min(w - 1, mm * PX_PER_MM)
        cv2.line(img, (x, 0), (x, h - 1), line, 3 if mm % 50 == 0 else 2)
    for mm in range(0, int(MAT_MM[1]) + 1, 10):
        y = min(h - 1, mm * PX_PER_MM)
        cv2.line(img, (0, y), (w - 1, y), line, 3 if mm % 50 == 0 else 2)
    cv2.imwrite(path, img)
    return path


def cam_in_pinch():
    """実物のカメラの位置（m）と向き（MuJoCo のカメラの向き）を、つかむ点（g_pinch）から見た値で返す。"""
    t = np.array(CAL["t_cam2tcp_mm"], float)
    t[2] += TCP2_TO_TIP + float(os.environ.get("CAM_DZ", 0))     # TCP_2 → 指先（つかむ点）。CAM_DZ は合わせ込みの試し用（mm）
    R_cv = np.array(CAL["R_cam2tcp"], float)                     # OpenCV のカメラ：x 右、y 下、z 前
    R_mj = R_cv @ np.diag([1.0, -1.0, -1.0])                     # MuJoCo のカメラ：x 右、y 上、-z を見る
    return t / 1000.0, R_mj, R_cv


def add_real_scene(spec):
    tex = mat_texture(os.path.join(HERE, "mat_texture.png"))
    spec.add_texture(name="cutmat", type=mujoco.mjtTexture.mjTEXTURE_2D, file=tex)
    m = spec.add_material(name="cutmat", texrepeat=[1, 1], texuniform=False)
    m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "cutmat"
    spec.worldbody.add_geom(name="mat", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[MAT_MM[0] / 2000, MAT_MM[1] / 2000, 0.01],
                            pos=[MAT_CENTER[0], MAT_CENTER[1], MAT_Z], material="cutmat",
                            contype=0, conaffinity=0)
    obj = dict(friction=[1.0, 0.005, 0.003], condim=6, solref=[0.005, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2])   # 硬め（sim_esg1 と同じ）
    vis = dict(contype=0, conaffinity=0, mass=0.0001)
    bx, by = BOX_XY
    if BOX_KIND == "tray":
        # ステンレスのトレー（動かない）：平らな底と、外へ広がる斜めの壁 4 枚、縁の外へ折り返した平らな縁
        spec.add_material(name="tray_steel", rgba=[0.62, 0.63, 0.65, 1], specular=0.9, shininess=0.8, reflectance=0.25)
        steel = dict(material="tray_steel", friction=[0.6, 0.005, 0.003], condim=4)
        rw, rl = BOX_RIM[0] / 2, BOX_RIM[1] / 2
        bw, bl = BOX_BOTTOM[0] / 2, BOX_BOTTOM[1] / 2
        spec.worldbody.add_geom(name="box_floor", type=mujoco.mjtGeom.mjGEOM_BOX, size=[bw, bl, BOX_T / 2],
                                pos=[bx, by, MAT_Z + BOX_T / 2], **steel)
        ang = math.atan2(BOX_SLOPE, BOX_H)                                         # 壁の傾き（縦から外へ）
        slant = math.hypot(BOX_SLOPE, BOX_H) / 2
        zc = MAT_Z + BOX_H / 2
        for name, sx, sy, half in (("box_wx0", -1, 0, bl + BOX_SLOPE / 2), ("box_wx1", 1, 0, bl + BOX_SLOPE / 2),
                                   ("box_wy0", 0, -1, bw + BOX_SLOPE / 2), ("box_wy1", 0, 1, bw + BOX_SLOPE / 2)):
            mid = (bw + rw) / 2 if sx else (bl + rl) / 2                          # 壁の真ん中の、箱の真ん中からの距離
            if sx:   # x の向きの壁：y のまわりに傾ける
                q = [math.cos(sx * ang / 2), 0, math.sin(sx * ang / 2), 0]      # y のまわりに +θ 回すと上が +x へ倒れる（外へ広がる）
                size, pos = [BOX_T / 2, half, slant], [bx + sx * mid, by, zc]
            else:    # y の向きの壁：x のまわりに傾ける
                q = [math.cos(-sy * ang / 2), math.sin(-sy * ang / 2), 0, 0]    # x のまわりに −θ 回すと上が +y へ倒れる
                size, pos = [half, BOX_T / 2, slant], [bx, by + sy * mid, zc]
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, quat=q, **steel)
        for name, pos, size in (("box_lx0", [bx - rw - 0.005, by], [0.005, rl + 0.010]), ("box_lx1", [bx + rw + 0.005, by], [0.005, rl + 0.010]),
                                ("box_ly0", [bx, by - rl - 0.005], [rw, 0.005]), ("box_ly1", [bx, by + rl + 0.005], [rw, 0.005])):
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size + [BOX_T / 2],
                                    pos=pos + [MAT_Z + BOX_H - BOX_T / 2], **steel)       # 縁の外へ 10mm 折り返した平らな縁
    else:
        # 前の仮の箱：上が開いた段ボール箱（底と壁 4 枚、動かない）
        iw, il = BOX_IN[0] / 2, BOX_IN[1] / 2
        card = dict(rgba=[0.66, 0.50, 0.33, 1], friction=[1.0, 0.005, 0.003], condim=4)
        spec.worldbody.add_geom(name="box_floor", type=mujoco.mjtGeom.mjGEOM_BOX, size=[iw + BOX_T, il + BOX_T, BOX_T / 2],
                                pos=[bx, by, MAT_Z + BOX_T / 2], **card)
        for name, pos, size in (("box_wx0", [bx - iw - BOX_T / 2, by], [BOX_T / 2, il + BOX_T]),
                                ("box_wx1", [bx + iw + BOX_T / 2, by], [BOX_T / 2, il + BOX_T]),
                                ("box_wy0", [bx, by - il - BOX_T / 2], [iw, BOX_T / 2]),
                                ("box_wy1", [bx, by + il + BOX_T / 2], [iw, BOX_T / 2])):
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size + [BOX_H / 2],
                                    pos=pos + [MAT_Z + BOX_H / 2], **card)
    # 消しゴム：白い本体と、青白のカバー（見た目だけ）
    L, W, T = (v / 1000 for v in ERASER_MM)
    b = spec.worldbody.add_body(name="eraser", pos=[MAT_CENTER[0] - 0.03, MAT_CENTER[1] - 0.02, T / 2])
    b.add_freejoint()
    b.add_geom(name="eraser_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, W / 2, T / 2],
               rgba=[0.95, 0.95, 0.95, 1], mass=0.012, **obj)
    b.add_geom(name="eraser_sleeve", type=mujoco.mjtGeom.mjGEOM_BOX, size=[L * 0.42, W / 2 + 0.0003, T / 2 + 0.0003],
               pos=[L * 0.06, 0, 0], rgba=[0.12, 0.3, 0.85, 1], **vis)
    # ペン：青い半透明の軸、濃い青の握り、銀の先と端（寝かせて置く。軸の向きが x）
    PL, PD = PEN_MM[0] / 1000, PEN_MM[1] / 1000
    b = spec.worldbody.add_body(name="r_pen", pos=[MAT_CENTER[0] + 0.03, MAT_CENTER[1] + 0.04, PD / 2])
    b.add_freejoint()
    b.add_geom(name="r_pen_barrel", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PD / 2, PL * 0.36, 0], quat=LYING,
               pos=[0.012, 0, 0], rgba=[0.25, 0.45, 0.95, 0.75], mass=0.010, **obj)
    b.add_geom(name="r_pen_grip", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PD / 2 + 0.0003, PL * 0.14, 0], quat=LYING,
               pos=[-PL * 0.27, 0, 0], rgba=[0.08, 0.15, 0.6, 1], **vis)
    b.add_geom(name="r_pen_tip", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[PD * 0.3, PL * 0.06, 0], quat=LYING,
               pos=[-PL * 0.43, 0, 0], rgba=[0.8, 0.8, 0.82, 1], **vis)
    b.add_geom(name="r_pen_end", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[PD * 0.35, PL * 0.03, 0], quat=LYING,
               pos=[PL * 0.47, 0, 0], rgba=[0.8, 0.8, 0.82, 1], **vis)
    if TOOL3 == "rod":
        # 銀の棒：光る金属の丸い棒と、片方の端に巻いた白いラベル（見た目だけ。寝かせて置く。軸の向きが x）
        RL, RD, LL = (v / 1000 for v in ROD_MM)
        # 明るさ 0.74 では白い棒に描かれ、d1-3B が「それ以外」と答えた（5 回中 4 回）。0.40 ではマットと見分けられず見つからなかった。
        # 候補を比べて（robot/probe_rod_look.py、3 場面）、見つかるのも d1-3B の答えも 3/3 だった 0.55 にした（10/19）。SIM_ROD_GRAY で変えられる
        g = float(os.environ.get("SIM_ROD_GRAY", "0.55"))
        spec.add_material(name="steel", rgba=[g, g + 0.01, g + 0.03, 1], specular=1.0, shininess=0.95, reflectance=0.15)
        b = spec.worldbody.add_body(name="r_rod", pos=[MAT_CENTER[0] - 0.05, MAT_CENTER[1] + 0.05, RD / 2])
        b.add_freejoint()
        b.add_geom(name="r_rod_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[RD / 2, RL / 2, 0], quat=LYING,
                   material="steel", mass=0.040, **obj)
        b.add_geom(name="r_rod_label", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[RD / 2 + 0.0003, LL / 2, 0], quat=LYING,
                   pos=[-RL / 2 + LL / 2, 0, 0], rgba=[0.96, 0.96, 0.94, 1], **vis)
    else:
        # スタビー型ドライバー：青の太い持ち手と、銀の短い軸（寝かせて置く。持ち手の向きが -x、軸が +x）
        HL, HD, SL, SD = (v / 1000 for v in DRIVER_MM)
        b = spec.worldbody.add_body(name="r_driver", pos=[MAT_CENTER[0] - 0.05, MAT_CENTER[1] + 0.05, HD / 2])
        b.add_freejoint()
        b.add_geom(name="r_driver_handle", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[HD / 2, HL / 2, 0], quat=LYING,
                   pos=[-HL / 2, 0, 0], rgba=[0.15, 0.3, 0.85, 0.85], mass=0.030, **obj)
        b.add_geom(name="r_driver_shaft", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[SD / 2, SL / 2, 0], quat=LYING,
                   pos=[SL / 2, 0, 0], rgba=[0.78, 0.78, 0.8, 1], mass=0.008, **obj)
    # 実物のカメラ（つかむ点 g_pinch は g_base の z = PINCH_Z にある）
    t, R_mj, _ = cam_in_pinch()
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R_mj.flatten())
    fovy = math.degrees(2 * math.atan((IMG_H / 2) / F))
    cam = [t[0], t[1], PINCH_Z + t[2]]
    spec.body("g_base").add_camera(name="wrist_real", pos=cam, quat=list(q), fovy=fovy)
    if HAND == "esg1":
        add_cam_mount(spec.body("g_base"), np.array(cam), q)
    # 照明を実機に近づける：カメラから出る光（ヘッドライト）を弱め、天井からの光を主にして影も付ける。
    # 実機の画像では物の側面が上の面より暗い（10/15）。ヘッドライトが強いと、カメラを向いた側面まで明るくなる
    spec.visual.headlight.diffuse = [LIGHT["head"]] * 3
    spec.visual.headlight.ambient = [LIGHT["ambient"]] * 3
    for lt in spec.lights:
        lt.diffuse = [LIGHT["top"]] * 3
        lt.castshadow = os.environ.get("SIMREAL_SHADOW", "1") == "1"
        lt.dir = [0.15, 0.25, -1.0]
    spec.visual.global_.offwidth = IMG_W
    spec.visual.global_.offheight = IMG_H


def add_cam_mount(body, cam, q):
    """カメラ台（赤い 3D プリント、写真 10/16）：ESG1 本体の横の面（-x の側）にねじ 4 本で留めた板と、
    そこから斜め下・外へ伸びる腕、その先の小さなカメラ。形のデータが無いので写真からざっくり。見た目だけ（当たり判定なし）。
    カメラの位置と向きは校正の値（calib_result.json）そのもの。"""
    E = sim_esg1
    vis = dict(contype=0, conaffinity=0, group=2)
    red, black = [0.8, 0.12, 0.1, 1], [0.05, 0.05, 0.05, 1]
    face_x = E.BIAS_X - E.BODY_X / 2
    plate = np.array([face_x - 0.003, 0.0, E.Z_BODY + 0.030])
    body.add_geom(name="cam_plate", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.003, 0.028, 0.022], pos=list(plate), rgba=red, **vis)
    # MuJoCo のカメラは自分の -z を見る。腕・箱・レンズはカメラの点より後ろ（+z）に置き、写る画像をふさがない
    back = R_from_quat(q) @ np.array([0, 0, 1.0])
    a, b = plate + np.array([-0.003, 0, 0]), cam + back * 0.02
    body.add_geom(name="cam_arm", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=list(a) + list(b), size=[0.007, 0, 0], rgba=red, **vis)
    body.add_geom(name="cam_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.012, 0.010], pos=list(cam + back * 0.0112),
                  quat=list(q), rgba=red, **vis)
    body.add_geom(name="cam_lens", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.004, 0.0005, 0],
                  pos=list(cam + back * 0.0007), quat=list(q), rgba=black, **vis)


def R_from_quat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


GRAVCOMP = os.environ.get("SIM_GRAVCOMP", "1") == "1"   # 腕とハンドの重さを打ち消す（0 にすると前と同じ）


# 三脚の固定カメラ（10/20、ざっくり。写真から：マットの左手前（ロボットから見て +x・+y の角の外）、机から 30〜40cm、マットを斜めに見る）
# 実機はまだつないでいない。つないだら校正して、この位置・向き・画角を合わせ直す
TRIPOD = os.environ.get("SIM_TRIPOD", "0") == "1"
TRIPOD_W, TRIPOD_H, TRIPOD_FOVY = 1280, 720, 43.0         # よくある 1080p の webカメラ（横の画角 約 70°）を小さめに描く


TRIPOD_POSE = os.environ.get("SIM_TRIPOD_POSE", "real")   # real＝本物の三脚の校正（real_ur/tripod_calib.json）から置く・guess＝前の見当の位置（10/20）
# 実機の校正・箱の位置は実機の日に書き替わる（10/23、箱を動かし三脚を校正し直したら、シミュレーターの三脚が約 320mm ずれて物が見えなくなった）。
# そこで、シミュレーターは 10/20 の校正と 10/19 の箱の控えを固定して使う（SIM_TRIPOD_CALIB・SIM_BOX_REAL で替えられる）
TRIPOD_REAL = os.environ.get("SIM_TRIPOD_CALIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                   LAYOUT_REAL["tripod_calib"] if LAYOUT_REAL else "sim_tripod_calib.json")
TRIPOD_BOX_REAL = os.environ.get("SIM_BOX_REAL") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_box_real.json")


def tripod_pose_real():
    """本物の三脚の校正から、シミュレーターの三脚の位置・向き・縦の画角を出す。無ければ None。
    実機の根元の座標とシミュレーターの座標は、真上から見て 180° 回っている（実機は箱が +x の端・マットが -y、
    シミュレーターは箱が -x の端・マットが +y）。そこで箱の真ん中を合わせ、箱からのカメラの位置と向きを 180° 回して置く。"""
    try:
        cal = json.load(open(TRIPOD_REAL, encoding="utf-8"))
        box = json.load(open(TRIPOD_BOX_REAL, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    R = cv2.Rodrigues(np.array(cal["rvec"], float))[0]
    cam = np.array(cal["cam_pos_mm"], float)
    table_z = min(p["tcp_mm"][2] for p in cal["points"]) - 50.0       # 低い方の点は机から 50mm
    flip = np.array([-1.0, -1.0, 1.0])
    if LAYOUT_REAL:                                                     # 根元を基準に置く（180° 回すだけ）
        pos = np.array([-cam[0] / 1000, -cam[1] / 1000, MAT_Z + (cam[2] - table_z) / 1000])
    else:
        rel = (cam[:2] - np.array(box["center"], float)) * flip[:2]
        pos = np.array([BOX_XY[0] + rel[0] / 1000, BOX_XY[1] + rel[1] / 1000, MAT_Z + (cam[2] - table_z) / 1000])
    front = MAT_CENTER[1] + MAT_MM[1] / 2000                            # マットの手前の端。三脚の前の脚（カメラから約 50mm 前）がその外に立つよう、手前へずらす（10/22）
    pos[1] += max(0.0, front + 0.065 - pos[1])
    x = (R.T @ np.array([1.0, 0, 0])) * flip                           # 写真の右
    y = -(R.T @ np.array([0, 1.0, 0])) * flip                          # 写真の上（写真の y は下向き）
    sc = TRIPOD_W / cal["image_size"][0]                               # 校正は 960×540、シミュレーターは 1280×720 で描く
    K = (cal["K"][0][0] * sc, cal["K"][0][2] * sc, cal["K"][1][2] * sc)  # 焦点距離・画像の中心（本物は中心が上へ約 90 画素ずれている）
    fovy = math.degrees(2 * math.atan((TRIPOD_H / 2) / K[0]))
    return pos, x, y, fovy, K


def tripod_pose():
    if TRIPOD_POSE == "real":
        r = tripod_pose_real()
        if r is not None:
            return r
    pos = np.array([MAT_CENTER[0] + MAT_MM[0] / 2000 + 0.08, MAT_CENTER0[1] + MAT_MM0[1] / 2000 + 0.16, MAT_Z + 0.35])
    look = np.array([MAT_CENTER[0], MAT_CENTER[1], MAT_Z])
    f = (look - pos) / np.linalg.norm(look - pos)
    x = np.cross(f, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return pos, x, y, TRIPOD_FOVY, None


def add_tripod_model(spec, pos, x, y):
    """三脚とカメラの形（机の上の小さな三脚・脚 3 本・真ん中の柱・雲台・webカメラの箱）。腕が当たれば保護停止になるよう、当たりも付ける。
    カメラの箱はレンズの後ろに置くので、三脚のカメラの写真には映らない。"""
    f = np.cross(y, x)                                           # レンズの向き（見る向き）
    G = mujoco.mjtGeom
    blk = dict(rgba=[0.08, 0.08, 0.09, 1], contype=1, conaffinity=1)
    R = np.stack([x, y, -f], 1)                                  # 箱の向き（カメラと同じ）
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    spec.worldbody.add_geom(name="tripod_cam", type=G.mjGEOM_BOX, size=[0.045, 0.015, 0.014], pos=list(pos - f * 0.02), quat=list(q), **blk)
    spec.worldbody.add_geom(name="tripod_lens", type=G.mjGEOM_CYLINDER, size=[0.008, 0.003], pos=list(pos - f * 0.004), quat=list(q),
                            rgba=[0.2, 0.25, 0.35, 1], contype=0, conaffinity=0)
    head = pos - f * 0.02 - np.array([0, 0, 0.03])
    spec.worldbody.add_geom(name="tripod_head", type=G.mjGEOM_SPHERE, size=[0.012], pos=list(head), **blk)
    low = np.array([head[0], head[1], MAT_Z + 0.12])             # 真ん中の柱の下の端（脚の付け根）
    spec.worldbody.add_geom(name="tripod_col", type=G.mjGEOM_CAPSULE, size=[0.007, 0], fromto=list(head) + list(low), **blk)
    b = -f[:2] / np.linalg.norm(f[:2])                          # 後ろ向き：1 本は後ろ、2 本は斜め前（写真に入らない）
    for i, a in enumerate((0.0, 2.094, -2.094)):
        c, s_ = math.cos(a), math.sin(a)
        foot = low[:2] + 0.11 * np.array([c * b[0] - s_ * b[1], s_ * b[0] + c * b[1]])
        spec.worldbody.add_geom(name=f"tripod_leg{i}", type=G.mjGEOM_CAPSULE, size=[0.005, 0],
                                fromto=list(low) + [foot[0], foot[1], MAT_Z + 0.005], **blk)


OLD_OBJECTS = ("screwdriver", "pen", "screw1", "screw2")   # 前の試験の物（sim_tools.OBJECTS）
DEL_OLD = os.environ.get("SIM_DEL_OLD", "1") == "1"   # 1（標準）：前の試験の物（ねじ回し・ペン・ねじ 2 本。マットの外にどけてあるだけで使わない）を模型から消す（10/23。消しても物の動きは同じと確かめた。0 で前のまま）


class RealLikeEnv(BaseEnv):
    OBJ_NAMES = [] if DEL_OLD else list(OLD_OBJECTS)
    def build_model(self):
        spec = build_spec()
        if DEL_OLD:
            for k in list(spec.keys):                            # 「home」の姿勢の長さが合わなくなるので先に消す（プログラムは使っていない。始めの姿勢は reset が HOME で入れる）
                spec.delete(k)
            for o in OLD_OBJECTS:
                spec.delete(spec.body(o))
        if TRIPOD:
            pos, x, y, fovy, K = tripod_pose()
            cam = spec.worldbody.add_camera(name="tripod", pos=list(pos), xyaxes=list(x) + list(y), fovy=fovy)
            if K is not None:                                    # 本物の焦点距離と画像の中心をそのまま入れる（画素で指定。10/20）
                pp = 1e-5
                cam.resolution = [TRIPOD_W, TRIPOD_H]
                cam.sensor_size = [TRIPOD_W * pp, TRIPOD_H * pp]
                cam.focal_pixel = [K[0], K[0]]
                cam.principal_pixel = [TRIPOD_W / 2 - K[1], TRIPOD_H / 2 - K[2]]   # MuJoCo はずれの向きが画像の画素と逆（左右・上下とも。10/20 に描いた絵と計算を比べて確かめた）
            self.tri_K = K if K is not None else ((TRIPOD_H / 2) / math.tan(math.radians(fovy) / 2), TRIPOD_W / 2, TRIPOD_H / 2)
            add_tripod_model(spec, pos, x, y)
        for name in ("bin", "wall_n", "wall_s", "wall_e", "wall_w"):   # 前の試験の黄色の入れ物（ur5e_tools.xml）は使わないので消す（10/19）
            g = spec.geom(name)
            if g is not None:
                spec.delete(g)
        add_real_scene(spec)
        m = spec.compile()
        if GRAVCOMP:
            # 腕とハンドの重さを打ち消す。関節はバネ（強さ 2000、手首 500）で目標を追うだけなので、重さの分だけ垂れて止まり、
            # 目標から横 5〜9mm・下 5〜10mm ずれていた（10/16。実機の UR5e は位置をきちんと保つ）。物や箱には入れない
            # 出来上がった模型の body_gravcomp を書き換えても効かない（打ち消す部品の数 ngravcomp が 0 のまま）ので、作る前に入れて作り直す
            base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
            for i in np.flatnonzero(m.body_rootid == m.body_rootid[base]):
                spec.body(m.body(int(i)).name).gravcomp = 1.0
            m = spec.compile()
        return m

    def __init__(self):
        super().__init__()
        self.rr = mujoco.Renderer(self.m, IMG_H, IMG_W)
        if TRIPOD:
            self.rt = mujoco.Renderer(self.m, TRIPOD_H, TRIPOD_W)
            self.rt_seg = mujoco.Renderer(self.m, TRIPOD_H // 2, TRIPOD_W // 2)   # ロボットの所は半分の大きさで描いて広げる（20ms → 約 5ms。どうせ太らせて使う）
            self.rt_seg.enable_segmentation_rendering()
            self.tripod_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "tripod")
        self.tool_q = {t: int(self.m.jnt_qposadr[self.m.body_jntadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, t)]])
                       for t in TOOLS}

    def render_tripod(self, robot_bodies=None):
        """三脚のカメラの画像（BGR）と、ロボットが写っている所（ロボットの形と姿勢は分かっているので、実機でも描ける）。"""
        self.rt.update_scene(self.d, camera="tripod")
        img = cv2.cvtColor(self.rt.render(), cv2.COLOR_RGB2BGR)
        if SIM_JPEG > 0:
            img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, SIM_JPEG])[1], cv2.IMREAD_COLOR)
        rmask = None
        if robot_bodies is not None:
            self.rt_seg.update_scene(self.d, camera="tripod")
            seg = self.rt_seg.render()[:, :, 0]
            lut = np.zeros(self.m.ngeom + 1, np.uint8)            # 部品の番号 → ロボットなら 255（-1 は何も無い所）
            lut[1:] = np.isin(self.m.geom_bodyid, list(robot_bodies)) * 255
            rmask = cv2.resize(lut[seg + 1], (TRIPOD_W, TRIPOD_H), interpolation=cv2.INTER_NEAREST)
        return img, rmask

    def tripod_on_plane(self, px, h_mm):
        """三脚のカメラの画素 → 机から h_mm の面の上の点（mm）。"""
        pos = self.d.cam_xpos[self.tripod_id]
        R = self.d.cam_xmat[self.tripod_id].reshape(3, 3)
        f, cx, cy = self.tri_K
        px = np.asarray(px, float).reshape(-1, 2)
        d = np.stack([(px[:, 0] - cx) / f, -(px[:, 1] - cy) / f, -np.ones(len(px))], 1) @ R.T
        z = MAT_Z + h_mm / 1000
        t = (z - pos[2]) / d[:, 2]
        return (pos[None, :] + t[:, None] * d) * 1000

    def tripod_px(self, P_mm):
        """机の上の点（mm）→ 三脚のカメラの画素。"""
        pos = self.d.cam_xpos[self.tripod_id]
        R = self.d.cam_xmat[self.tripod_id].reshape(3, 3)
        f, cx, cy = self.tri_K
        c = (np.asarray(P_mm, float).reshape(-1, 3) / 1000 - pos) @ R
        return np.stack([cx + f * c[:, 0] / -c[:, 2], cy - f * c[:, 1] / -c[:, 2]], 1)

    def render_real(self):
        """実物のカメラの画像（1920×1080、BGR）。"""
        self.rr.update_scene(self.d, camera="wrist_real")
        img = cv2.cvtColor(self.rr.render(), cv2.COLOR_RGB2BGR)
        if SIM_JPEG > 0:                                         # 実機のカメラを MJPG にした時の画質の荒れ（一度 JPEG にして戻す）
            img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, SIM_JPEG])[1], cv2.IMREAD_COLOR)
        return img

    def set_tool(self, t, xy, yaw):
        """道具 t を、机の上の xy（m）に、向き yaw で寝かせて置く。"""
        a = self.tool_q[t]
        self.d.qpos[a:a + 3] = [xy[0], xy[1], TOOL_H[t] / 2000 + 0.0002]
        self.d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    def set_eraser(self, xy, yaw):
        self.set_tool("eraser", xy, yaw)

    def tcp_like(self):
        """実機の getActualTCPPose と同じ形の値（つかむ点の位置 m と回転ベクトル）。実機用のプログラムに渡す用。"""
        p = self.d.site_xpos[self.site].copy()
        R = self.d.site_xmat[self.site].reshape(3, 3)
        rv, _ = cv2.Rodrigues(R)
        return list(p) + list(rv.ravel())
