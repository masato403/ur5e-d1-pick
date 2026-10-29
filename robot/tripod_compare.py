"""シミュレーターの三脚の写真と本物の三脚の写真（real_ur/cam_shots/tripod_latest.jpg）を並べる（10/20）。
シミュレーターの写真には tripod_px で箱の中心とマットの角を描き、描いた絵と計算が合うかも見る。"""
import os
os.environ["SIM_TRIPOD"] = "1"
import cv2
import mujoco
import numpy as np
import sim_real as sr

env = sr.RealLikeEnv()
mujoco.mj_forward(env.m, env.d)
img, _ = env.render_tripod()
bx, by = sr.BOX_XY
for P in [(bx * 1000, by * 1000, 0.0)]:
    u, v = env.tripod_px([P])[0]
    cv2.drawMarker(img, (int(u), int(v)), (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
real = cv2.imread(os.path.join(os.path.dirname(__file__), "..", "real_ur", "cam_shots", "tripod_latest.jpg"))
real = cv2.resize(real, (sr.TRIPOD_W, sr.TRIPOD_H))
out = np.hstack([img, real])
cv2.putText(out, "sim", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
cv2.putText(out, "real", (sr.TRIPOD_W + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
p = os.path.join(os.path.dirname(__file__), "logs", "tripod_sim_vs_real_K.jpg")
cv2.imwrite(p, cv2.resize(out, None, fx=0.5, fy=0.5))
print("K", env.tri_K, "fovy", float(env.m.cam_fovy[env.tripod_id]), "→", p)
