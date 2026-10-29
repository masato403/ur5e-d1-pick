# 1023b の焦点距離・ゆがみ（同じカメラ）を固定し、カメラの位置と向きだけを今の写真の交点で合わせる
import json, sys, cv2, numpy as np
sys.argv = [sys.argv[0]]; import os; sys.path.insert(0, os.getcwd())
import calib_tripod_grid as G
img = cv2.imread(os.environ.get("IMG", "tripod_ref_1026.jpg"))
b = json.load(open("tripod_calib_grid_1023b.json", encoding="utf-8"))
corners = np.array([float(v) for v in os.environ.get("CORN", "187 145 627 178 680 435 116 397").split()], np.float32).reshape(4, 2)
cm = np.array([[0, G.MAT_D], [G.MAT_W, G.MAT_D], [G.MAT_W, 0], [0, 0]], np.float32)
H = cv2.getPerspectiveTransform(cm, corners)
best = None
for it in range(4):
    Gm, Gp, ij, ph = G.find_points(img, H)
    H2, inl = cv2.findHomography(Gm.astype(np.float32), Gp.astype(np.float32), cv2.RANSAC, 1.5)
    e = np.linalg.norm(cv2.perspectiveTransform(Gm.reshape(-1,1,2).astype(np.float64), H2).reshape(-1,2) - Gp, axis=1)
    g = inl.ravel().astype(bool)
    print(it, len(Gm), int(g.sum()), round(float(np.median(e[g])),2), ph)
    if best is None or g.sum() > best[0]: best = (int(g.sum()), Gm[g], Gp[g])
    H = H2
_, Gm, Gp = best
SH=np.array([float(v) for v in os.environ["SH"].split(",")]); Gm = Gm + SH; print("ずらし", SH)
K = np.array(b["K"]); d = b["dist"]
p = np.r_[K[0,0], K[0,2], K[1,2], d[0], np.zeros(6), 0, 0, 0]
# マットの置き場所は 1023b と同じ（マットは動かしていない）
mc = np.array(b["mat_corners_mm"])
p[10:12] = mc[3,:2]; p[12] = np.radians(b["mat_yaw_deg"])
P3 = G.mat_to_base(p, Gm)
ok, rv, tv = cv2.solvePnP(P3, Gp, K, np.array(d, float))
p[4:7], p[7:10] = rv.ravel(), tv.ravel()
p, r = G.lm(p, (Gm, Gp, np.array([[200.0,-400,0]]), np.zeros((1,2)), 0.0), free=[4,5,6,7,8,9])
e = np.linalg.norm(r[:2*len(Gm)].reshape(-1,2), axis=1)
print("交点", len(Gm), "ずれ 中央", round(float(np.median(e)),2), "90%", round(float(np.percentile(e,90)),2), "最大", round(float(e.max()),2))
print("マット左手前の角の確かめ", mc[3], "→ 模型", G.mat_to_base(p, [[0,0]])[0])
R = cv2.Rodrigues(p[4:7])[0]; cam = -R.T @ p[7:10]; print("カメラ", cam.round(0), "机から", round(cam[2]-G.TABLE_Z))
mc2 = G.mat_to_base(p, cm.astype(float)); edge = G.project(p, mc2)
print("四隅の画素（模型）", edge.round(1).tolist())
vis = img.copy()
for (u,v) in Gp: cv2.circle(vis,(int(u),int(v)),2,(0,255,0),-1)
for (u,v) in G.project(p, G.mat_to_base(p, Gm)): cv2.circle(vis,(int(round(u)),int(round(v))),1,(0,0,255),-1)
cv2.polylines(vis,[edge.astype(int).reshape(-1,1,2)],True,(255,0,255),1)
cv2.imwrite("fixk.jpg", cv2.resize(vis,(1920,1080)))
out = dict(json.load(open("tripod_calib_grid_1026.json", encoding="utf-8")))
out.update(K=K.tolist(), dist=list(d), rvec=p[4:7].tolist(), tvec=p[7:10].tolist(), cam_pos_mm=cam.tolist(), grid_px_median=float(np.median(e)), grid_n=len(Gm),
           mat_corners_mm=mc2.tolist(), mat_yaw_deg=b["mat_yaw_deg"], mat_px=edge.round(1).tolist(), mat_px_calc=edge.round(1).tolist(),
           when="2026-10-26", note="10/26：三脚が動いたので、1023b の焦点距離・ゆがみ・マットの置き場所のまま、カメラの位置と向きだけ交点で合わせ直した（calib_fixk.py）")
json.dump(out, open(os.environ.get("OUT","tripod_calib_fixk_1026.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1)
