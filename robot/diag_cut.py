"""下りながら追う時の三脚の写真で、腕の所を消すと物がどう見えるか（10/22）。使い方: python robot/diag_cut.py 記録のフォルダ 物の番号"""
import json, os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_track_real as TT, sim_real as S, tripod_view as TV
env = S.RealLikeEnv()
TT.patch_tripod(env)
d, k = sys.argv[1], sys.argv[2]
TV.BG = cv2.imread(os.path.join(d, "bg.jpg")) if os.path.exists(os.path.join(d, "bg.jpg")) else None
st = json.load(open(os.path.join(d, "steps.json"), encoding="utf-8"))["steps"]
rows = [r for r in st if "t" in r]
per = []
for r_ in rows:                                                  # 物ごとの並び（t が 0 に戻る所で分ける）
    if r_["t"] == 0 or not per:
        per.append([])
    per[-1].append(r_)
grp = per[int(k) - 1]
for f in sorted(x for x in os.listdir(d) if x.startswith(f"obj{k}_t") or x.startswith(f"obj{k}_0_t")):
    t = int(f.split("_t")[1][:4]) / 1000
    r = min(grp, key=lambda r: abs(r["t"] - t))
    img = cv2.imread(os.path.join(d, f))
    rm = TV.robot_mask(img.shape, r["tcp"])
    a = [(TV.size_mm(c)) for _, c, _ in TV.blobs(img)]
    b = [(TV.size_mm(c)) for _, c, _ in TV.blobs(img, rm)]
    print(f, f"z {r['z']}", "消さない:", [f"{w:.0f}x{l:.0f}" for w, l in a], " 消す:", [f"{w:.0f}x{l:.0f}" for w, l in b])
    v = img.copy(); v[rm > 0] = (v[rm > 0] * 0.4 + np.array([0, 0, 150]) * 0.6).astype(np.uint8)
    cv2.imwrite(os.path.join(d, f"mask_{f}"), v)
