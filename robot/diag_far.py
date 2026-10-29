"""三脚の写真で、物の候補の塊を条件で捨てる前に全部出す（大きいマットの奥の物が見つからない件。10/22）。使い方: python robot/diag_far.py 記録のフォルダ 写真"""
import os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_track_real as TT, sim_real as S, tripod_view as TV
env = S.RealLikeEnv()
TT.patch_tripod(env)
d, f = sys.argv[1], sys.argv[2]
TV.BG = cv2.imread(os.path.join(d, "bg.jpg"))
img = cv2.imread(os.path.join(d, f))
fg = cv2.absdiff(img, TV.BG).max(axis=2) > 40
for er in (15, 1):
    inside = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(inside, (TV.MAT_PX * np.array([img.shape[1] / 960, img.shape[0] / 540])).astype(np.int32), 255)
    inside = cv2.erode(inside, np.ones((er, er), np.uint8))
    obj = ((inside > 0) & fg).astype(np.uint8) * 255
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(obj)
    print("縁を削る", er, [(tuple(cen[i].round()), int(st[i, 4]), int(st[i, 2]), int(st[i, 3])) for i in range(1, n)])
print("MAT_PX", TV.MAT_PX.round())
