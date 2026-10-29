"""手首のカメラで、今の形（DSHOW・YUY2）と MJPG のいくつかの開き方を比べる：1 秒の枚数・明るさ・カメラの設定・見つかる物（10/20）。
cam_view を止めてから。python real_ur/test_mjpg_bright.py [番号]"""
import sys, time
import cv2, numpy as np
import measure as M
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
PROPS = {"露出": cv2.CAP_PROP_EXPOSURE, "自動露出": cv2.CAP_PROP_AUTO_EXPOSURE, "利得": cv2.CAP_PROP_GAIN, "明るさ": cv2.CAP_PROP_BRIGHTNESS,
         "コントラスト": cv2.CAP_PROP_CONTRAST, "鮮やかさ": cv2.CAP_PROP_SATURATION, "白": cv2.CAP_PROP_WB_TEMPERATURE, "自動白": cv2.CAP_PROP_AUTO_WB, "ガンマ": cv2.CAP_PROP_GAMMA}

def open_cam(be, mj, setp=None):
    c = cv2.VideoCapture(N, be)
    if mj: c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    c.set(3, 1920); c.set(4, 1080)
    if mj: c.set(cv2.CAP_PROP_FPS, 30)
    for k, v in (setp or {}).items(): c.set(PROPS[k], v)
    return c

def grab(c, n=8, skip=30):
    fs = []; t = time.time(); k = 0
    while len(fs) < n and time.time() - t < 30:
        ok, f = c.read()
        if ok:
            k += 1
            if k > skip: fs.append(f)
    t2 = time.time(); m = 0
    while time.time() - t2 < 2.0: m += c.read()[0]
    return fs, m / 2.0

def desc(img):
    objs, _ = M.find_objects(img)
    out = []
    for cnt in objs:
        m = cv2.moments(cnt)
        if m["m00"] < 200: continue
        r = cv2.minAreaRect(cnt); out.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]), int(max(r[1])), int(min(r[1]))))
    return sorted(out)

def run(name, be, mj, setp=None):
    c = open_cam(be, mj, setp)
    p = {k: round(c.get(v), 2) for k, v in PROPS.items()}
    fs, fps = grab(c); c.release()
    if not fs: print(name, "写真が来ない"); return None, None
    a = np.mean([f.astype(float) for f in fs], 0)
    print(f"{name}：{fps:.1f} 枚/秒、明るさ {a.mean():.1f}（青緑赤 {np.round(a.mean((0, 1)), 1)}）、設定 {p}", flush=True)
    print("   物：", desc(fs[0]), "／", desc(fs[-1]), flush=True)
    cv2.imwrite(f"logs_mjpg/b_{name}.jpg", fs[-1])
    return p, a

if __name__ == "__main__":
  pY, aY = run("YUY2_DSHOW", cv2.CAP_DSHOW, False)
  run("MJPG_DSHOW", cv2.CAP_DSHOW, True)
  pJ, aJ = run("MJPG_MSMF", cv2.CAP_MSMF, True)
  if pY:   # 今の形の設定をそのまま MSMF の MJPG に入れる
      keep = {k: v for k, v in pY.items() if k in ("露出", "利得", "明るさ", "コントラスト", "鮮やかさ", "ガンマ", "白")}
      run("MJPG_MSMF_同じ設定", cv2.CAP_MSMF, True, keep)
