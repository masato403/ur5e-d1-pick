"""物の見つけ方（どこにあるか）を、実環境で使える方法どうしで比べる（学習なし・追加学習なし）。

  ref      ：今の方法。最初に撮った何もない机の画像との違い（シミュレーターならでは：照明が変わると弱い）
  colour   ：A。その場の画像から「背景の色」（机・箱の床など、広い面積を占める色）を推定し、それと違う所を物とする
  colour+e ：A に、輪郭（エッジ）も足したもの（色の差が小さい金属のネジ用）
  fastsam  ：B。FastSAM-s（物の輪郭を切り出すだけの公開モデル。種類は判断しない）
  mobilesam：B。MobileSAM（同上、SAM の小型版）

場面は 2 種類：
  clean：今のシミュレーターの画像そのまま
  real ：実環境に近づけた画像（机に木目、照明の明るさと色味の変化、カメラのノイズ、少しのぼけ）
        ref は、最初に撮った机の画像と照明がずれた状態で比べることになる（実環境で起きること）
  hard ：real より厳しい（木目が濃い、照明の変化が大きい、人や腕の影のような大きな影がかかる）

  python robot/test_detect.py [場面の数=30] [方法,方法,...]   → robot/logs/test_detect.jsonl
"""
import json
import os
import sys
import time

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
os.chdir(os.path.join(HERE, ".."))

from run_tools import PARK_Q, SegPerception  # noqa: E402
from sim_tools import OBJECTS, ToolEnv  # noqa: E402

MIN_AREA, MAX_AREA, MAX_SIDE = 30, 6000, 140      # 物の大きさの範囲（今の見る係と同じ）


# ------------------------------------------------------------ 実環境に近づける加工
def grain(rng, shape=(480, 640)):
    """木目の模様（机に固定。毎回同じ机なので、場面ごとには変えない）。"""
    h, w = shape
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    g = (np.sin(y * 0.09 + 3 * np.sin(x * 0.004 + y * 0.01)) * 0.5 + 0.5) * 0.06
    g += cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), 6) * 0.04
    return g.astype(np.float32)


def realistic(rgb, table_mask, g, rng, light):
    """light = (明るさ, 色味 RGB)。table_mask の所（机）には木目を乗せる。"""
    f = rgb.astype(np.float32)
    f[table_mask] *= (1 - g[table_mask])[:, None]
    f = f * light[0] * np.array(light[1], np.float32)
    f += rng.normal(0, 4, f.shape)
    f = cv2.GaussianBlur(f, (3, 3), 0.7)
    return np.clip(f, 0, 255).astype(np.uint8)


def realistic_fast(rgb, table_mask, g, light, _cache={}):
    """realistic と同じ加工を速くしたもの（動かしながら毎フレーム使う用）。乱数は OpenCV で作る（結果の並びは realistic と違う）。"""
    key = (id(table_mask), id(g))
    if key not in _cache:
        _cache.clear()
        _cache[key] = np.where(table_mask, 1 - g, 1).astype(np.float32)[:, :, None]
    f = rgb.astype(np.float32) * (_cache[key] * (light[0] * np.array(light[1], np.float32)))
    noise = np.empty_like(f)
    cv2.randn(noise, 0, 4)
    f = cv2.GaussianBlur(f + noise, (3, 3), 0.7)
    return np.clip(f, 0, 255).astype(np.uint8)


# ------------------------------------------------------------ 前景（物らしい所）の求め方
def fg_colour(rgb, edges=False, k=5, share=0.03, dist=16):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    small = cv2.resize(lab, (160, 120), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    _, lab_k, cen = cv2.kmeans(small, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lab_k.ravel(), minlength=k) / len(lab_k)
    bg = cen[cnt > share]                                       # 広い面積を占める色＝背景（机・箱の床・壁など）
    d = np.min(np.linalg.norm(lab[:, :, None, :] - bg[None, None], axis=3), axis=2)
    m = (d > dist).astype(np.uint8)
    if edges:
        e = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0), 60, 140)
        m |= (e > 0).astype(np.uint8)
    return m


class SamFG:
    def __init__(self, kind):
        from ultralytics import SAM, FastSAM
        self.kind = kind
        # fastsam / fastsam_h（半分の精度）/ fastsam480 / fastsam320 / fastsam320h のように、渡す画像の大きさと精度を名前で指定
        self.fast = kind.startswith("fastsam")
        rest = kind[len("fastsam"):] if self.fast else ""
        self.half = rest.endswith("h")
        self.imgsz = int(rest.rstrip("h").strip("_") or 640) if self.fast else 1024
        self.model = FastSAM("FastSAM-s.pt") if self.fast else SAM("mobile_sam.pt")

    def __call__(self, rgb):
        bgr = rgb[:, :, ::-1]
        import torch
        if self.fast:
            r = self.model(bgr, retina_masks=True, imgsz=self.imgsz, half=self.half, conf=0.25, iou=0.7, verbose=False)[0]
            masks = None if r.masks is None else r.masks.data.cpu().numpy()
        else:
            # 点を画像全体に格子状に並べ（20px おき）、点ごとに「この点を含む物」を切り出す。メモリ不足を防ぐため 32 点ずつ
            pts = [[x, y] for y in range(10, 480, 20) for x in range(10, 640, 20)]
            out = []
            for i in range(0, len(pts), 32):
                chunk = pts[i:i + 32]
                r = self.model(bgr, points=[[p] for p in chunk], labels=[[1] for _ in chunk], verbose=False)[0]
                if r.masks is not None:
                    out.append(r.masks.data.cpu().numpy())
            masks = np.concatenate(out) if out else None
        torch.cuda.empty_cache()
        m = np.zeros(rgb.shape[:2], np.uint8)
        if masks is None:
            return m
        for mk in masks:
            mk = mk > 0.5
            if mk.shape != m.shape:
                mk = cv2.resize(mk.astype(np.uint8), (m.shape[1], m.shape[0])) > 0
            area = int(mk.sum())
            if area < MIN_AREA or area > MAX_AREA:               # 机・箱の床のような広い面は捨てる
                continue
            ys, xs = np.nonzero(mk)
            if max(xs.max() - xs.min(), ys.max() - ys.min()) + 1 > MAX_SIDE:   # 箱の壁のような細長く大きい物も捨てる
                continue
            m |= mk.astype(np.uint8)
        return m


def group(m):
    """前景を、今の見る係と同じやり方で 1 つずつの物にまとめる。"""
    grp = cv2.dilate(m, np.ones((9, 9), np.uint8))
    n, lab = cv2.connectedComponents(grp)
    out = []
    for k in range(1, n):
        ys, xs = np.nonzero((lab == k) & (m > 0))
        if len(xs) < MIN_AREA:
            continue
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if len(xs) <= MAX_AREA and max(w, h) <= MAX_SIDE:
            out.append(ToolEnv.pixel_to_table(float(xs.mean()), float(ys.mean())))
    return out


def score(dets, truth):
    """物ごとに、4cm 以内に見つかった数を数える（1 つ＝正しい、0＝見落とし、2 以上＝分かれた）。物から遠い検出は誤検出。"""
    hit = {o: 0 for o in truth}
    err, fp = [], 0
    for xy in dets:
        o = min(truth, key=lambda o: np.linalg.norm(truth[o] - xy))
        dd = float(np.linalg.norm(truth[o] - xy))
        if dd < 0.04:
            hit[o] += 1
            err.append(dd)
        else:
            fp += 1
    return hit, err, fp


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    methods = sys.argv[2].split(",") if len(sys.argv) > 2 else ["ref", "colour", "colour+e", "fastsam", "mobilesam"]
    env = ToolEnv()
    rng = np.random.default_rng(5)
    g = grain(np.random.default_rng(1))
    sams = {k: SamFG(k) for k in methods if k.startswith("fastsam") or k == "mobilesam"}
    stat = {(c, k): {"ok": 0, "miss": 0, "split": 0, "fp": 0, "err": [], "ms": []} for c in ("clean", "real", "hard") for k in methods}
    os.makedirs("robot/detect", exist_ok=True)
    for i in range(n):
        env.reset(rng, n_in_bin=int(rng.integers(0, 2)))
        ref_clean = env.empty_table()
        env.d.qpos[:6] = PARK_Q
        mujoco.mj_forward(env.m, env.d)
        rgb = env.render()
        truth = {o: env.obj_pos(o)[:2] for o in OBJECTS}
        table = np.all(np.abs(ref_clean.astype(int) - ref_clean[240, 600].astype(int)) < 6, axis=2)   # 机の所（木目を乗せる所）
        light0 = (1.0, (1.0, 1.0, 1.0))
        light1 = (rng.uniform(0.75, 1.2), tuple(rng.uniform(0.92, 1.08, 3)))      # 机を撮った時から照明が変わった
        real = realistic(rgb, table, g, rng, light1)
        # hard：さらに、人や腕の影のような大きくぼやけた影がかかり、照明の変化も大きい
        light2 = (rng.uniform(0.6, 1.3), tuple(rng.uniform(0.88, 1.12, 3)))
        hard = realistic(rgb, table, g * 1.8, rng, light2).astype(np.float32)
        sh = np.zeros(rgb.shape[:2], np.float32)
        cv2.ellipse(sh, (int(rng.uniform(100, 540)), int(rng.uniform(80, 400))), (int(rng.uniform(80, 200)), int(rng.uniform(40, 120))),
                    float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), 25)
        hard = np.clip(hard * (1 - 0.35 * sh)[:, :, None], 0, 255).astype(np.uint8)
        ref_r = realistic(ref_clean, table, g, np.random.default_rng(i), light0)
        imgs = {"clean": (rgb, ref_clean), "real": (real, ref_r),
                "hard": (hard, realistic(ref_clean, table, g * 1.8, np.random.default_rng(i), light0))}
        for c, (img, ref) in imgs.items():
            for k in methods:
                t = time.perf_counter()
                if k == "ref":
                    per = SegPerception(ref)
                    dets = [ToolEnv.pixel_to_table(u, v) for u, v, _, _ in per.update(img, 0)]
                else:
                    m = fg_colour(img, edges=(k == "colour+e")) if k.startswith("colour") else sams[k](img)
                    dets = group(m)
                ms = (time.perf_counter() - t) * 1000
                hit, err, fp = score(dets, truth)
                s = stat[(c, k)]
                s["ok"] += sum(v == 1 for v in hit.values())
                s["miss"] += sum(v == 0 for v in hit.values())
                s["split"] += sum(v > 1 for v in hit.values())
                s["fp"] += fp
                s["err"] += err
                s["ms"].append(ms)
            if i < 3:
                cv2.imwrite(f"robot/detect/scene{i}_{c}.png", img[:, :, ::-1])
        print(i + 1, {f"{c}/{k}": s["ok"] for (c, k), s in stat.items()}, flush=True)
    total = n * len(OBJECTS)
    out = {"scenes": n, "objects": total, "results": {}}
    for (c, k), s in stat.items():
        out["results"][f"{c}/{k}"] = {"correct": f"{s['ok']}/{total}", "miss": s["miss"], "split": s["split"], "false_pos": s["fp"],
                                      "pos_err_mm": round(float(np.median(s["err"])) * 1000, 1) if s["err"] else None,
                                      "ms_median": round(float(np.median(s["ms"][1:] or s["ms"])), 1)}
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    with open("robot/logs/test_detect.jsonl", "a", encoding="utf-8") as fp:
        fp.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
