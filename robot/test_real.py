"""実物の写真で、見る係を試す（FastSAM で物を見つける → d1-3B で何の物かを見分ける。どちらも追加学習なし）。

  python robot/test_real.py 写真のパス [写真のパス ...]   → robot/real/<名前>_result.png と robot/logs/test_real.jsonl

写真の大きさはまちまちなので、物の大きさの範囲は画像の大きさに対する割合で決める。
いちばん大きくて明るい面（白い紙・箱）は「置き場所」とみなす。
"""
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]

from run_tools import PART_DESC  # noqa: E402

# 実物で使う部品（ネジが手元にないので、代わりに消しゴム）。PARTS=screwdriver,screw,pen のように環境変数で変えられる
PARTS = os.environ.get("PARTS", "screwdriver,eraser,pen").split(",")
MERGE = os.environ.get("MERGE", "line")         # 輪郭のまとめ方（detect の説明を参照）
# 試す設定（気になる点を全部比べるため、環境変数で切り替える）
SAM = os.environ.get("SAM", "fastsam")           # 物を切り出すモデル：fastsam / mobilesam
IMGSZ = int(os.environ.get("IMGSZ", 1024))       # FastSAM に渡す時の画像の大きさ
CONF = float(os.environ.get("CONF", 0.25))       # FastSAM：輪郭として出す自信の下限
IOU = float(os.environ.get("IOU", 0.7))          # FastSAM：重なった輪郭を 1 つに減らす基準
CROP = int(os.environ.get("CROP", 256))          # d1-3B に見せる切り抜きの大きさ
INPUT = int(os.environ.get("INPUT", 0))          # 0 以外なら、写真の長い辺をこの画素数に縮めてから全部の処理をする（カメラの解像度を下げた想定）
PLACE = os.environ.get("PLACE", "colour")        # 置き場所の見つけ方：colour（色の薄い所）／sam（切り出した輪郭のうち、大きくて色の薄い物）
SPLIT = os.environ.get("SPLIT", "none")          # 触れている物の分け方：none／colour（色の違いで 2 つに分ける）／ask（AI に物の数を聞き、2 つ以上なら色で分ける）
SAMEMERGE = os.environ.get("SAMEMERGE", "0") == "1"   # 見分けた後、くっついていて同じ種類と答えた 2 つを 1 つにまとめるか
TAG = os.environ.get("TAG", MERGE)               # 結果の名前

# 正解（写真を見て書き起こしたもの）：物の種類と、置き場所（P＝紙の上、T＝机、E＝紙のふちにまたがる＝どちらでも正解）
TRUTH = {"photo01": [("pen", "T"), ("eraser", "T"), ("screwdriver", "T")],
         "photo02": [("screwdriver", "T"), ("pen", "T"), ("eraser", "T")],
         "photo03": [("pen", "T"), ("eraser", "T"), ("screwdriver", "T")],
         "photo04": [("pen", "T"), ("eraser", "T"), ("screwdriver", "T")],
         "photo05": [("pen", "P"), ("eraser", "T"), ("screwdriver", "T")],
         "photo06": [("pen", "P"), ("screwdriver", "T"), ("eraser", "E")],
         "photo07": [("pen", "T"), ("screwdriver", "T"), ("eraser", "E")],
         "photo08": [("screwdriver", "P"), ("pen", "P"), ("eraser", "P")],
         "photo09": [("screwdriver", "P"), ("pen", "P"), ("eraser", "P")]}


def score(name, rows):
    """種類と置き場所が合っている物の数（正解の物 1 つにつき、答え 1 つまで）、余計な答えの数。"""
    truth = list(TRUTH.get(name, []))
    ok, kind_ok = 0, 0
    rest = []
    for r in rows:
        place = "P" if r["on_place"] else "T"
        hit = next((t for t in truth if t[0] == r["what"] and t[1] in (place, "E")), None)
        if hit:
            truth.remove(hit)
            ok += 1
        else:
            rest.append(r)
    for r in rest:                                  # 種類だけ合っている物（置き場所の判断だけまちがい）
        hit = next((t for t in truth if t[0] == r["what"]), None)
        if hit:
            truth.remove(hit)
            kind_ok += 1
    return {"correct": ok, "kind_only": kind_ok, "missed": len(truth), "extra": len(rows) - ok - kind_ok, "n_truth": len(TRUTH.get(name, []))}


def axis(mk):
    """輪郭の長い軸の向き（rad）と、長さ÷太さ。"""
    ys, xs = np.nonzero(mk)
    c = np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()]))
    ev, evec = np.linalg.eigh(c)
    return float(np.arctan2(evec[1, 1], evec[0, 1])), float(np.sqrt(ev[1] / max(ev[0], 1e-6)))


def collinear(a, b, max_deg=12):
    """2 つの輪郭が、1 本の細長い物の続きになっているか。
    ① 合わせた形の向きが、長い方の向きとそろっている　② 合わせても細長いまま
    ③ 短い方の中心が、長い方の軸の延長線上にある（横にずれていない。ずれが長い方の太さの半分以下）
    ③ がないと、横に平行に並んだ別の物（ペンの横のネジなど）までまとめてしまう。"""
    ta, ea = axis(a)
    tb, eb = axis(b)
    tc, ec = axis(a | b)

    def diff(x, y):
        d = abs(x - y) % np.pi
        return np.degrees(min(d, np.pi - d))
    longer, shorter = (a, b) if a.sum() >= b.sum() else (b, a)
    tl = ta if longer is a else tb
    if not (diff(tc, tl) < max_deg and ec > 3.0 and ec >= 0.8 * max(ea, eb)):
        return False
    ys, xs = np.nonzero(longer)
    cx, cy = xs.mean(), ys.mean()
    ux, uy = np.cos(tl), np.sin(tl)
    thick = np.abs(-(xs - cx) * uy + (ys - cy) * ux).max() * 2 + 1        # 長い方の太さ（軸に直角の向きの幅）
    ys2, xs2 = np.nonzero(shorter)
    off = abs(-(xs2.mean() - cx) * uy + (ys2.mean() - cy) * ux)          # 短い方の中心の、軸からの横のずれ
    return off <= 0.5 * thick


def detect(model, rgb, merge="line"):
    """merge：輪郭が物の一部分ずつに分かれた時のまとめ方
         touch   ：接している輪郭は全部まとめる（前の版。触れている別の物までまとまってしまった）
         contain ：まとめない。大きい輪郭にほとんど含まれる輪郭（物の一部分）だけ捨てる（A）
         colour  ：contain のあと、接していて色が似ている輪郭だけまとめる（B）
         line    ：contain のあと、接していて同じ向きに一直線に並ぶ輪郭だけまとめる（C。ペンの先と軸など）"""
    H, W = rgb.shape[:2]
    objs, place = [], None
    if SAM == "mobilesam":       # 点を画像全体に格子状に並べて（40 × 40 程度）、点ごとに物を切り出す
        step = max(H, W) // 40
        pts = [[x, y] for y in range(step // 2, H, step) for x in range(step // 2, W, step)]
        out = []
        for i in range(0, len(pts), 32):
            chunk = pts[i:i + 32]
            r = model(rgb[:, :, ::-1], points=[[p] for p in chunk], labels=[[1] for _ in chunk], verbose=False)[0]
            if r.masks is not None:
                out.append(r.masks.data.cpu().numpy() > 0.5)
        if not out:
            return objs, place
        masks = np.concatenate(out)
    else:
        r = model(rgb[:, :, ::-1], retina_masks=True, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)[0]
        if r.masks is None:
            return objs, place
        masks = r.masks.data.cpu().numpy() > 0.5
    # 置き場所（白い紙・箱の床）：明るくて色の薄い所のうち、いちばん大きいかたまりの外側を囲む形（凸包）。
    # 囲む形にすると、紙の上の物の所（白くない穴）や、影で少し暗い所も紙に含まれる
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # 紙は色がほとんどない（鮮やかさ S が 35 未満）。机の木目は S が 60 以上。
    # 影になった紙は机より暗いこともあるので、明るさでは分けない（暗すぎる所だけ除く）
    white = ((hsv[..., 1] < 40) & (hsv[..., 2] > 100)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8)))
    if n > 1:
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if st[k, cv2.CC_STAT_AREA] > 0.05 * H * W:
            pts = cv2.findNonZero((lab == k).astype(np.uint8))
            place = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(place, cv2.convexHull(pts), 1)
            place = place > 0
    if PLACE == "sam":
        # 切り出した輪郭のうち、画像の 5% 以上の大きさで、平均の鮮やかさが低い（白い紙）物。外側を囲む形（凸包）にする
        place = None
        best = None
        for mk in masks:
            if mk.sum() < 0.05 * H * W:
                continue
            sat = float(hsv[..., 1][mk].mean())
            if sat < 40 and (best is None or mk.sum() > best.sum()):
                best = mk
        if best is not None:
            place = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(place, cv2.convexHull(cv2.findNonZero(best.astype(np.uint8))), 1)
            place = place > 0
    lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    keep = []
    for mk in sorted(masks, key=lambda m: -m.sum()):
        frac = mk.sum() / (H * W)
        ys, xs = np.nonzero(mk)
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if frac > 0.15 or frac < 0.0008 or max(bw, bh) > 0.9 * min(H, W):    # 机・紙のような広い面と、小さすぎる物は除く
            continue
        edge = xs.min() <= 1 or ys.min() <= 1 or xs.max() >= W - 2 or ys.max() >= H - 2
        fill = mk.sum() / float(bw * bh)
        if edge and (min(bw, bh) < 0.05 * min(H, W) or fill < 0.35 or max(bw, bh) > 4 * min(bw, bh) and max(bw, bh) > 0.4 * min(H, W)):
            continue                                                         # 写真のふちの帯（細い・すかすか・長細い）は除く
        if merge == "small":
            keep.append(mk)                          # small：あとで小さい輪郭から選び直す
            continue
        if merge != "touch" and any((mk & q).sum() > 0.5 * mk.sum() for q in keep):   # 大きい輪郭にほとんど含まれる＝物の一部分
            continue
        keep.append(mk)
    if merge == "small":
        # 小さい輪郭から先に残す。大きい輪郭が、すでに残した 2 つ以上の輪郭でほとんど（8 割以上）覆われていたら、
        # それは別々の物をまとめた輪郭なので捨てる。1 つだけに覆われる場合は、その物の一部分なので大きい方と入れ替える
        chosen = []
        for mk in sorted(keep, key=lambda m: m.sum()):
            inside = [q for q in chosen if (mk & q).sum() > 0.6 * q.sum()]
            if len(inside) >= 2 and sum((mk & q).sum() for q in inside) > 0.8 * mk.sum():
                continue
            if len(inside) == 1:
                chosen = [q for q in chosen if q is not inside[0]]
            if any((mk & q).sum() > 0.5 * mk.sum() for q in chosen):
                continue
            chosen.append(mk)
        keep = chosen
        merge = "line"
    if not keep:
        return objs, place
    if merge == "touch":
        fg = np.any(keep, axis=0).astype(np.uint8)
        k = max(3, int(0.008 * min(H, W)))
        n, lab = cv2.connectedComponents(cv2.dilate(fg, np.ones((k, k), np.uint8)))
        groups = [(lab == i) & (fg > 0) for i in range(1, n)]
    else:
        groups = list(keep)
        if merge == "line":
            k = max(3, int(0.008 * min(H, W)))
            changed = True
            while changed:                                  # 接していて、同じ向きに一直線に並ぶ輪郭どうしをまとめる（ペンの先と軸など）
                changed = False
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        a, b = groups[i], groups[j]
                        if not (cv2.dilate(a.astype(np.uint8), np.ones((k, k), np.uint8)) > 0)[b].any():
                            continue
                        if collinear(a, b):
                            groups[i] = a | b
                            del groups[j]
                            changed = True
                            break
                    if changed:
                        break
        if merge == "colour":
            k = max(3, int(0.008 * min(H, W)))
            changed = True
            while changed:                                  # 接していて、平均の色が近い輪郭どうしをまとめる
                changed = False
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        a, b = groups[i], groups[j]
                        if not (cv2.dilate(a.astype(np.uint8), np.ones((k, k), np.uint8)) > 0)[b].any():
                            continue
                        if np.linalg.norm(lab_img[a].mean(0) - lab_img[b].mean(0)) < 25:
                            groups[i] = a | b
                            del groups[j]
                            changed = True
                            break
                    if changed:
                        break
    for mk in groups:
        ys, xs = np.nonzero(mk)
        if len(xs) < 0.0008 * H * W:
            continue
        objs.append({"mask": mk, "uv": (float(xs.mean()), float(ys.mean())), "box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))})
    return objs, place


def colour_split(rgb, o, min_share=0.2, min_de=40):
    """1 つの輪郭の中の色を 2 つのグループに分け、色の差が大きく、どちらもある程度の大きさなら、2 つの物に分ける。"""
    mk = o["mask"]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    px = lab[mk]
    if len(px) < 200:
        return [o]
    _, lb, cen = cv2.kmeans(px, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3, cv2.KMEANS_PP_CENTERS)
    if np.linalg.norm(cen[0] - cen[1]) < min_de:
        return [o]
    parts = []
    full = np.zeros(mk.shape, np.uint8)
    for c in (0, 1):
        full[:] = 0
        full[mk] = (lb.ravel() == c).astype(np.uint8)
        full2 = cv2.morphologyEx(full, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, cc, st, _ = cv2.connectedComponentsWithStats(full2)
        if n < 2:
            return [o]
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        part = cc == k
        if part.sum() < min_share * mk.sum():
            return [o]
        ys, xs = np.nonzero(part)
        parts.append({"mask": part, "uv": (float(xs.mean()), float(ys.mean())), "box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))})
    return parts


def zoom_split(model, rgb, o, pad=0.25, size=1024, min_share=0.12):
    """物のまわりだけを切り抜いて拡大し、もう一度 FastSAM で切り出す（手首のカメラで近づいて見るのと同じ考え方）。
    拡大した画像の中で、元の物の範囲に収まる別々の輪郭が 2 つ以上あれば、分けて返す。"""
    x1, y1, x2, y2 = o["box"]
    H, W = rgb.shape[:2]
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    a, b = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
    c, d = min(W, int(x2 + pad * bw) + 1), min(H, int(y2 + pad * bh) + 1)
    sub = rgb[b:d, a:c]
    f = size / max(sub.shape[:2])
    big = cv2.resize(sub, (round(sub.shape[1] * f), round(sub.shape[0] * f)), interpolation=cv2.INTER_CUBIC)
    r = model(big[:, :, ::-1], retina_masks=True, imgsz=size, conf=CONF, iou=IOU, verbose=False)[0]
    if r.masks is None:
        return [o]
    own = o["mask"][b:d, a:c]
    own_area = own.sum()
    parts = []
    for mk in sorted(r.masks.data.cpu().numpy() > 0.5, key=lambda m: -m.sum()):
        mk = cv2.resize(mk.astype(np.uint8), (sub.shape[1], sub.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        area = mk.sum()
        if area < min_share * own_area or area > 1.3 * own_area:        # 小さすぎる部分と、まわりの机や紙まで含む大きい輪郭は除く
            continue
        if (mk & own).sum() < 0.7 * area:                                # 元の物の範囲から大きくはみ出す輪郭は除く
            continue
        if any((mk & q).sum() > 0.5 * min(area, q.sum()) for q in parts):  # すでに選んだ輪郭とほとんど重なる物は除く
            continue
        parts.append(mk)
    if len(parts) < 2:
        return [o]
    out = []
    for mk in parts:
        full = np.zeros((H, W), bool)
        full[b:d, a:c] = mk & own
        ys, xs = np.nonzero(full)
        if len(xs) == 0:
            continue
        out.append({"mask": full, "uv": (float(xs.mean()), float(ys.mean())), "box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))})
    return out if len(out) >= 2 else [o]


def crop(rgb, o, pad=0.15):
    x1, y1, x2, y2 = o["box"]
    s = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = s / 2 * (1 + pad * 2)
    H, W = rgb.shape[:2]
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(W, cx + half)), int(min(H, cy + half))
    return cv2.resize(rgb[b:d, a:c], (CROP, CROP))


def main():
    from d1_fast import D1Fast
    from ultralytics import SAM as MSAM, FastSAM
    sam = MSAM("mobile_sam.pt") if SAM == "mobilesam" else FastSAM("FastSAM-s.pt")
    m = D1Fast(quant="fp8").model
    crit = {p: PART_DESC[p] for p in PARTS + ["other"]}
    os.makedirs("robot/real", exist_ok=True)
    for path in sys.argv[1:]:
        rgb = np.array(Image.open(path).convert("RGB"))
        if INPUT:
            f = INPUT / max(rgb.shape[:2])
            rgb = cv2.resize(rgb, (round(rgb.shape[1] * f), round(rgb.shape[0] * f)), interpolation=cv2.INTER_AREA)
        t = time.perf_counter()
        objs, place = detect(sam, rgb, MERGE)
        if SPLIT == "colour":
            objs = [p for o in objs for p in colour_split(rgb, o)]
        elif SPLIT == "zoom":                        # 物ごとに拡大して、もう一度切り出す
            objs = [p for o in objs for p in zoom_split(sam, rgb, o)]
        elif SPLIT == "askzoom":                     # AI が「2 つ以上」と答えた物だけ、拡大してもう一度切り出す
            new = []
            for o in objs:
                a = m.system_one("Close-up camera image of a work table.",
                                 {"q": {"type": "choice", "instructions": "How many separate objects are in this image?",
                                        "criteria": {"one": "one object", "two": "two objects", "three": "three or more objects"}}},
                                 images=[Image.fromarray(crop(rgb, o))])["answers"]["q"]["choice"]
                new += zoom_split(sam, rgb, o) if a != "one" else [o]
            objs = new
        elif SPLIT == "ask":                         # AI に切り抜きを見せて、物がいくつ写っているかを聞く
            new = []
            for o in objs:
                a = m.system_one("Close-up camera image of a work table.",
                                 {"q": {"type": "choice", "instructions": "How many separate objects are in this image?",
                                        "criteria": {"one": "one object", "two": "two objects", "three": "three or more objects"}}},
                                 images=[Image.fromarray(crop(rgb, o))])["answers"]["q"]["choice"]
                new += colour_split(rgb, o) if a != "one" else [o]
            objs = new
        det_ms = (time.perf_counter() - t) * 1000
        show = rgb[:, :, ::-1].copy()
        if place is not None:
            cnts, _ = cv2.findContours(place.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(show, cnts, -1, (0, 200, 255), 4)
        labs = []
        for o in objs:                               # 物ごとに、何の物かを AI に聞く
            t = time.perf_counter()
            q = m.system_one("Close-up camera image of one object on a work table.",
                             {"q": {"type": "choice", "instructions": "What is this object?", "criteria": crit}},
                             images=[Image.fromarray(crop(rgb, o))])["answers"]["q"]
            labs.append((q, round((time.perf_counter() - t) * 1000)))
        if SAMEMERGE:
            # 見分けた後のまとめ直し：
            #  ① 小さな切れ端（隣の物の 15% 未満）は、くっついている大きい物の一部とみなして足す（ペンの先など）
            #  ② 同じ種類と答えた 2 つ（「その他」どうしも含む）は、くっついているか、一直線に並んで近ければ 1 つにまとめる
            #     （ペンの上に消しゴムが乗って、ペンが 2 つに切れて見えた時など）。まとめた物は、もう一度 AI に聞き直す
            k = max(3, int(0.01 * min(rgb.shape[:2])))
            gap = 0.08 * min(rgb.shape[:2])

            def touching(a, b, d=k):
                return (cv2.dilate(a.astype(np.uint8), np.ones((d, d), np.uint8)) > 0)[b].any()

            def ask(o):
                t = time.perf_counter()
                q = m.system_one("Close-up camera image of one object on a work table.",
                                 {"q": {"type": "choice", "instructions": "What is this object?", "criteria": crit}},
                                 images=[Image.fromarray(crop(rgb, o))])["answers"]["q"]
                return q, round((time.perf_counter() - t) * 1000)

            def join(i, j):
                mk = objs[i]["mask"] | objs[j]["mask"]
                ys, xs = np.nonzero(mk)
                objs[i] = {"mask": mk, "uv": (float(xs.mean()), float(ys.mean())), "box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))}
                del objs[j], labs[j]

            changed = True
            while changed:
                changed = False
                for i in range(len(objs)):
                    for j in range(len(objs)):
                        if i == j:
                            continue
                        ai, aj = objs[i]["mask"].sum(), objs[j]["mask"].sum()
                        if aj < 0.15 * ai and touching(objs[i]["mask"], objs[j]["mask"]):        # ① 小さな切れ端
                            join(i, j) if i < j else join(i, j)
                            changed = True
                            break
                        same = labs[i][0]["choice"] == labs[j][0]["choice"]
                        if i < j and same and (touching(objs[i]["mask"], objs[j]["mask"]) or
                                               (touching(objs[i]["mask"], objs[j]["mask"], int(2 * gap)) and collinear(objs[i]["mask"], objs[j]["mask"]))):
                            join(i, j)                                                               # ② 同じ種類
                            labs[i] = ask(objs[i])
                            changed = True
                            break
                    if changed:
                        break
        # 最後にもう一度、写真のふちの細い帯を除く（拡大して見直した後の切れ端にも出ることがあるので）
        H, W = rgb.shape[:2]
        keep_i = []
        for i, o in enumerate(objs):
            x1, y1, x2, y2 = o["box"]
            bw, bh = x2 - x1 + 1, y2 - y1 + 1
            edge = x1 <= 1 or y1 <= 1 or x2 >= W - 2 or y2 >= H - 2
            fill = o["mask"].sum() / float(bw * bh)
            if edge and (min(bw, bh) < 0.05 * min(H, W) or fill < 0.35):
                continue
            keep_i.append(i)
        objs, labs = [objs[i] for i in keep_i], [labs[i] for i in keep_i]
        rows = []
        for i, (o, (q, ms)) in enumerate(zip(objs, labs), 1):
            on_place = bool(place is not None and place[int(o["uv"][1]), int(o["uv"][0])])
            rows.append({"id": i, "what": q["choice"], "prob": {k: round(v, 2) for k, v in q["probabilities"].items()},
                         "uv": [round(o["uv"][0]), round(o["uv"][1])], "on_place": on_place, "ms": ms})
            cnts, _ = cv2.findContours(o["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(show, cnts, -1, (0, 0, 255), 3)
            x, y = int(o["uv"][0]), int(o["uv"][1])
            cv2.circle(show, (x, y), 26, (255, 255, 255), -1)
            cv2.circle(show, (x, y), 26, (0, 0, 0), 2)
            cv2.putText(show, str(i), (x - 10, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.putText(show, f"{q['choice']} {q['probabilities'][q['choice']]:.2f}", (x + 32, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        name = os.path.splitext(os.path.basename(path))[0]
        out = f"robot/real/{name}_result_{TAG}.png"
        cv2.imwrite(out, show)
        res = {"image": path, "tag": TAG, "merge": MERGE, "sam": SAM, "imgsz": IMGSZ, "conf": CONF, "iou": IOU, "crop": CROP, "input": INPUT, "place": PLACE, "split": SPLIT, "samemerge": SAMEMERGE,
               "score": score(name, rows), "size": list(rgb.shape[:2]), "detect_ms": round(det_ms), "place_found": place is not None, "objects": rows}
        print(json.dumps(res, ensure_ascii=False), flush=True)
        with open("robot/logs/test_real.jsonl", "a", encoding="utf-8") as fp:
            fp.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
