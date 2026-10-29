"""実物の写真の上に「ロボットならこうする」を描く（ロボットは動かさない）。

  写真 → FastSAM で物を見つける → d1-3B で何の物か → 指示から運ぶ物を決める（d1-3B）
       → 物ごとに、つかむ位置・指の向き（物の長い向きに直角）・指を開く幅（mm）を計算
       → 写真の上に、運ぶ順番・指の形・置き場所（紙の上）への矢印を描く
  長さの換算：紙を A4（210 × 297 mm）とみなし、写った紙の大きさから 1 画素が何 mm かを求める。
  ROBOTIQ 2F-85 は指が最大 85 mm まで開く。それより太い物は「つかめない」とする。

  python robot/plan_photo.py "指示" 写真 [写真 ...]   → robot/real/<名前>_plan.png、robot/logs/plan_photo.jsonl
"""
import json
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]

import test_real as T  # noqa: E402

A4 = (210.0, 297.0)          # 紙の大きさ（mm）
MAX_OPEN = 85.0              # 2F-85 の指の最大の開き（mm）
JA = {"screwdriver": "ドライバー", "eraser": "消しゴム", "pen": "ペン", "other": "その他"}


def mm_per_px(place):
    """紙の外側を囲む長方形の、短い辺と長い辺の画素数から、1 画素あたりの mm を求める（2 つの平均）。"""
    pts = cv2.findNonZero(place.astype(np.uint8))
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    s, l = sorted((w, h))
    return (A4[0] / s + A4[1] / l) / 2


def grasp(o):
    """つかみ方：中心、長い向き（rad）、指の閉じる向きの太さ（画素）。"""
    ys, xs = np.nonzero(o["mask"])
    cx, cy = xs.mean(), ys.mean()
    c = np.cov(np.vstack([xs - cx, ys - cy]))
    ev, evec = np.linalg.eigh(c)
    ax = evec[:, 1]                                       # 長い向き
    perp = np.array([-ax[1], ax[0]])                      # 指が閉じる向き（長い向きに直角）
    # 中心を通る、長い向きの ±10% の帯の中で、直角の向きの広がり＝挟む所の太さ
    d_ax = (xs - cx) * ax[0] + (ys - cy) * ax[1]
    d_pp = (xs - cx) * perp[0] + (ys - cy) * perp[1]
    length = d_ax.max() - d_ax.min()
    band = np.abs(d_ax) < max(3, 0.1 * length)
    width = (d_pp[band].max() - d_pp[band].min()) if band.any() else (d_pp.max() - d_pp.min())
    return (float(cx), float(cy)), ax, perp, float(width), float(length)


def draw_gripper(img, c, perp, ax, open_px, colour, scale):
    """真上から見た 2 本の指（四角）と、つかむ点を描く。"""
    cx, cy = c
    half = open_px / 2 + 6 * scale
    for sgn in (-1, 1):
        fx, fy = cx + perp[0] * half * sgn, cy + perp[1] * half * sgn
        L, Wd = 22 * scale, 8 * scale                    # 指の長さ（物の長い向き）と厚み
        pts = np.array([[fx + ax[0] * L / 2 + perp[0] * Wd / 2 * sgn, fy + ax[1] * L / 2 + perp[1] * Wd / 2 * sgn],
                        [fx - ax[0] * L / 2 + perp[0] * Wd / 2 * sgn, fy - ax[1] * L / 2 + perp[1] * Wd / 2 * sgn],
                        [fx - ax[0] * L / 2 - perp[0] * Wd / 2 * sgn, fy - ax[1] * L / 2 - perp[1] * Wd / 2 * sgn],
                        [fx + ax[0] * L / 2 - perp[0] * Wd / 2 * sgn, fy + ax[1] * L / 2 - perp[1] * Wd / 2 * sgn]], np.int32)
        cv2.fillPoly(img, [pts], colour)
        cv2.polylines(img, [pts], True, (0, 0, 0), max(1, int(scale)))
    p1 = (int(cx + perp[0] * half), int(cy + perp[1] * half))
    p2 = (int(cx - perp[0] * half), int(cy - perp[1] * half))
    cv2.line(img, p1, p2, colour, max(1, int(2 * scale)))
    cv2.circle(img, (int(cx), int(cy)), int(5 * scale), (0, 0, 0), -1)


def main():
    instr = sys.argv[1]
    paths = sys.argv[2:]
    from d1_fast import D1Fast
    from ultralytics import FastSAM
    sam = FastSAM("FastSAM-s.pt")
    m = D1Fast(quant="fp8").model
    crit = {p: T.PART_DESC[p] for p in T.PARTS + ["other"]}
    wanted = {}
    for p in T.PARTS:                                    # 指示から、部品ごとに「入れる／残す」（文章だけ、最初に 1 回）
        o = m.system_one(f"Instruction to the robot: {instr}",
                         {"q": {"type": "choice", "instructions": f"Following this instruction, what should the robot do with the {p}?",
                                "criteria": {"put": "put it into the box", "leave": "leave it on the table"}}})
        wanted[p] = round(o["answers"]["q"]["probabilities"]["put"], 3)
    print("指示の読み取り", wanted, flush=True)
    for path in paths:
        rgb = np.array(Image.open(path).convert("RGB"))
        H, W = rgb.shape[:2]
        scale = max(H, W) / 1024
        objs, place = T.detect(sam, rgb, "line")
        new = []                                         # 触れている物は、AI に数を聞いて 2 つ以上なら色で分ける（写真の試験でいちばん良かった方法）
        for o in objs:
            a = m.system_one("Close-up camera image of a work table.",
                             {"q": {"type": "choice", "instructions": "How many separate objects are in this image?",
                                    "criteria": {"one": "one object", "two": "two objects", "three": "three or more objects"}}},
                             images=[Image.fromarray(T.crop(rgb, o))])["answers"]["q"]["choice"]
            new += T.colour_split(rgb, o) if a != "one" else [o]
        objs = new
        mmpp = mm_per_px(place) if place is not None else None
        show = rgb[:, :, ::-1].copy()
        if place is not None:
            cnts, _ = cv2.findContours(place.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(show, cnts, -1, (0, 200, 255), int(4 * scale))
        rows = []
        for o in objs:
            q = m.system_one("Close-up camera image of one object on a work table.",
                             {"q": {"type": "choice", "instructions": "What is this object?", "criteria": crit}},
                             images=[Image.fromarray(T.crop(rgb, o))])["answers"]["q"]
            c, ax, perp, wpx, lpx = grasp(o)
            on_place = bool(place is not None and place[int(c[1]), int(c[0])])
            rows.append({"o": o, "what": q["choice"], "p": q["probabilities"][q["choice"]], "c": c, "ax": ax, "perp": perp,
                         "w_mm": wpx * mmpp if mmpp else None, "l_mm": lpx * mmpp if mmpp else None, "w_px": wpx, "on_place": on_place})
        # 運ぶ物：指示で「入れる」になった部品で、まだ置き場所の上にない物。運ぶ順番は、置き場所から近い順
        todo = [r for r in rows if wanted.get(r["what"], 0) >= 0.5 and not r["on_place"]]
        if place is not None:
            ys, xs = np.nonzero(place)
            pc = np.array([xs.mean(), ys.mean()])
            todo.sort(key=lambda r: np.linalg.norm(np.array(r["c"]) - pc))
            # 置き場所：紙の中心のまわりに、物どうしが重ならないように縦に並べる
            pcs = cv2.minAreaRect(cv2.findNonZero(place.astype(np.uint8)))
            span = max(pcs[1]) * 0.6
            slots = [pc + np.array([0, (i - (len(todo) - 1) / 2) * span / max(1, len(todo))]) for i in range(len(todo))]
        else:
            slots = []
        for r in rows:                                    # 運ばない物：灰色の輪郭
            if any(r is t for t in todo):
                continue
            cnts, _ = cv2.findContours(r["o"]["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(show, cnts, -1, (150, 150, 150), int(3 * scale))
            why = "already on paper" if r["on_place"] and wanted.get(r["what"], 0) >= 0.5 else "leave"
            cv2.putText(show, f"{r['what']} ({why})", (int(r["c"][0]) + int(20 * scale), int(r["c"][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (90, 90, 90), max(1, int(2 * scale)))
        plan = []
        for i, r in enumerate(todo):
            ok = r["w_mm"] is None or r["w_mm"] + 10 <= MAX_OPEN
            col = (0, 170, 0) if ok else (0, 0, 220)
            cnts, _ = cv2.findContours(r["o"]["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(show, cnts, -1, col, int(3 * scale))
            draw_gripper(show, r["c"], r["perp"], r["ax"], r["w_px"], (255, 120, 0), scale)
            if i < len(slots):
                s = slots[i]
                cv2.arrowedLine(show, (int(r["c"][0]), int(r["c"][1])), (int(s[0]), int(s[1])), (255, 0, 200), max(2, int(3 * scale)),
                                tipLength=0.04)
                cv2.circle(show, (int(s[0]), int(s[1])), int(10 * scale), (255, 0, 200), -1)
            x, y = int(r["c"][0]), int(r["c"][1])
            cv2.circle(show, (x - int(40 * scale), y - int(40 * scale)), int(26 * scale), (255, 255, 255), -1)
            cv2.circle(show, (x - int(40 * scale), y - int(40 * scale)), int(26 * scale), (0, 0, 0), max(1, int(2 * scale)))
            cv2.putText(show, str(i + 1), (x - int(50 * scale), y - int(28 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale, (0, 0, 0),
                        max(1, int(3 * scale)))
            ang = (math.degrees(math.atan2(-r["perp"][1], r["perp"][0])) + 180) % 180
            label = f"{r['what']} {r['p']:.2f}  open {r['w_mm']:.0f}mm" if r["w_mm"] else f"{r['what']} {r['p']:.2f}"
            cv2.putText(show, label, (x + int(20 * scale), y + int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, col,
                        max(1, int(2 * scale)))
            plan.append({"order": i + 1, "what": r["what"], "prob": round(r["p"], 2), "grasp_px": [round(r["c"][0]), round(r["c"][1])],
                         "finger_angle_deg": round(ang, 1), "open_mm": round(r["w_mm"], 1) if r["w_mm"] else None,
                         "length_mm": round(r["l_mm"], 1) if r["l_mm"] else None, "graspable": ok})
        cv2.putText(show, f"instruction: {instr}", (int(12 * scale), int(36 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale,
                    (0, 0, 0), max(1, int(2 * scale)))
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(f"robot/real/{name}_plan.png", show)
        res = {"image": path, "instr": instr, "wanted": wanted, "mm_per_px": round(mmpp, 3) if mmpp else None, "plan": plan,
               "seen": [{"what": r["what"], "prob": round(r["p"], 2), "on_place": r["on_place"],
                         "width_mm": round(r["w_mm"], 1) if r["w_mm"] else None} for r in rows]}
        print(json.dumps(res, ensure_ascii=False), flush=True)
        with open("robot/logs/plan_photo.jsonl", "a", encoding="utf-8") as fp:
            fp.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
