"""画像に番号を書き込んで d1-3B に選ばせる方法（Set-of-Mark）を、UR5e のシミュレーターの場面で確かめる。

位置は画像処理（ここではシミュレーターの正解の枠）で分かっている前提で、d1-3B には「どれか」だけを選ばせる。
比べる聞き方（どれも追加学習なし・Jev と同じ「状況と選択肢」）：
  mark_color : 番号つき画像で「赤い積み木は何番？」
  mark_box   : 番号つき画像で「箱（黄色い四角）にいちばん近い積み木は何番？」
  mark_odd   : 番号つき画像で「箱の中に入っていない積み木は何番？」（1 個だけ外にある場面）
  where_color: 番号なし画像で「赤い積み木は 左・中央・右 のどこ？」（昨日までの聞き方）
  python robot/test_som.py [場面の数=40]
"""
import json
import os
import random
import statistics
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
os.chdir(os.path.join(HERE, ".."))

from sim_env import BIN, BIN_HALF, BLOCKS, HALF, Env  # noqa: E402

STATE = "Camera image of a robot work table seen from above. The yellow square is the box. Numbers mark the objects."


def boxes(env):
    """各積み木の画像上の枠（シミュレーターの正解。実機では YOLO などの検出結果）。"""
    out = {}
    for k, cx, cy, w, h in env.labels(min_px=20):
        name = ["red", "blue", "green", "box"][k]
        if name != "box":
            out[name] = (cx * 640, cy * 480, w * 640, h * 480)
    return out


def draw_marks(img, objs, order):
    im = img.copy()
    for num, name in enumerate(order, 1):
        cx, cy, w, h = objs[name]
        x, y = int(cx + w / 2 + 12), int(cy - h / 2 - 10)
        cv2.circle(im, (x, y), 13, (255, 255, 255), -1)
        cv2.circle(im, (x, y), 13, (0, 0, 0), 2)
        cv2.putText(im, str(num), (x - 7, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    return im


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    from d1_fast import D1Fast
    m = D1Fast(quant="fp8").model
    env = Env(seg=True)
    rng = np.random.default_rng(3)
    rr = random.Random(3)
    res = {k: {"ok": 0, "n": 0, "ms": []} for k in ("mark_color", "mark_box", "mark_odd", "where_color")}
    os.makedirs("robot/som", exist_ok=True)
    for i in range(n):
        odd = i % 2 == 1                                     # 半分の場面は、2 個を箱に入れて 1 個だけ外に置く
        env.reset(rng, n_in_bin=2 if odd else 0)
        if rr.random() < 0.7:
            env.random_arm(rng)
        objs = boxes(env)
        if len(objs) < 3:
            continue
        rgb = env.render()
        order = list(objs)
        rr.shuffle(order)                                    # 番号の付け方は毎回ばらばらにする
        marked = draw_marks(rgb, objs, order)
        if i < 6:
            cv2.imwrite(f"robot/som/{i:02d}.png", marked[:, :, ::-1])
        opts = {f"obj{k}": f"object number {k}" for k in range(1, len(order) + 1)}
        color = rr.choice(BLOCKS)
        pos = {b: env.block_pos(b)[:2] for b in BLOCKS}
        in_box = {b: bool(np.all(np.abs(pos[b] - BIN) < BIN_HALF)) for b in BLOCKS}
        closest = min(BLOCKS, key=lambda b: np.linalg.norm(pos[b] - BIN) if not in_box[b] else 99)
        tests = {
            "mark_color": (marked, f"Which numbered object is the {color} block?", opts, f"obj{order.index(color) + 1}"),
            "mark_box": (marked, "Which numbered block outside the box is closest to the yellow box?", opts,
                         f"obj{order.index(closest) + 1}"),
        }
        if odd:
            outside = [b for b in BLOCKS if not in_box[b]]
            if len(outside) == 1:
                tests["mark_odd"] = (marked, "Which numbered block is NOT inside the yellow box?", opts,
                                     f"obj{order.index(outside[0]) + 1}")
        u = objs[color][0]
        tests["where_color"] = (rgb, f"Where is the {color} block in the image?",
                                {"left": "left third", "center": "middle third", "right": "right third"},
                                "left" if u < 640 / 3 else ("center" if u < 640 * 2 / 3 else "right"))
        for name, (im, qtext, crit, ans) in tests.items():
            q = {"q": {"type": "choice", "instructions": qtext, "criteria": crit}}
            pil = Image.fromarray(cv2.resize(im, (512, 384)))
            torch.cuda.synchronize()
            t = time.perf_counter()
            o = m.system_one(STATE if name != "where_color" else "Camera image of a robot work table seen from above.",
                             q, images=[pil])
            torch.cuda.synchronize()
            r = res[name]
            r["n"] += 1
            r["ok"] += o["answers"]["q"]["choice"] == ans
            r["ms"].append((time.perf_counter() - t) * 1000)
    out = {k: {"acc": round(v["ok"] / max(1, v["n"]), 3), "n": v["n"],
               "ms_median": round(statistics.median(v["ms"]), 1) if v["ms"] else None} for k, v in res.items()}
    print(json.dumps(out, ensure_ascii=False), flush=True)
    with open("robot/logs/som.jsonl", "a", encoding="utf-8") as fp:
        fp.write(json.dumps({"n": n, "size": "512x384", "result": out}) + "\n")


if __name__ == "__main__":
    main()
