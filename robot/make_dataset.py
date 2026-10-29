"""シミュレーターで学習画像と正解の枠を自動で作る（人が印を付ける作業はいらない）。

  python robot/make_dataset.py 3000 300
出力：robot/data/images/{train,val}/*.jpg、robot/data/labels/{train,val}/*.txt、robot/data/blocks.yaml
"""
import os
import sys

import cv2
import numpy as np

from sim_env import CLASSES, Env

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    n_train, n_val = int(sys.argv[1]), int(sys.argv[2])
    env = Env(seg=True)
    rng = np.random.default_rng(1)
    for split, n in (("train", n_train), ("val", n_val)):
        os.makedirs(f"{OUT}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)
        for i in range(n):
            env.reset(rng, n_in_bin=int(rng.integers(0, 4)))
            env.randomize_look(rng)
            if rng.random() < 0.8:
                env.random_arm(rng)
            img = env.render()
            cv2.imwrite(f"{OUT}/images/{split}/{i:05d}.jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 96))])
            with open(f"{OUT}/labels/{split}/{i:05d}.txt", "w") as fp:
                for k, cx, cy, w, h in env.labels():
                    fp.write(f"{k} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    with open(f"{OUT}/blocks.yaml", "w") as fp:
        fp.write(f"path: {OUT.replace(os.sep, '/')}\ntrain: images/train\nval: images/val\nnames:\n")
        for k, c in enumerate(CLASSES):
            fp.write(f"  {k}: {c}\n")
    print("done", n_train, n_val)


if __name__ == "__main__":
    main()
