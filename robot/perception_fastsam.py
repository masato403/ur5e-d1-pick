"""見る係の FastSAM 版（何もない机の画像を使わない。実環境向け）。

FastSAM-s：物の輪郭を切り出すだけの公開モデル（物の種類は判断しない。追加の学習なし）。
切り出した輪郭のうち、物くらいの大きさの物だけを「物らしい画素」にする。番号の付け方などは SegPerception と同じ。
比べた結果（robot/test_detect.py、30 場面・物 120 個）：きれい・実環境ふう・大きな影ありのすべてで 120/120、約 26 ms。
"""
import cv2
import numpy as np
import torch

from run_tools import SegPerception

MIN_AREA, MAX_AREA, MAX_SIDE = 30, 6000, 140


class FastSamPerception(SegPerception):
    def __init__(self, min_area=MIN_AREA):
        from ultralytics import FastSAM
        self.model = FastSAM("FastSAM-s.pt")
        self.min_area = min_area
        self.objs, self.next_id, self.held = {}, 1, None
        self.placed, self.births = {}, []
        self._cache = (None, None)

    def foreground(self, rgb):
        if self._cache[0] is rgb:                    # 同じ画像なら、もう一度は計算しない
            return self._cache[1]
        with torch.inference_mode():
            r = self.model(rgb[:, :, ::-1], retina_masks=True, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
            if r.masks is None:
                m = np.zeros(rgb.shape[:2], np.uint8)
            else:
                # 輪郭の選別を GPU の上でまとめて行う（1 つずつ CPU に取り出すと、それだけで約 15 ms かかっていた）
                mk = r.masks.data > 0.5                                  # (輪郭の数, 高さ, 幅)
                area = mk.sum((1, 2))
                rows, cols = mk.any(2), mk.any(1)
                ar_h, ar_w = torch.arange(mk.shape[1], device=mk.device), torch.arange(mk.shape[2], device=mk.device)
                h = torch.where(rows, ar_h, -1).amax(1) - torch.where(rows, ar_h, 10 ** 6).amin(1) + 1
                w = torch.where(cols, ar_w, -1).amax(1) - torch.where(cols, ar_w, 10 ** 6).amin(1) + 1
                ok = (area >= MIN_AREA) & (area <= MAX_AREA) & (torch.maximum(h, w) <= MAX_SIDE)   # 机・箱の床・箱の壁は除く
                m = mk[ok].any(0).to(torch.uint8).cpu().numpy() if bool(ok.any()) else np.zeros(mk.shape[1:], np.uint8)
                if m.shape != rgb.shape[:2]:
                    m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        self._cache = (rgb, m)
        return m

    def refine(self, rgb, frame_no, robot_mask=None, min_share=0.05):
        """判断の前に、見えている物ごとに拡大してもう一度切り出す（手首のカメラで近づいて見るのと同じ考え方）。
        2 つ以上に分かれた物は、元の番号を消して、分かれた物それぞれに新しい番号をつける。分かれた物には輪郭（mask）も覚えておく。
        戻り値：分けた番号 → 新しい番号の一覧"""
        import test_real as T
        from sim_tools import ToolEnv
        fg = self.foreground(rgb).copy()
        if robot_mask is not None:
            fg[robot_mask] = 0
        n, lab = cv2.connectedComponents(cv2.dilate(fg, np.ones((9, 9), np.uint8)))
        changes = {}
        for k, o in list(self.visible(frame_no).items()):
            u, v = int(o["uv"][0]), int(o["uv"][1])
            c = lab[min(v, lab.shape[0] - 1), min(u, lab.shape[1] - 1)]
            if c == 0:
                continue
            mk = (lab == c) & (fg > 0)
            ys, xs = np.nonzero(mk)
            whole = {"mask": mk, "uv": (float(xs.mean()), float(ys.mean())), "box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))}
            with torch.inference_mode():
                parts = T.zoom_split(self.model, rgb, whole, min_share=min_share)
            # 分けすぎを戻す：接していて一直線に並ぶ 2 つ（ドライバーの持ち手と軸など）は 1 つにまとめる
            merged = True
            while merged and len(parts) >= 2:
                merged = False
                for i in range(len(parts)):
                    for j in range(i + 1, len(parts)):
                        a_, b_ = parts[i]["mask"], parts[j]["mask"]
                        if (cv2.dilate(a_.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)[b_].any() and T.collinear(a_, b_):
                            mk = a_ | b_
                            ys, xs = np.nonzero(mk)
                            parts[i] = {"mask": mk, "uv": (float(xs.mean()), float(ys.mean())),
                                        "box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))}
                            del parts[j]
                            merged = True
                            break
                    if merged:
                        break
            if len(parts) < 2:
                continue
            del self.objs[k]
            changes[k] = []
            for p in parts:
                pu, pv = p["uv"]
                x1, y1, x2, y2 = p["box"]
                nid = self.next_id
                self.next_id += 1
                self.objs[nid] = {"xy": ToolEnv.pixel_to_table(pu, pv), "uv": (pu, pv, x2 - x1 + 1, y2 - y1 + 1), "seen": frame_no, "mask": p["mask"]}
                self.births.append({"frame": frame_no, "id": nid, "split_from": k})
                changes[k].append(nid)
        return changes
