r"""判断の係：ローカルの AI「d1-3B」を 1 回だけ読み込んで裏で動かし続け、ほかのプログラムからの問い合わせに答える（10/19）。
実機のプログラム（.venv_ur、PyTorch 無し）からも、シミュレーター（.venv）からも、同じ形で使う（ハンドの係 hand_server.py と同じ考え）。
Jev 方式：今の状態（文章）と選択肢（説明付き）と画像を渡し、選択肢ごとの確率を返す。学習し直しはしない。
  POST /choice  {"state": 文章, "question": 問い, "choices": {名前: 説明}, "image": JPEG の base64（無くてもよい）}
             → {"choice": 名前, "probs": {名前: 確率}, "ms": かかった時間}
  POST /ask     {"state": 文章, "questions": {名前: {"type": "noul"|"choice"|"score", "instructions", "criteria"}}, "images": [JPEG の base64…]}
             → {"answers": {名前: model.system_one の答え（score は期待値 score・probabilities）}, "ms": かかった時間}
             いくつもの問いを 1 回の計算でまとめて聞く（毎コマの制御：dx・dy の score と、閉じるかの noul。10/20）
使い方（jev の場所で、PyTorch の入った .venv で）:  .venv\Scripts\python.exe real_ur\d1_server.py
"""
import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
PORT = int(os.environ.get("D1_PORT", "5006"))
# 速い動かし方（d1_judge_fast.FastJudge：CUDA グラフ・画像の前処理を GPU で）を使うか。0 で元の動かし方（model.system_one）。
# 問い合わせに "path": "ref" / "fast" を付けると、その問いだけ選べる（同じ答えになるかを比べるため。robot/eval_d1_fast.py）
# 10/19：種類と 4 つの判断の問い 169 個で、選んだ答えは全部同じ（確率の差は最大 0.05）。画像つき 202 → 66ms、文章だけ 150 → 28ms（中央値）
FAST = os.environ.get("D1_FAST", "1") == "1"


def main():
    os.chdir(ROOT)                                   # models/d1-3B を読むため
    import torch
    from PIL import Image
    from d1_fast import D1Fast
    t0 = time.perf_counter()
    d1 = D1Fast(quant="fp8", graph="cuda")
    model = d1.model
    from d1_judge_fast import FastJudge
    judge = FastJudge(d1)
    print(f"d1-3B を読み込んだ（{time.perf_counter() - t0:.0f} 秒）。127.0.0.1:{PORT} で待つ（{'速い動かし方' if FAST else '元の動かし方'}）", flush=True)
    lock = threading.Lock()                          # GPU は 1 つずつ使う
    state = {"asleep": False}

    def sleep():
        """GPU を空ける（紙芝居で絵を描く間。10/28）。モデルをメインメモリへ移す。CUDA グラフは捨てる。"""
        if not state["asleep"]:
            model.to("cpu")
            d1.graphs.clear()
            torch.cuda.empty_cache()
            state["asleep"] = True

    def wake():
        if state["asleep"]:
            model.to("cuda")
            state["asleep"] = False

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if self.path.startswith(("/sleep", "/wake")):    # POST /sleep・/wake（呼ばれた時だけ。眠っていても問い合わせが来たら起きて答える）
                with lock:
                    t = time.perf_counter()
                    (sleep if self.path.startswith("/sleep") else wake)()
                    out = json.dumps({"asleep": state["asleep"], "ms": round((time.perf_counter() - t) * 1000)}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            with lock:
                wake()
            if self.path.startswith("/ask"):
                return self.ask(body)
            images = None
            if body.get("image"):
                images = [Image.open(io.BytesIO(base64.b64decode(body["image"]))).convert("RGB")]
            qd = {"type": "choice", "instructions": body["question"], "criteria": body["choices"]}
            fast = {"fast": True, "ref": False}.get(body.get("path"), FAST)
            with lock, torch.inference_mode():
                if fast:
                    import numpy as np
                    a, ms = judge.choose(body["state"], qd, np.asarray(images[0]) if images else None)
                else:
                    t = time.perf_counter()
                    o = model.system_one(body["state"], {"q": qd}, images=images)
                    ms = (time.perf_counter() - t) * 1000
                    a = o["answers"]["q"]
            out = json.dumps({"choice": a["choice"], "probs": {k: round(float(v), 4) for k, v in a["probabilities"].items()},
                              "ms": round(ms, 1)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            try:
                self.end_headers()
                self.wfile.write(out)
            except ConnectionError:                      # 聞いた側が待ちきれずに切った（答えは捨てる）
                pass

        def ask(self, body):
            images = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in body.get("images") or []] or None
            with lock, torch.inference_mode():
                t = time.perf_counter()
                o = model.system_one(body["state"], body["questions"], images=images)
                ms = (time.perf_counter() - t) * 1000

            def plain(v):
                if isinstance(v, dict):
                    return {k: plain(x) for k, x in v.items()}
                if isinstance(v, (list, tuple)):
                    return [plain(x) for x in v]
                return round(float(v), 4) if hasattr(v, "__float__") and not isinstance(v, (str, bool, int)) else v
            out = json.dumps({"answers": plain(o["answers"]), "ms": round(ms, 1)}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            try:
                self.end_headers()
                self.wfile.write(out)
            except ConnectionError:
                pass

        def do_GET(self):
            out = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
