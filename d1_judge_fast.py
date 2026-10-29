"""d1-3B の判断を速くする（判断の係 real_ur/d1_server.py 用。10/19）。状態の文章と選択肢が問い合わせごとに変わっても使える形。

  ・文章だけの問い（指示・どれから運ぶか・つかむ高さ・やり直し方）：D1Fast の graph="cuda" と同じ。文字の並びを決まった長さ（段）まで
    右に詰め物をして、段ごとに記録した CUDA グラフを再生する
  ・画像つきの問い（物の種類）：画像はいつも 256×256 なので、画像を読む部分（前処理を GPU で → SigLIP2 → 変換）は 1 つの CUDA グラフにする。
    文章の部分は、画像の特徴を入れた埋め込みを段の長さまで右に詰めて、段ごとの CUDA グラフで読む
答えは問題文の最後の位置で読むので、右に詰め物を足しても答えは変わらない（前の位置の計算は後ろの文字の影響を受けない）。
元の動かし方（model.system_one）と同じ答えになるかは robot/eval_d1_fast.py で確かめる。
"""
import importlib
import time

import numpy as np
import torch

from d1_fast import BUCKETS, D1Fast

PATCH = 16
SIDE = 256


class FastJudge:
    def __init__(self, d1: D1Fast):
        self.d1, self.model, self.eng = d1, d1.model, d1.engine
        pkg = type(self.eng).__module__.rsplit(".", 1)[0]
        self.P = importlib.import_module(pkg + ".prompt")
        from PIL import Image
        self.blank = Image.new("RGB", (SIDE, SIDE))
        q = d1.as_question({"type": "choice", "instructions": "x", "criteria": {"a": "a", "b": "b"}})
        inp = self.eng._image_inputs(self._text("x", q), [self.blank])
        self.ss = inp["spatial_shapes"].cpu()                       # CUDA グラフの中で CPU に取り出されるので、最初から CPU に置く
        self.pam = inp["pixel_attention_mask"]
        self.n_patch = inp["pixel_values"].shape[1]
        self.gh, self.gw = (int(x) for x in self.ss[0])
        assert self.gh * self.gw == self.n_patch and self.gh * PATCH == SIDE, "画像は 256×256 の 1 枚（分割なし）になること"
        self.pix = inp["pixel_values"].clone()
        self.vg = None                                               # 画像を読む部分の CUDA グラフ
        self.tg = {}                                                 # 段の長さ → 文章を読む部分の CUDA グラフ
        self.img_id = self.model.config.image_token_id

    def _text(self, state, q):
        e, P = self.eng, self.P
        return (P.prefix_text(e.tokenizer, state, e.bos, e.state_style, e.system, e._image_markup(1))
                + P.suffix_text(e.tokenizer, q, e.lead, e.option_style))

    # ------------------------------------------------------------ 画像
    def _prep_gpu(self, rgb):
        """0〜1 → (x-0.5)/0.5 → 16×16 のマス目ごとに (縦, 横, 色) の順に並べる（元の前処理と同じ並び。d1_image_fast と同じ）。"""
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to("cuda").permute(2, 0, 1).unsqueeze(0).float()
        if x.shape[-2:] != (SIDE, SIDE):
            x = torch.nn.functional.interpolate(x, size=(SIDE, SIDE), mode="bicubic", align_corners=False)
        x = (x / 255.0 - 0.5) / 0.5
        x = x.reshape(1, 3, self.gh, PATCH, self.gw, PATCH).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(1, self.gh * self.gw, PATCH * PATCH * 3)

    def _vision(self, pix):
        mm = self.model.model
        vt = mm.vision_tower(pixel_values=pix, spatial_shapes=self.ss, pixel_attention_mask=self.pam,
                             return_dict=True).last_hidden_state
        f = vt[0, :self.n_patch].reshape(1, self.gh, self.gw, -1)
        return mm.multi_modal_projector(f).reshape(-1, mm.get_input_embeddings().weight.shape[-1])

    @staticmethod
    def _capture(fn):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        return g, out

    def _image_emb(self, rgb):
        if self.vg is None:
            self.vg = self._capture(lambda: self._vision(self.pix))
        self.pix.copy_(self._prep_gpu(rgb))
        self.vg[0].replay()
        return self.vg[1]

    # ------------------------------------------------------------ 文章（埋め込みを段の長さまで右に詰める）
    def _lm(self, L):
        if L not in self.tg:
            mm = self.model.model
            D = mm.get_input_embeddings().weight.shape[-1]
            x = torch.zeros(1, L, D, dtype=torch.bfloat16, device="cuda")
            idx = torch.zeros(1, dtype=torch.long, device="cuda")

            def fn():
                h = mm.language_model(inputs_embeds=x).last_hidden_state.index_select(1, idx)
                return self.model.lm_head(h)[0, 0].float()
            g, out = self._capture(fn)
            self.tg[L] = (g, x, idx, out)
        return self.tg[L]

    @torch.inference_mode()
    def choose(self, state, qd, rgb=None):
        """Jev と同じ形の答え（{"choice", "probabilities"} など）と、かかった ms。rgb：H×W×3 の uint8（RGB）か None。"""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        q = self.d1.as_question(qd)
        if rgb is None:
            row = self.d1.tok.encode(self.eng.render(state, q), add_special_tokens=False)
            lz = self.d1.logz(row)                                   # D1Fast の graph="cuda"
        else:
            ids = self.eng._image_inputs(self._text(state, q), [self.blank])["input_ids"].to("cuda")
            n = ids.shape[1]
            emb = self._image_emb(rgb)
            x0 = self.model.model.get_input_embeddings()(ids)
            x0[0, (ids[0] == self.img_id).nonzero().squeeze(1)] = emb.to(x0.dtype)
            g, x, idx, out = self._lm(self.d1._bucket(n))
            x.zero_()
            x[0, :n] = x0[0]
            idx.fill_(n - 1)
            g.replay()
            r = out
            lz = r - torch.logsumexp(r, dim=-1)
        probs = self.eng._readout(q, lz)
        ans = self.d1.answer(q, probs)
        torch.cuda.synchronize()
        return ans, (time.perf_counter() - t0) * 1000
