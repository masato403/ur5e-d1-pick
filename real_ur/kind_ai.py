"""物の種類を、ローカルの AI「d1-3B」に Jev 方式で選ばせる（10/19。d1-3B を使うのがこの課題の目的）。
  ・画像：見つけた物を、長い向きが横になるように回して切り抜いた絵（真ん中の写真から）
  ・今の状態（文章）：プログラムが測った値（上から見た長さ・幅・高さ、丸い棒に見えるか）
  ・選択肢：消しゴム・ペン・銀の棒・それ以外（説明に、それぞれの見た目と本当の大きさを書く）
d1-3B は判断の係（d1_server.py、127.0.0.1:5006）で動かす。実機（.venv_ur）からもシミュレーター（.venv）からも同じに使う。
"""
import base64
import json
import os
import time
import urllib.request

import cv2
import numpy as np

PORT = int(os.environ.get("D1_PORT", "5006"))
SIZE = 256                                     # d1-3B に渡す画像の大きさ（256×256 で 1 回 約 40ms、10/13 に測った）

# 選択肢（d1-3B は英語で読む）。説明には、見た目と、実物を測った大きさを書く
CHOICES = {
    "eraser": "an eraser: a white rubber block in a blue and white paper sleeve, about 52 mm long, 20 mm wide and 11 mm tall",
    "pen": "a pen: a blue ballpoint pen, about 145 mm long and 11 mm thick",
    "rod": "a silver rod: a shiny round metal bar, about 77 mm long and 11 mm thick, with a white label at one end",
    "other": "something else: a shadow, a marking on the mat, only a piece of an object, or an unknown item",
}
NAME = {"eraser": "消しゴム", "pen": "ペン", "rod": "銀の棒", "other": None}
# 本物の長さ（mm）。測った長さがこの LEN_RANGE 倍の外なら、その種類は d1-3B に聞く前に選択肢から外す（Jev 方式：数字の比べはプログラム）。
# 外さないと、36mm のかけらを「銀の棒」と答えた（10/19、実機の写真 113 個中 2 個）。ラベルが切れた銀の棒は 57mm（0.74 倍）に見えた
LEN = {"eraser": 52.0, "pen": 145.0, "rod": 77.0}
LEN_RANGE = (float(os.environ.get("KIND_LEN_LO", "0.6")), float(os.environ.get("KIND_LEN_HI", "1.35")))
P_MIN = float(os.environ.get("KIND_P_MIN", "0.4"))   # いちばん高い確率がこれより低い時は、決めない（種類は None）
USE_AI = os.environ.get("KIND_AI", "1") == "1"       # 0 の時は d1-3B を使わず、長さと幅の表で決める（前の決め方に戻す設定。10/19）
RETAKE = int(os.environ.get("KIND_RETAKE", "1"))     # 決められなかった物があれば、撮り直して聞き直す回数（0 で撮り直さない）
TABLE_OK = os.environ.get("KIND_TABLE", "0") == "1"
SAVE = os.environ.get("KIND_SAVE", "")                  # 決めた時に d1-3B に渡した絵を残すフォルダ（空なら残さない）   # 1 の時だけ、判断の係が動いていなければ長さと幅の表で決める（予備）

# 上の面の長さ（mm）の範囲と、幅の範囲（mm）。d1-3B で決めるようにした後は、判断の係が動いていない時の予備（KIND_TABLE=1）と、確かめの比べに使う。
# 10/16 の実機の写真 49 組では、消しゴムの幅 15.8〜19.2・銀の棒の幅 8.2〜13.3 だった。
# 長さだけで分けると、ラベルの所が切れて短く見えた銀の棒（長さ 57mm）を消しゴムと間違えたので、幅 14.5mm を境に分ける（10/19）
KINDS = {
    "消しゴム": (40.0, 65.0, 14.5, 23.0),
    "銀の棒": (50.0, 110.0, 0.0, 14.5),
    "ペン": (110.0, 170.0, 0.0, 23.0),
}


def kind_of(f):
    w, l = f["size"]
    for k, (lo, hi, wmin, wmax) in KINDS.items():
        if lo <= l < hi and wmin <= w <= wmax:
            return k
    return None


def crop_upright(img, contour, pad_long=0.12, pad_short=24):
    """物の輪郭を囲む長方形を、長い向きが横になるように回して切り抜く。長い向きは PAD_LONG の割合、短い向きは pad_short 画素だけ
    まわりを含め、残りは灰色で埋めて正方形にする。正方形のまま切り抜くと、長いペンのそばの消しゴムや銀の棒まで写り込み、
    d1-3B がペンを「銀の棒」と答えた（22 回中 17 回。10/19）ので、ほかの物が写らないようにする。"""
    (cx, cy), (w, h), ang = cv2.minAreaRect(contour)
    if w < h:
        w, h, ang = h, w, ang + 90.0
    cw, ch = int(w * (1 + pad_long)), int(h + 2 * pad_short)
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    M[:, 2] += [cw / 2 - cx, ch / 2 - cy]
    strip = cv2.warpAffine(img, M, (cw, ch), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    side = max(cw, ch)
    out = np.full((side, side, 3), 128, np.uint8)
    y0, x0 = (side - ch) // 2, (side - cw) // 2
    out[y0:y0 + ch, x0:x0 + cw] = strip
    return cv2.resize(out, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def facts(f):
    w, l = f["size"]
    s = (f"Measured by the robot's camera: seen from above it is {l:.0f} mm long and {w:.0f} mm wide, "
         f"and its top is {f['h']:.0f} mm above the mat.")
    if f.get("round"):
        s += " Its cross-section looks round, like a cylinder lying on its side."
    return s


def options(f):
    """測った長さで、ありえない種類を外した選択肢。「それ以外」は必ず残す。"""
    l = f["size"][1]
    return {k: v for k, v in CHOICES.items() if k not in LEN or LEN_RANGE[0] <= l / LEN[k] <= LEN_RANGE[1]}


def ask(img, f, timeout=10.0):
    """1 つの物の種類を d1-3B に聞く。戻り値：(種類の名前（消しゴム・ペン・銀の棒、それ以外は None）, 確率の辞書, かかった ms)。"""
    choices = options(f)
    if len(choices) == 1:                                        # 長さがどの種類にも合わない：聞かずに「それ以外」
        return None, {"other": 1.0}, 0.0
    crop = crop_upright(img, f["contour"])
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    body = {"state": "Close-up image from the robot's wrist camera of one object lying on a dark cutting mat. " + facts(f),
            "question": "What is this object?", "choices": choices,
            "image": base64.b64encode(buf.tobytes()).decode("ascii")}
    t0 = time.perf_counter()
    r = post(body, timeout)
    name = NAME[r["choice"]] if r["probs"][r["choice"]] >= P_MIN else None
    return name, r["probs"], (time.perf_counter() - t0) * 1000


def decide(img, found, table, want=lambda f: True, log=print):
    """見つけた物それぞれの種類を d1-3B に決めさせ、f["kind"]（名前か None）・f["kind_p"]（確率）・f["kind_by"]（"d1" か "表"）に入れる。
    want(f) が偽の物（平ら・マットの外など、つかむ候補にならない物）は聞かずに None。判断の係が動いていない時は止める
    （d1-3B を使うのがこの課題の目的。KIND_TABLE=1 の時だけ、表 table(f) で決めて、そのことを表示する）。"""
    up = USE_AI and ready()
    if not USE_AI:
        log("   ！ KIND_AI=0：種類は d1-3B を使わず、長さと幅の表で決める（前の決め方）")
    elif not up:
        if not TABLE_OK:
            raise SystemExit("判断の係（d1_server.py、d1-3B）が動いていない。先に .venv/Scripts/python.exe real_ur/d1_server.py を動かす")
        log("   ！ 判断の係が動いていないので、種類は長さと幅の表で決める（予備）")
    for f in found:
        f["kind"], f["kind_p"], f["kind_by"], f["kind_unsure"] = None, None, None, False
        if not want(f):
            continue
        if not up:
            f["kind"], f["kind_by"] = table(f), "表"
            continue
        name, probs, ms = ask(img, f)
        if SAVE:                                                 # 渡した絵を残す（確かめ用。OpenCV は日本語のファイル名を書けない）
            os.makedirs(SAVE, exist_ok=True)
            cv2.imwrite(os.path.join(SAVE, f"{time.strftime('%H%M%S')}_{int(f['size'][1])}mm_{name and {v: k for k, v in NAME.items()}[name] or 'other'}.jpg"),
                        crop_upright(img, f["contour"]))
        f["kind"], f["kind_p"], f["kind_by"], f["kind_ms"] = name, probs, "d1", ms
        f["kind_unsure"] = max(probs.values()) < P_MIN                 # 確率が低くて決められなかった（「それ以外」と答えたのとは違う）
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:2]
        log(f"   d1-3B：{name or 'それ以外'}（" + "、".join(f"{NAME.get(k) or 'それ以外'} {p:.2f}" for k, p in top) + f"、{ms:.0f}ms）")
    return found


def unsure(found):
    """確率が低くて種類を決められなかった物。"""
    return [f for f in found if f.get("kind_unsure")]


def recheck(found, again, near_mm=40.0, log=print):
    """決められなかった物を、撮り直した写真で測って種類を決めた物（again）と、中心の位置（near_mm 以内）で結び、決まった種類を写す。
    撮り直しても決められない物は、種類 None のまま（つかまない）。"""
    for f in unsure(found):
        g = min(again, key=lambda g: float(np.linalg.norm(np.asarray(g["center"][:2]) - np.asarray(f["center"][:2]))), default=None)
        if g is None or np.linalg.norm(np.asarray(g["center"][:2]) - np.asarray(f["center"][:2])) > near_mm:
            log("   撮り直したら、決められなかった物が見つからない → つかまない")
            continue
        if g.get("kind_unsure") or g.get("kind_by") is None:
            log("   撮り直しても決められない → つかまない")
            continue
        f.update(kind=g["kind"], kind_p=g["kind_p"], kind_by=g["kind_by"], kind_unsure=False)
        log(f"   撮り直して決まった：{g['kind'] or 'それ以外'}")
    return found


RECORD = None                                          # 確かめ用：並びを入れると、送った問い合わせを残す（robot/eval_d1_fast.py）


def post(body, timeout=10.0):
    """判断の係に 1 問送り、答え（{"choice", "probs", "ms"}）を返す。種類（ここ）と plan_ai の問いが共通で使う。"""
    if RECORD is not None:
        RECORD.append(body)
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/choice", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def ready():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=1.0).read()
        return True
    except OSError:
        return False
