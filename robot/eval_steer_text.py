r"""追い方の問い（track_real.steer_ask）の書き方を、シミュレーターを回さずに d1-3B で比べる（10/22）。
物の真上からのずれ・高さ・見えているかを変えた状態を作り、書き方ごとに「近い（指の間に入る）時は下りる、遠い時は下りない」を選べるかを数える。
使い方:  .venv\Scripts\python.exe robot\eval_steer_text.py   （判断の係 real_ur/d1_server.py を先に動かす）"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "real_ur"))
sys.path.insert(0, HERE)
import test_sim_grasp  # noqa: E402,F401  （試験と同じ順に読み込み、serial などの無い部品をにせものに差し替える）
import track_real as TR  # noqa: E402

VARIANTS = {"前の文": (0, False), "＋すき間の事実": (0, True), "結果を書く": (2, False), "結果を書く＋中立の問い": (3, False),
            "結果を強く": (4, False), "結果を強く＋中立の問い": (5, False), "シューティング式": (8, False), "7＋問いに確かめる": (9, False)}
if len(sys.argv) > 1:
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in sys.argv[1:]}
KINDS = {"消しゴム": 24.0, "ペン": 12.0, "銀の棒": 14.0}    # 三脚で測った幅（mm）
DISTS = (1, 3, 6, 10, 20)
HEIGHTS = (100, 60, 13)
AGES = (0.05, 0.6)


def main():
    rows = []
    for name, (text, facts) in VARIANTS.items():
        TR.STEER_TEXT, TR.STEER_FACTS = text, facts
        ok = n = n_desc_near = n_near = n_desc_far = n_far = 0
        ms = []
        for kind, w in KINDS.items():
            gap = max(2.0, (TR.OPEN_MM - w) / 2 - 1.0)
            for d in DISTS:
                for h in HEIGHTS:
                    for age in AGES:
                        fut, only = TR.steer_ask(kind, h, d, 10.0, age, True, h - 2.4, "", False, gap)
                        a = {"choice": only} if only else fut.result()
                        ms.append(a.get("ms", 0))
                        desc = a["choice"] == "descend"
                        inside = d <= gap
                        n += 1
                        ok += desc == inside
                        if inside:
                            n_near += 1
                            n_desc_near += desc
                        else:
                            n_far += 1
                            n_desc_far += desc
        rows.append((name, ok, n, n_desc_near, n_near, n_desc_far, n_far, sum(ms) / len(ms)))
        print(f"{name}: 正しい {ok}/{n}、近い時に下りた {n_desc_near}/{n_near}、遠い時に下りた {n_desc_far}/{n_far}、1 問 {sum(ms) / len(ms):.0f}ms", flush=True)


def main_close():
    """閉じるかの問い。正しい答え：ずれ＋見えない間のずれが指のすき間以内なら閉じる、超えれば追う。"""
    for name, ct in {"前の文": 0, "結果を書く＋中立の問い": 1, "結果を書く（問いは前のまま）": 2, "失敗をはっきり＋追う結果＋中立": 3, "失敗をはっきり＋追う結果＋今閉じる？": 4, "シューティング式": 5, "＋ずれの見積もり・追うの正直な結果": 6, "シューティング式＋ずれの見積もりだけ": 7}.items():
        TR.CLOSE_TEXT = ct
        ok = n = 0
        cnt = {"close_ok": [0, 0], "follow_ok": [0, 0]}
        for kind, w in KINDS.items():
            gap = max(2.0, (TR.OPEN_MM - w) / 2 - 1.0)
            for d in (0.5, 1, 3, 6, 10):
                for v in (10.0, 30.0):
                    for age in (0.05, 0.3, 1.0):
                        fut, only = TR.close_ask(kind, v, w, age, "", d)
                        a = {"choice": only} if only else fut.result()
                        unc = (TR.UNC_K * v * age + TR.UNC_BASE if ct >= 6 else v * age) if age >= 0.15 else 0
                        want = "close" if d + unc <= gap else "follow"
                        n += 1
                        ok += a["choice"] == want
                        cnt[want + "_ok"][0] += a["choice"] == want
                        cnt[want + "_ok"][1] += 1
        print(f"{name}: 正しい {ok}/{n}、閉じるべき時に閉じた {cnt['close_ok'][0]}/{cnt['close_ok'][1]}、"
              f"追うべき時に追った {cnt['follow_ok'][0]}/{cnt['follow_ok'][1]}", flush=True)


def main_look():
    """止まって手首で撮るか（追い方の問いに look がある時）。正しい答え：見えない間のずれ（速さ×秒）が指のすき間を超えたら撮る。
    ずれがすき間以内で、ずれ＋見えない間のずれもすき間以内なら下りる。それ以外は下りず・撮らず（寄せる）。"""
    for name, st in {"結果を強く＋中立の問い": 5, "＋見えない間のずれを書く": 6, "＋古い見当・撮れば分かり直す": 7, "シューティング式": 8, "7＋問いに確かめる": 9}.items():
        TR.STEER_TEXT, TR.STEER_FACTS = st, False
        ok = n = 0
        cnt = {}
        for kind, w in KINDS.items():
            gap = max(2.0, (TR.OPEN_MM - w) / 2 - 1.0)
            for d in (1, 3, 10):
                for h in (60, 13):
                    for v in (10.0, 30.0):
                        for age in (0.3, 1.0, 2.0):
                            fut, only = TR.steer_ask(kind, h, d, v, age, True, h - 2.4, "", True, gap)
                            a = {"choice": only} if only else fut.result()
                            unc = v * age
                            want = "look" if unc > gap else ("descend" if d + unc <= gap else "move")
                            got = a["choice"] if a["choice"] in ("look", "descend") else "move"
                            n += 1
                            ok += got == want
                            c = cnt.setdefault(want, [0, 0])
                            c[0] += got == want
                            c[1] += 1
        print(f"{name}: 正しい {ok}/{n}、" + "、".join(f"{ {'look': '撮るべき', 'descend': '下りるべき', 'move': '寄せるべき'}[k] } {c[0]}/{c[1]}"
                                                 for k, c in cnt.items()), flush=True)


def main_dir():
    """向き選び（TR_D1_STEER=2）。正しい答え：指の間に入る時は下りる。入らない時は、物の今の真上にいちばん近づく向き（いちばん良い向きと 2mm 差まで）。"""
    import numpy as np
    TR.on_mat = lambda p, margin=0.0: True                     # マットの真ん中で聞く形にする（向きを外さない）
    for name, dt in {"前の文": 0, "結果を書く＋中立の問い": 1, "シューティング式": 2}.items():
        TR.DIR_TEXT = dt
        ok = n = 0
        cnt = {}
        for kind, w in KINDS.items():
            gap = max(2.0, (TR.OPEN_MM - w) / 2 - 1.0)
            for d in (2, 5, 10, 20, 40):
                for ang in (0, 60, 135, 200, 290):
                    for h in (100, 13):
                        cur = np.zeros(2)
                        tgt = d * np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
                        vdir = np.array([np.cos(np.radians(ang + 90)), np.sin(np.radians(ang + 90))])
                        (fut, step) = TR.steer_ask_dir(kind, h, cur, tgt, tgt + vdir * 10 * TR.STEER_PRED, 10.0, 0.05, d < 30, h - 2.4, "", gap)
                        a = fut.result()
                        if d <= gap:
                            want, good = "descend", a["choice"] == "descend"
                        else:
                            dist = {k: np.linalg.norm(tgt - cur - step * np.array(v)) for k, v in TR.DIRS.items()}
                            best = min(dist.values())
                            want, good = "move", a["choice"] in dist and dist[a["choice"]] <= best + 2.0
                        n += 1
                        ok += good
                        c = cnt.setdefault(want, [0, 0])
                        c[0] += good
                        c[1] += 1
        print(f"{name}: 正しい {ok}/{n}、" + "、".join(f"{ {'descend': '下りるべき', 'move': '近づく向き'}[k] } {c[0]}/{c[1]}" for k, c in cnt.items()), flush=True)


def main_hes():
    """迷い 3 秒の決まりを外した時（TR_D1_HES）。下りれば指の間に入る時は迷いに関係なく下りる。入らない時は、迷いが 3 秒未満なら下りず、3 秒以上なら下りる（決まりと同じ答え）。"""
    TR.STEER_TEXT, TR.STEER_FACTS = 5, False
    for name, (on, ht) in {"迷いと待つと何が起きるかを書く（1）": (True, 1), "＋待つ選択肢に迷って見える・下りるに外れたらやり直す（2）": (True, 2)}.items():
        TR.HES_TEXT = ht
        ok = n = 0
        cnt = {}
        for kind, w in KINDS.items():
            gap = max(2.0, (TR.OPEN_MM - w) / 2 - 1.0)
            for over in (-1, 1, 3, 6):
                d = gap + over
                for hes in (0.0, 1.0, 2.0, 3.5, 5.0):
                    for v in (10.0, 30.0):
                        extra = TR.hes_fact(hes, 2.0 + hes, v) if (on and hes >= 0.05) else ""
                        fut, only = TR.steer_ask(kind, 13, d, v, 0.05, True, 10.6, extra, False, gap, hes)
                        a = {"choice": only} if only else fut.result()
                        want = "descend" if (d <= gap or hes >= TR.HESITATE_S) else "move"
                        got = "descend" if a["choice"] == "descend" else "move"
                        k = "入る" if d <= gap else ("迷い 3 秒以上" if hes >= TR.HESITATE_S else "迷い 3 秒未満")
                        c = cnt.setdefault(k, [0, 0])
                        c[0] += got == want
                        c[1] += 1
                        n += 1
                        ok += got == want
        print(f"{name}: 正しい {ok}/{n}、" + "、".join(f"{k} {c[0]}/{c[1]}" for k, c in cnt.items()), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["hes"]:
        main_hes()
    elif sys.argv[1:2] == ["dir"]:
        main_dir()
    elif sys.argv[1:2] == ["close"]:
        main_close()
    elif sys.argv[1:2] == ["look"]:
        main_look()
    else:
        main()
