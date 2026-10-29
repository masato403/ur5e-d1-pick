r"""実機で 1 回通し：ロボットとハンドにつなぎっぱなしのまま、決めた順に物を探してつかみ、元の位置に戻して離す（箱が無い時の試し用）。
1 回ごとにプログラムを起動し直し、ロボットにつなぎ直すと、1 つの物に 48 秒かかった（10/16）。シミュレーターと同じく 1 つの流れで動かす。
物は位置ではなく種類で選ぶ（丸いペンは置くと転がり、位置で選ぶと見失った。10/16）。種類は d1-3B が、切り抜いた画像と測った長さ・幅・高さを見て決める
（kind_ai.decide。10/19〜。判断の係 d1_server.py を先に動かしておく）。段階ごとの時間も出す。
指示（--say）を d1-3B が読んで運ぶ種類を決め、どれから運ぶか・つかむ高さ・持てなかった時のやり直し方も d1-3B が選ぶ（plan_ai。10/19〜）。
--order を付けた時だけ、前と同じく、決めた順に運ぶ（どれから運ぶかは d1-3B に聞かない）。
--box を付けると、手で覚えさせた箱（teach_box.py、box.json）へ運び、縁の上から落とし入れる。付けなければ、つかんだ所に戻して離す（箱が無かった時の試し方）。
使い方:  .venv_ur\Scripts\python.exe real_ur\pick_pass.py --ip 192.0.2.10 [--say ペン以外を箱に入れて] [--box] [--order 消しゴム ペン 銀の棒] [--force 60]
"""
import argparse
import math
import time

import cv2
import numpy as np

import height_by_motion as HM
import kind_ai as KA
import plan_ai as PL
import teach_box as TB
import pick_any as PA
import pick_by_camera as PB
from go_start import go_start
from hand import HandClient
import web_page
from kind_ai import KINDS, kind_of  # noqa: F401（長さと幅の表。予備と比べに使う。10/19 に kind_ai.py へ移した）

LIFT_BASE = PB.LIFT_SPEED                             # 持ち上げる速さ（やり直し方「ゆっくり持ち上げる」で半分にする）
FORCE = PA.FORCE                                      # 物ごとの挟む力（%）。シミュレーターでも同じにするため pick_any に移した（10/19）


class Clock:
    def __init__(self):
        self.t0 = self.t = time.perf_counter()
        self.rows = []

    def lap(self, what):
        now = time.perf_counter()
        self.rows.append((what, now - self.t))
        print(f"   ⏱ {what}：{now - self.t:.1f} 秒（通しで {now - self.t0:.1f} 秒）")
        self.t = now


def look(a, c, r, clock):
    h = HandClient(log=lambda s: None)
    h.open()                                                  # 撮る前に全開にする（閉じた指が画面の真ん中をふさぐ）
    h.close_port()
    clock.lap("ハンドを全開に")
    HM.clear_memo()
    imgA, tcpA, others = HM.capture_two(a.ip, c=c, r=r)
    clock.lap("3 枚撮る（動き・待ち）")
    found = PA.survey(imgA, tcpA, others)
    clock.lap("測る（計算）")
    kinds(imgA, found)
    clock.lap("種類を決める（d1-3B）")
    for _ in range(KA.RETAKE):                                # 決められなかった物があれば、撮り直して聞き直す
        if not KA.unsure(found):
            break
        print(f"   種類を決められなかった物が {len(KA.unsure(found))} 個 → 撮り直して聞き直す")
        imgB, tcpB, othersB = HM.capture_two(a.ip, c=c, r=r)
        KA.recheck(found, kinds(imgB, PA.survey(imgB, tcpB, othersB)))
        clock.lap("撮り直して種類を聞き直す")
    show(imgA, tcpA, others, found)
    return imgA, tcpA, others, found


def kinds(imgA, found):
    """つかむ候補になりうる物（マットの上で、平らでない物）の種類を d1-3B に決めさせる。"""
    return KA.decide(imgA, found, kind_of, want=lambda f: f["on_mat"] >= PA.MAT_RING and f["h"] >= PA.MIN_H)


def show(imgA, tcpA, others, found):
    """撮った 3 枚と、見つけた物の枠を、ページ（cam_view が出す Chrome のタブ）へ送る。緑＝消しゴム・ペン・銀の棒のどれかに当てはまる物。
    黄色の枠＝物を探さない箱の所。"""
    dbg = imgA.copy()
    ex = PA.box_mask(imgA, tcpA)
    if ex is not None:
        cv2.drawContours(dbg, cv2.findContours(ex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 220, 255), 3)
    for i, f in enumerate(found):
        k = f.get("kind")
        col = (0, 200, 0) if k else (0, 0, 255)
        box = cv2.boxPoints(cv2.minAreaRect(f["contour"])).astype(int)
        cv2.drawContours(dbg, [box], -1, col, 3)
        cv2.putText(dbg, f"{i}: h{f['h']:.0f} {f['size'][0]:.0f}x{f['size'][1]:.0f}", tuple(box.min(0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
    web_page.push_image("見つけた物（真ん中の写真）", dbg)
    for name, (im, _) in zip(("右", "左"), others):
        web_page.push_image(f"撮った写真（{name}）", im, 640)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--say", default="全部の物を箱に入れて", help="指示。種類ごとに 運ぶ／残す を d1-3B が決める")
    ap.add_argument("--order", nargs="+", help="運ぶ順を決めて試す時だけ（無ければ、どれから運ぶかを d1-3B が選ぶ）")
    ap.add_argument("--force", type=int, help="挟む力（%）。無ければ物ごとの FORCE")
    ap.add_argument("--box", action="store_true", help="覚えさせた箱へ運ぶ（teach_box.py で先に覚えさせる）")
    a = ap.parse_args()
    box = TB.load() if a.box else None
    if a.box and box is None:                                     # ロボットにつなぐ前に確かめる
        raise SystemExit("箱をまだ覚えていない。先に real_ur/teach_box.py --ip … で覚えさせる")
    # 箱はマットの上に置く（10/19 夜）。覚えた箱があれば、--box が無くても箱の所では物を探さない（光る箱を物と見間違えない・入れた物をまたつかまない）
    PA.BOX = TB.load()
    if PA.BOX is not None and TB.MASK:
        print(f"   箱の所（真ん中 x {PA.BOX['center'][0]:.0f}  y {PA.BOX['center'][1]:.0f} mm、{PA.BOX['when']} に覚えた）では物を探さない（BOX_MASK=0 で探す）")
    web_page.tee_stdout()                                         # print の文字をページにも出す（10/19）
    if KA.USE_AI and not KA.ready() and not KA.TABLE_OK:                        # ロボットにつなぐ前に確かめる（d1-3B を使うのがこの課題の目的）
        raise SystemExit("判断の係（d1_server.py、d1-3B）が動いていない。先に .venv/Scripts/python.exe real_ur/d1_server.py を動かす")
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(a.ip)
    c = RTDEControlInterface(a.ip)
    clock = Clock()
    result = []
    try:
        print(f"==== 指示：{a.say}")
        to_box = PL.instruction(a.say)
        clock.lap("指示を読む（d1-3B）")
        remaining = list(a.order) if a.order else [k for k, v in to_box.items() if v]
        n_carry, i_carry = len(remaining), 0                          # 箱の中で物ごとに位置をずらすための数
        print("   運ぶ物：" + ("・".join(remaining) or "なし") + "（残す物：" + ("・".join(k for k, v in to_box.items() if not v) or "なし") + "）")
        while remaining:
            want = remaining[0] if a.order else None
            print(f"==== {want or '次に運ぶ物（d1-3B が選ぶ）：' + '・'.join(remaining)}")
            go_start(c, r)
            clock.lap("最初の位置へ")
            imgA, tcpA, others, found = look(a, c, r, clock)
            cands = []
            for f in found:
                far = math.hypot(f["center"][0] - tcpA[0] * 1000, f["center"][1] - tcpA[1] * 1000)
                why = PA.why_not(f, far)
                k = f["kind"]
                print(f"   {k or '？'}（{f['kind_by'] or '聞かず'}）：高さ {f['h']:.1f}mm、上の面 {f['size'][0]:.1f} × {f['size'][1]:.1f} mm、"
                      f"中心 x {f['center'][0]:.1f}  y {f['center'][1]:.1f}" + ("（写真の端に触れている）" if f.get("by_edge") else "")
                      + (f" → 外す（{'、'.join(why)}）" if why else ""))
                if (k == want if want else k in remaining) and not why:
                    cands.append(f)
            best = {}
            for g in cands:                                     # 同じ種類が 2 つあれば、合い具合の良い方（銀の棒が二重に見つかった）
                if g["kind"] not in best or g["match"] > best[g["kind"]]["match"]:
                    best[g["kind"]] = g
            cands = list(best.values())
            if not cands:
                if want:
                    print(f"   {want} が見つからないので、次へ")
                    result.append((want, "見つからない"))
                    remaining.pop(0)
                    continue
                print(f"   {'・'.join(remaining)} が見つからないので、終わる")
                result += [(w, "見つからない") for w in remaining]
                break
            f = cands[0] if want else PL.order(cands, tcpA, found)   # どれから運ぶかは d1-3B が選ぶ
            want = f["kind"]
            remaining.remove(want)
            clock.lap("どれから運ぶか（d1-3B）")
            if box:                                                    # 箱の長い向きに、物ごとに位置をずらして落とす
                place, place_kw = TB.spot(box, i_carry, n_carry), {"place_far": PB.BOX_FAR}
                i_carry += 1
            else:
                place, place_kw = [float(f["center"][0]), float(f["center"][1])], {}
            ok, empty, drop = False, False, 0.0
            PB.LIFT_SPEED = LIFT_BASE                                  # 物ごとに、持ち上げる速さを元に戻す
            try:
                for attempt in range(1 + PA.RETRY):
                    grip = PL.grip(f, drop, empty)                     # 指先を下ろす高さは d1-3B が候補から選ぶ
                    print(f"→ {want} をつかむ{'' if attempt == 0 else f'（やり直し {attempt} 回目）'}：指先を机から {grip:.1f}mm まで下ろし、"
                          + (f"箱の x {place[0]:.1f}  y {place[1]:.1f} の上から落とす" if box else f"元の位置 x {place[0]:.1f}  y {place[1]:.1f} に戻す")
                          + 
                          f"（力 {a.force if a.force is not None else FORCE.get(want, 60)}%）")
                    force = a.force if a.force is not None else FORCE.get(want, 60)
                    if box:
                        place_kw["place_h"] = TB.release_h(box, grip)       # 物の底が縁の上 10mm に来る高さで離す
                    res = PB.grasp(c, r, r.getActualTCPPose(), f["center"], f["longv"], grip, force, place, False,
                                   miss="retry" if attempt < PA.RETRY else "stop", **place_kw)
                    clock.lap("つかむ・運ぶ・置く" if res is not False else "つかみ損ねて上がる")
                    if res is not False:
                        ok = True
                        break
                    empty = PB.LAST["closed"] is not None and PB.LAST["closed"] <= PB.HOLD_MM
                    how = PL.retry(f, attempt, grip, empty)            # やり直し方は d1-3B が選ぶ（できない選択肢は先に外す）
                    if how == "skip":
                        print("   この物はやめる")
                        break
                    if how == "lower":
                        drop += PA.GRIP_STEP
                    if how == "slow":                                      # 滑った時：次は持ち上げる速さを半分にする（10/19 夜）
                        PB.LIFT_SPEED = LIFT_BASE / 2
                    # 物の真上・手首を回したまま撮ると、長い物（ペン）が写る範囲から外れて形で選べなかった（10/16）→ 最初の位置から撮り直す
                    go_start(c, r)
                    imgA, tcpA2, others = HM.capture_two(a.ip, c=c, r=r)
                    near = [g for g in PA.survey(imgA, tcpA2, others)
                            if np.linalg.norm(g["center"][:2] - f["center"][:2]) < PA.REFIND_MM]
                    again = [g for g in kinds(imgA, near) if g["kind"] == want]
                    clock.lap("撮り直して測り直す")
                    if not again:
                        print("   撮り直したら見つからないので、次へ")
                        break
                    f = min(again, key=lambda g: np.linalg.norm(g["center"][:2] - f["center"][:2]))
            except SystemExit as e:                               # 持てずに運ばず止めた、など：次の物へ
                print(f"   止めた：{e}")
                clock.lap("止めた")
            result.append((want, "成功" if ok else "失敗"))
        go_start(c, r)
        clock.lap("最初の位置へ")
    finally:
        c.stopScript()
        c.disconnect()
        r.disconnect()
    print("==== 結果：" + "、".join(f"{w} {s}" for w, s in result) + f"（通しで {time.perf_counter() - clock.t0:.0f} 秒）")
    tot = {}
    for what, dt in clock.rows:
        tot[what] = tot.get(what, 0.0) + dt
    print("   時間の内訳：" + "、".join(f"{k} {v:.1f} 秒" for k, v in sorted(tot.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
