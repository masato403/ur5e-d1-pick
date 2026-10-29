r"""箱の位置を、手で動かした腕の手先の位置から覚えさせる（10/19）。腕は動かさない（RTDE の受け側だけを使う）。
ティーチペンダントのフリードライブで、指先（道具の点）を箱の底の長い向きの真ん中の線の「片方の端の真上・縁の高さ」に合わせて Enter、
次に「もう片方の端の真上・縁の高さ」に合わせて Enter。2 か所から、箱の真ん中・長い向き・物を並べられる長さ・縁の高さが分かるので、
箱の大きさを測らなくてよい。結果は real_ur/box.json に書き、pick_pass.py --box が使う。
使い方:  .venv_ur\Scripts\python.exe real_ur\teach_box.py --ip 192.0.2.10
         .venv_ur\Scripts\python.exe real_ur\teach_box.py --show        （覚えた値を出すだけ。ロボットにつながない）
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "box.json")
MARGIN = 30.0       # 並べる時、端からあける長さ（mm）。物の半分の幅と、落ちた時のずれの分
DROP_GAP = 10.0     # 離す時の、物の底と縁の上とのすき間（mm）。シミュレーターの試験と同じ
# 箱をマットの上に置く時（10/19 夜）、探す範囲から箱の所を外す。光る箱を物と見間違えず、箱に入れた物をまたつかみに行かない。
# 覚えさせるのは底の長い向きの両端だけなので、幅は箱の大きさ（ステンレスのトレー 縦 15cm）を使う
BOX_W = float(os.environ.get("BOX_W", 150))        # 箱の幅（mm、外の大きさ）
BOX_END = float(os.environ.get("BOX_END", 15))     # 覚えた底の端から、縁の外側までの長さ（mm）。トレーの斜めの壁と折り返しの分
BOX_PAD = float(os.environ.get("BOX_PAD", 15))     # さらに外へ広げる余裕（mm）。覚えた位置のずれ・箱の映り込み・縁の影の分
MASK = os.environ.get("BOX_MASK", "1") == "1"      # 0 で前と同じ（箱の所も探す）


def load():
    """覚えた箱（無ければ None）。{"center": [x, y] mm, "axis": [ux, uy], "length": mm, "rim": mm（机から）}"""
    try:
        with open(PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except OSError:
        return None


def spot(box, i, n):
    """n 個の物の i 番目（0 から）を落とす位置（根元の座標 x, y mm）。長い向きに、端から MARGIN あけて等間隔に並べる。"""
    use = max(0.0, box["length"] - 2 * MARGIN)
    t = 0.0 if n <= 1 else -use / 2 + use * i / (n - 1)
    return [box["center"][0] + box["axis"][0] * t, box["center"][1] + box["axis"][1] * t]


def release_h(box, grip):
    """離す時の指先の高さ（机から mm）：物の底（指先より grip 下）が縁の上 DROP_GAP に来る高さ。"""
    return box["rim"] + grip + DROP_GAP


def footprint(box):
    """箱の所の 4 隅（根元の座標 x, y mm）。長い向きに 底の長さの半分 ＋ BOX_END ＋ BOX_PAD、横に 幅の半分 ＋ BOX_PAD。"""
    u = np.array(box["axis"], np.float64)
    v = np.array([-u[1], u[0]])
    hl, hw = box["length"] / 2 + BOX_END + BOX_PAD, BOX_W / 2 + BOX_PAD
    c = np.array(box["center"], np.float64)
    return np.array([c + su * hl * u + sv * hw * v for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def mask_px(box, tcp, shape):
    """箱の所が写る画素（255）。4 隅を机の高さと縁の高さの両方で写真に写し、全部を囲む凸の形を塗る（箱の壁も入る）。"""
    import height_by_motion as HM
    xy = footprint(box)
    pts = []
    for h in (0.0, max(0.0, box["rim"])):
        p0 = HM.L.G0 + HM.GN * h                                   # 机から高さ h の面の上で、x, y が 4 隅になる点
        z = (p0 @ HM.GN - xy @ HM.GN[:2]) / HM.GN[2]
        pts.append(HM.project_fast(np.column_stack([xy, z]), tcp))
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.vstack(pts).astype(np.int32)), 255)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--point", type=int, choices=(1, 2),
                    help="Enter を待たずに、今の指先を 1 点目／2 点目として読む（2 で箱を決める。プログラムから動かす時用、10/19 夜）")
    a = ap.parse_args()
    if a.show or not a.ip:
        print(json.dumps(load(), ensure_ascii=False, indent=1) if load() else "まだ覚えていない")
        return
    from rtde_receive import RTDEReceiveInterface
    from plan_hover import TABLE_Z
    r = RTDEReceiveInterface(a.ip)
    pts = []
    first = os.path.join(HERE, "box_point1.json")
    try:
        names = ("片方の端", "もう片方の端")
        if a.point == 2:
            with open(first, encoding="utf-8") as fh:
                pts.append(tuple(json.load(fh)))
        for k, name in enumerate(names):
            if a.point and k + 1 != a.point:
                continue
            if not a.point:
                input(f"指先を、箱の底の長い向きの{name}の真上・縁の高さに合わせて Enter：")
            p = r.getActualTCPPose()
            pts.append((p[0] * 1000, p[1] * 1000, p[2] * 1000 - TABLE_Z))
            print(f"   {name}：x {pts[-1][0]:.1f}  y {pts[-1][1]:.1f} mm、指先の高さ（机から）{pts[-1][2]:.1f} mm")
    finally:
        r.disconnect()
    if a.point == 1:
        with open(first, "w", encoding="utf-8") as fh:
            json.dump(pts[0], fh)
        print("1 点目を覚えた。もう片方の端に合わせて --point 2")
        return
    (x1, y1, z1), (x2, y2, z2) = pts
    L = math.hypot(x2 - x1, y2 - y1)
    if L < 20:
        raise SystemExit("2 か所が近すぎる（20mm 未満）。長い向きの両端で覚えさせ直す")
    box = {"center": [(x1 + x2) / 2, (y1 + y2) / 2], "axis": [(x2 - x1) / L, (y2 - y1) / L], "length": L,
           "rim": (z1 + z2) / 2, "points": pts, "when": time.strftime("%Y-%m-%d %H:%M")}
    if abs(z1 - z2) > 10:
        print(f"   ！ 2 か所の高さが {abs(z1 - z2):.0f}mm 違う。平均 {box['rim']:.0f}mm を縁の高さにする")
    with open(PATH, "w", encoding="utf-8") as fh:
        json.dump(box, fh, ensure_ascii=False, indent=1)
    print(f"覚えた：真ん中 x {box['center'][0]:.1f}  y {box['center'][1]:.1f} mm、並べられる長さ {L:.0f}mm、縁の高さ {box['rim']:.0f}mm → {PATH}")


if __name__ == "__main__":
    main()
