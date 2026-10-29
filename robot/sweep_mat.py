"""マットを見分ける明るさの下限（MAT_V_MIN）と色の濃さの上限（MAT_S_MAX）を何通りか変えて、実機で残した写真を全部測り直し、
種類ごとに「ちょうど 1 つ見つかった組」「2 つ以上の組」と、どの種類にも当てはまらない候補の数を比べる（10/19、暗い時のマットの件）。
候補 ＝ まわりがマットの割合 60% 以上で、高さ 3mm 以上の物（pick_pass が選ぶ物）。
使い方:  python robot/sweep_mat.py [日付 20261016] "90,60" "75,60" ...
"""
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(day, v, s, extra=()):
    env = dict(os.environ, MAT_V_MIN=str(v), MAT_S_MAX=str(s), PYTHONIOENCODING="utf-8")
    env.update(e.split("=", 1) for e in extra)                  # ほかの設定（例 MAT_HULL=1）。10/19
    subprocess.run([PY, os.path.join(HERE, "scan_archive.py"), day], env=env, check=True, capture_output=True)
    rows = json.load(open(os.path.join(HERE, "logs", f"scan_{day}", "scan.json"), encoding="utf-8"))
    os.replace(os.path.join(HERE, "logs", f"scan_{day}", "scan.json"), os.path.join(HERE, "logs", f"scan_{day}", f"scan_v{v}_s{s}{'_' + '_'.join(extra) if extra else ''}.json"))
    by = defaultdict(list)
    for r in rows:
        by[r["t"]].append(r)
    c = Counter()
    for t, rs in by.items():
        cand = [r for r in rs if r["on_mat"] >= 0.6 and r["h"] >= 3]
        ks = Counter(r["kind"] for r in cand)
        for k in ("消しゴム", "ペン", "銀の棒"):
            c[f"{k}1"] += ks[k] == 1
            c[f"{k}2+"] += ks[k] >= 2
        c["？"] += ks[None]
    return len(by), c


def main():
    day = sys.argv[1]
    for vs in sys.argv[2:]:
        v, s, *extra = vs.split(",")                           # "90,60,MAT_HULL=1,MAT_DARK=1" のように、ほかの設定も付けられる
        n, c = run(day, v, s, extra)
        print(f"明るさの下限 {v}・色の濃さの上限 {s}{'・' + '・'.join(extra) if extra else ''}（{n} 組）：消しゴム {c['消しゴム1']}（2 つ以上 {c['消しゴム2+']}）、"
              f"ペン {c['ペン1']}（{c['ペン2+']}）、銀の棒 {c['銀の棒1']}（{c['銀の棒2+']}）、どれでもない候補 {c['？']}", flush=True)


if __name__ == "__main__":
    main()
