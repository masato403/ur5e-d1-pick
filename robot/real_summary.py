"""real_suite.sh の結果を、設定ごとにまとめる（robot/logs/test_real.jsonl の最新の結果を使う）。"""
import json
import sys

rows = {}
for l in open("robot/logs/test_real.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if "score" in r:
        rows[(r["tag"], r["image"])] = r
tags = []
for (t, _) in rows:
    if t not in tags:
        tags.append(t)
print("設定 | 正解 | 種類だけ正解 | 見落とし・まとまり | 余計 | 物を見つける時間(中央)")
for t in tags:
    rs = [r for (tt, _), r in rows.items() if tt == t]
    s = {k: sum(r["score"][k] for r in rs) for k in ("correct", "kind_only", "missed", "extra", "n_truth")}
    ms = sorted(r["detect_ms"] for r in rs)[len(rs) // 2]
    print(f"{t} | {s['correct']}/{s['n_truth']} | {s['kind_only']} | {s['missed']} | {s['extra']} | {ms} ms ({len(rs)}枚)")
