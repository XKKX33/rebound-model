"""正式库的本金测算（正式模型的分数） vs 十年实验（它自己的分数），同一本金、同一准入池版本，逐条规则比主要数字。

两边不会逐位相同：分数来自两套模型（特征 float32 末位差、正式版修了标签泄漏），计分区间也不一样
（十年实验到 2026-10-19、少最后 5 天；正式版到最后一个能成交的日子），十年实验还多一种重训频率（每半年）。
用法：python -X utf8 -B compare_capital.py <正式版 --json 输出> <本金> <nocap|cap70>   输出 compare_capital_<本金>_<版本>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORK2 = HERE.parents[0] / "decade"
RULES = ["DB-K3-b16", "DB-K2-b6", "DB-K2-b4", "DB-K3-b4", "DB-K10-b12", "DB-K30-b8", "DB-K30-b4", "DB-K1-b12",
         "FH-K6-next-o1o5", "FH-K2-next-o1o5"]
FIELDS = ("cagr", "excess", "max_dd", "final")


def full_name(short: str) -> str:
    return short if short.startswith("FH") else f"{short}-o1o5"


def main() -> None:
    ours_path, capital, variant = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    ours = json.loads(ours_path.read_text(encoding="utf-8"))
    src = FORK2 / ("nocap" if variant == "nocap" else "") / "rules" / str(capital)
    theirs = {c: json.loads((src / f"real_{c}.json").read_text(encoding="utf-8")) for c in ("Y", "Q", "M")}
    fresh_file = FORK2 / ("nocap/fresh_nocap_20000.json" if variant == "nocap" else "fresh_20000_all.json")
    their_fresh = json.loads(fresh_file.read_text(encoding="utf-8")) if capital == 20000 else {}
    rows = {(r["cadence"], r["rule"]): r for r in ours["rows"]}
    out = {"ours": {"first": ours["first"], "last": ours["last"], "pool": ours["pool"], "best_picker": ours["best_picker"],
                    "walk_forward": {p: {k: v for k, v in w.items() if k != "picks"} for p, w in ours["walk_forward"].items()},
                    "picks_by_year": ours["walk_forward"][ours["best_picker"]]["picks"],
                    "recommended": _brief(ours["rows"][ours["recommended"]]) if ours["recommended"] is not None else None,
                    "picker_choice": _brief(ours["rows"][ours["picker_choice"]]),
                    "steadiest": _brief(ours["rows"][ours["steadiest"]]) if ours["steadiest"] is not None else None,
                    "robust_count": ours["robust_count"]},
           "rules": {}}
    for cadence in ("M", "Q"):
        for short in RULES:
            name = full_name(short)
            mine, theirs_row = rows.get((cadence, name)), theirs[cadence].get(name)
            if mine is None or theirs_row is None:
                continue
            item = {"ours": {f: mine[f] for f in FIELDS}, "fork2": {f: theirs_row[f] for f in FIELDS}}
            if cadence == "M" and mine["fresh"]:
                item["ours_fresh"] = {y: v["cagr"] for y, v in mine["fresh"]["starts"].items()}
                got = {k.split("|")[4]: v["cagr"] for k, v in their_fresh.items()
                       if k.split("|")[1] == "M" and k.split("|")[2] == name and k.split("|")[3] == "min5"}
                if got:
                    item["fork2_fresh"] = got
            out["rules"][f"{cadence}|{short}"] = item
    target = HERE / f"compare_capital_{capital}_{variant}.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["ours"], ensure_ascii=False, indent=1)[:3000])
    for key, item in out["rules"].items():
        print(key, "ours", item["ours"], "fork2", item["fork2"], "| fresh", item.get("ours_fresh"), item.get("fork2_fresh"))


def _brief(row: dict) -> dict:
    return {k: row[k] for k in ("cadence", "rule", "plain", "cagr", "excess", "excess_ci", "max_dd", "final", "ir",
                                "years_positive", "worst_year")} | {"fresh": row["fresh"]}


if __name__ == "__main__":
    main()
