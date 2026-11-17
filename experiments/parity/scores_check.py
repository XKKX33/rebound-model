"""正式库的 model_score vs 十年实验的打分（`scores10_{Y,Q,M}.parquet`，限 70 元和不限价）。

每条序列、每个共同的交易日：两边池子差几只；共同的票里，两边分数的秩相关、前 6 / 前 30 重合几只。
两边的模型不会逐位相同（特征的 float32 末位差、正式版修了标签泄漏），这里量差多少。两边都只读。
用法：python -X utf8 -B scores_check.py   输出 scores_check.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORMAL = ROOT / "data/market.duckdb"
FORK2 = ROOT / "experiments/decade"
SOURCES = {"cap70": FORK2, "nocap": FORK2 / "nocap"}
CADENCES = ("Y", "Q", "M")
OUT = HERE / "scores_check.json"
T0 = time.time()

PER_DAY = """
WITH f AS (SELECT trade_date, ts_code, score FROM formal.model_score WHERE variant = $variant AND cadence = $cadence),
r AS (SELECT CAST(d.trade_date AS DATE) AS trade_date, c.ts_code, s.score
      FROM read_parquet($scores) s JOIN axis_dates d USING (di) JOIN axis_codes c USING (ji)),
days AS (SELECT trade_date FROM f INTERSECT SELECT trade_date FROM r),
both_ AS (SELECT coalesce(f.trade_date, r.trade_date) AS trade_date, f.score AS a, r.score AS b,
                 coalesce(f.ts_code, r.ts_code) AS ts_code
          FROM f FULL OUTER JOIN r USING (trade_date, ts_code)
          WHERE coalesce(f.trade_date, r.trade_date) IN (SELECT trade_date FROM days)),
ranked AS (SELECT trade_date, ts_code,
                  rank() OVER (PARTITION BY trade_date ORDER BY a DESC, ts_code) AS ra,
                  rank() OVER (PARTITION BY trade_date ORDER BY b DESC, ts_code) AS rb
           FROM both_ WHERE a IS NOT NULL AND b IS NOT NULL)
SELECT p.trade_date, p.only_formal, p.only_research, q.n, q.rho, q.top6, q.top30
FROM (SELECT trade_date, count(*) FILTER (WHERE b IS NULL) AS only_formal,
             count(*) FILTER (WHERE a IS NULL) AS only_research FROM both_ GROUP BY 1) p
JOIN (SELECT trade_date, count(*) AS n, corr(ra, rb) AS rho,
             count(*) FILTER (WHERE ra <= 6 AND rb <= 6) AS top6,
             count(*) FILTER (WHERE ra <= 30 AND rb <= 30) AS top30 FROM ranked GROUP BY 1) q USING (trade_date)
ORDER BY 1
"""


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [{time.time() - T0:5.0f}s] {msg}", flush=True)


def compare(con: duckdb.DuckDBPyConnection, variant: str, cadence: str) -> dict:
    src = SOURCES[variant]
    axes = json.loads((src / "axes.json").read_text(encoding="utf-8"))
    con.register("axis_dates", pd.DataFrame({"di": range(len(axes["dates"])), "trade_date": pd.to_datetime(axes["dates"])}))
    con.register("axis_codes", pd.DataFrame({"ji": range(len(axes["codes"])), "ts_code": axes["codes"]}))
    day = con.execute(PER_DAY, {"variant": variant, "cadence": cadence,
                                "scores": str(src / f"scores10_{cadence}.parquet")}).df()
    con.unregister("axis_dates")
    con.unregister("axis_codes")
    worst = day.loc[day.rho.idxmin()]
    year = day.assign(year=pd.to_datetime(day.trade_date).dt.year).groupby("year")
    return {
        "days": int(len(day)), "first": str(day.trade_date.min()), "last": str(day.trade_date.max()),
        "pool_days_differ": int(((day.only_formal > 0) | (day.only_research > 0)).sum()),
        "only_formal_rows": int(day.only_formal.sum()), "only_research_rows": int(day.only_research.sum()),
        "rho_mean": round(float(day.rho.mean()), 4), "rho_median": round(float(day.rho.median()), 4),
        "rho_min": round(float(day.rho.min()), 4), "rho_ge_099": round(float((day.rho >= 0.99).mean()), 3),
        "top6_mean": round(float(day.top6.mean()), 2), "top30_mean": round(float(day.top30.mean()), 2),
        "worst_day": {"day": str(worst.trade_date), "rho": round(float(worst.rho), 4), "top30": int(worst.top30)},
        "by_year": {int(y): {"rho": round(float(g.rho.mean()), 4), "top6": round(float(g.top6.mean()), 2),
                             "top30": round(float(g.top30.mean()), 2)} for y, g in year},
    }


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")
    con.execute("SET threads = 8")
    con.execute(f"ATTACH '{FORMAL.as_posix()}' AS formal (READ_ONLY)")
    result = {}
    for variant in SOURCES:
        for cadence in CADENCES:
            result[f"{variant}|{cadence}"] = got = compare(con, variant, cadence)
            say(f"{variant} {cadence}: {got['days']} 天，池子不同 {got['pool_days_differ']} 天，"
                f"秩相关均值 {got['rho_mean']}（最低 {got['rho_min']}），前 6 重合 {got['top6_mean']}，前 30 重合 {got['top30_mean']}")
            OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    say("DONE")


if __name__ == "__main__":
    main()
