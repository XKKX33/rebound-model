"""阶段 2 核对：正式版逐日打分（正式库 model_score）和研究版打分（scores_<格子>_floor2.parquet）。

判定（计划）：每天全准入池两边打分的相关系数 ≥ 0.99。另外如实报告每天前 6 只重合几只
（直接取前 6，和每行业最多 2 只去重后的前 6）。
两边的区别：正式版修了标签泄漏（只在段起点前已卖出的行里排百分位），线程数固定为 24。
用法：python -X utf8 phase2_check.py   （只读两个库；结果写 phase2_check.json）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAB = HERE.parent / "shared"
sys.path[:0] = [str(LAB), str(ROOT)]

import engine as rs  # noqa: E402

CELLS = ("o1o5", "o1c5", "c1o5")


def research_scores(m: rs.Market, cell: str) -> pd.DataFrame:
    s = pd.read_parquet(LAB / f"scores_{cell}_floor2.parquet")
    return pd.DataFrame({"trade_date": m.dates[s.di.to_numpy()], "ts_code": m.codes[s.ji.to_numpy()],
                         "score_r": s.score.to_numpy(np.float64)})


def formal_scores(con: duckdb.DuckDBPyConnection, cell: str) -> pd.DataFrame:
    f = con.execute("SELECT trade_date, ts_code, score FROM model_score WHERE cell = ?", [cell]).df()
    return pd.DataFrame({"trade_date": pd.to_datetime(f.trade_date).astype("datetime64[ns]"), "ts_code": f.ts_code,
                         "score_f": f.score.to_numpy(np.float64)})


def top6(day: pd.DataFrame, col: str, industry: dict[str, str], dedup: bool) -> set[str]:
    order = day.sort_values([col, "ts_code"], ascending=[False, True], kind="mergesort").ts_code.to_numpy()
    if not dedup:
        return set(order[:6])
    picks, count = [], {}
    for code in order:
        g = industry.get(code, "")
        if count.get(g, 0) >= 2:
            continue
        picks.append(code)
        count[g] = count.get(g, 0) + 1
        if len(picks) == 6:
            break
    return set(picks)


def main() -> None:
    m = rs.load_market()
    con = duckdb.connect(str(ROOT / "data/market.duckdb"), read_only=True)
    industry = dict(con.execute("SELECT ts_code, coalesce(industry, '') FROM stock_basic").fetchall())
    report = {}
    for cell in CELLS:
        r, f = research_scores(m, cell), formal_scores(con, cell)
        both = r.merge(f, on=["trade_date", "ts_code"], how="inner")
        days = sorted(set(r.trade_date) & set(f.trade_date))
        per_day = []
        for day, g in both.groupby("trade_date", sort=True):
            per_day.append({
                "day": day, "n": len(g), "pearson": g.score_r.corr(g.score_f),
                "spearman": g.score_r.rank().corr(g.score_f.rank()),
                "top6": len(top6(g, "score_r", industry, False) & top6(g, "score_f", industry, False)),
                "top6_dedup": len(top6(g, "score_r", industry, True) & top6(g, "score_f", industry, True)),
            })
        d = pd.DataFrame(per_day)
        on_common = r[r.trade_date.isin(days)]
        report[cell] = {
            "days_common": len(days), "rows_research": len(on_common), "rows_formal": int(f.trade_date.isin(days).sum()),
            "rows_matched": len(both),
            "pearson": {"min": d.pearson.min(), "p5": d.pearson.quantile(0.05), "median": d.pearson.median(),
                        "share_ge_099": float((d.pearson >= 0.99).mean())},
            "spearman_median": d.spearman.median(),
            "top6_mean": d.top6.mean(), "top6_dedup_mean": d.top6_dedup.mean(),
            "top6_dedup_hist": d.top6_dedup.value_counts().sort_index().to_dict(),
            "worst_days": d.nsmallest(5, "pearson")[["day", "pearson", "n"]].astype(str).to_dict("records"),
        }
        x = report[cell]
        print(f"{cell}: {x['days_common']} 天，相关系数中位数 {x['pearson']['median']:.4f}、最低 {x['pearson']['min']:.4f}、"
              f"≥0.99 的天数占 {x['pearson']['share_ge_099']:.1%}；前 6 重合平均 {x['top6_mean']:.2f}、"
              f"去重后 {x['top6_dedup_mean']:.2f}", flush=True)
    con.close()
    (HERE / "phase2_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                                            encoding="utf-8")


if __name__ == "__main__":
    main()
