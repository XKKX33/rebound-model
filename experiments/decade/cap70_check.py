"""准入池的 70 元单价上限挡掉了多少票、挡掉的票表现如何（只读面板，探索性诊断，不改任何结果）。

"被上限挡掉"：未复权收盘 >= 70，其余准入条件都满足（成交额 >= 0.8 亿、至少 150 根日线；面板本来只有主板）。
ST 没有单独判断（70 元以上的 ST 极少）。收益口径和准入池基准相同（engine.pool_daily_return）：
昨天在组里、今天有收盘的票，等权收盘对收盘，复权价，不扣成本。计分日和 decade_sim 相同。
输出 cap70_check.json。用法：python -X utf8 cap70_check.py
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from common import DATA, PANEL, axes, row_index, signal_range
from decade_sim import YEAR, scoring_days

COLS = ["ts_code", "trade_date", "c", "raw_c", "amount", "bar_no", "eligible", "below2"]
CAP = 70.0
OUT = DATA / "cap70_check.json"


def flag(s) -> np.ndarray:
    return s.fillna(False).to_numpy(dtype=bool)


def load() -> tuple:
    """面板 → 复权收盘、成交额、准入池（去掉 < 2 元）、被上限挡掉的票，都是 日 × 股。"""
    codes, dates = axes()
    shape = (len(dates), len(codes))
    close = np.full(shape, np.nan, dtype=np.float32)
    money = np.zeros(shape, dtype=np.float32)
    pool = np.zeros(shape, dtype=bool)
    capped = np.zeros(shape, dtype=bool)
    for batch in pq.ParquetFile(PANEL).iter_batches(batch_size=500_000, columns=COLS):
        b = batch.to_pandas()
        di, ji = row_index(b.ts_code, b.trade_date, codes, dates)
        close[di, ji] = b.c.to_numpy(dtype=np.float32)
        money[di, ji] = b.amount.fillna(0).to_numpy(dtype=np.float32)
        pool[di, ji] = flag(b.eligible) & ~flag(b.below2)
        capped[di, ji] = flag((b.raw_c >= CAP) & (b.amount >= 80_000) & (b.bar_no >= 149))
    return dates, close, money, pool, capped


def daily(r: np.ndarray, fin: np.ndarray, member: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """昨天在组里、今天有收盘的票：每天的等权收益和只数（第 0 天为空）。"""
    ok = member[:-1] & fin
    n = ok.sum(axis=1)
    ret = np.full(len(member), np.nan)
    ret[1:] = np.where(n > 0, np.where(ok, r, 0).sum(axis=1, dtype=np.float64) / np.maximum(n, 1), np.nan)
    return ret, np.concatenate([[0], n])


def compound(ret: np.ndarray, days: np.ndarray) -> float:
    return float(np.prod(1 + np.nan_to_num(ret[days])) - 1) * 100


def main() -> None:
    dates, close, money, pool, capped = load()
    with np.errstate(invalid="ignore", divide="ignore"):
        r = close[1:] / close[:-1] - 1
    fin = np.isfinite(r)
    del close
    nocap = pool | capped
    rets = {"pool": daily(r, fin, pool), "capped": daily(r, fin, capped), "nocap": daily(r, fin, nocap)}
    amt_capped, amt_all = (money * capped).sum(axis=1), (money * nocap).sum(axis=1)
    idx = scoring_days(dates, *signal_range(dates))
    years = dates[idx].year
    out = {"years": {}}
    for y in np.unique(years):
        d = idx[years == y]
        row = {k: round(compound(v[0], d), 1) for k, v in rets.items()}
        row.update({f"n_{k}": round(float(v[1][d].mean())) for k, v in rets.items() if k != "nocap"})
        # 信号日（昨天）被挡组的成交额 / 准入池 + 被挡组的成交额
        row["capped_amount_pct"] = round(float(amt_capped[d - 1].sum() / amt_all[d - 1].sum()) * 100, 1)
        out["years"][str(y)] = row
    span = len(idx) / YEAR
    out["all_cagr"] = {k: round(((1 + compound(v[0], idx) / 100) ** (1 / span) - 1) * 100, 1) for k, v in rets.items()}
    ref = json.loads((DATA / "sim10_results.json").read_text(encoding="utf-8"))["context"]["pool"]["years"]
    out["pool_check_max_diff"] = round(max(abs(compound(rets["pool"][0], idx[years == int(y)]) - v["total"])
                                           for y, v in ref.items()), 3)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("年份 | 准入池 | 被挡组 | 不设上限 | 每天只数 池/被挡 | 被挡组成交额占比")
    for y, v in out["years"].items():
        print(f"{y} {v['pool']:+6.1f} {v['capped']:+6.1f} {v['nocap']:+6.1f} | {v['n_pool']}/{v['n_capped']} | "
              f"{v['capped_amount_pct']:.1f}%")
    print("全期年化", out["all_cagr"], "| 和 sim10 准入池逐年最大差", out["pool_check_max_diff"])


if __name__ == "__main__":
    main()
