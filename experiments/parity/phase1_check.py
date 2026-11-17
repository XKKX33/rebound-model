"""阶段 1 核对：正式版的特征、准入池、标签，和研究版逐格比（抽 30 个信号日）。

研究版：`shared/ 的面板（panel.parquet）`（特征、准入池）+ 它的 `engine.trades`（carry 口径的逐笔收益）+
`floor.below_floor`（2 元下限）。正式版：`rebound.features` / `model_labels`。
正式版各跑两次：读研究库（同一份数据，比的是代码），读正式库（2026-10-31 从研究库导入，比的是数据）。
判定：特征误差 ≤ 1e-6 且空值位置一致；准入池、标签（收益、状态、卖出日、百分位）完全一致。
用法：python -X utf8 phase1_check.py   （只读两个库；结果写 phase1_check.json）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAB = HERE.parent / "shared"
sys.path[:0] = [str(LAB), str(ROOT)]

import engine as rs  # noqa: E402  研究版
from floor import below_floor  # noqa: E402

from rebound.config import Universe  # noqa: E402
from rebound.features import FEATURES, eligible, feature_panel  # noqa: E402
from rebound.labels import CELLS, STATUS, carry_trades, percentile, price_matrix  # noqa: E402
from rebound.store import Store  # noqa: E402

UNIVERSE = Universe(board="mainboard", exclude_st=True, price_max=70.0, min_amount_yi=0.8, price_min=2.0)
N_DAYS, SEED, FLOOR, MIN_BARS = 30, 20261031, 2.0, 150
DBS = {"research": ROOT / "data/research.duckdb", "formal": ROOT / "data/market.duckdb"}
KEY = ["ts_code", "trade_date"]


def research_side() -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    m = rs.load_market()
    first = int(np.flatnonzero(m.eligible.any(axis=1))[0])
    pick = np.sort(np.random.default_rng(SEED).choice(np.arange(first, len(m.dates) - 30), N_DAYS, replace=False))
    days = [m.dates[i] for i in pick]
    cols = ["ts_code", "trade_date", "bar_no", "eligible", "up_open", "up_close", "dn_open", "dn_close", *FEATURES]
    panel = pd.read_parquet(rs.PANEL, columns=cols, filters=[("trade_date", "in", days)])
    panel["ts_code"] = panel["ts_code"].astype(str)
    ji = m.code_index(panel.ts_code)
    di = m.date_index(panel.trade_date.astype("datetime64[ns]"))
    panel["eligible"] = panel.eligible.to_numpy() & ~below_floor(m, FLOOR)[di, ji]
    labels = []
    rows = panel[panel.eligible]
    rdi, rji = m.date_index(rows.trade_date.astype("datetime64[ns]")), m.code_index(rows.ts_code)
    for cell in CELLS:
        got = rs.trades(m, rdi, rji, rs.Cell(cell.entry, cell.x, cell.exit), "carry")
        exit_date = np.where(got["exit_day"] >= 0, m.dates.values[np.clip(got["exit_day"], 0, None)], np.datetime64("NaT"))
        labels.append(pd.DataFrame({"ts_code": rows.ts_code.to_numpy(), "trade_date": rows.trade_date.to_numpy(),
                                    "cell": cell.name, "ret": got["ret"], "exit_date": exit_date,
                                    "status": np.array(rs.STATUS)[got["status"]],  # 两边编号不同，比名称
                                    "pct": pd.Series(got["ret"]).groupby(rdi).rank(pct=True).to_numpy()}))
    return panel, pd.concat(labels, ignore_index=True), days


def formal_side(db: Path, days: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    with Store(db) as store:
        store.connection.execute("SET memory_limit = '900MB'; SET threads = 4")
        feats = pd.concat([feature_panel(store, d.date(), d.date()) for d in days], ignore_index=True)
        sessions = list(store.sessions(days[-1].date() + pd.Timedelta(days=60), 100_000))
        first = store.connection.execute("SELECT min(trade_date) FROM daily").fetchone()[0]
        slim, edges = [], [pd.Timestamp(first)] + [pd.Timestamp(f"{y}-01-01") for y in range(first.year + 1, 2028)]
        for lo, hi in zip(edges[:-1], edges[1:]):
            part = feature_panel(store, lo.date(), (hi - pd.Timedelta(days=1)).date())
            slim.append(part[["ts_code", "trade_date", "o", "c", "up_open", "up_close", "dn_open", "dn_close"]]
                        .assign(ts_code=lambda f: f.ts_code.astype(str)))
            del part
    feats["ts_code"] = feats["ts_code"].astype(str)
    feats["eligible"] = eligible(feats.assign(ts_code=feats.ts_code.astype("category")), UNIVERSE, min_bars=MIN_BARS)
    fm = price_matrix(pd.concat(slim, ignore_index=True), sessions)
    rows = feats[feats.eligible]
    di = fm.dates.get_indexer(pd.DatetimeIndex(rows.trade_date))
    ji = pd.Index(fm.codes).get_indexer(rows.ts_code)
    labels = []
    for cell in CELLS:
        got = carry_trades(fm, di, ji, cell)
        exit_date = np.where(got["exit_day"] >= 0, fm.dates.values[np.clip(got["exit_day"], 0, None)], np.datetime64("NaT"))
        labels.append(pd.DataFrame({"ts_code": rows.ts_code.to_numpy(), "trade_date": rows.trade_date.to_numpy(),
                                    "cell": cell.name, "ret": got["ret"], "exit_date": exit_date,
                                    "status": np.array(STATUS)[got["status"]], "pct": percentile(got["ret"], di)}))
    return feats, pd.concat(labels, ignore_index=True)


def _close(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    """两边都有值的格子里的最大绝对误差，和空值位置不一致的格数。"""
    x, y = a.to_numpy(np.float64), b.to_numpy(np.float64)
    both = ~np.isnan(x) & ~np.isnan(y)
    return (float(np.max(np.abs(x[both] - y[both]))) if both.any() else 0.0), int((np.isnan(x) != np.isnan(y)).sum())


def _same(a: pd.Series, b: pd.Series) -> int:
    """不相等的格数（两边都空算相等）。"""
    return int((~((a == b) | (a.isna() & b.isna()))).sum())


def compare(research: pd.DataFrame, r_labels: pd.DataFrame, feats: pd.DataFrame, labels: pd.DataFrame) -> dict:
    both = research.merge(feats, on=KEY, how="outer", suffixes=("_r", "_f"), indicator=True)
    out = {"research_only": int((both._merge == "left_only").sum()), "formal_only": int((both._merge == "right_only").sum())}
    both = both[both._merge == "both"]
    elig_diff = both[both.eligible_r.astype(bool) != both.eligible_f.astype(bool)]
    out |= {"rows": len(both), "bar_no_diff": int((both.bar_no_r != both.bar_no_f).sum()),
            "flag_diff": {f: _same(both[f"{f}_r"].astype(bool), both[f"{f}_f"].astype(bool))
                          for f in ("up_open", "up_close", "dn_open", "dn_close")},
            "eligible_rows": int(both.eligible_r.sum()), "eligible_diff": len(elig_diff),
            "eligible_diff_codes": sorted(set(elig_diff.ts_code))[:20],
            "eligible_diff_names": sorted(set(elig_diff.name.astype(str)))[:20]}
    errors = {f: _close(both[f"{f}_r"], both[f"{f}_f"]) for f in FEATURES}
    out["features"] = {f: {"max_abs": e, "nan_diff": n} for f, (e, n) in errors.items()}
    out["features_ok"] = all(e <= 1e-6 and n == 0 for e, n in errors.values())
    out["features_max_abs"] = max(e for e, _ in errors.values())
    lab = r_labels.merge(labels, on=[*KEY, "cell"], how="outer", suffixes=("_r", "_f"), indicator=True)
    out["labels"] = {"research_only": int((lab._merge == "left_only").sum()),
                     "formal_only": int((lab._merge == "right_only").sum())}
    lab = lab[lab._merge == "both"]
    for cell in CELLS:
        g = lab[lab.cell == cell.name]
        (ret_err, ret_nan), (pct_err, pct_nan) = _close(g.ret_r, g.ret_f), _close(g.pct_r, g.pct_f)
        out["labels"][cell.name] = {"rows": len(g), "status_diff": int((g.status_r != g.status_f).sum()),
                                    "exit_diff": _same(g.exit_date_r, g.exit_date_f), "ret_max_abs": ret_err,
                                    "ret_nan_diff": ret_nan, "pct_max_abs": pct_err, "pct_nan_diff": pct_nan}
    cells = [out["labels"][c.name] for c in CELLS]
    out["labels_ok"] = (out["labels"]["research_only"] == out["labels"]["formal_only"] == 0
                        and all(c["status_diff"] == c["exit_diff"] == c["ret_nan_diff"] == c["pct_nan_diff"] == 0
                                and c["ret_max_abs"] == 0.0 and c["pct_max_abs"] == 0.0 for c in cells))
    return out


def _ns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(trade_date=pd.to_datetime(frame.trade_date).astype("datetime64[ns]"))


def main() -> None:
    t0 = time.time()
    research, r_labels, days = research_side()
    research, r_labels = _ns(research), _ns(r_labels)
    print(f"研究版：{len(research)} 行特征、{len(r_labels)} 行标签（{time.time() - t0:.0f}s）", flush=True)
    report = {"days": [d.date().isoformat() for d in days], "seed": SEED}
    for name, db in DBS.items():
        feats, labels = formal_side(db, days)
        report[name] = compare(research, r_labels, _ns(feats), _ns(labels))
        r = report[name]
        print(f"正式版读{name}：{r['rows']} 行；特征 ok={r['features_ok']}（最大误差 {r['features_max_abs']:.2e}），"
              f"准入池不一致 {r['eligible_diff']} 行，标签 ok={r['labels_ok']}（{time.time() - t0:.0f}s）", flush=True)
    (HERE / "phase1_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()