"""任务 1.2 构建主面板：28 个原特征 + o1o5 收益标签，只留不限价（nocap）准入池的行。

直接复用仓库的 `rebound.model.train.load_rows`，口径和作者的正式程序完全一致：
- 特征：`rebound.features`（按年分批算，主板、有复权因子的行）；
- 准入池：主板、按当时名称排除 ST 和退市整理、成交额 ≥ 0.8 亿、收盘 ≥ 2 元、至少 150 根日线；
- 标签：T+1 开盘买、T+5 开盘卖的复权收益；买不进记空，卖不出最多顺延 20 天。

运行：python -X utf8 -m ml.p1_2_panel
输出：
- data/ml/panel_nocap.parquet                  主面板（不进 Git）
- ml/results/p1_2_panel/report.md              构建记录与核对结果
- ml/results/p1_2_panel/yearly_rows.csv        逐年行数与标签有效率
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

from ml.common import DB_PATH, connect, data_dir, file_fingerprint, register, result_dir, rss_gb, write_md
from rebound import params as model_params
from rebound.features import FEATURES, eligible, feature_panel, fingerprint
from rebound.model.train import load_rows, open_store

PANEL = "panel_nocap.parquet"
VARIANT = "nocap"
CHECK_DAYS = (date(2016, 3, 15), date(2021, 4, 23), date(2025, 9, 10))   # 核对用：早期、拼接缝当天、近期


def build() -> tuple[pd.DataFrame, dict]:
    log: list[str] = []
    t0 = time.time()
    rows = load_rows(DB_PATH, log=lambda line: (print(line, flush=True), log.append(line)))
    secs = time.time() - t0
    keep = rows.member[VARIANT]
    dates = rows.dates
    di = rows.di[keep]
    exit_day = rows.exit_day[keep]
    panel = pd.DataFrame(rows.x[keep], columns=list(FEATURES))
    panel.insert(0, "trade_date", dates[di])
    panel.insert(1, "ts_code", pd.Categorical(rows.codes[rows.ji[keep]]))
    panel.insert(2, "di", di.astype(np.int32))
    panel["ret_o1o5"] = rows.ret[keep].astype(np.float64)
    panel["exit_di"] = exit_day.astype(np.int32)                       # -1 = 没有有效收益
    panel["exit_date"] = pd.NaT
    ok = exit_day >= 0
    panel.loc[ok, "exit_date"] = dates[exit_day[ok]]
    panel["in_cap70"] = rows.member["cap70"][keep]
    # 评估用标签：当天准入池内收益的百分位（全部已知，用于看结果；训练用的标签另按训练截止日重排，见 ml/evaluate.py 的 train_labels）
    panel["label_pct"] = panel.groupby("di")["ret_o1o5"].rank(pct=True)
    # 评估用超额收益：减去当天准入池的平均收益
    panel["excess_o1o5"] = panel["ret_o1o5"] - panel.groupby("di")["ret_o1o5"].transform("mean")
    info = {"seconds": round(secs, 1), "mem_gb": round(rss_gb(), 2), "log": log, "n_dates": len(dates), "first_date": dates[0].date(), "last_date": dates[-1].date()}
    del rows
    return panel, info


def check_features(panel: pd.DataFrame) -> pd.DataFrame:
    """抽 3 个交易日，用 feature_panel 单独重算这一天，和面板逐格比较。"""
    out = []
    with open_store(DB_PATH) as store:
        for day in CHECK_DAYS:
            one = feature_panel(store, day, day)
            mask = eligible(one, model_params.VARIANTS[VARIANT], min_bars=model_params.MIN_BARS)
            one = one[mask].set_index(one[mask]["ts_code"].astype(str))
            mine = panel[panel["trade_date"] == pd.Timestamp(day)]
            mine = mine.set_index(mine["ts_code"].astype(str))
            same_pool = set(one.index) == set(mine.index)
            common = one.index.intersection(mine.index)
            a = one.loc[common, list(FEATURES)].to_numpy(np.float64)
            b = mine.loc[common, list(FEATURES)].to_numpy(np.float64)
            both_nan = np.isnan(a) & np.isnan(b)
            diff = np.where(both_nan, 0.0, np.abs(a - b))
            nan_mismatch = int((np.isnan(a) ^ np.isnan(b)).sum())
            scale = np.nanmax(np.abs(b), axis=0)
            worst = np.nanmax(diff, axis=0)
            rel = worst / np.where(scale > 0, scale, 1)
            flips = int(((diff > 0.5) & ~both_nan).sum())          # 0/1 特征翻转（ma20_gt_ma60 的浮点并列）
            out.append({"日期": day, "单日重算行数": len(one), "面板行数": len(mine), "准入池一致": same_pool,
                        "最大绝对差": float(np.nanmax(np.where(diff > 0.5, 0, diff))),
                        "最大相对差": float(np.nanmax(rel[worst < 0.5])) if (worst < 0.5).any() else 0.0,
                        "空值不一致格数": nan_mismatch, "0/1 翻转格数": flips})
    return pd.DataFrame(out)


def check_labels(panel: pd.DataFrame, n: int = 2000, seed: int = 1) -> dict:
    """抽 n 行“没有顺延”的有效样本，用 SQL 直接算 T+1 开盘到 T+5 开盘的复权收益，和面板对照。"""
    rng = np.random.default_rng(seed)
    valid = panel[(panel["exit_di"] == panel["di"] + 5)]
    sample = valid.iloc[rng.choice(len(valid), size=min(n, len(valid)), replace=False)]
    con = connect()
    sessions = pd.DatetimeIndex(pd.to_datetime(con.execute(
        "SELECT DISTINCT trade_date FROM daily ORDER BY 1").df()["trade_date"]))
    q = pd.DataFrame({"ts_code": sample["ts_code"].astype(str).to_numpy(),
                      "buy": sessions[sample["di"].to_numpy() + 1].date,
                      "sell": sessions[sample["di"].to_numpy() + 5].date})
    con.register("q", q)
    got = con.execute("""
        SELECT q.ts_code, q.buy, (s.open * sa.adj_factor) / (b.open * ba.adj_factor) - 1 AS ret
        FROM q
        JOIN daily b ON b.ts_code = q.ts_code AND b.trade_date = q.buy
        JOIN adj_factor ba ON ba.ts_code = q.ts_code AND ba.trade_date = q.buy
        JOIN daily s ON s.ts_code = q.ts_code AND s.trade_date = q.sell
        JOIN adj_factor sa ON sa.ts_code = q.ts_code AND sa.trade_date = q.sell
    """).df()
    con.close()
    got["buy"] = pd.to_datetime(got["buy"]).dt.date
    merged = q.assign(panel_ret=sample["ret_o1o5"].to_numpy()).merge(got, on=["ts_code", "buy"], how="left")
    diff = (merged["panel_ret"] - merged["ret"]).abs()
    return {"sampled": len(q), "matched": int(merged["ret"].notna().sum()), "max_abs_diff": float(diff.max())}


def main() -> None:
    """默认：面板已存在就直接读（只重出报告和核对）；加 --rebuild 才重新构建。"""
    out = result_dir("p1_2_panel")
    path = data_dir() / PANEL
    meta_path = data_dir() / "panel_nocap.meta.json"
    if path.exists() and meta_path.exists() and "--rebuild" not in sys.argv:
        panel = pd.read_parquet(path)
        info = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"读取已有面板 {path.name}（重建请加 --rebuild）")
    else:
        panel, info = build()
        panel.to_parquet(path, index=False)
        meta_path.write_text(json.dumps(info, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
        print(f"写出 {path}（{path.stat().st_size / 1024**2:.0f} MB）")

    yearly = panel.assign(year=panel["trade_date"].dt.year).groupby("year").agg(
        行数=("ret_o1o5", "size"), 交易日=("di", "nunique"), 有效标签=("ret_o1o5", "count"),
        o1o5收益均值=("ret_o1o5", "mean"), o1o5收益中位数=("ret_o1o5", "median"))
    yearly["日均准入股票数"] = (yearly["行数"] / yearly["交易日"]).round(0)
    yearly["有效标签%"] = (yearly["有效标签"] / yearly["行数"] * 100).round(2)
    for c in ["o1o5收益均值", "o1o5收益中位数"]:
        yearly[c] = (yearly[c] * 100).round(3)
    yearly.to_csv(out / "yearly_rows.csv", encoding="utf-8-sig")

    fcheck = check_features(panel)
    lcheck = check_labels(panel)
    carried = int(((panel["exit_di"] > panel["di"] + 5)).sum())
    invalid = int(panel["ret_o1o5"].isna().sum())
    missing = panel[list(FEATURES)].isna().mean().mul(100).round(2)
    miss_md = missing[missing > 0].sort_values(ascending=False).to_frame("缺失%").to_markdown()

    text = f"""
# 1.2 主面板

- 运行：`python -X utf8 -m ml.p1_2_panel`
- 产物：`data/ml/{PANEL}`（{path.stat().st_size / 1024**2:.0f} MB，不进 Git）、`yearly_rows.csv`
- 代码口径：直接调用仓库 `rebound.model.train.load_rows`；特征指纹 `{fingerprint()}`；数据库指纹 `{file_fingerprint(DB_PATH)}`
- 耗时 {info['seconds']:.0f} 秒，进程内存 {info['mem_gb']} GB

## 一、结果

| 项目 | 数值 |
|---|---|
| 交易日 | {info['n_dates']}（{info['first_date']} ~ {info['last_date']}） |
| **不限价准入池（nocap）行数** | **{len(panel):,}**，涉及 {panel['ts_code'].nunique():,} 只股票 |
| 其中也在限 70 元池里的 | {int(panel['in_cap70'].sum()):,}（限 70 元池是不限价池的子集） |
| 有效标签（能买进、能卖出） | {len(panel) - invalid:,}（{(1 - invalid / len(panel)) * 100:.2f}%） |
| 其中卖出日顺延过的 | {carried:,} |
| 没有有效收益的 | {invalid:,}（T+1 一字涨停或停牌买不进、20 天内卖不出，或数据到头） |

- 最后 5 个交易日的信号还没有 T+5 的价格，所以没有标签，这是正常的（2026 年有效率略低就是这个原因）。
- 2013 年只有 {int(panel.loc[panel['trade_date'].dt.year == 2013, 'di'].nunique())} 个交易日有样本：准入池要求库里至少 150 根日线，库从 2013-01-04 开始，所以要到 2013 年 8 月才有股票进池。

## 二、面板的列

| 列 | 含义 |
|---|---|
| `trade_date`、`ts_code` | 信号日、股票代码 |
| `di` | 信号日在全部交易日里的下标（从 0 起），用于按“第几个交易日”切分和对齐 |
| 28 个特征 | `{'`、`'.join(FEATURES)}` |
| `ret_o1o5` | T+1 开盘买、T+5 开盘卖的复权收益（卖不出时顺延）；空 = 无效 |
| `exit_di`、`exit_date` | 实际卖出日的下标和日期（-1 / 空 = 无效） |
| `in_cap70` | 是否也在限 70 元版的池里 |
| `label_pct` | 当天池内收益的百分位（**只用于评估**；训练标签另按训练截止日重排，防泄漏） |
| `excess_o1o5` | 收益减去当天池子的平均收益（评估 Top-K 超额用） |

## 三、逐年

{yearly[['交易日', '日均准入股票数', '行数', '有效标签%', 'o1o5收益均值', 'o1o5收益中位数']].to_markdown()}

（收益单位：%。几乎每年中位数都低于均值，说明 5 日收益分布右偏：少数大涨的股票拉高了均值，典型的一只股票其实是小亏。
各年均值有正有负，2018、2026 年最差，2015、2014、2025 年最好，和市场的牛熊对应。）

## 四、核对

### 特征：抽 3 天单独重算，和面板逐格比较

{fcheck.to_markdown(index=False)}

- 仓库文档说明过：按年分段算和单日算的起点不同，DuckDB 窗口求和顺序、MACD 指数均线的起点不同，会差在 float32 的末位；
  `ma20_gt_ma60` 在两条均线数学上相等时可能翻转。所以判断标准是“差在 1e-5 量级以内、准入池完全一致”。

### 标签：抽 {lcheck['sampled']} 个没有顺延的样本，直接用 SQL 算 T+1 开盘 → T+5 开盘的复权收益

- 对上 {lcheck['matched']} / {lcheck['sampled']} 行，最大绝对差 {lcheck['max_abs_diff']:.2e}。

### 特征缺失率（%）

{miss_md if missing.gt(0).any() else "28 个特征在准入池里**没有缺失**：准入池要求至少 150 根日线，而最长的窗口是 120 根（`r120`）、MACD 预热 33 根，进池时所有窗口都已填满。"}
"""
    write_md(out / "report.md", text)
    register("1.2_panel", rows=len(panel), stocks=panel["ts_code"].nunique(), valid_labels=len(panel) - invalid,
             seconds=info["seconds"], feature_fingerprint=fingerprint(), db_fingerprint=file_fingerprint(DB_PATH),
             feature_check=fcheck.to_dict("records"), label_check=lcheck)


if __name__ == "__main__":
    main()
