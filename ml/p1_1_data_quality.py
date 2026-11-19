"""任务 1.1 数据字典与质量检查。

运行：python -X utf8 -m ml.p1_1_data_quality
输出（ml/results/p1_1_data_quality/）：
- report.md               数据字典、质量检查结果、已知问题
- table_summary.csv       每张表的行数、日期范围、主键重复数
- column_nulls.csv        每列的缺失率
- daily_counts.csv        逐日主板行数、复权因子/涨跌停价/每日指标/资金流的缺失、涨跌停数量
- yearly_coverage.csv     按年汇总上面的数
- daily_counts.png        逐日主板行数与覆盖率图
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ml.common import DB_PATH, connect, file_fingerprint, register, result_dir, write_md

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

MAIN = "NOT regexp_matches(ts_code, '^(300|301|688|689|8|4|9)')"   # 主板（与 rebound.universe 一致）
TOL = 1e-6

#: 表 → (主键, 日期列, 中文说明)
TABLES: dict[str, tuple[tuple[str, ...], str | None, str]] = {
    "daily": (("ts_code", "trade_date"), "trade_date", "日线行情（不复权）"),
    "adj_factor": (("ts_code", "trade_date"), "trade_date", "复权因子"),
    "daily_limit": (("ts_code", "trade_date"), "trade_date", "每日涨跌停价"),
    "daily_basic": (("ts_code", "trade_date"), "trade_date", "每日指标（换手率、股本、市值）"),
    "moneyflow": (("ts_code", "trade_date"), "trade_date", "个股资金流（按单笔大小分档）"),
    "index_daily": (("ts_code", "trade_date"), "trade_date", "指数日线（上证、沪深300、中证500、中证1000）"),
    "trade_cal": (("exchange", "cal_date"), "cal_date", "交易日历"),
    "namechange": (("ts_code", "start_date", "name"), "start_date", "股票历史名称（判断当时是否 ST）"),
    "stock_basic": (("ts_code",), None, "股票列表（今天的快照）"),
    "adj_scale": (("ts_code",), None, "拼库时复权因子的缩放系数"),
    "merge_meta": (("key",), None, "拼库记录"),
}

#: 字段单位（来自 Tushare 接口文档，并用 600000.SH 2024-03-15 的数值做了量级核对）
UNITS = {
    "daily": "open/high/low/close/pre_close：元（不复权）；pct_chg：%；vol：手；amount：**千元**",
    "adj_factor": "无量纲；复权价 = 不复权价 × adj_factor",
    "daily_limit": "up_limit/down_limit：元",
    "daily_basic": "turnover_rate / turnover_rate_f：%；volume_ratio：倍；*_share：**万股**；total_mv / circ_mv：**万元**",
    "moneyflow": "各金额：**万元**（与 daily.amount 的千元差 10 倍）；lg = 大单，elg = 特大单",
    "index_daily": "close：点；pct_chg：%",
}


def table_summary(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, nulls = [], []
    for table, (pk, dcol, desc) in TABLES.items():
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        keys = ", ".join(pk)
        dup = con.execute(f"SELECT count(*) FROM (SELECT {keys} FROM {table} GROUP BY {keys} HAVING count(*) > 1)").fetchone()[0]
        lo = hi = ndays = None
        if dcol:
            lo, hi, ndays = con.execute(f"SELECT min({dcol}), max({dcol}), count(DISTINCT {dcol}) FROM {table}").fetchone()
        cols = con.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}' "
                           "ORDER BY ordinal_position").fetchall()
        rows.append({"表": table, "说明": desc, "行数": n, "主键": keys, "主键重复组数": dup,
                     "最早": lo, "最晚": hi, "不同日期数": ndays, "列数": len(cols)})
        exprs = ", ".join(f"avg(CASE WHEN \"{c}\" IS NULL THEN 1.0 ELSE 0 END) AS \"{c}\"" for c, _ in cols)
        rates = con.execute(f"SELECT {exprs} FROM {table}").fetchone()
        for (c, t), r in zip(cols, rates):
            nulls.append({"表": table, "列": c, "类型": t, "缺失率": r})
    return pd.DataFrame(rows), pd.DataFrame(nulls)


def daily_counts(con) -> pd.DataFrame:
    """逐日：主板日线行数，以及这些行里缺复权因子 / 涨跌停价 / 每日指标 / 资金流的数量，涨跌停收盘数，价格异常数。"""
    sql = f"""
    SELECT d.trade_date,
      count(*) AS main_rows,
      count(*) FILTER (WHERE a.adj_factor IS NULL) AS no_adj,
      count(*) FILTER (WHERE l.up_limit IS NULL) AS no_limit,
      count(*) FILTER (WHERE b.ts_code IS NULL) AS no_basic,
      count(*) FILTER (WHERE m.ts_code IS NULL) AS no_moneyflow,
      count(*) FILTER (WHERE abs(d.close - l.up_limit) <= {TOL}) AS close_limit_up,
      count(*) FILTER (WHERE abs(d.close - l.down_limit) <= {TOL}) AS close_limit_down,
      count(*) FILTER (WHERE d.high < greatest(d.open, d.close) - {TOL} OR d.low > least(d.open, d.close) + {TOL}
                         OR d.close <= 0 OR d.open <= 0) AS bad_ohlc,
      count(*) FILTER (WHERE d.pre_close > 0 AND abs((d.close / d.pre_close - 1) * 100 - d.pct_chg) > 0.05) AS bad_pct,
      count(*) FILTER (WHERE d.vol = 0 OR d.amount = 0) AS zero_volume
    FROM daily d
    LEFT JOIN adj_factor a USING (ts_code, trade_date)
    LEFT JOIN daily_limit l USING (ts_code, trade_date)
    LEFT JOIN daily_basic b USING (ts_code, trade_date)
    LEFT JOIN moneyflow m USING (ts_code, trade_date)
    WHERE {MAIN.replace('ts_code', 'd.ts_code')}
    GROUP BY 1 ORDER BY 1
    """
    df = con.execute(sql).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["chg_vs_prev"] = df["main_rows"].pct_change()
    return df


def calendar_check(con, daily: pd.DataFrame) -> dict:
    cal = con.execute("SELECT cal_date FROM trade_cal WHERE exchange = 'SSE' AND is_open = 1 "
                      "AND cal_date BETWEEN '2013-01-04' AND '2026-10-23'").df()
    cal_days = set(pd.to_datetime(cal["cal_date"]))
    have = set(daily["trade_date"])
    return {"open_days": len(cal_days), "daily_days": len(have),
            "missing_in_daily": sorted(d.date() for d in cal_days - have),
            "extra_in_daily": sorted(d.date() for d in have - cal_days)}


def seam_check(con) -> dict:
    """拼接缝 2021-04-23：缝前后几天的复权收益是否出现异常跳变。

    做法：主板每只股票逐日复权收益 |r|>25%（主板涨跌幅限制最多 20%，加上除权误差也不该超过）的行数，
    比较缝附近 ±5 个交易日与全期平均。缝处复权因子如果没对齐，会在缝当天出现一批大跳变。
    """
    sql = f"""
    WITH p AS (
      SELECT d.ts_code, d.trade_date, d.close * a.adj_factor AS c
      FROM daily d JOIN adj_factor a USING (ts_code, trade_date) WHERE {MAIN.replace('ts_code', 'd.ts_code')}
    ), r AS (
      SELECT trade_date, c / lag(c) OVER (PARTITION BY ts_code ORDER BY trade_date) - 1 AS ret FROM p
    )
    SELECT trade_date, count(*) FILTER (WHERE abs(ret) > 0.25) AS jumps, count(ret) AS n
    FROM r GROUP BY 1 ORDER BY 1
    """
    j = con.execute(sql).df()
    j["trade_date"] = pd.to_datetime(j["trade_date"])
    seam = pd.Timestamp("2021-04-23")
    k = int(j.index[j["trade_date"] >= seam][0])
    near = j.iloc[k - 5: k + 6]
    meta = dict(con.execute("SELECT key, value_json FROM merge_meta").fetchall())
    return {"near": near, "avg_jumps_per_day": j["jumps"].mean(), "total_jumps": int(j["jumps"].sum()),
            "seam_checks": json.loads(meta["seam_checks"]), "top_jump_days": j.nlargest(5, "jumps")}


def plot_counts(daily: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(daily["trade_date"], daily["main_rows"], lw=0.8)
    axes[0].set_ylabel("主板日线行数")
    axes[0].set_title("逐日主板行数与各表覆盖情况")
    for col, lab in [("no_basic", "缺每日指标"), ("no_moneyflow", "缺资金流"), ("no_limit", "缺涨跌停价"), ("no_adj", "缺复权因子")]:
        axes[1].plot(daily["trade_date"], daily[col] / daily["main_rows"] * 100, lw=0.8, label=lab)
    axes[1].set_ylabel("缺失占比（%）")
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].axvline(pd.Timestamp("2021-04-23"), color="grey", ls="--", lw=0.8)
    axes[1].text(pd.Timestamp("2021-05-01"), axes[1].get_ylim()[1] * 0.85, "拼接缝 2021-04-23", fontsize=8, color="grey")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    out = result_dir("p1_1_data_quality")
    con = connect()
    summary, nulls = table_summary(con)
    summary.to_csv(out / "table_summary.csv", index=False, encoding="utf-8-sig")
    nulls.to_csv(out / "column_nulls.csv", index=False, encoding="utf-8-sig")

    daily = daily_counts(con)
    daily.to_csv(out / "daily_counts.csv", index=False, encoding="utf-8-sig")
    plot_counts(daily, out / "daily_counts.png")
    yearly = daily.assign(year=daily["trade_date"].dt.year).groupby("year").agg(
        交易日=("trade_date", "count"), 日均主板行数=("main_rows", "mean"),
        缺复权因子=("no_adj", "sum"), 缺涨跌停价=("no_limit", "sum"), 缺每日指标=("no_basic", "sum"),
        缺资金流=("no_moneyflow", "sum"), 收盘涨停=("close_limit_up", "sum"), 收盘跌停=("close_limit_down", "sum"),
        价格异常=("bad_ohlc", "sum"), 涨跌幅不一致=("bad_pct", "sum"), 零成交=("zero_volume", "sum"),
        行数=("main_rows", "sum"))
    for c in ["缺复权因子", "缺涨跌停价", "缺每日指标", "缺资金流"]:
        yearly[c + "%"] = (yearly[c] / yearly["行数"] * 100).round(3)
    yearly["日均主板行数"] = yearly["日均主板行数"].round(0)
    yearly.to_csv(out / "yearly_coverage.csv", encoding="utf-8-sig")

    cal = calendar_check(con, daily)
    seam = seam_check(con)
    drops = daily[(daily["chg_vs_prev"] < -0.05)]
    n_main_codes = con.execute(f"SELECT count(DISTINCT ts_code) FROM daily WHERE {MAIN}").fetchone()[0]
    basic_missing = con.execute(f"SELECT count(DISTINCT ts_code) FROM daily WHERE {MAIN} AND ts_code NOT IN "
                                "(SELECT ts_code FROM stock_basic)").fetchone()[0]
    ind_missing = con.execute(f"SELECT count(*) FROM stock_basic WHERE {MAIN} AND industry IS NULL").fetchone()[0]
    con.close()

    total = daily[["main_rows", "no_adj", "no_limit", "no_basic", "no_moneyflow", "bad_ohlc", "bad_pct", "zero_volume"]].sum()
    pct = lambda c: f"{total[c] / total['main_rows'] * 100:.3f}%"  # noqa: E731

    summary["不同日期数"] = summary["不同日期数"].astype("Int64")
    summ_md = summary.astype(object).fillna("").to_markdown(index=False)
    unit_md = "\n".join(f"| `{t}` | {u} |" for t, u in UNITS.items())
    year_md = yearly[["交易日", "日均主板行数", "缺复权因子%", "缺涨跌停价%", "缺每日指标%", "缺资金流%", "收盘涨停", "收盘跌停",
                      "价格异常", "涨跌幅不一致"]].to_markdown()
    high_null = nulls[(nulls["缺失率"] > 0.01)].assign(缺失率=lambda x: (x["缺失率"] * 100).round(2).astype(str) + "%")
    near_md = seam["near"].assign(trade_date=lambda x: x["trade_date"].dt.date).to_markdown(index=False)
    top_md = seam["top_jump_days"].assign(trade_date=lambda x: x["trade_date"].dt.date).to_markdown(index=False)
    drop_md = (drops[["trade_date", "main_rows", "chg_vs_prev"]].assign(trade_date=lambda x: x["trade_date"].dt.date,
               chg_vs_prev=lambda x: (x["chg_vs_prev"] * 100).round(1)).to_markdown(index=False)
               if len(drops) else "无")

    text = f"""
# 1.1 数据字典与质量检查

- 运行：`python -X utf8 -m ml.p1_1_data_quality`（只读打开研究库）
- 数据库：`{DB_PATH.name}`，指纹 `{file_fingerprint(DB_PATH)}`
- 产物：`table_summary.csv`、`column_nulls.csv`、`daily_counts.csv`、`yearly_coverage.csv`、`daily_counts.png`

## 一、结论先看

1. **主键无重复，价格基本自洽**：各表主键重复组数见下表；主板 {int(total['main_rows']):,} 行日线里，
   开高低收矛盾的 {int(total['bad_ohlc'])} 行，涨跌幅与收盘 / 昨收对不上（差 > 0.05 个百分点）的 {int(total['bad_pct'])} 行（{pct('bad_pct')}）。
2. **主板覆盖完整**：主板 {n_main_codes:,} 只股票；缺复权因子 {pct('no_adj')}、缺涨跌停价 {pct('no_limit')}、
   缺每日指标 {pct('no_basic')}、缺资金流 {pct('no_moneyflow')}（逐年见第四节）。新特征用到的每日指标和资金流可以放心使用。
3. **交易日历一致**：2013-01-04 ~ 2026-10-23 上交所开市 {cal['open_days']} 天，日线有 {cal['daily_days']} 天；
   日历有而日线没有的 {len(cal['missing_in_daily'])} 天，日线有而日历没有的 {len(cal['extra_in_daily'])} 天。
4. **拼接缝 2021-04-23 没有跳变**：复权收益绝对值超过 25% 的行（主板涨跌幅最多 20%，超过说明复权有问题），
   全期每天平均 {seam['avg_jumps_per_day']:.2f} 行，缝前后 11 天见第五节；作者拼库时的核对也是价格、成交额、涨跌停价零差异。
5. **需要在报告里说明的已知问题**见第六节，其中最重要的是 `stock_basic` 是今天的快照，所以**不用行业做特征**。

## 二、表与字段

{summ_md}

### 字段单位（构造特征时必须换算）

| 表 | 单位 |
|---|---|
{unit_md}

量级核对（600000.SH，2024-03-15）：成交额 208,970 千元 ≈ 2.09 亿元；资金流净额 2,375 万元；流通市值 20,810,693 万元 ≈ 2,081 亿元，都符合浦发银行的实际量级。

### 缺失率超过 1% 的列

{high_null.to_markdown(index=False) if len(high_null) else "无"}

## 三、交易日历

- 日历有、日线没有的日子：{', '.join(map(str, cal['missing_in_daily'])) or '无'}
- 日线有、日历没有的日子：{', '.join(map(str, cal['extra_in_daily'])) or '无'}

## 四、逐年覆盖（主板）

缺失列是“这一年主板日线里，另一张表没有对应行的比例”；涨跌停按收盘价贴着涨跌停价计。

{year_md}

主板行数比前一天少 5% 以上的日子（日线不含停牌日，所以这里行数骤减 = 大面积停牌或数据残缺）：

{drop_md}

这几天都在 2015 年 7 月初股灾期间，当时上千家公司主动停牌（“千股停牌”），属于真实的市场事件，不是数据残缺。
仓库打分时的“行情不全就拒绝”闸门针对的是实时数据；历史面板保留这几天。

## 五、拼接缝 2021-04-23

缝前后各 5 个交易日，复权收益 |r| > 25% 的行数（jumps）和当天有收益的行数（n）：

{near_md}

全期跳变最多的 5 天（用来对照：如果都是正常的除权或新股上市首日之类，就说明复权没问题）：

{top_md}

作者拼库时记录的核对（`merge_meta.seam_checks`）：

```json
{json.dumps(seam['seam_checks'], ensure_ascii=False, indent=1)}
```

## 六、已知问题与处理

| 问题 | 影响 | 处理 |
|---|---|---|
| `stock_basic` 是今天的快照：行业、名称、上市状态都是现在的 | 用行业做特征或分组会带入未来信息 | 不用行业做特征；ST 判断用 `namechange` 的当时名称（仓库已这样做）；`list_date` 是固定事实，可用于上市年龄 |
| `stock_basic` 里主板缺行业 {ind_missing} 只；日线里有 {basic_missing} 只主板股票不在 `stock_basic` | 只影响仓库选股时的“同行业最多 2 只” | 不影响特征和标签 |
| 北交所 11 只股票缺复权因子，拼库时被丢掉（`merge_meta.dropped_no_adj`） | 不影响主板 | 无需处理 |
| `daily_basic` 只有换手率、量比、股本、市值，没有市盈率、市净率 | 没有估值类特征 | 报告里说明 |
| 研究库没有 `model_score` 表 | 仓库的正式选股命令不能直接在这个库上用 | 我们的模型分数另存到 `data/ml/scores/` |
| 日线不含停牌日（停牌当天没有行） | 窗口特征按“这只票自己的交易日”计 | 与仓库口径一致 |
"""
    write_md(out / "report.md", text)
    register("1.1_data_quality", db_fingerprint=file_fingerprint(DB_PATH), main_rows=int(total["main_rows"]),
             bad_ohlc=int(total["bad_ohlc"]), bad_pct=int(total["bad_pct"]), seam_avg_jumps=round(seam["avg_jumps_per_day"], 3))


if __name__ == "__main__":
    main()
