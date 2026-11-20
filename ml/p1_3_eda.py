"""任务 1.3 探索性分析（EDA）。

运行：python -X utf8 -m ml.p1_3_eda
输入：data/ml/panel_nocap.parquet（1.2 的产物）、研究库的 index_daily
输出（ml/results/p1_3_eda/）：
- report.md                 结论与图表说明
- fig1_pool_size.png        准入池每天的股票数
- fig2_pool_vs_index.png    准入池等权净值 vs 四个指数（对数坐标）
- fig3_label_dist.png       o1o5 收益分布、逐年箱线图、每天截面离散度
- fig4_feature_dist.png     28 个特征的分布（按 1%~99% 截尾画）
- fig5_feature_corr.png     28 个特征的 Spearman 相关矩阵
- feature_stats.csv         特征的描述统计
- high_corr_pairs.csv       |相关系数| > 0.8 的特征对
- yearly_returns.csv        准入池与指数的逐年收益
另存 data/ml/pool_daily_return.parquet（准入池等权日收益，后面评估、回测都要用）
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ml.common import DB_PATH, connect, data_dir, register, result_dir, write_md
from rebound.backtest.market import load_markets, pool_daily_return
from rebound.features import FEATURES

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

INDEX_NAMES = {"000001.SH": "上证指数", "000300.SH": "沪深300", "000905.SH": "中证500", "000852.SH": "中证1000"}
FEATURE_CN = {
    "pct_chg": "当日涨跌幅", "r1": "1日收益", "r5": "5日收益", "r10": "10日收益", "r20": "20日收益", "r60": "60日收益",
    "r120": "120日收益", "from_low60": "距60日低点", "from_high60": "距60日高点", "dev_ma5": "偏离5日均线",
    "dev_ma20": "偏离20日均线", "dev_ma60": "偏离60日均线", "ma20_gt_ma60": "20日线在60日线上", "ma20_slope5": "20日线5日斜率",
    "ma60_slope10": "60日线10日斜率", "dd20": "20日回撤", "vol5": "成交额/前5日均", "vol20": "成交额/前20日均",
    "amp20": "20日平均振幅", "launch10": "10日内大涨天数", "up_days10": "10日内上涨天数", "max_up20": "20日最大涨幅",
    "limit_up_today": "当日涨停", "limit_up20": "20日涨停次数", "dif_ratio": "MACD DIF/价", "hist_ratio": "MACD柱/价",
    "rank_r20": "20日收益截面排名", "rank_r60": "60日收益截面排名",
}


def pool_returns() -> pd.Series:
    """准入池等权日收益（仓库 rebound.backtest.market.pool_daily_return，口径和作者一致），缓存成 parquet。"""
    path = data_dir() / "pool_daily_return.parquet"
    if path.exists():
        s = pd.read_parquet(path)
        return s.set_index("trade_date")["pool_ret"]
    m = load_markets(DB_PATH, log=print)["nocap"]
    s = pd.Series(pool_daily_return(m), index=m.dates, name="pool_ret")
    s.rename_axis("trade_date").reset_index().to_parquet(path, index=False)
    return s


def index_returns() -> pd.DataFrame:
    con = connect()
    idx = con.execute("SELECT ts_code, trade_date, close FROM index_daily").df()
    con.close()
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    wide = idx.pivot(index="trade_date", columns="ts_code", values="close").rename(columns=INDEX_NAMES)
    return wide.pct_change()


def fig_pool_size(panel: pd.DataFrame, path) -> pd.Series:
    n = panel.groupby("trade_date").size()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(n.index, n.values, lw=0.8)
    ax.set_title("不限价准入池每天的股票数")
    ax.set_ylabel("股票数")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return n


def fig_pool_vs_index(pool: pd.Series, idx: pd.DataFrame, path) -> pd.DataFrame:
    start = pd.Timestamp("2013-08-01")          # 准入池从 2013 年 8 月起才有股票（见 1.2）
    rets = pd.concat([pool.rename("准入池等权"), idx], axis=1).loc[start:"2026-10-23"].fillna(0)
    nav = (1 + rets).cumprod()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col in nav.columns:
        ax.plot(nav.index, nav[col], lw=1.6 if col == "准入池等权" else 0.9, label=col)
    ax.set_yscale("log")
    ax.set_title("准入池等权净值 vs 指数（2013-08 起，对数坐标，不含分红、不扣成本）")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    yearly = rets.groupby(rets.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    return yearly.round(1)


def fig_label(panel: pd.DataFrame, path) -> dict:
    r = panel["ret_o1o5"].dropna()
    q01, q99 = r.quantile([0.01, 0.99])
    disp = panel.groupby("trade_date")["ret_o1o5"].std()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(r.clip(q01, q99) * 100, bins=100, color="#4C72B0")
    axes[0].set_title("o1o5 收益分布（1%~99% 截尾）")
    axes[0].set_xlabel("收益（%）")
    years = panel.assign(year=panel["trade_date"].dt.year)
    data = [g["ret_o1o5"].dropna().clip(q01, q99) * 100 for _, g in years.groupby("year")]
    axes[1].boxplot(data, tick_labels=sorted(years["year"].unique()), showfliers=False)
    axes[1].set_title("逐年分布（不画离群点）")
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].axhline(0, color="grey", lw=0.6)
    axes[2].plot(disp.index, disp.values * 100, lw=0.6)
    axes[2].set_title("每天的截面离散度（o1o5 收益的标准差）")
    axes[2].set_ylabel("%")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    # 截面百分位标签的意义：一天之内的收益差（离散度）远大于大盘涨跌在不同日子之间的差别吗？
    day_mean = panel.groupby("trade_date")["ret_o1o5"].mean()
    total_var = r.var()
    between = panel.groupby("trade_date")["ret_o1o5"].transform("mean").loc[r.index].var()
    return {"n": len(r), "mean": r.mean(), "median": r.median(), "std": r.std(), "skew": stats.skew(r),
            "kurt": stats.kurtosis(r), "q01": q01, "q99": q99, "p_pos": (r > 0).mean(),
            "disp_median": disp.median(), "day_mean_std": day_mean.std(), "between_share": between / total_var}


def fig_features(panel: pd.DataFrame, dist_path, corr_path, sample: int = 300_000, seed: int = 0):
    feats = list(FEATURES)
    x = panel[feats]
    desc = x.describe(percentiles=[0.01, 0.5, 0.99]).T
    desc["skew"] = x.sample(sample, random_state=seed).skew()
    desc["中文名"] = [FEATURE_CN[f] for f in feats]
    desc["不同取值数"] = [x[f].nunique() if f in ("ma20_gt_ma60", "limit_up_today", "launch10", "up_days10", "limit_up20") else np.nan
                     for f in feats]

    s = x.sample(sample, random_state=seed)
    fig, axes = plt.subplots(4, 7, figsize=(20, 10))
    for ax, f in zip(axes.flat, feats):
        v = s[f].dropna()
        lo, hi = v.quantile([0.01, 0.99])
        ax.hist(v.clip(lo, hi), bins=50, color="#55A868")
        ax.set_title(f"{f}\n{FEATURE_CN[f]}", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle("28 个原特征的分布（随机抽 30 万行，按 1%~99% 截尾）")
    fig.tight_layout()
    fig.savefig(dist_path, dpi=110)
    plt.close(fig)

    corr = s.rank().corr()   # Spearman = 秩的 Pearson
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={"shrink": 0.7},
                xticklabels=feats, yticklabels=feats)
    ax.set_title("28 个原特征的 Spearman 相关矩阵（随机抽 30 万行）")
    fig.tight_layout()
    fig.savefig(corr_path, dpi=110)
    plt.close(fig)
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:] if abs(corr.loc[a, b]) > 0.8]
    pairs = pd.DataFrame(pairs, columns=["特征1", "特征2", "Spearman"]).sort_values("Spearman", key=abs, ascending=False)
    return desc, pairs


def main() -> None:
    out = result_dir("p1_3_eda")
    panel = pd.read_parquet(data_dir() / "panel_nocap.parquet")
    pool = pool_returns()
    idx = index_returns()

    n = fig_pool_size(panel, out / "fig1_pool_size.png")
    yearly = fig_pool_vs_index(pool, idx, out / "fig2_pool_vs_index.png")
    yearly.to_csv(out / "yearly_returns.csv", encoding="utf-8-sig")
    lab = fig_label(panel, out / "fig3_label_dist.png")
    desc, pairs = fig_features(panel, out / "fig4_feature_dist.png", out / "fig5_feature_corr.png")
    desc.to_csv(out / "feature_stats.csv", encoding="utf-8-sig")
    pairs.to_csv(out / "high_corr_pairs.csv", index=False, encoding="utf-8-sig")

    ann = lambda s: ((1 + s.loc["2013-08-01":"2026-10-23"].fillna(0)).prod() ** (242 / s.loc["2013-08-01":"2026-10-23"].size) - 1) * 100  # noqa: E731
    ann_rows = {"准入池等权": ann(pool), **{c: ann(idx[c]) for c in idx.columns}}
    ann_md = "\n".join(f"| {k} | {v:+.1f}% |" for k, v in ann_rows.items())

    text = f"""
# 1.3 探索性分析（EDA）

- 运行：`python -X utf8 -m ml.p1_3_eda`
- 输入：`data/ml/panel_nocap.parquet`；准入池日收益用仓库的 `pool_daily_return`（口径和作者一致），缓存到 `data/ml/pool_daily_return.parquet`
- 产物：fig1 ~ fig5、`feature_stats.csv`、`high_corr_pairs.csv`、`yearly_returns.csv`

## 一、准入池规模（fig1）

每天平均 {n.mean():,.0f} 只，最少 {n.min():,} 只（{n.idxmin().date()}），最多 {n.max():,} 只（{n.idxmax().date()}）。
规模跟着成交额走：成交额门槛是固定的 0.8 亿元，市场活跃时（2015、2025 年起）进池的股票明显变多。
这意味着**不同年份的截面大小差别很大**，所以评估时 Top-K 用固定的 K（6、30），IC 按天算再平均。

## 二、准入池 vs 指数（fig2）

2013-08 ~ 2026-10 的年化收益（价格指数，不含分红；准入池等权、每天再平衡、不扣成本）：

| 序列 | 年化 |
|---|---|
{ann_md}

逐年收益（%）：

{yearly.to_markdown()}

准入池（成交额门槛下的主板股票等权）长期跑输大盘指数。这和仓库十年实验的记录一致（作者记录准入池十年年化 −5.6%，
口径从 2015 年起算，这里从 2013 年 8 月起，数字会略有不同）。所以模型的“超额”要相对准入池来算，而不是相对指数；
否则模型只要“少拿一点池子里的票”看起来就有超额。

## 三、标签：o1o5 收益（fig3）

| 统计量 | 数值 |
|---|---|
| 有效样本 | {lab['n']:,} |
| 均值 / 中位数 | {lab['mean'] * 100:+.3f}% / {lab['median'] * 100:+.3f}% |
| 标准差 | {lab['std'] * 100:.2f}% |
| 偏度 / 超额峰度 | {lab['skew']:.2f} / {lab['kurt']:.1f} |
| 1% / 99% 分位 | {lab['q01'] * 100:+.1f}% / {lab['q99'] * 100:+.1f}% |
| 收益为正的比例 | {lab['p_pos'] * 100:.1f}% |
| 每天截面标准差（中位数） | {lab['disp_median'] * 100:.2f}% |
| “日均收益”在不同日子之间的标准差 | {lab['day_mean_std'] * 100:.2f}% |
| 总方差中“日与日之间”的占比 | {lab['between_share'] * 100:.1f}% |

要点：
1. **厚尾、右偏**：峰度远大于 0（正态分布是 0），极端收益多。直接用收益做回归目标，少数极端值会主导损失函数，
   这是仓库把标签换成“当天截面百分位”的原因之一：百分位在 0 ~ 1 之间均匀分布，对极端值不敏感。
2. **大盘涨跌约占三成方差，七成来自个股之间**：总方差里约 {lab['between_share'] * 100:.0f}% 来自“哪一天”（大盘整体涨跌），
   约 {100 - lab['between_share'] * 100:.0f}% 来自同一天里股票之间的差别。大盘涨跌很难预测，而且选股模型也控制不了；
   我们要预测的是后者（同一天里谁涨得多），所以标签按天做截面排名、去掉大盘涨跌是合理的。
3. **截面离散度随时间变化**（fig3 右图）：2015 年股灾前后最大。离散度大的日子，选对选错的差距也大。

## 四、28 个原特征（fig4、fig5）

- 描述统计见 `feature_stats.csv`（均值、标准差、1%/50%/99% 分位、偏度）。
- **大多数特征厚尾**（例如各期收益、偏离均线、成交额比），线性模型和 MLP 必须先做截面排名或缩尾（任务 1.5）；树模型对单调变换不敏感，可以直接用原值。
- 有 5 个是离散特征：`ma20_gt_ma60`、`limit_up_today` 是 0/1，`launch10`、`up_days10`、`limit_up20` 是计数。
- **高度相关的特征对**（|Spearman| > 0.8）共 {len(pairs)} 对：

{pairs.round(3).to_markdown(index=False) if len(pairs) else "无"}

  这些大多是同一信息的不同写法：`pct_chg`（不复权涨跌幅）和 `r1`（复权日收益）几乎完全相同；
  各期收益和对应周期的均线偏离、均线斜率、MACD 也高度相关。
  对树模型影响不大；对 Ridge 会造成系数不稳定（这正是用 Ridge 而不是 OLS 的原因之一）。
  做 SHAP 解释时，高相关特征会互相“分走”重要性，报告里要说明。

## 五、对后续任务的影响

| 发现 | 后续做法 |
|---|---|
| 准入池长期跑输指数 | 超额一律相对准入池等权计算 |
| 标签厚尾，约七成方差来自个股之间 | 沿用截面百分位标签；评估用 Rank IC（对极端值稳健） |
| 特征厚尾、量纲不同 | 1.5：Ridge 和 MLP 用截面排名标准化，树模型用原值 |
| 每天截面大小差别大 | IC 按天算再平均；Top-K 用固定 K |
| 部分特征高度相关 | 线性模型用 Ridge；SHAP 解释时说明共线性 |
"""
    write_md(out / "report.md", text)
    register("1.3_eda", label_stats={k: float(v) for k, v in lab.items()}, high_corr_pairs=len(pairs),
             annualized={k: round(float(v), 2) for k, v in ann_rows.items()})


if __name__ == "__main__":
    main()
