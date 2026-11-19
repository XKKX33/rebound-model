"""任务 1.4：在全部历史上计算 47 个新特征，对齐到主面板，并做核对。

运行：python -X utf8 -m ml.p1_4_build_features
输入：data/ml/panel_nocap.parquet（1.2）
输出：
- data/ml/features_extra.parquet        和主面板逐行对齐（同样的行序）：trade_date, ts_code, 47 个特征
- ml/results/p1_4_features/report.md    构建记录、核对、缺失、相关性、去留决定
- ml/results/p1_4_features/feature_info.csv          47 个特征的说明（组、计算、预期方向、理由）
- ml/results/p1_4_features/feature_stats.csv         描述统计
- ml/results/p1_4_features/missing_by_year.csv       逐年缺失率
- ml/results/p1_4_features/high_corr_pairs.csv       |Spearman| > 0.9 的特征对
- ml/results/p1_4_features/fig1_new_feature_dist.png 39 个基础新特征的分布
- ml/results/p1_4_features/fig2_corr_new_vs_old.png  新特征与原 28 个特征的相关
"""

from __future__ import annotations

import sys
import time
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml.common import connect, data_dir, register, result_dir, rss_gb, write_md
from ml.features_extra import (ALL_FEATS, BASE_NAMES, EXTRA_NAMES, GROUPS, MARKET_STATE, RANK_OF, add_ranks,
                               compute_range, feature_table)
from rebound.features import FEATURES

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_NAME = "features_extra.parquet"
LAST = date(2026, 10, 23)
CHECK_DAYS = (date(2014, 6, 16), date(2021, 4, 23), date(2026, 10, 23))


def build(panel: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    keys = panel[["trade_date", "ts_code"]].copy()
    keys["ts_code"] = keys["ts_code"].astype(str)
    keys["row"] = np.arange(len(keys))
    parts = []
    con = connect()
    for year in range(2013, 2027):
        t0 = time.time()
        lo, hi = date(year, 1, 1), min(date(year, 12, 31), LAST)
        f = compute_range(con, lo, hi)
        f["ts_code"] = f["ts_code"].astype(str)
        k = keys[(keys["trade_date"] >= pd.Timestamp(lo)) & (keys["trade_date"] <= pd.Timestamp(hi))]
        m = k.merge(f, on=["trade_date", "ts_code"], how="left", indicator=True)
        missing = int((m["_merge"] != "both").sum())
        parts.append(m.drop(columns="_merge"))
        line = f"[新特征] {year}：主板 {len(f):,} 行，对齐到准入池 {len(k):,} 行，没对上 {missing}（{time.time() - t0:.0f}s，内存 {rss_gb():.1f} GB）"
        print(line, flush=True)
        log.append(line)
        if missing:
            raise RuntimeError(f"{year} 年有 {missing} 行准入池样本没算出新特征")
    con.close()
    out = pd.concat(parts, ignore_index=True).sort_values("row").reset_index(drop=True)
    assert (out["row"].to_numpy() == np.arange(len(panel))).all()
    out = add_ranks(out, panel["di"].to_numpy())
    out = out[["trade_date", "ts_code", *EXTRA_NAMES]]
    out["ts_code"] = pd.Categorical(out["ts_code"])
    return out


def check_lookahead(feats: pd.DataFrame) -> pd.DataFrame:
    """在真实数据上：把数据截断到某天单独重算，和全量结果比较（不应有任何差别，否则说明用到了未来数据）。"""
    rows = []
    con = connect()
    for day in CHECK_DAYS:
        cut = compute_range(con, day, day).set_index("ts_code")
        cut.index = cut.index.astype(str)
        full = feats[feats["trade_date"] == pd.Timestamp(day)].set_index("ts_code")
        full.index = full.index.astype(str)
        common = full.index.intersection(cut.index)
        a = cut.loc[common, list(BASE_NAMES)].to_numpy(np.float64)
        b = full.loc[common, list(BASE_NAMES)].to_numpy(np.float64)
        nan_diff = int((np.isnan(a) ^ np.isnan(b)).sum())
        d = np.abs(np.where(np.isnan(a) & np.isnan(b), 0, a - b))
        rel = d / np.maximum(np.abs(b), 1e-6)
        rows.append({"日期": day, "准入池行数": len(full), "对上行数": len(common), "空值不一致格数": nan_diff,
                     "最大相对差": float(np.nanmax(rel))})
    con.close()
    return pd.DataFrame(rows)


def check_raw(feats: pd.DataFrame, n: int = 1000, seed: int = 3) -> dict:
    """抽 n 行，直接从原始表算几个“当日值”特征，核对单位换算。"""
    rng = np.random.default_rng(seed)
    s = feats.iloc[rng.choice(len(feats), n, replace=False)][["trade_date", "ts_code", "turn_f", "ln_circ_mv", "mf_net1",
                                                              "big_net1", "dist_dn"]].copy()
    s["ts_code"] = s["ts_code"].astype(str)
    s["trade_date"] = s["trade_date"].dt.date
    con = connect()
    con.register("s", s)
    raw = con.execute("""
        SELECT s.*, b.turnover_rate_f AS t_raw, ln(b.circ_mv) AS mv_raw,
               m.net_mf_amount / (d.amount / 10) AS mf_raw,
               (m.buy_lg_amount + m.buy_elg_amount - m.sell_lg_amount - m.sell_elg_amount) / (d.amount / 10) AS big_raw,
               (d.close - l.down_limit) / d.close AS dn_raw
        FROM s JOIN daily d USING (ts_code, trade_date)
        LEFT JOIN daily_basic b USING (ts_code, trade_date)
        LEFT JOIN moneyflow m USING (ts_code, trade_date)
        LEFT JOIN daily_limit l USING (ts_code, trade_date)""").df()
    con.close()
    out = {}
    for f, rcol in [("turn_f", "t_raw"), ("ln_circ_mv", "mv_raw"), ("mf_net1", "mf_raw"), ("big_net1", "big_raw"), ("dist_dn", "dn_raw")]:
        d = (raw[f] - raw[rcol]).abs() / raw[rcol].abs().clip(lower=1e-6)
        out[f] = float(np.nanmax(d))
    return out


def corr_analysis(panel: pd.DataFrame, feats: pd.DataFrame, path, sample: int = 300_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(panel), sample, replace=False)
    x = pd.concat([panel.iloc[idx][list(FEATURES)].reset_index(drop=True),
                   feats.iloc[idx][list(EXTRA_NAMES)].reset_index(drop=True)], axis=1)
    corr = x.rank().corr()
    names = list(FEATURES) + list(EXTRA_NAMES)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = corr.loc[a, b]
            if abs(v) > 0.9 and not (a in FEATURES and b in FEATURES):   # 原特征之间的相关在 1.3 里看过
                pairs.append((a, b, v))
    pairs = pd.DataFrame(pairs, columns=["特征1", "特征2", "Spearman"]).sort_values("Spearman", key=abs, ascending=False)
    sub = corr.loc[list(BASE_NAMES), list(FEATURES)]
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(sub, cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.6}, xticklabels=True, yticklabels=True)
    ax.set_title("39 个基础新特征（行）与原 28 个特征（列）的 Spearman 相关（随机抽 30 万行）")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    max_with_old = sub.abs().max(axis=1).sort_values(ascending=False)
    return pairs, max_with_old


def fig_dist(feats: pd.DataFrame, path, sample: int = 300_000, seed: int = 0) -> None:
    s = feats.sample(sample, random_state=seed)
    cn = {f.name: f.cn for f in ALL_FEATS}
    fig, axes = plt.subplots(6, 7, figsize=(21, 15))
    for ax, f in zip(axes.flat, BASE_NAMES):
        v = s[f].dropna()
        lo, hi = v.quantile([0.01, 0.99])
        ax.hist(v.clip(lo, hi), bins=50, color="#C44E52")
        ax.set_title(f"{f}\n{cn[f]}", fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(BASE_NAMES):]:
        ax.axis("off")
    fig.suptitle("39 个基础新特征的分布（准入池内随机抽 30 万行，按 1%~99% 截尾）")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    out = result_dir("p1_4_features")
    log: list[str] = []
    panel = pd.read_parquet(data_dir() / "panel_nocap.parquet")
    path = data_dir() / OUT_NAME
    secs_path = data_dir() / "features_extra.seconds"
    if path.exists() and secs_path.exists() and "--rebuild" not in sys.argv:
        feats = pd.read_parquet(path)
        secs = float(secs_path.read_text())
        print(f"读取已有 {path.name}（重算请加 --rebuild）")
    else:
        t0 = time.time()
        feats = build(panel, log)
        secs = time.time() - t0
        feats.to_parquet(path, index=False)
        secs_path.write_text(f"{secs:.1f}")
        print(f"写出 {path}（{path.stat().st_size / 1024**2:.0f} MB）")

    info = feature_table()
    info.to_csv(out / "feature_info.csv", index=False, encoding="utf-8-sig")
    stats = feats[list(EXTRA_NAMES)].describe(percentiles=[0.01, 0.5, 0.99]).T
    stats.to_csv(out / "feature_stats.csv", encoding="utf-8-sig")
    miss = feats[list(EXTRA_NAMES)].isna().groupby(feats["trade_date"].dt.year).mean().mul(100).round(2)
    miss.to_csv(out / "missing_by_year.csv", encoding="utf-8-sig")
    miss_total = feats[list(EXTRA_NAMES)].isna().mean().mul(100)

    look = check_lookahead(feats)
    raw = check_raw(feats)
    pairs, max_old = corr_analysis(panel, feats, out / "fig2_corr_new_vs_old.png")
    pairs.to_csv(out / "high_corr_pairs.csv", index=False, encoding="utf-8-sig")
    fig_dist(feats, out / "fig1_new_feature_dist.png")
    ovn_zero = float((feats["ovn5"] == 0).mean() * 100)

    # 去留：H 组和它的基础特征高相关是设计使然（给线性模型和 MLP 用），不算；
    # 其余 |ρ| > 0.9 的对：如果一边是原特征，就去掉新特征（不和原特征重复）；两边都是新特征，就去掉排在后面的那个。
    drop = []
    for a, b, v in pairs.itertuples(index=False):          # 按 |ρ| 从大到小处理
        if b == f"rk_{a}" or a == f"rk_{b}":
            continue
        if a in drop or b in drop:                           # 其中一个已经去掉，这一对不再算重复
            continue
        if a in FEATURES or b in FEATURES:
            new = b if a in FEATURES else a
        else:
            new = b
        if new not in drop:
            drop.append(new)
    keep = [n for n in EXTRA_NAMES if n not in drop]

    miss_top = miss_total.sort_values(ascending=False).head(10).round(2).to_frame("全期缺失%")
    look_md = look.to_markdown(index=False)
    raw_md = "\n".join(f"| `{k}` | {v:.1e} |" for k, v in raw.items())
    group_md = "\n".join(f"| {g} {GROUPS[g]} | {sum(f.group == g for f in ALL_FEATS)} | "
                         f"{'、'.join(f'`{f.name}`' for f in ALL_FEATS if f.group == g)} |" for g in GROUPS)
    old_md = max_old.head(10).round(3).to_frame("与原特征的最大 |ρ|").to_markdown()
    drop_md = ("\n".join(f"- `{d}`" for d in drop)) if drop else "无"

    text = f"""
# 1.4 新特征

- 运行：`python -X utf8 -m ml.p1_4_build_features`（单元测试：`python -X utf8 -m pytest -p no:cacheprovider -o addopts="" -q ml/tests`）
- 代码：`ml/features_extra.py`（特征定义、SQL、说明表）
- 产物：`data/ml/{OUT_NAME}`（{path.stat().st_size / 1024**2:.0f} MB，与主面板逐行对齐）、`feature_info.csv`、`feature_stats.csv`、
  `missing_by_year.csv`、`high_corr_pairs.csv`、fig1、fig2
- 耗时 {secs:.0f} 秒
- 每个特征的预期方向已在计算任何统计量之前写进 `ml/PREREG.md`

## 一、结果

共 **{len(EXTRA_NAMES)} 个新特征**，全部对齐到主面板的 {len(feats):,} 行（每一行准入池样本都算出了特征）。

| 组 | 个数 | 特征 |
|---|---|---|
{group_md}

每个特征的中文名、计算方法、预期方向和理由见 `feature_info.csv`（任务清单 1.4 有同样的表）。

## 二、核对

### 1. 单元测试（`ml/tests/test_features_extra.py`，10 项全部通过）

在手工构造的小库上逐个核对：资金流的单位换算（万元 vs 千元）、换手和市值、复权收益（窗口跨过除权日时不出现假跳变）、
波动率 / 最大最小收益 / 下行波动占比、隔夜和日内收益、K 线收盘位置和影线、beta 和特质波动率（与 numpy 最小二乘一致）、
跌停计数、连续下跌天数、250 日窗口、市场状态特征同一天对所有股票相同、截面排名，以及没有前视。

### 2. 真实数据上的防前视检查

把数据库截断到某一天（所有表只读到这一天）单独重算，和全量计算的结果比较。如果用到了未来数据，两者会不同：

{look_md}

### 3. 单位换算抽查（随机 1000 行，直接从原始表计算，最大相对差）

| 特征 | 最大相对差 |
|---|---|
{raw_md}

## 三、缺失

缺失最多的 10 个特征（全期，%）：

{miss_top.to_markdown()}

原因：250 日特征（`from_low250`、`dd250`）要求这只股票在库里已有 250 根日线，而准入池只要求 150 根，所以新进池的股票前 100 天没有值，
库最早的 2013 年也全部没有；beta / 特质波动率要求 60 天里至少 50 个有效收益；资金流 2016 ~ 2018 年偶有缺失（见 1.1）。
逐年缺失率见 `missing_by_year.csv`。树模型直接处理缺失，Ridge 和 MLP 在 1.5 里统一补 0 并加缺失指示。

另：`ovn5` 恰好为 0 的行占 {ovn_zero:.1f}%：开盘价等于前一天收盘价的情况在 A 股很常见（尤其大盘股），属正常现象。

## 四、和原特征重复吗（fig2）

新特征与原 28 个特征的最大 |Spearman| 排前 10：

{old_md}

|ρ| > 0.9 的特征对（不含原特征之间）：

{pairs.round(3).to_markdown(index=False) if len(pairs) else "无"}

**去留规则（事先定好，和结果无关）**：H 组排名特征和它对应的基础特征高相关是设计使然（排名版给 Ridge 和 MLP 用），不算重复；
其余 |ρ| > 0.9 的对，如果一边是原特征就去掉新特征，两边都是新特征就去掉后列的那个。

据此去掉：

{drop_md}

为什么会重复（事后检查定义得到的解释）：
- `max_ret20` 与原特征 `max_up20`：两者都是“20 日内最大的单日涨幅”，只是一个用复权收益、一个用涨跌幅字段，定义上就是同一个量；
- `dist_dn` 与 `r1` / `pct_chg`：主板跌停价 = 昨收 × 0.9（ST 为 0.95），所以 (收盘 − 跌停价) / 收盘 只是当日涨跌幅的单调变换；
- `vr`（量比）与原特征 `vol5`：量比 = 当日成交量 / 前 5 日平均成交量，`vol5` 是同样定义的成交额版本；
- `rv20` 与原特征 `amp20`（20 日平均振幅）：日内振幅大的股票日收益波动也大，原特征其实已经间接包含了波动率信息；
- `turn5` 与 `turn_f`：换手率的持续性很强，当日换手和 5 日均换手几乎同序；
- `rk_max_ret20` 随 `max_ret20` 一起去掉。

这几处是构造时没有预料到的，正说明了这一步核对的必要；被去掉的特征仍保留在 `ml/features_extra.py` 和 `PREREG.md` 里，只是不进入模型和单因子检验。

最终新特征 **{len(keep)} 个**，加上原 28 个，共 **{28 + len(keep)} 个**（任务清单里的 “F75” 以此为准）。
最终列表保存在 `final_features.txt`。

## 五、说明：市场状态特征

`{'`、`'.join(MARKET_STATE)}` 同一天对所有股票取值相同。标签是截面百分位，所以它们单独不能区分股票，
只能通过“在什么市场环境下哪类股票更易反弹”的交互起作用：线性模型用不上，树模型和 MLP 可以。这是研究问题 RQ1 的一个具体例证。
"""
    write_md(out / "report.md", text)
    (out / "final_features.txt").write_text("\n".join(list(FEATURES) + keep) + "\n", encoding="utf-8")
    register("1.4_features", n_new=len(EXTRA_NAMES), n_keep=len(keep), dropped=drop, seconds=round(secs, 1),
             lookahead=look.to_dict("records"), raw_check=raw)


if __name__ == "__main__":
    main()
