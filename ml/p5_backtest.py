"""P5 组合回测：复用仓库的模拟器（`rebound.backtest`），不重写。只报测试期（2023-01 起），分数是测试段按年重训的正式分数。

- 5.1 适配器：`data/ml/scores/<model>__final.parquet` → (di, ji, score) → `market.candidates` → `sim.simulate`；
  行情矩阵 `load_markets` 读一次（不限价准入池 nocap，和主面板一致）。
- 5.2 固定 3 条规则，不做规则搜索（PREREG 2.3）：20 万 `DB-K30-b8`（主）、2 万 `DB-K3-b16`、20 万 `FH-K6-h4`；
  费用按仓库真实口径（`CostModel`：佣金万五、最低 5 元、过户费、印花税 0.05%、一手取整）。买卖时点按仓库现行口径（开盘买、尾盘卖）。
- 5.4 稳健性：费用上调 50%、历史印花税（主规则）；逐年（单列 2026）；种子差异（主规则，每个种子的分数单独回测）。
- 指标（`rebound.backtest.stats`）：年化、相对准入池等权的年化超额 [按月整块重抽 95% 区间]、β、最大回撤、换手、费用占比；T1 / T2 / 逐年。
- 组合层结果只报告，不用来挑模型（PREREG 2.5）。

运行：python -X utf8 -m ml.p5_backtest
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

from ml.common import DB_PATH, RESULTS, data_dir, load_config, register, result_dir, write_md

OUT = "p5_backtest"
RULES = {"DB-K30-b8": 200_000, "DB-K3-b16": 20_000, "FH-K6-h4": 200_000}
MAIN_RULE = "DB-K30-b8"
MODELS = ["random", "rev_r5", "xgb_repro_F28", "rf_F28", "rf_F69", "xgb_F28", "xgb_F69", "lstm_S", "lstm_SF69",
          "xgb_clf_F69", "xgb_rank_F69", "ens_top3", "ens_family"]


def to_market_index(m, sc: pd.DataFrame, col: str = "score", name: str = "") -> pd.DataFrame:
    """5.1 适配器：分数表（trade_date, ts_code, 分数列）→ 行情矩阵轴上的 (di, ji, score)，交给 `market.candidates`。

    di = 日期在 `m.dates` 里的下标，ji = 代码在 `m.codes` 里的下标；有任何对不上的日期或代码就报错（不悄悄丢掉）。
    """
    di = m.dates.get_indexer(pd.DatetimeIndex(sc["trade_date"]))
    ji = m.code_index(sc["ts_code"])
    if (di < 0).any() or (ji < 0).any():
        raise ValueError(f"{name}：分数里有日期或代码不在行情矩阵的轴上")
    if pd.DataFrame({"di": di, "ji": ji}).duplicated().any():
        raise ValueError(f"{name}：同一天同一只票有重复的分数")
    return pd.DataFrame({"di": di.astype(np.int64), "ji": ji.astype(np.int64), "score": sc[col].to_numpy(np.float64)})


def _scores(m, model_id: str, col: str = "score") -> pd.DataFrame:
    return to_market_index(m, pd.read_parquet(data_dir() / "scores" / f"{model_id}__final.parquet"), col, model_id)


def main() -> None:
    from rebound.backtest import stats
    from rebound.backtest.costs import CostModel
    from rebound.backtest.market import candidates, load_markets, pool_daily_return
    from rebound.backtest.rules import rule_of
    from rebound.backtest.sim import simulate
    t0 = time.time()
    s = load_config()["split"]
    m = load_markets(DB_PATH, start=date(2022, 1, 1))["nocap"]
    pool, dates = pool_daily_return(m), m.dates
    first = pd.read_parquet(data_dir() / "scores" / "xgb_F69__final.parquet", columns=["trade_date"])["trade_date"]
    d_first = int(dates.get_indexer([first.min()])[0])
    d_last = int(dates.get_indexer([first.max()])[0])
    idx = stats.scoring_days(d_first, d_last)
    cut = pd.Timestamp(s["test_split_at"])
    parts = {"全段": idx, "T1": idx[dates[idx] < cut], "T2": idx[dates[idx] >= cut]}
    print(f"行情读完 {time.time() - t0:.0f}s；信号日 {dates[d_first].date()} ~ {dates[d_last].date()}，计分 {len(idx)} 天", flush=True)
    rows, years, navs = [], [], {}

    def one(model_id, rule_name, cost, tag, col="score"):
        cands, rank = candidates(m, _scores(m, model_id, col))
        res = simulate(m, cands, rank, rule_of(rule_name), cost, d_first, d_last)
        summ = stats.summary(res, pool, dates, idx)
        row = {"model_id": model_id, "规则": rule_name, "本金": cost.capital, "口径": tag,
               "换手（倍/年）": summ["turnover"], "费用占比（%/年）": summ["cost_pct"], "平均持股": summ["avg_holdings"],
               "空仓现金（%）": summ["idle_cash_pct"], "平均持有天数": summ["hold_days"]}
        for p, ix in parts.items():
            per = stats.period(res.nav, pool, dates, ix)
            row.update({f"{p} 年化%": per["cagr"], f"{p} 超额%": per["excess"], f"{p} 超额区间": per["excess_ci"],
                        f"{p} β": per["beta"], f"{p} 最大回撤%": per["max_dd"]})
        for y, v in summ["years"].items():
            years.append({"model_id": model_id, "规则": rule_name, "口径": tag, "年": int(y), "收益%": v["total"], "超额%": v["excess"]})
        return row, res

    for rule_name, capital in RULES.items():
        for mid in MODELS:
            row, res = one(mid, rule_name, CostModel(capital=float(capital)), "基准")
            rows.append(row)
            navs[(rule_name, mid)] = res.nav[idx[0] - 1:idx[-1] + 1] / res.nav[idx[0] - 1]
            print(f"[{rule_name}] {mid}：年化 {row['全段 年化%']:+.1f}%，超额 {row['全段 超额%']:+.1f}%（{time.time() - t0:.0f}s）", flush=True)
    # 5.4 稳健性：费用上调 50%（主规则）
    base = CostModel(capital=float(RULES[MAIN_RULE]))
    high = replace(base, commission=base.commission * 1.5, min_commission=base.min_commission * 1.5, stamp=base.stamp * 1.5,
                   transfer=base.transfer * 1.5)
    for mid in MODELS:
        row, _ = one(mid, MAIN_RULE, high, "费用 ×1.5")
        rows.append(row)
    # 5.4 稳健性：历史印花税（2023-08-28 之前是 0.1%，基准口径按 PREREG 全期用现行 0.05%）
    hist = replace(base, stamp_before=0.001)
    for mid in MODELS:
        row, _ = one(mid, MAIN_RULE, hist, "历史印花税")
        rows.append(row)
    # 5.4 稳健性：种子差异（主规则，每个种子的分数单独回测）
    for mid in MODELS:
        cols = [c for c in pd.read_parquet(data_dir() / "scores" / f"{mid}__final.parquet").columns if c[:1] == "s" and c[1:].isdigit()]
        if len(cols) < 2:
            continue
        for c in cols:
            row, _ = one(mid, MAIN_RULE, base, f"种子 {c[1:]}", col=c)
            rows.append(row)
    print(f"全部完成 {time.time() - t0:.0f}s", flush=True)
    out = result_dir(OUT)
    tab = pd.DataFrame(rows)
    tab.to_json(out / "results.json", orient="records", force_ascii=False, indent=1)
    tab.drop(columns=[c for c in tab.columns if c.endswith("区间")]).to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(years).to_csv(out / "yearly.csv", index=False, encoding="utf-8-sig")
    pool_nav = np.r_[1.0, np.cumprod(1 + np.nan_to_num(pool[idx]))]
    for rule_name in RULES:                                  # 每条规则一份净值表（api.nav 读它）
        nav = pd.DataFrame({mid: navs[(rule_name, mid)] for mid in MODELS}, index=dates[idx[0] - 1:idx[-1] + 1])
        nav["准入池等权"] = pool_nav
        nav.index.name = "date"
        nav.to_csv(out / f"nav_{rule_name}.csv", encoding="utf-8-sig")
    register("P5_backtest", n=len(tab), seconds=round(time.time() - t0, 1))
    report()


def report() -> None:
    out = result_dir(OUT)
    tab = pd.DataFrame(json.loads((out / "results.json").read_text(encoding="utf-8")))
    yearly = pd.read_csv(out / "yearly.csv")
    lb = pd.read_csv(RESULTS / "leaderboard.csv").set_index("model_id")
    fmt = lambda v: f"[{v[0]:+.1f}, {v[1]:+.1f}]"  # noqa: E731
    main = []
    for rule in RULES:
        t = tab[(tab["规则"] == rule) & (tab["口径"] == "基准")].copy()
        t["全段 超额区间"] = t["全段 超额区间"].map(fmt)
        main.append((rule, t[["model_id", "全段 年化%", "全段 超额%", "全段 超额区间", "T1 超额%", "T2 超额%", "全段 β", "全段 最大回撤%",
                              "换手（倍/年）", "费用占比（%/年）", "平均持股"]]))
    cost = tab[tab["口径"].isin(["基准", "费用 ×1.5", "历史印花税"]) & (tab["规则"] == MAIN_RULE)].pivot(index="model_id", columns="口径", values="全段 超额%")
    seeds = tab[tab["口径"].str.startswith("种子") & (tab["规则"] == MAIN_RULE)].pivot(index="model_id", columns="口径", values="全段 超额%")
    seeds["极差"] = seeds.max(axis=1) - seeds.min(axis=1)
    yr = yearly[(yearly["规则"] == MAIN_RULE) & (yearly["口径"] == "基准")].pivot(index="model_id", columns="年", values="超额%")
    ic = lb["测试 IC 全段"]
    rq4 = tab[(tab["规则"] == MAIN_RULE) & (tab["口径"] == "基准")].set_index("model_id")[["全段 超额%", "换手（倍/年）", "费用占比（%/年）"]]
    rq4["测试 IC"] = ic
    rq4["测试前30超额（单笔，%）"] = lb["测试前30超额"] * 100
    _figures(out, tab, rq4)
    notes_path = out / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "（读法待补）"
    order = lambda d: d.reindex([m for m in MODELS if m in d.index])  # noqa: E731
    text = f"""
# P5 组合回测（测试期 2023-01 ~ 2026-10）

- 代码：`ml/p5_backtest.py`；结果 `results.json / .csv`（全部规则、口径）、`yearly.csv`、`nav_<规则>.csv`（每条规则各模型的净值）
- 模拟器：仓库的 `rebound.backtest`（`load_markets` → `candidates` → `simulate`，`stats.summary / period`），没有重写；
  不限价准入池（nocap）；真实费用口径（佣金万五、最低 5 元、过户费、印花税 0.05%、一手取整）；开盘买、尾盘卖（仓库现行时点）。
- 分数：测试段按年重训的正式分数（PREREG 补充登记十）。超额 = 相对准入池等权（每天收盘对收盘）的年化超额，区间是按月整块重抽 95% 区间。
- 组合层结果只报告，不用来挑模型（PREREG 2.5）。

## 一、5.2 三条固定规则
""" + "\n".join(f"""
### {rule}（本金 {RULES[rule] // 10000} 万）

{order(t.set_index("model_id")).reset_index().to_markdown(index=False)}
""" for rule, t in main) + f"""
## 二、5.4 稳健性（主规则 {MAIN_RULE}）

**费用敏感性：全段年化超额（%）**（费用 ×1.5 = 佣金、最低佣金、印花税、过户费都上调 50%；历史印花税 = 2023-08-28 之前按 0.1%）

{order(cost).round(2).to_markdown()}

**种子差异：每个种子的分数单独回测的全段年化超额（%）**

{order(seeds).round(2).to_markdown()}

**逐年超额（%，2026 年只到 10 月）**

{order(yr).round(1).to_markdown()}

![净值](fig1_nav.png)

## 三、5.5 RQ4：IC 高不一定组合收益高

{order(rq4).round(4).to_markdown()}

![IC vs 超额](fig2_ic_vs_excess.png)

## 四、读法

{notes}
"""
    write_md(out / "report.md", text)


def _figures(out, tab, rq4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    nav = pd.read_csv(out / f"nav_{MAIN_RULE}.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for c, st in (("准入池等权", "k--"), ("xgb_repro_F28", "-"), ("xgb_F28", "-"), ("xgb_F69", "-"), ("rf_F69", "-"),
                  ("lstm_SF69", "-"), ("ens_family", ":"), ("rev_r5", "-.")):
        ax.plot(nav.index, nav[c], st, lw=1.2, label=c)
    ax.set_ylabel("净值（起点 = 1）")
    ax.set_title(f"{MAIN_RULE}、20 万本金、测试期净值")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_nav.png", dpi=130)
    plt.close(fig)

    d = rq4.drop(index=["random"], errors="ignore")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})
    for k, ax in enumerate(axes):
        dd = d if k == 0 else d[d["测试 IC"] > 0.09]
        sc = ax.scatter(dd["测试 IC"], dd["全段 超额%"], c=dd["换手（倍/年）"], cmap="viridis", s=60,
                        vmin=d["换手（倍/年）"].min(), vmax=d["换手（倍/年）"].max())
        for n, (mid, r) in enumerate(dd.iterrows()):
            off = (5, 3) if k == 0 else (6, -10 + 7 * (n % 3))
            ax.annotate(mid, (r["测试 IC"], r["全段 超额%"]), fontsize=8, xytext=off, textcoords="offset points")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("测试期逐日 Rank IC")
        ax.set_ylabel(f"{MAIN_RULE} 年化超额（%）")
        ax.set_title("全部模型" if k == 0 else "放大：IC > 0.09 的模型")
    fig.colorbar(sc, ax=axes, label="换手（倍/年）")
    fig.suptitle("RQ4：预测准不准（IC）与组合赚不赚（超额），颜色 = 换手")
    fig.savefig(out / "fig2_ic_vs_excess.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if "--report" in sys.argv:
        report()
    else:
        main()
