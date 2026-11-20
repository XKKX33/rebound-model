"""P8 交付版结果：只含三个 F69 模型（PREREG 补充登记十三）。

不训练任何模型，只读已有的分数：
- 测试段（2023-01 ~ 2026-10，正式数字）：排行榜（`leaderboard.json` 的三行）、模型两两的 IC 差、组合回测（P5 基准口径）和组合超额差；
- 长期回测（2015 ~ 2026，补充分析）：`data/ml/history/scores_<模型>.parquet`（LSTM 由 `ml/p7_history_lstm.py` 补做），
  用与 P7 完全相同的模拟器、规则、费用重算三个模型，加逐年 IC、极端行情、月度收益、换起点、交易统计；
- LSTM 长期分数与测试段正式分数的一致性（GPU 训练不能逐位复现，看相关系数和 IC 差）。

输出全部在 `ml/results/delivery/`；LSTM 的交易明细、每日持仓、每日前 30 名写到 `data/ml/history/`（树模型的 P7 已有）。

运行：python -X utf8 -m ml.p8_delivery            # 约 15 分钟
      python -X utf8 -m ml.p8_delivery --report   # 只重写报告
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import RESULTS, data_dir, register, result_dir, write_md

OUT = "delivery"
M3 = ["xgb_F69", "rf_F69", "lstm_SF69"]
NAMES = {"xgb_F69": "XGBoost F69", "rf_F69": "随机森林 F69", "lstm_SF69": "LSTM S+F69"}
PAIRS = [("rf_F69", "xgb_F69", "随机森林 − XGBoost（RQ1，事先登记）"), ("lstm_SF69", "xgb_F69", "LSTM S+F69 − XGBoost（只描述）"),
         ("lstm_SF69", "rf_F69", "LSTM S+F69 − 随机森林（只描述）")]
BENCH = ["准入池等权", "沪深300", "中证500", "中证1000"]
LB_COLS = ["model_id", "验证 IC（用于选择，偏乐观）", "测试 IC 全段", "测试 IC 全段 区间", "测试 IC T1", "测试 IC T2", "测试 ICIR", "测试 NW t",
           "测试前30超额", "测试多空", "测试 IC 2023", "测试 IC 2024", "测试 IC 2025", "测试 IC 2026", "测试种子极差", "训练秒（最终阶段）"]


def _test_ic(keys: pd.DataFrame) -> dict[str, pd.Series]:
    from ml.p7_history_analyze import daily_ic
    return {m: daily_ic(keys, pd.read_parquet(data_dir() / "scores" / f"{m}__final.parquet", columns=["trade_date", "ts_code", "score"]))
            for m in M3}


def test_part(keys: pd.DataFrame) -> dict:
    lb = pd.DataFrame(json.loads((RESULTS / "leaderboard.json").read_text(encoding="utf-8"))).set_index("model_id").reindex(M3).reset_index()
    lb = lb[LB_COLS]
    lb.insert(1, "模型", [NAMES[m] for m in lb["model_id"]])
    ics = _test_ic(keys)
    spread = lb.set_index("model_id")["测试种子极差"]
    cmp_rows = []
    for a, b, label in PAIRS:
        c = ev.compare(ics[a], ics[b])
        cmp_rows.append({"比较": label, "IC 差": round(c["diff"], 4), "95% 区间": f"[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]",
                         "NW t": round(c["nw_t"], 2), "判定（PREREG 2.4 的规则）": ev.verdict(c, spread[a], spread[b])})
    p5 = pd.DataFrame(json.loads((RESULTS / "p5_backtest" / "results.json").read_text(encoding="utf-8")))
    p5 = p5[(p5["口径"] == "基准") & p5["model_id"].isin(M3)].copy()
    p5["模型"] = p5["model_id"].map(NAMES)
    port = p5[["模型", "规则", "本金", "全段 年化%", "全段 超额%", "全段 超额区间", "全段 最大回撤%", "全段 β", "换手（倍/年）", "费用占比（%/年）", "平均持有天数"]]
    order = {"DB-K30-b8": 0, "FH-K6-h4": 1, "DB-K3-b16": 2}
    port = port.assign(_r=port["规则"].map(order), _m=port["模型"].map({v: i for i, v in enumerate(NAMES.values())}))
    port = port.sort_values(["_r", "_m"]).drop(columns=["_r", "_m"])
    from rebound.backtest.stats import month_ci
    pdiff = []
    for rule in ("DB-K30-b8", "FH-K6-h4"):
        nav = pd.read_csv(RESULTS / "p5_backtest" / f"nav_{rule}.csv", index_col=0, parse_dates=True)
        r = nav.pct_change().iloc[1:]
        for a, b, label in PAIRS:
            d = r[a] - r[b]
            lo, hi = month_ci(d)
            pdiff.append({"规则": rule, "比较": label.split("（")[0], "年化超额差%": round(d.mean() * 242 * 100, 1),
                          "95% 区间": f"[{lo * 242 * 100:+.1f}, {hi * 242 * 100:+.1f}]"})
    return {"leaderboard": lb, "ic_cmp": pd.DataFrame(cmp_rows), "portfolio": port, "port_diff": pd.DataFrame(pdiff),
            "yearly_ic": pd.DataFrame({NAMES[m]: s.groupby(s.index.year).mean() for m, s in ics.items()}).round(4)}


def lstm_consistency(keys: pd.DataFrame) -> pd.DataFrame:
    from ml.p7_history_analyze import daily_ic, load_scores
    h = load_scores("lstm_SF69")
    f = pd.read_parquet(data_dir() / "scores" / "lstm_SF69__final.parquet", columns=["trade_date", "ts_code", "score"])
    m = f.merge(h[["trade_date", "ts_code", "score"]], on=["trade_date", "ts_code"], suffixes=("_正式", "_长期"))
    ic_f, ic_h = daily_ic(keys, f), daily_ic(keys, m.rename(columns={"score_长期": "score"})[["trade_date", "ts_code", "score"]])
    daily_corr = m.groupby("trade_date").apply(lambda g: g["score_正式"].corr(g["score_长期"], method="spearman"))
    return pd.DataFrame([{"对上的行数": len(m), "测试段正式分数行数": len(f), "逐日秩相关（中位数）": round(float(daily_corr.median()), 4),
                          "逐日秩相关（最低）": round(float(daily_corr.min()), 4), "正式分数测试 IC": round(float(ic_f.mean()), 4),
                          "长期回测分数测试 IC": round(float(ic_h.mean()), 4)}])


def long_part(keys: pd.DataFrame) -> dict:
    from ml.p7_history_analyze import (MAIN, SEGMENTS, backtests, daily_ic, export_details, holding_stats, load_scores, monthly,
                                       stress_table)
    from ml.p7_starts import run as run_starts
    ics = {m: daily_ic(keys, load_scores(m)) for m in M3}
    yearly_ic = pd.DataFrame({NAMES[m]: s.groupby(s.index.year).mean() for m, s in ics.items()}).round(4)
    ic_cmp = []
    for a, b, label in PAIRS:
        row = {"比较": label.split("（")[0]}
        for seg, (y0, y1) in SEGMENTS.items():
            sel = (ics[a].index.year >= y0) & (ics[a].index.year <= y1)
            c = ev.compare(ics[a][sel], ics[b].reindex(ics[a].index)[sel])
            row[seg] = f"{c['diff']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]"
        ic_cmp.append(row)
    rows, years_df, nav_tabs, trades_all, _, _, _, ind, cands_cache, m = backtests(keys, models=M3, extras=False)
    export_details({k: v for k, v in trades_all.items() if k[0] == "lstm_SF69"}, ind, {"lstm_SF69": cands_cache["lstm_SF69"]}, m, nav_tabs)
    navs = {r: t[M3 + BENCH] for r, t in nav_tabs.items()}
    stress = stress_table(navs[MAIN], M3 + ["准入池等权", "中证1000", "沪深300"])
    starts = run_starts(M3)
    hs = {mid: holding_stats(mid, MAIN) for mid in M3}
    return {"ics": ics, "yearly_ic": yearly_ic, "ic_cmp": pd.DataFrame(ic_cmp), "results": rows, "yearly": years_df, "navs": navs,
            "stress": stress, "starts": starts, "holding": hs, "monthly": {r: monthly(t) for r, t in navs.items()}}


def figures(out, navs: dict, ics: dict, years_df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ml.p7_history_analyze import LAST_SIGNAL, MAIN, monthly
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"xgb_F69": "#1f5fa8", "rf_F69": "#2e8b57", "lstm_SF69": "#c0392b", "准入池等权": "k", "中证1000": "#999"}
    nav = navs[MAIN]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    for c in M3 + ["准入池等权", "中证1000"]:
        ax.plot(nav.index, nav[c], ":" if c == "准入池等权" else "-", color=colors[c], lw=1.5 if c in M3 else 1.0, label=NAMES.get(c, c))
    ax.set_yscale("log")
    for a, b, col in (("2021-01-01", "2022-12-31", "#fff3cd"), ("2023-01-01", LAST_SIGNAL, "#d4edda")):
        for x in (ax, ax2):
            x.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=col, alpha=0.6, lw=0)
    ax.set_ylabel("净值（对数刻度，2015 年初 = 1）")
    ax.set_title(f"三个模型的长期回测：{MAIN}，20 万（黄 = 挑参数的年份，绿 = 样本外；之前仅供参考）")
    ax.legend(fontsize=9, ncol=3)
    for c in M3 + ["准入池等权"]:
        ax2.plot(nav.index, (nav[c] / nav[c].cummax() - 1) * 100, color=colors[c], lw=1, label=NAMES.get(c, c))
    ax2.set_ylabel("回撤（%）")
    fig.tight_layout()
    fig.savefig(out / "fig1_long_nav.png", dpi=130)
    plt.close(fig)

    y = years_df[years_df["规则"] == MAIN].pivot(index="年", columns="model_id", values="超额%")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for k, c in enumerate(M3):
        ax.bar(y.index + (k - 1) * 0.27, y[c], width=0.27, color=colors[c], label=NAMES[c])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(y.index)
    ax.set_ylabel("当年年化超额（%，相对准入池等权）")
    ax.set_title(f"逐年超额（{MAIN}）")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig2_yearly_excess.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for mid in M3:
        s = ics[mid]
        yy = s.groupby(s.index.year).mean()
        ax.plot(yy.index, yy.values, marker="o", color=colors[mid], label=NAMES[mid])
    ax.axvspan(2020.5, 2022.5, color="#fff3cd", alpha=0.6, lw=0)
    ax.axvspan(2022.5, 2026.5, color="#d4edda", alpha=0.6, lw=0)
    ax.set_ylabel("当年逐日 Rank IC 均值")
    ax.set_title("逐年 IC（黄 = 挑参数的年份，绿 = 样本外）")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig3_yearly_ic.png", dpi=130)
    plt.close(fig)

    mo = monthly(nav[M3 + ["准入池等权"]])
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    for a, mid in zip(axes, M3):
        ex = (mo[mid] - mo["准入池等权"]).to_frame("v")
        ex["年"], ex["月"] = ex.index.year, ex.index.month
        h = ex.pivot(index="年", columns="月", values="v")
        im = a.imshow(h.to_numpy(), cmap="RdYlGn", vmin=-10, vmax=10, aspect="auto")
        a.set_xticks(range(12))
        a.set_xticklabels(range(1, 13))
        a.set_yticks(range(len(h.index)))
        a.set_yticklabels(h.index)
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                v = h.iat[i, j]
                if np.isfinite(v):
                    a.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6)
        a.set_title(f"{NAMES[mid]} 月度超额（%，减准入池等权）")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(out / "fig4_monthly_excess.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def save(t: dict, lp: dict, cons: pd.DataFrame) -> None:
    out = result_dir(OUT)
    t["leaderboard"].to_json(out / "leaderboard.json", orient="records", force_ascii=False, indent=1)
    t["leaderboard"].to_csv(out / "leaderboard.csv", index=False, encoding="utf-8-sig")
    t["ic_cmp"].to_csv(out / "test_ic_compare.csv", index=False, encoding="utf-8-sig")
    t["portfolio"].to_csv(out / "test_portfolio.csv", index=False, encoding="utf-8-sig")
    t["port_diff"].to_csv(out / "test_portfolio_diff.csv", index=False, encoding="utf-8-sig")
    t["yearly_ic"].to_csv(out / "test_yearly_ic.csv", encoding="utf-8-sig")
    cons.to_csv(out / "lstm_consistency.csv", index=False, encoding="utf-8-sig")
    lp["results"].to_json(out / "long_results.json", orient="records", force_ascii=False, indent=1)
    lp["yearly"].to_csv(out / "long_yearly.csv", index=False, encoding="utf-8-sig")
    lp["yearly_ic"].to_csv(out / "long_yearly_ic.csv", encoding="utf-8-sig")
    lp["ic_cmp"].to_csv(out / "long_ic_compare.csv", index=False, encoding="utf-8-sig")
    lp["stress"].to_csv(out / "long_stress.csv", index=False, encoding="utf-8-sig")
    lp["starts"].to_csv(out / "long_starts.csv", index=False, encoding="utf-8-sig")
    for rule, nav in lp["navs"].items():
        nav.to_csv(out / f"long_nav_{rule}.csv", encoding="utf-8-sig")
        lp["monthly"][rule].to_csv(out / f"long_monthly_{rule}.csv", encoding="utf-8-sig")
    json.dump({mid: [v[0].reset_index().to_dict("records"), v[1].reset_index().to_dict("records"), v[2].reset_index().to_dict("records")]
               for mid, v in lp["holding"].items()}, open(out / "long_holding_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=str)
    figures(out, lp["navs"], lp["ics"], lp["yearly"])


def main() -> None:
    keys = ev.load_keys()
    cons = lstm_consistency(keys)
    print(cons.to_string(index=False), flush=True)
    t = test_part(keys)
    print(t["ic_cmp"].to_string(index=False), flush=True)
    lp = long_part(keys)
    save(t, lp, cons)
    register("P8_delivery", lstm_consistency=cons.to_dict("records"))
    from ml.p8_delivery_report import report
    report()


if __name__ == "__main__":
    if "--report" in sys.argv:
        from ml.p8_delivery_report import report
        report()
    else:
        main()
