"""P7 长期回测的分析部分（由 `python -m ml.p7_history analyze` 调用）。设置见 PREREG 补充登记十一。

读 data/ml/history/scores_<模型>.parquet（`p7_history train` 的产物），做：
1. 一致性核对：2023 ~ 2026 有效行上的分数 vs 测试段正式分数；
2. 逐年 IC（只在有有效收益的行上算）和模型之间的逐年比较；
3. 回测：7 个模型 × 3 条规则（印花税按历史税率），作者十年实验逐年对照，候选名单修正前后对比（2023 ~ 2026）；
4. 交易明细、每日持仓、每日候选名单（data/ml/history/），持仓统计、极端行情、参数敏感性、月度收益；
5. 图和报告（ml/results/p7_history/）。
"""

from __future__ import annotations

import json
import re
from datetime import date

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import DB_PATH, RESULTS, connect, data_dir, register, result_dir, write_md
from ml.p7_history import NAMES, OUT, START_YEAR, hist_dir

MODELS = ["xgb_F69", "xgb_F69_s5", "xgb_F69_s4", "xgb_F28", "xgb_repro_F28", "rf_F69", "ens_top3"]
ENS = ["xgb_F69", "rf_F69", "xgb_F28"]
RULES = {"DB-K30-b8": 200_000, "FH-K6-h4": 200_000, "DB-K3-b16": 20_000}
MAIN = "DB-K30-b8"
AUTHOR_RULE = "FH-K6-next-o1o5"
LAST_SIGNAL = "2026-10-16"                       # 与测试段相同：最后一个有有效收益的信号日
SEGMENTS = {"2015 ~ 2020（仅供参考）": (2015, 2020), "2021 ~ 2022（挑参数的年份，偏乐观）": (2021, 2022),
            "2023 ~ 2026（样本外）": (2023, 2026), "全期 2015 ~ 2026": (2015, 2026)}
STRESS = {"2015 股灾": ("2015-06-12", "2015-09-30"), "2016 熔断": ("2016-01-01", "2016-01-31"),
          "2018 熊市": ("2018-01-29", "2018-12-31"), "2020 疫情": ("2020-01-20", "2020-03-31"),
          "2024 小盘股暴跌": ("2024-01-01", "2024-02-07"), "2026 年 7 月": ("2026-07-01", "2026-07-31")}
INDICES = {"沪深300": "000300.SH", "中证500": "000905.SH", "中证1000": "000852.SH"}


# --------------------------------------------------------------------------- 分数


def load_scores(mid: str) -> pd.DataFrame:
    if mid != "ens_top3":
        return pd.read_parquet(hist_dir() / f"scores_{mid}.parquet")
    parts = [pd.read_parquet(hist_dir() / f"scores_{m}.parquet") for m in ENS]
    base = parts[0][["trade_date", "ts_code"]].copy()
    for p in parts[1:]:
        if not (p["ts_code"].to_numpy() == base["ts_code"].to_numpy()).all():
            raise RuntimeError("集成成员的分数行对不上")
    for j in range(3):
        r = [p.groupby("trade_date")[f"s{j}"].rank(pct=True).to_numpy() for p in parts]
        base[f"s{j}"] = np.mean(r, axis=0).astype(np.float32)
    base["score"] = base[["s0", "s1", "s2"]].mean(axis=1).astype(np.float32)
    return base


def consistency(keys: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mid in ("xgb_F69", "xgb_F28", "xgb_repro_F28", "rf_F69"):
        h = load_scores(mid)
        f = pd.read_parquet(data_dir() / "scores" / f"{mid}__final.parquet", columns=["trade_date", "ts_code", "score"])
        m = f.merge(h[["trade_date", "ts_code", "score"]], on=["trade_date", "ts_code"], suffixes=("_正式", "_长期"))
        rows.append({"模型": NAMES[mid], "对上的行数": len(m), "测试段正式分数行数": len(f),
                     "最大绝对差": float(np.abs(m["score_正式"] - m["score_长期"]).max())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 逐年 IC


def daily_ic(keys: pd.DataFrame, scores: pd.DataFrame) -> pd.Series:
    """只在有有效收益的行上算逐日 Rank IC；算两遍，一致才用（测试段出现过偶发计算错误）。"""
    k = keys[keys["ret_o1o5"].notna()]
    m = k.merge(scores[["trade_date", "ts_code", "score"]], on=["trade_date", "ts_code"])
    kv = m[keys.columns].reset_index(drop=True)
    a = ev.daily_rank_ic(ev._frame(m["score"].to_numpy(), kv))
    b = ev.daily_rank_ic(ev._frame(m["score"].to_numpy(), kv))
    if not a.equals(b):
        raise RuntimeError("逐日 IC 两次计算不一致")
    return a


def ic_tables(keys: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ics = {mid: daily_ic(keys, load_scores(mid)) for mid in MODELS}
    yearly = pd.DataFrame({NAMES[m]: s.groupby(s.index.year).mean() for m, s in ics.items()}).round(4)
    seg_rows = []
    for a, b, label in (("xgb_F69", "xgb_F28", "XGBoost F69 − F28（新特征）"), ("xgb_F69", "xgb_repro_F28", "XGBoost F69 − 作者原模型"),
                        ("rf_F69", "xgb_F69", "随机森林 F69 − XGBoost F69"), ("xgb_F69_s5", "xgb_F69", "敏感性：第 5 组参数 − 选定参数"),
                        ("xgb_F69_s4", "xgb_F69", "敏感性：第 4 组参数 − 选定参数"), ("ens_top3", "xgb_F69", "集成 top3 − XGBoost F69")):
        row = {"比较": label}
        for seg, (y0, y1) in SEGMENTS.items():
            da, db = ics[a], ics[b]
            sel = (da.index.year >= y0) & (da.index.year <= y1)
            c = ev.compare(da[sel], db.reindex(da.index)[sel])
            row[seg] = f"{c['diff']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]"
        seg_rows.append(row)
    return yearly, pd.DataFrame(seg_rows), ics


# --------------------------------------------------------------------------- 回测


def _names_at_time(codes: pd.Series, when: pd.Series) -> np.ndarray:
    """交易当时的股票名称（namechange 表：start_date 起生效），找不到就用当前名称。"""
    with connect() as con:
        nc = con.execute("SELECT ts_code, name, start_date FROM namechange WHERE start_date IS NOT NULL").df()
        cur = con.execute("SELECT ts_code, name FROM stock_basic").df().set_index("ts_code")["name"]
    nc["start_date"] = pd.to_datetime(nc["start_date"]).astype("datetime64[ns]")          # 两边统一成纳秒精度，merge_asof 要求一致
    q = pd.DataFrame({"ts_code": codes.to_numpy(), "when": pd.to_datetime(when.to_numpy()).astype("datetime64[ns]"), "i": np.arange(len(codes))})
    m = pd.merge_asof(q.sort_values("when"), nc.sort_values("start_date"), left_on="when", right_on="start_date", by="ts_code",
                      direction="backward").sort_values("i")
    return m["name"].fillna(m["ts_code"].map(cur)).fillna("").to_numpy()


def backtests(keys: pd.DataFrame, models: list[str] | None = None, extras: bool = True):
    """models 默认 MODELS；extras=False 时不做作者对照和候选名单修正核对（交付版 P8 用），对应的返回值是 None。"""
    models = models or MODELS
    from rebound.backtest import stats
    from rebound.backtest.costs import CostModel
    from rebound.backtest.market import candidates, load_markets, pool_daily_return
    from rebound.backtest.rules import rule_of
    from rebound.backtest.sim import simulate
    from ml.p5_backtest import to_market_index
    m = load_markets(DB_PATH, start=date(2014, 1, 1))["nocap"]
    pool, dates = pool_daily_return(m), m.dates
    d_first = int(dates.searchsorted(pd.Timestamp(f"{START_YEAR}-01-01")))
    d_last = int(dates.get_indexer([pd.Timestamp(LAST_SIGNAL)])[0])
    idx = stats.scoring_days(d_first, d_last)
    with connect() as con:
        ind = con.execute("SELECT basic.industry, basic.ts_code FROM stock_basic basic").df().set_index("ts_code")["industry"]
        idxp = con.execute("SELECT ts_code, trade_date, close FROM index_daily WHERE trade_date >= '2014-01-01'").df()
    idxp["trade_date"] = pd.to_datetime(idxp["trade_date"])
    index_close = {n: idxp[idxp["ts_code"] == c].set_index("trade_date")["close"].reindex(dates).ffill() for n, c in INDICES.items()}

    def seg_idx(y0, y1):
        yy = dates[idx].year
        return idx[(yy >= y0) & (yy <= y1)]

    navs, rows, years_rows, trades_all, cands_cache = {}, [], [], {}, {}
    for mid in models:
        sc = load_scores(mid)
        sc = sc[sc["trade_date"] <= pd.Timestamp(LAST_SIGNAL)]
        cands_cache[mid] = candidates(m, to_market_index(m, sc, "score", mid))
        for rule, capital in RULES.items():
            cost = CostModel(capital=float(capital), stamp_before=0.001)
            cands, rank = cands_cache[mid]
            res = simulate(m, cands, rank, rule_of(rule), cost, d_first, d_last)
            summ = stats.summary(res, pool, dates, idx)
            navs[(mid, rule)] = res.nav
            row = {"model_id": mid, "模型": NAMES[mid], "规则": rule, "本金": capital, "换手（倍/年）": summ["turnover"],
                   "费用占比（%/年）": summ["cost_pct"], "平均持有天数": summ["hold_days"], "平均持股": summ["avg_holdings"]}
            for seg, (y0, y1) in SEGMENTS.items():
                per = stats.period(res.nav, pool, dates, seg_idx(y0, y1))
                row.update({f"{seg}｜年化%": per["cagr"], f"{seg}｜超额%": per["excess"], f"{seg}｜超额区间": per["excess_ci"],
                            f"{seg}｜最大回撤%": per["max_dd"], f"{seg}｜β": per["beta"]})
            rows.append(row)
            for y, v in stats.by_year(res.nav, pool, dates, idx).items():
                years_rows.append({"model_id": mid, "规则": rule, "年": int(y), "收益%": v["total"], "超额%": v["excess"],
                                   "最大回撤%": v["max_dd"], "β": v["beta"]})
            t = res.trades.copy()
            t["ts_code"] = m.codes[t["j"].to_numpy()]
            for c in ("signal_day", "entry_day", "exit_day"):
                t[c.replace("_day", "日")] = dates[t[c].to_numpy()]
            t["买入时排名"] = rank[t["signal_day"].to_numpy(), t["j"].to_numpy()]
            t["持有天数"] = t["exit_day"] - t["entry_day"]
            trades_all[(mid, rule)] = t
            print(f"[{rule}] {mid}：全期年化 {row['全期 2015 ~ 2026｜年化%']:+.1f}%，超额 {row['全期 2015 ~ 2026｜超额%']:+.1f}%", flush=True)
    author_ours = author_full = fix_df = None
    if extras:
        author_ours, author_full, fix_df = _extras(m, cands_cache, pool, dates, idx, d_first, d_last)
    bench = {"准入池等权": np.r_[1.0, np.cumprod(1 + np.nan_to_num(pool[idx]))]}
    for n, s in index_close.items():
        v = s.to_numpy()
        bench[n] = v[idx[0] - 1:idx[-1] + 1] / v[idx[0] - 1]
    span = dates[idx[0] - 1:idx[-1] + 1]
    nav_tabs = {}
    for rule in RULES:
        t = pd.DataFrame({mid: navs[(mid, rule)][idx[0] - 1:idx[-1] + 1] / navs[(mid, rule)][idx[0] - 1] for mid in models}, index=span)
        for n, v in bench.items():
            t[n] = v
        t.index.name = "date"
        nav_tabs[rule] = t
    return (pd.DataFrame(rows), pd.DataFrame(years_rows), nav_tabs, trades_all, author_ours, author_full,
            fix_df, ind, cands_cache, m)


def _extras(m, cands_cache, pool, dates, idx, d_first, d_last):
    """作者十年实验对照（作者原模型 × 作者当时的规则）和候选名单修正前后核对（2023 ~ 2026）。"""
    from rebound.backtest import stats
    from rebound.backtest.costs import CostModel
    from rebound.backtest.rules import rule_of
    from rebound.backtest.sim import simulate
    # 作者十年实验对照：作者原模型 × 作者当时的规则
    res = simulate(m, *cands_cache["xgb_repro_F28"], rule_of(AUTHOR_RULE), CostModel(capital=200_000.0, stamp_before=0.001), d_first, d_last)
    author_ours = {int(y): v for y, v in stats.by_year(res.nav, pool, dates, idx).items()}
    author_full = stats.period(res.nav, pool, dates, idx)
    # 候选名单修正前后（2023 ~ 2026，主规则，印花税口径与 P5 基准相同）
    d23 = int(dates.searchsorted(pd.Timestamp("2023-01-01")))
    idx23 = stats.scoring_days(d23, d_last)
    p5 = pd.DataFrame(json.loads((RESULTS / "p5_backtest" / "results.json").read_text(encoding="utf-8")))
    p5 = p5[(p5["规则"] == MAIN) & (p5["口径"] == "基准")].set_index("model_id")
    fix_rows = []
    for mid in ("xgb_F69", "xgb_F28", "xgb_repro_F28", "rf_F69", "ens_top3"):
        r23 = simulate(m, *cands_cache[mid], rule_of(MAIN), CostModel(capital=200_000.0), d23, d_last)
        per = stats.period(r23.nav, pool, dates, idx23)
        fix_rows.append({"模型": NAMES[mid], "P5 原来（只用有有效收益的候选）年化超额%": p5.loc[mid, "全段 超额%"],
                         "修正后（全部股票都是候选）年化超额%": per["excess"], "差（百分点）": round(per["excess"] - p5.loc[mid, "全段 超额%"], 2),
                         "P5 原来最大回撤%": p5.loc[mid, "全段 最大回撤%"], "修正后最大回撤%": per["max_dd"]})
    return author_ours, author_full, pd.DataFrame(fix_rows)


def author_table(ours: dict, full: dict) -> pd.DataFrame:
    """读作者 experiments/decade/REPORT.md 的“按年”一栏（FH-K6-next-o1o5，按年重训），和我们的逐年结果并排。"""
    text = (DB_PATH.parents[1] / "experiments" / "decade" / "REPORT.md").read_text(encoding="utf-8")
    sec = text[text.index("## FH-K6-next-o1o5"):text.index("## FH-K2-next-o1o5-small")]
    rows = []
    for line in sec.splitlines():
        mm = re.match(r"\|\s*(\d{4}|全期年化)\s*\|\s*([+-][\d.]+)\s*\|\s*([+-][\d.]+)\s*/\s*([+-][\d.]+)\s*\|", line)
        if not mm:
            continue
        y = mm.group(1)
        if y == "全期年化":
            o_tot, o_ex = full["cagr"], full["excess"]
        else:
            v = ours.get(int(y))
            o_tot, o_ex = (v["total"], v["excess"]) if v else (np.nan, np.nan)
        rows.append({"年份": y, "准入池（作者）": float(mm.group(2)), "作者：收益%": float(mm.group(3)), "作者：超额%": float(mm.group(4)),
                     "我们：收益%": o_tot, "我们：超额%": o_ex})
    t = pd.DataFrame(rows)
    t["收益差"] = (t["我们：收益%"] - t["作者：收益%"]).round(1)
    t["超额差"] = (t["我们：超额%"] - t["作者：超额%"]).round(1)
    return t


# --------------------------------------------------------------------------- 明细与统计


def export_details(trades_all, ind, cands_cache, m, nav_tabs) -> None:
    from rebound.select import dedup_top
    hd = hist_dir()
    for (mid, rule), t in trades_all.items():
        out = pd.DataFrame({"信号日": t["signal日"], "买入日": t["entry日"], "卖出日": t["exit日"], "ts_code": t["ts_code"],
                            "名称（当时）": _names_at_time(t["ts_code"], t["entry日"]), "行业": t["ts_code"].map(ind).fillna("").to_numpy(),
                            "买入时排名": t["买入时排名"], "买入价（复权）": t["entry_px"], "卖出价（复权）": t["exit_px"],
                            "收益（扣费前）": t["gross"], "收益（扣费后）": t["net"], "持有天数": t["持有天数"],
                            "被卡住（跌停 / 停牌晚卖）": t["carried"]})
        out.to_parquet(hd / f"trades_{mid}__{rule}.parquet", index=False)
        # 每日持仓：买入日起、卖出日前一天止每天收盘后都持有（卖出日当天卖掉）
        n = np.maximum((t["exit_day"] - t["entry_day"]).to_numpy(), 1)
        rep = np.repeat(np.arange(len(t)), n)
        dayi = t["entry_day"].to_numpy()[rep] + (np.arange(len(rep)) - np.repeat(np.cumsum(n) - n, n))
        pd.DataFrame({"date": m.dates[dayi], "ts_code": out["ts_code"].to_numpy()[rep], "名称（当时）": out["名称（当时）"].to_numpy()[rep],
                      "行业": out["行业"].to_numpy()[rep]}).to_parquet(hd / f"holdings_{mid}__{rule}.parquet", index=False)
    for mid, (cands, rank) in cands_cache.items():
        rows = []
        for d, js in cands.items():
            codes = m.codes[js]
            inds = [ind.get(c, "") or "" for c in codes]
            keep = dedup_top(inds, max_per_industry=2, top_n=30)
            for r, i in enumerate(keep, 1):
                rows.append((m.dates[d], r, int(rank[d, js[i]]), codes[i], inds[i]))
        pd.DataFrame(rows, columns=["date", "名单排名", "全池排名", "ts_code", "行业"]).to_parquet(hd / f"picks_{mid}.parquet", index=False)


def holding_stats(mid: str, rule: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t = pd.read_parquet(hist_dir() / f"trades_{mid}__{rule}.parquet")
    t["年"] = pd.to_datetime(t["买入日"]).dt.year
    by_year = t.groupby("年").agg(交易笔数=("ts_code", "size"), 平均持有天数=("持有天数", "mean"),
                                 胜率=("收益（扣费后）", lambda s: (s > 0).mean() * 100), 平均单笔收益=("收益（扣费后）", lambda s: s.mean() * 100),
                                 被卡住笔数=("被卡住（跌停 / 停牌晚卖）", "sum")).round(2)
    h = pd.read_parquet(hist_dir() / f"holdings_{mid}__{rule}.parquet")
    h["年"] = pd.to_datetime(h["date"]).dt.year
    top = h.groupby(["ts_code"]).agg(名称=("名称（当时）", "last"), 行业=("行业", "first"), 持有天数=("date", "size")).sort_values("持有天数", ascending=False).head(15)
    share = h.groupby(["年", "行业"]).size().groupby(level=0, group_keys=False).apply(lambda s: s / s.sum() * 100)
    ind_top = share.groupby(level=0, group_keys=False).nlargest(3).reset_index()
    ind_top["行业（占比%）"] = ind_top["行业"] + "（" + ind_top[0].round(1).astype(str) + "）"
    ind_year = ind_top.groupby("年")["行业（占比%）"].apply("、".join).to_frame()
    return by_year, top, ind_year


def stress_table(nav: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for name, (a, b) in STRESS.items():
        seg = nav.loc[:pd.Timestamp(b)]
        before = seg.loc[:pd.Timestamp(a) - pd.Timedelta(days=1)]
        if before.empty:
            continue
        base = before.iloc[-1]
        win = seg.loc[pd.Timestamp(a):]
        row = {"时期": f"{name}（{a} ~ {b}）"}
        for c in cols:
            path = pd.concat([pd.Series([base[c]], index=[before.index[-1]]), win[c]])
            ret = path.iloc[-1] / path.iloc[0] - 1
            dd = (path / path.cummax() - 1).min()
            trough = (path / path.cummax() - 1).idxmin()
            peak = path.loc[:trough].max()
            after = nav[c].loc[trough:]
            rec = after[after >= peak]
            days = int(nav.index.get_indexer([rec.index[0]])[0] - nav.index.get_indexer([trough])[0]) if len(rec) else None
            row[f"{c}｜区间收益%"] = round(ret * 100, 1)
            row[f"{c}｜区间最大回撤%"] = round(dd * 100, 1)
            row[f"{c}｜回到前高（交易日）"] = days if days is not None else "未回到"
        rows.append(row)
    return pd.DataFrame(rows)


def monthly(nav: pd.DataFrame) -> pd.DataFrame:
    me = nav.groupby(nav.index.to_period("M")).last()
    first = nav.iloc[0]
    r = me.pct_change()
    r.iloc[0] = me.iloc[0] / first - 1
    return (r * 100).round(2)


# --------------------------------------------------------------------------- 图


def figures(out, nav_tabs, ics, years_df) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    nav = nav_tabs[MAIN]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    for c, st in (("xgb_F69", "-"), ("ens_top3", ":"), ("rf_F69", "-"), ("xgb_F28", "-"), ("xgb_repro_F28", "--"),
                  ("准入池等权", "k:"), ("中证1000", "-")):
        ax.plot(nav.index, nav[c], st, lw=1.6 if c == "xgb_F69" else 1.0, label=NAMES.get(c, c), color="#888" if c == "中证1000" else None)
    ax.set_yscale("log")
    for a, b, col in (("2021-01-01", "2022-12-31", "#fff3cd"), ("2023-01-01", LAST_SIGNAL, "#d4edda")):
        for x in (ax, ax2):
            x.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=col, alpha=0.6, lw=0)
    ax.set_ylabel("净值（对数刻度，2015 年初 = 1）")
    ax.set_title(f"长期回测：{MAIN}，20 万，印花税按历史税率（黄 = 挑参数的年份，绿 = 样本外；之前仅供参考）")
    ax.legend(fontsize=8, ncol=3)
    for c in ("xgb_F69", "准入池等权"):
        ax2.plot(nav.index, (nav[c] / nav[c].cummax() - 1) * 100, lw=1, label=NAMES.get(c, c))
    ax2.set_ylabel("回撤（%）")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_nav_drawdown.png", dpi=130)
    plt.close(fig)

    y = years_df[(years_df["规则"] == MAIN)].pivot(index="年", columns="model_id", values="超额%")
    cols = ["xgb_F69", "xgb_repro_F28", "rf_F69", "xgb_F28"]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    w = 0.2
    for k, c in enumerate(cols):
        ax.bar(y.index + (k - 1.5) * w, y[c], width=w, label=NAMES[c])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(y.index)
    ax.set_ylabel("当年年化超额（%，相对准入池等权）")
    ax.set_title(f"逐年超额（{MAIN}）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_yearly_excess.png", dpi=130)
    plt.close(fig)

    mo = monthly(nav[["xgb_F69", "准入池等权"]])
    for col, fname, title in (("xgb_F69", "fig3_monthly_return.png", "XGBoost F69 月度收益（%）"),):
        z = mo[col].to_frame("v")
        z["年"], z["月"] = z.index.year, z.index.month
        hm = z.pivot(index="年", columns="月", values="v")
        ex = (mo["xgb_F69"] - mo["准入池等权"]).to_frame("v")
        ex["年"], ex["月"] = ex.index.year, ex.index.month
        hx = ex.pivot(index="年", columns="月", values="v")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for a, h, t in ((axes[0], hm, title), (axes[1], hx, "XGBoost F69 月度超额（%，减准入池等权）")):
            lim = np.nanpercentile(np.abs(h.to_numpy()), 95)
            im = a.imshow(h.to_numpy(), cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
            a.set_xticks(range(12))
            a.set_xticklabels(range(1, 13))
            a.set_yticks(range(len(h.index)))
            a.set_yticklabels(h.index)
            for i in range(h.shape[0]):
                for j in range(h.shape[1]):
                    v = h.iat[i, j]
                    if np.isfinite(v):
                        a.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5)
            a.set_title(t)
            fig.colorbar(im, ax=a)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for mid in ("xgb_F69", "xgb_F69_s5", "xgb_F69_s4", "xgb_F28", "xgb_repro_F28", "rf_F69"):
        s = ics[mid]
        yy = s.groupby(s.index.year).mean()
        ax.plot(yy.index, yy.values, marker="o", ls="--" if "_s" in mid else "-", lw=1.4 if mid == "xgb_F69" else 1, label=NAMES[mid])
    ax.axvspan(2020.5, 2022.5, color="#fff3cd", alpha=0.6, lw=0)
    ax.axvspan(2022.5, 2026.5, color="#d4edda", alpha=0.6, lw=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("当年逐日 Rank IC 均值")
    ax.set_title("逐年 IC（黄 = 挑参数的年份，绿 = 样本外）")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig4_yearly_ic.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- 主流程


def analyze() -> None:
    out = result_dir(OUT)
    keys = ev.load_keys()
    cons = consistency(keys)
    print(cons.to_string(index=False), flush=True)
    yearly_ic, ic_cmp, ics = ic_tables(keys)
    print(yearly_ic.to_string(), flush=True)
    rows, years_df, nav_tabs, trades_all, a_ours, a_full, fix, ind, cands_cache, m = backtests(keys)
    export_details(trades_all, ind, cands_cache, m, nav_tabs)
    auth = author_table(a_ours, a_full)
    rows.to_json(out / "results.json", orient="records", force_ascii=False, indent=1)
    years_df.to_csv(out / "yearly.csv", index=False, encoding="utf-8-sig")
    yearly_ic.to_csv(out / "yearly_ic.csv", encoding="utf-8-sig")
    ic_cmp.to_csv(out / "ic_compare.csv", index=False, encoding="utf-8-sig")
    fix.to_csv(out / "fix_check.csv", index=False, encoding="utf-8-sig")
    auth.to_csv(out / "author_check.csv", index=False, encoding="utf-8-sig")
    cons.to_csv(out / "consistency.csv", index=False, encoding="utf-8-sig")
    for rule, t in nav_tabs.items():
        t.to_csv(out / f"nav_{rule}.csv", encoding="utf-8-sig")
        monthly(t).to_csv(out / f"monthly_{rule}.csv", encoding="utf-8-sig")
    stress = stress_table(nav_tabs[MAIN], ["xgb_F69", "xgb_repro_F28", "准入池等权", "中证1000", "沪深300"])
    stress.to_csv(out / "stress.csv", index=False, encoding="utf-8-sig")
    hs = {mid: holding_stats(mid, MAIN) for mid in ("xgb_F69", "xgb_repro_F28")}
    figures(out, nav_tabs, ics, years_df)
    json.dump({"stats": {mid: [v[0].reset_index().to_dict("records"), v[1].reset_index().to_dict("records"),
                               v[2].reset_index().to_dict("records")] for mid, v in hs.items()}},
              open(out / "holding_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    register("P7_history_analyze", fix_max_abs=float(fix["差（百分点）"].abs().max()))
    from ml.p7_history_report import report
    report()
