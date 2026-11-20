"""P8 补充：三个模型换起点的净值曲线和图（PREREG 补充登记十二、十三，描述性分析）。

`ml/p8_delivery.py` 里的换起点只存了汇总表（`delivery/long_starts.csv`）；这里用同样的分数、同样的模拟器和设置
（`ml/p7_starts.py`：2015 ~ 2024 每年第一个交易日拿 20 万起步，跑到 2026-10，印花税按历史税率）重跑一遍，把每条净值曲线存下来并画图。
运行时核对：每条曲线算出的年化必须和 `long_starts.csv` 一致，否则停止。

输出（`ml/results/delivery/`）：
- `long_starts_nav_<规则>.csv`：列名 `<模型>|<起步年>`（模型 = xgb_F69 / rf_F69 / lstm_SF69 / 准入池），每条曲线起步日 = 1，起步前为空；
- `fig5_starts_DB-K30-b8.png`、`fig6_starts_FH-K6-h4.png`：每个起点一格，对数坐标。

运行：python -X utf8 -m ml.p8_starts_nav          # 约 3 分钟（60 次模拟）
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ml.common import DB_PATH, result_dir
from ml.p7_history_analyze import LAST_SIGNAL, load_scores
from ml.p7_starts import RULES, STARTS
from ml.p8_delivery import M3, OUT

FIGS = {"DB-K30-b8": "fig5_starts_DB-K30-b8.png", "FH-K6-h4": "fig6_starts_FH-K6-h4.png"}
STYLE = {"xgb_F69": ("XGBoost", "#2a78d6", "-"), "rf_F69": ("随机森林", "#eb6834", "--"), "lstm_SF69": ("LSTM", "#1baf7a", ":")}
POOL = "准入池"


def run() -> dict[str, pd.DataFrame]:
    from rebound.backtest import stats
    from rebound.backtest.costs import CostModel
    from rebound.backtest.market import candidates, load_markets, pool_daily_return
    from rebound.backtest.rules import rule_of
    from rebound.backtest.sim import simulate
    from ml.p5_backtest import to_market_index
    m = load_markets(DB_PATH, start=date(2014, 1, 1))["nocap"]
    pool, dates = np.nan_to_num(np.asarray(pool_daily_return(m), float)), m.dates
    d_last = int(dates.get_indexer([pd.Timestamp(LAST_SIGNAL)])[0])
    d_end = len(dates) - 1
    ref = pd.read_csv(result_dir(OUT) / "long_starts.csv", encoding="utf-8-sig")
    cols: dict[str, dict[str, np.ndarray]] = {r: {} for r in RULES}
    for mid in M3:
        sc = load_scores(mid)
        cands, rank = candidates(m, to_market_index(m, sc[sc["trade_date"] <= pd.Timestamp(LAST_SIGNAL)], "score", mid))
        for rule, capital in RULES.items():
            for y in STARTS:
                d0 = int(dates.searchsorted(pd.Timestamp(f"{y}-01-01")))
                res = simulate(m, cands, rank, rule_of(rule), CostModel(capital=float(capital), stamp_before=0.001), d0, d_last)
                cagr = stats.window(res.nav, pool, d0, d_end, dates)["cagr"]
                want = ref[(ref["model_id"] == mid) & (ref["规则"] == rule) & (ref["起步年"] == y)]["年化%"].iloc[0]
                if abs(cagr - want) > 0.05:
                    raise RuntimeError(f"{mid} {rule} {y}：年化 {cagr:.2f}% 与 long_starts.csv 的 {want:.2f}% 不一致")
                nav = np.full(len(dates), np.nan)
                v = np.asarray(res.nav, float)[d0:]
                nav[d0:] = v / v[0]
                cols[rule][f"{mid}|{y}"] = nav
            print(f"{mid} {rule} 完成（年化与汇总表一致）", flush=True)
    out = {}
    for rule in RULES:
        for y in STARTS:
            d0 = int(dates.searchsorted(pd.Timestamp(f"{y}-01-01")))
            p = np.full(len(dates), np.nan)
            p[d0:] = np.concatenate([[1.0], np.cumprod(1 + pool[d0 + 1:])])
            cols[rule][f"{POOL}|{y}"] = p
        t = pd.DataFrame(cols[rule], index=pd.DatetimeIndex(dates, name="date"))
        out[rule] = t.loc[t.notna().any(axis=1)].round(4)
    return out


def figure(rule: str, t: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    ref = pd.read_csv(result_dir(OUT) / "long_starts.csv", encoding="utf-8-sig")
    ref = ref[ref["规则"] == rule]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.2))
    for ax, y in zip(axes.flat, STARTS):
        r = ref[ref["起步年"] == y].set_index("model_id")
        for mid in M3:
            nm, c, ls = STYLE[mid]
            s = t[f"{mid}|{y}"].dropna()
            ax.plot(s.index, s.to_numpy(), color=c, ls=ls, lw=2.2 if ls == ":" else 1.8,
                    label=f"{nm} 年化 {r.loc[mid, '年化%']:.1f}% 回撤 {r.loc[mid, '最大回撤%']:.0f}%")
        s = t[f"{POOL}|{y}"].dropna()
        ax.plot(s.index, s.to_numpy(), color="#888780", ls=(0, (4, 3)), lw=1, label=f"准入池 年化 {r['准入池同期年化%'].iloc[0]:.1f}%")
        ax.axhline(1, color="#c3c2b7", lw=0.6)
        ax.set_yscale("log")
        ax.set_title(f"{y} 年初起步（{r['持有年数'].iloc[0]:.1f} 年）", fontsize=12)
        lo, hi = ax.get_ylim()
        ax.set_yticks([v for v in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12) if lo <= v <= hi])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(mdates.YearLocator(2 if y < 2020 else 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y"))
        ax.grid(alpha=0.3, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7.6, loc="upper left", frameon=False, handlelength=2.2)
    fig.suptitle(f"换起点：{rule}（20 万本金）—— 2015 ~ 2024 每年第一个交易日起步，跑到 2026-10；净值对数坐标，起点 = 1", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    out = result_dir(OUT)
    for rule, t in run().items():
        t.to_csv(out / f"long_starts_nav_{rule}.csv", encoding="utf-8-sig")
        figure(rule, t, out / FIGS[rule])
        print(f"写出 {out / f'long_starts_nav_{rule}.csv'}、{FIGS[rule]}", flush=True)


if __name__ == "__main__":
    main()
