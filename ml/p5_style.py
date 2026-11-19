"""风格归因（全面检查时补做，测试期）：模型的预测能力和组合超额里，有多少来自常见的风格，有多少是风格以外的选股能力。

研究库只有行情、换手、市值、资金流，没有股息率和估值，所以能控制的风格是 4 种（全部用信号日 T 已知的数据）：
- 市值：流通市值（log）；
- 低波动：60 日特质波动；
- 低换手：自由流通换手率；
- 短期反转：20 日收益。

一、预测层：每天在截面上把模型分数（排名）对 4 个风格（排名）做回归，取残差，算残差和收益的 Rank IC。
   残差 IC = 风格以外的选股能力；同时报告“只用 4 个风格拟合出的部分”的 IC。
二、组合层：主规则 DB-K30-b8 的每日组合超额（相对准入池等权）对 4 个风格因子收益和市场项回归：
   风格因子收益 = 前一天按该风格分 5 组、多头组（大市值 / 低波动 / 低换手 / 近期跌得多）减空头组的等权次日收益；
   市场项 = 准入池等权当日收益（组合 β < 1 的部分）。截距 × 242 = 扣掉风格后的年化 alpha，t 值用 Newey-West（滞后 10）。

运行：python -X utf8 -m ml.p5_style
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import RESULTS, data_dir, register, result_dir, write_md
from ml.dataset import load_panel

OUT = "p5_style"
STYLES = {"ln_circ_mv": "市值（大）", "ivol60": "特质波动（低）", "turn_f": "换手（低）", "r20": "20 日收益（低 = 反转）"}
SIGN = {"ln_circ_mv": 1, "ivol60": -1, "turn_f": -1, "r20": -1}      # 多头方向：大市值、低波动、低换手、近期跌得多
MODELS = {"xgb_repro_F28": "仓库原模型 F28", "xgb_F28": "XGBoost F28", "xgb_F69": "XGBoost F69", "rf_F69": "随机森林 F69",
          "lstm_S": "LSTM-S", "lstm_SF69": "LSTM-S+F69", "ens_family": "集成 family"}


def _panel() -> pd.DataFrame:
    keys = ev.load_keys()
    p = load_panel(["trade_date", "ts_code", "r1", "r20"])
    fx = pd.read_parquet(data_dir() / "features_extra.parquet", columns=["trade_date", "ts_code", "ln_circ_mv", "ivol60", "turn_f"])
    d = keys[["trade_date", "ts_code", "di", "ret_o1o5"]].merge(p, on=["trade_date", "ts_code"]).merge(fx, on=["trade_date", "ts_code"])
    return d


def _neutral(score: np.ndarray, X: np.ndarray, day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐日截面回归：score ~ 1 + X（都已是当天排名），返回 (残差, 拟合值)。"""
    resid, fit = np.empty_like(score), np.empty_like(score)
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(day)]):
        A = np.column_stack([np.ones(b - a), X[a:b]])
        beta, *_ = np.linalg.lstsq(A, score[a:b], rcond=None)
        fit[a:b] = A @ beta
        resid[a:b] = score[a:b] - fit[a:b]
    return resid, fit


def predictive(d: pd.DataFrame) -> pd.DataFrame:
    rows, daily = [], {}
    k = pd.read_parquet(data_dir() / "scores" / "xgb_F69__final.parquet", columns=["trade_date", "ts_code"])
    base = k.merge(d, on=["trade_date", "ts_code"], how="left")
    g = base.groupby("trade_date")
    X = np.column_stack([g[c].rank(pct=True).fillna(0.5).to_numpy() for c in STYLES])
    day = base["di"].to_numpy()
    kv = base[["trade_date", "ts_code", "ret_o1o5"]].copy()
    kv["excess_o1o5"] = kv["ret_o1o5"] - kv.groupby("trade_date")["ret_o1o5"].transform("mean")
    kv["label_pct"] = kv.groupby("trade_date")["ret_o1o5"].rank(pct=True)
    for mid, name in MODELS.items():
        sc = pd.read_parquet(data_dir() / "scores" / f"{mid}__final.parquet", columns=["trade_date", "ts_code", "score"])
        if not (sc["ts_code"].to_numpy() == base["ts_code"].to_numpy()).all():
            raise RuntimeError(f"{mid}：分数行和面板对不上")
        s = sc.groupby("trade_date")["score"].rank(pct=True).to_numpy()
        resid, fit = _neutral(s, X, day)
        ic_raw = ev.daily_rank_ic(ev._frame(s, kv))
        ic_res = ev.daily_rank_ic(ev._frame(resid, kv))
        ic_fit = ev.daily_rank_ic(ev._frame(fit, kv))
        r2 = pd.Series(1 - pd.Series(resid).groupby(day).var().to_numpy() / pd.Series(s).groupby(day).var().to_numpy()).mean()
        daily[mid] = ic_res
        rows.append({"模型": name, "测试 IC（原始）": ic_raw.mean(), "风格能解释的部分的 IC": ic_fit.mean(),
                     "去掉风格后的 IC": ic_res.mean(), "去风格 IC 的 NW t": ev.newey_west_t(ic_res),
                     "分数被风格解释的比例（R²）": r2})
    tab = pd.DataFrame(rows)
    c = ev.compare(daily["xgb_F69"], daily["xgb_F28"])
    tab.attrs["rq3"] = c
    c2 = ev.compare(daily["xgb_F69"], daily["xgb_repro_F28"])
    tab.attrs["vs_repro"] = c2
    # 4 个风格合起来本身的 IC（作参照：一个只按风格打分的“模型”）
    style_score = (X * np.array([SIGN[c] for c in STYLES])).sum(axis=1)
    tab.attrs["style_only_ic"] = float(ev.daily_rank_ic(ev._frame(style_score, kv)).mean())
    return tab


def factor_returns(d: pd.DataFrame) -> pd.DataFrame:
    """风格因子的日收益：前一天按风格分 5 组，多头组减空头组的等权次日收益（准入池内）。"""
    d = d.sort_values(["ts_code", "di"])
    d["r1_next"] = d.groupby("ts_code")["r1"].shift(-1)
    d["di_next"] = d.groupby("ts_code")["di"].shift(-1)
    d = d[d["di_next"] == d["di"] + 1]                       # 只用次日仍在准入池里的票
    days = d[["di", "trade_date"]].drop_duplicates().set_index("di")["trade_date"]
    out = {}
    for c in STYLES:
        q = d.groupby("di")[c].rank(pct=True) * SIGN[c]
        hi = d[q >= (0.8 if SIGN[c] > 0 else -0.2)].groupby("di")["r1_next"].mean()
        lo = d[q <= (0.2 if SIGN[c] > 0 else -0.8)].groupby("di")["r1_next"].mean()
        out[STYLES[c]] = hi - lo
    f = pd.DataFrame(out)
    f.index = days.reindex(f.index + 1).to_numpy()             # 收益发生在次日
    return f.dropna()


def portfolio(f: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.api as sm
    nav = pd.read_csv(RESULTS / "p5_backtest" / "nav_DB-K30-b8.csv", index_col=0, parse_dates=True)
    r = nav.pct_change().iloc[1:]
    rows = []
    for mid, name in MODELS.items():
        y = (r[mid] - r["准入池等权"]).rename("y")
        X = pd.concat([r["准入池等权"].rename("市场（准入池）"), f], axis=1)
        df = pd.concat([y, X], axis=1, join="inner").dropna()
        res = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
        row = {"模型": name, "年化超额（原始，%）": df["y"].mean() * 242 * 100,
               "扣风格后的年化 alpha（%）": res.params["const"] * 242 * 100, "alpha 的 t": res.tvalues["const"], "R²": res.rsquared}
        for c in X.columns:
            row[f"暴露：{c}"] = res.params[c]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    d = _panel()
    pred = predictive(d)
    f = factor_returns(d)
    port = portfolio(f)
    out = result_dir(OUT)
    pred.to_csv(out / "predictive.csv", index=False, encoding="utf-8-sig")
    port.to_csv(out / "portfolio.csv", index=False, encoding="utf-8-sig")
    fy = f.groupby(f.index.year).sum().mul(100).round(1)
    rq3, vr = pred.attrs["rq3"], pred.attrs["vs_repro"]
    register("style_attribution", style_only_ic=pred.attrs["style_only_ic"])
    notes_path = out / "notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "（读法待补）"
    write_md(out / "report.md", f"""
# 风格归因（测试期 2023-01 ~ 2026-10，全面检查时补做）

- 代码：`ml/p5_style.py`；结果 `predictive.csv`、`portfolio.csv`
- 能控制的风格：市值、特质波动、换手、20 日收益（研究库没有股息率和估值，**“高股息 / 低估值”控制不了**）
- 不改动任何模型，只是把已有的测试段分数和组合收益拆开看

## 一、预测层：去掉风格后还剩多少 IC

{pred.round(4).to_markdown(index=False)}

- 只按这 4 个风格打分（大市值、低波动、低换手、近期跌得多，等权）的 IC：{pred.attrs['style_only_ic']:.4f}
- 去风格后 XGBoost F69 − F28（RQ3）：{rq3['diff']:+.4f}，95% 区间 [{rq3['ci'][0]:+.4f}, {rq3['ci'][1]:+.4f}]
- 去风格后 XGBoost F69 − 仓库原模型：{vr['diff']:+.4f}，95% 区间 [{vr['ci'][0]:+.4f}, {vr['ci'][1]:+.4f}]

## 二、组合层：主规则 DB-K30-b8 的超额对风格因子回归

{port.round(3).to_markdown(index=False)}

**风格因子各年累计收益（%，多头组 − 空头组，日收益相加）**

{fy.to_markdown()}

## 三、读法

{notes}
""")
    print(pred.round(4).to_string(index=False))
    print(port.round(3).to_string(index=False))
    print(fy)
    print("style_only", pred.attrs["style_only_ic"], "rq3", rq3, "vs_repro", vr)


if __name__ == "__main__":
    main()
