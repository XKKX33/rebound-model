"""数据与模型部分对外的只读接口（任务 6.2 定稿；6.1 早期版本的函数名、参数和返回列都没有变）。

给队友的智能体工作流和 UI 看板调用。**只读**：不训练、不写任何文件、研究库只读打开。说明见 `ml/INTERFACE.md`。

    from ml import api
    api.models()                                   # 有哪些模型（三个：xgb_F69、rf_F69、lstm_SF69）
    api.leaderboard()                              # 排行榜
    api.scores("xgb_F69", "2025-03-14")            # 某天全池的分数和排名
    api.picks("xgb_F69", "2025-03-14", top_n=6)    # 候选名单（每个行业最多 2 只）
    api.explain("xgb_F69", "2025-03-14", "600000.SH")   # 这只票为什么得这个分（测试段）
    api.nav("xgb_F69", "DB-K30-b8")                # 测试期组合净值；period="long" 是 2015 ~ 2026 长期回测
    api.feature_info()                             # 特征和序列通道的说明
    api.data_summary()                             # 数据概况

分数从哪来（全部是事先算好的，接口只是读）：
- 2021-01-04 ~ 2022-12-30：验证段（`select`），模型用 2013 ~ 2020 训练；
- 2023-01-03 ~ 2026-10-16：测试段（`final`），模型每年第一个交易日用此前全部数据重训（PREREG 补充登记十）。
两段都是“模型当时没见过的日子”，可以拿来展示历史名单和事后表现。

交付范围（PREREG 补充登记十三）：只有三个用 69 个特征的模型。要给数据之外的新日子打分，用 `python -m ml.predict`（不在本接口里）。
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from ml.common import RESULTS, connect, data_dir, load_config

__all__ = ["models", "leaderboard", "dates", "scores", "picks", "explain", "nav", "feature_info", "data_summary",
           "ApiError", "EXPLAINED", "RULES", "DEFAULT_MODEL"]


class ApiError(ValueError):
    """参数不对（模型不存在、日期没有分数等），信息是中文，可以直接展示给用户。"""


#: model_id → (显示名, 类别, 输入)。交付范围只有这三个（PREREG 补充登记十三）；其他模型的结果是过程记录，不经接口提供。
MODELS: dict[str, tuple[str, str, str]] = {
    "xgb_F69": ("XGBoost", "推荐默认展示", "69 个特征"),
    "rf_F69": ("随机森林", "模型", "69 个特征"),
    "lstm_SF69": ("LSTM", "模型", "60 天日线序列 + 69 个特征"),
}
DEFAULT_MODEL = "xgb_F69"
STAGES = ("select", "final")
STAGE_NAMES = {"select": "验证段（2021 ~ 2022）", "final": "测试段（2023 起，按年重训）"}

#: 仓库 28 个原特征的说明（定义取自 rebound/features.py 的计算代码；c = 复权收盘价）
F28_INFO = {
    "pct_chg": ("当日涨跌幅（%）", "交易所公布的当日涨跌幅"),
    "r1": ("1 日收益", "c / 前 1 日 c − 1"), "r5": ("5 日收益", "c / 前 5 日 c − 1"),
    "r10": ("10 日收益", "c / 前 10 日 c − 1"), "r20": ("20 日收益", "c / 前 20 日 c − 1"),
    "r60": ("60 日收益", "c / 前 60 日 c − 1"), "r120": ("120 日收益", "c / 前 120 日 c − 1"),
    "from_low60": ("距 60 日最低", "c / 60 日最低 c − 1"), "from_high60": ("距 60 日最高", "c / 60 日最高 c − 1"),
    "dev_ma5": ("偏离 5 日均线", "c / 5 日均价 − 1"), "dev_ma20": ("偏离 20 日均线", "c / 20 日均价 − 1"),
    "dev_ma60": ("偏离 60 日均线", "c / 60 日均价 − 1"), "ma20_gt_ma60": ("20 日线在 60 日线上方", "1 / 0"),
    "ma20_slope5": ("20 日线 5 日斜率", "20 日均价 / 5 天前的 20 日均价 − 1"),
    "ma60_slope10": ("60 日线 10 日斜率", "60 日均价 / 10 天前的 60 日均价 − 1"),
    "dd20": ("20 日回撤", "c / 20 日最高 c − 1"),
    "vol5": ("相对 5 日量能", "当日成交额 / 前 5 日平均成交额"), "vol20": ("相对 20 日量能", "当日成交额 / 前 20 日平均成交额"),
    "amp20": ("20 日平均振幅", "20 日平均 (最高 − 最低) / 收盘"),
    "launch10": ("10 日内大涨天数", "前 10 日中涨幅 ≥ 5% 或涨停的天数"),
    "up_days10": ("10 日内上涨天数", "前 10 日中上涨的天数"), "max_up20": ("20 日最大单日涨幅", "20 日内最大的当日涨跌幅（%）"),
    "limit_up_today": ("当日涨停", "收盘价等于涨停价为 1"), "limit_up20": ("20 日涨停次数", "20 日内收盘涨停的天数"),
    "dif_ratio": ("MACD DIF / 价格", "(12 日 EMA − 26 日 EMA) / c"),
    "hist_ratio": ("MACD 柱 / 价格", "2 × (DIF − DEA) / c"),
    "rank_r20": ("20 日收益截面排名", "当天全池 r20 的百分位"), "rank_r60": ("60 日收益截面排名", "当天全池 r60 的百分位"),
}
#: LSTM 的 11 个序列通道（ml/seq_data.py）
CHANNEL_INFO = {
    "ret": ("日收益", "ln(复权收盘 / 前一交易日复权收盘)"), "ovn": ("隔夜收益", "ln(复权开盘 / 前一交易日复权收盘)"),
    "intra": ("日内收益", "ln(收盘 / 开盘)"), "hl": ("日内振幅", "ln(最高 / 最低)"),
    "clv": ("收盘位置", "(2C − H − L) / (H − L)，一字板记 0"), "ln_amt": ("成交额（log）", "ln(成交额)"),
    "ln_turn": ("换手率（log）", "ln(自由流通换手率)"), "mf": ("资金净流入占比", "净流入额 / 成交额"),
    "big": ("大单净买占比", "(大单 + 特大单买入 − 卖出) / 成交额"), "limit": ("涨跌停", "收盘涨停 +0.5，跌停 −0.5，其余 0"),
    "traded": ("当天有交易", "有交易 1，停牌 / 未上市 / 已退市 0"),
}


def _check_model(model_id: str) -> None:
    if model_id not in MODELS:
        raise ApiError(f"没有模型 {model_id}；可选：{', '.join(MODELS)}")


def models() -> pd.DataFrame:
    """全部模型：model_id、名称、类别、输入，以及哪些阶段已经有分数。"""
    rows = []
    for mid, (name, kind, inp) in MODELS.items():
        have = [s for s in STAGES if _export("scores", f"{mid}__{s}.parquet") is not None
                or (data_dir() / "scores" / f"{mid}__{s}.parquet").exists()]
        rows.append({"model_id": mid, "名称": name, "类别": kind, "输入": inp, "已有分数": "、".join(STAGE_NAMES[s] for s in have)})
    return pd.DataFrame(rows)


def leaderboard() -> pd.DataFrame:
    """三个模型的排行榜（测试段正式数字，`ml/results/delivery/leaderboard.json`，由 `ml/p8_delivery.py` 生成）。

    小数口径：IC 是相关系数；超额、多空是小数（0.003 = 0.3%）。
    """
    path = RESULTS / "delivery" / "leaderboard.json"
    if path.exists():
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    rows = []
    for mid, (name, kind, inp) in MODELS.items():
        p = RESULTS / "p3_models" / mid / "select.json"
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))["summary"]["全部"]
            rows.append({"model_id": mid, "模型": name, "类别": kind, "特征集": inp,
                         "验证 IC（用于选择，偏乐观）": s["ic_mean"], "验证前30超额": s["top30"]})
    return pd.DataFrame(rows)


@lru_cache(maxsize=16)
def _export(*parts: str):
    """交接用的精简数据包（`ml/p6_export.py` 导出，解压到 data/ml/api_export/）里的文件；没有就返回 None，退回读原始文件。"""
    p = data_dir().joinpath("api_export", *parts)
    return p if p.exists() else None


def _scores(model_id: str, stage: str) -> pd.DataFrame:
    path = _export("scores", f"{model_id}__{stage}.parquet") or data_dir() / "scores" / f"{model_id}__{stage}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["trade_date", "ts_code", "score"])
    return pd.read_parquet(path, columns=["trade_date", "ts_code", "score"])


def dates(model_id: str) -> list[pd.Timestamp]:
    """这个模型有分数的全部交易日（验证段 + 测试段）。"""
    _check_model(model_id)
    out = [d for s in STAGES for d in _scores(model_id, s)["trade_date"].unique()]
    return sorted(pd.Timestamp(d) for d in out)


@lru_cache(maxsize=1)
def _stocks() -> pd.DataFrame:
    """股票名称和行业（研究库 stock_basic，只读；行业是当前口径，不随时间变化）。有数据包时读数据包，不需要研究库。"""
    if (p := _export("stocks.parquet")) is not None:
        return pd.read_parquet(p).set_index("ts_code")
    with connect() as con:
        return con.execute("SELECT ts_code, name, industry FROM stock_basic").df().set_index("ts_code")


@lru_cache(maxsize=1)
def _outcomes() -> pd.DataFrame:
    if (p := _export("outcomes.parquet")) is not None:
        return pd.read_parquet(p).set_index(["trade_date", "ts_code"])
    from ml.dataset import load_panel
    return load_panel(["trade_date", "ts_code", "ret_o1o5", "excess_o1o5"]).set_index(["trade_date", "ts_code"])


def scores(model_id: str, date, with_outcome: bool = False) -> pd.DataFrame:
    """某天全池的分数和排名，按分数从高到低（同分按代码）。

    列：ts_code、name、industry、score（模型原始分数，只用于排序，不同模型之间不可比）、rank（1 = 最好）、
    pct（全池百分位，越大越好，不同模型之间可比）。
    with_outcome=True 时多两列事后表现：ret_o1o5（T+1 开盘买、T+5 开盘卖的收益）、excess_o1o5（减去当天全池平均）。
    这是“当时不知道”的信息，只能用于复盘展示，不能用于选股。
    """
    _check_model(model_id)
    day = pd.Timestamp(date)
    frames = [_scores(model_id, s) for s in STAGES]
    sc = pd.concat([f[f["trade_date"] == day] for f in frames])
    if sc.empty:
        ds = dates(model_id)
        rng = f"{ds[0].date()} ~ {ds[-1].date()}" if ds else "（还没有）"
        raise ApiError(f"{model_id} 在 {day.date()} 没有分数（不是交易日，或不在有分数的区间 {rng} 内）")
    sc = sc.sort_values(["score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    sc["rank"] = np.arange(1, len(sc) + 1)
    sc["pct"] = 1 - (sc["rank"] - 1) / len(sc)
    info = _stocks().reindex(sc["ts_code"])
    sc["name"] = info["name"].to_numpy()
    sc["industry"] = info["industry"].fillna("").to_numpy()
    cols = ["ts_code", "name", "industry", "score", "rank", "pct"]
    if with_outcome:
        o = _outcomes().reindex(pd.MultiIndex.from_arrays([sc["trade_date"], sc["ts_code"]]))
        sc["ret_o1o5"], sc["excess_o1o5"] = o["ret_o1o5"].to_numpy(), o["excess_o1o5"].to_numpy()
        cols += ["ret_o1o5", "excess_o1o5"]
    return sc[cols]


def picks(model_id: str, date, top_n: int = 6, max_per_industry: int = 2, with_outcome: bool = False) -> pd.DataFrame:
    """候选名单：全池按分数从高到低，每个行业最多 max_per_industry 只，取前 top_n（行业去重复用仓库 `rebound.select.dedup_top`）。

    返回列同 `scores`。这是信号日 T 收盘后的名单，按仓库口径 T+1 开盘买入。
    """
    from rebound.select import dedup_top
    if top_n < 1 or max_per_industry < 1:
        raise ApiError("top_n 和 max_per_industry 至少为 1")
    pool = scores(model_id, date, with_outcome=with_outcome)
    keep = dedup_top(pool["industry"].tolist(), max_per_industry=max_per_industry, top_n=top_n)
    return pool.iloc[keep].reset_index(drop=True)


#: 预先算好的解释（测试段每行贡献最大的 8 项）：XGBoost 用 TreeSHAP（任务 4.4），随机森林用路径分解，LSTM 用积分梯度（P8）
EXPLAINED = {"xgb_F69": "xgb_F69.parquet", "rf_F69": "rf_F69.parquet", "lstm_SF69": "lstm_SF69.parquet"}
EXPLAIN_METHOD = {"xgb_F69": "TreeSHAP（精确）", "rf_F69": "路径分解（Saabas，900 棵树平均）",
                  "lstm_SF69": "积分梯度（基线 = 截面中位数的股票，3 个种子平均）"}
RULES = ("DB-K30-b8", "DB-K3-b16", "FH-K6-h4")


@lru_cache(maxsize=2)
def _explain_table(model_id: str) -> pd.DataFrame:
    path = _export(f"explain_{EXPLAINED[model_id]}") or data_dir() / "explain" / EXPLAINED[model_id]
    if not path.exists():
        raise ApiError(f"还没有 {model_id} 的解释数据（{path}）：先运行 python -m ml.p4_4_shap_stability / ml.p8_explain_rf / ml.p8_explain_lstm")
    return pd.read_parquet(path)


def explain(model_id: str, date, ts_code: str, top: int = 8) -> pd.DataFrame:
    """这只股票当天分数的解释：贡献最大的 top 项（按 |贡献| 从大到小，top 最多 8）。三个模型都有，只覆盖测试段（2023-01-03 ~ 2026-10-16）。

    列：特征、中文名、数值（模型看到的值）、贡献（正 = 让分数变高）、方向（“加分” / “减分”）。
    另在 `.attrs` 里给出 score（模型分数）、base（基线分数）和 method（解释方法）：score ≈ base + 全部项的贡献之和，
    这里只列出最大的 top 项，所以列出的贡献之和不一定等于 score − base。各模型的方法：
    - xgb_F69：TreeSHAP；数值是特征原值；
    - rf_F69：路径分解（每次分裂带来的分数变化记到该特征上）；数值是特征原值；
    - lstm_SF69：积分梯度；解释项除了特征，还有 11 个序列通道（`seq_<通道>`，60 天归因之和，数值是信号日当天的通道值），
      特征的数值是截面排名等 dense 预处理之后的值；score 来自同配置重训的模型，与 `scores()` 的正式分数一致（最大差 4e-8）。
    """
    _check_model(model_id)
    if model_id not in EXPLAINED:
        raise ApiError(f"{model_id} 没有预先算好的解释")
    t = _explain_table(model_id)
    day = pd.Timestamp(date)
    row = t[(t["trade_date"] == day) & (t["ts_code"] == ts_code)]
    if row.empty:
        raise ApiError(f"{model_id} 在 {day.date()} 没有 {ts_code} 的解释（不在测试段、不是交易日，或这只票当天不在准入池）")
    r = row.iloc[0]
    n = min(top, sum(1 for c in t.columns if c.startswith("f") and c[1:].isdigit()))
    out = pd.DataFrame([{"特征": str(r[f"f{k}"]), "中文名": _cn_name(str(r[f"f{k}"])), "数值": float(r[f"x{k}"]),
                         "贡献": float(r[f"v{k}"]), "方向": "加分" if r[f"v{k}"] > 0 else "减分"} for k in range(1, n + 1)])
    out.attrs.update({"score": float(r["score"]), "base": float(r["base"]), "method": EXPLAIN_METHOD[model_id], "model_id": model_id,
                      "date": str(day.date()),
                      "ts_code": ts_code})
    return out


def nav(model_id: str, rule: str = "DB-K30-b8", period: str = "test") -> pd.DataFrame:
    """组合净值曲线（仓库模拟器、真实费用、每年初重训的分数）。

    rule：`DB-K30-b8`（20 万，分散，主规则）、`DB-K3-b16`（2 万，集中）、`FH-K6-h4`（20 万，每天买前 6 名拿 4 天）。
    period：`test` = 测试段 2023-01 ~ 2026-10（任务 P5，正式结果，现行印花税）；
            `long` = 2015 ~ 2026 长期回测（补充分析，印花税按历史税率；2023 年以前仅供参考），多三列指数净值。
    返回列：date、nav（模型净值，起点 1）、pool_nav（准入池等权净值，不可交易、不扣费用，作对照）；long 另有 hs300、zz500、zz1000。
    """
    _check_model(model_id)
    if rule not in RULES:
        raise ApiError(f"没有规则 {rule}；可选：{', '.join(RULES)}")
    if period not in ("test", "long"):
        raise ApiError("period 只能是 test 或 long")
    if period == "long":
        path = RESULTS / "delivery" / f"long_nav_{rule}.csv"
        if not path.exists():
            raise ApiError(f"还没有长期回测结果（{path}）：先运行 python -m ml.p8_delivery")
        t = pd.read_csv(path, parse_dates=["date"])
        return pd.DataFrame({"date": t["date"], "nav": t[model_id], "pool_nav": t["准入池等权"], "hs300": t["沪深300"],
                             "zz500": t["中证500"], "zz1000": t["中证1000"]})
    path = RESULTS / "p5_backtest" / f"nav_{rule}.csv"
    if not path.exists():
        raise ApiError(f"还没有回测结果（{path}）：先运行 python -m ml.p5_backtest")
    t = pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame({"date": t["date"], "nav": t[model_id], "pool_nav": t["准入池等权"]})


def _cn_name(name: str) -> str:
    """解释项的中文名：特征、LSTM 序列通道（seq_<通道>）、缺失指示（na_<特征>）。"""
    cn = _feature_info().set_index("名称")["中文名"]
    if name.startswith("seq_"):
        return "序列：" + CHANNEL_INFO.get(name[4:], ("",))[0] + "（近 60 天）"
    if name.startswith("na_"):
        return "缺失：" + str(cn.get(name[3:], name[3:]))
    return str(cn.get(name, ""))


def feature_info() -> pd.DataFrame:
    """F69 的 69 个特征和 LSTM 的 11 个序列通道：名称、来源、分组、中文名、计算方法、预期方向（新特征有，原特征没有）。"""
    return _feature_info().copy()


@lru_cache(maxsize=1)
def _feature_info() -> pd.DataFrame:
    from ml.dataset import F28, f69
    from ml.features_extra import feature_table
    rows = [{"名称": f, "来源": "原特征（仓库 F28）", "分组": "仓库原特征", "中文名": F28_INFO[f][0], "计算": F28_INFO[f][1],
             "预期方向": "", "理由": ""} for f in F28]
    new = feature_table().set_index("特征")
    for f in f69()[len(F28):]:
        r = new.loc[f]
        rows.append({"名称": f, "来源": "新特征（1.4）", "分组": r["组"], "中文名": r["中文名"], "计算": r["计算"],
                     "预期方向": r["预期方向"], "理由": r["理由"]})
    rows += [{"名称": c, "来源": "LSTM 序列通道（1.6）", "分组": "日线序列", "中文名": v[0], "计算": v[1], "预期方向": "",
              "理由": ""} for c, v in CHANNEL_INFO.items()]
    return pd.DataFrame(rows)


def data_summary() -> dict:
    """数据概况：区间、股票数、样本数、时间划分、特征数。"""
    if (p := _export("data_summary.json")) is not None:
        return json.loads(p.read_text(encoding="utf-8"))
    from ml.dataset import load_panel
    k = load_panel(["trade_date", "ts_code", "ret_o1o5"])
    s = load_config()["split"]
    return {
        "研究库": "2013-01-04 ~ 2026-10-23 的 A 股日线（只用主板），只读",
        "样本": "主面板每行 = 一只股票在一个信号日；标签 = T+1 开盘买、T+5 开盘卖的收益（不限价准入池）",
        "信号日区间": f"{k['trade_date'].min().date()} ~ {k['trade_date'].max().date()}",
        "交易日数": int(k["trade_date"].nunique()),
        "股票数": int(k["ts_code"].nunique()),
        "样本行数": int(len(k)),
        "有效收益行数": int(k["ret_o1o5"].notna().sum()),
        "时间划分": {"训练": f"{s['train_start']} ~ {s['train_end']}", "验证": f"{s['valid_start']} ~ {s['valid_end']}",
                     "测试": f"{s['test_start']} ~ {s['test_end']}（每年第一个交易日重训）"},
        "特征": {"F28": "仓库原有 28 个量价特征", "F69": "F28 + 41 个新特征（换手、规模、资金流、市场状态等）",
                 "S": "LSTM 输入：过去 60 个交易日 × 11 个日线通道"},
    }
