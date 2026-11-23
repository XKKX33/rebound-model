"""Real feature slices and archived experiments; demonstration data is marked explicitly."""
from __future__ import annotations
import json
import math
from functools import lru_cache
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
from rebound.store import Store
from rebound.debate import runs as debate_runs

from paths import PROJECT, DATABASE as BASE, MODEL_REPO, MODEL_ARTIFACTS, PROFILE, FEATURE_CACHE, FEATURE_STAMP
MODEL_IDS = ("xgb_F69", "rf_F69", "lstm_SF69")
MODEL_LABELS = {
    "xgb_F69": "XGBoost F69",
    "rf_F69": "随机森林 F69",
    "lstm_SF69": "LSTM · 69 特征 + 60×11 序列",
}
DELIVERY_MODEL_LABELS = {
    "xgb_F69": "XGBoost F69",
    "rf_F69": "随机森林 F69",
    "lstm_SF69": "LSTM S+F69",
}
STRATEGIES = {
    "DB-K30-b8": {"label": "排名缓冲 · 30只", "description": "最多持有30只，排名跌出前240名后换仓。", "capital": 200000},
    "DB-K3-b16": {"label": "集中持仓 · 3只", "description": "最多持有3只，排名缓冲换仓。", "capital": 20000},
    "FH-K6-h4": {"label": "分批持有 · 6只×4日", "description": "每日买入排名靠前的6只，按持有周期分批退出。", "capital": 200000},
}
MODEL_PERIODS = {"test": "测试期（2023—2026）", "long": "长期回测（2015—2026）"}
FEATURE_INFO = [
    ("r20", "20日收益", "动量", "%", "复权收盘价 / 20根前复权收盘价 − 1"),
    ("dev_ma20", "20日均线偏离", "趋势", "%", "复权收盘价 / 20根复权收盘均值 − 1"),
    ("vol20", "20日成交额比", "活跃度", "倍", "当日成交额 / 前20根成交额均值（不含当日）"),
    ("dd20", "20日高点回落", "回撤", "%", "复权收盘价 / 20根复权收盘最高值 − 1"),
    ("from_low60", "距60日低点", "位置", "%", "复权收盘价 / 60根复权收盘最低值 − 1"),
    ("amp20", "20日平均振幅", "波动", "%", "20根内每日（最高价 − 最低价）/ 收盘价的均值"),
]


@lru_cache(maxsize=1)
def model_api():
    """Read the delivered data package in place; keep the member clone read-only."""
    if not (MODEL_REPO / "ml" / "api.py").is_file():
        raise RuntimeError("本地克隆缺少 ml.api，请先在 GitHub Desktop 拉取 main")
    if not (MODEL_ARTIFACTS / "api_export" / "manifest.json").is_file():
        raise RuntimeError("未找到特征与模型/api_export 数据包")
    import sys

    repo = str(MODEL_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import ml.common as common

    common.DATA_OUT = MODEL_ARTIFACTS
    common.DB_PATH = BASE
    from ml import api

    return api


@lru_cache(maxsize=1)
def model_metadata():
    try:
        api = model_api()
        dates = api.dates("xgb_F69")
        return {
            "available": True,
            "first_score_date": str(min(dates).date()),
            "last_score_date": str(max(dates).date()),
            "models": list(MODEL_IDS),
            "message": "模型评分与归因读取自本地交付数据包。",
        }
    except Exception as exc:
        return {"available": False, "message": str(exc)}


def _records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=8))


def _model_candidates(as_of, model_id, market, *, limit=20):
    api = model_api()
    scores = api.scores(model_id, as_of).head(limit)
    prices = market.set_index("ts_code")
    peer_scores = {model_id: scores.set_index("ts_code")}
    for peer_id in MODEL_IDS:
        if peer_id != model_id:
            peer_scores[peer_id] = api.scores(peer_id, as_of).set_index("ts_code")
    candidates = []
    for row in scores.itertuples(index=False):
        code = str(row.ts_code)
        if code not in prices.index:
            continue
        quote = prices.loc[code]
        peers = {}
        for peer_id, peer_frame in peer_scores.items():
            if code in peer_frame.index:
                peer = peer_frame.loc[code]
                peers[peer_id] = {"rank": int(peer["rank"]), "pct": round(float(peer["pct"]) * 100, 2)}
        candidates.append({
            "ts_code": code,
            "name": str(row.name or code),
            "industry": str(row.industry or "未分类"),
            "close": float(quote.close),
            "pct_chg": float(quote.pct_chg),
            "amount_yi": float(quote.amount_yi),
            "rank": int(row.rank),
            "pct": round(float(row.pct) * 100, 2),
            "model_id": model_id,
            "peer_models": peers,
        })
    return candidates


def model_evidence(as_of, code, model_id):
    api = model_api()
    try:
        scores = api.scores(model_id, as_of)
        hit = scores[scores["ts_code"] == code]
        if hit.empty:
            return {"available": False, "reason": "该股票不在所选日期的模型评分池中"}
        if not (date(2023, 1, 3) <= as_of <= date(2026, 10, 16)):
            return {"available": False, "reason": "该日期没有预先计算的个股归因"}
        factors = api.explain(model_id, as_of, code, top=8)
        row = hit.iloc[0]
        return {
            "available": True,
            "model_id": model_id,
            "model_name": MODEL_LABELS[model_id],
            "rank": int(row["rank"]),
            "pool_size": int(len(scores)),
            "percentile": round(float(row["pct"]) * 100, 2),
            "method": str(factors.attrs.get("method", "")),
            "factors": _records(factors),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


@lru_cache(maxsize=16)
def _performance_rows(period):
    repo_results = MODEL_REPO / "ml" / "results" / "delivery"
    if period == "test":
        return pd.read_csv(repo_results / "test_portfolio.csv").to_dict(orient="records")
    return json.loads((repo_results / "long_results.json").read_text(encoding="utf-8"))


def model_comparison(as_of, model_id, market, *, rule="DB-K30-b8", period="test"):
    if rule not in STRATEGIES or period not in MODEL_PERIODS:
        raise ValueError("请选择已交付的策略与回测区间")
    api = model_api()
    board = api.leaderboard().set_index("model_id")
    nav_frames = {mid: api.nav(mid, rule, period) for mid in MODEL_IDS}
    series = []
    summaries = []
    for mid in MODEL_IDS:
        nav = nav_frames[mid]
        summary_rows = _performance_rows(period)
        summary = next(
            row for row in summary_rows
            if row.get("规则") == rule and
            (row.get("model_id") == mid if period == "long" else row.get("模型") == DELIVERY_MODEL_LABELS[mid])
        )
        if period == "test":
            annual = float(summary["全段 年化%"])
            excess = float(summary["全段 超额%"])
            drawdown = float(summary["全段 最大回撤%"])
            turnover = float(summary["换手（倍/年）"])
            costs = float(summary["费用占比（%/年）"])
        else:
            annual = float(summary["全期 2015 ~ 2026｜年化%"])
            excess = float(summary["全期 2015 ~ 2026｜超额%"])
            drawdown = float(summary["全期 2015 ~ 2026｜最大回撤%"])
            turnover = float(summary["换手（倍/年）"])
            costs = float(summary["费用占比（%/年）"])
        info = board.loc[mid]
        series.append({
            "model_id": mid,
            "label": MODEL_LABELS[mid],
            "points": [{"date": str(pd.Timestamp(row.date).date()), "value": float(row.nav)} for row in nav.itertuples(index=False)],
        })
        summaries.append({
            "model_id": mid,
            "label": MODEL_LABELS[mid],
            "rank_ic": round(float(info["测试 IC 全段"]), 4),
            "rank_ic_interval": str(info["测试 IC 全段 区间"]),
            "training_seconds": float(info["训练秒（最终阶段）"]),
            "annual_pct": annual,
            "excess_pct": excess,
            "max_drawdown_pct": drawdown,
            "turnover": turnover,
            "annual_cost_pct": costs,
            "capital": int(summary["本金"]),
        })
    shared = nav_frames[MODEL_IDS[0]]
    result = {
        "as_of": str(as_of),
        "model_id": model_id,
        "rule": rule,
        "rule_label": STRATEGIES[rule]["label"],
        "rule_description": STRATEGIES[rule]["description"],
        "period": period,
        "period_label": MODEL_PERIODS[period],
        "score_last_date": model_metadata()["last_score_date"],
        "nav_last_date": str(pd.Timestamp(shared.date.iloc[-1]).date()),
        "series": series,
        "benchmarks": [
            {"label": "准入池等权", "points": [{"date": str(pd.Timestamp(row.date).date()), "value": float(row.pool_nav)} for row in shared.itertuples(index=False)]},
        ],
        "summaries": summaries,
        "candidates": _model_candidates(as_of, model_id, market),
        "discipline": "测试期三个模型的 Rank IC 与组合超额差异均未达到统计显著；跨模型不比较原始分数。",
        "source": "XKKX33/rebound-model · ml/api.py 与 ml/results/delivery",
    }
    if period == "long":
        for key, label in (("hs300", "沪深300"), ("zz500", "中证500"), ("zz1000", "中证1000")):
            result["benchmarks"].append({"label": label, "points": [{"date": str(pd.Timestamp(row.date).date()), "value": float(getattr(row, key))} for row in shared.itertuples(index=False)]})
    return result


@lru_cache(maxsize=3)
def panel(as_of):
    cache = FEATURE_CACHE
    stamp = FEATURE_STAMP
    fingerprint = json.loads(PROFILE.read_text(encoding="utf-8"))["provenance"]["sha256"]
    identity = None
    if stamp.is_file():
        try:
            identity = json.loads(stamp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            identity = None
    if cache.is_file() and identity == {"as_of": str(as_of), "sha256": fingerprint}:
        return pd.read_parquet(cache)
    if cache.is_file() and isinstance(identity, dict) and identity.get("sha256") == fingerprint:
        raise RuntimeError(
            f"量价特征截面目前缓存于 {identity.get('as_of')}，当前请求为 {as_of}；未自动重算全市场特征。"
        )
    raise RuntimeError("量价特征缓存不存在或与当前历史库不匹配；未自动重算全市场特征。")


def feature_data(as_of, code, raw, model_id="xgb_F69", score_as_of=None):
    frame = panel(as_of)
    selected = frame[frame.ts_code == code]
    history = raw.copy().tail(180)
    c = history.close * history.adj_factor
    history["r20"] = c / c.shift(20) - 1
    history["dev_ma20"] = c / c.rolling(20).mean() - 1
    history["vol20"] = history.amount / history.amount.shift(1).rolling(20).mean()
    history["dd20"] = c / c.rolling(20).max() - 1
    history["from_low60"] = c / c.rolling(60).min() - 1
    history["amp20"] = ((history.high - history.low) / history.close).rolling(20).mean()
    features = []
    for key, label, group, unit, formula in FEATURE_INFO:
        multiplier = 100 if unit == "%" else 1
        values = frame[key].dropna().astype(float) * multiplier
        value = float(selected.iloc[0][key]) * multiplier if len(selected) and pd.notna(selected.iloc[0][key]) else None
        lo, hi = values.quantile([.01, .99]) if len(values) else (0, 1)
        if lo == hi:
            hi = lo + 1
        counts, edges = np.histogram(np.clip(values, lo, hi), bins=np.linspace(lo, hi, 21))
        times = history[["trade_date", key]].dropna().tail(80)
        features.append({
            "key": key, "label": label, "group": group, "unit": unit, "formula": formula,
            "value": value, "percentile": float((values <= value).mean() * 100) if value is not None and len(values) else None,
            "median": float(values.median()) if len(values) else None, "valid": len(values),
            "missing": len(frame) - len(values),
            "distribution": [{"lo": float(edges[i]), "hi": float(edges[i+1]), "count": int(n)} for i, n in enumerate(counts)],
            "timeseries": [{"date": str(pd.Timestamp(r.trade_date).date()), "value": float(r[key]) * multiplier}
                           for _, r in times.iterrows()],
        })
    attribution_date = score_as_of or as_of
    attribution = model_evidence(attribution_date, code, model_id)
    model_catalog = _records(model_api().feature_info())
    model_feature_count = sum(row["来源"] != "LSTM 序列通道（1.6）" for row in model_catalog)
    return {"provenance": "historical", "as_of": str(as_of), "score_as_of": str(attribution_date),
            "code": code, "pipeline": "rebound.features",
            "feature_count": model_feature_count, "population": "当日有复权因子的沪深主板股票，包含 ST；未应用选股准入",
            "population_count": len(frame), "features": features,
            "catalog": [row[0] for row in FEATURE_INFO],
            "model_feature_catalog": model_catalog,
            "model_attribution": attribution,
            "missing_reason": None if len(selected) else "该股票不在原模型的主板特征截面内"}


@lru_cache(maxsize=1)
def archived():
    directory = MODEL_REPO / "experiments/execgap"
    output = []
    for key, label, buffer, timing in (
        ("DB-K2-b8_now", "XGBoost · 开盘换仓", 8, "开盘卖出，并按同一开盘价买入"),
        ("DB-K2-b6_close_sell_prev_rank", "XGBoost · 尾盘换仓", 6, "尾盘卖出，下一交易日开盘补仓"),
    ):
        detail = json.loads((directory / f"detail_{key}.json").read_text(encoding="utf-8"))
        # Use one internally consistent JSON artifact, rather than mixing rounded CSV summaries.
        output.append({
            "experiment_id": "archive-" + key, "label": label, "model": "XGBoost",
            "strategy": "排名缓冲换仓", "strategy_id": key,
            "strategy_description": {
                "holdings": 2, "buffer": buffer, "sell_below_rank": 2 * buffer,
                "buy": "空位按上一交易日模型评分由高到低补齐；无法成交或不足一手则顺延。",
                "sell": f"持仓排名跌出前{2 * buffer}名时卖出；当天买入的股票不卖。",
                "timing": timing, "industry_cap": 2,
                "execution_assumption": (
                    "开盘换仓假设卖出资金可立即用于同一开盘价买入。满仓账户难以在开盘集合竞价中完成这一过程，"
                    "这项历史实验的执行假设偏乐观。"
                    if buffer == 8 else
                    "卖出决策使用上一交易日收盘后的排名，尾盘卖出的资金在下一交易日开盘补仓。"
                ),
            },
            "provenance": "repository_archive", "source": f"experiments/execgap/detail_{key}.json",
            "metrics": detail["row"], "drawdown": detail["drawdown"],
            "monthly": detail["monthly"], "annual": detail["yearly"],
            "capital_cny": 20000,
            "evaluation_end": detail["last_day"]["day"],
        })
    return output


def _debate_record(as_of, model_id, candidate_signature):
    expected_codes = [item[0] for item in candidate_signature]
    expected = {item[0]: item[1:] for item in candidate_signature}
    try:
        _, entries = debate_runs.list_runs(
            BASE,
            variant="nocap",
            cadence="M",
            as_of=date.fromisoformat(as_of),
            model_source=model_id,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "message": f"读取本地辩论记录失败：{exc}"}

    completed = [item for item in entries if item.get("completed")]
    incomplete = [item for item in entries if not item.get("completed")]
    for entry in [*completed, *incomplete]:
        folder = Path(entry["run_dir"])
        try:
            request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        context = request.get("context") or {}
        stored_candidates = request.get("candidates") or []
        stored_codes = [row.get("ts_code") for row in stored_candidates]
        if request.get("as_of") != as_of or context.get("model_source") != model_id:
            continue
        if stored_codes != expected_codes:
            continue
        input_hash = request.get("input_hash")
        if not input_hash or entry.get("hash_prefix") != input_hash[:12]:
            continue

        matches = True
        for row in stored_candidates:
            code = row["ts_code"]
            rank, pct, close, pct_chg, amount_yi = expected[code]
            model = row.get("model") or {}
            bars = row.get("bars") or []
            # Compare at the precision stored in the debate request contract.
            if (
                model.get("rank") != rank
                or not math.isclose(float(model.get("pct", math.nan)), pct, rel_tol=0, abs_tol=0.051)
                or not bars
                or bars[-1].get("date") != as_of
                or round(float(bars[-1].get("close", math.nan)), 2) != round(close, 2)
                or round(float(bars[-1].get("pct_chg", math.nan)), 2) != round(pct_chg, 2)
                or round(float(bars[-1].get("amount_yi", math.nan)), 2) != round(amount_yi, 2)
            ):
                matches = False
                break
        if not matches:
            continue

        if not entry.get("completed"):
            done = int(entry.get("debated_count", 0))
            return {
                "status": "incomplete",
                "submitted_count": len(stored_codes),
                "debated_count": done,
                "hash_prefix": input_hash[:12],
                "usage": entry.get("usage") or {},
                "message": (
                    f"找到对应的运行记录，目前已完成 {done}/{len(stored_codes)} 只，"
                    f"尚无终选名单。输入指纹 {input_hash[:12]}。"
                ),
            }

        final = entry.get("final") or {}
        result = entry.get("result") or {}
        if final.get("input_hash") != input_hash:
            continue
        usage = final.get("usage") or result.get("usage") or {}
        debated_count = int(final.get("debated_count", len(result.get("verdicts", []))))
        return {
            "status": "completed",
            "submitted_count": int(final.get("submitted_count", len(stored_codes))),
            "debated_count": debated_count,
            "failed_count": int(final.get("failed_count", 0)),
            "final_count": int(final.get("final_count", 0)),
            "final_picks": final.get("picks", []),
            "usage": usage,
            "hash_prefix": input_hash[:12],
            "completed_at": str(final.get("completed_at") or ""),
            "dialogue": {
                "analysts": result.get("analysts", []),
                "debate": result.get("debate", []),
                "verdicts": result.get("verdicts", []),
            },
            "message": (
                f"已载入本地真实辩论记录：完成 {debated_count}/{len(stored_codes)} 只，"
                f"终选 {final.get('final_count', 0)} 只。运行指纹 {input_hash[:12]}。"
            ),
        }
    if entries:
        return {
            "status": "mismatch",
            "submitted_count": 0,
            "debated_count": 0,
            "message": "本机有该日期的辩论记录，但模型候选或信号日行情与当前看板不一致，未合并显示。",
        }
    return {
        "status": "unavailable",
        "submitted_count": 0,
        "debated_count": 0,
        "message": "本机没有该日期的已保存辩论原文；当前仅显示已交付模型的候选与评分证据。",
    }


def _selected_debate_dialogue(record, selected_code):
    dialogue = record.get("dialogue")
    if not dialogue:
        return None
    return {
        "analysts": [row for row in dialogue["analysts"] if row.get("ts_code") == selected_code],
        "debate": [row for row in dialogue["debate"] if row.get("ts_code") == selected_code],
        "verdict": next(
            (row for row in dialogue["verdicts"] if row.get("ts_code") == selected_code),
            None,
        ),
    }


def module_data(as_of, market, model_id="xgb_F69", selected_code=None):
    candidates = _model_candidates(as_of, model_id, market)
    selected = next((item for item in candidates if item["ts_code"] == selected_code), None)
    evidence = model_evidence(as_of, selected_code, model_id) if selected else None
    signature = tuple(
        (
            item["ts_code"],
            item["rank"],
            round(item["pct"], 2),
            item["close"],
            item["pct_chg"],
            item["amount_yi"],
        )
        for item in candidates
    )
    debate_record = _debate_record(str(as_of), model_id, signature)
    agent_run = {key: value for key, value in debate_record.items() if key != "dialogue"}
    agent_run["available"] = debate_record.get("status") == "completed"
    agent_run["dialogue"] = _selected_debate_dialogue(debate_record, selected_code) if selected else None
    return {"as_of": str(as_of), "model_id": model_id, "model_label": MODEL_LABELS[model_id], "archived": archived(),
            "backtest_capability": {
                "status": "available",
                "reason": "三个模型与三条固定策略均有预先计算的净值和正式指标。任意本金或自定义规则需要另行重跑模拟，不会由页面参数自动改写历史回测。",
                "required": [
                    "当前页面的持仓预算只用于按收盘参考价测算股数、费用与现金。",
                    "组合页当前只展示一只信号日的模型候选参考计划。",
                ],
                "scope": "正式逐日评分截至2026-10-16；净值曲线与回测指标读取仓库已交付结果。",
            },
            "candidates": candidates,
            "selected_candidate": selected,
            "model_evidence": evidence,
            "candidate_method": f"{MODEL_LABELS[model_id]} 在信号日全池排序前20名；名单未作行业去重。",
            "agent_run": agent_run,
            "workflow": [
                {"label": "模型候选", "detail": "读取指定模型当天全池前20名，候选阶段不作行业去重。"},
                {"label": "独立分析", "detail": "模型分析师与量价分析师分别读取同一候选快照。"},
                {"label": "多空交锋", "detail": "看多、看空、看多回应依次读取前序观点。"},
                {"label": "风险裁决", "detail": "风险主席结合整批候选与个股辩论记录给出裁决。"},
                {"label": "整批终选", "detail": "终选名单由程序校验候选范围、名额和行业上限。"},
            ],
            "source": "XKKX33/rebound-model · rebound/debate 与 ml/api.py"}
