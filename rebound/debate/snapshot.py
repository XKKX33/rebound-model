"""候选快照：从 DuckDB 只读组装辩论 `RunRequest`（第 3.4 节）。

不改写 `pick_day` 与 `DayPick` 口径；组装完立刻关闭数据库连接，辩论期间不占着库。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rebound import config as app_config
from rebound import params as model_params
from rebound.data.errors import DataError
from rebound.debate.contracts import (
    Candidate,
    CandidateBar,
    CandidateFlag,
    CandidateModel,
    IndustryStats,
    LlmSettings,
    Methodology,
    ModelPickItem,
    PoolSummary,
    RiskVerdict,
    RunContext,
    RunLimits,
    RunRequest,
    fingerprint,
    input_fingerprint,
)
from rebound.debate.methodology import load_methodology
from rebound.features import FEATURES, TOL, feature_panel
from rebound.model.train import open_store
from rebound.select import DayPick, SelectError, pick_day

__all__ = [
    "EMPTY_INDUSTRY",
    "Snapshot",
    "build_request",
    "convert_features",
    "industry_name",
]

EMPTY_INDUSTRY = "（无行业）"

_PCT_TIMES_100 = frozenset(
    {
        "r1",
        "r5",
        "r10",
        "r20",
        "r60",
        "r120",
        "from_low60",
        "from_high60",
        "dev_ma5",
        "dev_ma20",
        "dev_ma60",
        "ma20_slope5",
        "ma60_slope10",
        "dd20",
        "amp20",
        "dif_ratio",
        "hist_ratio",
    }
)
_PCT_RAW = frozenset({"pct_chg", "max_up20"})
_RATIO_RAW = frozenset({"vol5", "vol20"})
_RANK_TIMES_100 = frozenset({"rank_r20", "rank_r60"})
_INT_FEATURES = frozenset({"launch10", "up_days10", "limit_up20", "ma20_gt_ma60", "limit_up_today"})
_BINARY_FEATURES = frozenset({"ma20_gt_ma60", "limit_up_today"})
_CONTINUOUS_FEATURES: tuple[str, ...] = tuple(f for f in FEATURES if f not in _BINARY_FEATURES)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """一次快照组装结果：携带协议请求以及模型原名单供展示对照。"""

    day_pick: DayPick
    request: RunRequest


def industry_name(value: Any) -> str:
    """空行业统一归为「（无行业）」，与 `select.dedup_top` 口径一致。"""
    if value is None or pd.isna(value):
        return EMPTY_INDUSTRY
    text = str(value).strip()
    return text if text else EMPTY_INDUSTRY


def _stock_name(value: Any, ts_code: str) -> str:
    if value is None or pd.isna(value):
        return ts_code
    text = str(value).strip()
    return text if text else ts_code


def _convert_value(name: str, raw: float) -> float | int:
    if name in _PCT_TIMES_100:
        return round(raw * 100.0, 2)
    if name in _PCT_RAW or name in _RATIO_RAW:
        return round(raw, 2)
    if name in _RANK_TIMES_100:
        return round(raw * 100.0, 1)
    if name in _INT_FEATURES:
        return int(round(raw))
    raise DataError(f"未知特征名：{name!r}")


def convert_features(row: pd.Series) -> tuple[dict[str, float | int], list[str]]:
    """按第 3.4 节把特征换算为展示单位，非有限值省略并记入缺失清单。"""
    values: dict[str, float | int] = {}
    missing: list[str] = []
    for name in FEATURES:
        raw = row.get(name)
        if raw is None or pd.isna(raw):
            missing.append(name)
            continue
        num = float(raw)
        if not math.isfinite(num):
            missing.append(name)
            continue
        values[name] = _convert_value(name, num)
    return values, missing


def _query_bars(store: Any, codes: list[str], day: date, limit_bars: int) -> pd.DataFrame:
    placeholders = ", ".join(["?"] * len(codes))
    sql = f"""
    WITH ranked AS (
        SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close, d.pct_chg, d.amount,
               a.adj_factor, l.up_limit, l.down_limit,
               row_number() OVER (PARTITION BY d.ts_code ORDER BY d.trade_date DESC) AS rn
        FROM daily d
        JOIN adj_factor a USING (ts_code, trade_date)
        LEFT JOIN daily_limit l USING (ts_code, trade_date)
        WHERE d.trade_date <= ? AND d.ts_code IN ({placeholders})
    )
    SELECT ts_code, trade_date, open, high, low, close, pct_chg, amount, adj_factor, up_limit, down_limit
    FROM ranked
    WHERE rn <= ?
    ORDER BY ts_code, trade_date ASC
    """
    return store.connection.execute(sql, [day, *codes, int(limit_bars)]).df()


def _limit_tag(close: float, up_limit: Any, down_limit: Any) -> str | None:
    if up_limit is not None and pd.notna(up_limit) and abs(close - float(up_limit)) <= TOL:
        return "up"
    if down_limit is not None and pd.notna(down_limit) and abs(close - float(down_limit)) <= TOL:
        return "down"
    return None


def _build_bars(
    code: str,
    raw_bars: pd.DataFrame,
    panel_lookup: dict[tuple[str, date], tuple[float | None, float | None, float | None]],
    as_of: date,
) -> tuple[CandidateBar, ...]:
    if raw_bars.empty:
        raise DataError(f"{code} 在 {as_of} 及之前没有关联到复权因子的日线")
    last_date = pd.Timestamp(raw_bars.iloc[-1]["trade_date"]).date()
    if last_date != as_of:
        raise DataError(f"{code} 最后一根日线日期是 {last_date}，不是信号日 {as_of}")
    adj_d = float(raw_bars.iloc[-1]["adj_factor"])
    if not math.isfinite(adj_d) or adj_d <= 0:
        raise DataError(f"{code} 在 {as_of} 的复权因子非法：{adj_d!r}")

    out: list[CandidateBar] = []
    for row in raw_bars.itertuples(index=False):
        bar_date = pd.Timestamp(row.trade_date).date()
        factor = float(row.adj_factor) / adj_d
        vol5, dif_ratio, hist_ratio = panel_lookup.get((code, bar_date), (None, None, None))
        out.append(
            CandidateBar(
                date=bar_date.isoformat(),
                open=round(float(row.open) * factor, 2),
                high=round(float(row.high) * factor, 2),
                low=round(float(row.low) * factor, 2),
                close=round(float(row.close) * factor, 2),
                pct_chg=round(float(row.pct_chg), 2),
                amount_yi=round(float(row.amount) / 100_000.0, 2),
                limit=_limit_tag(float(row.close), row.up_limit, row.down_limit),
                vol5=vol5,
                dif_ratio=dif_ratio,
                hist_ratio=hist_ratio,
            )
        )
    return tuple(out)


def _panel_bar_metrics(panel: pd.DataFrame) -> dict[tuple[str, date], tuple[float | None, float | None, float | None]]:
    lookup: dict[tuple[str, date], tuple[float | None, float | None, float | None]] = {}
    for row in panel.itertuples(index=False):
        code = str(row.ts_code)
        day = pd.Timestamp(row.trade_date).date()
        v5 = round(float(row.vol5), 2) if pd.notna(row.vol5) and math.isfinite(float(row.vol5)) else None
        dif = round(float(row.dif_ratio) * 100.0, 2) if pd.notna(row.dif_ratio) and math.isfinite(float(row.dif_ratio)) else None
        hist = round(float(row.hist_ratio) * 100.0, 2) if pd.notna(row.hist_ratio) and math.isfinite(float(row.hist_ratio)) else None
        lookup[(code, day)] = (v5, dif, hist)
    return lookup


def _pool_summary(pool_today: pd.DataFrame) -> PoolSummary:
    valid_chg = pool_today["pct_chg"].dropna()
    if valid_chg.empty:
        raise DataError("当天股票池没有任何有效涨跌幅")
    med_chg = float(valid_chg.median())
    med_r5 = float(pool_today["r5"].dropna().median() * 100.0) if pool_today["r5"].notna().any() else 0.0
    med_r20 = float(pool_today["r20"].dropna().median() * 100.0) if pool_today["r20"].notna().any() else 0.0
    up_ratio = float((valid_chg > 0).sum() / len(valid_chg) * 100.0)
    return PoolSummary(
        median_pct_chg=round(med_chg, 2),
        median_r5=round(med_r5, 2),
        median_r20=round(med_r20, 2),
        up_ratio=round(up_ratio, 2),
    )


def _median_pct(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    val = float(clean.median() * 100.0)
    return round(val, 2) if math.isfinite(val) else None


def _build_flags(
    *,
    today_row: pd.Series,
    bars: tuple[CandidateBar, ...],
    missing_cols: list[str],
    top_count: int,
    max_per_industry: int,
) -> tuple[CandidateFlag, ...]:
    flags: list[CandidateFlag] = []
    if bool(today_row.get("up_close")) or bars[-1].limit == "up":
        flags.append(CandidateFlag("limit_up_today", "信号日收盘涨停"))
    if bool(today_row.get("up_open")):
        flags.append(CandidateFlag("one_word_limit_up", "信号日一字涨停"))

    streak = 0
    for bar in reversed(bars):
        if bar.limit == "up":
            streak += 1
        else:
            break
    if streak >= 2:
        flags.append(CandidateFlag("limit_up_streak", f"连续 {streak} 天收盘涨停"))

    recent_downs = sum(1 for bar in bars[-5:] if bar.limit == "down")
    if recent_downs >= 1:
        flags.append(CandidateFlag("limit_down_recent", f"最近 5 个交易日有 {recent_downs} 天收盘跌停"))

    if missing_cols:
        flags.append(CandidateFlag("missing_features", f"数据不够，缺 {'、'.join(missing_cols)}"))

    if top_count >= 3:
        flags.append(
            CandidateFlag(
                "industry_crowded",
                f"同行业在送辩候选里有 {top_count} 只，终选最多选 {max_per_industry} 只",
            )
        )
    return tuple(flags)


def build_request(
    db_path: Path,
    *,
    settings: app_config.Settings,
    as_of: date | None = None,
    variant: str | None = None,
    cadence: str | None = None,
    methodology: Methodology | None = None,
) -> Snapshot:
    """只读组装一次辩论请求。组装完立刻关闭数据库。"""
    resolved_variant = variant or settings.model.variant
    resolved_cadence = cadence or settings.model.cadence
    day_pick = pick_day(
        db_path,
        as_of=as_of,
        variant=resolved_variant,
        cadence=resolved_cadence,
        top_n=settings.pick.top_n,
        max_per_industry=settings.pick.max_per_industry,
        candidate_limit=settings.pick.candidate_limit,
    )
    pool_df = day_pick.pool
    if pool_df.empty:
        raise SelectError(f"{day_pick.day} {model_params.VARIANT_NAMES[resolved_variant]}股票池为空")

    head = pool_df.head(min(settings.debate.candidates, len(pool_df)))
    submitted_codes = [str(c) for c in head["ts_code"]]
    day = day_pick.day

    with open_store(db_path) as store:
        sessions = store.sessions(day, settings.debate.bars + 10, exchange=settings.sync.exchange)
        if not sessions:
            raise DataError(f"交易日历缺少 {day} 及之前的开市日")
        panel = feature_panel(store, sessions[0], day)
        bars_df = _query_bars(store, submitted_codes, day, settings.debate.bars)

    panel["ts_code"] = panel["ts_code"].astype(str)
    panel["trade_date_d"] = pd.to_datetime(panel["trade_date"]).dt.date
    today_panel = panel[panel["trade_date_d"] == day].set_index("ts_code")
    pool_codes = [str(c) for c in pool_df["ts_code"]]
    pool_today = today_panel.reindex(pool_codes)
    if pool_today.empty or pool_today["close"].isna().all():
        raise DataError(f"{day} 特征面板缺失股票池行")

    pct_table = pd.DataFrame(index=pool_today.index)
    for col in _CONTINUOUS_FEATURES:
        pct_table[col] = pool_today[col].rank(pct=True, method="average") * 100.0

    pool_industries = [industry_name(v) for v in pool_df["industry"]]
    pool_ind_counts: dict[str, int] = {}
    rank_in_industry_by_code: dict[str, int] = {}
    for code, ind in zip(pool_codes, pool_industries):
        next_rank = pool_ind_counts.get(ind, 0) + 1
        pool_ind_counts[ind] = next_rank
        rank_in_industry_by_code[code] = next_rank

    pool_today = pool_today.assign(_industry=pool_industries)
    ind_medians = {
        ind: (_median_pct(group["r5"]), _median_pct(group["r20"]))
        for ind, group in pool_today.groupby("_industry", sort=False)
    }

    head_industries = [industry_name(v) for v in head["industry"]]
    top_ind_counts: dict[str, int] = {}
    for ind in head_industries:
        top_ind_counts[ind] = top_ind_counts.get(ind, 0) + 1

    panel_lookup = _panel_bar_metrics(panel)
    bars_by_code = {str(code): group for code, group in bars_df.groupby("ts_code", sort=False)}
    model_pick_codes = {str(c) for c in day_pick.picks["ts_code"]}

    candidates: list[Candidate] = []
    for row in head.to_dict(orient="records"):
        code = str(row["ts_code"])
        ind = industry_name(row["industry"])
        today_row = pool_today.loc[code]
        features, missing_cols = convert_features(today_row)
        feature_pct = {
            col: round(float(pct_table.at[code, col]), 1)
            for col in _CONTINUOUS_FEATURES
            if pd.notna(pct_table.at[code, col]) and math.isfinite(float(pct_table.at[code, col]))
        }
        cand_bars = _build_bars(code, bars_by_code.get(code, pd.DataFrame()), panel_lookup, day)
        med_r5, med_r20 = ind_medians.get(ind, (None, None))
        top_count = top_ind_counts[ind]
        flags = _build_flags(
            today_row=today_row,
            bars=cand_bars,
            missing_cols=missing_cols,
            top_count=top_count,
            max_per_industry=settings.pick.max_per_industry,
        )
        candidates.append(
            Candidate(
                ts_code=code,
                name=_stock_name(row["name"], code),
                industry=ind,
                model=CandidateModel(
                    rank=int(row["rank"]),
                    pct=round(float(row["pct"]) * 100.0, 1),
                    score=round(float(row["score"]), 4),
                    in_model_picks=code in model_pick_codes,
                ),
                features=features,
                feature_pct=feature_pct,
                bars=cand_bars,
                industry_stats=IndustryStats(
                    pool_count=pool_ind_counts[ind],
                    top_count=top_count,
                    rank_in_industry=rank_in_industry_by_code[code],
                    median_r5=med_r5,
                    median_r20=med_r20,
                ),
                flags=flags,
            )
        )

    debated_set = set(submitted_codes)
    model_picks = tuple(
        ModelPickItem(
            ts_code=str(row["ts_code"]),
            name=_stock_name(row["name"], str(row["ts_code"])),
            industry=industry_name(row["industry"]),
            rank=int(row["rank"]),
            debated=str(row["ts_code"]) in debated_set,
        )
        for row in day_pick.picks.to_dict(orient="records")
    )

    meth = methodology if methodology is not None else load_methodology(root=settings.root)
    limits = RunLimits(
        candidates=len(candidates),
        final=min(settings.pick.top_n, len(candidates)),
        max_per_industry=settings.pick.max_per_industry,
        concurrency=settings.debate.concurrency,
        call_timeout_seconds=settings.debate.call_timeout_seconds,
    )
    llm = LlmSettings(
        id=settings.debate.llm.model,
        base_url=settings.debate.llm.base_url,
        reasoning_effort=settings.debate.llm.reasoning_effort,
        max_tokens=settings.debate.llm.max_tokens,
        temperature=settings.debate.llm.temperature,
    )
    context = RunContext(
        as_of=day.isoformat(),
        model_label="反弹模型",
        variant=resolved_variant,
        cadence=resolved_cadence,
        fit_date=day_pick.fit_date.isoformat(),
        model_id=day_pick.model_id,
        scores_stored=day_pick.stored,
        pool_size=len(pool_df),
        candidates=len(candidates),
        final=limits.final,
        max_per_industry=settings.pick.max_per_industry,
        pool=_pool_summary(pool_today),
    )
    cands_tuple = tuple(candidates)
    cand_hash = fingerprint([c.payload() for c in cands_tuple])
    inp_hash = input_fingerprint(
        as_of=day.isoformat(),
        limits=limits,
        methodology=meth,
        llm=llm,
        context=context,
        model_picks=model_picks,
        candidates=cands_tuple,
    )
    run_id = f"{day.isoformat()}-{resolved_variant}-{resolved_cadence}-{inp_hash[:12]}"
    request = RunRequest(
        run_id=run_id,
        as_of=day.isoformat(),
        candidate_hash=cand_hash,
        input_hash=inp_hash,
        limits=limits,
        methodology=meth,
        llm=llm,
        context=context,
        model_picks=model_picks,
        candidates=cands_tuple,
    )
    return Snapshot(day_pick=day_pick, request=request)
