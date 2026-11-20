"""多模型（XGBoost F69 / 随机森林 F69 / LSTM S+F69）与 F69 特征的只读桥接层。

封装对 `ml.api` 与 `ml.features_extra` 的调用，供 Web UI、AI 复盘对话工具与辩论快照共用。
设计纪律：
- 严格只读，不训练、不改写 `data/ml/` 或研究库；
- `data/ml/api_export/` 不存在时自动降级，不影响原有 `return_model`（F28）流程与隔离测试；
- 遵循统计纪律：三个 F69 模型测试段差异不显著，跨模型比较一律用百分位 `pct`，不直接比原始 `score`。
"""

from __future__ import annotations

import functools
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd

from rebound.debate.runs import LEGACY_MODEL_SOURCE

__all__ = [
    "ALL_MODEL_SOURCES",
    "LEGACY_MODEL_SOURCE",
    "MAIN_BACKTEST_RULE",
    "ML_MODEL_IDS",
    "ML_MODEL_META",
    "START_YEARS",
    "build_ml_day_pick",
    "compute_extra_features_for_day",
    "get_day_consensus",
    "get_feature_catalog",
    "get_leaderboard",
    "get_ml_dates",
    "get_nav_compare",
    "get_starts_summary",
    "get_stock_explain_all",
    "is_ml_available",
    "is_ml_model",
    "list_model_sources",
    "validate_model_source",
]

ML_MODEL_IDS: tuple[str, ...] = ("xgb_F69", "rf_F69", "lstm_SF69")
ALL_MODEL_SOURCES: tuple[str, ...] = (*ML_MODEL_IDS, LEGACY_MODEL_SOURCE)
MAIN_BACKTEST_RULE: str = "DB-K30-b8"
START_YEARS: tuple[int, ...] = tuple(range(2015, 2025))
TRADING_DAYS_PER_YEAR: float = 242.0
CAGR_MIN_DAYS: int = 60
INITIAL_CAPITAL: float = 200_000.0

ML_MODEL_META: dict[str, dict[str, Any]] = {
    "xgb_F69": {
        "id": "xgb_F69",
        "name": "XGBoost F69",
        "short_name": "XGBoost",
        "feature_set": "F69",
        "feature_count": 69,
        "explain_method": "TreeSHAP（精确）",
        "tagline": "推荐默认 · 69 特征 · 长期 IC 最高、换起点最稳",
    },
    "rf_F69": {
        "id": "rf_F69",
        "name": "随机森林 F69",
        "short_name": "随机森林",
        "feature_set": "F69",
        "feature_count": 69,
        "explain_method": "路径分解（Saabas）",
        "tagline": "69 特征 · 2015 股灾回撤最小（-40.1%）",
    },
    "lstm_SF69": {
        "id": "lstm_SF69",
        "name": "LSTM S+F69",
        "short_name": "LSTM",
        "feature_set": "S+F69",
        "feature_count": 69,
        "explain_method": "积分梯度（含 60×11 序列通道）",
        "tagline": "60 天日线序列 + 69 特征 · 2019 后起步收益最高、回撤与换手最大",
    },
    LEGACY_MODEL_SOURCE: {
        "id": LEGACY_MODEL_SOURCE,
        "name": "原版反弹 F28",
        "short_name": "原版 F28",
        "feature_set": "F28",
        "feature_count": 28,
        "explain_method": None,
        "tagline": "仓库原版 28 量价特征月度重训 XGBoost（兼容历史账本）",
    },
}

DISCIPLINE_NOTES: tuple[str, ...] = (
    "三个 F69 模型在测试段（2023-01 ~ 2026-10）的逐日 IC（0.119 ~ 0.120）和组合收益差异在统计上均不显著（95% 区间均跨 0），切勿解读为某个模型显著更优。",
    "不同模型的原始 score 量纲不同、不可直接横向比较；跨模型对比一律使用全池百分位 pct（0 ~ 100，越大越靠前）。",
    "事后收益（T+1 开盘买 → T+5 开盘卖）属于信号日当时未知的未来信息，仅限复盘核验，不得作为信号日选股依据。",
)


def is_ml_model(model_source: str | None) -> bool:
    return bool(model_source and model_source in ML_MODEL_IDS)


def validate_model_source(model_source: str | None) -> str:
    clean = (model_source or "").strip()
    if not clean:
        return LEGACY_MODEL_SOURCE
    if clean not in ALL_MODEL_SOURCES:
        raise ValueError(f"不支持的模型源：{clean}；可选：{', '.join(ALL_MODEL_SOURCES)}")
    return clean


def _import_ml_api() -> Any:
    from ml import api as ml_api

    return ml_api


def is_ml_available() -> bool:
    """检查 `ml.api` 与本地 `data/ml/api_export/` 是否就绪。"""
    try:
        ml_api = _import_ml_api()
        return ml_api._export("manifest.json") is not None
    except Exception:
        return False


def _clean_float(val: Any, digits: int | None = None) -> float | None:
    if val is None or pd.isna(val):
        return None
    num = float(val)
    if not math.isfinite(num):
        return None
    return round(num, digits) if digits is not None else num


def _clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        cleaned: dict[str, Any] = {}
        for k, v in rec.items():
            if isinstance(v, (float, np.floating)):
                cleaned[str(k)] = _clean_float(v)
            elif isinstance(v, (np.integer,)):
                cleaned[str(k)] = int(v)
            elif pd.isna(v):
                cleaned[str(k)] = None
            else:
                cleaned[str(k)] = v
        rows.append(cleaned)
    return rows


@functools.lru_cache(maxsize=64)
def _cached_scores_df(model_id: str, as_of_str: str, with_outcome: bool) -> pd.DataFrame:
    ml_api = _import_ml_api()
    return ml_api.scores(model_id, as_of_str, with_outcome=with_outcome)


@functools.lru_cache(maxsize=64)
def _cached_picks_codes(model_id: str, as_of_str: str, top_n: int, max_per_industry: int) -> frozenset[str]:
    ml_api = _import_ml_api()
    pk = ml_api.picks(model_id, as_of_str, top_n=top_n, max_per_industry=max_per_industry)
    return frozenset(pk["ts_code"].astype(str))


@functools.lru_cache(maxsize=24)
def _cached_explain_day(model_id: str, as_of_str: str) -> dict[str, dict[str, Any]]:
    """用 PyArrow 谓词下推按单日读取 explain parquet（避免 3 个百兆表在 lru_cache(2) 中互相驱逐抖动）。"""
    ml_api = _import_ml_api()
    if model_id not in ml_api.EXPLAINED:
        return {}
    fname = ml_api.EXPLAINED[model_id]
    path = ml_api._export(f"explain_{fname}") or (ml_api.data_dir() / "explain" / fname)
    if not path.exists():
        return {}
    try:
        df = duckdb.execute(
            "SELECT * FROM read_parquet(?) WHERE CAST(trade_date AS DATE) = CAST(? AS DATE)",
            [str(path), as_of_str],
        ).df()
    except Exception:
        t = ml_api._explain_table(model_id)
        df = t[t["trade_date"] == pd.Timestamp(as_of_str)]
    if df.empty:
        return {}
    n_cols = sum(1 for c in df.columns if c.startswith("f") and c[1:].isdigit())
    method = ml_api.EXPLAIN_METHOD.get(model_id, ML_MODEL_META[model_id]["explain_method"])
    cat = get_feature_catalog().get("by_name") or {}
    ch_info = getattr(ml_api, "CHANNEL_INFO", {})

    def _fast_cn(name: str) -> str:
        if name.startswith("seq_"):
            ch = ch_info.get(name[4:])
            return f"序列：{ch[0]}（近 60 天）" if ch else name
        if name.startswith("na_"):
            base = name[3:]
            return f"缺失：{cat.get(base, {}).get('cn_name') or base}"
        return cat.get(name, {}).get("cn_name") or name

    out: dict[str, dict[str, Any]] = {}
    for rec in df.to_dict(orient="records"):
        code = str(rec["ts_code"])
        factors = []
        for k in range(1, n_cols + 1):
            fname_k = str(rec[f"f{k}"])
            contrib_k = _clean_float(rec[f"v{k}"], 5)
            factors.append(
                {
                    "feature": fname_k,
                    "cn_name": _fast_cn(fname_k),
                    "value": _clean_float(rec[f"x{k}"], 5),
                    "contribution": contrib_k if contrib_k is not None else 0.0,
                    "direction": "加分" if (contrib_k or 0.0) > 0 else "减分",
                }
            )
        out[code] = {
            "method": method,
            "base": _clean_float(rec.get("base"), 5),
            "score": _clean_float(rec.get("score"), 5),
            "top_factors": factors,
        }
    return out


@functools.lru_cache(maxsize=8)
def _cached_ml_dates(model_id: str) -> tuple[str, ...]:
    ml_api = _import_ml_api()
    ds = ml_api.dates(model_id)
    return tuple(d.date().isoformat() for d in reversed(ds))


def get_ml_dates(model_id: str, limit: int | None = None) -> list[str]:
    if not is_ml_model(model_id) or not is_ml_available():
        return []
    days = _cached_ml_dates(model_id)
    return list(days[:limit]) if limit is not None else list(days)


def list_model_sources() -> list[dict[str, Any]]:
    ml_ready = is_ml_available()
    items: list[dict[str, Any]] = []
    if ml_ready:
        for mid in ML_MODEL_IDS:
            days = _cached_ml_dates(mid)
            meta = ML_MODEL_META[mid]
            items.append(
                {
                    **meta,
                    "available": True,
                    "is_default": mid == "xgb_F69",
                    "date_min": days[-1] if days else None,
                    "date_max": days[0] if days else None,
                    "date_count": len(days),
                }
            )
        return items
    legacy_meta = ML_MODEL_META[LEGACY_MODEL_SOURCE]
    items.append(
        {
            **legacy_meta,
            "available": True,
            "is_default": True,
            "date_min": None,
            "date_max": None,
            "date_count": None,
        }
    )
    return items


def get_leaderboard() -> dict[str, Any]:
    if not is_ml_available():
        return {
            "available": False,
            "models": list_model_sources(),
            "leaderboard": [],
            "long_summary": {},
            "starts_summary": {"available": False, "rows": [], "summary": {}},
            "data_summary": {},
            "discipline_notes": list(DISCIPLINE_NOTES),
        }
    ml_api = _import_ml_api()
    from ml.common import RESULTS

    lb_rows = _clean_records(ml_api.leaderboard())
    long_summary: dict[str, Any] = {}
    long_path = RESULTS / "delivery" / "long_results.json"
    if long_path.is_file():
        try:
            long_summary = json.loads(long_path.read_text(encoding="utf-8"))
        except Exception:
            long_summary = {}

    return {
        "available": True,
        "models": list_model_sources(),
        "leaderboard": lb_rows,
        "long_summary": long_summary,
        "starts_summary": get_starts_summary(MAIN_BACKTEST_RULE),
        "data_summary": ml_api.data_summary(),
        "rules": [MAIN_BACKTEST_RULE],
        "main_rule": MAIN_BACKTEST_RULE,
        "available_start_years": list(START_YEARS),
        "discipline_notes": list(DISCIPLINE_NOTES),
    }


@functools.lru_cache(maxsize=4)
def _cached_starts_nav_df(rule: str) -> pd.DataFrame | None:
    from ml.common import RESULTS

    p = RESULTS / "delivery" / f"long_starts_nav_{rule}.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p, encoding="utf-8-sig")
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


@functools.lru_cache(maxsize=2)
def _cached_starts_summary_df() -> pd.DataFrame | None:
    from ml.common import RESULTS

    p = RESULTS / "delivery" / "long_starts.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p, encoding="utf-8-sig")


@functools.lru_cache(maxsize=1)
def _cached_index_closes() -> dict[str, dict[str, float]]:
    """读取三大对照指数（沪深300 / 中证500 / 中证1000）收盘点位，优先研究库 index_daily，降级到 long_nav CSV。"""
    out: dict[str, dict[str, float]] = {"hs300": {}, "zz500": {}, "zz1000": {}}
    code_map = {"000300.SH": "hs300", "000905.SH": "zz500", "000852.SH": "zz1000"}
    try:
        from ml.common import DB_PATH

        if DB_PATH.is_file():
            con = duckdb.connect(str(DB_PATH), read_only=True)
            try:
                rows = con.execute(
                    "SELECT CAST(trade_date AS VARCHAR) AS d, ts_code, close "
                    "FROM index_daily WHERE ts_code IN ('000300.SH', '000905.SH', '000852.SH') "
                    "ORDER BY trade_date"
                ).fetchall()
                for d_raw, ts_code, close_val in rows:
                    key = code_map.get(str(ts_code))
                    fv = _clean_float(close_val, 6)
                    if key and fv is not None and fv > 0:
                        out[key][str(d_raw)[:10]] = fv
            finally:
                con.close()
    except Exception:
        pass

    if any(out.values()):
        return out

    # 降级：从交付包长期净值 CSV 提取指数相对点位
    try:
        from ml.common import RESULTS

        p = RESULTS / "delivery" / f"long_nav_{MAIN_BACKTEST_RULE}.csv"
        if p.is_file():
            df = pd.read_csv(p, encoding="utf-8-sig")
            d_list = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").tolist()
            for col_cn, key in (("沪深300", "hs300"), ("中证500", "zz500"), ("中证1000", "zz1000")):
                if col_cn in df.columns:
                    for d_str, v in zip(d_list, df[col_cn].tolist()):
                        fv = _clean_float(v, 6)
                        if fv is not None and fv > 0:
                            out[key][d_str] = fv
    except Exception:
        pass
    return out


def _rebased_index_values(dates: list[str]) -> dict[str, list[float | None]]:
    if not dates:
        return {}
    closes = _cached_index_closes()
    rebased: dict[str, list[float | None]] = {}
    for key in ("hs300", "zz500", "zz1000"):
        series_map = closes.get(key) or {}
        base: float | None = None
        vals: list[float | None] = []
        for d in dates:
            c = series_map.get(d)
            if c is not None and c > 0:
                if base is None:
                    base = c
                vals.append(round(c / base, 4))
            else:
                vals.append(None)
        if base is not None and sum(1 for v in vals if v is not None) >= 2:
            rebased[key] = vals
    return rebased


def _rolling_cagr_series(values: list[float | None], min_days: int = CAGR_MIN_DAYS) -> list[float | None]:
    """计算从起点出发截至每一交易日的累计年化收益率（%，按每年 242 个交易日折算）。"""
    n = len(values)
    if n == 0:
        return []
    base_idx: int | None = None
    base_val: float | None = None
    for i, v in enumerate(values):
        if v is not None and v > 0:
            base_idx = i
            base_val = v
            break
    if base_idx is None or base_val is None:
        return [None] * n

    out: list[float | None] = []
    for i, v in enumerate(values):
        elapsed = i - base_idx
        if elapsed < min_days or v is None or v <= 0:
            out.append(None)
            continue
        ratio = v / base_val
        cagr = (ratio ** (TRADING_DAYS_PER_YEAR / elapsed) - 1.0) * 100.0
        out.append(round(cagr, 2) if math.isfinite(cagr) else None)
    return out


def _nav_stats(
    values: list[float | None],
    dates: list[str],
    pool_values: list[float | None] | None = None,
    *,
    capital: float = INITIAL_CAPITAL,
) -> dict[str, Any]:
    clean = [(i, v) for i, v in enumerate(values) if v is not None and v > 0]
    if len(clean) < 2:
        return {
            "final_nav": None,
            "end_capital": None,
            "multiple": None,
            "total_return": None,
            "cagr": None,
            "max_drawdown": None,
            "excess_total": None,
            "annual_excess": None,
            "trading_days": 0,
            "hold_years": 0.0,
        }
    start_val = clean[0][1]
    end_val = clean[-1][1]
    multiple = end_val / start_val
    total_ret = multiple - 1.0

    # 与仓库 rebound.backtest.stats 及交付表一致：按 242 个交易日/年折算年化
    n_days = max(len(clean) - 1, 1)
    years = n_days / TRADING_DAYS_PER_YEAR
    cagr = multiple ** (1.0 / years) - 1.0

    peak = start_val
    max_dd = 0.0
    for _, v in clean:
        if v > peak:
            peak = v
        dd = v / peak - 1.0
        if dd < max_dd:
            max_dd = dd

    excess_total = None
    annual_excess = None
    if pool_values is not None and len(pool_values) == len(values):
        p_clean = [v for v in pool_values if v is not None and v > 0]
        if len(p_clean) >= 2:
            pool_ret = p_clean[-1] / p_clean[0] - 1.0
            excess_total = total_ret - pool_ret
        diffs: list[float] = []
        for idx in range(1, len(values)):
            v0, v1 = values[idx - 1], values[idx]
            p0, p1 = pool_values[idx - 1], pool_values[idx]
            if v0 and v1 and p0 and p1 and v0 > 0 and p0 > 0:
                diffs.append((v1 / v0 - 1.0) - (p1 / p0 - 1.0))
        if diffs:
            annual_excess = float(np.mean(diffs)) * TRADING_DAYS_PER_YEAR

    return {
        "final_nav": round(end_val, 4),
        "end_capital": round(capital * multiple, 0),
        "multiple": round(multiple, 2),
        "total_return": round(total_ret * 100.0, 2),
        "cagr": round(cagr * 100.0, 2),
        "max_drawdown": round(max_dd * 100.0, 2),
        "excess_total": round(excess_total * 100.0, 2) if excess_total is not None else None,
        "annual_excess": round(annual_excess * 100.0, 2) if annual_excess is not None else None,
        "trading_days": n_days,
        "hold_years": round(years, 1),
    }


def _build_yearly_breakdown(
    dates: list[str],
    series_values: Mapping[str, list[float | None]],
    *,
    capital: float = INITIAL_CAPITAL,
) -> list[dict[str, Any]]:
    """按自然年汇总期末资金、年收益率与年内最大回撤（口径对齐逐年回测表）。"""
    if not dates:
        return []
    year_indices: dict[str, list[int]] = {}
    for idx, d in enumerate(dates):
        year_indices.setdefault(d[:4], []).append(idx)

    prev_end: dict[str, float | None] = {}
    for key, vals in series_values.items():
        first_v = next((v for v in vals if v is not None and v > 0), None)
        prev_end[key] = first_v

    rows: list[dict[str, Any]] = []
    for yr in sorted(year_indices.keys()):
        idxs = year_indices[yr]
        end_date = dates[idxs[-1]]
        models_out: dict[str, dict[str, Any]] = {}
        benchmarks_out: dict[str, float | None] = {}

        for mid in ML_MODEL_IDS:
            vals = series_values.get(mid) or []
            yr_vals = [vals[i] for i in idxs if i < len(vals) and vals[i] is not None and vals[i] > 0]
            base_v = prev_end.get(mid)
            if not yr_vals or not base_v:
                models_out[mid] = {"end_capital": None, "return_pct": None, "max_drawdown": None}
                continue
            end_v = yr_vals[-1]
            ret_pct = (end_v / base_v - 1.0) * 100.0
            peak = base_v
            max_dd = 0.0
            for v in yr_vals:
                if v > peak:
                    peak = v
                dd = v / peak - 1.0
                if dd < max_dd:
                    max_dd = dd
            models_out[mid] = {
                "end_capital": round(capital * end_v, 0),
                "return_pct": round(ret_pct, 2),
                "max_drawdown": round(max_dd * 100.0, 2),
            }
            prev_end[mid] = end_v

        for bkey in ("pool_nav", "hs300", "zz500", "zz1000"):
            vals = series_values.get(bkey)
            if not vals:
                continue
            yr_vals = [vals[i] for i in idxs if i < len(vals) and vals[i] is not None and vals[i] > 0]
            base_v = prev_end.get(bkey)
            if yr_vals and base_v:
                end_v = yr_vals[-1]
                benchmarks_out[bkey] = round((end_v / base_v - 1.0) * 100.0, 2)
                prev_end[bkey] = end_v
            else:
                benchmarks_out[bkey] = None

        rows.append(
            {
                "year": int(yr),
                "end_date": end_date,
                "is_partial": yr == dates[-1][:4] and not end_date.endswith("-12-31"),
                "models": models_out,
                "benchmarks": benchmarks_out,
            }
        )
    return rows


def get_starts_summary(rule: str = MAIN_BACKTEST_RULE) -> dict[str, Any]:
    """返回 2015 ~ 2024 共 10 个独立起步年至今（2026-10）的年化、超额、回撤与汇总统计。"""
    if not is_ml_available():
        return {"available": False, "rule": rule, "rows": [], "summary": {}}
    ref_df = _cached_starts_summary_df()
    nav_df = _cached_starts_nav_df(rule)
    if ref_df is None or nav_df is None:
        return {"available": False, "rule": rule, "rows": [], "summary": {}}

    sub_ref = ref_df[ref_df["规则"] == rule].copy()
    if sub_ref.empty:
        return {"available": False, "rule": rule, "rows": [], "summary": {}}

    end_date = str(nav_df["date"].iloc[-1])
    rows: list[dict[str, Any]] = []
    for y in START_YEARS:
        col_x = f"xgb_F69|{y}"
        if col_x not in nav_df.columns:
            continue
        valid = nav_df.loc[nav_df[col_x].notna()]
        if valid.empty:
            continue
        start_date = str(valid["date"].iloc[0])
        n_days = max(len(valid) - 1, 1)
        hold_years = round(n_days / TRADING_DAYS_PER_YEAR, 1)
        pool_col = f"准入池|{y}"
        pool_end = _clean_float(valid[pool_col].iloc[-1], 4) if pool_col in valid.columns else None
        pool_cagr = (
            round(((pool_end ** (TRADING_DAYS_PER_YEAR / n_days)) - 1.0) * 100.0, 2)
            if pool_end is not None and pool_end > 0
            else None
        )

        yr_ref = sub_ref[sub_ref["起步年"] == y].set_index("model_id")
        models_row: dict[str, dict[str, Any]] = {}
        best_mid = ""
        best_cagr = -1e9
        for mid in ML_MODEL_IDS:
            mcol = f"{mid}|{y}"
            end_nav = _clean_float(valid[mcol].iloc[-1], 4) if mcol in valid.columns else None
            exact_cagr = (
                round(((end_nav ** (TRADING_DAYS_PER_YEAR / n_days)) - 1.0) * 100.0, 2)
                if end_nav is not None and end_nav > 0
                else None
            )
            ref_row = yr_ref.loc[mid] if mid in yr_ref.index else None
            cagr_val = exact_cagr if exact_cagr is not None else (_clean_float(ref_row["年化%"], 2) if ref_row is not None else None)
            if cagr_val is not None and cagr_val > best_cagr:
                best_cagr = cagr_val
                best_mid = mid
            models_row[mid] = {
                "end_nav": end_nav,
                "end_capital": round(INITIAL_CAPITAL * end_nav, 0) if end_nav is not None else None,
                "multiple": round(end_nav, 2) if end_nav is not None else None,
                "cagr": cagr_val,
                "annual_excess": _clean_float(ref_row["年化超额%"], 2) if ref_row is not None else None,
                "max_drawdown": _clean_float(ref_row["最大回撤%"], 2) if ref_row is not None else None,
                "first_year_return": _clean_float(ref_row["第一年收益%"], 2) if ref_row is not None else None,
            }

        rows.append(
            {
                "start_year": y,
                "start_date": start_date,
                "end_date": end_date,
                "trading_days": n_days,
                "hold_years": hold_years,
                "is_oos": y >= 2023,
                "pool_end_nav": pool_end,
                "pool_cagr": pool_cagr,
                "best_model_id": best_mid,
                "best_model_name": ML_MODEL_META[best_mid]["name"] if best_mid in ML_MODEL_META else best_mid,
                "models": models_row,
            }
        )

    # 10 个起点与 2016~2024（不含 2015 股灾）汇总统计
    def _agg(subset: list[dict[str, Any]]) -> dict[str, Any]:
        out_m: dict[str, dict[str, Any]] = {}
        for mid in ML_MODEL_IDS:
            cagrs = [r["models"][mid]["cagr"] for r in subset if r["models"][mid]["cagr"] is not None]
            excs = [r["models"][mid]["annual_excess"] for r in subset if r["models"][mid]["annual_excess"] is not None]
            dds = [r["models"][mid]["max_drawdown"] for r in subset if r["models"][mid]["max_drawdown"] is not None]
            wins = sum(1 for r in subset if r["best_model_id"] == mid)
            out_m[mid] = {
                "median_cagr": round(float(np.median(cagrs)), 2) if cagrs else None,
                "min_cagr": round(float(np.min(cagrs)), 2) if cagrs else None,
                "max_cagr": round(float(np.max(cagrs)), 2) if cagrs else None,
                "mean_cagr": round(float(np.mean(cagrs)), 2) if cagrs else None,
                "median_excess": round(float(np.median(excs)), 2) if excs else None,
                "median_dd": round(float(np.median(dds)), 2) if dds else None,
                "best_count": wins,
            }
        pool_cagrs = [r["pool_cagr"] for r in subset if r["pool_cagr"] is not None]
        out_m["pool_nav"] = {
            "median_cagr": round(float(np.median(pool_cagrs)), 2) if pool_cagrs else None,
            "min_cagr": round(float(np.min(pool_cagrs)), 2) if pool_cagrs else None,
            "max_cagr": round(float(np.max(pool_cagrs)), 2) if pool_cagrs else None,
            "mean_cagr": round(float(np.mean(pool_cagrs)), 2) if pool_cagrs else None,
        }
        return out_m

    ex2015 = [r for r in rows if r["start_year"] >= 2016]
    return {
        "available": bool(rows),
        "rule": rule,
        "capital": int(INITIAL_CAPITAL),
        "end_date": end_date,
        "years": [r["start_year"] for r in rows],
        "rows": rows,
        "summary": {
            "all_10": _agg(rows),
            "ex_2015": _agg(ex2015),
        },
        "note": (
            "每个起点均于当年首个交易日拿 20 万现金独立起步、连续交易至 2026-10-23（历史印花税口径），"
            "年化按 (期末资金/20万)^(242/交易日数)−1 折算。随机森林仅在 2015 年起步最高，2016~2024 任一起步年均垫底；"
            "2023 年以前起点仅供参考，正式统计检验以 2023~2026 测试段为准。"
        ),
    }


def get_nav_compare(
    rule: str = MAIN_BACKTEST_RULE,
    period: str = "test",
    start_year: int | None = None,
) -> dict[str, Any]:
    if not is_ml_available():
        raise ValueError("尚未就绪 data/ml/api_export 数据包，无法加载三模型净值对比")
    ml_api = _import_ml_api()
    if rule not in ml_api.RULES:
        raise ValueError(f"不支持的规则：{rule}；可选：{', '.join(ml_api.RULES)}")

    capital = 20_000.0 if rule == "DB-K3-b16" else INITIAL_CAPITAL

    if start_year is not None:
        if start_year not in START_YEARS:
            raise ValueError(f"不支持的起步年份：{start_year}；可选：{START_YEARS[0]} ~ {START_YEARS[-1]}")
        starts_df = _cached_starts_nav_df(rule)
        if starts_df is None or f"xgb_F69|{start_year}" not in starts_df.columns:
            raise ValueError(f"规则 {rule} 暂无 {start_year} 年独立起步净值曲线数据")
        valid = starts_df.loc[starts_df[f"xgb_F69|{start_year}"].notna()].reset_index(drop=True)
        dates = valid["date"].tolist()
        pool_vals = [_clean_float(v, 4) for v in valid[f"准入池|{start_year}"].tolist()]

        ref_df = _cached_starts_summary_df()
        yr_ref = (
            ref_df[(ref_df["规则"] == rule) & (ref_df["起步年"] == start_year)].set_index("model_id")
            if ref_df is not None
            else pd.DataFrame()
        )

        series: dict[str, dict[str, Any]] = {}
        raw_series_vals: dict[str, list[float | None]] = {}
        for mid in ML_MODEL_IDS:
            vals = [_clean_float(v, 4) for v in valid[f"{mid}|{start_year}"].tolist()]
            raw_series_vals[mid] = vals
            st = _nav_stats(vals, dates, pool_vals, capital=capital)
            if not yr_ref.empty and mid in yr_ref.index:
                ref_row = yr_ref.loc[mid]
                if pd.notna(ref_row.get("年化超额%")):
                    st["annual_excess"] = _clean_float(ref_row["年化超额%"], 2)
                if pd.notna(ref_row.get("第一年收益%")):
                    st["first_year_return"] = _clean_float(ref_row["第一年收益%"], 2)
                if pd.notna(ref_row.get("最大回撤%")):
                    st["max_drawdown"] = _clean_float(ref_row["最大回撤%"], 2)
            series[mid] = {
                "id": mid,
                "label": ML_MODEL_META[mid]["name"],
                "kind": "model",
                "values": vals,
                "cagr_values": _rolling_cagr_series(vals),
                "stats": st,
            }

        raw_series_vals["pool_nav"] = pool_vals
        series["pool_nav"] = {
            "id": "pool_nav",
            "label": "准入池等权（基准）",
            "kind": "benchmark",
            "values": pool_vals,
            "cagr_values": _rolling_cagr_series(pool_vals),
            "stats": _nav_stats(pool_vals, dates, None, capital=capital),
        }

        idx_rebased = _rebased_index_values(dates)
        for col, label in (("hs300", "沪深 300"), ("zz500", "中证 500"), ("zz1000", "中证 1000")):
            vals = idx_rebased.get(col)
            if vals:
                raw_series_vals[col] = vals
                series[col] = {
                    "id": col,
                    "label": label,
                    "kind": "index",
                    "values": vals,
                    "cagr_values": _rolling_cagr_series(vals),
                    "stats": _nav_stats(vals, dates, None, capital=capital),
                }

        n_days = max(len(dates) - 1, 1)
        hold_years = round(n_days / TRADING_DAYS_PER_YEAR, 1)
        if start_year >= 2023:
            note = (
                f"主规则 {rule} · {start_year} 年独立起步（{dates[0]} 拿 20 万现金起步 → {dates[-1]}，"
                f"共 {n_days} 个交易日 / {hold_years} 年，历史印花税口径，属 2023 年后样本外区间）"
            )
        elif start_year == 2016:
            note = (
                f"主规则 {rule} · 2016 年独立起步（{dates[0]} 拿 20 万现金起步 → {dates[-1]}，"
                f"共 {n_days} 个交易日 / {hold_years} 年，避开 2015 股灾且空仓躲过熔断首日；"
                "2023 年以前仅供参考，建议与 2015 年起步对照同看）"
            )
        else:
            note = (
                f"主规则 {rule} · {start_year} 年独立起步（{dates[0]} 拿 20 万现金起步 → {dates[-1]}，"
                f"共 {n_days} 个交易日 / {hold_years} 年，历史印花税口径；2023 年以前仅供参考）"
            )

        return {
            "rule": rule,
            "period": f"start_{start_year}",
            "start_year": start_year,
            "capital": int(capital),
            "cagr_min_days": CAGR_MIN_DAYS,
            "available_start_years": list(START_YEARS),
            "date_start": dates[0] if dates else "",
            "date_end": dates[-1] if dates else "",
            "count": len(dates),
            "trading_days": n_days,
            "hold_years": hold_years,
            "dates": dates,
            "series": series,
            "yearly_breakdown": _build_yearly_breakdown(dates, raw_series_vals, capital=capital),
            "note": note,
        }

    if period not in ("test", "long"):
        raise ValueError("period 只能是 test、long 或指定 start_year (2015~2024)")

    dfs = {mid: ml_api.nav(mid, rule=rule, period=period) for mid in ML_MODEL_IDS}
    base_df = dfs["xgb_F69"]
    dates = [pd.Timestamp(d).date().isoformat() for d in base_df["date"]]
    pool_vals = [_clean_float(v, 4) for v in base_df["pool_nav"]]

    series = {}
    raw_series_vals = {}
    for mid in ML_MODEL_IDS:
        vals = [_clean_float(v, 4) for v in dfs[mid]["nav"]]
        raw_series_vals[mid] = vals
        series[mid] = {
            "id": mid,
            "label": ML_MODEL_META[mid]["name"],
            "kind": "model",
            "values": vals,
            "cagr_values": _rolling_cagr_series(vals),
            "stats": _nav_stats(vals, dates, pool_vals, capital=capital),
        }

    raw_series_vals["pool_nav"] = pool_vals
    series["pool_nav"] = {
        "id": "pool_nav",
        "label": "准入池等权（基准）",
        "kind": "benchmark",
        "values": pool_vals,
        "cagr_values": _rolling_cagr_series(pool_vals),
        "stats": _nav_stats(pool_vals, dates, None, capital=capital),
    }

    if period == "long":
        for col, label in (("hs300", "沪深 300"), ("zz500", "中证 500"), ("zz1000", "中证 1000")):
            if col in base_df.columns:
                vals = [_clean_float(v, 4) for v in base_df[col]]
                raw_series_vals[col] = vals
                series[col] = {
                    "id": col,
                    "label": label,
                    "kind": "index",
                    "values": vals,
                    "cagr_values": _rolling_cagr_series(vals),
                    "stats": _nav_stats(vals, dates, None, capital=capital),
                }
    else:
        idx_rebased = _rebased_index_values(dates)
        for col, label in (("hs300", "沪深 300"), ("zz500", "中证 500"), ("zz1000", "中证 1000")):
            vals = idx_rebased.get(col)
            if vals:
                raw_series_vals[col] = vals
                series[col] = {
                    "id": col,
                    "label": label,
                    "kind": "index",
                    "values": vals,
                    "cagr_values": _rolling_cagr_series(vals),
                    "stats": _nav_stats(vals, dates, None, capital=capital),
                }

    n_days = max(len(dates) - 1, 1)
    return {
        "rule": rule,
        "period": period,
        "start_year": None,
        "capital": int(capital),
        "cagr_min_days": CAGR_MIN_DAYS,
        "available_start_years": list(START_YEARS),
        "date_start": dates[0] if dates else "",
        "date_end": dates[-1] if dates else "",
        "count": len(dates),
        "trading_days": n_days,
        "hold_years": round(n_days / TRADING_DAYS_PER_YEAR, 1),
        "dates": dates,
        "series": series,
        "yearly_breakdown": _build_yearly_breakdown(dates, raw_series_vals, capital=capital),
        "note": (
            f"主规则 {rule} · 正式测试段（{dates[0]} ~ {dates[-1]}，每年初重训，现行印花税口径，统计检验主基准）"
            if period == "test"
            else f"主规则 {rule} · 长期补充分析（{dates[0]} ~ {dates[-1]}，2023 年以前仅供参考；2015 年股灾最大回撤 -40% ~ -56%）"
        ),
    }


def get_day_consensus(
    as_of: str,
    *,
    top_n: int = 6,
    max_per_industry: int = 2,
    with_outcome: bool = False,
) -> dict[str, Any]:
    if not is_ml_available():
        raise ValueError("尚未就绪 data/ml/api_export 数据包，无法查询三模型共识")
    ml_api = _import_ml_api()
    clean_date = as_of.strip()
    if not clean_date:
        days = get_ml_dates("xgb_F69", limit=1)
        if not days:
            raise ValueError("三模型尚无可用日期")
        clean_date = days[0]

    picks_by_model: dict[str, list[dict[str, Any]]] = {}
    scores_lookup: dict[str, dict[str, dict[str, Any]]] = {}
    picked_sets: dict[str, set[str]] = {}

    for mid in ML_MODEL_IDS:
        sc = _cached_scores_df(mid, clean_date, with_outcome)
        lookup: dict[str, dict[str, Any]] = {}
        for row in sc.itertuples(index=False):
            rec: dict[str, Any] = {
                "ts_code": str(row.ts_code),
                "name": str(row.name or row.ts_code),
                "industry": str(row.industry or "（无行业）"),
                "rank": int(row.rank),
                "pct": round(float(row.pct) * 100.0, 2),
                "score": round(float(row.score), 5),
            }
            if with_outcome:
                ret = _clean_float(getattr(row, "ret_o1o5", None))
                exc = _clean_float(getattr(row, "excess_o1o5", None))
                rec["ret_o1o5_pct"] = round(ret * 100.0, 2) if ret is not None else None
                rec["excess_o1o5_pct"] = round(exc * 100.0, 2) if exc is not None else None
            lookup[rec["ts_code"]] = rec
        scores_lookup[mid] = lookup

        pk = ml_api.picks(mid, clean_date, top_n=top_n, max_per_industry=max_per_industry, with_outcome=with_outcome)
        pk_codes = [str(c) for c in pk["ts_code"]]
        picked_sets[mid] = set(pk_codes)
        picks_by_model[mid] = [lookup[c] for c in pk_codes if c in lookup]

    # 汇总三模型入选并集
    union_codes: list[str] = []
    seen: set[str] = set()
    for mid in ML_MODEL_IDS:
        for item in picks_by_model[mid]:
            if item["ts_code"] not in seen:
                seen.add(item["ts_code"])
                union_codes.append(item["ts_code"])

    consensus_rows: list[dict[str, Any]] = []
    for code in union_codes:
        picked_by = [mid for mid in ML_MODEL_IDS if code in picked_sets[mid]]
        per_model: dict[str, Any] = {}
        pcts: list[float] = []
        base_info: dict[str, Any] | None = None
        for mid in ML_MODEL_IDS:
            rec = scores_lookup[mid].get(code)
            if rec is not None:
                if base_info is None:
                    base_info = rec
                pcts.append(rec["pct"])
                per_model[mid] = {
                    "rank": rec["rank"],
                    "pct": rec["pct"],
                    "in_picks": code in picked_sets[mid],
                }
        assert base_info is not None
        row_obj: dict[str, Any] = {
            "ts_code": code,
            "name": base_info["name"],
            "industry": base_info["industry"],
            "picked_by": picked_by,
            "consensus_count": len(picked_by),
            "avg_pct": round(sum(pcts) / len(pcts), 2) if pcts else None,
            "models": per_model,
        }
        if with_outcome:
            row_obj["ret_o1o5_pct"] = base_info.get("ret_o1o5_pct")
            row_obj["excess_o1o5_pct"] = base_info.get("excess_o1o5_pct")
        consensus_rows.append(row_obj)

    consensus_rows.sort(key=lambda r: (-r["consensus_count"], -(r["avg_pct"] or 0.0), r["ts_code"]))

    # 回填每只票的共识度到分模型名单
    count_map = {r["ts_code"]: r["consensus_count"] for r in consensus_rows}
    picked_map = {r["ts_code"]: r["picked_by"] for r in consensus_rows}
    for mid in ML_MODEL_IDS:
        for item in picks_by_model[mid]:
            item["consensus_count"] = count_map.get(item["ts_code"], 1)
            item["picked_by"] = picked_map.get(item["ts_code"], [mid])

    pairwise = {
        "xgb_rf": len(picked_sets["xgb_F69"] & picked_sets["rf_F69"]),
        "xgb_lstm": len(picked_sets["xgb_F69"] & picked_sets["lstm_SF69"]),
        "rf_lstm": len(picked_sets["rf_F69"] & picked_sets["lstm_SF69"]),
        "all_three": len(picked_sets["xgb_F69"] & picked_sets["rf_F69"] & picked_sets["lstm_SF69"]),
    }

    return {
        "as_of": clean_date,
        "top_n": top_n,
        "max_per_industry": max_per_industry,
        "with_outcome": with_outcome,
        "models": {
            mid: {
                "meta": ML_MODEL_META[mid],
                "picks": picks_by_model[mid],
            }
            for mid in ML_MODEL_IDS
        },
        "consensus_rows": consensus_rows,
        "pairwise_overlap": pairwise,
    }


def get_stock_explain_all(
    as_of: str,
    ts_code: str,
    *,
    model_id: str | None = None,
    explain_only_model: str | None = None,
    top: int = 8,
    top_n: int = 6,
    max_per_industry: int = 2,
) -> dict[str, Any]:
    """查询单只股票在指定信号日下的三模型百分位对照 + Top-N 打分归因（TreeSHAP / Saabas / 积分梯度）。"""
    if not is_ml_available():
        return {"available": False, "as_of": as_of, "ts_code": ts_code, "models": {}}

    clean_code = ts_code.strip().upper()
    targets = (model_id,) if (model_id and model_id in ML_MODEL_IDS) else ML_MODEL_IDS

    models_out: dict[str, Any] = {}
    stock_name = clean_code
    stock_industry = ""

    for mid in targets:
        entry: dict[str, Any] = {
            "model_id": mid,
            "name": ML_MODEL_META[mid]["name"],
            "short_name": ML_MODEL_META[mid]["short_name"],
            "feature_set": ML_MODEL_META[mid]["feature_set"],
            "score_available": False,
            "explain_available": False,
            "explain_error": None,
            "rank": None,
            "pct": None,
            "score": None,
            "in_model_picks": False,
            "method": ML_MODEL_META[mid]["explain_method"],
            "base": None,
            "top_factors": [],
        }
        try:
            sc = _cached_scores_df(mid, as_of, False)
            hit = sc[sc["ts_code"] == clean_code]
            if not hit.empty:
                r = hit.iloc[0]
                stock_name = str(r["name"] or stock_name)
                stock_industry = str(r["industry"] or stock_industry)
                pk_codes = _cached_picks_codes(mid, as_of, top_n, max_per_industry)
                entry.update(
                    {
                        "score_available": True,
                        "rank": int(r["rank"]),
                        "pct": round(float(r["pct"]) * 100.0, 2),
                        "score": round(float(r["score"]), 5),
                        "in_model_picks": clean_code in pk_codes,
                    }
                )
        except Exception as exc:
            entry["explain_error"] = str(exc)

        if explain_only_model is None or explain_only_model == mid:
            try:
                day_map = _cached_explain_day(mid, as_of)
                exp_rec = day_map.get(clean_code)
                if exp_rec is not None:
                    entry.update(
                        {
                            "explain_available": True,
                            "method": exp_rec["method"],
                            "base": exp_rec["base"],
                            "score": exp_rec["score"],
                            "top_factors": exp_rec["top_factors"][:top],
                        }
                    )
                elif not entry["explain_error"]:
                    entry["explain_error"] = f"{mid} 在 {as_of} 没有 {clean_code} 的归因（仅覆盖测试段 2023-01-03 ~ 2026-10-16）"
            except Exception as exc:
                if not entry["explain_error"]:
                    entry["explain_error"] = str(exc)

        models_out[mid] = entry

    return {
        "available": True,
        "as_of": as_of,
        "ts_code": clean_code,
        "name": stock_name,
        "industry": stock_industry,
        "models": models_out,
    }


@functools.lru_cache(maxsize=1)
def get_feature_catalog() -> dict[str, Any]:
    """返回 F69 全部 69 个特征与 LSTM 11 个序列通道的结构化中文说明字典。"""
    if not is_ml_available():
        return {"available": False, "items": [], "by_name": {}, "groups": []}
    ml_api = _import_ml_api()
    df = ml_api.feature_info()
    items: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    groups_order: list[str] = []
    for rec in _clean_records(df):
        item = {
            "name": str(rec["名称"]),
            "source": str(rec["来源"]),
            "group": str(rec["分组"]),
            "cn_name": str(rec["中文名"]),
            "formula": str(rec["计算"]),
            "expected_sign": str(rec.get("预期方向") or ""),
            "reason": str(rec.get("理由") or ""),
        }
        items.append(item)
        by_name[item["name"]] = item
        if item["group"] not in groups_order:
            groups_order.append(item["group"])
    return {
        "available": True,
        "items": items,
        "by_name": by_name,
        "groups": groups_order,
    }


def _db_has_extra_tables(con: duckdb.DuckDBPyConnection) -> bool:
    try:
        rows = con.execute("SHOW TABLES").fetchall()
        names = {str(r[0]).lower() for r in rows}
        return {"daily", "adj_factor", "daily_basic", "moneyflow", "index_daily"}.issubset(names)
    except Exception:
        return False


@functools.lru_cache(maxsize=8)
def _cached_extra_panel_for_day(db_resolved: str, day_iso: str) -> pd.DataFrame | None:
    """按单日只读计算主板全部股票的 39 个基础新特征并缓存（避免切换模型重复扫描 DuckDB）。"""
    from ml.features_extra import compute_range

    con = duckdb.connect(db_resolved, read_only=True)
    try:
        if not _db_has_extra_tables(con):
            return None
        df = compute_range(con, date.fromisoformat(day_iso), date.fromisoformat(day_iso))
        return df
    finally:
        con.close()


def compute_extra_features_for_day(
    db_path: Path,
    day: date,
    pool_codes: Sequence[str],
    target_codes: Sequence[str],
) -> tuple[dict[str, dict[str, float | int]], dict[str, dict[str, float]]]:
    """计算指定信号日下 `target_codes` 的 41 个 F69 新增特征原值及准入池内 0~100 百分位。"""
    if not pool_codes or not target_codes:
        return {}, {}
    try:
        from ml.dataset import F28, f69
        from ml.features_extra import MARKET_STATE, add_ranks

        extra41 = [c for c in f69() if c not in F28]
    except Exception:
        return {}, {}

    raw_df = _cached_extra_panel_for_day(str(Path(db_path).resolve()), day.isoformat())
    if raw_df is None or raw_df.empty:
        return {}, {}

    pool_set = set(str(c) for c in pool_codes)
    sub = raw_df[raw_df["ts_code"].astype(str).isin(pool_set)].copy()
    if sub.empty:
        return {}, {}

    day_arr = np.full(len(sub), day.isoformat(), dtype=object)
    sub = add_ranks(sub, day_arr)
    sub["ts_code"] = sub["ts_code"].astype(str)
    sub = sub.set_index("ts_code")

    market_set = set(MARKET_STATE)
    rank_cols = [c for c in extra41 if c in sub.columns and c not in market_set]
    pct_table = pd.DataFrame(index=sub.index)
    for col in rank_cols:
        pct_table[col] = sub[col].rank(pct=True, method="average") * 100.0

    values_by_code: dict[str, dict[str, float | int]] = {}
    pct_by_code: dict[str, dict[str, float]] = {}

    for code in target_codes:
        c = str(code)
        if c not in sub.index:
            continue
        row = sub.loc[c]
        vals: dict[str, float | int] = {}
        pcts: dict[str, float] = {}
        for col in extra41:
            if col not in sub.columns:
                continue
            v = row[col]
            if v is None or pd.isna(v):
                continue
            num = float(v)
            if not math.isfinite(num):
                continue
            if col in ("ld20", "down_streak"):
                vals[col] = int(round(num))
            else:
                vals[col] = round(num, 4)
            if col in pct_table.columns:
                pv = pct_table.at[c, col]
                if pd.notna(pv) and math.isfinite(float(pv)):
                    pcts[col] = round(float(pv), 1)
        values_by_code[c] = vals
        pct_by_code[c] = pcts

    return values_by_code, pct_by_code


def build_ml_day_pick(
    model_id: str,
    as_of: date | None,
    *,
    top_n: int,
    max_per_industry: int,
) -> tuple[date, pd.DataFrame, pd.DataFrame, date]:
    """从 `ml.api` 拉取指定模型的全池评分与行业去重前 N 名单。

    返回 `(day, pool_df, picks_df, fit_date)`。
    """
    if not is_ml_model(model_id):
        raise ValueError(f"不是三模型 ID：{model_id}")
    if not is_ml_available():
        raise ValueError("尚未就绪 data/ml/api_export 数据包")

    ml_api = _import_ml_api()
    if as_of is None:
        days = get_ml_dates(model_id, limit=1)
        if not days:
            raise ValueError(f"{model_id} 没有可用交易日分数")
        day = date.fromisoformat(days[0])
    else:
        day = as_of

    pool_df = _cached_scores_df(model_id, day.isoformat(), False).copy()
    picks_df = ml_api.picks(model_id, day.isoformat(), top_n=top_n, max_per_industry=max_per_industry).copy()
    fit_date = date(2020, 12, 31) if day.year <= 2022 else date(day.year, 1, 1)
    return day, pool_df, picks_df, fit_date
