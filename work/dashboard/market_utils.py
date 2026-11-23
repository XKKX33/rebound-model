"""Stock search and OHLC aggregation for the dashboard."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True, slots=True)
class Stock:
    ts_code: str
    symbol: str
    name: str
    industry: str
    market: str
    list_date: str | None

    def as_dict(self):
        return asdict(self)


def rank(stocks: Iterable[Stock], query: str):
    """Order exact codes, code prefixes, names and industries, with stable ties."""
    needle = (query or "").strip().casefold()
    if not needle:
        return []
    matched = []
    for stock in stocks:
        code, symbol, name, industry = (
            str(value or "").casefold()
            for value in (stock.ts_code, stock.symbol, stock.name, stock.industry)
        )
        conditions = (
            needle in (code, symbol),
            code.startswith(needle) or symbol.startswith(needle),
            name == needle,
            name.startswith(needle),
            needle in code or needle in symbol,
            needle in name,
            bool(industry) and needle in industry,
        )
        score = next((index for index, found in enumerate(conditions) if found), None)
        if score is not None:
            matched.append((stock, score))
    return sorted(matched, key=lambda item: (item[1], item[0].ts_code))


def aggregate(frame: pd.DataFrame, *, period: str, sums: Sequence[str] = ()):
    """Use actual trading dates and preserve missing amounts in weekly/monthly bars."""
    if period == "day":
        return frame
    if period not in {"week", "month"}:
        raise ValueError("周期应为日线、周线或月线")
    if frame.empty:
        return frame
    frequency = "W-FRI" if period == "week" else "ME"
    work = frame.assign(_date=pd.to_datetime(frame["trade_date"])).sort_values("_date")
    rules = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "pre_close": "first", "vol": "sum", "amount": "sum",
    }
    for name in sums:
        if name in frame.columns:
            rules[name] = lambda values: values.sum(min_count=1)
    rules = {name: operation for name, operation in rules.items() if name in frame.columns}
    grouped = work.set_index("_date").resample(frequency).agg(rules).dropna(subset=["close"])
    if grouped.empty:
        return frame.iloc[0:0]
    labels = work.groupby(pd.Grouper(key="_date", freq=frequency))["_date"].max()
    grouped["trade_date"] = [pd.Timestamp(day).date() for day in labels.reindex(grouped.index)]
    return grouped.reset_index(drop=True)[["trade_date", *rules]]
