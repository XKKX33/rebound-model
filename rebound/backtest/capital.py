"""`rebound capital`：给定本金，150 条持股规则 × 三种重训频率逐日模拟真实账户，按十年实验的标准选规则。

- 全期表：2015-01 起到最后一个打分日，每条规则的期末金额、年化、超额 [按月重抽区间]、β、回撤、逐年、费用、空仓。
- 滚动样本外：2017 ~ 2026 每年年初只用之前的数据选规则（四种选法），拼成样本外。四种都列出来；推荐固定按信息比率
  （`PICKER`）：十年实验的结论是信息比率挑出的是分散、换手低的组合，滚动选每次落在同一片；按超额挑会挑到集中持仓，
  哪组领先每年都在换。哪种选法这一次的滚动超额最高也列出来（`best_picker`），只作参考：每次都重新四选一，本身就是在挑最好的。
- 换起点体检：每月重训的 150 条，2017 / 2019 / 2021 / 2023 年第一个开市日用同样的本金重新起步（小本金的结果
  很依赖开头一两年：账户先涨大了，最低佣金就摊薄了）。
- 推荐（十年实验对两种本金用了两个标准）：
  - 有最低佣金、本金不到 10 万（`SMALL_CAPITAL`）：按换起点体检，推四个起点都赚、最差起点年化最高的那条。
    十年实验 2 万那一节："2 万选规则以换起点体检为准"——小账户的全期和滚动数字来自 2015 年起步、早早涨大的那个账户；
    30 只的规则 5 万还会亏光、10 万能用。
  - 其余（20 万这一类，或者免五）：按信息比率今天会选的那条（全期信息比率最高）；它在某个起点亏钱，就改推上面那条。
  - 两条不一样时都列出来，`basis` 记着按的是哪个标准。
- 每条规则：从 2015 年一路按它做的账户现在拿着什么、下一个开市日开盘买什么、尾盘卖什么（规则都是开盘买、尾盘卖，
  前一晚定好）；今天开新账户下一个开市日开盘买什么。只列清单，不下单。
结果按（本金、费率、准入池版本、打分到哪天、用的模型）存成 JSON，同样的输入再跑直接读。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rebound.data.errors import DataError
from rebound import params as model_params
from rebound.store import Store
from rebound.backtest import stats
from rebound.backtest.costs import CostModel
from rebound.backtest.market import candidates, load_markets, model_scores, pool_daily_return, stock_names
from rebound.backtest.plan import holdings, next_orders
from rebound.backtest.rules import Spec, specs
from rebound.backtest.sim import simulate

#: 模拟器、规则网格、选法、输出格式变了就加一，旧的缓存不再用。3：2026-11-01 起规则改成开盘买、尾盘卖（前一晚定）。
VERSION = 3
CADENCES: tuple[str, ...] = ("Y", "Q", "M")
FRESH_YEARS: tuple[int, ...] = (2017, 2019, 2021, 2023)
#: 第一个信号日（和十年实验相同：2015 年第一个开市日）。
SIGNAL_FROM = pd.Timestamp("2015-01-01")
#: 有最低佣金时，本金低于这个数按换起点体检选规则（十年实验：30 只的规则 5 万还会亏光，10 万能用）。
SMALL_CAPITAL = 100_000.0
#: 推荐用的选法：十年实验定的信息比率（`docs/return-model-strategy.md`）。
PICKER = "ir"
PICKER_NAMES = {"smooth": "平滑后超额", "raw": "不平滑的超额", "ir": "信息比率", "growth": "平滑后复利增长"}


class CapitalError(DataError):
    """本金测算的输入不对，或者模型打分还没准备好。"""


@dataclass(frozen=True)
class CapitalRequest:
    capital: float = 20_000.0
    variant: str = model_params.DEFAULT_VARIANT
    commission: float = 0.0005      # 佣金（双边）
    min_commission: float = 5.0     # 每笔最低佣金；免五填 0
    stamp: float = 0.0005           # 印花税（卖出）
    transfer: float = 0.00001       # 过户费（双边）

    def check(self) -> None:
        if not 1_000 <= self.capital <= 1e9:
            raise CapitalError(f"本金 {self.capital:g} 元不在 1000 元 ~ 10 亿元之间")
        if self.variant not in model_params.VARIANTS:
            raise CapitalError(f"准入池版本 {self.variant!r} 未知；可选：{sorted(model_params.VARIANTS)}")
        for name, value, high in (("commission", self.commission, 0.01), ("stamp", self.stamp, 0.01),
                                  ("transfer", self.transfer, 0.001), ("min_commission", self.min_commission, 100)):
            if not 0 <= value <= high:
                raise CapitalError(f"{name}={value:g} 不在 0 ~ {high:g} 之间")

    @property
    def small(self) -> bool:
        """小账户：最低佣金让结果很依赖从哪年起步，选规则以换起点体检为准。"""
        return self.min_commission > 0 and self.capital < SMALL_CAPITAL

    @property
    def cost(self) -> CostModel:
        return CostModel(capital=float(self.capital), commission=self.commission, min_commission=self.min_commission,
                         stamp=self.stamp, transfer=self.transfer)


def cache_root(db_path: Path) -> Path:
    db_path = Path(db_path)
    return db_path.with_name(f"{db_path.stem}.capital")


def _inputs(db_path: Path, variant: str) -> dict[str, Any]:
    """打分到哪天、每种频率用到的模型：变了就要重算。"""
    with Store(db_path) as store:
        if not store.has_table("model_score"):
            raise CapitalError("库里还没有反弹模型的打分：先跑 rebound model train，再跑 rebound model score")
        rows = store.connection.execute(
            "SELECT cadence, max(trade_date), count(*), min(trade_date) FROM model_score WHERE variant = ? GROUP BY 1",
            [variant]).fetchall()
        models = store.connection.execute(
            "SELECT DISTINCT model_id FROM model_score WHERE variant = ? ORDER BY 1", [variant]).fetchall()
        # 模拟用到的行情和口径也要进指纹：改了行业（行业上限）、补了历史名称（准入池基准）、改了日线或复权因子，
        # 同样的本金和分数也不该再读旧结果。行数 + 最后一天抓得住追加和整天补漏；某一行的数改了抓不住。
        tables = {t: [str(v) for v in store.connection.execute(f"SELECT count(*), max({col}) FROM {t}").fetchone()]
                  for t, col in (("daily", "trade_date"), ("adj_factor", "trade_date"), ("daily_limit", "trade_date"),
                                 ("namechange", "start_date"))}
        industry = hashlib.sha256(json.dumps(store.connection.execute(
            "SELECT ts_code, coalesce(industry, '') FROM stock_basic ORDER BY 1").fetchall()).encode()).hexdigest()[:12]
    found = {c: {"last": str(last), "rows": int(n), "first": str(first)} for c, last, n, first in rows}
    missing = [c for c in CADENCES if c not in found]
    if missing:
        raise CapitalError(f"{model_params.VARIANT_NAMES[variant]}的 {missing} 还没打分：先跑 rebound model score")
    lasts = {found[c]["last"] for c in CADENCES}
    if len(lasts) != 1:
        raise CapitalError(f"三种重训频率打分到的日子不一样（{lasts}）：先跑 rebound model score 补齐")
    digest = hashlib.sha256(json.dumps([m for (m,) in models]).encode()).hexdigest()[:12]
    return {"scores": found, "last": lasts.pop(), "models": digest, "tables": tables, "industry": industry}


def fingerprint(request: CapitalRequest, inputs: dict[str, Any]) -> str:
    blob = json.dumps({"version": VERSION, "request": asdict(request), "inputs": inputs}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _check_stop(stop: Any) -> None:
    if stop is not None:
        stop.checkpoint()


def cached(db_path: Path, request: CapitalRequest) -> dict[str, Any] | None:
    """同样的输入算过就直接读；没算过返回 None。"""
    request.check()
    path = cache_root(db_path) / f"{fingerprint(request, _inputs(db_path, request.variant))}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load(db_path: Path, key: str) -> dict[str, Any]:
    if not key.isalnum():
        raise CapitalError(f"测算编号 {key!r} 不合法")
    path = cache_root(db_path) / f"{key}.json"
    if not path.is_file():
        raise CapitalError(f"没有编号 {key} 的测算结果")
    out = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(out, dict) or out.get("version") != VERSION:  # 旧版本是另一套规则或时点算的：不拿来给清单
        raise CapitalError(f"编号 {key} 是旧版本的测算结果（规则或买卖时点变过）：重新运行一次")
    return out


def run(db_path: Path, request: CapitalRequest, *, log: Callable[[str], None] | None = None,
        stop: Any = None, refresh: bool = False) -> dict[str, Any]:
    """算一次本金测算（同样的输入算过就直接读，`refresh=True` 强制重算）。返回结果，也存成 JSON。"""
    say = log or (lambda _line: None)
    request.check()
    inputs = _inputs(db_path, request.variant)
    key = fingerprint(request, inputs)
    path = cache_root(db_path) / f"{key}.json"
    if path.is_file() and not refresh:
        say(f"[本金测算] 同样的输入算过了（编号 {key}），直接读")
        return json.loads(path.read_text(encoding="utf-8"))
    t0 = time.time()
    m = load_markets(db_path, end=date.fromisoformat(inputs["last"]), log=say, stop=stop)[request.variant]
    names, pool, dates = stock_names(db_path), pool_daily_return(m), m.dates
    today = len(dates) - 1
    # 最后一个能成交的信号日是昨天：今天的信号下一个开市日才成交（开盘买、尾盘卖）。计分日一直到今天（研究版少最后 5 天，那是标签要用的）
    d_first, d_last = int(dates.searchsorted(SIGNAL_FROM)), today - 1
    idx = stats.scoring_days(d_first, d_last)
    starts = {y: max(int(dates.searchsorted(pd.Timestamp(y, 1, 1))), d_first) for y in FRESH_YEARS}
    grid, cost = specs(), request.cost
    configs, book, found, fresh = [], [], [], {}
    total, done = len(grid) * (len(CADENCES) + len(FRESH_YEARS)), 0
    for cadence in CADENCES:
        cands, rank = candidates(m, model_scores(db_path, m, request.variant, cadence))
        for spec in grid:
            _check_stop(stop)
            res = simulate(m, cands, rank, spec.rule, cost, d_first, d_last)
            configs.append((cadence, spec))
            book.append(res.nav[idx] / res.nav[idx - 1] - 1)
            found.append({"summary": stats.summary(res, pool, dates, idx),
                          "account": {"holdings": holdings(m, names, res.positions, today),
                                      "orders": next_orders(m, names, cands, rank, spec.rule, cost, res.positions,
                                                            res.tranche_cash, today)},
                          "new": next_orders(m, names, cands, rank, spec.rule, cost, (), (), today)})
            done += 1
            if done % 50 == 0:
                say(f"[本金测算] {done}/{total} 组（{time.time() - t0:.0f}s）")
        if cadence == "M":
            for spec in grid:
                _check_stop(stop)
                for year, d0 in starts.items():
                    res = simulate(m, cands, rank, spec.rule, cost, d0, d_last)
                    fresh[("M", spec.name), year] = stats.window(res.nav, pool, d0, today, dates)
                done += len(starts)
                if done % 50 < len(starts):
                    say(f"[本金测算] {done}/{total} 组（{time.time() - t0:.0f}s）")
        del cands, rank
    book = np.vstack(book)
    if not np.isfinite(book).all():
        raise CapitalError("逐日收益里有空值：模拟没跑完整")
    walked, full = _walk(configs, book, pool[idx], dates[idx])
    best = int(np.argmax(full[PICKER]))
    cadence, spec = configs[best]
    if cadence != "M":  # 换起点体检只算了每月重训的：推荐的这条另外补四个起点
        cands, rank = candidates(m, model_scores(db_path, m, request.variant, cadence))
        for year, d0 in starts.items():
            res = simulate(m, cands, rank, spec.rule, cost, d0, d_last)
            fresh[(cadence, spec.name), year] = stats.window(res.nav, pool, d0, today, dates)
        del cands, rank
    out = _assemble(configs, book, pool, dates, idx, found, fresh, walked, full, best, d_first, small=request.small)
    out.update({"version": VERSION, "key": key, "request": asdict(request), "inputs": inputs,
                "variant_name": model_params.VARIANT_NAMES[request.variant], "today": str(dates[today].date()),
                "fresh_years": list(FRESH_YEARS), "seconds": round(time.time() - t0)})
    path.parent.mkdir(parents=True, exist_ok=True)
    # 先写摘要再写结果：列表按结果文件列，结果在、摘要缺了会从结果补；反过来摘要先到、结果没写成的不会列出来
    _write_json(_meta_path(path), _meta(out, time.strftime("%Y-%m-%dT%H:%M:%S")))
    _write_json(path, out)
    say(f"[本金测算] 算完（{out['seconds']}s），编号 {key}")
    return out


def _write_json(path: Path, value: Any) -> None:
    # 临时文件名带线程号：页面上两个请求可能同时给同一份老结果补摘要
    tmp = path.with_name(f"{path.stem}.tmp-{os.getpid()}-{threading.get_ident()}.json")
    tmp.write_text(json.dumps(value, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _meta_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.meta.json")


def _meta(out: dict[str, Any], created_at: str) -> dict[str, Any]:
    """一次测算的摘要（本金测算页的下拉用）：输入、打分到哪天、推荐哪条。"""
    pick = out["rows"][out["recommended"]] if out.get("recommended") is not None else None
    return {"key": out["key"], "version": out.get("version"), "request": out["request"], "today": out["today"],
            "variant_name": out.get("variant_name"), "created_at": created_at,
            "recommended": None if pick is None else {"cadence": pick["cadence"], "name": pick["name"],
                                                      "rule": pick["rule"], "plain": pick["plain"]}}


def runs(db_path: Path) -> dict[str, Any]:
    """算过的测算，打分日新的在前（本金测算页的下拉）。

    只列当前版本（`VERSION`）的结果：旧版本的缓存永远不会再被同样的输入命中，页面也不认它们的格式。
    老结果没有摘要文件的，读一遍结果补上（只补一次）。读不了的文件不跳过不报：放进 `problems`。
    """
    root = cache_root(db_path)
    found, problems = [], []
    for path in sorted(root.glob("*.json")) if root.is_dir() else ():
        if not path.stem.isalnum():  # 摘要（<编号>.meta.json）和写到一半的临时文件
            continue
        meta_path = _meta_path(path)
        try:
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime))
                out = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(out, dict):
                    raise TypeError(f"结果不是一个对象（是 {type(out).__name__}）")
                meta = _meta(out, stamp) if out.get("version") == VERSION else {"key": path.stem,
                                                                                 "version": out.get("version")}
                _write_json(meta_path, meta)
            if not isinstance(meta, dict):
                raise TypeError(f"摘要不是一个对象（是 {type(meta).__name__}）")
            if meta.get("version") == VERSION and not (isinstance(meta.get("request"), dict)
                                                       and isinstance(meta.get("key"), str)):
                raise TypeError("摘要缺 request 或 key")
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            problems.append(f"{path.name}：{type(exc).__name__}: {exc}")
            continue
        if meta.get("version") == VERSION:
            found.append(meta)
    found.sort(key=lambda m: (str(m.get("today")), str(m.get("created_at"))), reverse=True)
    return {"runs": found, "problems": problems}


def _walk(configs: list[tuple[str, Spec]], book: np.ndarray, pool: np.ndarray,
          dates: pd.DatetimeIndex) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    """四种选法：滚动样本外（逐年选中的规则）和全期打分（最后一年之后会选的就是全期最高的那条）。"""
    excess = book - pool
    near = stats.neighbor_index(configs)
    # 2017 起到数据里最后一年（十年实验是 2017 ~ 2026；写死的话，2027 年起最新那年就不进样本外了）
    years = tuple(range(stats.TEST_YEARS[0], int(dates[-1].year) + 1))
    walked = {}
    for picker in stats.PICKERS:
        w = stats.walk_forward(book, excess, dates, near, picker, years=years)
        walked[picker] = {**stats.series_stats(w["book"], w["excess"], w["dates"]),
                          "picks": {y: [configs[i][0], configs[i][1].short] for y, i in w["picks"].items()}}
    return walked, stats.scores(book, excess, near)


def recommend(rows: list[dict[str, Any]], best: int, *, small: bool) -> tuple[int | None, int | None, str]:
    """（推荐、换起点最稳的那条、按的标准）。最稳 = 四个起点都赚、最差起点年化最高（只算了每月重训的 150 条）。"""
    robust = [i for i, r in enumerate(rows) if r["fresh"] and r["fresh"]["all_positive"]]
    steady = max(robust, key=lambda i: rows[i]["fresh"]["worst_cagr"]) if robust else None
    if small:
        return steady, steady, "fresh"
    best_ok = bool(rows[best]["fresh"] and rows[best]["fresh"]["all_positive"])
    return (best, steady, "picker") if best_ok else (steady, steady, "fresh")


def _assemble(configs, book, pool, dates, idx, found, fresh, walked, full, best, d_first, *,
              small: bool) -> dict[str, Any]:
    excess = book - pool[idx]
    sd = excess.std(axis=1, ddof=1)
    whole = [str(y) for y in np.unique(dates[idx].year) if (dates[idx].year == y).sum() >= 200]

    def fresh_of(i: int) -> dict[str, Any] | None:
        cadence, spec = configs[i]
        got = {str(y): fresh.get(((cadence, spec.name), y)) for y in FRESH_YEARS}
        if any(v is None for v in got.values()):
            return None
        worst = min(v["cagr"] for v in got.values())
        return {"starts": got, "worst_cagr": worst, "all_positive": worst > 0}

    rows = []
    for i, ((cadence, spec), item) in enumerate(zip(configs, found)):
        s = item["summary"]
        yearly = [s["years"][y]["excess"] for y in whole if y in s["years"]]
        rows.append({"cadence": cadence, "cadence_name": model_params.CADENCE_NAMES[cadence], "name": spec.short,
                     "rule": spec.name, "plain": spec.plain, "family": spec.family, "k": spec.k,
                     "ir": round(float(excess[i].mean() / sd[i] * stats.YEAR ** 0.5), 2) if sd[i] > 0 else None,
                     "smooth": round(float(full["smooth"][i]) * stats.YEAR * 100, 2),
                     "worst_year": min(yearly) if yearly else None,
                     "years_positive": f"{sum(v > 0 for v in yearly)}/{len(yearly)}",
                     **s, "fresh": fresh_of(i), "account": item["account"], "new": item["new"]})
    picker = stats.best_of(walked)
    pick, steady, basis = recommend(rows, best, small=small)
    robust = [r for r in rows if r["fresh"] and r["fresh"]["all_positive"]]
    pool_nav = np.full(len(pool), np.nan)
    pool_nav[d_first] = 1.0
    pool_nav[d_first + 1:] = np.cumprod(1 + np.nan_to_num(pool[d_first + 1:]))
    return {"start": str(dates[d_first].date()), "first": str(dates[idx[0]].date()), "last": str(dates[idx[-1]].date()),
            "whole_years": whole,
            "pool": stats.period(pool_nav, pool, dates, idx), "rows": rows, "walk_forward": walked,
            "best_picker": picker, "picker_choice": best, "steadiest": steady, "recommended": pick,
            "picker": PICKER, "basis": basis, "small_capital": small, "robust_count": len(robust)}
