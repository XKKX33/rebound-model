"""纸面跟踪：选定持股规则和起步日，算出从那天起照规则做的账户现在怎样、下一个开市日开盘买什么、尾盘卖什么。

- 一次跟踪 = 持股规则 + 重训频率 + 准入池版本 + 本金与费率 + 第一次开盘买的那天（`start`）。
  起步日前一个开市日（`signal_day`）收盘时账户是本金，第一笔买单用那天的打分。
- 同一时间可以跟踪多组；根目录的 `selected.json` 只决定页面当前显示哪一组，不影响其他组继续更新。
- 纸面账户和本金测算用同一个逐日模拟器（`sim.simulate`）：开盘价买、收盘价卖（前一晚定好，尾盘收盘竞价卖）、
  按费率表收费、一手取整，买不进顺延到下一名，卖不出就拿着。所以起步日可以往回补：照清单做的，回放出来就是这个账户。
  2026-11-01 之前开的跟踪存的是开盘卖的老名字，照原来的时点回放（`rules.rule_of`）。
- 每个信号日发出的清单冻结成一个文件，之后不再改。重新打分后（比如修了准入池）同一天的清单变了，页面两个都给看。
  以后记实际成交，也往这本流水账里加。
- 存在库旁边的 `<库名>.tracking/`，不进库：页面上开始、结束跟踪，不和流水线抢写库。
"""

from __future__ import annotations

import bisect
import json
import os
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rebound.data.errors import DataError
from rebound import params as model_params
from rebound.store import Store
from rebound.model.train import open_store
from rebound.backtest import capital
from rebound.backtest.market import Market, candidates, load_markets, model_scores, pool_daily_return, stock_names
from rebound.backtest.plan import _stock, holdings, next_orders
from rebound.backtest.rules import Spec, parse, rule_of
from rebound.backtest.sim import SimResult, simulate

#: 结果的格式变了就加一：页面不认旧格式的结果，等下一次算。
VERSION = 1
#: 当前显示项的独立格式版本；它不改变任何账户本身。
SELECTION_VERSION = 1
#: 跟踪编号 = 建的时刻 + 6 位随机数。也是目录名：只许这几种字符（整串匹配），挡住 `../` 之类。
_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")
#: 创建、切换和结束都会改显示指针：同一个进程里两个请求（两个标签页）不能交错。
_LOCK = threading.RLock()
#: 结果里最多留多少笔卖掉的（新的在前）；总笔数在 `summary.trades`。从很早起步的跟踪，结果文件不会越长越大。
MAX_TRADES = 200


class TrackError(DataError):
    """跟踪的状态不让做（显示项损坏、没有能结束的、还没有打分）：接口回 409。"""


class TrackInputError(TrackError):
    """输入不对（起步日、规则名、本金和费率）：接口回 422。"""


class TrackBatchError(TrackError):
    """独立账户已尽量更新，但至少一组失败；成功件保留，整批不能冒充成功。"""

    def __init__(self, updated: list[dict[str, Any]], failures: list[dict[str, str]]) -> None:
        self.updated = updated
        self.failures = failures
        detail = "；".join(f"{row['track_id']}：{row['error']}" for row in failures)
        super().__init__(f"{len(updated)} 组更新成功，{len(failures)} 组失败；{detail}")


def tracking_root(db_path: Path) -> Path:
    db_path = Path(db_path)
    return db_path.with_name(f"{db_path.stem}.tracking")


@dataclass(frozen=True)
class Tracker:
    track_id: str
    rule: str                # 规则全名，如 DB-K2-b6-o1c5（2026-11-01 之前开的是 -o1o5，开盘卖）
    cadence: str             # Y / Q / M
    variant: str             # nocap / cap70
    capital: float
    commission: float
    min_commission: float
    stamp: float
    transfer: float
    start: str               # 第一次开盘买的那天（ISO 日期）
    signal_day: str          # 它前一个开市日：收盘时账户是本金
    created_at: str
    closed_at: str | None = None

    @property
    def spec(self) -> Spec:
        return parse(self.rule, legacy_ok=True)  # 2026-11-01 之前开的跟踪存的是开盘卖的老名字

    @property
    def request(self) -> capital.CapitalRequest:
        return capital.CapitalRequest(capital=self.capital, variant=self.variant, commission=self.commission,
                                      min_commission=self.min_commission, stamp=self.stamp, transfer=self.transfer)

    def check(self) -> None:
        """设置可能来自手改过的文件：哪一项不对都是 TrackInputError，不让别的异常从读文件的地方漏出去。"""
        try:
            if not _ID.fullmatch(self.track_id):
                raise TrackInputError(f"跟踪编号 {self.track_id!r} 不合法")
            if self.cadence not in model_params.CADENCES:
                raise TrackInputError(f"重训频率 {self.cadence!r} 未知；可选：{sorted(model_params.CADENCES)}")
            parse(self.rule, legacy_ok=True)
            self.request.check()
            for day in (self.start, self.signal_day):
                date.fromisoformat(day)
        except TrackInputError:
            raise
        except (ValueError, TypeError, DataError) as exc:
            raise TrackInputError(str(exc)) from exc

    def describe(self) -> dict[str, Any]:
        """`sell_leg`：卖在开盘（2026-11-01 之前开的跟踪）还是尾盘；短名两种时点一样，页面靠它说清是哪种。"""
        spec = self.spec
        return {**asdict(self), "name": spec.short, "plain": spec.plain, "sell_leg": rule_of(self.rule).cell.exit,
                "cadence_name": model_params.CADENCE_NAMES[self.cadence],
                "variant_name": model_params.VARIANT_NAMES[self.variant]}

    @classmethod
    def from_dict(cls, raw: Any) -> Tracker:
        if not isinstance(raw, dict):
            raise TrackInputError(f"跟踪设置不是一个对象（是 {type(raw).__name__}）")
        texts = ("track_id", "rule", "cadence", "variant", "start", "signal_day", "created_at")
        numbers = ("capital", "commission", "min_commission", "stamp", "transfer")
        bad = [k for k in texts if not isinstance(raw.get(k), str)]
        bad += [k for k in numbers if isinstance(raw.get(k), bool) or not isinstance(raw.get(k), (int, float))]
        if raw.get("closed_at") is not None and not isinstance(raw.get("closed_at"), str):
            bad.append("closed_at")
        if bad:
            raise TrackInputError(f"跟踪设置这几项缺了或类型不对：{bad}")
        tracker = cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})
        tracker.check()
        return tracker


# --------------------------------------------------------------------------- #
# 起步日
# --------------------------------------------------------------------------- #


def _calendar(store: Store) -> list[date]:
    """日历里全部开市日（含往后排的），升序。"""
    return list(store.sessions(date(9999, 12, 31), 10_000_000))


def _scored(store: Store, variant: str, cadence: str) -> tuple[date, date]:
    """这个版本、这种重训频率最早和最后一个有分的交易日。"""
    first, last = store.connection.execute(
        "SELECT min(trade_date), max(trade_date) FROM model_score WHERE variant = ? AND cadence = ?",
        [variant, cadence]).fetchone() if store.has_table("model_score") else (None, None)
    if last is None:
        raise TrackError(f"{model_params.VARIANT_NAMES[variant]}、{model_params.CADENCE_NAMES[cadence]}重训还没有打分："
                         "先跑 rebound model train，再跑 rebound model score")
    return first, last


def resolve_start(store: Store, variant: str, cadence: str, start: date | None) -> tuple[date, date]:
    """（第一次开盘买的那天，它前一个开市日）。不填 = 最后一个打分日之后的那个开市日；不开市的日子顺延到下一个开市日。"""
    sessions = _calendar(store)
    first, last = _scored(store, variant, cadence)
    wanted = start if start is not None else last + timedelta(days=1)
    at = bisect.bisect_left(sessions, wanted)
    if at >= len(sessions):
        raise TrackInputError(f"{wanted} 超出了交易日历（日历排到 {sessions[-1] if sessions else '—'}）：先 rebound sync 同步日历")
    if at == 0 or sessions[at - 1] < first:
        later = bisect.bisect_right(sessions, first)
        earliest = sessions[later] if later < len(sessions) else first
        raise TrackInputError(f"起步日最早是 {earliest}：前一个开市日要有打分（打分从 {first} 起）")
    return sessions[at], sessions[at - 1]


def create(db_path: Path, *, rule: str, cadence: str, request: capital.CapitalRequest,
           start: date | None = None, replace_active: bool = False) -> Tracker:
    """新增并选中一条规则；显式 `replace_active=True` 才结束此前显示的那一条。"""
    if cadence not in model_params.CADENCES:
        raise TrackInputError(f"重训频率 {cadence!r} 未知；可选：{sorted(model_params.CADENCES)}")
    try:
        spec = parse(rule)  # 老时点（开盘卖）的名字在这里就不收：只留给以前的跟踪文件回放
        request.check()
    except (ValueError, DataError) as exc:
        raise TrackInputError(str(exc)) from exc
    with open_store(db_path) as store:
        first_buy, signal = resolve_start(store, request.variant, cadence, start)
    with _LOCK:
        current = active(db_path) if replace_active else None
        tracker = Tracker(track_id=_new_id(), rule=spec.name, cadence=cadence, variant=request.variant,
                          capital=float(request.capital), commission=request.commission,
                          min_commission=request.min_commission, stamp=request.stamp, transfer=request.transfer,
                          start=first_buy.isoformat(), signal_day=signal.isoformat(), created_at=_now())
        tracker.check()
        # 先写账户再选中它：中间断了最多留下一个未显示的合法账户，不会让指针指向不存在的目录。
        _write(_dir(db_path, tracker.track_id) / "tracker.json", asdict(tracker))
        _write_selection(db_path, tracker)
        if current is not None:
            stop(db_path, current.track_id)
    return tracker


# --------------------------------------------------------------------------- #
# 存取
# --------------------------------------------------------------------------- #


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _new_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def _dir(db_path: Path, track_id: str) -> Path:
    if not (isinstance(track_id, str) and _ID.fullmatch(track_id)):
        raise TrackInputError(f"跟踪编号 {track_id!r} 不合法")
    return tracking_root(db_path) / track_id


def _write(path: Path, value: Any) -> None:
    """整个文件换掉（先写临时文件再改名）：读的人看到的要么是旧的、要么是新的，不会是半截。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp-{os.getpid()}-{threading.get_ident()}{path.suffix}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _selection_path(db_path: Path) -> Path:
    return tracking_root(db_path) / "selected.json"


def _write_selection(db_path: Path, tracker: Tracker | None) -> None:
    """原子写当前显示项；没有未结束账户时删除这个单一指针。"""
    path = _selection_path(db_path)
    if tracker is None:
        path.unlink(missing_ok=True)
        return
    _write(path, {"version": SELECTION_VERSION, "track_id": tracker.track_id, "selected_at": _now()})


def list_trackers(db_path: Path) -> tuple[list[Tracker], list[str]]:
    """全部跟踪（新建的在后）和读不了的那些（一条一句，不跳过不报）。"""
    root = tracking_root(db_path)
    found, problems = [], []
    for folder in sorted(root.iterdir()) if root.is_dir() else ():
        if not (folder.is_dir() and _ID.fullmatch(folder.name)):
            continue
        try:
            tracker = Tracker.from_dict(json.loads((folder / "tracker.json").read_text(encoding="utf-8")))
        except (OSError, ValueError, TrackError) as exc:
            problems.append(f"{folder.name}：{type(exc).__name__}: {exc}")
            continue
        if tracker.track_id != folder.name:  # 整个目录复制了一份：两个"在跟踪"共用一个编号，结束不掉其中一个
            problems.append(f"{folder.name}：里面的编号是 {tracker.track_id}，和目录名对不上，没算它")
            continue
        found.append(tracker)
    return found, problems


def live_trackers(db_path: Path) -> list[Tracker]:
    """全部未结束的跟踪；顺序沿用稳定的目录顺序。"""
    return [tracker for tracker in list_trackers(db_path)[0] if tracker.closed_at is None]


def _selected_id(db_path: Path) -> str | None:
    """读取显式显示指针；文件不存在表示升级前的数据，不表示损坏。"""
    path = _selection_path(db_path)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrackError(f"当前显示项读不了（{type(exc).__name__}: {exc}）：{path}") from exc
    if not isinstance(raw, dict) or raw.get("version") != SELECTION_VERSION:
        raise TrackError("当前显示项格式或版本不正确")
    track_id = raw.get("track_id")
    if not isinstance(track_id, str) or not _ID.fullmatch(track_id):
        raise TrackError("当前显示项的跟踪编号不合法")
    return track_id


def selected(db_path: Path) -> Tracker | None:
    """页面当前显示的跟踪；旧数据没有指针时沿用原来的目录最大值。"""
    live = live_trackers(db_path)
    track_id = _selected_id(db_path)
    if track_id is None:
        return max(live, key=lambda tracker: tracker.track_id) if live else None
    for tracker in live:
        if tracker.track_id == track_id:
            return tracker
    raise TrackError(f"当前显示项 {track_id} 不存在或已经结束")


def active(db_path: Path) -> Tracker | None:
    """兼容旧调用方：`active` 现在明确表示当前显示项。"""
    return selected(db_path)


def load(db_path: Path, track_id: str) -> Tracker:
    path = _dir(db_path, track_id) / "tracker.json"
    if not path.is_file():
        raise TrackError(f"没有编号 {track_id} 的跟踪")
    return Tracker.from_dict(json.loads(path.read_text(encoding="utf-8")))


def select(db_path: Path, track_id: str) -> Tracker:
    """选择页面显示项；只改指针，不重算、不结束任何账户。"""
    with _LOCK:
        tracker = load(db_path, track_id)
        if tracker.closed_at is not None:
            raise TrackError(f"跟踪 {track_id} 已经在 {tracker.closed_at} 结束了")
        _write_selection(db_path, tracker)
        return tracker


def stop(db_path: Path, track_id: str | None = None) -> Tracker:
    """结束跟踪（不填编号 = 当前显示项）。设置、结果和每天的清单都留着。"""
    with _LOCK:
        current = active(db_path)
        tracker = current if track_id is None else load(db_path, track_id)
        if tracker is None:
            raise TrackError("现在没有在跟踪的策略")
        if tracker.closed_at is not None:
            raise TrackError(f"跟踪 {tracker.track_id} 已经在 {tracker.closed_at} 结束了")
        closed = replace(tracker, closed_at=_now())
        _write(_dir(db_path, closed.track_id) / "tracker.json", asdict(closed))
        if current is not None and current.track_id == closed.track_id:
            remaining = live_trackers(db_path)
            next_tracker = max(remaining, key=lambda item: item.track_id) if remaining else None
            _write_selection(db_path, next_tracker)
    return closed


# --------------------------------------------------------------------------- #
# 算账户
# --------------------------------------------------------------------------- #


def compute(db_path: Path, tracker: Tracker, *, log: Callable[[str], None] | None = None,
            stop: Any = None) -> dict[str, Any]:
    """从起步日照规则做到最后一个打分日的纸面账户，和下一个开市日的清单（开盘买、尾盘卖）。只读库。

    `state`：pending = 起步日前一个开市日还没打分（清单那天收盘后才有）；waiting = 下一个开市日就是起步日，
    还没买；running = 已经买过。`stop`：后台任务的句柄，读行情时每读完一年看一次要不要停。
    """
    signal = date.fromisoformat(tracker.signal_day)
    with open_store(db_path) as store:
        last = _scored(store, tracker.variant, tracker.cadence)[1]
        sessions = _calendar(store)
    base = {"version": VERSION, "computed_at": _now(), "track_id": tracker.track_id, "tracker": tracker.describe(),
            "signal_day": tracker.signal_day, "start": tracker.start, "scored_through": last.isoformat(),
            "next_open": _next_open(sessions, last)}
    if signal > last:
        return {**base, "state": "pending"}
    m = load_markets(db_path, start=signal, end=last, log=log, stop=stop)[tracker.variant]
    cands, rank = candidates(m, model_scores(db_path, m, tracker.variant, tracker.cadence))
    d0, today = int(m.dates.searchsorted(pd.Timestamp(signal))), len(m.dates) - 1
    if d0 > today or m.dates[d0] != pd.Timestamp(signal):
        raise TrackError(f"行情里没有起步前一天 {signal}（交易日历里有）：先 rebound sync 补齐行情")
    rule, cost = rule_of(tracker.rule), tracker.request.cost  # 老跟踪文件存的老名字照原来的时点回放
    res = simulate(m, cands, rank, rule, cost, d0, today - 1)
    names, pool = stock_names(db_path), _pool_nav(m, d0, today)
    plan = next_orders(m, names, cands, rank, rule, cost, res.positions, res.tranche_cash, today)
    out = {**base, "state": "waiting" if today == d0 else "running",
           "summary": _summary(res, pool, d0, today), "nav": _nav(res, m, pool, d0, today),
           "holdings": holdings(m, names, res.positions, today), "plan": plan,
           "trades": _trades(m, names, res, rule.cell.entry, rule.cell.exit)[:MAX_TRADES]}
    out["frozen"] = _freeze(db_path, tracker, out)
    return out


def _next_open(sessions: list[date], day: date) -> str | None:
    at = bisect.bisect_right(sessions, day)
    return sessions[at].isoformat() if at < len(sessions) else None


def _num(value: float, digits: int = 2) -> float | None:
    """结果要能当严格的 JSON 读（页面用 JSON.parse）：算不出来的写 null，不写 NaN。"""
    return round(float(value), digits) if np.isfinite(value) else None


def _pool_nav(m: Market, d0: int, today: int) -> np.ndarray:
    """准入池等权（本金测算的基准）从起步前一天收盘 = 1 起的净值。"""
    pool = pool_daily_return(m)
    return np.concatenate([[1.0], np.cumprod(1 + np.nan_to_num(pool[d0 + 1:today + 1]))])


def _summary(res: SimResult, pool: np.ndarray, d0: int, today: int) -> dict[str, Any]:
    nav = res.nav[d0:today + 1]
    start, end = float(nav[0]), float(nav[-1])
    before = float(nav[-2]) if len(nav) > 1 else None  # 前一个收盘日：还没开始买（只有起步那天）就没有"今天"
    trades = res.trades
    return {"value": _num(end), "start_value": _num(start), "profit": _num(end - start),
            "day_profit": None if before is None else _num(end - before),
            "day_pct": None if before is None else _num((end / before - 1) * 100),
            "return_pct": _num((end / start - 1) * 100), "pool_pct": _num((pool[-1] - 1) * 100),
            "excess_pct": _num((end / start - pool[-1]) * 100),
            "max_dd_pct": _num(float((nav / np.maximum.accumulate(nav) - 1).min()) * 100),
            "cash": _num(res.cash[today]), "days": int(today - d0), "trades": int(len(trades)),
            "wins": int((trades.net > 0).sum()), "fees": _num(res.costs[d0 + 1:today + 1].sum()),
            "hold_days": _num(float((trades.exit_day - trades.entry_day).mean()), 1) if len(trades) else None,
            "holding_count": int(len(res.positions))}


def _nav(res: SimResult, m: Market, pool: np.ndarray, d0: int, today: int) -> dict[str, list]:
    """逐日净值（元）和准入池等权（起步前一天 = 1），画小图用。"""
    return {"dates": [d.date().isoformat() for d in m.dates[d0:today + 1]],
            "value": [_num(v) for v in res.nav[d0:today + 1]],
            "pool": [_num(v, 5) for v in pool]}


def _trades(m: Market, names: dict[str, tuple[str, str]], res: SimResult, entry: str, exit_: str) -> list[dict]:
    """卖掉了的每一笔，新的在前。价格是不复权的成交价；赚的钱扣过双边费用。"""
    rows = []
    for t in res.trades.itertuples(index=False):
        buy, sell = int(t.entry_day), int(t.exit_day)
        rows.append({**_stock(m, names, int(t.j)), "bought": m.dates[buy].date().isoformat(),
                     "sold": m.dates[sell].date().isoformat(), "buy_px": _num(m.raw[entry][buy, int(t.j)]),
                     "sell_px": _num(m.raw[exit_][sell, int(t.j)]), "return_pct": _num(t.net * 100),
                     "profit": _num(t.net * t.cost_basis), "late": bool(t.carried)})
    return rows[::-1]


def _plan_key(plan: dict[str, Any]) -> list[tuple[str, str, int]]:
    rows = [("卖", r["ts_code"], r["shares"] or 0) for r in plan["sells"]]
    rows += [("买", r["ts_code"], r["shares"] or 0) for r in plan["buys"]]
    return sorted(rows)


def _freeze(db_path: Path, tracker: Tracker, out: dict[str, Any]) -> dict[str, Any]:
    """这个信号日第一次算出的清单冻结成文件，之后不再改；再算出来不一样就把原来那份带出来。"""
    day = out["scored_through"]
    path = _dir(db_path, tracker.track_id) / "plans" / f"{day}.json"
    record = {"signal_day": day, "issued_at": out["computed_at"], "next_open": out["next_open"],
              "plan": out["plan"], "holdings": out["holdings"]}
    text = json.dumps(record, ensure_ascii=False, allow_nan=False)  # 先序列化：出错就不留下空文件
    mine = _plan_key(out["plan"])  # 刚算出来的：这里出错是代码的错，照样抛
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
        return {"signal_day": day, "issued_at": record["issued_at"], "issued_now": True, "matches": True}
    except FileExistsError:
        pass
    try:
        first = json.loads(path.read_text(encoding="utf-8"))
        same = _plan_key(first["plan"]) == mine
        issued = first.get("issued_at")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:  # 文件坏了、被改过：说清，不让整次失败
        return {"signal_day": day, "issued_at": None, "issued_now": False, "matches": None,
                "error": f"冻结的清单读不了（{type(exc).__name__}: {exc}）：{path}"}
    return {"signal_day": day, "issued_at": issued, "issued_now": False, "matches": same,
            "first_plan": None if same else first["plan"]}


def refresh(db_path: Path, track_id: str | None = None, *, log: Callable[[str], None] | None = None,
            stop: Any = None) -> dict[str, Any] | None:
    """重算一个跟踪（不填编号 = 当前显示项），结果写进 `latest.json`。没有显示项就返回 None。"""
    tracker = active(db_path) if track_id is None else load(db_path, track_id)
    if tracker is None:
        return None
    out = compute(db_path, tracker, log=log, stop=stop)
    _write(_dir(db_path, tracker.track_id) / "latest.json", out)
    if log is not None:
        about, summary = tracker.describe(), out.get("summary")
        tail = (f"账户 {summary['value']} 元，累计 {summary['return_pct']}%，下一个开市日买 {len(out['plan']['buys'])} 只、"
                f"卖 {len(out['plan']['sells'])} 只" if summary else
                f"{out['start']} 起步，{tracker.signal_day} 收盘后出第一份清单")
        log(f"[纸面跟踪] {about['cadence_name']}重训 + {about['name']}：打分到 {out['scored_through']}，{tail}")
    return out


def refresh_all(db_path: Path, *, log: Callable[[str], None] | None = None,
                stop: Any = None) -> dict[str, Any]:
    """更新调用开始时全部未结束账户；独立失败全部尝试后统一抛出。"""
    updated: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    results: dict[str, dict[str, Any]] = {}
    for tracker in live_trackers(db_path):
        if stop is not None:
            stop.checkpoint()
        try:
            out = refresh(db_path, tracker.track_id, log=log, stop=stop)
        except (DataError, OSError, ValueError, TypeError) as exc:
            failure = {"track_id": tracker.track_id, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            if log is not None:
                log(f"[纸面跟踪] {tracker.track_id} 更新失败：{failure['error']}")
            continue
        if out is not None:
            results[tracker.track_id] = out
            updated.append({"track_id": tracker.track_id, "state": out["state"],
                            "scored_through": out["scored_through"]})
    if failures:
        raise TrackBatchError(updated, failures)
    return {"updated": len(updated), "track_ids": [row["track_id"] for row in updated], "results": results}


def payload(db_path: Path) -> dict[str, Any]:
    """今天页：全部未结束项和当前显示项；只读文件，不开库、不写迁移指针。"""
    trackers, problems = list_trackers(db_path)
    live = [t for t in trackers if t.closed_at is None]
    try:
        current = selected(db_path)
    except TrackError as exc:
        current = None
        problems = [*problems, str(exc)]
    result, recommendation = None, None
    if current is not None:
        path = _dir(db_path, current.track_id) / "latest.json"
        try:
            loaded = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        except (OSError, ValueError) as exc:
            loaded, problems = None, [*problems, f"{path}：{type(exc).__name__}: {exc}"]
        result = loaded if isinstance(loaded, dict) and loaded.get("version") == VERSION else None
        try:  # 本金测算的结果目录坏了，不该连跟踪卡片也看不到
            recommendation = _recommendation(db_path, current)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            problems = [*problems, f"本金测算的推荐读不了：{type(exc).__name__}: {exc}"]
    closed = [t.describe() for t in trackers if t.closed_at is not None][-5:]
    ordered = sorted(live, key=lambda tracker: tracker.track_id, reverse=True)
    if current is not None:
        ordered = [current, *(tracker for tracker in ordered if tracker.track_id != current.track_id)]
    return {"tracker": None if current is None else current.describe(),
            "trackers": [tracker.describe() for tracker in ordered], "result": result,
            "recommendation": recommendation, "closed": closed[::-1], "problems": problems}


def _same_request(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("variant") != b.get("variant"):
        return False
    keys = ("capital", "commission", "min_commission", "stamp", "transfer")
    return all(isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float))
               and abs(a[k] - b[k]) <= 1e-9 * max(1.0, abs(a[k])) for k in keys)


def _recommendation(db_path: Path, tracker: Tracker) -> dict[str, Any] | None:
    """同样本金和费率最近一次本金测算推荐的规则（只是提示：跟踪的规则不会跟着换）。"""
    mine = asdict(tracker.request)
    for run in capital.runs(db_path)["runs"]:
        if _same_request(run["request"], mine):
            pick = run.get("recommended")
            return {"key": run["key"], "today": run["today"], "pick": pick,
                    "same": pick is not None and (pick["cadence"], pick["rule"]) == (tracker.cadence, tracker.rule)}
    return None
