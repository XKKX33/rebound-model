"""选股前把反弹模型补到最新：到期的重训日训掉，没打分的日子打上分。

每天收盘后的流水线（`rebound daily`、`rebound pick --refresh`）选股之前调这里：平常只补当天的分（几十秒），
每月第一个开市日多训两次（两个准入池版本各一次，几分钟）。

这几种情况拒绝，一行都不写，并说清先做什么：

- 缺的重训日太多（新库、或者很久没跑）：全量训练要几个小时，不该藏在一次选股里，先单独跑 `rebound model train`；
- 缺分的交易日太多：先单独跑 `rebound model score`；
- 要打分的日子行情不全（主板日线比前一个交易日少一截，或者有日线没复权因子）：分一旦写下就不再改，
  拿半截的准入池打分，那一天的名单就永远是半截的。

没有缺的重训日就不调训练：训练会逐个核对已训目录的口径（含 xgboost 版本），升级 xgboost 不该让每天的选股停掉。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from rebound import params as model_params
from rebound.store import Store
from rebound.model.artifacts import ModelError, list_blocks
from rebound.model.score import ScoreReport, fill_scores
from rebound.model.train import _NON_MAIN, TrainReport, active_fit, data_sessions, fit_days, train

#: 选股前最多顺手训几次（两个版本合计）：两个版本各落后一个月是 2 次。
MAX_NEW_FITS = 4
#: 选股前最多顺手补几个交易日的分（六条序列的并集）：一个季度左右，几分钟。
MAX_SCORE_DAYS = 60
#: 要打分的那天，主板日线行数至少是前一个交易日的这个比例（实测 2019 年以来最低 0.985，是节假日前后）。
MIN_SESSION_RATIO = 0.95
#: 要打分的那天，有复权因子的主板日线至少占这个比例（准入池内连接复权因子，缺了的票会悄悄掉出池子）。
MIN_FACTOR_COVERAGE = 0.995


@dataclass(frozen=True)
class RefreshReport:
    trained: TrainReport
    scored: ScoreReport


def missing_fits(db_path: Path) -> dict[str, list[date]]:
    """每个版本按计划该有、还没训的重训日。"""
    with Store(db_path) as store:
        fits = fit_days(data_sessions(store), "M")
    out = {}
    for variant in model_params.VARIANTS:
        have = {block.fit_date for block in list_blocks(db_path, variant)}
        out[variant] = [fit for fit in fits if fit not in have]
    return out


def pending_days(db_path: Path) -> list[date]:
    """六条序列里，最后一个有分的日子之后、已经有重训日可用的交易日（并集，升序）。"""
    with Store(db_path) as store:
        sessions = data_sessions(store)
        last = {}
        if store.has_table("model_score"):
            last = {(v, c): d for v, c, d in store.connection.execute(
                "SELECT variant, cadence, max(trade_date) FROM model_score GROUP BY 1, 2").fetchall()}
    days: set[date] = set()
    for variant in model_params.VARIANTS:
        for cadence in model_params.CADENCES:
            fits, after = fit_days(sessions, cadence), last.get((variant, cadence))
            days.update(d for d in sessions if (after is None or d > after) and active_fit(d, fits) is not None)
    return sorted(days)


def incomplete_days(db_path: Path, days: list[date]) -> list[str]:
    """`days` 里行情看起来不全的日子，每天一句话说明；都齐就是空列表。"""
    if not days:
        return []
    with Store(db_path) as store:
        previous = store.sessions(days[0], 2)
        start = previous[0] if previous else days[0]
        rows = store.connection.execute(
            f"SELECT d.trade_date, count(*), count(a.ts_code) FROM daily d "
            f"LEFT JOIN adj_factor a USING (ts_code, trade_date) "
            f"WHERE d.trade_date BETWEEN ? AND ? AND NOT regexp_matches(d.ts_code, '{_NON_MAIN}') GROUP BY 1",
            [start, days[-1]]).fetchall()
        sessions = store.sessions(days[-1], 1_000_000)
    counts = {day: (main, factored) for day, main, factored in rows}
    problems = []
    for day in days:
        main, factored = counts.get(day, (0, 0))
        at = sessions.index(day) if day in sessions else 0
        before = counts.get(sessions[at - 1], (0, 0))[0] if at else 0
        if before and main < before * MIN_SESSION_RATIO:
            problems.append(f"{day} 主板日线 {main} 行，前一个交易日 {before} 行")
        elif main and factored < main * MIN_FACTOR_COVERAGE:
            problems.append(f"{day} 主板日线 {main} 行，有复权因子的只有 {factored} 行")
    return problems


def refresh(db_path: Path, *, log: Callable[[str], None] | None = None,
            max_new_fits: int = MAX_NEW_FITS, max_score_days: int = MAX_SCORE_DAYS) -> RefreshReport:
    say = log or (lambda _line: None)
    missing = missing_fits(db_path)
    count = sum(len(days) for days in missing.values())
    if count > max_new_fits:
        first = min(day for days in missing.values() for day in days)
        raise ModelError(f"反弹模型缺 {count} 次重训（最早 {first}），不在选股里顺手补：全量训练要几个小时，"
                         "先单独跑 rebound model train，再跑 rebound model score")
    pending = pending_days(db_path)
    if len(pending) > max_score_days:
        raise ModelError(f"反弹模型有 {len(pending)} 个交易日还没打分（{pending[0]} ~ {pending[-1]}），"
                         "不在选股里顺手补：先单独跑 rebound model score")
    problems = incomplete_days(db_path, pending)
    if problems:
        raise ModelError("要打分的日子行情不全：" + "；".join(problems) + "。分一旦写下就不再改，这次不打分也不选股。"
                         "多半是同步时上游还没写完：删掉这天的日线和复权因子再 rebound sync；"
                         "确认就是这么多（比如大面积停牌），单独跑 rebound model score")
    trained = TrainReport()
    if count:
        say(f"[反弹模型] 到了重训日，先训 {count} 次（" + "、".join(
            f"{model_params.VARIANT_NAMES[v]} {d}" for v, days in missing.items() for d in days) + "），要几分钟。")
        trained = train(db_path, log=say)
    scored = fill_scores(db_path, log=say)
    if scored.days:
        say(f"[反弹模型] 补了 {scored.days} 个交易日的分，{scored.rows} 行。")
    return RefreshReport(trained=trained, scored=scored)
