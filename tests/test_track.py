"""纸面跟踪（`rebound/track.py`）：起步日、和本金测算同一个模拟器、冻结清单、开始与结束。

用 conftest 的小世界（训好小模型、补好打分，到 2023-05-31），日历往后多排几天（2023-06），
好测"还没买"和"还没到起步日"。每个测试一份自己的库：跟踪写在库旁边的目录里。
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from conftest import TEST_FIRST_FIT

from rebound.data.db import WriteStore
from rebound.data.errors import DataError
from rebound.model.artifacts import models_root
from rebound.model.score import fill_scores
from rebound.model.train import open_store
from rebound import track
from rebound.backtest import capital
from rebound.backtest.market import candidates, load_markets, model_scores, stock_names
from rebound.backtest.plan import holdings, next_orders
from rebound.backtest.rules import parse
from rebound.backtest.sim import simulate

#: 小世界最后一个打分日之后，日历上多排的开市日。
LATER = [date(2023, 6, d) for d in (1, 2, 5, 6, 7, 8, 9)]
REQUEST = capital.CapitalRequest(capital=200_000.0)


@pytest.fixture(scope="module")
def scored_world(trained_world_pristine: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("track-world") / "market.duckdb"
    shutil.copy2(trained_world_pristine, path)
    shutil.copytree(models_root(trained_world_pristine), models_root(path))
    fill_scores(path, first_fit=TEST_FIRST_FIT)
    with WriteStore(path) as store:
        store.upsert("trade_cal", pd.DataFrame({"exchange": "SSE", "cal_date": LATER, "is_open": 1}))
    return path


@pytest.fixture
def world(scored_world: Path, tmp_path: Path) -> Path:
    """每个测试一份：库、模型目录都复制，跟踪目录从空的开始。"""
    path = tmp_path / "market.duckdb"
    shutil.copy2(scored_world, path)
    shutil.copytree(models_root(scored_world), models_root(path))
    return path


def _create(path: Path, start: date | None, rule: str = "DB-K2-b4", cadence: str = "M", **kw) -> track.Tracker:
    return track.create(path, rule=rule, cadence=cadence, request=REQUEST, start=start, **kw)


def test_the_start_snaps_to_an_open_day_and_needs_a_scored_day_before_it(world: Path) -> None:
    with open_store(world) as store:
        # 周六 → 下个周一买，信号是周五收盘
        assert track.resolve_start(store, "nocap", "M", date(2023, 1, 7)) == (date(2023, 1, 9), date(2023, 1, 6))
        # 第一个打分日当天不能起步（前一天没分），后一天可以
        with pytest.raises(track.TrackInputError, match="最早是 2022-09-02"):
            track.resolve_start(store, "nocap", "M", TEST_FIRST_FIT)
        assert track.resolve_start(store, "nocap", "M", date(2022, 9, 2)) == (date(2022, 9, 2), TEST_FIRST_FIT)
        with pytest.raises(track.TrackInputError, match="超出了交易日历"):
            track.resolve_start(store, "nocap", "M", date(2023, 7, 3))
        # 不填 = 最后一个打分日之后的那个开市日
        assert track.resolve_start(store, "nocap", "M", None) == (date(2023, 6, 1), date(2023, 5, 31))


@pytest.mark.parametrize(("rule", "cadence"), [("DB-K2-b4", "M"), ("FH-K2-h4", "M"), ("DB-K3-b4", "Q")])
def test_a_tracker_is_the_capital_simulator_started_on_that_day(world: Path, rule: str, cadence: str) -> None:
    """只读起步那年起的行情，和读全部历史、同一天起步直接调模拟器逐项相同（FH 的分批也一样）。"""
    tracker = _create(world, date(2023, 3, 1), rule, cadence)
    out = track.compute(world, tracker)
    assert out["state"] == "running" and out["signal_day"] == "2023-02-28" and out["scored_through"] == "2023-05-31"

    m = load_markets(world)["nocap"]
    cands, rank = candidates(m, model_scores(world, m, "nocap", cadence))
    d0, today = int(m.dates.searchsorted(pd.Timestamp(2023, 2, 28))), len(m.dates) - 1
    spec, cost = parse(rule), REQUEST.cost
    res = simulate(m, cands, rank, spec.rule, cost, d0, today - 1)
    names = stock_names(world)
    assert out["nav"]["value"] == [round(float(v), 2) for v in res.nav[d0:today + 1]]
    assert out["nav"]["dates"][0] == "2023-02-28" and out["nav"]["dates"][-1] == "2023-05-31"
    assert out["holdings"] == holdings(m, names, res.positions, today)
    # FH 的批号是"开市日下标 mod 批数"：窗口从哪天读起，批号就整体轮换一下，每批的票和钱不变
    plan = next_orders(m, names, cands, rank, spec.rule, cost, res.positions, res.tranche_cash, today)
    assert {k: v for k, v in out["plan"].items() if k != "tranche"} == {k: v for k, v in plan.items() if k != "tranche"}
    full = [(str(m.codes[int(t.j)]), m.dates[int(t.entry_day)].date().isoformat(),
             m.dates[int(t.exit_day)].date().isoformat(), round(float(t.net) * 100, 2)) for t in res.trades.itertuples()]
    assert [(t["ts_code"], t["bought"], t["sold"], t["return_pct"]) for t in out["trades"]] == full[::-1]
    assert out["summary"]["trades"] == len(full) > 0
    assert out["summary"]["value"] == round(float(res.nav[today]), 2)
    json.dumps(out, allow_nan=False)  # 页面用 JSON.parse：不能有 NaN


def test_before_the_first_buy_the_plan_is_a_new_accounts_buys(world: Path) -> None:
    out = track.compute(world, _create(world, None))
    assert out["state"] == "waiting" and out["start"] == "2023-06-01" and out["next_open"] == "2023-06-01"
    assert out["holdings"] == [] and out["summary"]["value"] == REQUEST.capital and out["summary"]["days"] == 0
    m = load_markets(world)["nocap"]
    cands, rank = candidates(m, model_scores(world, m, "nocap", "M"))
    fresh = next_orders(m, stock_names(world), cands, rank, parse("DB-K2-b4").rule, REQUEST.cost, (), (),
                        len(m.dates) - 1)
    assert out["plan"] == fresh and out["plan"]["buys"]


def test_a_start_after_the_next_open_waits_for_its_signal_day(world: Path) -> None:
    out = track.compute(world, _create(world, date(2023, 6, 6)))
    assert out["state"] == "pending" and out["signal_day"] == "2023-06-05"
    assert "plan" not in out and "summary" not in out and out["next_open"] == "2023-06-01"


def test_the_first_list_of_a_signal_day_is_frozen_and_a_changed_one_is_shown_next_to_it(world: Path) -> None:
    tracker = _create(world, date(2023, 5, 2))
    first = track.refresh(world)
    assert first["frozen"]["issued_now"] and first["frozen"]["matches"]
    plan_file = track.tracking_root(world) / tracker.track_id / "plans" / "2023-05-31.json"
    assert json.loads(plan_file.read_text(encoding="utf-8"))["plan"] == first["plan"]
    time.sleep(1.1)
    again = track.refresh(world)
    assert not again["frozen"]["issued_now"] and again["frozen"]["matches"]
    assert again["frozen"]["issued_at"] == first["computed_at"] != again["computed_at"]
    # 假装重新打分后，同一天算出的清单变了：原来冻结的那份不改，带出来给页面看
    record = json.loads(plan_file.read_text(encoding="utf-8"))
    record["plan"]["buys"] = [{"ts_code": "600999.SH", "shares": 100}]
    plan_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    changed = track.refresh(world)
    assert changed["frozen"]["matches"] is False
    assert changed["frozen"]["first_plan"]["buys"] == [{"ts_code": "600999.SH", "shares": 100}]
    latest = json.loads((track.tracking_root(world) / tracker.track_id / "latest.json").read_text(encoding="utf-8"))
    assert latest["frozen"]["matches"] is False


def test_multiple_trackers_stay_live_and_the_selected_one_can_change(world: Path) -> None:
    first = _create(world, date(2023, 5, 2))
    assert track.active(world) == first
    second = _create(world, date(2023, 5, 3), "DB-K3-b4")
    assert track.active(world) == second
    assert {t.track_id for t in track.live_trackers(world)} == {first.track_id, second.track_id}

    assert track.select(world, first.track_id) == first
    assert track.active(world) == first
    pointer = json.loads((track.tracking_root(world) / "selected.json").read_text(encoding="utf-8"))
    assert pointer["version"] == 1 and pointer["track_id"] == first.track_id

    closed = track.stop(world)
    assert closed.track_id == first.track_id and closed.closed_at
    assert track.active(world) == second
    closed = track.stop(world)
    assert closed.track_id == second.track_id and closed.closed_at and track.active(world) is None
    assert not (track.tracking_root(world) / "selected.json").exists()
    with pytest.raises(track.TrackError, match="没有在跟踪"):
        track.stop(world)
    with pytest.raises(track.TrackError, match="已经在"):
        track.stop(world, first.track_id)
    with pytest.raises(track.TrackInputError, match="不合法"):
        track.load(world, "../../etc")
    with pytest.raises(track.TrackInputError):
        _create(world, None, "DB-K7-b4")  # 不在网格里
    with pytest.raises(track.TrackInputError):
        _create(world, None, cadence="H")


def test_explicit_replacement_only_ends_the_previously_selected_tracker(world: Path) -> None:
    first = _create(world, date(2023, 5, 2))
    second = _create(world, date(2023, 5, 3), "DB-K3-b4")
    third = _create(world, date(2023, 5, 4), "DB-K2-b6", replace_active=True)
    states = {t.track_id: t.closed_at is None for t in track.list_trackers(world)[0]}
    assert states == {first.track_id: True, second.track_id: False, third.track_id: True}
    assert track.active(world) == third


def test_missing_selection_uses_legacy_rule_but_a_broken_pointer_is_not_guessed(world: Path) -> None:
    first = _create(world, date(2023, 5, 2))
    second = _create(world, date(2023, 5, 3), "DB-K3-b4")
    pointer = track.tracking_root(world) / "selected.json"
    pointer.unlink()
    assert track.active(world) in (first, second)

    pointer.write_text("[]", encoding="utf-8")
    body = track.payload(world)
    assert body["tracker"] is None
    assert {t["track_id"] for t in body["trackers"]} == {first.track_id, second.track_id}
    assert any("当前显示项" in problem for problem in body["problems"])

    track.select(world, first.track_id)
    assert track.active(world) == first


def test_a_broken_tracker_file_is_reported_not_skipped(world: Path) -> None:
    good = _create(world, date(2023, 5, 2))
    broken = track.tracking_root(world) / "20230101-000000-abcdef"
    broken.mkdir(parents=True)
    (broken / "tracker.json").write_text("{not json", encoding="utf-8")
    body = track.payload(world)
    assert body["tracker"]["track_id"] == good.track_id and body["result"] is None
    assert len(body["problems"]) == 1 and "20230101-000000-abcdef" in body["problems"][0]


def test_the_card_says_whether_capital_still_recommends_the_tracked_rule(world: Path, monkeypatch) -> None:
    tracker = _create(world, date(2023, 5, 2))
    mine = {"cadence": "M", "name": "DB-K2-b4", "rule": tracker.rule, "plain": ""}
    other = {**mine, "name": "DB-K3-b4", "rule": "DB-K3-b4-o1c5"}

    def fake(runs: list[dict]) -> None:
        monkeypatch.setattr(capital, "runs", lambda _path: {"runs": runs, "problems": []})

    request = {"capital": 200000.0, "variant": "nocap", "commission": 0.0005, "min_commission": 5.0,
               "stamp": 0.0005, "transfer": 0.00001}
    fake([{"key": "a", "today": "2023-05-31", "request": request, "recommended": mine}])
    assert track.payload(world)["recommendation"]["same"] is True
    fake([{"key": "b", "today": "2023-05-31", "request": request, "recommended": other}])
    assert track.payload(world)["recommendation"] == {"key": "b", "today": "2023-05-31", "pick": other, "same": False}
    fake([{"key": "c", "today": "2023-05-31", "request": {**request, "capital": 20000.0}, "recommended": other}])
    assert track.payload(world)["recommendation"] is None


def test_capital_runs_lists_current_results_and_backfills_their_summaries(tmp_path: Path) -> None:
    db = tmp_path / "market.duckdb"
    root = capital.cache_root(db)
    root.mkdir()
    row = {"cadence": "M", "name": "DB-K2-b8", "rule": "DB-K2-b8-o1c5", "plain": "最多 2 只"}
    good = {"version": capital.VERSION, "key": "aaaa", "request": {"capital": 20000.0}, "today": "2026-10-30",
            "variant_name": "不限价", "rows": [row], "recommended": 0}
    (root / "aaaa.json").write_text(json.dumps(good), encoding="utf-8")
    (root / "bbbb.json").write_text(json.dumps({"version": 1, "key": "bbbb"}), encoding="utf-8")
    (root / "cccc.json").write_text("{broken", encoding="utf-8")
    (root / "dddd.tmp-1.json").write_text("{half", encoding="utf-8")
    out = capital.runs(db)
    assert [r["key"] for r in out["runs"]] == ["aaaa"]
    assert out["runs"][0]["recommended"] == row and out["runs"][0]["today"] == "2026-10-30"
    assert (root / "aaaa.meta.json").is_file() and (root / "bbbb.meta.json").is_file()
    assert len(out["problems"]) == 1 and "cccc.json" in out["problems"][0]
    assert capital.runs(db)["runs"] == out["runs"]  # 第二次读摘要，不再读结果


# --------------------------------------------------------------------------- #
# 文件被改坏了（手改、同步盘、以后改格式）：报出来，不让页面 500、不让流水线失败
# --------------------------------------------------------------------------- #


def test_an_old_tracker_file_keeps_its_open_sell_and_new_trackers_need_the_new_name(world: Path) -> None:
    """2026-11-01 之前开的跟踪存的是老名字（开盘卖）：照原来的时点回放，不悄悄改成尾盘卖。新开的跟踪不收老名字。"""
    from rebound.labels import Cell
    from rebound.backtest.sim import Rule

    tracker = _create(world, date(2023, 3, 1))
    assert tracker.rule == "DB-K2-b4-o1c5"
    now = track.compute(world, tracker)
    path = track.tracking_root(world) / tracker.track_id / "tracker.json"
    path.write_text(json.dumps({**asdict(tracker), "rule": "DB-K2-b4-o1o5"}), encoding="utf-8")
    old = track.load(world, tracker.track_id)
    assert old.describe()["name"] == "DB-K2-b4"
    before = track.compute(world, old)
    m = load_markets(world)["nocap"]
    cands, rank = candidates(m, model_scores(world, m, "nocap", "M"))
    d0, today = int(m.dates.searchsorted(pd.Timestamp(2023, 2, 28))), len(m.dates) - 1
    res = simulate(m, cands, rank, Rule("DB", 2, Cell("open", 5, "open"), buffer=4.0), REQUEST.cost, d0, today - 1)
    assert before["summary"]["value"] == round(float(res.nav[today]), 2) != now["summary"]["value"]
    assert "sell_leg" not in before["plan"] and now["plan"]["sell_leg"] == "close"
    assert (old.describe()["sell_leg"], tracker.describe()["sell_leg"]) == ("open", "close")
    # 卖掉的每一笔：尾盘卖的成交价是卖出那天的收盘价（不复权），老时点是开盘价；买的都是开盘价
    day = {d.date().isoformat(): i for i, d in enumerate(m.dates)}
    code = {str(c): j for j, c in enumerate(m.codes)}
    for out, sell_leg in ((now, "close"), (before, "open")):
        assert out["trades"]
        for t in out["trades"]:
            j = code[t["ts_code"]]
            assert t["buy_px"] == round(float(m.raw["open"][day[t["bought"]], j]), 2)
            assert t["sell_px"] == round(float(m.raw[sell_leg][day[t["sold"]], j]), 2)
    with pytest.raises(track.TrackInputError, match="DB-K2-b4-o1c5"):
        _create(world, None, "DB-K2-b4-o1o5", replace_active=True)


def test_rule_names_of_the_wrong_shape_are_value_errors() -> None:
    for bad in ("DB", "FH-K6", "DB-Kx-b2", "DB-K2-b1e400", 7, None):
        with pytest.raises(ValueError):
            parse(bad)


def test_a_tracker_file_of_the_wrong_shape_is_a_problem_not_a_crash(world: Path) -> None:
    good = _create(world, date(2023, 5, 2))
    bad = track.tracking_root(world) / "20230101-000000-abcdef"
    bad.mkdir(parents=True)
    wrong_rule = json.dumps({**asdict(good), "track_id": bad.name, "rule": "DB"})
    for content in ("null", "5", '"rule"', '{"track_id": 7}', "[]", wrong_rule):
        (bad / "tracker.json").write_text(content, encoding="utf-8")
        trackers, problems = track.list_trackers(world)
        assert [t.track_id for t in trackers] == [good.track_id] and len(problems) == 1, content
        assert track.payload(world)["tracker"]["track_id"] == good.track_id
    track.refresh_all(world)
    assert track.payload(world)["result"]["state"] == "running"


def test_a_copied_tracker_folder_is_not_a_second_tracker(world: Path) -> None:
    good = _create(world, date(2023, 5, 2))
    root = track.tracking_root(world)
    shutil.copytree(root / good.track_id, root / "20230101-000000-abcdef")
    trackers, problems = track.list_trackers(world)
    assert [t.track_id for t in trackers] == [good.track_id]
    assert len(problems) == 1 and "对不上" in problems[0]


def test_a_damaged_frozen_list_is_reported_on_the_card(world: Path) -> None:
    tracker = _create(world, date(2023, 5, 2))
    plan_file = track.tracking_root(world) / tracker.track_id / "plans" / "2023-05-31.json"
    plan_file.parent.mkdir(parents=True)
    for content in ("", "[]", '{"issued_at": "x"}'):
        plan_file.write_text(content, encoding="utf-8")
        out = track.refresh(world)
        assert out["frozen"]["matches"] is None and "读不了" in out["frozen"]["error"], content
        assert plan_file.read_text(encoding="utf-8") == content  # 冻结的那份不被改写


def test_refresh_all_updates_every_live_tracker(world: Path) -> None:
    first = _create(world, date(2023, 5, 2))
    second = _create(world, date(2023, 5, 3), "DB-K3-b4")
    batch = track.refresh_all(world)
    assert batch["updated"] == 2
    assert set(batch["track_ids"]) == {first.track_id, second.track_id}
    for tracker in (first, second):
        latest = track.tracking_root(world) / tracker.track_id / "latest.json"
        assert json.loads(latest.read_text(encoding="utf-8"))["track_id"] == tracker.track_id


def test_refresh_all_finishes_independent_trackers_then_reports_failures(world: Path, monkeypatch) -> None:
    first = _create(world, date(2023, 5, 2))
    second = _create(world, date(2023, 5, 3), "DB-K3-b4")
    original = track.compute

    def compute(path: Path, tracker: track.Tracker, **kwargs):
        if tracker.track_id == first.track_id:
            raise track.TrackError("第一组坏了")
        return original(path, tracker, **kwargs)

    monkeypatch.setattr(track, "compute", compute)
    with pytest.raises(track.TrackBatchError, match="1 组更新成功，1 组失败") as caught:
        track.refresh_all(world)
    assert caught.value.failures[0]["track_id"] == first.track_id
    assert (track.tracking_root(world) / second.track_id / "latest.json").is_file()


def test_a_broken_capital_cache_does_not_hide_the_tracking_card(world: Path, monkeypatch) -> None:
    tracker = _create(world, date(2023, 5, 2))

    def broken(_path):
        raise TypeError("坏了")

    monkeypatch.setattr(capital, "runs", broken)
    body = track.payload(world)
    assert body["tracker"]["track_id"] == tracker.track_id and body["recommendation"] is None
    assert any("本金测算的推荐读不了" in p for p in body["problems"])


def test_capital_runs_skips_results_and_summaries_of_the_wrong_shape(tmp_path: Path) -> None:
    db = tmp_path / "market.duckdb"
    root = capital.cache_root(db)
    root.mkdir()
    (root / "aaaa.json").write_text("[]", encoding="utf-8")
    (root / "bbbb.json").write_text("{}", encoding="utf-8")
    (root / "bbbb.meta.json").write_text("[]", encoding="utf-8")
    (root / "cccc.json").write_text("{}", encoding="utf-8")
    (root / "cccc.meta.json").write_text(json.dumps({"version": capital.VERSION, "key": "cccc"}), encoding="utf-8")
    out = capital.runs(db)
    assert out["runs"] == [] and len(out["problems"]) == 3


def test_the_result_keeps_the_latest_trades_and_the_total(world: Path, monkeypatch) -> None:
    monkeypatch.setattr(track, "MAX_TRADES", 3)
    out = track.compute(world, _create(world, date(2023, 3, 1)))
    assert len(out["trades"]) == 3 < out["summary"]["trades"]
    assert out["trades"][0]["sold"] >= out["trades"][-1]["sold"]


def test_a_refresh_can_be_stopped_between_years_and_the_stop_is_not_swallowed(world: Path) -> None:
    _create(world, date(2022, 9, 2))  # 2022 年起步：读 2022、2023 两年

    class Count:
        def __init__(self) -> None:
            self.calls = 0

        def checkpoint(self) -> None:
            self.calls += 1

    count = Count()
    track.refresh(world, stop=count)
    assert count.calls == 2

    class Stopped(RuntimeError):
        pass

    class StopNow:
        def checkpoint(self) -> None:
            raise Stopped("停")

    with pytest.raises(Stopped):  # 停止不被吞成"跟踪没算成"
        track.refresh_all(world, stop=StopNow())


# --------------------------------------------------------------------------- #
# 账户摘要
# --------------------------------------------------------------------------- #


def test_the_summary_says_what_the_account_made_today(world: Path) -> None:
    out = track.compute(world, _create(world, date(2023, 3, 1)))
    values, s = out["nav"]["value"], out["summary"]
    assert s["day_profit"] == pytest.approx(values[-1] - values[-2], abs=0.011)
    assert s["day_pct"] == pytest.approx((values[-1] / values[-2] - 1) * 100, abs=0.011)
    waiting = track.compute(world, _create(world, None, replace_active=True))
    assert waiting["state"] == "waiting" and waiting["summary"]["day_profit"] is None
    assert waiting["summary"]["day_pct"] is None


