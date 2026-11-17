"""北京时间：系统的"今天""现在"不跟这台电脑的时区走。"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rebound import clock
from rebound.data import freshness

ROOT = Path(__file__).resolve().parents[1]
BEIJING_OFFSET = 8 * 3600


def test_beijing_now_is_utc_plus_eight() -> None:
    gap = clock.beijing_now() - datetime.now(timezone.utc).replace(tzinfo=None)
    assert abs(gap - timedelta(hours=8)) < timedelta(seconds=5)


@pytest.mark.parametrize("before", ["", "import time; time.localtime()\n"], ids=["rebound-first", "local-time-first"])
def test_a_process_runs_on_beijing_time_whatever_the_machine_timezone_is(before: str) -> None:
    """电脑在美东（这里用 TZ 模拟）：导入 rebound 之后，datetime.now() 和 time.localtime() 都是北京时间。

    `local-time-first`：导入 rebound 之前就取过本地时间（Windows 的 C 运行库那时已记住电脑的时区），照样要切过来。
    """
    code = before + (
        "import rebound, datetime, time\n"
        "utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)\n"
        "local = datetime.datetime.now()\n"
        "c_local = datetime.datetime(*time.localtime()[:6])\n"
        "print((local - utc).total_seconds(), (c_local - utc).total_seconds())\n"
    )
    env = {**os.environ, "TZ": "EST5EDT", "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT,
                         check=True).stdout.split()
    py_gap, c_gap = (float(item) for item in out)
    assert abs(py_gap - BEIJING_OFFSET) < 120
    assert abs(c_gap - BEIJING_OFFSET) < 120


def test_the_freshness_gate_asks_the_beijing_clock(monkeypatch) -> None:
    """闸门判断"今天收盘没有"用的是北京时间的现在，不是本机时间。"""
    monkeypatch.setattr(freshness, "beijing_now", lambda: datetime(2026, 11, 11, 16, 30))

    class NoCalendar:
        def open_dates(self, **_kw: object) -> tuple:
            return ()

    gate = freshness.check_freshness(NoCalendar())
    assert gate.today.isoformat() == "2026-11-11"
    assert (gate.now.hour, gate.now.minute) == (16, 30)
