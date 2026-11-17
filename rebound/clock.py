"""北京时间。A 股的"今天""现在"一律按北京时间算，不跟这台电脑的时区走（这台机器在美东）。

- `rebound` 包一导入就调 `use_beijing_time()`，把本进程的本地时区设成北京时间：之后的 `datetime.now()`、
  `date.today()`、`time.strftime()`、`.astimezone()` 都是北京时间。命令行、页面服务、脚本、测试都一样，
  启动前不用另外设 `TZ`。
- Windows 没有 `time.tzset()`；C 运行库第一次取本地时间时读一次 `TZ`，之后一直用读到的时区。进程在导入
  rebound 之前取过本地时间（先导入了别的库、pytest 等），只改环境变量就不起作用。所以直接调 C 运行库
  （ucrtbase）的 `_tzset()` 让它重读，什么时候调都生效。万一调不到，`local_is_beijing()` 是假，命令行启动时会警告。
- 最要紧的两处（数据新鲜度闸门判断"今天收盘没有"、同步补到哪天）直接用 `beijing_now()`，不靠进程时区。
- 中国不用夏令时：北京时间全年 UTC+8，`CST-8` 是 POSIX 写法（名字 CST、本地比 UTC 早 8 小时）。
- `time.timezone`、`time.tzname` 这几个模块常量在 Windows 上仍是电脑的时区（Python 导入 time 时就定了），
  系统里没用它们；要时区对象就用这里的 `BEIJING`。
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from datetime import datetime, timedelta, timezone

BEIJING = timezone(timedelta(hours=8), "北京时间")
TZ = "CST-8"


def use_beijing_time() -> None:
    """把本进程的本地时区设成北京时间。什么时候调都生效，调几次都行。"""
    os.environ["TZ"] = TZ
    if hasattr(time, "tzset"):
        time.tzset()
    elif sys.platform == "win32":
        _reload_windows_timezone()


def _reload_windows_timezone() -> None:
    """让 Windows 的 C 运行库按新的 `TZ` 重算本地时区。

    调不到（极少见）就不管：本地时间仍是电脑的时区，`local_is_beijing()` 是假，命令行启动时会警告；
    新鲜度闸门和同步用的是 `beijing_now()`，不受影响。
    """
    try:
        ctypes.CDLL("ucrtbase")._tzset()
    except (OSError, AttributeError):
        return


def beijing_now() -> datetime:
    """北京时间的现在：不带时区的墙上时间，和库里、页面上的日期时间同一种写法。"""
    return datetime.now(BEIJING).replace(tzinfo=None)


def local_is_beijing(tolerance_seconds: float = 120.0) -> bool:
    """本进程的本地时间是不是北京时间。"""
    return abs((datetime.now() - beijing_now()).total_seconds()) <= tolerance_seconds
