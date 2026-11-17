"""反弹模型：A 股主板短线选股的机器学习全流程。

    rebound.data       数据：Tushare -> DuckDB，增量、幂等、报缺口；表结构只在 data/schema.py
    rebound.universe   股票池：主板、按当时的名称排除 ST 和退市整理、成交额、价格下限
    rebound.features   28 个量价特征（只用信号日及以前的行情）
    rebound.labels     标签：T+1 开盘买、T+5 开盘卖的收益在当天股票池里的分位
    rebound.params     训练口径：股票池版本、XGBoost 参数、种子、重训日
    rebound.model      滚动训练、逐日打分、模型文件
    rebound.select     每日选股：分数从高到低、每个行业最多 2 只、取前 6
    rebound.backtest   回测：持股规则、真实账户逐日模拟、费用、统计、本金测算
    rebound.track      前向纸面跟踪
    rebound.cli        命令行（python -m rebound ...）

依赖只往下走：data 不依赖模型和回测，模型不依赖回测。
"""

from __future__ import annotations

from rebound.clock import use_beijing_time

# A 股的"今天""现在"一律按北京时间（今天收盘没有、数据新不新、选股是哪天），不跟这台电脑的时区走。
# 放在包一导入的地方：凡是用到 rebound 的进程（命令行、脚本、测试）都切成北京时间。见 rebound/clock.py。
use_beijing_time()

__version__ = "0.1.0"

__all__ = ["__version__"]
