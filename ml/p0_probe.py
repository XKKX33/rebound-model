"""任务 0.5 计算量摸底：只算一年（2020）的 28 个原特征，测行数、耗时、内存，推算全量。

运行：python -X utf8 -m ml.p0_probe
输出：ml/results/p0_setup/compute_probe.md
"""

from __future__ import annotations

import time
from datetime import date

import numpy as np

from ml.common import DB_PATH, connect, register, result_dir, rss_gb, write_md
from rebound import params as model_params
from rebound.features import eligible, feature_panel
from rebound.model.train import open_store


def main() -> None:
    out = result_dir("p0_setup")
    mem0 = rss_gb()
    t0 = time.time()
    with open_store(DB_PATH) as store:
        panel = feature_panel(store, date(2020, 1, 1), date(2020, 12, 31))
    secs = time.time() - t0
    mem1 = rss_gb()
    mask = eligible(panel, model_params.VARIANTS["nocap"], min_bars=model_params.MIN_BARS)
    n_all, n_pool = len(panel), int(mask.sum())
    days = panel["trade_date"].nunique()
    panel_mb = panel.memory_usage(deep=True).sum() / 1024**2

    con = connect()
    n_days_total = con.execute("SELECT count(DISTINCT trade_date) FROM daily").fetchone()[0]
    con.close()
    est_rows = n_pool / days * n_days_total
    est_mb_75 = est_rows * (75 + 6) * 4 / 1024**2   # 75 个 float32 特征 + 几列键 / 标签

    text = f"""
# 0.5 计算量摸底

运行：`python -X utf8 -m ml.p0_probe`（只算 2020 年一年的 28 个原特征）

| 项目 | 数值 |
|---|---|
| 2020 年交易日数 | {days} |
| 主板全部行数 | {n_all:,} |
| 其中不限价准入池（nocap）行数 | {n_pool:,}（每天约 {n_pool / days:,.0f} 只） |
| 一年特征耗时 | {secs:.0f} 秒 |
| 进程内存 | 计算前 {mem0:.2f} GB → 计算后 {mem1:.2f} GB；一年面板 DataFrame 约 {panel_mb:,.0f} MB |

## 推算全量（{n_days_total} 个交易日）

- 准入池总行数约 **{est_rows / 1e6:.1f} 百万行**。
- 75 个特征按 float32 存约 **{est_mb_75 / 1024:.1f} GB**（在 15 GB 内存的机器上可以整体放进内存，但要按年分批计算）。
- 28 个原特征全量计算约 {secs * n_days_total / days / 60:.0f} 分钟（按年分批，内存峰值约等于一年）。

## 由此定下的做法（写进后续任务）

1. 面板和新特征都**按年分批**计算、分批写 parquet，避免一次性占满内存（本机 15 GB，日常空闲 3 ~ 8 GB）。
2. 随机森林、MLP 训练时如果超出 30 分钟，按交易日分层抽样（比例在 P3 摸底后写进 config）。
3. XGBoost 用 `hist` 方法可直接吃全量。
"""
    write_md(out / "compute_probe.md", text)
    register("0.5_probe", year=2020, days=days, rows_all=n_all, rows_pool=n_pool, seconds=round(secs, 1),
             est_rows_total=int(est_rows), mem_gb_after=round(mem1, 2))


if __name__ == "__main__":
    main()
