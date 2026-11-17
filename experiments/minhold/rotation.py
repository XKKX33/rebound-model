"""诊断：早卖空出来的位置换上的新票，和卖掉的那只拿到第 4 天相比，谁赚得多。

minhold.rest_of_window 只量了卖掉的那只"如果接着拿"会怎样。早卖的好处还有另一半：同一个开盘买进的新票。
这里对每笔不到 4 天就卖的成交，取同一天开盘新买进的票（同一次模拟里的成交），量它们从这个开盘到原来那只的
第 4 天开盘、相对同期准入池赚多少（repl），再和卖掉的那只同一段的超额（gap）相减：rotation = repl − gap，
> 0 表示换得好。两段起止相同、准入池是同一个数，所以 rotation 就是两只票同一段收益之差。
新票之后自己什么时候卖不管（按拿到那天算）；那天没有新买进的不算。只解释机制，结论以完整模拟为准。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from minhold import Market


def rotation(m: Market, trades: pd.DataFrame, early: pd.DataFrame) -> pd.DataFrame:
    """early 是 rest_of_window 的输出（含 exit_day、target、stock、pool、gap）。加上 repl（同一天新买进的票同一段的
    平均超额）、n_new（几只新票）、rotation = repl − gap；那天没有新买进、或新票两头没有开盘价的，repl 是 NaN。"""
    o = m.price["open"]
    buys = {d: g.j.to_numpy() for d, g in trades.groupby("entry_day")}
    cache: dict[tuple[int, int], tuple[float, int]] = {}
    repl, n_new = [], []
    for a, b, pool in zip(early.exit_day.tolist(), early.target.tolist(), early.pool.tolist()):
        if (a, b) not in cache:
            js = buys.get(a, np.array([], dtype=np.int64))
            with np.errstate(invalid="ignore", divide="ignore"):
                r = o[b, js] / o[a, js] - 1 if len(js) else np.array([])
            r = r[np.isfinite(r)]
            cache[a, b] = (float(r.mean()) - pool if len(r) else np.nan, int(len(r)))
        value, count = cache[a, b]
        repl.append(value)
        n_new.append(count)
    out = early.assign(repl=repl, n_new=n_new)
    return out.assign(rotation=out.repl - out.gap)
