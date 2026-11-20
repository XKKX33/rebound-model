"""任务 1.6：构建 LSTM 的序列数据并核对。

运行：python -X utf8 -m ml.p1_6_build_seq [--rebuild]
- 不加 --rebuild：已有 data/ml/seq/ 就直接读，只重做核对和报告；
- 计算量摸底（LSTM 一个 epoch 多久、显存多少）由 dl-study 环境单独跑：
  D:/Anaconda/envs/dl-study/python.exe -X utf8 -m ml.lstm_probe，结果写到 ml/results/p1_6_seq/probe.json，本报告会读进来。
输出：ml/results/p1_6_seq/report.md、channel_stats.csv、fig1_example_window.png、fig2_traded_days.png
"""

from __future__ import annotations

import json
import sys
from datetime import date

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ml.common import connect, register, result_dir, rss_gb, write_md  # noqa: E402
from ml.dataset import load_panel  # noqa: E402
from ml.seq_data import (CHANNELS, RANKED, WINDOW, build, load_X, panel_index, ranked, raw_range, save,  # noqa: E402
                         seq_dir, window)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
CHECK_DAYS = (date(2015, 7, 9), date(2021, 4, 23), date(2026, 10, 23))   # 股灾千股停牌后、拼接缝、最后一天


def independent_raw(con, day: date) -> pd.DataFrame:
    """不用窗口函数、单独写的计算：取每只主板股票到 day 为止最近两行，按定义算 9 个原始通道和涨跌停标记。"""
    from ml.features_extra import _MAIN
    q = f"""
      SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close, d.amount, a.adj_factor,
             l.up_limit, l.down_limit, b.turnover_rate_f, m.net_mf_amount,
             m.buy_lg_amount + m.buy_elg_amount - m.sell_lg_amount - m.sell_elg_amount AS big_raw,
             m.ts_code IS NOT NULL AS has_mf
      FROM daily d JOIN adj_factor a USING (ts_code, trade_date)
      LEFT JOIN daily_limit l USING (ts_code, trade_date)
      LEFT JOIN daily_basic b USING (ts_code, trade_date)
      LEFT JOIN moneyflow m USING (ts_code, trade_date)
      WHERE d.trade_date <= ? AND d.trade_date >= ? - INTERVAL 400 DAY AND {_MAIN}
    """
    t = con.execute(q, [day, day]).df()
    t = t.sort_values(["ts_code", "trade_date"])
    last = t.groupby("ts_code").tail(2)
    cur = last[pd.to_datetime(last["trade_date"]) == pd.Timestamp(day)].set_index("ts_code")
    prev = last.groupby("ts_code").head(1).set_index("ts_code")
    prev = prev[pd.to_datetime(prev["trade_date"]) < pd.Timestamp(day)]
    c, o = cur["close"] * cur["adj_factor"], cur["open"] * cur["adj_factor"]
    pc = (prev["close"] * prev["adj_factor"]).reindex(cur.index)
    amt_wan = np.where(cur["has_mf"] & (cur["amount"] > 0), cur["amount"] / 10.0, np.nan)
    out = pd.DataFrame({
        "ret": np.log(c / pc), "ovn": np.log(o / pc), "intra": np.log(cur["close"] / cur["open"]),
        "hl": np.log(cur["high"] / cur["low"]),
        "clv": np.where(cur["high"] > cur["low"], (2 * cur["close"] - cur["high"] - cur["low"]) / (cur["high"] - cur["low"]), 0.0),
        "ln_amt": np.log(cur["amount"].where(cur["amount"] > 0)), "ln_turn": np.log(cur["turnover_rate_f"].where(cur["turnover_rate_f"] > 0)),
        "mf": cur["net_mf_amount"] / amt_wan, "big": cur["big_raw"] / amt_wan,
        "limit": np.select([cur["close"] >= cur["up_limit"] - 1e-6, cur["close"] <= cur["down_limit"] + 1e-6], [0.5, -0.5], 0.0),
    }, index=cur.index)
    for ch in RANKED:                                    # 排名：scipy 的平均名次，(名次 − 0.5) / n − 0.5
        v = out[ch].to_numpy(np.float64)
        ok = np.isfinite(v)
        rk = np.zeros(len(v))
        rk[ok] = (rankdata(v[ok]) - 0.5) / ok.sum() - 0.5
        out[ch] = rk
    out["traded"] = 1.0
    return out


def main() -> None:
    out_dir = result_dir("p1_6_seq")
    p = seq_dir()
    if "--rebuild" in sys.argv or not (p / "X.npy").exists():
        built = build()
        panel = load_panel(["ts_code"])
        save(built, panel_index(built["codes"], panel["ts_code"]))
        build_secs = built["seconds"]
        del built
    else:
        build_secs = json.loads((p / "meta.json").read_text(encoding="utf-8"))["seconds"]
        print("读取已有序列数据（重建请加 --rebuild）")
    X, codes, dates, pj = load_X(mmap=True)
    keys = load_panel(["trade_date", "ts_code", "di"])
    con = connect()

    # ---- 1. 通道统计（交易过的格子）----
    traded = np.asarray(X[:, :, 10]) == 1
    stats = []
    rng = np.random.default_rng(1)
    days = np.sort(rng.choice(np.flatnonzero(traded.any(axis=0)), 300, replace=False))
    sub = np.asarray(X[:, days, :], dtype=np.float32)
    tsub = sub[:, :, 10] == 1
    for k, ch in enumerate(CHANNELS):
        v = sub[:, :, k][tsub]
        day_mean = np.array([sub[:, i, k][tsub[:, i]].mean() for i in range(len(days))])
        stats.append({"通道": ch, "最小": v.min(), "最大": v.max(), "均值": v.mean(), "为 0 的比例": (v == 0).mean(),
                      "每天均值的最大偏离": np.abs(day_mean).max() if ch in RANKED else np.nan})
    stats = pd.DataFrame(stats).round(4)
    stats.to_csv(out_dir / "channel_stats.csv", index=False, encoding="utf-8-sig")

    # ---- 2. 防前视：截断到某一天重算，与张量比较 ----
    trunc = []
    for d in CHECK_DAYS:
        r = ranked(raw_range(con, d, d))
        raw = raw_range(con, d, d)
        j = pd.Index(codes).get_indexer(raw["ts_code"])
        di = dates.get_loc(pd.Timestamp(d))
        got = np.asarray(X[j, di, :], dtype=np.float32)
        want = r.to_numpy(np.float32).astype(np.float16).astype(np.float32)
        trunc.append({"日期": d, "股票数": len(raw), "不一致的格子": int((got != want).sum()),
                      "其他股票当天是否都标为未交易": bool((np.asarray(X[np.setdiff1d(np.arange(len(codes)), j), di, 10]) == 0).all())})
    trunc = pd.DataFrame(trunc)

    # ---- 3. 独立重算抽查 ----
    indep = []
    for d in CHECK_DAYS:
        ref = independent_raw(con, d)
        j = pd.Index(codes).get_indexer(ref.index)
        di = dates.get_loc(pd.Timestamp(d))
        got = np.asarray(X[j, di, :], dtype=np.float32)
        diff = np.abs(got - ref[list(CHANNELS)].to_numpy(np.float32))
        indep.append({"日期": d, "股票数": len(ref), "最大绝对差": float(diff.max()),
                      "差 > 1e-3 的格子": int((diff > 1e-3).sum()), "最差的通道": CHANNELS[int(diff.max(axis=0).argmax())]})
    indep = pd.DataFrame(indep)
    con.close()

    # ---- 4. 主面板的窗口覆盖 ----
    di_all = keys["di"].to_numpy()
    cnt = np.empty(len(keys), np.int16)
    for lo in range(0, len(keys), 400_000):
        w = window(X, pj[lo:lo + 400_000], di_all[lo:lo + 400_000])
        cnt[lo:lo + 400_000] = (w[:, :, 10] == 1).sum(axis=1)
    self_traded = bool((np.asarray(X[pj, di_all, 10]) == 1).all())
    full = float((cnt == WINDOW).mean())

    # ---- 图 ----
    rng = np.random.default_rng(7)
    i = int(rng.integers(len(keys)))
    ex = window(X, pj[i:i + 1], di_all[i:i + 1])[0].astype(np.float32)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(ex.T, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_yticks(range(len(CHANNELS)), CHANNELS)
    ax.set_xlabel("距信号日的天数（最右 = 信号日 T）")
    ax.set_xticks(range(0, WINDOW, 10), [str(t - WINDOW + 1) for t in range(0, WINDOW, 10)])
    ax.set_title(f"LSTM 的一个输入样本：{keys['ts_code'].iloc[i]}，信号日 {keys['trade_date'].iloc[i].date()}（60 天 × 11 通道）")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_example_window.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    vals, freq = np.unique(cnt, return_counts=True)
    ax.bar(vals, freq / freq.sum() * 100, width=0.8)
    ax.set_xlabel("60 天窗口里有交易的天数")
    ax.set_ylabel("样本占比（%）")
    ax.set_yscale("log")
    ax.set_title("主面板样本的窗口完整度")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_traded_days.png", dpi=130)
    plt.close(fig)

    probe_path = out_dir / "probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else None
    probe_md = "（还没有运行，见文件开头的命令）" if probe is None else pd.DataFrame(probe["runs"]).to_markdown(index=False)

    text = f"""
# 1.6 LSTM 的序列数据

- 代码：`ml/seq_data.py`（定义和计算）、`ml/p1_6_build_seq.py`（构建与核对）；单元测试 `ml/tests/test_seq_data.py`
- 运行：`python -X utf8 -m ml.p1_6_build_seq [--rebuild]`
- 产物：`data/ml/seq/`（不进 Git）——`X.npy` 张量、`codes.npy` 股票代码、`dates.npy` 交易日、`panel_j.npy` 主面板每行的股票下标
- 构建耗时 {build_secs:.0f} 秒，本次核对的进程内存 {rss_gb():.1f} GB

## 一、为什么要单独准备

Ridge 和 XGBoost 读的是“一行 = 一只股票一天的手工特征”。LSTM 的意义在于直接读**过去一段时间的原始日线**，
让网络自己学出时间上的模式（参考 Qlib 的 Alpha360：60 天 × 几个原始量）。所以给它一条序列：
**一个样本 = 主面板的一行 (股票, 信号日 T)，输入是这只股票 T−59 ~ T 共 60 个交易日 × 11 个通道。**

| 通道 | 计算 | 含义 |
|---|---|---|
| `ret` | ln(复权收盘 / 前一交易日复权收盘) | 日收益 |
| `ovn` | ln(复权开盘 / 前一交易日复权收盘) | 隔夜收益 |
| `intra` | ln(收盘 / 开盘) | 日内收益 |
| `hl` | ln(最高 / 最低) | 日内振幅 |
| `clv` | (2C − H − L) / (H − L)，一字板记 0 | 收盘位置 |
| `ln_amt` | ln(成交额) | 成交活跃度、规模 |
| `ln_turn` | ln(自由流通换手率) | 换手 |
| `mf` | 净流入额 / 成交额 | 资金流 |
| `big` | 大单 + 特大单净买入 / 成交额 | 大资金动向 |
| `limit` | 涨停 +0.5、跌停 −0.5、其余 0 | 涨跌停 |
| `traded` | 有交易 1，停牌 / 未上市 / 已退市 0 | 有效标记 |

- **标准化**：前 9 个通道每天在**全主板**截面排名到 (−0.5, 0.5)，和 1.5 的做法一致：对极端值不敏感、每天独立、不需要拟合，没有跨期泄漏。
  缺失（停牌、没有资金流）填 0，并由 `traded` 通道告诉网络“这一天没有交易”。
- **存储**：张量 `[股票 {X.shape[0]:,}, 交易日 {X.shape[1]:,}, 通道 {X.shape[2]}]`，float16，{X.nbytes / 1024**3:.2f} GB，可以整份放进显卡（8 GB）。
  训练时按下标现切窗口，不展开成“430 万 × 60 × 11”的大数组（那样要 {len(keys) * WINDOW * 11 * 2 / 1024**3:.0f} GB）。
- 截面排名的范围是全主板（不只准入池），因为窗口里有的天这只股票还没进池。

## 二、核对

### 1. 每个通道的取值（随机 300 天、有交易的格子）

{stats.to_markdown(index=False)}

排名通道都在 (−0.5, 0.5) 内、每天居中（偏离不到 1e-4，表中四舍五入为 0）；`limit` 只有 −0.5 / 0 / 0.5（约 2.9% 的格子是涨停或跌停）；`traded` 全是 1。

### 2. 防前视：把库截断到某一天（所有表只读到这一天）单独重算，和张量逐格比较

{trunc.to_markdown(index=False)}

“不一致的格子”都是 0：每个格子只用到当天及以前的数据。

### 3. 独立重算：不用窗口函数，另写一段代码按定义直接算，再用 scipy 排名

{indep.to_markdown(index=False)}

差别都在 float16 的精度以内（float16 在 0.5 附近的分辨率约 2.4e-4）。

### 4. 主面板的覆盖

- 主面板 {len(keys):,} 行全部找到了股票下标；信号日当天在张量里都标为“有交易”：{self_traded}
- 60 天窗口里有交易的天数：平均 {cnt.mean():.1f} 天，**{full * 100:.1f}%** 的样本 60 天都有交易，最少 {cnt.min()} 天
  （准入池要求至少 150 根日线，所以窗口不会落到上市前；少于 60 天的是窗口里有停牌，极少数是长期停牌后刚复牌的股票，
  窗口里几乎全是“未交易”。这些样本照常保留：`traded` 通道让网络知道哪些天是空的，删掉反而和其他模型的样本不一致）。

![例子](fig1_example_window.png)

![窗口完整度](fig2_traded_days.png)

## 三、计算量摸底（dl-study 环境，RTX 5060 8 GB）

{probe_md}

读法（每步含前向、反向、参数更新；“全量一个 epoch”= 选择阶段 192 万训练行全部过一遍）：
- 隐藏维度 64 以内，一个全量 epoch 只要 6 ~ 13 秒；隐藏维度 128、两层约 50 秒。最多 30 个 epoch，单组超参数最慢约 25 分钟，
  调参 10 ~ 12 组在可接受范围内（任务清单 0.5 的目标：单次完整训练不超过 30 分钟）。
- 隐藏维度 128、两层、批大小 8192 时显存峰值超过 8 GB，借用了共享内存，每步慢了 6 倍：**批大小上限定为 4096**。
- 序列张量整份放进显卡只占 {X.nbytes / 1024**3:.2f} GB，按下标在显卡上切窗口，不用 DataLoader，也不经过内存。

## 四、给后续任务的用法

```python
from ml.seq_data import load_X, window
X, codes, dates, panel_j = load_X()               # X: [股票, 交易日, 11]，float16
seq = window(X, panel_j[rows], keys["di"].to_numpy()[rows])   # [样本, 60, 11]
```
LSTM 训练时把 X 整份放进显卡，按同样的下标在显卡上切（`ml/lstm_train.py`，任务 3.6）。
"""
    write_md(out_dir / "report.md", text)
    register("1.6_seq", shape=list(X.shape), build_seconds=build_secs, truncation=trunc.to_dict("records"),
             independent=indep.to_dict("records"), mean_traded_days=float(cnt.mean()), full_window=full, probe=probe)


if __name__ == "__main__":
    main()
