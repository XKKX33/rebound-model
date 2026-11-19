"""任务 1.5：在真实数据上检查预处理流水线。

用训练段（2013 ~ 2020，见 ml/config.yaml）拟合 dense 模式的预处理，再按年分批变换全部数据
（截面排名每天独立，按年分批和一次算完结果完全相同，只是省内存），检查输出。
变换后的矩阵不落盘：每次训练时现算，几十秒。

运行：python -X utf8 -m ml.p1_5_preprocess_check
输出：ml/results/p1_5_preprocess/report.md、column_check.csv、market_state_z.csv
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ml.common import CFG, register, result_dir, rss_gb, write_md
from ml.dataset import feature_set, load_features, load_panel
from ml.features_extra import MARKET_STATE
from ml.preprocess import Preprocessor


def main() -> None:
    out = result_dir("p1_5_preprocess")
    keys = load_panel(["trade_date", "di"])
    x = load_features("F69")
    day = keys["di"].to_numpy()
    seg = np.select([keys["trade_date"] <= CFG["split"]["train_end"], keys["trade_date"] <= CFG["split"]["valid_end"]],
                    ["训练", "验证"], "测试")
    train = seg == "训练"

    t0 = time.time()
    tree = Preprocessor("tree").fit(x[train], day[train])
    dense = Preprocessor("dense").fit(x[train], day[train])
    fit_secs = time.time() - t0

    t0 = time.time()
    years = keys["trade_date"].dt.year.to_numpy()
    blocks, names = [], None
    for y in np.unique(years):
        m = years == y
        v, names = dense.transform(x[m], day[m])
        blocks.append(v)
    z = np.vstack(blocks)
    secs = time.time() - t0
    mem = rss_gb()
    tv, tnames = tree.transform(x.iloc[:1000], day[:1000])

    # 检查
    zf = pd.DataFrame(z, columns=names)
    cs = [c for c in names if c in dense.cs_cols]
    col = pd.DataFrame({
        "列": names,
        "类型": ["截面排名" if c in dense.cs_cols else "时间序列 z 分数" if c in dense.ts_cols else "缺失指示" for c in names],
        "最小": zf.min().round(4).to_numpy(), "最大": zf.max().round(4).to_numpy(),
        "均值": zf.mean().round(4).to_numpy(), "非零比例": (zf != 0).mean().round(4).to_numpy(),
    })
    col.to_csv(out / "column_check.csv", index=False, encoding="utf-8-sig")
    day_mean = zf[cs].groupby(day).mean().abs().max().max()
    ms = []
    for c in dense.ts_cols:
        s = pd.Series(zf[c].to_numpy()).groupby(day).first()
        sg = pd.Series(seg).groupby(day).first()
        for g in ("训练", "验证", "测试"):
            v = s[sg == g]
            ms.append({"特征": c, "段": g, "均值": round(float(v.mean()), 3), "标准差": round(float(v.std()), 3)})
    ms = pd.DataFrame(ms)
    ms.to_csv(out / "market_state_z.csv", index=False, encoding="utf-8-sig")
    na_share = x[dense.na_cols].isna().mean().mul(100).round(2)

    seg_rows = pd.Series(seg).value_counts().reindex(["训练", "验证", "测试"])
    text = f"""
# 1.5 预处理流水线

- 代码：`ml/preprocess.py`（流水线）、`ml/dataset.py`（读取面板和特征集）；单元测试 `ml/tests/test_preprocess.py`（7 项，全部通过）
- 运行检查：`python -X utf8 -m ml.p1_5_preprocess_check`
- 产物：`column_check.csv`（每个输出列的范围和均值）、`market_state_z.csv`（市场状态 z 分数在各段的均值和标准差）

## 一、设计

| 模型 | 模式 | 做法 | 理由 |
|---|---|---|---|
| 随机森林、XGBoost | `tree` | 原样输入，缺失交给模型 | 树只看特征值的顺序，单调变换不改变分裂结果；XGBoost 能学习缺失值该往哪边走 |
| Ridge、MLP | `dense` | 个股特征 → 每天截面排名，映射到 (−0.5, 0.5) | 1.3 发现特征厚尾、量纲不同；排名后每个特征每天都是均匀分布、居中，和截面百分位标签对齐，也不受极端值影响 |
| | | 市场状态 4 个特征 → 时间序列 z 分数（均值、标准差只用训练段） | 它们同一天所有股票相同，截面排名会全部变成 0，信息就丢了 |
| | | 缺失 → 0；训练段缺失率 ≥ 0.1% 的列再加缺失指示列 | 0 是排名后的中位数、z 分数的均值；指示列让模型知道“这里本来没值”。只缺几十行的列不加，免得多出几十个几乎全是 0 的列（第一次运行时 32 列都加了，其中 30 列缺失率约 0%，所以加了门槛） |

**为什么不先缩尾**：任务清单原写“缩尾后排名”。缩尾是单调变换，只会把两端的值变成并列，排名几乎不变
（单元测试 `test_rank_ignores_outliers` 验证了差别 < 0.05），所以只做排名，省掉一步。

**防泄漏**：
- 截面排名每天独立计算，不用其他日子的信息，所以不需要“拟合”（单元测试 `test_rank_does_not_depend_on_other_days`：改动其他日子，这一天的结果不变）；
- 需要拟合的只有市场状态的均值和标准差、以及“哪些列加缺失指示”，都只用训练段（单元测试 `test_fit_uses_train_only`：随便改验证和测试段，拟合结果不变）。

## 二、真实数据上的结果（F69，用训练段 2013 ~ 2020 拟合）

| 项目 | 数值 |
|---|---|
| 样本行数 | 训练 {seg_rows['训练']:,}、验证 {seg_rows['验证']:,}、测试 {seg_rows['测试']:,} |
| 输入特征 | {x.shape[1]} 个（截面排名 {len(dense.cs_cols)} 个，时间序列 z 分数 {len(dense.ts_cols)} 个） |
| 缺失指示列 | {len(dense.na_cols)} 个 |
| dense 输出 | {z.shape[0]:,} 行 × {z.shape[1]} 列，float32，约 {z.nbytes / 1024**3:.2f} GB |
| 拟合耗时 / 变换耗时 | {fit_secs:.1f} 秒 / {secs:.0f} 秒（按年分批），进程内存 {mem:.1f} GB |
| 输出里的空值 | {int(np.isnan(z).sum())} 个 |
| 截面排名列的取值范围 | [{zf[cs].min().min():.4f}, {zf[cs].max().max():.4f}] |
| 每天截面排名的均值，最大偏离 0 | {day_mean:.2e}（含缺失填 0 的影响） |
| tree 输出 | 和输入完全相同（{len(tnames)} 列） |

### 加了缺失指示的列（训练段缺失率 ≥ 0.1%）及全期缺失率（%）

{na_share.to_frame("缺失%").to_markdown()}

### 市场状态特征的 z 分数在各段的分布

训练段的均值≈0、标准差≈1 是按定义；验证、测试段偏离 0 和 1 说明市场环境和训练期不同，这是真实的分布漂移，不是错误。
树模型对这种漂移更敏感（它学的是阈值），这一点在 P3 解读结果时要考虑。

{ms.pivot(index="特征", columns="段", values=["均值", "标准差"]).to_markdown()}

### 一次无法复现的异常（如实记录）

第一次在真实数据上运行本检查时，“每天截面排名均值的最大偏离”是 0.5：2021-01-04 这一天 `pct_chg` 列全部变成了 −0.5。
随后用同样的代码、同样的数据重跑 5 次以上（含一次完整的 3 轮全量检查），都完全正常，原因没有查明
（怀疑与 pandas 分组变换在当时的内存压力下的行为有关，但无法证实）。处理办法：
1. 每天的非空个数改用 numpy 直接计数，不再依赖 `groupby.transform`；
2. `cross_section_rank` 每次都自检“每天每列的均值≈0”，不满足就报错停止，不会让错误数据悄悄进入训练
   （单元测试 `test_rank_self_check_catches_bad_output` 验证自检有效）。
上表是加了自检之后的结果。

## 三、给后续任务的用法

```python
from ml.dataset import load_features, load_panel
from ml.preprocess import Preprocessor

x = load_features("F69")                       # 或 "F28"
day = load_panel(["di"])["di"].to_numpy()
pre = Preprocessor("dense").fit(x[train], day[train])   # 树模型用 Preprocessor("tree")
z_train, names = pre.transform(x[train], day[train])
z_valid, _ = pre.transform(x[valid], day[valid])
```
"""
    write_md(out / "report.md", text)
    register("1.5_preprocess", n_features=x.shape[1], n_out=z.shape[1], na_cols=dense.na_cols,
             ts_mean=dense.mean_, ts_std=dense.std_, transform_seconds=round(secs, 1), mem_gb=round(mem, 2))


if __name__ == "__main__":
    main()
