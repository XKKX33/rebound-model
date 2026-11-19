# 数据与模型部分的接口说明（给智能体工作流和 UI 看板）

> 版本：6.2 定稿（2026-11-19）。和 6.1 早期版本相比只增加了内容（`explain`、`nav` 可用，排行榜有了测试段），已有函数的名字、参数、返回列都没变。

## 一句话

`from ml import api` 之后调用几个函数，就能拿到：有哪些模型、排行榜、某天全池的分数和排名、候选名单、特征说明、数据概况。
接口**只读**：不训练模型、不写任何文件；研究库只读打开。所有分数都是事先算好存在 `data/ml/scores/` 里的。

## 环境与数据（队友拉代码后要做的两步）

1. **代码与依赖**：拉取仓库（`ml/` 目录），在仓库根目录 `rebound-model/` 下运行。Python 3.11+，按 `ml/requirements.txt` 装“接口”那一组：
   pandas、numpy、pyarrow、duckdb、PyYAML、psutil、xgboost、scikit-learn、scipy（仓库原有的 `pyproject.toml` 里**没有** pyarrow、psutil、scikit-learn，要另外装）。
2. **数据包**：`data/` 不进 Git，向数据负责人要 `api_export.zip`（约 300 MB），**解压到 `rebound-model/data/ml/`**，
   得到 `rebound-model/data/ml/api_export/`。里面是接口要读的全部数据：各模型的分数、测试段的解释、事后收益、股票名称和行业、数据概况。
   有了它就**不需要研究库和主面板**，也不用训练任何模型。

检查是否装好：

```powershell
& 'D:\Anaconda\python.exe' -X utf8 -c "from ml import api; print(api.models()); print(api.picks('xgb_F69', '2025-03-14'))"
& 'D:\Anaconda\python.exe' -X utf8 -m pytest -p no:cacheprovider -o addopts="" -q ml/tests/test_api.py   # 6 项应全部通过
```

数据包由 `python -m ml.p6_export` 从完整数据导出（数据负责人这边）；模型或分数更新后重新导出、重新发一次即可，接口代码不用改。
`nav` 和 `leaderboard` 读的是 `ml/results/` 里的结果文件，已经随代码进了 Git。

## 函数一览

| 函数 | 返回 | 状态 |
|---|---|---|
| `models()` | 全部模型：`model_id`、名称、类别、输入、已有哪些阶段的分数 | 可用 |
| `leaderboard()` | 排行榜（每个模型一行；列见下） | 可用 |
| `dates(model_id)` | 这个模型有分数的全部交易日 | 可用 |
| `scores(model_id, date, with_outcome=False)` | 某天全池的分数和排名 | 可用 |
| `picks(model_id, date, top_n=6, max_per_industry=2, with_outcome=False)` | 候选名单（行业去重后的前 top_n） | 可用 |
| `feature_info()` | 69 个特征 + 11 个序列通道的说明 | 可用 |
| `data_summary()` | 数据概况（字典） | 可用 |
| `explain(model_id, date, ts_code, top=8)` | 这只股票分数的解释：SHAP 贡献最大的特征 | 可用：只有 `xgb_F69`，只覆盖测试段 |
| `nav(model_id, rule="DB-K30-b8")` | 测试期组合净值曲线（仓库模拟器、真实费用） | 可用：3 条规则 `DB-K30-b8`、`DB-K3-b16`、`FH-K6-h4` |

参数不对（模型不存在、那天没有分数）时抛 `api.ApiError`，信息是中文，可以直接展示给用户。

## 返回列

**`scores` / `picks`**（按分数从高到低，同分按代码）：

| 列 | 含义 |
|---|---|
| `ts_code` | 股票代码 |
| `name`、`industry` | 名称、行业（研究库 `stock_basic`，当前口径；少数已退市股票行业为空） |
| `score` | 模型原始分数，只用于排序；**不同模型之间不可比** |
| `rank` | 全池排名，1 = 最好 |
| `pct` | 全池百分位，1 = 最好；**不同模型之间可比** |
| `ret_o1o5`、`excess_o1o5` | 只在 `with_outcome=True` 时有：T+1 开盘买、T+5 开盘卖的实际收益，以及减去当天全池平均后的超额。这是事后才知道的信息，只能用于复盘展示 |

名单的口径和仓库一致：信号日 T 收盘后出名单，T+1 开盘买入；行业去重直接复用仓库的 `rebound.select.dedup_top`。

**`leaderboard`**（测试段跑完后）：`model_id`、模型、类别、特征集、验证 IC（用于选择，偏乐观）、测试 IC（T1 / T2 / 全段 / 逐年，带 95% 区间）、
ICIR、Newey-West t、前 6 / 前 30 超额、多空、10 组单调性、种子标准差和极差、训练耗时，以及“只训练一次”的对照列。
超额、多空是小数（0.003 = 0.3%）。测试段还没跑完时，只有验证段的几列。

**`explain`**：特征、中文名、数值（模型看到的原值）、贡献（SHAP；正 = 让分数变高）、方向（加分 / 减分），按 |贡献| 从大到小。
`.attrs` 里有 `score`（模型分数）和 `base`（全部样本的平均分数）：score = base + 全部 69 个特征的贡献之和，列出的只是最大的几个。
适合给智能体组织“为什么推荐这只票”的解释，比如“换手低、离一年高点近，所以加分”。

**`nav`**：date、nav（模型的组合净值，测试期第一天 = 1）、pool_nav（准入池等权净值，作对照，不可交易、不扣费用）。
规则：`DB-K30-b8`（20 万、最多 30 只，主规则）、`DB-K3-b16`（2 万、最多 3 只）、`FH-K6-h4`（20 万、每天买前 6 名拿 4 天）。

**`feature_info`**：名称、来源（原特征 / 新特征 / LSTM 序列通道）、分组、中文名、计算、预期方向（+ / − / ?，只有新特征有）、理由。

## 模型

| model_id | 说明 |
|---|---|
| `xgb_F69` | XGBoost 调参版，69 个特征。**推荐默认展示这个**：验证段和测试段都在最好的一档（测试 IC 0.120），而且只有它有 `explain` |
| `xgb_F28`、`rf_F28`、`rf_F69` | XGBoost / 随机森林，28 或 69 个特征 |
| `lstm_S`、`lstm_SF69` | LSTM，只用 60 天日线序列 / 再加 69 个特征 |
| `ens_top3`、`ens_family` | 几个模型的排名平均 |
| `xgb_repro_F28` | 仓库原模型的复现（参照） |
| `rev_r5`、`random` | 参照：反转因子、随机打分 |
| `xgb_clf_F69`、`xgb_rank_F69` | 任务形式对比用（二分类 / 排序），一般不用于展示 |

## 有分数的日期

- 2021-01-04 ~ 2022-12-30：验证段，模型用 2013 ~ 2020 年训练；
- 2023-01-03 ~ 2026-10-16：测试段，模型在每年第一个交易日用此前全部数据重训。

两段都是模型训练时没见过的日子，可以拿来展示历史名单和事后表现。最后一个有分数的信号日是 2026-10-16
（标签要等 T+5 卖出，研究库到 2026-10-23）。**接口不提供“明天”的实时名单**：那需要用最新数据重训并打分，属于仓库原有的 `rebound` 命令。

## 例子

```python
from ml import api

api.picks("xgb_F69", "2025-03-14", top_n=6)                       # 那天的 6 只候选
api.picks("xgb_F69", "2025-03-14", top_n=6, with_outcome=True)    # 复盘：它们后来涨了多少
top = api.scores("ens_top3", "2025-03-14").head(30)               # 集成模型的前 30 名
api.feature_info().query("来源 == '新特征（1.4）'")                 # 新特征的说明（智能体解释用）
api.explain("xgb_F69", "2025-03-14", "600926.SH")                 # 这只票为什么排第一
api.nav("xgb_F69", "DB-K30-b8")                                   # 测试期净值，画曲线用
```

## 注意

- 同一个 Python 进程里，分数文件读一次后会缓存；数据负责人更新了分数文件，重启进程才能读到新的。
- 分数文件、研究库都不进 Git；接口代码（`ml/api.py`）和本说明进 Git。
