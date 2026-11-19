# 数据与模型部分的接口说明（给智能体工作流和 UI 看板）

> 版本：交付版（2026-11-19，PREREG 补充登记十三）。交付范围只有**三个用 69 个特征的模型**：XGBoost、随机森林、LSTM。
> 和 6.2 版相比：模型列表只剩这三个；`explain` 三个模型都能用；`nav` 多了 2015 ~ 2026 长期回测；另外交付了模型文件和打分脚本 `ml/predict.py`，可以给新的交易日出名单。
> 函数名、参数、返回列都没有变（只加了可选参数和 `.attrs` 里的字段）。

## 一句话

`from ml import api` 之后调用几个函数，就能拿到：有哪些模型、排行榜、某天全池的分数和排名、候选名单、打分原因、组合净值、特征说明、数据概况。
接口**只读**：不训练模型、不写任何文件。所有分数都是事先算好的。要给数据之外的新日子打分，用 `python -m ml.predict`（见最后一节）。

## 环境与数据（队友拉代码后要做的两步）

1. **代码与依赖**：拉取仓库（`ml/` 目录），在仓库根目录 `rebound-model/` 下运行。Python 3.11+，按 `ml/requirements.txt` 装“接口”那一组：
   pandas、numpy、pyarrow、duckdb、PyYAML、psutil、xgboost、scikit-learn、scipy（仓库原有的 `pyproject.toml` 里**没有** pyarrow、psutil、scikit-learn，要另外装）。
2. **数据包**：`data/` 不进 Git，向数据负责人要 `api_export.zip`，**解压到 `rebound-model/data/ml/`**，得到两个目录：
   - `data/ml/api_export/`：接口要读的全部数据（三个模型的分数、测试段的解释、事后收益、股票名称和行业、数据概况）。
     有了它就**不需要研究库和主面板**，也不用训练任何模型；
   - `data/ml/deploy/`：三个模型训练好的文件，只有 `ml/predict.py` 用（见最后一节）。

检查是否装好：

```powershell
& 'D:\Anaconda\python.exe' -X utf8 -c "from ml import api; print(api.models()); print(api.picks('xgb_F69', '2025-03-14'))"
& 'D:\Anaconda\python.exe' -X utf8 -m pytest -p no:cacheprovider -o addopts="" -q ml/tests/test_api.py   # 应全部通过
```

数据包由 `python -m ml.p6_export` 从完整数据导出（数据负责人这边）；模型或分数更新后重新导出、重新发一次即可，接口代码不用改。
`nav` 和 `leaderboard` 读的是 `ml/results/` 里的结果文件，已经随代码进了 Git。

## 模型

| model_id | 说明 |
|---|---|
| `xgb_F69` | XGBoost（梯度提升树），69 个特征。**推荐默认展示**（`api.DEFAULT_MODEL`）：测试 IC 0.120；2015 ~ 2026 长期 IC 三个里最高，主规则下换起点从不垫底；不需要 PyTorch |
| `rf_F69` | 随机森林，69 个特征。测试 IC 0.119，和 XGBoost 分不出高下；2015 年股灾回撤最小，但 2016 年以后起步的长期收益最低 |
| `lstm_SF69` | LSTM：过去 60 个交易日 × 11 个日线通道的序列 + 69 个特征。测试 IC 0.120；组合收益略高，但换手和 2015 年股灾回撤也最大 |

三个模型用的信息完全相同（69 个特征），差别只在模型本身。测试段的 IC 和组合收益两两都**分不出高下**；它们的分数高度相关，名单大部分重合。
详细比较见 `ml/results/delivery/report.md`；为什么只交这三个，见 `ml/PREREG.md` 补充登记十三。

## 函数一览

| 函数 | 返回 |
|---|---|
| `models()` | 三个模型：`model_id`、名称、类别、输入、已有哪些阶段的分数 |
| `leaderboard()` | 排行榜（每个模型一行，测试段正式数字；列见下） |
| `dates(model_id)` | 这个模型有分数的全部交易日 |
| `scores(model_id, date, with_outcome=False)` | 某天全池的分数和排名 |
| `picks(model_id, date, top_n=6, max_per_industry=2, with_outcome=False)` | 候选名单（行业去重后的前 top_n） |
| `explain(model_id, date, ts_code, top=8)` | 这只股票当天分数的解释：贡献最大的几项（三个模型都有，只覆盖测试段） |
| `nav(model_id, rule="DB-K30-b8", period="test")` | 组合净值曲线：`test` = 测试段 2023 ~ 2026，`long` = 2015 ~ 2026 长期回测 |
| `feature_info()` | 69 个特征 + 11 个序列通道的说明 |
| `data_summary()` | 数据概况（字典） |

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

**`leaderboard`**：`model_id`、模型、验证 IC（用于选择，偏乐观）、测试 IC（全段带 95% 区间、T1 / T2、逐年）、ICIR、Newey-West t、
前 30 超额、多空、种子极差、训练耗时。超额、多空是小数（0.003 = 0.3%）。

**`explain`**：特征、中文名、数值、贡献（正 = 让分数变高）、方向（加分 / 减分），按 |贡献| 从大到小。
`.attrs` 里有 `score`（模型分数）、`base`（基线分数）和 `method`（解释方法）：score ≈ base + 全部项的贡献之和，列出的只是最大的几项。
三个模型的方法不同，读法相同：

| 模型 | 方法 | 说明 |
|---|---|---|
| `xgb_F69` | TreeSHAP | 数值是特征原值 |
| `rf_F69` | 路径分解 | 每次分裂带来的分数变化记到该特征上；靠近树根的特征会被多记一些。数值是特征原值 |
| `lstm_SF69` | 积分梯度 | 除了 69 个特征，还有 11 个序列通道（`seq_ret` 等，中文名“序列：日收益（近 60 天）”），表示过去 60 天这个通道的整体影响；特征的数值是截面排名等预处理之后的值 |

适合给智能体组织“为什么推荐这只票”的解释，比如“换手低、离一年高点近，所以加分”。

**`nav`**：date、nav（模型的组合净值，第一天 = 1）、pool_nav（准入池等权净值，作对照，不可交易、不扣费用）；`period="long"` 时另有 `hs300`、`zz500`、`zz1000`。
规则：`DB-K30-b8`（20 万、最多 30 只，主规则）、`DB-K3-b16`（2 万、最多 3 只，噪声很大，不建议用来比较模型）、`FH-K6-h4`（20 万、每天买前 6 名拿 4 天）。
`test` 是正式结果；`long` 是补充分析，2023 年以前的部分仅供参考（参数、特征、规则都是看过那段数据后定的）。

**`feature_info`**：名称、来源（原特征 / 新特征 / LSTM 序列通道）、分组、中文名、计算、预期方向（+ / − / ?，只有新特征有）、理由。

## 有分数的日期

- 2021-01-04 ~ 2022-12-30：验证段，模型用 2013 ~ 2020 年训练；
- 2023-01-03 ~ 2026-10-16：测试段，模型在每年第一个交易日用此前全部数据重训。

两段都是模型训练时没见过的日子，可以拿来展示历史名单和事后表现。最后一个有分数的信号日是 2026-10-16（标签要等 T+5 卖出，研究库到 2026-10-23）。

## 例子

```python
from ml import api

api.picks("xgb_F69", "2025-03-14", top_n=6)                       # 那天的 6 只候选
api.picks("xgb_F69", "2025-03-14", top_n=6, with_outcome=True)    # 复盘：它们后来涨了多少
api.picks("rf_F69", "2025-03-14", top_n=6)                        # 换个模型看看
api.explain("lstm_SF69", "2025-03-14", "600926.SH")               # 这只票为什么得这个分
api.nav("xgb_F69", "DB-K30-b8")                                   # 测试期净值
api.nav("xgb_F69", "DB-K30-b8", period="long")                    # 2015 ~ 2026 长期净值，附指数
api.feature_info().query("来源 == '新特征（1.4）'")                 # 新特征的说明（智能体解释用）
```

## 给新的交易日打分：`ml/predict.py`

接口只读事先算好的分数，到 2026-10-16 为止。要给之后的日子出名单，用交付的模型文件（`data/ml/deploy/`）：

```powershell
& 'D:\Anaconda\python.exe' -X utf8 -m ml.predict --rebuild          # 研究库更新到新交易日后：重建面板、特征、序列，再给最后一天打分
& 'D:\Anaconda\python.exe' -X utf8 -m ml.predict --date 2026-10-23 --models xgb_F69 --top 6
```

- **需要研究库**（`data/research-2013-2026.duckdb`，更新到要打分的那一天）和完整依赖（`ml/requirements.txt` 的“完整复现”组）。
  `--rebuild` 会重建主面板、特征和序列数据（约 15 分钟），没有新数据时不用加。
- 输出 `data/ml/deploy/out/scores_<日期>.csv`（全池每个模型的分数、名次、百分位）和 `picks_<日期>.csv`（每个模型的名单），屏幕上也会打印前 10 名。
- 模型是用截至 2026-10-23 已经知道结果的全部样本训练的（冻结参数，3 个种子）。按项目的做法，**每年初应重训一次**：`python -m ml.p8_deploy`（先删掉 `data/ml/deploy/<模型>/`）。
- LSTM 打分要用装了 PyTorch 的环境（本机是 dl-study，路径写在 `ml/predict.py` 的 `DL_PY`）；没有的话加 `--models xgb_F69,rf_F69`。
- 随机森林的模型文件是 scikit-learn 对象，读取时 scikit-learn 版本要和训练时相同（见 `deploy/rf_F69/meta.json` 的 `versions`）；XGBoost 用自己的格式，不受版本影响。
- 流程核对：`python -m ml.predict --check` 从文件读回模型再打分，和训练时内存里的分数比较（树模型应完全相同）。

## 注意

- 同一个 Python 进程里，分数文件读一次后会缓存；数据负责人更新了分数文件，重启进程才能读到新的。
- 分数文件、模型文件、研究库都不进 Git；接口代码（`ml/api.py`）、打分脚本（`ml/predict.py`）和本说明进 Git。
- 其他模型（28 个特征的模型、作者原模型、只用序列的 LSTM、集成等）的结果是“三个模型怎么选出来的”过程记录，在 `ml/results/` 各步目录里，接口不提供。
