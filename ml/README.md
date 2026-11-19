# ml/：期末项目的数据与模型部分

在队友的反弹模型仓库上新增的目录，不改动仓库原有代码（`rebound/`），只通过 import 复用。
执行依据是 `期末项目/任务清单_数据与模型.md`（第 8 版：比较 3 个模型——随机森林、XGBoost、LSTM；测试段按年重训）；每一步的详细结果在 `ml/results/<步骤>/report.md`。

**先看这三个**：报告草稿 [results/REPORT.md](results/REPORT.md)、总排行榜 [results/leaderboard.md](results/leaderboard.md)、给队友的接口说明 [INTERFACE.md](INTERFACE.md)。

## 交接给队友（2026-11-19）

课程不需要提交报告，数据与模型部分的交付物是：
1. **代码**（本目录 `ml/`，进 Git）：数据、特征、模型、评估、回测、接口、测试，全部可以复现；依赖见 [requirements.txt](requirements.txt)（仓库原有的 `pyproject.toml` 不够）；
2. **结果**（`ml/results/`，进 Git）：每一步的报告、排行榜、调参记录、图，以及 `REPORT.md`（全部结论的汇总）；
3. **接口**：`ml/api.py` + 说明 [INTERFACE.md](INTERFACE.md)，智能体和看板只需要调用它；
4. **数据包**（不进 Git）：`data/ml/api_export.zip`（约 300 MB，`python -m ml.p6_export` 导出），解压到队友的 `rebound-model/data/ml/` 后接口即可使用，
   不需要研究库、不需要训练。已在“只有代码 + 数据包”的干净目录里验证过（接口测试 6 项通过）。
5. **事先登记与全部决策记录**：[PREREG.md](PREREG.md)；**执行记录**：[任务清单_数据与模型.md](任务清单_数据与模型.md)（交接时的副本；原件在仓库外的上一级目录）。

推荐默认展示的模型是 `xgb_F69`，理由见 [INTERFACE.md](INTERFACE.md) 和 `REPORT.md`。

**不包含的**（需要的话找数据负责人）：
- **训练好的模型文件**：接口只读事先算好的分数，所以不能给数据之外的新日子打分。有分数的最后一个信号日是 2026-10-16，
  之后的“明天名单”要用仓库原有的 `rebound` 命令，或者用 `ml/` 的代码在最新数据上重训、打分；
- 解释（`explain`）只覆盖 `xgb_F69` 的测试段；
- 研究库和约 6 GB 的中间数据（主面板、特征、LSTM 训练数据等），都可以用 `python -m ml.run` 从研究库重新生成。

## 进度

| 步骤 | 状态 | 代码 | 结果报告 | 一句话结论 |
|---|---|---|---|---|
| P0 准备 | 完成（0.1 课程要求待提供） | `config.yaml`、`common.py`、`p0_probe.py` | [p0_setup](results/p0_setup/compute_probe.md)、[ENV.md](ENV.md) | 一年特征 6 秒，全量约 430 万行，内存够用 |
| 1.1 数据质量 | 完成 | `p1_1_data_quality.py` | [p1_1_data_quality](results/p1_1_data_quality/report.md) | 主键无重复、价格自洽、日历一致、拼接缝无跳变；行业是今天的快照，不用 |
| 1.2 主面板 | 完成 | `p1_2_panel.py` | [p1_2_panel](results/p1_2_panel/report.md) | 4,278,794 行；特征与单日重算差 1e-9，标签抽查 2000/2000 一致 |
| 1.3 EDA | 完成 | `p1_3_eda.py` | [p1_3_eda](results/p1_3_eda/report.md) | 准入池长期跑输指数；标签厚尾，约七成方差来自个股之间；13 对原特征高相关 |
| 1.4 新特征 | 完成 | `features_extra.py`、`p1_4_build_features.py` | [p1_4_features](results/p1_4_features/report.md) | 构造 47 个，去掉 6 个重复，保留 41 个；特征集 F69 |
| 1.5 预处理 | 完成 | `preprocess.py`、`dataset.py`、`p1_5_preprocess_check.py` | [p1_5_preprocess](results/p1_5_preprocess/report.md) | 树模型用原值；LSTM 的 F69 部分用截面排名 + 市场状态 z 分数 |
| 1.6 LSTM 序列数据 | 完成 | `seq_data.py`、`p1_6_build_seq.py`、`lstm_probe.py` | [p1_6_seq](results/p1_6_seq/report.md) | [3,413 股 × 3,350 天 × 11 通道] float16 张量 0.23 GB；防前视、独立重算都通过；LSTM 一个 epoch 6 ~ 50 秒 |
| P2 评估协议 | 完成（2.3 组合层的代码在 P5；2.6 按 3 个模型补充，见 PREREG 补充登记三、四） | `evaluate.py`、`p2_protocol_check.py`；协议文字见 [PREREG.md](PREREG.md) 第二节 | [p2_protocol](results/p2_protocol/report.md) | 训练行都在重训日前卖出；随机打分 IC≈0、完美打分 IC=1；NW t 比普通 t 小三到四成 |
| 3.1 统一接口 | 完成 | `models.py`、`train_eval.py` | [p3_1_interface](results/p3_1_interface/report.md) | `python -m ml.train_eval <model_id>` 一条命令训练、打分、评估、落盘 |
| 3.2 参照 | 完成（验证段） | `train_eval.py`（`random`、`rev_r5`） | [p3_2_references](results/p3_2_references/report.md) | 随机 IC≈0；反转 −r5 IC 0.040，但前 6 名超额为负 |
| 3.3 XGBoost 复现 | **通过** | `p3_baselines.py` | [p3_3_xgb_repro](results/p3_3_xgb_repro/report.md) | 验证段 IC 0.0946（作者约 0.095）；与仓库训练代码逐位相同 |
| 3.4 随机森林 | 完成（验证段） | `models.py`（`RandomForest`）、`p3_4_random_forest.py` | [p3_4_random_forest](results/p3_4_random_forest/report.md)、[tuning.md](results/tuning.md) | 验证 IC：F28 0.0973、F69 0.0972（冻结前补了 6 组网格）；加新特征 IC 不变、前 6 名超额下降 |
| 3.5 XGBoost 调参版 | 完成（验证段） | `models.py`（`XGB`，IC 早停）、`p3_5_xgboost.py` | [p3_5_xgboost](results/p3_5_xgboost/report.md)、[tuning.md](results/tuning.md) | 验证 IC：F28 0.0967、F69 0.0987；第一轮（平方损失早停）停得太早，已归档 |
| 3.6 LSTM | 完成（验证段） | `lstm_prep.py`、`lstm_train.py`、`p3_6_lstm.py` | [p3_6_lstm](results/p3_6_lstm/report.md)、[tuning.md](results/tuning.md) | 验证 IC：S 0.0893、S+F69 0.0935（第二轮：学习率衰减 + 权重平均，曲线稳定）；第一轮已存档 |
| 3.7、3.8 调参预算、两轮比较 | 完成（验证段） | | [tuning.md](results/tuning.md) 3.7、3.8 节 | 加新信息 IC：RF ±0、XGB +0.002、LSTM +0.004；头部超额都降 |
| 3.9 任务形式对比 | 完成（验证段） | `p3_9_task_form.py` | [p3_9_task_form](results/p3_9_task_form/report.md) | 排序 ≈ 回归（0.0985 / 0.0987）；二分类 IC ≈ 0 但头部超额更高（学到的是波动） |
| 3.10 简单集成（选做） | 完成（验证段） | `p3_10_ensemble.py` | [p3_10_ensemble](results/p3_10_ensemble/report.md) | 验证 IC 0.1020 / 0.0994，比最好的单个模型高 0.001 ~ 0.003（区间跨 0） |
| 3.11 总排行榜 | 完成（测试段，按年重训） | `p3_11_run_final.py`、`p3_11_leaderboard.py` | [leaderboard.md](results/leaderboard.md) | 新特征显著有用（RQ3 更好，+0.014）；模型之间分不出（RQ1、RQ2） |
| 4.1 单因子检验 | 完成（测试期） | `p4_1_single_factor.py` | [p4_1_single_factor](results/p4_1_single_factor/report.md) | 65 个特征 42 个显著、40 个为负：最近“热”的股票之后更差；新特征 支持 10 / 相反 9 / 无证据 13 |
| 4.2 按组消融 | 完成（验证段） | `p4_2_ablation.py` | [p4_2_ablation](results/p4_2_ablation/report.md) | 各组作用都在噪声内；排名版本（换手、特质波动等）可能有用，市场状态可能有害 |
| 4.3 解释性 | 完成（验证段） | `p4_3_explain.py`、`lstm_ig.py` | [p4_3_explain](results/p4_3_explain/report.md) | 换手和短期涨幅最重要；20 日内跌停过的票加分；LSTM-S+F69 约 98% 靠手工特征 |
| 4.4 重要性的时间稳定性（选做） | 完成 | `p4_4_shap_stability.py` | [p4_4_shap_stability](results/p4_4_shap_stability/report.md) | 各年重要性排名相关 0.84 ~ 0.94；市场状态类最不稳定 |
| P5 组合回测（5.1 ~ 5.5） | 完成（测试期） | `p5_backtest.py` | [p5_backtest](results/p5_backtest/report.md) | 主规则年化超额：F69 模型 18% ~ 22%、F28 模型 12% ~ 13%；组合层差别都不显著；复现与作者同一量级 |
| 风格归因（全面检查时补做） | 完成（测试期） | `p5_style.py` | [p5_style](results/p5_style/report.md) | 模型 IC 约八成可由 4 种风格解释；新特征的增益来自风格；F69 组合扣风格后仍有显著 alpha |
| 6.1 早期接口 | 完成 | `api.py`、`INTERFACE.md` | [INTERFACE.md](INTERFACE.md) | 队友可以调用：模型列表、排行榜、某天分数和名单、特征说明 |
| 6.2 接口定稿 | 完成 | `api.py`、`INTERFACE.md` | [INTERFACE.md](INTERFACE.md) | 增加 explain（测试段 SHAP，xgb_F69）、nav（3 条规则的净值） |
| 6.3 图表 | 完成 | `p6_figures.py` | [figures/](results/figures/) | 排行榜、逐月 IC、10 组、净值、消融瀑布、SHAP、IC vs 超额 |
| 6.4 报告草稿 | 完成 | — | [REPORT.md](results/REPORT.md) | 按 RQ1 ~ RQ4 组织，含 8 条局限 |
| 6.5 一键复现 | 完成 | `run.py` | 见下文“怎么重跑” | `python -m ml.run panel / features / train / evaluate / backtest / report / test / all` |
| 6.6 测试 | 完成 | `tests/` | — | 53 项通过；仓库原有测试见执行记录 |

## 目录

```text
ml/
├─ README.md                 本文件：进度与导航
├─ p4_1_single_factor.py     4.1 单因子检验（测试期，对照 PREREG 登记的方向）
├─ p4_2_ablation.py          4.2 按组消融（验证段，同一套 XGBoost 参数）
├─ p4_3_explain.py           4.3 解释性：gain、排列重要性、SHAP、浅决策树（XGBoost 与随机森林对照）
├─ lstm_ig.py                4.3 LSTM 积分梯度（dl-study 环境）
├─ p4_4_shap_stability.py    4.4 重要性的时间稳定性；顺带为 api.explain 预算测试段 SHAP
├─ p5_backtest.py            P5 组合回测：仓库模拟器 × 3 条固定规则（测试期），费用和种子稳健性，RQ4
├─ p5_style.py               风格归因：去风格后的 IC、组合超额对风格因子回归
├─ api.py                    6.1 对外只读接口（给智能体和看板）：models、leaderboard、scores、picks、feature_info、data_summary
├─ INTERFACE.md              6.1 接口说明（给队友）
├─ config.yaml               路径、时间划分、种子
├─ ENV.md                    运行环境
├─ requirements.txt          ml/ 需要的 Python 包（接口 / 完整复现 / LSTM 三组）
├─ PREREG.md                 事先登记：新特征的预期方向、评估协议（时间划分、指标、显著性、选择规则）
├─ common.py                 公共工具：路径、只读连接、计时、实验登记
├─ dataset.py                读取主面板和特征集（F28 / F69）
├─ features_extra.py         47 个新特征的定义、SQL 和说明表
├─ preprocess.py             预处理流水线（tree / dense 两种模式）
├─ seq_data.py               1.6 LSTM 的序列数据：11 个原始日线通道、窗口切片
├─ evaluate.py               评估协议：时间划分（含测试段按年重训）、训练标签、IC / Top-K 等指标、模型比较和判定
├─ models.py                 统一模型接口：随机打分、单因子、XGBoost（回归 / 二分类 / 排序，可按 IC 早停）、随机森林；LSTM 在 dl-study 环境单独训练
├─ train_eval.py             统一入口：python -m ml.train_eval <model_id>，训练 → 打分 → 评估 → 落盘
├─ p3_baselines.py           3.1 ~ 3.3 的报告（含与仓库训练代码的逐位核对）
├─ p3_4_random_forest.py     3.4 随机森林：调参（F28、F69 各 12 组）→ 3 个种子 → 报告
├─ p3_5_xgboost.py           3.5 XGBoost 调参版：调参（各 12 组，IC 早停）→ 3 个种子 → 报告
├─ lstm_prep.py              3.6 为 LSTM 准备训练行 / 评估行和手工特征（主环境，写 .npy）
├─ lstm_train.py             3.6 LSTM 训练与打分（dl-study 环境，只依赖 numpy、torch）
├─ p3_6_lstm.py              3.6 LSTM：调度 dl-study 训练（S、S+F69 各 12 组）→ 3 个种子 → 统一评估 → 报告
├─ p3_9_task_form.py         3.9 任务形式对比：XGBoost F69 的回归 / 二分类 / 排序
├─ p3_10_ensemble.py         3.10 简单集成：成员分数当天名次等权平均（不训练）
├─ p3_11_run_final.py        3.11 冻结后一次性跑全部模型的测试段（按年重训 + 只训练一次的对照），再生成排行榜
├─ p3_11_leaderboard.py      3.11 总排行榜（只读已落盘结果）和 4 个主要比较的正式判定
├─ p0_probe.py               0.5 计算量摸底
├─ p1_1_data_quality.py      1.1 数据字典与质量检查
├─ p1_2_panel.py             1.2 构建主面板
├─ p1_3_eda.py               1.3 探索性分析
├─ p1_4_build_features.py    1.4 计算新特征并核对
├─ p1_5_preprocess_check.py  1.5 预处理在真实数据上的检查
├─ p1_6_build_seq.py         1.6 构建序列张量并核对
├─ lstm_probe.py             1.6 LSTM 计算量摸底（dl-study 环境运行，只依赖 numpy、torch）
├─ p2_protocol_check.py      P2 评估协议在真实数据上的检查（不训练模型、不碰测试段收益）
├─ p6_figures.py            6.3 报告用图（统一放到 results/figures/）
├─ p6_export.py             交接：导出接口用的精简数据包 data/ml/api_export（+ zip）
├─ run.py                    6.5 一键复现：python -m ml.run <步骤>|all
├─ tests/                    单元测试（53 项）
└─ results/                  每一步的报告、表格和图（进 Git）
   ├─ registry.jsonl         每次运行的记录（时间、参数、关键数字）
   ├─ p0_setup/  p1_1_data_quality/  p1_2_panel/  p1_3_eda/  p1_4_features/  p1_5_preprocess/  p1_6_seq/
   ├─ p2_protocol/  p3_1_interface/  p3_2_references/  p3_3_xgb_repro/  p3_4_random_forest/  p3_5_xgboost/
   ├─ p3_6_lstm/  p3_9_task_form/  p3_10_ensemble/
   ├─ tuning.md              所有模型的调参网格（运行前写定）和结果
   └─ p3_models/<model_id>/  每个模型的指标（select.json）和逐日序列（select_daily.csv）

data/ml/                      大文件（不进 Git，可由脚本重新生成）
├─ panel_nocap.parquet        主面板：键、28 个原特征、标签（491 MB）
├─ features_extra.parquet     47 个新特征，与主面板逐行对齐（689 MB）
├─ pool_daily_return.parquet  准入池等权日收益
├─ seq/                       LSTM 序列张量 X.npy（[股票, 交易日, 11]，float16）、codes / dates / panel_j.npy
└─ scores/                    每个模型的逐行分数（<model_id>__select.parquet / __final.parquet）
```

## 怎么重跑

在仓库根目录 `rebound-model/` 打开终端。**一键复现**（6.5）：

```powershell
& 'D:\Anaconda\python.exe' -X utf8 -m ml.run all --dry-run   # 先看看要执行哪些命令
& 'D:\Anaconda\python.exe' -X utf8 -m ml.run all             # 全部按顺序跑，约 10 ~ 12 小时
```

也可以只跑某一步（按这个顺序，后面的步骤依赖前面的产物）：

| 步骤 | 内容 | 耗时 |
|---|---|---|
| `panel` | 1.1 数据质量、1.2 主面板、1.3 探索性分析 | 约 5 分钟 |
| `features` | 1.4 新特征、1.5 预处理检查、1.6 序列数据（含 GPU 摸底）、P2 协议检查 | 约 10 分钟 |
| `train` | 3.1 ~ 3.3 参照与复现、3.4 随机森林、3.5 XGBoost、3.6 LSTM（GPU）、3.9 任务形式、3.10 集成（都是验证段） | 约 4 ~ 5 小时 |
| `evaluate` | 3.11 测试段（按年重训）与排行榜、4.1 ~ 4.4 特征分析（含 LSTM 积分梯度，GPU） | 约 5 ~ 6 小时 |
| `backtest` | P5 组合回测、风格归因 | 约 6 分钟 |
| `report` | 6.3 报告用图、导出接口数据包 | 约 3 分钟 |
| `test` | 单元测试 | 约 10 秒 |

环境：主环境 `D:\Anaconda\python.exe`（pandas、pyarrow、duckdb 1.5.5、xgboost、scikit-learn、matplotlib）；LSTM 步骤自动调用
dl-study 环境 `D:\Anaconda\envs\dl-study\python.exe`（torch + CUDA，8 GB 显存够用）。详见 [ENV.md](ENV.md)。
注意：调参脚本可续跑（已完成的组跳过），要完全从头跑先删结果目录；`evaluate` 需要 PREREG 里已有冻结记录；一次只跑一步重活。

逐条命令（和 `ml.run` 调用的完全一样）：

```powershell
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p1_1_data_quality          # 约 10 秒
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p1_2_panel --rebuild       # 约 2 分钟
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p1_3_eda                   # 约 2 分钟（第一次要算准入池收益）
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p1_4_build_features --rebuild   # 约 4 分钟
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p1_5_preprocess_check      # 约 1 分钟
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p1_6_build_seq --rebuild    # 约 2 分钟
& 'D:\Anaconda\envs\dl-study\python.exe' -X utf8 ml/lstm_probe.py    # 约 1 分钟（GPU）
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p2_protocol_check          # 约 1 分钟
& 'D:\Anaconda\python.exe' -X utf8 -m ml.train_eval random rev_r5 xgb_repro_F28   # 约 2 分钟
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p3_baselines               # 约 1 分钟（含用仓库代码重训一次做核对）
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p3_4_random_forest         # 约 2.5 小时（36 组调参 + 2 × 3 个种子）
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p3_5_xgboost              # 约 1.5 小时（24 组调参 + 2 × 3 个种子）
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p3_6_lstm                 # 约 1.5 小时（GPU，第二轮设置；自动调用 dl-study 环境）
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p3_9_task_form            # 约 20 分钟
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p3_10_ensemble            # 约 1 分钟（不训练）
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p3_11_run_final           # 冻结后才能跑：全部模型的测试段（按年重训），约 3 ~ 4 小时
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p4_1_single_factor       # 约 3 分钟
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p4_2_ablation            # 约 1.5 小时
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p4_3_explain             # 约 7 分钟
& 'D:\Anaconda\envs\dl-study\python.exe' -X utf8 ml/lstm_ig.py    # 约 5 分钟（GPU）
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p4_3_explain --report    # 把 LSTM 结果写进 4.3 报告
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p4_4_shap_stability      # 约 25 分钟
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p5_backtest              # 约 3 分钟
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p5_style                 # 约 3 分钟
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p6_figures               # 约 1 分钟
& 'D:\Anaconda\python.exe' -X utf8 -m ml.p6_export                # 约 2 分钟：交接用的数据包
& 'D:\Anaconda\python.exe' -X utf8 -m pytest -p no:cacheprovider -o addopts="" -q ml/tests
```

