# 3.1 统一模型接口

- 代码：`ml/models.py`（模型）、`ml/train_eval.py`（统一入口）；单元测试 `ml/tests/test_models.py`（6 项，全部通过）
- 运行：`python -X utf8 -m ml.train_eval <model_id> ...`（选择阶段：训练段 → 验证段）；加 `--final` 是最终阶段（训练 + 验证 → 测试，所有选择冻结后才用）

## 一、接口

每个模型实现两个方法：

| 方法 | 输入 | 说明 |
|---|---|---|
| `fit(X, y, X_valid, y_valid, day_valid)` | `X`：预处理后的特征（float32）；`y`：训练标签（当天收益百分位，只在已卖出的行里排） | 后三个参数只给需要早停的模型（XGBoost 调参版、MLP），其他模型忽略 |
| `predict(X)` / `predict_seeds(X)` | 评估段的特征 | 前者返回各种子的平均分数（正式分数），后者返回每个种子各自的分数（算种子间波动） |

每个模型还声明 `preprocess`（`tree` 用原值，`dense` 用截面排名，见 1.5）和 `complexity`（选参数打平时用，越小越简单）。

## 二、统一入口做的事（`train_eval.run`）

1. 按协议 A 取训练行和评估行（`ml/evaluate.py`，训练行的卖出日都早于重训日）；
2. 训练标签 `train_labels`（只在已知行里排百分位，和仓库 `training_set` 逐位相同）；
3. 预处理：只用训练行拟合（`ml/preprocess.py`）；
4. 训练 → 对评估段打分（每个种子一列，正式分数取平均）；
5. 用 `evaluate` 算全部预测层指标，另算每个种子单独的 IC；
6. 落盘：
   - 分数 `data/ml/scores/<model_id>__<阶段>.parquet`（`trade_date, ts_code, score, s0, s1, ...`，给 P5 回测和队友的接口用）；
   - 指标 `ml/results/p3_models/<model_id>/<阶段>.json`、逐日序列 `<阶段>_daily.csv`（模型两两比较用）；
   - `ml/results/registry.jsonl` 追加一条。

模型在 `train_eval.CONFIGS` 里登记（model_id → 构造函数、特征集）。目前有：`random`、`rev_r5`、`xgb_repro_F28`。

## 三、跑通的结果

| 模型          |   验证 IC |   NW t |   前6 超额% |   前30 超额% |   训练秒 |
|:--------------|----------:|-------:|------------:|-------------:|---------:|
| random        |   -0.0009 |   -0.8 |       0.174 |       -0.028 |      0   |
| rev_r5        |    0.04   |    4.2 |      -0.555 |       -0.178 |      0   |
| xgb_repro_F28 |    0.0946 |    9.9 |       0.348 |        0.316 |     36.4 |

参照的数字和 P2 检查（`ml/results/p2_protocol/report.md`）完全相同：统一入口和评估代码前后一致。
