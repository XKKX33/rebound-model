# 运行环境（任务 0.2，2026-11-17 记录）

> 安装用的包清单见 [requirements.txt](requirements.txt)（2026-11-19 交接时补）。

## 主环境：Anaconda base（`D:\Anaconda\python.exe`）

数据、特征、随机森林 / XGBoost、评估、回测都用它。

| 包 | 版本 | 说明 |
|---|---|---|
| Python | 3.13.9 | |
| duckdb | 1.5.5 | 和仓库钉的版本一致（2026-11-17 新装） |
| pandas | 2.3.3 | 和仓库一致 |
| numpy | 2.4.3 | 仓库钉 2.3.5 |
| xgboost | 3.4.1 | 仓库钉 3.2.0：分数末几位会和作者不同，所以复现只要求“同一量级” |
| scikit-learn | 1.7.2 | |
| scipy / statsmodels | 1.16.3 / 0.14.5 | 统计检验 |
| pyarrow | 21.0.0 | parquet |
| matplotlib / seaborn | 3.10.6 / 0.13.2 | 作图 |
| shap | 未装（不需要） | 4.3 的 SHAP 用 xgboost 自带的精确 TreeSHAP（`pred_contribs=True`）；LSTM 的积分梯度用 PyTorch 自己写（`ml/lstm_ig.py`），也没装 captum |

## 神经网络环境：`dl-study`（`D:\Anaconda\envs\dl-study\python.exe`）

只给 LSTM 用（第 5 版；原计划给 MLP）：Python 3.11.15，PyTorch 2.10.0+cu128，GPU 为 RTX 5060 Laptop 8 GB（CUDA 可用）。

该环境没有 pyarrow、duckdb，pandas 是 3.0.2（主环境 2.3.3）。**不补装**：LSTM 只通过 numpy 文件（.npy）和主环境交换数据
——主环境准备序列张量、样本下标和标签，dl-study 只负责训练和打分，分数回到主环境用同一套 `ml/evaluate.py` 评估。
实测 0.5 GB 的数组，npy 写 0.33 秒、读 0.16 秒；parquet 写 5.3 秒、读 1.0 秒。

## 机器

32 个逻辑核，内存 15 GB（日常空闲 3 ~ 8 GB），所以面板和特征都按年分批计算。

## 运行方式

在仓库根目录 `rebound-model/` 下：

```powershell
& 'D:\Anaconda\python.exe' -X utf8 -m ml.<脚本名>
```

## 运行时的经验（2026-11-19 补）

- **一次只跑一个重活**：内存 15 GB，几个进程同时读大文件、训练时，测试段出现过偶发的计算错误（PREREG“冻结后的运行记录”）。
  CPU 任务（树模型）和 GPU 任务（LSTM）可以并行，但不要同时跑两个读全量特征的任务。
- **显卡**：笔记本显卡同时给其他程序用。LSTM 训练已改为把一批再切成 ≤ 2048 行的小块累加梯度（峰值约 3 GB），
  并且遇到显存不足会等 60 秒重试（`ml/p3_6_lstm.py`）。
- **Bash 里**要写全路径 `/d/Anaconda/python.exe`：裸 `python` 不是 Anaconda 主环境。
