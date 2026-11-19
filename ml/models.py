"""任务 3.1 统一模型接口：每个模型都是 `fit(X, y, ...)` / `predict(X)`，由 `ml/train_eval.py` 统一调度。

- `X`：预处理后的特征矩阵（float32，numpy），树模型是原值，dense 模式是截面排名（见 `ml/preprocess.py`）；
- `y`：训练标签 = 当天准入池内收益的百分位，只在已卖出的行里排（`ml/evaluate.py` 的 `train_labels`）；
- `X_valid`、`y_valid`、`day_valid`：只给需要早停的模型（XGBoost 调参版）用，其余模型忽略；
- `predict` 返回每行一个分数，分数越高 = 预测之后 5 天的收益排名越高。只有分数的**排序**有意义。

有随机性的模型在内部跑多个种子：`predict` 返回各种子的平均，`predict_seeds` 返回每个种子各自的分数
（PREREG 2.4：正式分数取平均，另报各种子的 IC）。

本文件目前有：随机打分、单因子（参照，任务 3.2）、随机森林（任务 3.4）、XGBoost（任务 3.3 复现 / 3.5 调参版）。
LSTM（3.6）在 dl-study 环境单独训练（`ml/lstm_train.py`），分数回到这里评估。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from rebound import params as repo_params
from rebound.model.train import CHUNK, booster_params, no_power_throttling

#: 排序目标（任务 3.9）的相关度等级数：训练标签（百分位）切成这么多档
RANK_GRADES = 100


class Model:
    """所有模型的基类。"""

    name: str = "model"
    preprocess: str = "tree"               # "tree" 或 "dense"，见 ml/preprocess.py
    seeds: tuple[int, ...] = (0,)
    complexity: float = 0.0                # 选择规则打平时用：越小越简单（PREREG 2.5）

    def fit(self, X: np.ndarray, y: np.ndarray, X_valid=None, y_valid=None, day_valid=None) -> "Model":
        raise NotImplementedError

    def predict_seeds(self, X: np.ndarray) -> np.ndarray:
        """[行, 种子] 的分数。没有随机性的模型只有一列。"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_seeds(X).mean(axis=1)

    def info(self) -> dict[str, Any]:
        """写进 registry 的配置。"""
        return {"name": self.name, "preprocess": self.preprocess, "seeds": list(self.seeds)}


# --------------------------------------------------------------------------- #
# 参照（任务 3.2）
# --------------------------------------------------------------------------- #


@dataclass
class RandomScore(Model):
    """随机打分：IC 应≈0，用来检验评估流程本身有没有虚假的预测能力。"""

    seed: int = 20261117
    name: str = "random"

    def fit(self, X, y, X_valid=None, y_valid=None, day_valid=None):
        return self

    def predict_seeds(self, X):
        return np.random.default_rng(self.seed).normal(size=(len(X), 1))

    def info(self):
        return {**super().info(), "seed": self.seed}


@dataclass
class SingleFactor(Model):
    """单因子：分数 = sign × 某一列特征（缺失记为当列中位数，等于不表态）。"""

    column: int = 0
    sign: float = -1.0
    label: str = "-r5"
    name: str = "factor"

    def fit(self, X, y, X_valid=None, y_valid=None, day_valid=None):
        return self

    def predict_seeds(self, X):
        v = np.asarray(X[:, self.column], dtype=np.float64)
        v = np.where(np.isfinite(v), v, np.nanmedian(v))
        return (self.sign * v)[:, None]

    def info(self):
        return {**super().info(), "factor": self.label}


# --------------------------------------------------------------------------- #
# XGBoost（任务 3.3 复现基准、3.5 调参版、3.9 任务形式）
# --------------------------------------------------------------------------- #


def daily_ic_metric(day: np.ndarray, y_pct: np.ndarray):
    """XGBoost 早停用的自定义指标：验证段逐日“分数排名与收益百分位”的相关系数，再对天平均（越大越好）。

    要求验证行按交易日排好（统一入口给的评估行就是按日期排的）。分数排名用序数名次（并列按出现顺序），
    只用于决定在第几棵树停下；正式报告的 IC 仍由 `ml/evaluate.py` 计算。每一轮约 0.1 秒。
    """
    day = np.asarray(day)
    if len(day) > 1 and (np.diff(day) < 0).any():
        raise ValueError("验证行必须按交易日排好")
    codes = np.searchsorted(np.unique(day), day)
    n = np.bincount(codes).astype(np.float64)
    starts = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64)
    y = np.asarray(y_pct, dtype=np.float64)
    my = np.bincount(codes, y) / n
    vy = np.bincount(codes, y * y) / n - my ** 2

    def metric(predt: np.ndarray, _dmat) -> tuple[str, float]:
        order = np.lexsort((predt, codes))
        r = np.empty(len(predt))
        r[order] = np.arange(len(predt)) - starts[codes[order]]
        mr = (n - 1) / 2
        vr = (n * n - 1) / 12
        cov = np.bincount(codes, r * y) / n - mr * my          # 当天名次 0..n−1 的均值 (n−1)/2、方差 (n²−1)/12
        with np.errstate(invalid="ignore", divide="ignore"):
            ic = cov / np.sqrt(vr * vy)
        return "ic", float(np.nanmean(ic))

    return metric


class _Chunks(xgb.DataIter):
    """训练矩阵分块送进 QuantileDMatrix（和仓库一样每块 CHUNK 行），不再整份拷贝。"""

    def __init__(self, X: np.ndarray, y: np.ndarray, group: np.ndarray | None = None) -> None:
        self._X, self._y, self._g, self._at = X, y, group, 0
        super().__init__()

    def next(self, input_data) -> bool:
        if self._at >= len(self._X):
            return False
        n = len(self._X)
        stop = min(self._at + CHUNK, n)
        if self._g is not None and stop < n:            # 排序目标：块的边界要落在“一天”的边界上（行已按日期排好）
            stop = int(np.searchsorted(self._g, self._g[stop], side="left"))
            if stop <= self._at:
                stop = int(np.searchsorted(self._g, self._g[self._at], side="right"))
        kw = {"data": self._X[self._at:stop], "label": self._y[self._at:stop]}
        if self._g is not None:
            kw["qid"] = self._g[self._at:stop]
        input_data(**kw)
        self._at = stop
        return True

    def reset(self) -> None:
        self._at = 0


@dataclass
class XGB(Model):
    """XGBoost。默认参数 = 仓库原参数（`rebound.params.PARAMS`），种子 0、1、2，16 线程，和作者完全一致。

    objective：
    - "reg"：回归，预测收益百分位（仓库现行做法）；
    - "clf"：二分类，是否进当天前 20%（任务 3.9）；
    - "rank"：`rank:pairwise`，按交易日分组排序（任务 3.9）。
    early_stop：> 0 时用验证集早停，记下最佳树数 best_rounds，最终阶段重训时直接用这个树数。
    early_metric：早停看什么——
    - "rmse"：验证段平方损失（XGBoost 默认；3.5 第一轮用的，停得太早，见 PREREG 补充登记六）；
    - "ic"：验证段逐日截面秩相关的均值（和选择指标一致，也和 LSTM 的早停规则一致；3.5 第二轮起用）。
    """

    params: dict[str, Any] = field(default_factory=lambda: dict(repo_params.PARAMS))
    seeds: tuple[int, ...] = tuple(repo_params.SEEDS)
    objective: str = "reg"
    early_stop: int = 0
    early_metric: str = "rmse"
    name: str = "xgb"
    boosters: list = field(default_factory=list, repr=False)
    best_rounds: list = field(default_factory=list)

    def _booster_params(self, seed: int) -> dict[str, Any]:
        p = booster_params(self.params, seed)            # 仓库的写法：reg:squarederror、eta、hist、nthread=16
        for k in ("min_child_weight", "reg_lambda", "gamma"):
            if k in self.params:
                p[k] = self.params[k]
        if self.objective == "clf":
            p.update(objective="binary:logistic", eval_metric="logloss")
        elif self.objective == "rank":
            p.update(objective="rank:pairwise", eval_metric="ndcg@30")
            # 排序的成对抽样方式（任务 3.9 定为 "mean"：每行随机配对，学整个截面的顺序），原样传给 xgboost
            p.update({k: v for k, v in self.params.items() if k.startswith("lambdarank_") or k == "ndcg_exp_gain"})
        return p

    def _target(self, y: np.ndarray) -> np.ndarray:
        if self.objective == "clf":
            return (y > 0.8).astype(np.float32)
        if self.objective == "rank":
            # xgboost 3.x 的排序目标要求标签是非负整数（相关度等级）：百分位切成 100 档 0 ~ 99，同一档内的两行不配对
            return np.minimum(np.floor(np.asarray(y) * RANK_GRADES), RANK_GRADES - 1).astype(np.float32)
        return y.astype(np.float32)

    @property
    def complexity(self) -> float:
        return self.params["n_estimators"] * 2 ** self.params["max_depth"]

    def fit(self, X, y, X_valid=None, y_valid=None, day_valid=None, day_train=None):
        no_power_throttling()
        group = day_train if self.objective == "rank" else None
        if self.objective == "rank" and group is None:
            raise ValueError("排序目标需要 day_train（每行的交易日）")
        dtrain = xgb.QuantileDMatrix(_Chunks(X, self._target(y), group), nthread=repo_params.THREADS)
        evals = []
        if self.early_stop and X_valid is not None:
            kw = {"qid": day_valid} if self.objective == "rank" else {}
            evals = [(xgb.QuantileDMatrix(X_valid, label=self._target(y_valid), ref=dtrain,
                                          nthread=repo_params.THREADS, **kw), "valid")]
        self.boosters, self.best_rounds = [], []
        kw = {}
        if evals and self.early_metric == "ic":
            kw = {"custom_metric": daily_ic_metric(day_valid, y_valid), "maximize": True}
        for seed in self.seeds:
            p = self._booster_params(seed)
            if kw:
                p["disable_default_eval_metric"] = 1       # 只用自定义的 IC 做早停
            b = xgb.train(p, dtrain, num_boost_round=self.params["n_estimators"],
                          evals=evals, early_stopping_rounds=self.early_stop or None, verbose_eval=False, **kw)
            self.best_rounds.append(int(b.best_iteration) + 1 if evals else self.params["n_estimators"])
            self.boosters.append(b)
        return self

    def predict_seeds(self, X):
        d = xgb.DMatrix(X, nthread=repo_params.THREADS)
        out = []
        for b, n in zip(self.boosters, self.best_rounds):
            out.append(b.predict(d, iteration_range=(0, n)))
        return np.column_stack(out)

    def info(self):
        return {**super().info(), "params": self.params, "objective": self.objective,
                "early_stop": self.early_stop, "early_metric": self.early_metric, "best_rounds": self.best_rounds}


# --------------------------------------------------------------------------- #
# 随机森林（任务 3.4）
# --------------------------------------------------------------------------- #

#: 随机森林固定的部分（PREREG 补充登记四）
RF_FIXED: dict[str, Any] = {"n_estimators": 300, "bootstrap": True, "n_jobs": 24}


@dataclass
class RandomForest(Model):
    """scikit-learn `RandomForestRegressor`：Bagging，很多棵树各自在有放回抽取的样本、随机的特征子集上生长，再取平均。

    params：调的 3 个参数 `max_samples`、`max_features`、`min_samples_leaf`（其余见 RF_FIXED）。
    每个种子是一整片独立的森林（300 棵树）；种子决定每棵树抽哪些样本、每次分裂看哪些特征。
    原值输入，缺失值由 sklearn 处理（分裂时把缺失样本试着分到左右两边，取更好的一边）。
    """

    params: dict[str, Any] = field(default_factory=lambda: {"max_samples": 0.15, "max_features": 0.4, "min_samples_leaf": 200})
    seeds: tuple[int, ...] = (0, 1, 2)
    name: str = "rf"
    forests: list = field(default_factory=list, repr=False)

    @property
    def complexity(self) -> float:
        """预期叶子数 ∝ 每棵树的样本比例 / 叶子最少样本数（打平时选更小的）。"""
        return self.params["max_samples"] / self.params["min_samples_leaf"]

    def fit(self, X, y, X_valid=None, y_valid=None, day_valid=None):
        self.forests = [RandomForestRegressor(**RF_FIXED, **self.params, random_state=s).fit(X, y) for s in self.seeds]
        return self

    def predict_seeds(self, X):
        return np.column_stack([f.predict(X) for f in self.forests])

    def info(self):
        n_leaves = [float(np.mean([e.get_n_leaves() for e in f.estimators_])) for f in self.forests]
        return {**super().info(), "params": {**RF_FIXED, **self.params}, "mean_leaves": n_leaves}
