"""任务 1.5 预处理流水线。

两种模式：
- `tree`（随机森林、XGBoost）：原样使用特征，缺失值交给模型处理。树模型只看特征的顺序，对单调变换不敏感，不需要标准化。
- `dense`（原计划给 Ridge、MLP；第 6 版只给 LSTM-S+F69 的手工特征部分）：
  1. 个股特征 → 每天在准入池内做截面排名，映射到 (-0.5, 0.5)：
     (平均名次 − 0.5) / 当天非空个数 − 0.5。排名对极端值不敏感，所以不需要先缩尾
     （缩尾是单调变换，只会把两端变成并列，排名后几乎一样）。每天单独算，不依赖其他日期，所以没有跨期泄漏，也不需要“拟合”。
  2. 市场状态特征（同一天所有股票相同，截面排名会全部变成 0）→ 时间序列 z 分数，
     均值和标准差**只用训练段**估计，再套到验证集和测试集。
  3. 缺失 → 填 0（排名后的中位数 / z 分数的均值）；训练段缺失率 ≥ `na_min_rate`（默认 0.1%）的列再加一个 0/1 缺失指示列。
     缺失极少的列（只缺几十行）不加指示列，免得多出几十个几乎全是 0 的噪声列。

`fit` 只接收训练段的数据；`transform` 可以用于任何时段。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.features_extra import MARKET_STATE


def cross_section_rank(x: pd.DataFrame, day: np.ndarray, *, check: bool = True) -> pd.DataFrame:
    """每天截面排名映射到 (-0.5, 0.5)；空值保持为空。

    每天的非空个数用 numpy 直接计数（不用 pandas 的 groupby.transform），并自检：
    每天每列的均值必须≈0（排名映射后按定义居中）。2026-11-17 第一次在真实数据上运行时出现过一次
    “某天某列全部变成 −0.5”的异常，之后多次重跑无法复现，原因没有查明；所以加了这道自检，
    再出现就直接报错，而不是让错误的数据悄悄进入训练。
    """
    codes, uniq = pd.factorize(np.asarray(day))
    r = x.groupby(codes, sort=False).rank(method="average")
    out = {}
    for c in x.columns:
        ok = x[c].notna().to_numpy()
        n = np.bincount(codes[ok], minlength=len(uniq)).astype(np.float64)[codes]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[c] = ((r[c].to_numpy(np.float64) - 0.5) / n - 0.5).astype(np.float32)
    res = pd.DataFrame(out, index=x.index)
    if check and len(res):
        dev = res.groupby(codes, sort=False).mean().abs()
        if (dev > 1e-3).to_numpy().any():
            col = dev.max().idxmax()
            raise RuntimeError(f"截面排名自检失败：{col} 在第 {uniq[int(dev[col].idxmax())]} 天均值偏离 0 达 {dev[col].max():.3f}")
    return res


@dataclass
class Preprocessor:
    mode: str                                  # "tree" 或 "dense"
    features: list[str] = field(default_factory=list)
    ts_cols: list[str] = field(default_factory=list)          # 做时间序列 z 分数的列（市场状态）
    cs_cols: list[str] = field(default_factory=list)          # 做截面排名的列
    mean_: dict[str, float] = field(default_factory=dict)
    std_: dict[str, float] = field(default_factory=dict)
    na_cols: list[str] = field(default_factory=list)          # 加缺失指示的列
    na_min_rate: float = 0.001                                # 训练段缺失率达到这个比例才加指示列

    def fit(self, x_train: pd.DataFrame, day_train: np.ndarray) -> "Preprocessor":
        if self.mode not in ("tree", "dense"):
            raise ValueError("mode 只能是 tree 或 dense")
        self.features = list(x_train.columns)
        if self.mode == "tree":
            return self
        self.ts_cols = [c for c in self.features if c in MARKET_STATE]
        self.cs_cols = [c for c in self.features if c not in MARKET_STATE]
        for c in self.ts_cols:
            # 市场状态每天一个值：先按天去重再估计，避免股票多的日子权重更大
            per_day = x_train[c].groupby(day_train, sort=False).first()
            self.mean_[c] = float(per_day.mean())
            self.std_[c] = float(per_day.std(ddof=1)) or 1.0
        rate = x_train.isna().mean()
        self.na_cols = [c for c in self.features if rate[c] >= self.na_min_rate]
        return self

    def transform(self, x: pd.DataFrame, day: np.ndarray) -> tuple[np.ndarray, list[str]]:
        if list(x.columns) != self.features:
            raise ValueError("特征列和 fit 时不一致")
        if self.mode == "tree":
            return x.to_numpy(np.float32), list(self.features)
        parts, names = [], []
        if self.cs_cols:
            parts.append(cross_section_rank(x[self.cs_cols], day))
            names += self.cs_cols
        if self.ts_cols:
            z = pd.DataFrame({c: (x[c] - self.mean_[c]) / self.std_[c] for c in self.ts_cols}, index=x.index)
            parts.append(z.astype(np.float32))
            names += self.ts_cols
        out = pd.concat(parts, axis=1)[names]
        ind = x[self.na_cols].isna().astype(np.float32).to_numpy() if self.na_cols else np.empty((len(x), 0), np.float32)
        values = np.nan_to_num(out.to_numpy(np.float32), nan=0.0)
        return np.hstack([values, ind]).astype(np.float32), names + [f"na_{c}" for c in self.na_cols]
