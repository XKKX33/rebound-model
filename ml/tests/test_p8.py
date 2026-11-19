"""P8：随机森林路径分解的完备性（各特征贡献之和 + 基线 = 预测），以及交付范围。"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ml.p8_explain_rf import forest_contrib


def test_saabas_complete_with_missing_values():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5)).astype(np.float32)
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(size=2000)
    X[rng.random(X.shape) < 0.05] = np.nan                                  # 有缺失值（sklearn 树原生处理）
    forests = [RandomForestRegressor(n_estimators=15, min_samples_leaf=20, max_samples=0.5, random_state=s).fit(X, y) for s in (0, 1)]
    contrib, base = forest_contrib(forests, X)
    pred = np.mean([f.predict(X) for f in forests], axis=0)
    assert np.abs(base + contrib.sum(axis=1) - pred).max() < 1e-10
    assert np.abs(contrib).mean(axis=0).argmax() == 0                       # 最重要的是第 0 个特征


def test_delivery_models_consistent():
    from ml import api, predict
    from ml.p8_delivery import M3
    assert list(api.MODELS) == list(predict.MODELS) == M3


def test_numexpr_disabled():
    """本机 numexpr 2.14.1 偶尔把大数组运算整列算成 0（PREREG 冻结后的运行记录），导入 ml 时必须关掉。"""
    import pandas as pd
    import ml  # noqa: F401
    assert pd.get_option("compute.use_numexpr") is False
