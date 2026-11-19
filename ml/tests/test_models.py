"""任务 3.1 统一模型接口的单元测试：接口形状、参照模型、XGBoost 与仓库原参数一致、早停、分类 / 排序目标。"""

from __future__ import annotations

import numpy as np
import pytest

from ml.models import XGB, RandomScore, SingleFactor, _Chunks
from rebound import params as repo_params
from rebound.model.train import booster_params


def _toy(n_days: int = 30, n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(n_days), n)
    X = rng.normal(size=(n_days * n, 4)).astype(np.float32)
    signal = X[:, 0] - 0.5 * X[:, 1] ** 2 + 0.3 * rng.normal(size=len(X))
    y = np.empty(len(X))
    for d in range(n_days):
        m = day == d
        y[m] = (signal[m].argsort().argsort() + 1) / m.sum()           # 当天百分位
    return X, y.astype(np.float32), day


def test_reference_models():
    X, y, _ = _toy()
    r = RandomScore().fit(X, y)
    assert r.predict(X).shape == (len(X),)
    np.testing.assert_array_equal(r.predict(X), RandomScore().predict(X))        # 固定种子可复现
    f = SingleFactor(column=1, sign=-1.0).fit(X, y)
    np.testing.assert_allclose(f.predict(X), -X[:, 1])
    X2 = X.copy()
    X2[0, 1] = np.nan
    assert np.isfinite(f.predict(X2)).all()


def test_xgb_defaults_match_repo():
    m = XGB()
    assert m.params == repo_params.PARAMS and m.seeds == repo_params.SEEDS
    for s in m.seeds:
        assert m._booster_params(s) == booster_params(repo_params.PARAMS, s)


def test_xgb_learns_and_is_deterministic():
    X, y, day = _toy()
    p = {**repo_params.PARAMS, "n_estimators": 30}
    a = XGB(params=p, seeds=(0, 1)).fit(X, y)
    b = XGB(params=p, seeds=(0, 1)).fit(X, y)
    S = a.predict_seeds(X)
    assert S.shape == (len(X), 2)
    np.testing.assert_array_equal(S, b.predict_seeds(X))                         # 同种子同结果
    assert np.corrcoef(a.predict(X), y)[0, 1] > 0.5


def test_xgb_early_stopping_records_rounds():
    X, y, day = _toy()
    tr, va = day < 20, day >= 20
    m = XGB(params={**repo_params.PARAMS, "n_estimators": 400, "learning_rate": 0.3}, seeds=(0,), early_stop=10)
    m.fit(X[tr], y[tr], X[va], y[va], day[va])
    assert 1 <= m.best_rounds[0] < 400
    full = m.boosters[0].predict(__import__("xgboost").DMatrix(X[va]))
    assert not np.allclose(full, m.predict(X[va]))                              # 预测只用到最佳树数


def test_xgb_classification_and_ranking():
    X, y, day = _toy()
    c = XGB(params={**repo_params.PARAMS, "n_estimators": 20}, seeds=(0,), objective="clf").fit(X, y)
    pc = c.predict(X)
    assert ((pc >= 0) & (pc <= 1)).all()
    with pytest.raises(ValueError):
        XGB(objective="rank").fit(X, y)
    r = XGB(params={**repo_params.PARAMS, "n_estimators": 20}, seeds=(0,), objective="rank").fit(X, y, day_train=day)
    assert np.corrcoef(r.predict(X), y)[0, 1] > 0.3
    # 3.9 的成对抽样设置（lambdarank_*）要原样传给 xgboost
    import json
    r2 = XGB(params={**repo_params.PARAMS, "n_estimators": 5, "lambdarank_pair_method": "mean",
                     "lambdarank_num_pair_per_sample": 1}, seeds=(0,), objective="rank").fit(X, y, day_train=day)
    lp = json.loads(r2.boosters[0].save_config())["learner"]["objective"]["lambdarank_param"]
    assert lp["lambdarank_pair_method"] == "mean" and lp["lambdarank_num_pair_per_sample"] == "1"
    # 排序标签：百分位 → 0 ~ 99 的整数等级（xgboost 3.x 要求），1.0 归到最高档
    assert XGB(objective="rank")._target(np.array([0.0, 0.004, 0.5, 0.999, 1.0])).tolist() == [0, 0, 50, 99, 99]


def test_chunks_respect_day_boundaries(monkeypatch):
    import ml.models as mm
    monkeypatch.setattr(mm, "CHUNK", 100)
    X, y, day = _toy(n_days=10, n=60)
    seen = []
    it = _Chunks(X, y, day)
    while it.next(lambda **kw: seen.append(kw["qid"])):
        pass
    assert sum(len(q) for q in seen) == len(X)
    for a, b in zip(seen, seen[1:]):
        assert a[-1] != b[0]                                                      # 同一天不被拆到两块


def test_random_forest():
    from ml.models import RandomForest
    X, y, _ = _toy()
    X[::11, 2] = np.nan                                                          # 缺失值直接交给 sklearn
    p = {"max_samples": 0.5, "max_features": 0.5, "min_samples_leaf": 20}
    a = RandomForest(params=p, seeds=(0, 1)).fit(X, y)
    S = a.predict_seeds(X)
    assert S.shape == (len(X), 2) and np.isfinite(S).all()
    # 可复现：同种子长出的树完全相同；sklearn 并行把各棵树的预测相加，相加顺序不固定，只会差在最后一位（~1e-16）
    np.testing.assert_allclose(S, RandomForest(params=p, seeds=(0, 1)).fit(X, y).predict_seeds(X), rtol=0, atol=1e-12)
    assert not np.array_equal(S[:, 0], S[:, 1])                                  # 不同种子不同森林
    assert np.corrcoef(a.predict(X), y)[0, 1] > 0.5
    assert a.info()["params"]["n_estimators"] == 300
    simple = RandomForest(params={**p, "min_samples_leaf": 200})
    assert simple.complexity < a.complexity


def test_daily_ic_metric_matches_evaluate():
    """早停用的 IC 指标和 evaluate 的逐日 Spearman 一致（没有并列时完全相同）。"""
    import pandas as pd
    from ml import evaluate as ev
    from ml.models import daily_ic_metric
    X, y, day = _toy(n_days=8, n=50)
    pred = X[:, 0] + 0.5 * np.random.default_rng(3).normal(size=len(X))
    name, val = daily_ic_metric(day, y)(pred, None)
    k = pd.DataFrame({"trade_date": pd.to_datetime("2021-01-04") + pd.to_timedelta(day, "D"), "ret_o1o5": y,
                      "excess_o1o5": y - 0.5, "label_pct": y})
    # 玩具标签是 float32，和 evaluate 内部的 float64 排名差在 1e-9 量级
    assert name == "ic" and val == pytest.approx(ev.daily_rank_ic(ev._frame(pred, k)).mean(), abs=1e-8)
    with pytest.raises(ValueError):
        daily_ic_metric(day[::-1], y)


def test_xgb_ic_early_stopping():
    X, y, day = _toy()
    tr, va = day < 20, day >= 20
    m = XGB(params={**repo_params.PARAMS, "n_estimators": 300, "learning_rate": 0.3}, seeds=(0,), early_stop=10, early_metric="ic")
    m.fit(X[tr], y[tr], X[va], y[va], day[va])
    assert 1 <= m.best_rounds[0] < 300 and m.info()["early_metric"] == "ic"
