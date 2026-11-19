"""任务 1.5 预处理的单元测试：不跨日泄漏、只用训练段拟合、取值范围、缺失处理。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.preprocess import Preprocessor, cross_section_rank


def _toy(seed: int = 0, days: int = 6, n: int = 50) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(days), n)
    x = pd.DataFrame({
        "r5": rng.standard_t(3, days * n),                       # 厚尾
        "turn_f": rng.lognormal(0, 1, days * n),
        "mkt_ret20": np.repeat(rng.normal(0, 0.05, days), n),     # 市场状态：同一天相同
    })
    x.loc[[3, 70, 140], "turn_f"] = np.nan
    return x, day


def test_rank_range_and_center():
    x, day = _toy()
    r = cross_section_rank(x[["r5"]], day)
    assert r["r5"].between(-0.5, 0.5).all()
    assert r.groupby(day)["r5"].mean().abs().max() < 1e-6          # 每天居中


def test_rank_does_not_depend_on_other_days():
    x, day = _toy()
    full = cross_section_rank(x[["r5", "turn_f"]], day)
    x2 = x.copy()
    x2.loc[day == 5, "r5"] *= 1000                                 # 改动最后一天
    other = cross_section_rank(x2[["r5", "turn_f"]], day)
    pd.testing.assert_frame_equal(full[day < 5], other[day < 5])
    one = cross_section_rank(x[day == 2][["r5", "turn_f"]], day[day == 2])
    pd.testing.assert_frame_equal(full[day == 2], one)


def test_rank_ignores_outliers():
    x, day = _toy()
    clipped = x.copy()
    lo, hi = x["r5"].quantile([0.01, 0.99])
    clipped["r5"] = x["r5"].clip(lo, hi)
    a = cross_section_rank(x[["r5"]], day)["r5"]
    b = cross_section_rank(clipped[["r5"]], day)["r5"]
    assert (a - b).abs().max() < 0.05                               # 缩尾只让两端出现并列，排名几乎不变


def test_fit_uses_train_only():
    x, day = _toy()
    train = day < 4
    p1 = Preprocessor("dense").fit(x[train], day[train])
    x2 = x.copy()
    x2.loc[~train, ["r5", "mkt_ret20"]] = 99.0                      # 验证 / 测试段随便改
    p2 = Preprocessor("dense").fit(x2[train], day[train])
    assert p1.mean_ == p2.mean_ and p1.std_ == p2.std_ and p1.na_cols == p2.na_cols
    per_day = x.loc[train, "mkt_ret20"].groupby(day[train]).first()
    assert p1.mean_["mkt_ret20"] == pytest.approx(per_day.mean())


def test_dense_output():
    x, day = _toy()
    p = Preprocessor("dense").fit(x[day < 4], day[day < 4])
    values, names = p.transform(x, day)
    assert names == ["r5", "turn_f", "mkt_ret20", "na_turn_f"]
    assert not np.isnan(values).any()
    na = values[:, names.index("na_turn_f")]
    assert na.sum() == 3 and (values[na == 1, names.index("turn_f")] == 0).all()


def test_tree_passthrough():
    x, day = _toy()
    p = Preprocessor("tree").fit(x, day)
    values, names = p.transform(x, day)
    assert names == list(x.columns)
    np.testing.assert_array_equal(values, x.to_numpy(np.float32))


def test_rank_self_check_catches_bad_output(monkeypatch):
    """自检：如果排名结果不居中，就报错。用一个故意算错的 rank 模拟异常。"""
    x, day = _toy()
    import pandas.core.groupby.generic as gg
    orig = gg.DataFrameGroupBy.rank
    monkeypatch.setattr(gg.DataFrameGroupBy, "rank", lambda self, **k: orig(self, **k) * 0 + 1)
    with pytest.raises(RuntimeError, match="自检失败"):
        cross_section_rank(x[["r5"]], day)
