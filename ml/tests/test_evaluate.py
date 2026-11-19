"""P2 评估协议的单元测试：划分有间隔、训练标签不泄漏、各项指标和手算 / 第三方库一致、Newey-West、判定规则。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from ml import evaluate as ev
from rebound.model.train import TrainingRows, training_set


def _keys(days: int = 40, n: int = 50, seed: int = 0, carry_every: int = 7) -> pd.DataFrame:
    """玩具面板：每天 n 只，卖出日一般是 T+5，每隔几行顺延几天，最后 5 天没有有效收益。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-12-01", periods=days)
    di = np.repeat(np.arange(days), n)
    ret = rng.normal(0, 0.05, days * n)
    exit_di = di + 5 + np.where(np.arange(days * n) % carry_every == 0, 3, 0)
    bad = exit_di >= days
    ret[bad] = np.nan
    exit_di[bad] = -1
    k = pd.DataFrame({"trade_date": dates[di], "ts_code": np.tile([f"S{j}" for j in range(n)], days),
                      "di": di, "ret_o1o5": ret, "exit_di": exit_di})
    k["label_pct"] = k.groupby("di")["ret_o1o5"].rank(pct=True)
    k["excess_o1o5"] = k["ret_o1o5"] - k.groupby("di")["ret_o1o5"].transform("mean")
    return k


# ---- 划分与训练标签 ----------------------------------------------------------


def test_fit_rows_have_gap():
    k = _keys()
    fit = 20
    rows = ev.fit_rows(k, fit)
    assert k.loc[rows, "exit_di"].max() < fit                     # 训练行都已在重训日前卖出
    assert k.loc[rows, "di"].max() <= fit - 6                       # 没顺延的行，最后一个信号日是 fit − 6
    assert not rows[(k["di"] >= fit - 5).to_numpy()].any()          # 重训日前 5 天的信号全部清掉


def test_train_labels_match_repo_training_set():
    """和仓库 training_set 逐位相同（同一个 fit、同一批行）。"""
    k = _keys()
    fit = 25
    rows = TrainingRows(dates=pd.DatetimeIndex(k["trade_date"].unique()), codes=np.array([]),
                        di=k["di"].to_numpy(), ji=np.zeros(len(k), int), x=np.zeros((len(k), 1)),
                        member={"nocap": np.ones(len(k), bool)}, ret=k["ret_o1o5"].to_numpy(),
                        exit_day=k["exit_di"].to_numpy())
    ts = training_set(rows, "nocap", fit)
    mine = ev.fit_rows(k, fit)
    np.testing.assert_array_equal(np.flatnonzero(mine), ts.rows)
    np.testing.assert_array_equal(ev.train_labels(k, mine), ts.y[ts.keep])


def test_train_labels_ignore_unknown_rows():
    """未卖出的行不影响已知行的百分位：把未卖出行的收益随便改，标签不变。"""
    k = _keys()
    fit = 25
    rows = ev.fit_rows(k, fit)
    y1 = ev.train_labels(k, rows)
    k2 = k.copy()
    k2.loc[~rows & k2["ret_o1o5"].notna().to_numpy(), "ret_o1o5"] *= -100
    np.testing.assert_array_equal(y1, ev.train_labels(k2, rows))


def test_protocol_a_on_config_dates():
    k = _keys(days=600, n=5, carry_every=11)
    k["trade_date"] = pd.bdate_range("2020-06-01", periods=600)[k["di"]]
    cfg = {"split": {"valid_start": "2021-01-01", "valid_end": "2021-06-30", "test_start": "2021-07-01",
                     "test_end": "2022-12-31", "test_split_at": "2022-01-01"}}
    p = ev.protocol_a(k, cfg)
    first_valid = k.loc[p.select.eval_rows, "di"].min()
    assert k.loc[p.select.fit_rows, "exit_di"].max() < first_valid
    assert k.loc[p.final.fit_rows, "exit_di"].max() < k.loc[p.final.eval_rows, "di"].min()
    assert not (p.select.eval_rows & p.final.eval_rows).any()
    t1, t2 = p.final.parts["T1"], p.final.parts["T2"]
    assert not (t1 & t2).any() and ((t1 | t2) == p.final.eval_rows).all()
    years = [p.final.parts[y] for y in ("2021", "2022")]                # 测试段逐年
    assert ((years[0] | years[1]) == p.final.eval_rows).all() and not (years[0] & years[1]).any()
    # 测试段按年重训（PREREG 补充登记十）：每年一次，第一次 = 原最终阶段；各年评估行不重叠、合起来 = 全部测试行；
    # 训练行都在重训日前卖出，且训练行逐年只增不减（扩张窗口）
    assert [r.name for r in p.refits] == ["2021", "2022"]
    assert p.refits[0].fit_di == p.final.fit_di and (p.refits[0].fit_rows == p.final.fit_rows).all()
    assert ((p.refits[0].eval_rows | p.refits[1].eval_rows) == p.final.eval_rows).all()
    assert not (p.refits[0].eval_rows & p.refits[1].eval_rows).any()
    for r in p.refits:
        assert k.loc[r.fit_rows, "exit_di"].max() < r.fit_di <= k.loc[r.eval_rows, "di"].min()
    assert (p.refits[1].fit_rows >= p.refits[0].fit_rows).all() and p.refits[1].fit_rows.sum() > p.refits[0].fit_rows.sum()
    assert k.loc[p.refits[1].fit_rows, "trade_date"].dt.year.min() == 2020                      # 每次都从头开始


# ---- 指标 --------------------------------------------------------------------


def _scored(seed: int = 1, signal: float = 0.3):
    k = _keys(seed=seed).dropna(subset=["ret_o1o5"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    score = signal * k["ret_o1o5"].to_numpy() / 0.05 + rng.normal(size=len(k))
    return k, score


def test_rank_ic_matches_scipy():
    k, score = _scored()
    f = ev._frame(score, k)
    ic = ev.daily_rank_ic(f)
    for d, g in f.groupby("date"):
        assert ic[d] == pytest.approx(spearmanr(g["score"], g["ret"]).statistic, abs=1e-12)


def test_rank_ic_handles_ties():
    k, _ = _scored()
    score = np.round(k["ret_o1o5"].to_numpy(), 2)                    # 大量并列
    f = ev._frame(score, k)
    ic = ev.daily_rank_ic(f)
    d0 = f["date"].iloc[0]
    g = f[f["date"] == d0]
    assert ic[d0] == pytest.approx(spearmanr(g["score"], g["ret"]).statistic, abs=1e-12)


def test_newey_west():
    import statsmodels.api as sm
    rng = np.random.default_rng(3)
    e = rng.normal(size=500)
    x = 0.1 + np.convolve(e, np.ones(5) / 5, mode="same")         # 有自相关
    for lags in (0, 5, 10):
        fit = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})
        assert ev.newey_west_t(x, lags) == pytest.approx(float(fit.tvalues[0]), rel=1e-10)
    assert abs(ev.newey_west_t(x, 10)) < abs(ev.newey_west_t(x, 0))   # 正自相关时 NW t 更小


def test_topk_and_groups_by_hand():
    dates = pd.to_datetime(["2021-01-04"] * 40)
    ex = np.arange(40, dtype=float) - 19.5
    k = pd.DataFrame({"trade_date": dates, "ret_o1o5": ex, "excess_o1o5": ex, "label_pct": (np.arange(40) + 1) / 40})
    f = ev._frame(ex, k)                                             # 分数 = 超额：完美排序
    assert ev.topk_excess(f, 6).iloc[0] == pytest.approx(ex[-6:].mean())
    g = ev.group_returns(f)
    np.testing.assert_allclose(g.iloc[0].to_numpy(), ex.reshape(10, 4).mean(axis=1))
    assert ev.daily_ndcg(f, 30).iloc[0] == pytest.approx(1.0)
    assert ev.daily_rank_ic(f).iloc[0] == pytest.approx(1.0)


def test_auc_matches_sklearn():
    k, score = _scored()
    f = ev._frame(score, k)
    auc = ev.daily_auc(f, top=0.2)
    for d, g in list(f.groupby("date"))[:5]:
        assert auc[d] == pytest.approx(roc_auc_score(g["pct"] > 0.8, g["score"]), abs=1e-12)


def test_evaluate_random_vs_signal():
    k, score = _scored(signal=0.5)
    good = ev.evaluate(score, k, auc=True, ndcg=True)["summary"]["全部"]
    rnd = ev.evaluate(np.random.default_rng(9).normal(size=len(k)), k)["summary"]["全部"]
    assert good["ic_mean"] > 0.2 and good["top6"] > 0 and good["long_short"] > 0 and good["monotonic"] > 0.8
    assert good["auc"] > 0.6 and 0 < good["ndcg30"] <= 1
    assert abs(rnd["ic_mean"]) < 0.05


def test_evaluate_parts_and_bad_scores():
    k, score = _scored()
    half = (k["trade_date"] < k["trade_date"].iloc[len(k) // 2]).to_numpy()
    res = ev.evaluate(score, k, parts={"前": half, "后": ~half})
    assert res["summary"]["前"]["days"] + res["summary"]["后"]["days"] == len(res["daily"])
    bad = score.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="空值"):
        ev.evaluate(bad, k)
    with pytest.raises(ValueError, match="不一致"):
        ev.evaluate(score[:-1], k)


# ---- 比较与判定 ----------------------------------------------------------------


def test_compare_and_verdict():
    idx = pd.bdate_range("2021-01-04", periods=480)
    rng = np.random.default_rng(5)
    a = pd.Series(0.08 + rng.normal(0, 0.1, len(idx)), index=idx)
    same = ev.compare(a, a)
    assert same["diff"] == 0 and same["ci"] == [0.0, 0.0]
    better = ev.compare(a + 0.03, a)
    assert better["ci"][0] > 0
    assert ev.verdict(better, 0.0, 0.01) == "更好"
    assert ev.verdict(better, 0.0, 0.05) == "可能更好"                # 差值小于种子极差
    noise = ev.compare(a, pd.Series(0.08 + rng.normal(0, 0.1, len(idx)), index=idx))
    assert ev.verdict(noise, 0, 0) == "分不出"
    assert ev.seed_spread([0.090, 0.095, 0.092]) == pytest.approx(0.005)


def test_pick_best_tie_prefers_simpler():
    res = [{"valid_ic": 0.100, "complexity": 3}, {"valid_ic": 0.0985, "complexity": 1}, {"valid_ic": 0.090, "complexity": 0}]
    assert ev.pick_best(res)["complexity"] == 1                    # 差 0.0015 < 0.002，取更简单的
    res[1]["valid_ic"] = 0.097
    assert ev.pick_best(res)["complexity"] == 3
