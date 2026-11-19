"""对外接口 ml/api.py（任务 6.1）：返回列、排序、行业去重、错误提示。需要真实的分数文件，没有就跳过。"""

import pandas as pd
import pytest

from ml import api
from ml.common import data_dir

HAVE = any((data_dir() / sub / "xgb_F69__select.parquet").exists() for sub in ("scores", "api_export/scores"))
pytestmark = pytest.mark.skipif(not HAVE, reason="没有分数文件 data/ml/scores/")
DAY = "2022-06-15"


def test_scores_sorted_and_ranked():
    s = api.scores("xgb_F69", DAY)
    assert list(s.columns) == ["ts_code", "name", "industry", "score", "rank", "pct"]
    assert s["score"].is_monotonic_decreasing and s["rank"].tolist() == list(range(1, len(s) + 1))
    assert s["pct"].iloc[0] == 1 and 0 < s["pct"].iloc[-1] < 0.01
    assert s["ts_code"].is_unique


def test_picks_respect_industry_cap():
    p = api.picks("xgb_F69", DAY, top_n=10, max_per_industry=1)
    assert len(p) == 10 and p["industry"].value_counts().max() == 1
    pool = api.scores("xgb_F69", DAY)
    assert p["ts_code"].iloc[0] == pool["ts_code"].iloc[0]                 # 第一名一定入选
    assert p["rank"].is_monotonic_increasing


def test_outcome_columns():
    p = api.picks("xgb_F69", DAY, top_n=6, with_outcome=True)
    assert {"ret_o1o5", "excess_o1o5"} <= set(p.columns) and p["ret_o1o5"].notna().all()


def test_errors_are_api_errors():
    with pytest.raises(api.ApiError):
        api.scores("no_such_model", DAY)
    with pytest.raises(api.ApiError):
        api.scores("xgb_F69", "2022-06-18")                               # 星期六
    with pytest.raises(api.ApiError):
        api.explain("xgb_F69", DAY, "600000.SH")                          # 验证段的日子没有预先算好的解释


def test_feature_info_complete():
    fi = api.feature_info()
    assert len(fi) == 69 + 11 and fi["名称"].is_unique
    assert (fi["中文名"] != "").all() and (fi["计算"] != "").all()
    assert isinstance(api.data_summary()["股票数"], int)
    assert isinstance(api.leaderboard(), pd.DataFrame) and len(api.models()) == len(api.MODELS)


def test_only_three_models():
    assert list(api.MODELS) == ["xgb_F69", "rf_F69", "lstm_SF69"] and api.DEFAULT_MODEL == "xgb_F69"
    with pytest.raises(api.ApiError):
        api.scores("xgb_F28", DAY)                                          # 不在交付范围（PREREG 补充登记十三）
    assert api._cn_name("seq_ret").startswith("序列：日收益") and api._cn_name("na_dd250").startswith("缺失：")


HAVE_TEST = all((data_dir() / "explain" / f).exists() or (data_dir() / "api_export" / f"explain_{f}").exists()
                for f in api.EXPLAINED.values())


@pytest.mark.skipif(not HAVE_TEST, reason="没有测试段的解释数据")
@pytest.mark.parametrize("mid", ["xgb_F69", "rf_F69", "lstm_SF69"])
def test_explain(mid):
    day = "2025-03-14"
    code = api.picks(mid, day, top_n=1)["ts_code"].iloc[0]
    e = api.explain(mid, day, code)
    assert list(e.columns) == ["特征", "中文名", "数值", "贡献", "方向"] and len(e) == 8
    assert e["贡献"].abs().is_monotonic_decreasing                         # 按 |贡献| 从大到小
    assert e.attrs["score"] > e.attrs["base"] and e.attrs["method"]         # 第一名的分数高于基线
    assert (e["中文名"] != "").all()


def test_nav():
    for period in ("test", "long"):
        n = api.nav("xgb_F69", "DB-K30-b8", period=period)
        assert n["nav"].iloc[0] == 1.0 and n["nav"].notna().all() and "pool_nav" in n
    with pytest.raises(api.ApiError):
        api.nav("xgb_F69", "DB-K99-b1")
    with pytest.raises(api.ApiError):
        api.nav("xgb_F69", period="2015")
