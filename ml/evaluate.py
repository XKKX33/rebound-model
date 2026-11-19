"""任务 P2 评估协议的实现：时间划分、训练标签、预测层指标、显著性。

协议的文字版（为什么这样定、哪些沿用仓库、哪些是新增）见 `ml/PREREG.md` 第二节；本文件只负责照着算。

用法（P3 起每个模型都这样评估）：

    from ml import evaluate as ev
    keys = ev.load_keys()                          # 主面板的键和标签列
    plan = ev.protocol_a(keys)                     # 两次拟合、各段的行
    fit = plan.select                              # 选择阶段：训练段 → 验证段
    y = ev.train_labels(keys, fit.fit_rows)        # 训练标签（只在已知行里排百分位）
    ... 模型在 keys[fit.fit_rows] 上训练，对 keys[fit.eval_rows] 打分 → score ...
    res = ev.evaluate(score, keys[fit.eval_rows])  # 指标 + 逐日序列

逐日序列（IC、Top-K 超额）保存下来，模型两两比较时用 `compare`。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.common import CFG
from rebound.backtest.stats import month_ci
from rebound.labels import percentile

#: 一天至少这么多有效行才计算当天的指标
MIN_ROWS = 30
#: Newey-West 的滞后天数（标签期限 5 天的 2 倍，偏保守），敏感性另报 5、20
NW_LAGS = 10
TOP_K = (6, 30)
N_GROUPS = 10
#: 打平：验证 IC 相差不到这么多，选更简单的那组
TIE_IC = 0.002
KEY_COLS = ["trade_date", "ts_code", "di", "ret_o1o5", "exit_di", "label_pct", "excess_o1o5"]


# --------------------------------------------------------------------------- #
# 时间划分
# --------------------------------------------------------------------------- #


def load_keys() -> pd.DataFrame:
    """主面板的键和标签列（不含特征）。"""
    from ml.dataset import load_panel
    return load_panel(KEY_COLS)


def first_di_on_or_after(keys: pd.DataFrame, day: str | pd.Timestamp) -> int:
    """某个日期当天或之后的第一个交易日的下标。"""
    d = keys.loc[keys["trade_date"] >= pd.Timestamp(day), "di"]
    if d.empty:
        raise ValueError(f"{day} 之后没有交易日")
    return int(d.min())


def fit_rows(keys: pd.DataFrame, fit_di: int, start_di: int = 0) -> np.ndarray:
    """在交易日 fit_di 重训时能用的行：信号日在 [start_di, fit_di) 内，收益有效，且实际卖出日早于 fit_di。

    和仓库 `rebound.model.train.training_set` 的规则完全相同（`exit_day >= 0 & exit_day < fit`）。
    没有顺延的行卖出日是 T+5，所以 fit_di 之前最后 5 个交易日的信号自动被清掉，这就是段与段之间的间隔。
    """
    di = keys["di"].to_numpy()
    ex = keys["exit_di"].to_numpy()
    ret = keys["ret_o1o5"].to_numpy()
    return (di >= start_di) & (di < fit_di) & np.isfinite(ret) & (ex >= 0) & (ex < fit_di)


def train_labels(keys: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """训练标签：当天收益百分位，只在 rows（已知的行）里排；返回长度 = rows 里 True 的个数。

    直接调用仓库的 `rebound.labels.percentile(known=...)`：没卖出的行不影响别人的百分位。
    """
    y = percentile(keys["ret_o1o5"].to_numpy(), keys["di"].to_numpy(), known=rows)
    return y[rows]


def eval_rows(keys: pd.DataFrame, start: str, end: str) -> np.ndarray:
    """评估段：信号日在 [start, end] 内、收益有效的行。"""
    d = keys["trade_date"]
    return ((d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end)) & keys["ret_o1o5"].notna()).to_numpy()


@dataclass
class Stage:
    name: str
    fit_di: int                 # 重训日（下一段第一个交易日）
    fit_rows: np.ndarray        # 训练行
    eval_rows: np.ndarray       # 评估行
    parts: dict[str, np.ndarray] = field(default_factory=dict)   # 评估段再分的小段（测试期 T1 / T2）


@dataclass
class ProtocolA:
    select: Stage               # 训练段 → 验证段
    final: Stage                # 测试段的评估（eval_rows、T1 / T2 / 全段 / 逐年）；fit_rows = 第一次重训（2023 年）的训练行
    refits: list[Stage] = field(default_factory=list)   # 测试段按年重训（PREREG 补充登记十）：每年一次，eval_rows = 这一年的测试行


def protocol_a(keys: pd.DataFrame, cfg: dict | None = None) -> ProtocolA:
    """主协议 A（PREREG 2.1）。"""
    s = (cfg or CFG)["split"]
    fit1 = first_di_on_or_after(keys, s["valid_start"])
    fit2 = first_di_on_or_after(keys, s["test_start"])
    select = Stage("select", fit1, fit_rows(keys, fit1), eval_rows(keys, s["valid_start"], s["valid_end"]))
    test = eval_rows(keys, s["test_start"], s["test_end"])
    cut = pd.Timestamp(s["test_split_at"])
    before = (keys["trade_date"] < cut).to_numpy()
    final = Stage("final", fit2, fit_rows(keys, fit2), test,
                  parts={"T1": test & before, "T2": test & ~before, "全段": test, **year_parts(keys, test)})
    return ProtocolA(select, final, test_refits(keys, test, s["test_start"]))


def test_refits(keys: pd.DataFrame, test: np.ndarray, test_start: str) -> list[Stage]:
    """测试段按年重训（PREREG 补充登记十）：测试期每年第一个交易日重训一次（扩张窗口），模型只给这一年的测试行打分。

    第一次重训日 = 测试段第一天（与原“训练一次”的最终阶段相同）。各年的 eval_rows 互不重叠，合起来正好是全部测试行。
    """
    years = keys["trade_date"].dt.year.to_numpy()
    out = []
    for y in np.unique(years[test]):
        fit = first_di_on_or_after(keys, max(pd.Timestamp(test_start), pd.Timestamp(f"{y}-01-01")))
        out.append(Stage(str(y), fit, fit_rows(keys, fit), test & (years == y)))
    return out


def year_parts(keys: pd.DataFrame, rows: np.ndarray) -> dict[str, np.ndarray]:
    """评估行按自然年再分（PREREG 补充登记二第 4 条：测试段逐年 IC，只描述）。"""
    years = keys["trade_date"].dt.year.to_numpy()
    return {str(y): rows & (years == y) for y in np.unique(years[rows])}


def protocol_b(keys: pd.DataFrame, years=range(2017, 2027)) -> list[Stage]:
    """协议 B（PREREG 2.1）：扩张窗口、每年第一个交易日重训，评估这一年。"""
    out = []
    for y in years:
        fit = first_di_on_or_after(keys, f"{y}-01-01")
        out.append(Stage(str(y), fit, fit_rows(keys, fit), eval_rows(keys, f"{y}-01-01", f"{y}-12-31")))
    return out


# --------------------------------------------------------------------------- #
# 预测层指标（PREREG 2.2）
# --------------------------------------------------------------------------- #


def _frame(score, keys: pd.DataFrame) -> pd.DataFrame:
    score = np.asarray(score, dtype=np.float64)
    if len(score) != len(keys):
        raise ValueError(f"分数 {len(score)} 行，评估行 {len(keys)} 行，不一致")
    if np.isnan(score).any():
        raise ValueError("分数里有空值：模型必须给每一行打分")
    f = pd.DataFrame({"date": keys["trade_date"].to_numpy(), "score": score,
                      "ret": keys["ret_o1o5"].to_numpy(np.float64),
                      "excess": keys["excess_o1o5"].to_numpy(np.float64),
                      "pct": keys["label_pct"].to_numpy(np.float64)})
    f = f[np.isfinite(f["ret"])]
    n = f.groupby("date")["ret"].transform("size")
    return f[n >= MIN_ROWS].reset_index(drop=True)


def daily_rank_ic(f: pd.DataFrame) -> pd.Series:
    """逐日 Spearman(分数, 收益)：当天内各自排名（并列取平均）后求 Pearson 相关。"""
    g = f.groupby("date", sort=True)
    a = g["score"].rank(method="average").to_numpy()
    b = g["ret"].rank(method="average").to_numpy()
    t = pd.DataFrame({"date": f["date"].to_numpy(), "a": a, "b": b, "ab": a * b, "aa": a * a, "bb": b * b})
    s = t.groupby("date", sort=True).mean()
    cov = s["ab"] - s["a"] * s["b"]
    va, vb = s["aa"] - s["a"] ** 2, s["bb"] - s["b"] ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = cov / np.sqrt(va * vb)
    return ic.rename("ic")


def newey_west_t(x, lags: int = NW_LAGS) -> float:
    """序列均值的 Newey-West t 值（Bartlett 核）。lags=0 时等于普通 t（方差用 1/n）。"""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else float("nan")


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    sd = float(ic.std(ddof=1))
    return {"days": int(len(ic)), "ic_mean": float(ic.mean()), "ic_std": sd,
            "icir": float(ic.mean() / sd) if sd > 0 else float("nan"),
            "ic_pos": float((ic > 0).mean()),
            "nw_t": newey_west_t(ic, NW_LAGS), "nw_t_lag5": newey_west_t(ic, 5), "nw_t_lag20": newey_west_t(ic, 20),
            "naive_t": newey_west_t(ic, 0)}


def _order(f: pd.DataFrame) -> pd.DataFrame:
    """每天按分数从高到低排好；并列按原行顺序（稳定排序）。"""
    return f.sort_values(["date", "score"], ascending=[True, False], kind="mergesort")


def topk_excess(f: pd.DataFrame, k: int) -> pd.Series:
    """逐日：分数最高的 k 只的平均超额。"""
    return _order(f).groupby("date", sort=True).head(k).groupby("date", sort=True)["excess"].mean().rename(f"top{k}")


def group_returns(f: pd.DataFrame, n_groups: int = N_GROUPS) -> pd.DataFrame:
    """逐日按分数排名等分 n 组（第 1 组分数最低），每组平均超额；返回 [日, 组] 的表。"""
    r = f.groupby("date", sort=True)["score"].rank(method="first").to_numpy()
    n = f.groupby("date", sort=True)["score"].transform("size").to_numpy()
    grp = np.floor((r - 1) * n_groups / n).astype(int) + 1
    return f.assign(grp=grp).groupby(["date", "grp"])["excess"].mean().unstack("grp")


def daily_auc(f: pd.DataFrame, top: float = 0.2) -> pd.Series:
    """逐日 AUC：正类 = 当天收益百分位排名前 top（label_pct > 1 − top）。"""
    pos = (f["pct"] > 1 - top).to_numpy()
    r = f.groupby("date", sort=True)["score"].rank(method="average").to_numpy()
    t = pd.DataFrame({"date": f["date"].to_numpy(), "pos": pos, "rpos": np.where(pos, r, 0.0)})
    s = t.groupby("date", sort=True).agg(n=("pos", "size"), n1=("pos", "sum"), rsum=("rpos", "sum"))
    n0 = s["n"] - s["n1"]
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (s["rsum"] - s["n1"] * (s["n1"] + 1) / 2) / (s["n1"] * n0)
    return auc.rename("auc")


def daily_ndcg(f: pd.DataFrame, k: int = 30) -> pd.Series:
    """逐日 NDCG@k：相关度 = 当天收益百分位，线性增益，折扣 1/log2(名次+1)。"""
    disc = 1 / np.log2(np.arange(2, k + 2))
    by_score = _order(f).groupby("date", sort=True).head(k)
    pos = by_score.groupby("date", sort=True).cumcount().to_numpy()
    dcg = pd.Series(by_score["pct"].to_numpy() * disc[pos], index=by_score["date"].to_numpy()).groupby(level=0).sum()
    ideal = f.sort_values(["date", "pct"], ascending=[True, False], kind="mergesort").groupby("date", sort=True).head(k)
    ipos = ideal.groupby("date", sort=True).cumcount().to_numpy()
    idcg = pd.Series(ideal["pct"].to_numpy() * disc[ipos], index=ideal["date"].to_numpy()).groupby(level=0).sum()
    return (dcg / idcg).rename("ndcg")


def evaluate(score, keys: pd.DataFrame, *, parts: dict[str, np.ndarray] | None = None,
             auc: bool = False, ndcg: bool = False) -> dict:
    """一组分数在一段评估行上的全部预测层指标。

    score 与 keys 逐行对齐。parts（可选）：在 keys 里再划的小段（如测试期 T1 / T2），布尔数组与 keys 等长，
    每段单独汇总。返回 {"summary": {段: 指标}, "daily": 逐日序列 DataFrame, "groups": {段: 10 组均值}}。
    """
    f = _frame(score, keys)
    daily = pd.concat([daily_rank_ic(f)] + [topk_excess(f, k) for k in TOP_K], axis=1)
    groups = group_returns(f)
    daily["long_short"] = groups[N_GROUPS] - groups[1]
    if auc:
        daily = daily.join(daily_auc(f))
    if ndcg:
        daily = daily.join(daily_ndcg(f))
    daily.index.name = "date"

    seg_days = {"全部": daily.index}
    if parts:
        dates = keys["trade_date"].to_numpy()
        seg_days = {name: daily.index.intersection(pd.DatetimeIndex(np.unique(dates[np.asarray(m)]))) for name, m in parts.items()}
    summary, gtab = {}, {}
    for name, days in seg_days.items():
        d = daily.loc[days]
        g = groups.loc[days].mean()
        row = ic_summary(d["ic"])
        for k in TOP_K:
            row[f"top{k}"] = float(d[f"top{k}"].mean())
            row[f"top{k}_nw_t"] = newey_west_t(d[f"top{k}"])
        row["long_short"] = float(d["long_short"].mean())
        row["monotonic"] = float(pd.Series(g.to_numpy()).corr(pd.Series(np.arange(1, N_GROUPS + 1)), method="spearman"))
        if auc:
            row["auc"] = float(d["auc"].mean())
        if ndcg:
            row["ndcg30"] = float(d["ndcg"].mean())
        summary[name] = row
        gtab[name] = g
    return {"summary": summary, "daily": daily, "groups": gtab}


# --------------------------------------------------------------------------- #
# 显著性与判定（PREREG 2.4）
# --------------------------------------------------------------------------- #


def compare(a: pd.Series, b: pd.Series, *, seed: int | None = None) -> dict:
    """两条逐日序列（同名指标，按日期对齐）之差 a − b：均值、按月整块重抽 95% 区间、NW t。"""
    d = (a - b).dropna()
    d.index = pd.DatetimeIndex(d.index)
    kw = {} if seed is None else {"seed": seed}
    lo, hi = month_ci(d, **kw)
    return {"days": int(len(d)), "diff": float(d.mean()), "ci": [lo, hi], "nw_t": newey_west_t(d)}


def seed_spread(per_seed_ic: list[float]) -> float:
    """种子极差：各种子 IC 均值的最大 − 最小；没有随机性的模型传一个值，得 0。"""
    return float(np.max(per_seed_ic) - np.min(per_seed_ic)) if len(per_seed_ic) else 0.0


def verdict(cmp: dict, spread_a: float, spread_b: float) -> str:
    """A 比 B 好的判定（PREREG 2.4）：区间下限 > 0 且差值 > 较大的种子极差 → 更好；只有前者 → 可能更好；
    对称地判更差；区间跨 0 → 分不出。"""
    lo, hi = cmp["ci"]
    spread = max(spread_a, spread_b)
    if lo > 0:
        return "更好" if cmp["diff"] > spread else "可能更好"
    if hi < 0:
        return "更差" if -cmp["diff"] > spread else "可能更差"
    return "分不出"


def pick_best(results: list[dict], key: str = "valid_ic", simpler: str = "complexity") -> dict:
    """选择规则（PREREG 2.5）：验证 IC 最高；与最高差不到 TIE_IC 的算打平，打平时取 complexity 最小的。"""
    top = max(r[key] for r in results)
    tied = [r for r in results if r[key] >= top - TIE_IC]
    return min(tied, key=lambda r: (r[simpler], -r[key]))
