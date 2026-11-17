"""候选池构建：硬性准入过滤 + 行业热度 + 候选种子排序。

这是"哪些票有资格被评分"的唯一入口，也是**公式的一部分**：`rank` 归一化是在
候选池内做的，所以池子的成员与顺序直接改变每一只票的分数。改动本模块等于改动
打分口径，不是"选数据"。

涨停不剔除是有意设计：`exclude_limit_up` 默认 false，依据是
"涨停不是扣分项，本质是市场确认"。真正的降级理由是涨停后承接失败/结构走坏/
板块退潮/资金派发，由 `money_overlay` 与结构因子处理，不用硬门槛一刀切。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from rebound.config import Universe

#: 非主板前缀（创业板/科创板/北交所式）
NON_MAIN_PREFIXES: tuple[str, ...] = ("300", "301", "688", "689", "8", "4", "9")
#: 名称里出现就不进准入池：ST、*ST，以及退市整理期的"退市XX""XX退"（没有涨跌幅限制，第一天常常腰斩）。
#: 2026-10-31 核对：namechange 里带"退"的名称 155 个，变更原因全是"退市整理期"，没有别的意思的"退"。
#: 反弹模型在 SQL 里用同一个式子判当时的名称（这个式子进特征指纹：改了就要重训）。
ST_PATTERN = r"ST|\*|退"

#: 行业进入热度排名所需的最少成分股（成员太少的行业热度不可信）
MIN_INDUSTRY_MEMBERS = 3
#: 参与"热门行业内领涨"这一路的行业数
TOP_INDUSTRIES = 18
#: 三段并集各自的取数上限
HOT_INDUSTRY_HEAD = 220
LEADER_HEAD = 180
AMOUNT_HEAD = 160


def board_code(ts_code: object) -> str:
    """从 `ts_code` 取交易所内的数字代码（`600000.SH` -> `600000`）。

    为什么不取快照的 `symbol` 列：那列来自 `LEFT JOIN stock_basic`，某只票没有
    `stock_basic` 行时是 NaN；而 `ts_code` 是 join 的键，恒定存在。
    """
    return str(ts_code).split(".", 1)[0]


def is_mainboard(ts_code: object) -> bool:
    """按 `ts_code` 判断是否主板。参数必须是完整 ts_code，不是 symbol。"""
    return not board_code(ts_code).startswith(NON_MAIN_PREFIXES)


def apply_universe(snap: pd.DataFrame, universe: Universe) -> pd.DataFrame:
    """按策略 universe 段过滤截面快照。返回新 DataFrame（不改原表）。

    一条纪律：**判据缺失一律排除，不当成通过**。截面是
    `daily LEFT JOIN stock_basic`，某票缺 `stock_basic` 行时 `name` 是 NaN。
    旧实现先 `name.fillna("")` 再判 ST，于是缺名字的票"不含 ST"从而放行；
    板块判据取 NaN 的 `symbol`，`str(nan)="nan"` 也不以非主板前缀开头，同样放行。
    结果是一只信息不全、连板块都不知道的票静默进入候选池——比少一只候选严重得多。
    所以：板块从 `ts_code` 推，ST 判据缺失即排除。
    """
    return snap[universe_mask(snap, universe)].reset_index(drop=True)


def universe_mask(frame: pd.DataFrame, universe: Universe) -> pd.Series:
    """逐行判断是否在准入池里（判据同 `apply_universe`）。

    反弹模型在几百万行的面板上用它，所以按"每个不同的名称/代码"判一次再铺回各行，
    不逐行转字符串。
    """
    keep = pd.Series(True, index=frame.index)
    if universe.exclude_st:
        keep &= _per_value(frame["name"], lambda name: not re.search(ST_PATTERN, name), missing=False)
    if universe.price_max is not None:
        keep &= frame["close"].astype(float) < float(universe.price_max)
    if universe.price_min is not None:
        # 低价票有面值退市风险（连续 20 个交易日收盘 < 1 元）：退市前连续跌停卖不出，模型学不到。
        keep &= frame["close"].astype(float) >= float(universe.price_min)
    if universe.board == "mainboard":
        keep &= _per_value(frame["ts_code"], is_mainboard, missing=True)
    if universe.min_amount_yi is not None:
        # Tushare amount 单位 = 千元；1 亿元 = 100000 千元。
        keep &= frame["amount"].astype(float) >= universe.min_amount_qian
    return keep


def _per_value(column: pd.Series, test, *, missing: bool) -> pd.Series:  # noqa: ANN001
    """对列里每个不同的值判一次，再按行铺开；空值一律给 `missing`。"""
    values = column.astype("category")
    verdict = np.array([bool(test(str(v))) for v in values.cat.categories] + [missing], dtype=bool)
    return pd.Series(verdict[values.cat.codes.to_numpy()], index=column.index)


def industry_heat(df: pd.DataFrame) -> pd.DataFrame:
    """行业热度表，按热度降序。

    `heat = avg_pct*0.25 + med_pct*0.25 + up_ratio*5 + strong_ratio*10 + log1p(total_amount)/5`

    系数是为"当日强度"服务的：涨跌幅均值与中位数各占 0.25（中位数压住个别极端票），
    上涨家数占比 ×5 与涨幅 ≥5% 家数占比 ×10 表达"普涨程度"，成交额取对数后除以 5
    避免大行业仅凭体量霸榜。每个系数都直接改变最终选股，不要"调参式"微调。
    """
    ind = (
        df.dropna(subset=["industry"])
        .groupby("industry")
        .agg(
            count=("ts_code", "count"),
            avg_pct=("pct_chg", "mean"),
            med_pct=("pct_chg", "median"),
            up_ratio=("pct_chg", lambda s: (s > 0).mean()),
            strong_ratio=("pct_chg", lambda s: (s >= 5).mean()),
            total_amount=("amount", "sum"),
        )
        .reset_index()
    )
    ind = ind[ind["count"] >= MIN_INDUSTRY_MEMBERS]
    ind["heat"] = (
        ind["avg_pct"] * 0.25
        + ind["med_pct"] * 0.25
        + ind["up_ratio"] * 5
        + ind["strong_ratio"] * 10
        + np.log1p(ind["total_amount"]) / 5
    )
    return ind.sort_values("heat", ascending=False).reset_index(drop=True)


def industry_meta(ind: pd.DataFrame) -> tuple[dict[str, float], dict[str, int], list[str]]:
    """返回 `(heat_map, rank_map, top_inds)`。

    **`rank_map` 覆盖全部入选行业，不截断到 18**——所以 `industry_rank` 可以到 75。
    这与"榜单只落库 15 条"是两件事：`rank_map` 决定因子取值，落库只决定页面能看到
    几条。曾误以为 `rank_map` 按 `head(18)` 建，实测数据（有的票 rank 到 75）证明不是。
    """
    heat_map = dict(zip(ind["industry"], ind["heat"]))
    rank_map = {name: i + 1 for i, name in enumerate(ind["industry"].tolist())}
    top_inds = ind.head(TOP_INDUSTRIES)["industry"].tolist()
    return heat_map, rank_map, top_inds


def build_candidates(df: pd.DataFrame, top_inds: list[str], limit: int) -> pd.DataFrame:
    """候选种子 = 三段并集去重后按 `seed_score` 取前 `limit`。

    三路各有分工，缺一路就会漏掉一整类机会：
    - 热门行业内领涨（前 220）：板块共振的主线票；
    - 全场领涨（前 180）：不走板块、个股独强的票；
    - 全场成交额（前 160）：大资金战场。
    `seed_score = pct_chg*2 + log1p(amount)/2`——**涨幅主导，成交额只做次级排序**。
    这个乘数决定了"谁进候选池"，也就决定了归一化分母，进而决定所有分数。
    """
    parts = [
        df[df["industry"].isin(top_inds)]
        .sort_values(["pct_chg", "amount"], ascending=False)
        .head(HOT_INDUSTRY_HEAD),
        df.sort_values(["pct_chg", "amount"], ascending=False).head(LEADER_HEAD),
        df.sort_values("amount", ascending=False).head(AMOUNT_HEAD),
    ]
    cand = pd.concat(parts).drop_duplicates("ts_code").copy()
    cand["seed_score"] = cand["pct_chg"].fillna(0) * 2 + np.log1p(cand["amount"].fillna(0)) / 2
    return cand.sort_values("seed_score", ascending=False).head(limit).reset_index(drop=True)


def amount_top15pct_threshold(frame: pd.DataFrame) -> float:
    """候选池内成交额的 85 分位（`amount_top15pct` 因子的门槛）。

    注意分母是**候选池**而不是全市场：这个因子问的是"在这批候选里成交额是否靠前"，
    而不是"在全市场里是否靠前"。
    """
    return float(frame["amount"].astype(float).quantile(0.85))
