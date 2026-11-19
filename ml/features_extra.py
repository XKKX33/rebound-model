"""任务 1.4：新构造的 47 个特征（7 组 39 个 + 8 个截面排名）。

设计原则（详见 期末项目/任务清单_数据与模型.md 的 1.4 和 ml/PREREG.md）：
1. 只补原 28 个特征完全没有的信息维度；
2. 每个特征事先写好理由和预期方向；
3. 只用信号日 T 收盘后能拿到的数据，窗口只往回看，按这只票自己的交易日计（停牌日不补）；
4. 按组评估。

时点保证：SQL 里所有表都先按 `trade_date <= $end` 截断，窗口都是 `... PRECEDING AND CURRENT ROW`。
`ml/p1_4_build_features.py` 用“截断到某天重算、和全量结果比较”的办法验证没有用到未来数据。

单位：daily.amount 千元；moneyflow 各金额万元；daily_basic.circ_mv 万元、*_share 万股、换手率 %。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from rebound.universe import NON_MAIN_PREFIXES

TOL = 1e-6                     # 价格贴着跌停价的容差（元），和仓库一致
LOOKBACK_DAYS = 600            # 每批往前多读的自然日（约 400 个交易日，最长窗口 250 根，足够）
BENCH = "000852.SH"            # 中证 1000
_MAIN = "NOT regexp_matches(d.ts_code, '^(" + "|".join(NON_MAIN_PREFIXES) + ")')"


@dataclass(frozen=True)
class Feat:
    name: str
    group: str
    cn: str
    formula: str
    sign: str          # 预期方向：+ / - / ? / 交互
    reason: str


GROUPS = {
    "A": "换手与流动性", "B": "规模与上市年龄", "C": "资金流", "D": "波动与收益分布",
    "E": "K线结构与隔夜/日内", "F": "与市场的关系、市场状态", "G": "超跌与跌停", "H": "截面排名版本",
}

FEATS: tuple[Feat, ...] = (
    # A 换手与流动性
    Feat("turn_f", "A", "自由流通换手率", "当日 turnover_rate_f（%）", "-", "高换手代表投机情绪，往往对应后续低收益（Liu-Stambaugh-Yuan 2019 中国四因子含换手因子）"),
    Feat("turn5", "A", "5日平均换手（log）", "ln(5 日平均 turnover_rate_f)", "-", "同上，较长窗口降噪"),
    Feat("turn20", "A", "20日平均换手（log）", "ln(20 日平均 turnover_rate_f)", "-", "同上"),
    Feat("turn_abn", "A", "异常换手", "5 日均换手 / 60 日均换手", "-", "异常放量是关注度冲击，之后回吐"),
    Feat("amihud20", "A", "Amihud 非流动性（log）", "ln(20 日平均 |日收益| / 成交额(亿元))", "+", "非流动性溢价（Amihud 2002）；反弹在流动性差的票上可能更强"),
    Feat("vr", "A", "量比", "当日 volume_ratio", "?", "放量的另一种度量，作对照"),
    # B 规模与上市年龄
    Feat("ln_circ_mv", "B", "流通市值（log）", "ln(circ_mv，万元)", "-", "A 股小市值效应和壳价值；仓库方法文档指出超额可能集中在小盘股"),
    Feat("free_ratio", "B", "自由流通比例", "free_share / float_share", "?", "控盘程度：比例低说明筹码集中"),
    Feat("ln_age", "B", "上市年龄（log）", "ln(1 + 信号日 − 上市日，自然日)", "-", "次新股的炒作和回落；上市日是固定事实，无前视"),
    # C 资金流
    Feat("mf_net1", "C", "当日净流入占比", "net_mf_amount / 成交额（统一成万元）", "?", "短期可能追涨（负）也可能是承接（正），由数据决定"),
    Feat("mf_net5", "C", "5日净流入占比", "5 日净流入合计 / 5 日成交额合计", "+", "持续流入代表中期资金认可"),
    Feat("mf_net20", "C", "20日净流入占比", "20 日净流入合计 / 20 日成交额合计", "+", "同上"),
    Feat("big_net1", "C", "当日大单净买入占比", "(大单+特大单买 − 卖) / 成交额", "+", "大资金在下跌中吸筹，是“反弹”的直接证据"),
    Feat("big_net5", "C", "5日大单净买入占比", "5 日合计 / 5 日成交额合计", "+", "同上"),
    # D 波动与收益分布
    Feat("rv20", "D", "20日波动率", "20 日复权日收益的标准差", "-", "低波动异象；原特征里没有真正的波动率（vol5/vol20 是成交额比）"),
    Feat("skew20", "D", "20日偏度", "20 日日收益的样本偏度", "-", "彩票偏好：正偏的股票被高估"),
    Feat("kurt20", "D", "20日峰度", "20 日日收益的超额峰度", "?", "尾部风险的度量"),
    Feat("max_ret20", "D", "20日最大日收益", "20 日内最大的日收益", "-", "MAX 效应（Bali-Cakici-Whitelaw 2011），散户彩票偏好"),
    Feat("min_ret20", "D", "20日最小日收益", "20 日内最小的日收益", "-", "越深的单日暴跌越容易反弹（反弹模型的核心假设之一）"),
    Feat("down_vol_ratio", "D", "下行波动占比", "sqrt(mean(min(r,0)²)) / sqrt(mean(r²))，20 日", "+", "跌得集中而不是持续阴跌，更像情绪超卖"),
    # E K线结构与隔夜/日内
    Feat("clv1", "E", "当日收盘位置", "(2C − H − L) / (H − L)，一字板记 0", "+", "收在高位说明尾盘有承接"),
    Feat("clv5", "E", "5日平均收盘位置", "clv1 的 5 日均值", "+", "同上"),
    Feat("low_shadow5", "E", "5日平均下影线", "5 日平均 (min(O,C) − L) / C", "+", "下影线长代表探底回升，经典反弹信号"),
    Feat("up_shadow5", "E", "5日平均上影线", "5 日平均 (H − max(O,C)) / C", "-", "上方抛压"),
    Feat("ovn5", "E", "5日隔夜收益", "5 日 ln(开盘 / 前收) 之和（复权）", "-", "隔夜和日内收益预测性不同（Lou-Polk-Skouras 2019）；A 股 T+1 下开盘价有系统性偏差（Qiao & Dam）"),
    Feat("ovn20", "E", "20日隔夜收益", "20 日 ln(开盘 / 前收) 之和", "-", "同上"),
    Feat("intra5", "E", "5日日内收益", "5 日 ln(收盘 / 开盘) 之和", "+", "日内买盘更多代表知情交易"),
    Feat("intra20", "E", "20日日内收益", "20 日 ln(收盘 / 开盘) 之和", "+", "同上"),
    # F 与市场的关系、市场状态
    Feat("beta60", "F", "60日 beta", "60 日日收益对中证 1000 的回归斜率", "?", "仓库发现模型 β < 1，检验低 beta 是不是收益来源"),
    Feat("ivol60", "F", "60日特质波动率", "同一回归的残差标准差", "-", "特质波动率之谜（Ang 等 2006）"),
    Feat("mkt_ret20", "F", "市场20日收益", "中证 1000 的 20 日收益", "交互", "同一天所有股票相同，只能通过交互起作用（见 1.4 说明）"),
    Feat("mkt_vol20", "F", "市场20日波动率", "中证 1000 日收益的 20 日标准差", "交互", "同上"),
    Feat("breadth5", "F", "5日上涨家数占比", "全主板当日上涨家数占比的 5 日均值", "交互", "同上"),
    Feat("limit_down_pct", "F", "当日跌停家数占比", "全主板收盘跌停家数 / 主板家数", "交互", "恐慌程度，同上"),
    # G 超跌与跌停
    Feat("ld20", "G", "20日跌停次数", "近 20 日收盘贴着跌停价的天数", "?", "情绪性跌停后可能反弹，事件性跌停会继续跌，由数据决定"),
    Feat("dist_dn", "G", "距跌停价", "(收盘 − 当日跌停价) / 收盘", "+", "离跌停越远，T+5 卖出被锁死的风险越小"),
    Feat("down_streak", "G", "连续下跌天数", "截至当天连续收跌的交易日数", "+", "连跌后的均值回归"),
    Feat("from_low250", "G", "距250日低点", "收盘 / 250 日最低收盘 − 1（复权）", "-", "接近一年低点，超跌更深"),
    Feat("dd250", "G", "250日回撤", "收盘 / 250 日最高收盘 − 1（复权）", "-", "回撤越深越可能反弹，与原 dd20 互补，覆盖长周期"),
)

#: H 组：这些特征再加一个当天准入池内的截面百分位（标签是截面百分位，排名特征和它对齐，也不受极端值影响）
RANK_OF: tuple[str, ...] = ("turn20", "amihud20", "ln_circ_mv", "mf_net5", "big_net5", "max_ret20", "ivol60", "ovn20")
RANK_FEATS: tuple[Feat, ...] = tuple(
    Feat(f"rk_{n}", "H", f"{next(f.cn for f in FEATS if f.name == n)}（截面排名）", f"{n} 在当天准入池内的百分位",
         next(f.sign for f in FEATS if f.name == n), "去掉整体漂移、与百分位标签对齐")
    for n in RANK_OF)

BASE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATS)
ALL_FEATS: tuple[Feat, ...] = FEATS + RANK_FEATS
EXTRA_NAMES: tuple[str, ...] = tuple(f.name for f in ALL_FEATS)
MARKET_STATE: tuple[str, ...] = ("mkt_ret20", "mkt_vol20", "breadth5", "limit_down_pct")


def _need(expr: str, col: str, win: str, n: int) -> str:
    """窗口里 `col` 至少 n 个非空值才算，否则为空。"""
    return f"CASE WHEN count({col}) OVER {win} >= {n} THEN {expr} END"


SQL = f"""
WITH base AS (
  SELECT d.ts_code, d.trade_date, d.open, d.high, d.low, d.close, d.pct_chg, d.amount,
         d.close * a.adj_factor AS c, d.open * a.adj_factor AS o,
         l.down_limit,
         b.turnover_rate_f AS tr, b.volume_ratio, b.circ_mv, b.free_share, b.float_share,
         m.net_mf_amount AS mf,
         m.buy_lg_amount + m.buy_elg_amount - m.sell_lg_amount - m.sell_elg_amount AS big,
         CASE WHEN m.ts_code IS NOT NULL THEN d.amount / 10.0 END AS amt_wan,      -- 千元 → 万元（与资金流同单位）
         sb.list_date
  FROM daily d
  JOIN adj_factor a USING (ts_code, trade_date)
  LEFT JOIN daily_limit l USING (ts_code, trade_date)
  LEFT JOIN daily_basic b USING (ts_code, trade_date)
  LEFT JOIN moneyflow m USING (ts_code, trade_date)
  LEFT JOIN stock_basic sb ON sb.ts_code = d.ts_code
  WHERE d.trade_date BETWEEN $src_start AND $end AND {_MAIN}
),
r AS (
  SELECT *,
    c / lag(c) OVER s - 1 AS ret,
    ln(o / lag(c) OVER s) AS ovn,
    ln(c / o) AS intra,
    CASE WHEN high > low THEN (2 * close - high - low) / (high - low) ELSE 0 END AS clv,
    (least(open, close) - low) / close AS lshadow,
    (high - greatest(open, close)) / close AS ushadow,
    CASE WHEN down_limit IS NOT NULL AND abs(close - down_limit) <= {TOL} THEN 1.0 ELSE 0.0 END AS at_dn
  FROM base WINDOW s AS (PARTITION BY ts_code ORDER BY trade_date)
),
mkt_day AS (      -- 全主板（不只是准入池）的当日统计
  SELECT trade_date, avg(CASE WHEN pct_chg > 0 THEN 1.0 ELSE 0.0 END) AS up_ratio, avg(at_dn) AS ld_ratio
  FROM r GROUP BY trade_date
),
idx AS (
  SELECT trade_date, close / lag(close) OVER (ORDER BY trade_date) - 1 AS ri,
         close / lag(close, 20) OVER (ORDER BY trade_date) - 1 AS mkt_ret20
  FROM index_daily WHERE ts_code = '{BENCH}' AND trade_date <= $end
),
mkt AS (
  SELECT k.trade_date, i.ri, i.mkt_ret20,
    stddev_samp(i.ri) OVER (ORDER BY k.trade_date ROWS 19 PRECEDING) AS mkt_vol20,
    avg(k.up_ratio) OVER (ORDER BY k.trade_date ROWS 4 PRECEDING) AS breadth5,
    k.ld_ratio AS limit_down_pct
  FROM mkt_day k LEFT JOIN idx i USING (trade_date)
),
j AS (
  SELECT r.*, mkt.ri, mkt.mkt_ret20, mkt.mkt_vol20, mkt.breadth5, mkt.limit_down_pct,
    CASE WHEN ret < 0 THEN 0 ELSE 1 END AS reset
  FROM r JOIN mkt USING (trade_date)
),
g AS (
  SELECT *, sum(reset) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS UNBOUNDED PRECEDING) AS grp FROM j
),
f AS (
  SELECT ts_code, trade_date,
    -- A 换手与流动性
    tr AS turn_f,
    {_need("ln(avg(tr) OVER w5 + 1e-6)", "tr", "w5", 4)} AS turn5,
    {_need("ln(avg(tr) OVER w20 + 1e-6)", "tr", "w20", 16)} AS turn20,
    {_need("avg(tr) OVER w5 / nullif(avg(tr) OVER w60, 0)", "tr", "w60", 48)} AS turn_abn,
    {_need("ln(avg(abs(ret) / (amount / 1e5)) OVER w20 + 1e-8)", "ret", "w20", 16)} AS amihud20,
    volume_ratio AS vr,
    -- B 规模与上市年龄
    ln(circ_mv) AS ln_circ_mv,
    free_share / nullif(float_share, 0) AS free_ratio,
    ln(1 + date_diff('day', list_date, trade_date)) AS ln_age,
    -- C 资金流（只在有资金流数据的行上求和）
    mf / nullif(amt_wan, 0) AS mf_net1,
    {_need("sum(mf) OVER w5 / nullif(sum(amt_wan) OVER w5, 0)", "mf", "w5", 4)} AS mf_net5,
    {_need("sum(mf) OVER w20 / nullif(sum(amt_wan) OVER w20, 0)", "mf", "w20", 16)} AS mf_net20,
    big / nullif(amt_wan, 0) AS big_net1,
    {_need("sum(big) OVER w5 / nullif(sum(amt_wan) OVER w5, 0)", "big", "w5", 4)} AS big_net5,
    -- D 波动与收益分布（要求 20 个日收益都在）
    {_need("stddev_samp(ret) OVER w20", "ret", "w20", 20)} AS rv20,
    {_need("skewness(ret) OVER w20", "ret", "w20", 20)} AS skew20,
    {_need("kurtosis(ret) OVER w20", "ret", "w20", 20)} AS kurt20,
    {_need("max(ret) OVER w20", "ret", "w20", 20)} AS max_ret20,
    {_need("min(ret) OVER w20", "ret", "w20", 20)} AS min_ret20,
    {_need("sqrt(avg(least(ret, 0) ^ 2) OVER w20) / nullif(sqrt(avg(ret ^ 2) OVER w20), 0)", "ret", "w20", 20)} AS down_vol_ratio,
    -- E K线结构与隔夜/日内
    clv AS clv1,
    {_need("avg(clv) OVER w5", "clv", "w5", 5)} AS clv5,
    {_need("avg(lshadow) OVER w5", "lshadow", "w5", 5)} AS low_shadow5,
    {_need("avg(ushadow) OVER w5", "ushadow", "w5", 5)} AS up_shadow5,
    {_need("sum(ovn) OVER w5", "ovn", "w5", 5)} AS ovn5,
    {_need("sum(ovn) OVER w20", "ovn", "w20", 20)} AS ovn20,
    {_need("sum(intra) OVER w5", "intra", "w5", 5)} AS intra5,
    {_need("sum(intra) OVER w20", "intra", "w20", 20)} AS intra20,
    -- F 与市场的关系、市场状态
    CASE WHEN regr_count(ret, ri) OVER w60 >= 50 THEN regr_slope(ret, ri) OVER w60 END AS beta60,
    CASE WHEN regr_count(ret, ri) OVER w60 >= 50 THEN
      sqrt(greatest(regr_syy(ret, ri) OVER w60 - (regr_sxy(ret, ri) OVER w60) ^ 2 / nullif(regr_sxx(ret, ri) OVER w60, 0), 0)
           / (regr_count(ret, ri) OVER w60 - 2)) END AS ivol60,
    mkt_ret20, mkt_vol20, breadth5, limit_down_pct,
    -- G 超跌与跌停
    CASE WHEN count(*) OVER w20 >= 20 THEN sum(at_dn) OVER w20 END AS ld20,
    (close - down_limit) / close AS dist_dn,
    CASE WHEN ret < 0 THEN row_number() OVER (PARTITION BY ts_code, grp ORDER BY trade_date)
                           - CASE WHEN grp = 0 THEN 0 ELSE 1 END ELSE 0 END AS down_streak,
    CASE WHEN count(*) OVER w250 >= 250 THEN c / min(c) OVER w250 - 1 END AS from_low250,
    CASE WHEN count(*) OVER w250 >= 250 THEN c / max(c) OVER w250 - 1 END AS dd250
  FROM g
  WINDOW w5 AS (PARTITION BY ts_code ORDER BY trade_date ROWS 4 PRECEDING),
         w20 AS (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING),
         w60 AS (PARTITION BY ts_code ORDER BY trade_date ROWS 59 PRECEDING),
         w250 AS (PARTITION BY ts_code ORDER BY trade_date ROWS 249 PRECEDING)
)
SELECT * FROM f WHERE trade_date BETWEEN $start AND $end ORDER BY ts_code, trade_date
"""


def compute_range(con: duckdb.DuckDBPyConnection, start: date, end: date) -> pd.DataFrame:
    """[start, end] 内主板每只股票每个交易日的 39 个基础新特征（float32）。只读 `end` 及以前的数据。"""
    params = {"start": start, "end": end, "src_start": start - timedelta(days=LOOKBACK_DAYS)}
    df = con.execute(SQL, params).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for col in BASE_NAMES:
        v = df[col].to_numpy(dtype=np.float64)
        v[~np.isfinite(v)] = np.nan
        df[col] = v.astype(np.float32)
    return df


def add_ranks(frame: pd.DataFrame, day: np.ndarray) -> pd.DataFrame:
    """H 组：当天（同一个 `day` 值）内的百分位，空值不参与。`frame` 只应包含准入池的行。"""
    for n in RANK_OF:
        frame[f"rk_{n}"] = frame[n].groupby(day).rank(pct=True).astype(np.float32)
    return frame


def feature_table() -> pd.DataFrame:
    """47 个新特征的说明表（面板展示、智能体解释、报告都用它）。"""
    return pd.DataFrame([{"特征": f.name, "组": f"{f.group} {GROUPS[f.group]}", "中文名": f.cn, "计算": f.formula,
                          "预期方向": f.sign, "理由": f.reason} for f in ALL_FEATS])
