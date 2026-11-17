"""两日反弹模型实验的公用部分：变体目录、要比的模型、退市整理期名单、选法。

脚本都由 run_all.py 起，环境变量 DECADE_DATA 指到本目录下的变体目录（nocap/ 或 cap70/）：十年实验的 common.py
在 import 时读它，面板、标签、打分、登记都落在变体目录里，不碰十年实验的目录。
"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent
DECADE = HERE.parent / "decade"
sys.path.insert(0, str(DECADE))
sys.path.insert(0, str(HERE.parent / "shared"))

from engine import Cell  # noqa: E402

#: 变体 → 十年实验里对应的目录（面板、o1o5 标签、每月重训的 CPU 版五日打分从那里硬链接过来）
VARIANTS = {"nocap": DECADE / "nocap", "cap70": DECADE}
VARIANT_NAMES = {"nocap": "不限价", "cap70": "限 70 元"}
CAPITAL = 20_000.0
FRESH_YEARS = (2017, 2019, 2021, 2023)
#: 退市整理期：当时名称带"退"（正式 rebound/universe.py 的 ST_PATTERN 比十年实验多出来的那一项）
DELIST = "退"
#: 硬链接进变体目录的文件（o1o5 标签另外找：文件名带面板指纹）
LINKED = ("panel10.parquet", "panel10.meta.json", "axes.json", "scores10_M.parquet")


@dataclass(frozen=True)
class Model:
    key: str      # 文件名用
    cell: str     # 标签的买卖时点
    device: str   # cpu = 十年实验存好的打分；cuda = 本实验训练
    label: str    # 报告里的名字

    @property
    def scores(self) -> str:
        """变体目录里的每月重训打分文件。"""
        return "scores10_M.parquet" if self.device == "cpu" else f"scores_{self.cell}_{self.device}_M.parquet"


MODELS = {m.key: m for m in (
    Model("o1o5-cpu", "o1o5", "cpu", "五日（现行）"),
    Model("o1o2-cuda", "o1o2", "cuda", "两日"),
    Model("o1o5-cuda", "o1o5", "cuda", "五日（GPU 重训）"),
    Model("o1o3-cuda", "o1o3", "cuda", "三日"),
)}


def cell_of(name: str) -> Cell:
    """o1o2 → T+1 开盘买、T+2 开盘卖。"""
    legs = {"o": "open", "c": "close"}
    if len(name) != 4 or name[1] != "1" or name[0] not in legs or name[2] not in legs or not name[3].isdigit():
        raise ValueError(f"不认识的买卖时点：{name}")
    return Cell(legs[name[0]], int(name[3]), legs[name[2]])


def data_dir() -> Path:
    """本次运行的变体目录（DECADE_DATA）。必须在本目录下：防止误写十年实验的目录。"""
    raw = os.environ.get("DECADE_DATA")
    if not raw:
        raise SystemExit("先设 DECADE_DATA 指到本目录下的变体目录（run_all.py 会设好）")
    d = Path(raw).resolve()
    if HERE not in d.parents or d.name not in VARIANTS:
        raise SystemExit(f"DECADE_DATA={d} 不是 {HERE} 下的变体目录 {sorted(VARIANTS)}")
    return d


def link_inputs(variant: str, root: Path = HERE) -> Path:
    """变体目录里放十年实验的面板、坐标轴、o1o5 标签和 CPU 版五日打分（硬链接，同一个盘不占地方；链不上就复制）。"""
    src, dst = VARIANTS[variant], root / variant
    labels = sorted(src.glob("labels10_o1o5_*.parquet"))
    if len(labels) != 1:
        raise SystemExit(f"{src} 应该正好有一份 o1o5 标签，现在是 {[p.name for p in labels]}")
    dst.mkdir(parents=True, exist_ok=True)
    for f in [src / name for name in LINKED] + labels:
        target = dst / f.name
        if target.exists():
            continue
        try:
            os.link(f, target)
        except OSError:
            shutil.copy2(f, target)
    return dst


def delisting_rows(panel: Path, db: Path) -> pd.DataFrame:
    """准入池里当时名称带"退"的股票日（ts_code, trade_date）。当时名称和十年实验 panel10 同一算法：
    namechange 里生效日不晚于当天的最后一个名字（同一生效日取公告最晚的），没有记录才用现在的名称。"""
    con = duckdb.connect()
    try:
        con.execute("SET memory_limit='1GB'; SET threads=4")
        con.execute(f"ATTACH '{db.as_posix()}' AS db (READ_ONLY)")
        return con.execute(f"""
            WITH names AS (
                SELECT ts_code, name, CAST(start_date AS TIMESTAMP) AS start_ts FROM (
                    SELECT *, row_number() OVER (PARTITION BY ts_code, start_date ORDER BY ann_date DESC NULLS LAST, name) AS k
                    FROM db.namechange) WHERE k = 1
            ), r AS (
                SELECT CAST(ts_code AS VARCHAR) AS ts_code, CAST(trade_date AS TIMESTAMP) AS trade_date
                FROM read_parquet('{panel.as_posix()}') WHERE eligible
            )
            SELECT r.ts_code, r.trade_date FROM r
            ASOF LEFT JOIN names n ON r.ts_code = n.ts_code AND r.trade_date >= n.start_ts
            LEFT JOIN db.stock_basic b ON r.ts_code = b.ts_code
            WHERE coalesce(regexp_matches(coalesce(n.name, b.name), '{DELIST}'), false)
            ORDER BY 1, 2""").df()
    finally:
        con.close()


def fresh_stats(starts: dict[str, dict]) -> dict:
    """四个起点的年化：最差的那个、是不是都赚。"""
    worst = min(v["cagr"] for v in starts.values())
    return {"worst_cagr": worst, "all_positive": worst > 0}


def steadiest(rows: list[tuple[str, dict]]) -> str | None:
    """四个起点都赚、最差起点年化最高的规则（rows 按网格顺序；并列取靠前的，和正式 rebound capital 的 max 一样）。"""
    robust = [(name, r["worst_cagr"]) for name, r in rows if r["all_positive"]]
    if not robust:
        return None
    return max(robust, key=lambda item: item[1])[0]


def verdict(two: float | None, five: float | None, noise: float) -> str:
    """两日推荐 vs 五日推荐的最差起点年化：差距超过重训噪声才算分出高下（预注册第 2 条）。"""
    if two is None or five is None:
        return {(True, True): "两个模型都没有四个起点都赚的规则", (True, False): "五日更好（两日没有四个起点都赚的规则）",
                (False, True): "两日更好（五日没有四个起点都赚的规则）"}[(two is None, five is None)]
    if abs(two - five) <= noise:
        return "打平"
    return "两日更好" if two > five else "五日更好"
