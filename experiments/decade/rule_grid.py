"""持股规则网格：两类规则、一组固定的参数，以及网格上的"邻居"（选优时做平滑用）。

两类都是 T+1 开盘买、开盘卖（和 o1o5 打分同一个时点）：
- FH（固定持有、错开分批）：每天买当天前 K 名，每只拿 h 个交易日，到期开盘卖。同时最多 K·h 只。
- DB（每日调仓带缓冲）：最多拿 K 只；每天开盘把排名掉出 b·K 的卖掉，空位按排名补。
共同：每个行业最多 2 只；买不进（涨停、停牌、钱不够一手）就顺延到下一名。
"""
from __future__ import annotations

from dataclasses import dataclass

from common import DATA
from engine import Cell
from sim import Rule

CADENCES = ("Y", "H", "Q", "M")
CADENCE_NAMES = {"Y": "每年", "H": "每半年", "Q": "每季度", "M": "每月"}
FH_K = (1, 2, 3, 4, 6, 8, 10, 15, 20)
FH_H = (1, 2, 3, 4, 6, 9, 14, 19)
DB_K = (1, 2, 3, 4, 6, 8, 10, 15, 20, 30)
DB_B = (1, 1.5, 2, 3, 4, 6, 8, 12, 16)
#: FH 同时持有 K·h 只的上限。再多每只分到的钱太少，一手买不起、最低佣金吃掉收益，不值得算。
MAX_BOOK = 100
REFERENCE = ("Y", "FH-K6-next-o1o5")  # 原方案，作参照


@dataclass(frozen=True)
class Spec:
    family: str   # "FH" 或 "DB"
    k: int
    p: float      # FH：持有天数 h；DB：缓冲倍数 b

    @property
    def rule(self) -> Rule:
        if self.family == "FH":
            return Rule("FH", self.k, Cell("open", int(self.p) + 1, "open"))
        return Rule("DB", self.k, Cell("open", 5, "open"), buffer=float(self.p))

    @property
    def name(self) -> str:
        return self.rule.name

    @property
    def plain(self) -> str:
        if self.family == "FH":
            return f"每天买前 {self.k} 名，每只拿 {self.p:g} 天（最多 {self.k * int(self.p)} 只）"
        return f"最多 {self.k} 只，排名掉出前 {self.p * self.k:g} 名才卖"


def specs() -> list[Spec]:
    fh = [Spec("FH", k, h) for k in FH_K for h in FH_H if k * h <= MAX_BOOK]
    db = [Spec("DB", k, b) for k in DB_K for b in DB_B]
    return fh + db


def axes_of(family: str) -> tuple[tuple, tuple]:
    return (FH_K, FH_H) if family == "FH" else (DB_K, DB_B)


def neighbors(spec: Spec, pool: set[Spec]) -> list[Spec]:
    """网格上相邻的点（K 和 h/b 各差一格以内，含自己），只取 pool 里有的。"""
    ks, ps = axes_of(spec.family)
    i, j = ks.index(spec.k), ps.index(spec.p)
    near = [Spec(spec.family, ks[a], ps[b]) for a in range(max(i - 1, 0), min(i + 2, len(ks)))
            for b in range(max(j - 1, 0), min(j + 2, len(ps)))]
    return [s for s in near if s in pool]


def parse(name: str) -> Spec:
    """规则名 → Spec（FH-K6-next-o1o5 / DB-K10-b4-o1o5）。"""
    family, k, mid, cell = name.split("-")
    if family == "FH":
        return Spec("FH", int(k[1:]), int(cell[3:]) - 1)
    b = float(mid[1:])
    return Spec("DB", int(k[1:]), int(b) if b.is_integer() else b)


def out_dir(capital: float):
    return DATA / "rules" / f"{int(capital)}"
