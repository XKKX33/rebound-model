"""持股规则网格（十年实验 `rule_grid.py` 的参数）：两类规则、一组固定的参数，以及网格上的"邻居"（选优时做平滑用）。

两类都是 T+1 开盘买；2026-11-01 起都是尾盘（收盘竞价）卖、前一晚定好（满仓账户同一个开盘先卖后买实盘做不到：
买单的钱要等 9:25 卖出成交才到账，见 `data/research/execgap-20261101`）：
- FH（固定持有、错开分批）：每天买当天前 K 名，买进后第 h 个开市日尾盘卖（原来是那天开盘卖）。
  隔夜最多 K·h 只；资金分 h + 1 批（卖的那天的钱第二天开盘才回到那一批手上）。
- DB（每日调仓带缓冲）：最多拿 K 只；前一晚排名掉出 b·K 的，第二天尾盘卖；空出来的位置再下一个开盘按排名补。
共同：每个行业最多 2 只；买不进（涨停、停牌、钱不够一手）就顺延到下一名。共 60 + 90 = 150 条。

老名字（`-o1o5` 这类，开盘卖）仍然认：跟踪文件里存着的照原来的时点回放（`rule_of`）；本金测算只算新的时点。
"""

from __future__ import annotations

from dataclasses import dataclass

from rebound.labels import Cell
from rebound.backtest.sim import Rule

FH_K = (1, 2, 3, 4, 6, 8, 10, 15, 20)
FH_H = (1, 2, 3, 4, 6, 9, 14, 19)
DB_K = (1, 2, 3, 4, 6, 8, 10, 15, 20, 30)
DB_B = (1, 1.5, 2, 3, 4, 6, 8, 12, 16)
#: FH 隔夜同时持有 K·h 只的上限（尾盘卖时，白天开盘买进到尾盘卖出之间会多拿一批：K·(h+1) 只）。
#: 再多每只分到的钱太少，一手买不起、最低佣金吃掉收益，不值得算。
MAX_BOOK = 100


@dataclass(frozen=True)
class Spec:
    family: str   # "FH" 或 "DB"
    k: int
    p: float      # FH：持有天数 h；DB：缓冲倍数 b

    @property
    def rule(self) -> Rule:
        """现在的时点：开盘买、尾盘卖。FH 卖在原来那个卖出日（T+h+1）的尾盘；DB 的 5 只是名字（DB 没有固定卖出日）。"""
        return self._rule("close")

    @property
    def legacy_rule(self) -> Rule:
        """2026-11-01 之前的时点：开盘买、开盘卖（只给跟踪文件里的老名字回放用）。"""
        return self._rule("open")

    def _rule(self, exit_leg: str) -> Rule:
        if self.family == "FH":
            return Rule("FH", self.k, Cell("open", int(self.p) + 1, exit_leg))
        return Rule("DB", self.k, Cell("open", 5, exit_leg), buffer=float(self.p))

    @property
    def name(self) -> str:
        return self.rule.name

    @property
    def legacy(self) -> str:
        return self.legacy_rule.name

    @property
    def short(self) -> str:
        """FH-K6-next-o1c5 → FH-K6-h4，DB-K10-b4-o1c5 → DB-K10-b4。"""
        return f"FH-K{self.k}-h{self.p:g}" if self.family == "FH" else f"DB-K{self.k}-b{self.p:g}"

    @property
    def plain(self) -> str:
        if self.family == "FH":
            return f"每天买前 {self.k} 名，每只拿 {self.p:g} 天（隔夜最多 {self.k * int(self.p)} 只）"
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


def parse(name: str, *, legacy_ok: bool = False) -> Spec:
    """规则名 → Spec。认短名（FH-K6-h4 / DB-K10-b4）和全名（FH-K6-next-o1c5 / DB-K10-b4-o1c5）；
    `legacy_ok=True` 时也认 2026-11-01 之前的全名（`-o1o5` 这类，开盘卖），只给读老跟踪文件用。

    Spec 不带买卖时点：`.rule` 永远是现在的时点（尾盘卖）。所以老名字默认不认，免得拿老名字取 `.rule`
    悄悄按尾盘卖算；要照名字本身的时点算用 `rule_of`。
    形状不对的名字（段数不够、不是文字）一律是 ValueError：名字可能来自跟踪文件，读坏了也得说清是名字的问题。
    全名的后缀是买卖时点，对不上的不认（比如 -c1c5）：不然会悄悄按另一个时点算。
    """
    try:
        parts = name.split("-")
        family, k = parts[0], int(parts[1][1:])
        if family == "FH":
            h = int(parts[3][3:]) - 1 if len(parts) == 4 else int(parts[2][1:])
            spec = Spec("FH", k, h)
        elif family == "DB":
            b = float(parts[2][1:])
            spec = Spec("DB", k, int(b) if b.is_integer() else b)
        else:
            raise ValueError(f"不认识的规则名：{name}")
    except (IndexError, AttributeError, OverflowError) as exc:
        raise ValueError(f"不认识的规则名：{name!r}") from exc
    if spec not in set(specs()):
        raise ValueError(f"{name} 不在 150 条规则的网格里")
    if name == spec.legacy and not legacy_ok:
        raise ValueError(f"{name} 是 2026-11-01 之前的开盘卖时点，已经不用了：改成 {spec.name}（开盘买、尾盘卖）")
    if name not in (spec.short, spec.name, spec.legacy):
        raise ValueError(f"不认识的规则名：{name!r}（可以写 {spec.short} 或 {spec.name}）")
    return spec


def rule_of(name: str) -> Rule:
    """规则名 → 模拟器用的规则：老名字（开盘卖）照原来的时点，其余是现在的时点（开盘买、尾盘卖）。"""
    spec = parse(name, legacy_ok=True)
    return spec.legacy_rule if name == spec.legacy else spec.rule
