"""2026 年（截至 10-19）各规则的收益与超额分布：十年实验的 150 条规则。

回答"2026 年是不是全都不好"：看规则收益中位数、超额中位数、跑赢准入池的比例，
并和 2023~2025 对比；再按月看 2026 年的超额是在哪几个月丢的。
用法：python -X utf8 year2026_check.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
YEARS = (2023, 2024, 2025, 2026)
VARIANTS = (("限价", ROOT), ("不限价", ROOT / "nocap"))
CAPITALS = ("200000", "20000")
CADENCES = ("Y", "H", "Q", "M")
PCT = 100.0


def load(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, pd.DatetimeIndex]:
    z = np.load(path, allow_pickle=True)
    return list(z["names"]), np.asarray(z["ret"], dtype=float), np.asarray(z["pool"], dtype=float), pd.DatetimeIndex(z["dates"])


def compound(x: np.ndarray) -> np.ndarray | float:
    return np.prod(1.0 + x, axis=-1) - 1.0


def pct(v: float) -> str:
    return "—" if not np.isfinite(v) else f"{v * PCT:+.1f}%"


def main() -> None:
    for vname, vroot in VARIANTS:
        for cap in CAPITALS:
            for cad in CADENCES:
                path = vroot / "rules" / cap / f"real_{cad}.npz"
                if not path.exists():
                    continue
                names, ret, pool, dates = load(path)
                print(f"\n## {vname} {cap} {cad}（{len(names)} 条规则，{dates[0].date()} ~ {dates[-1].date()}）")
                print("| 年份 | 准入池 | 规则中位数 | 超额中位数 | 超额>0 | 收益>0 |")
                print("|---|---|---|---|---|---|")
                for y in YEARS:
                    mask = dates.year == y
                    if not mask.any():
                        continue
                    r = compound(ret[:, mask])
                    p = compound(pool[mask])
                    ex = r - p
                    print(f"| {y} | {pct(p)} | {pct(float(np.median(r)))} | "
                          f"{pct(float(np.median(ex)))} | {np.mean(ex > 0):.0%} | {np.mean(r > 0):.0%} |")
                print()
                print("| 2026 月份 | 准入池 | 规则中位数 | 超额中位数 | 超额>0 |")
                print("|---|---|---|---|---|")
                for m in range(1, 13):
                    mask = (dates.year == 2026) & (dates.month == m)
                    if not mask.any():
                        continue
                    r = compound(ret[:, mask])
                    p = compound(pool[mask])
                    ex = r - p
                    print(f"| 2026-{m:02d} | {pct(p)} | {pct(float(np.median(r)))} | "
                          f"{pct(float(np.median(ex)))} | {np.mean(ex > 0):.0%} |")
                mask = dates.year == 2026
                r26 = compound(ret[:, mask])
                p26 = compound(pool[mask])
                ex26 = r26 - p26
                order = np.argsort(ex26)
                print("\n2026 超额最差 5 条：", "; ".join(
                    f"{names[i]} {pct(ex26[i])}" for i in order[:5]))
                print("2026 超额最好 5 条：", "; ".join(
                    f"{names[i]} {pct(ex26[i])}" for i in order[-5:]))
                for want in ("DB-K30-b4-o1o5", "DB-K30-b8-o1o5", "DB-K3-b16-o1o5", "DB-K10-b12-o1o5"):
                    if want in names:
                        i = names.index(want)
                        print(f"  {want}: 2026 {pct(r26[i])}，超额 {pct(ex26[i])}")


if __name__ == "__main__":
    main()
