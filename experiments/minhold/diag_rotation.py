"""诊断（N = 1）：四条主规则早卖的票"接着拿"能赚多少（gap），同一个开盘换上的新票赚多少（repl），差多少（rotation）。

先核对：这里重跑的 N = 1 逐日收益必须和 run_minhold 存的一样。输出 <变体>/rotation.json。
用法（DECADE_DATA 指到 nocap 或 cap70）：python -X utf8 diag_rotation.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from run_minhold import DATA, fee_model, load_done, out_paths, parse, picks, prepare
from minhold import LABEL_HOLD, PRIMARY, month_ci, rest_of_window, simulate_minhold
from rotation import rotation


def block_mean(values: np.ndarray, when: pd.DatetimeIndex) -> dict:
    lo, hi = month_ci(pd.Series(values, index=when))
    return {"mean": round(float(values.mean()) * 100, 3), "ci": [round(lo * 100, 3), round(hi * 100, 3)]}


def main() -> None:
    start = time.time()
    s = prepare(delist=True)
    cands, rank = picks(s)
    out = {}
    for capital, rules in PRIMARY.items():
        _, stored = load_done(*out_paths(capital, "min5", True))
        for name in rules:
            res = simulate_minhold(s.m, cands, rank, parse(name).rule, fee_model(capital, "min5"), s.d_first, s.d_last)
            ret = res.nav[s.idx] / res.nav[s.idx - 1] - 1
            if f"1|{name}" in stored and not np.array_equal(ret, stored[f"1|{name}"]):
                raise SystemExit(f"{capital} {name}：重跑的 N = 1 和 run_minhold 存的不一样")
            r = rotation(s.m, res.trades, rest_of_window(s.m, res.trades, LABEL_HOLD))
            r = r[np.isfinite(r.repl)]
            when = s.m.dates[r.exit_day.to_numpy()]
            out[f"{capital}|{name}"] = {
                "trades": int(len(res.trades)), "early_with_new_buy": int(len(r)),
                "gap": block_mean(r.gap.to_numpy(), when), "repl": block_mean(r.repl.to_numpy(), when),
                "rotation": block_mean(r.rotation.to_numpy(), when),
                "rotation_positive": round(float((r.rotation > 0).mean()) * 100, 1)}
            print(capital, name, out[f"{capital}|{name}"], f"({time.time() - start:.0f}s)", flush=True)
    (DATA / "rotation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {DATA.name}/rotation.json", flush=True)


if __name__ == "__main__":
    main()
