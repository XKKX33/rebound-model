"""正式代码 vs 十年实验（experiments/decade/）逐位核对。两边的库和文件都只读。

1. 正式库 vs 十年合成库：主板 daily / adj_factor / daily_limit 在十年库的日期范围内逐年比哈希；stock_basic 的名称。
2. 用正式代码（rebound.model.train.load_rows）读十年合成库，和十年实验的面板比：两个版本的准入池、28 个特征、
   o1o5 卖出日、标签（研究口径：当天整个准入池里的百分位）。
3. 模型：几个重训日，正式代码的训练集和分块（研究口径的标签）训种子 0，预测和十年实验缓存的 fits/*.npy 比。
   对不上时，再把十年实验的特征和标签原样喂给正式的训练代码，分清是数据的差别还是训练代码的差别。
4. 修掉标签泄漏的影响：同一个重训日，正式口径和研究口径的标签差多少、两个模型的打分有多像。
用法：python -X utf8 -B decade_check.py   输出 decade_check.json、decade_check.log
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from rebound import params as model_params  # noqa: E402
from rebound.features import FEATURES  # noqa: E402
from rebound.labels import percentile  # noqa: E402
from rebound.model import train as T  # noqa: E402

DECADE_DB, FORMAL_DB = ROOT / "data/research-2013-2026.duckdb", ROOT / "data/market.duckdb"
FORK2 = ROOT / "experiments/decade"
SOURCES = {"cap70": FORK2, "nocap": FORK2 / "nocap"}
#: 十年实验 train_cadence.PARAMS 原样（键的顺序进它缓存文件名的哈希）
THEIR_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    tree_method="hist")
FIT_MONTHS = {"cap70": ["2015-01", "2019-01", "2023-01", "2026-10"], "nocap": ["2015-01", "2021-01", "2026-10"]}
FILTERS = [("eligible", "==", True), ("below2", "==", False)]
OUT, LOG = HERE / "decade_check.json", HERE / "decade_check.log"
RESULT: dict = {}
T0 = time.time()


def say(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} [{time.time() - T0:5.0f}s] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def save() -> None:
    OUT.write_text(json.dumps(RESULT, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def raw_tables() -> dict:
    con = duckdb.connect()
    con.execute("SET memory_limit='1GB'; SET threads=4")
    con.execute(f"ATTACH '{FORMAL_DB.as_posix()}' AS f (READ_ONLY)")
    con.execute(f"ATTACH '{DECADE_DB.as_posix()}' AS d (READ_ONLY)")
    last = con.execute("SELECT max(trade_date) FROM d.daily").fetchone()[0]
    out: dict = {"decade_last": str(last)}
    cols = {"daily": "open, high, low, close, pre_close, pct_chg, vol, amount", "adj_factor": "adj_factor",
            "daily_limit": "up_limit, down_limit"}
    for table, c in cols.items():
        q = (f"SELECT year(trade_date), count(*), sum(hash(ts_code, trade_date, {c})) FROM {{db}}.{table} "
             f"WHERE trade_date <= ? AND NOT regexp_matches(ts_code, '{T._NON_MAIN}') GROUP BY 1 ORDER BY 1")
        a = {r[0]: r[1:] for r in con.execute(q.format(db="f"), [last]).fetchall()}
        b = {r[0]: r[1:] for r in con.execute(q.format(db="d"), [last]).fetchall()}
        diff = [str(y) for y in sorted(set(a) | set(b)) if a.get(y) != b.get(y)]
        out[table] = {"rows_formal": sum(v[0] for v in a.values()), "rows_decade": sum(v[0] for v in b.values()),
                      "years_differ": diff}
    out["stock_basic_name_differ"] = con.execute(f"""SELECT count(*) FROM f.stock_basic f JOIN d.stock_basic d
        ON f.ts_code = d.ts_code WHERE f.name IS DISTINCT FROM d.name
        AND NOT regexp_matches(f.ts_code, '{T._NON_MAIN}')""").fetchone()[0]
    out["stock_basic_name_samples"] = con.execute(f"""SELECT f.ts_code, d.name, f.name FROM f.stock_basic f
        JOIN d.stock_basic d ON f.ts_code = d.ts_code WHERE f.name IS DISTINCT FROM d.name
        AND NOT regexp_matches(f.ts_code, '{T._NON_MAIN}') LIMIT 8""").fetchall()
    con.close()
    return out


def panel_key(src: Path) -> str:
    meta = json.loads((src / "panel10.meta.json").read_text(encoding="utf-8"))
    key = {k: meta[k] for k in ("rows", "eligible_rows", "first", "last", "seam")}
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:8]


def their_side(variant: str) -> dict:
    src = SOURCES[variant]
    axes = json.loads((src / "axes.json").read_text(encoding="utf-8"))
    codes = np.array(axes["codes"])
    dates = pd.DatetimeIndex(pd.to_datetime(axes["dates"])).astype("datetime64[ns]")
    key = panel_key(src)
    labels = pd.read_parquet(src / f"labels10_o1o5_{key}.parquet")
    t = pq.read_table(src / "panel10.parquet", columns=["ts_code", "trade_date"], filters=FILTERS,
                      use_threads=False, read_dictionary=["ts_code"]).to_pandas()
    cat = pd.Categorical(t.ts_code)
    ji = pd.Index(codes).get_indexer(cat.categories)[cat.codes]
    di = dates.get_indexer(pd.DatetimeIndex(t.trade_date).astype("datetime64[ns]"))
    del t
    if len(di) != len(labels) or (di != labels.di.to_numpy()).any() or (ji != labels.ji.to_numpy()).any():
        raise SystemExit(f"{variant}：十年实验的面板和标签行对不上")
    order = np.argsort(di, kind="stable")
    return {"src": src, "key": key, "codes": codes, "dates": dates, "order": order, "di": di[order],
            "ji": ji[order], "y": labels.y.to_numpy()[order], "exit_day": labels.exit_day.to_numpy()[order]}


def keys(dates_from: pd.DatetimeIndex, di: np.ndarray, codes: np.ndarray, ji: np.ndarray,
         axis: pd.DatetimeIndex, common: np.ndarray) -> np.ndarray:
    d = axis.get_indexer(dates_from)[di]
    c = np.searchsorted(common, codes)[ji]
    if (d < 0).any():
        raise SystemExit("日期轴对不上")
    return d.astype(np.int64) * len(common) + c


def sample(k: np.ndarray, axis: pd.DatetimeIndex, common: np.ndarray, n: int = 6) -> list:
    return [(str(axis[i // len(common)].date()), str(common[i % len(common)])) for i in k[:n]]


def day_stats(day: np.ndarray, a: np.ndarray, b: np.ndarray, top: int = 30) -> dict:
    df = pd.DataFrame({"d": day, "a": a, "b": b})
    g = df.groupby("d")
    ra, rb = g["a"].rank(), g["b"].rank()
    da, db = g["a"].rank(ascending=False, method="first"), g["b"].rank(ascending=False, method="first")
    tmp = pd.DataFrame({"d": df.d, "ra": ra, "rb": rb, "both": (da <= top) & (db <= top)})
    rho = tmp.groupby("d")[["ra", "rb"]].corr().unstack().iloc[:, 1]
    overlap = tmp.groupby("d")["both"].sum() / np.minimum(top, tmp.groupby("d").size())
    return {"days": int(len(rho)), "spearman_mean": float(rho.mean()), "spearman_min": float(rho.min()),
            f"top{top}_overlap_mean": float(overlap.mean()), f"top{top}_overlap_min": float(overlap.min())}


def compare_predictions(day: np.ndarray, mine: np.ndarray, theirs: np.ndarray) -> dict:
    out = {"rows": int(len(mine)), "exact": int((mine == theirs).sum()),
           "max_abs_diff": float(np.max(np.abs(mine.astype(np.float64) - theirs))) if len(mine) else 0.0}
    if out["exact"] != out["rows"]:
        out.update(day_stats(day, mine, theirs))
    return out


def score_end(dates: pd.DatetimeIndex, f: int) -> int:
    return min(int(dates.searchsorted(dates[f] + pd.DateOffset(years=1))), len(dates))


def train_seed0(x: np.ndarray, data: T.TrainingSet) -> xgb.Booster:
    m = xgb.QuantileDMatrix(T._Batches(x, data), nthread=model_params.THREADS)
    return xgb.train(T.booster_params(model_params.PARAMS, 0), m, num_boost_round=model_params.PARAMS["n_estimators"])


#: 量修掉标签泄漏的影响用的重训日（打分区间满一年）
LEAK_MONTH = {"cap70": "2023-01", "nocap": "2021-01"}


def check_variant(variant: str, rows: T.TrainingRows) -> None:
    res = RESULT.setdefault(variant, {})
    them = their_side(variant)
    common = np.union1d(rows.codes, them["codes"])
    pool = np.flatnonzero(rows.member[variant])
    mk = keys(rows.dates, rows.di[pool], rows.codes, rows.ji[pool], them["dates"], common)
    tk = keys(them["dates"], them["di"], them["codes"], them["ji"], them["dates"], common)
    if not ((np.diff(mk) > 0).all() and (np.diff(tk) > 0).all()):
        raise SystemExit("行序不是按日、同日按代码")
    _, im, it = np.intersect1d(mk, tk, assume_unique=True, return_indices=True)
    only_mine, only_theirs = np.setdiff1d(mk, tk, assume_unique=True), np.setdiff1d(tk, mk, assume_unique=True)
    res["pool"] = {"mine": int(len(mk)), "theirs": int(len(tk)), "only_mine": int(len(only_mine)),
                   "only_theirs": int(len(only_theirs)), "same_order": bool(len(only_mine) == len(only_theirs) == 0),
                   "only_mine_samples": sample(only_mine, them["dates"], common),
                   "only_theirs_samples": sample(only_theirs, them["dates"], common)}
    say(f"{variant} pool {res['pool']}")
    save()

    dmap = them["dates"].get_indexer(rows.dates)
    ex = rows.exit_day[pool]
    ex = np.where(ex >= 0, dmap[np.maximum(ex, 0)], -1)
    y_mine = percentile(rows.ret[pool], rows.di[pool])
    ya, yb = y_mine[im], them["y"][it]
    same = (ya == yb) | (np.isnan(ya) & np.isnan(yb))
    d = np.abs(ya - yb)[~same]
    res["labels"] = {"exit_day_differ": int((ex[im] != them["exit_day"][it]).sum()), "y_differ": int((~same).sum()),
                     "y_max_abs_diff": float(np.nanmax(d)) if np.isfinite(d).any() else 0.0}
    say(f"{variant} labels {res['labels']}")
    save()

    feats = {}
    for k, name in enumerate(FEATURES):
        col = pq.read_table(them["src"] / "panel10.parquet", columns=[name], filters=FILTERS,
                            use_threads=False).column(0).to_numpy()[them["order"]]
        a, b = rows.x[pool[im], k], col[it].astype(np.float32)
        ok = (a == b) | (np.isnan(a) & np.isnan(b))
        if not ok.all():
            diff = np.abs(a.astype(np.float64) - b)[~ok]
            feats[name] = {"differ": int((~ok).sum()), "nan_mismatch": int((np.isnan(a) != np.isnan(b)).sum()),
                           "max_abs": float(np.nanmax(diff)) if np.isfinite(diff).any() else None}
        del col
    res["features"] = {"cells": int(len(im)) * len(FEATURES), "differ": feats}
    say(f"{variant} features differ {feats}")
    save()

    res["fits"] = {}
    leak_booster = None
    for month in FIT_MONTHS[variant]:
        f = int(them["dates"].searchsorted(pd.Timestamp(month + "-01")))
        fit_day = them["dates"][f]
        n = int(np.searchsorted(them["di"], f))
        keep = np.isfinite(them["y"][:n]) & (them["exit_day"][:n] >= 0) & (them["exit_day"][:n] < f)
        hi = int(np.searchsorted(them["di"], score_end(them["dates"], f)))
        n_train, n_test = int(keep.sum()), hi - n
        tag = hashlib.sha256(json.dumps([them["key"], THEIR_PARAMS, list(FEATURES), n_train, n_test, "batches"])
                             .encode()).hexdigest()[:8]
        path = them["src"] / "fits" / f"o1o5_{fit_day:%Y%m%d}_{tag}_s0.npy"
        entry = res["fits"].setdefault(month, {"fit": str(fit_day.date()), "their_train": n_train,
                                               "their_test": n_test, "file": path.name, "found": path.is_file()})
        if not path.is_file():
            say(f"{variant} {month}: 缓存文件 {path.name} 不在")
            continue
        theirs = np.load(path)
        data = T.training_set(rows, variant, int(rows.dates.searchsorted(fit_day)), research_labels=True)
        entry["my_train"] = int(data.keep.sum())
        t0 = time.time()
        booster = train_seed0(rows.x, data)
        sel = np.arange(n, hi)
        pos = np.minimum(np.searchsorted(it, sel), len(it) - 1)
        found = it[pos] == sel
        pred = booster.inplace_predict(rows.x[pool[im[pos[found]]]]).astype(np.float32)
        entry["predict"] = compare_predictions(them["di"][sel[found]], pred, theirs[found])
        entry["seconds"] = round(time.time() - t0)
        say(f"{variant} {month} {entry}")
        save()
        if month == LEAK_MONTH[variant]:
            leak_booster = booster
        if month == FIT_MONTHS[variant][0] and entry["predict"]["exact"] != entry["predict"]["rows"]:
            x_their = np.empty((hi, len(FEATURES)), np.float32)
            for k, name in enumerate(FEATURES):
                x_their[:, k] = pq.read_table(them["src"] / "panel10.parquet", columns=[name], filters=FILTERS,
                                              use_threads=False).column(0).to_numpy()[them["order"]][:hi]
            their_set = T.TrainingSet(prefix=np.arange(n), keep=keep, y=np.where(keep, them["y"][:n], np.nan))
            p2 = train_seed0(x_their, their_set).inplace_predict(x_their[n:hi]).astype(np.float32)
            entry["their_data_my_trainer"] = compare_predictions(them["di"][n:hi], p2, theirs)
            say(f"{variant} {month} their data through my trainer: {entry['their_data_my_trainer']}")
            save()
            del x_their
        del booster, data

    if leak_booster is not None:
        fit_day = them["dates"][int(them["dates"].searchsorted(pd.Timestamp(LEAK_MONTH[variant] + "-01")))]
        f = int(rows.dates.searchsorted(fit_day))
        live = T.training_set(rows, variant, f)
        research = T.training_set(rows, variant, f, research_labels=True)
        k = live.keep
        dy = np.abs(live.y[k] - research.y[k])
        changed = dy > 0
        leak = {"fit": str(fit_day.date()), "rows": int(k.sum()), "rows_changed": int(changed.sum()),
                "days_changed": int(len(np.unique(rows.di[live.prefix[k][changed]]))),
                "max_abs": float(dy.max()), "mean_abs_changed": float(dy[changed].mean()) if changed.any() else 0.0}
        test = pool[(rows.di[pool] >= f) & (rows.di[pool] < score_end(rows.dates, f))]
        p_live = train_seed0(rows.x, live).inplace_predict(rows.x[test]).astype(np.float32)
        p_res = leak_booster.inplace_predict(rows.x[test]).astype(np.float32)
        leak["scores_live_vs_research"] = day_stats(rows.di[test], p_live, p_res)
        res["leak_fix"] = leak
        say(f"{variant} leak fix {leak}")
        save()


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    say(f"power throttling off: {T.no_power_throttling()}")
    RESULT["raw_tables"] = raw_tables()
    say(f"raw tables {RESULT['raw_tables']}")
    save()
    rows = T.load_rows(DECADE_DB, log=say)
    RESULT["rows"] = {"total": int(len(rows.di)), **{v: int(m.sum()) for v, m in rows.member.items()},
                      "first": str(rows.dates[0].date()), "last": str(rows.dates[-1].date())}
    say(f"rows {RESULT['rows']}")
    save()
    for variant in ("cap70", "nocap"):
        check_variant(variant, rows)
    say("done")


if __name__ == "__main__":
    main()
