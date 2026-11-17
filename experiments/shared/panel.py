"""从研究库建一张主板日线面板：复权开收盘、涨跌停受阻标记、准入池、28 个量价特征。

特征和准入池与 stage_data.load_bars 完全同一套（直接用它的 SQL 和同样的后处理）。
内存紧（本机空闲约 2GB），所以 SQL 结果先由 DuckDB 直接写盘，pandas 只分批读需要的列，最后再由 DuckDB 合并。
只读研究库。输出 panel.parquet 和 panel.meta.json。

用法：python -X utf8 panel.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from stage_data import FEATURES, ROOT, SQL, TOL, _ema  # noqa: E402

RAW, EXTRA, OUT = HERE / "panel.raw.parquet", HERE / "panel.extra.parquet", HERE / "panel.parquet"
DB = ROOT / "data/research.duckdb"
SETTINGS = "SET memory_limit='900MB'; SET threads=4; SET preserve_insertion_order=true; SET enable_progress_bar=false"


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DB), read_only=True)
    con.execute(SETTINGS + f"; SET temp_directory='{(HERE / 'duckdb-tmp').as_posix()}'")
    return con


def extra_columns() -> pd.DataFrame:
    """stage_data.load_bars 在 pandas 里补的列：MACD 比值、截面排名、涨跌停受阻、准入池。"""
    base = pd.read_parquet(RAW, columns=["ts_code", "trade_date", "bar_no", "c", "r20", "r60"],
                           read_dictionary=["ts_code"])
    by_code = base.groupby("ts_code", sort=False, observed=True)["c"]
    dif = by_code.transform(_ema, 12) - by_code.transform(_ema, 26)
    dea = dif.groupby(base.ts_code, sort=False, observed=True).transform(_ema, 9)
    warm = base.bar_no < 33
    out = base[["ts_code", "trade_date"]].copy()
    out["dif_ratio"] = (dif / base.c).mask(warm)
    out["hist_ratio"] = (2 * (dif - dea) / base.c).mask(warm)
    for col in ("r20", "r60"):
        out[f"rank_{col}"] = base.groupby("trade_date")[col].rank(pct=True)
    del base, dif, dea, by_code
    px = pd.read_parquet(RAW, columns=["open", "high", "low", "close", "amount", "bar_no", "up_limit", "down_limit"])

    def touch(limit: str, cols: tuple[str, ...]) -> np.ndarray:
        hits = [(px[col] - px[limit]).abs() <= TOL for col in cols]
        return (px[limit].notna() & np.logical_and.reduce(hits)).to_numpy()

    four = ("open", "high", "low", "close")
    out["up_open"], out["dn_open"] = touch("up_limit", four), touch("down_limit", four)
    out["up_close"], out["dn_close"] = touch("up_limit", ("close",)), touch("down_limit", ("close",))
    con = _con()
    try:
        names = con.execute("SELECT ts_code, name FROM stock_basic").df()
    finally:
        con.close()
    st = names.name.isna() | names.name.astype(str).str.contains(r"ST|\*", regex=True)
    ok_code = out.ts_code.isin(names.loc[~st, "ts_code"]).to_numpy()
    out["eligible"] = ok_code & (px.close < 70).to_numpy() & (px.amount >= 80_000).to_numpy() & (px.bar_no >= 149).to_numpy()
    return out


def main() -> None:
    start = time.time()
    con = _con()
    try:
        # trade_date 转成 TIMESTAMP：pandas 读 DATE 会变成 Python 对象列，几百万行太占内存。
        con.execute(f"COPY (SELECT * REPLACE (CAST(trade_date AS TIMESTAMP) AS trade_date) FROM ({SQL}) "
                    f"ORDER BY ts_code, trade_date) TO '{RAW.as_posix()}' (FORMAT PARQUET)")
        cal = [pd.Timestamp(d) for (d,) in con.execute(
            "SELECT cal_date FROM trade_cal WHERE exchange='SSE' AND is_open=1 ORDER BY 1").fetchall()]
    finally:
        con.close()
    print(f"raw written ({time.time() - start:.0f}s)", flush=True)
    extra_columns().to_parquet(EXTRA, index=False)
    print(f"extra written ({time.time() - start:.0f}s)", flush=True)
    feats = ", ".join(f"CAST(CASE WHEN isfinite(r.{f}) THEN r.{f} END AS FLOAT) AS {f}"
                      if f not in ("dif_ratio", "hist_ratio", "rank_r20", "rank_r60")
                      else f"CAST(CASE WHEN isfinite(x.{f}) THEN x.{f} END AS FLOAT) AS {f}" for f in FEATURES)
    con = duckdb.connect()
    con.execute(SETTINGS + f"; SET temp_directory='{(HERE / 'duckdb-tmp').as_posix()}'")
    con.execute(f"""COPY (
        SELECT r.ts_code, CAST(r.trade_date AS TIMESTAMP) AS trade_date, CAST(r.bar_no AS INTEGER) AS bar_no,
               r.o, r.c, r.amount, x.up_open, x.up_close, x.dn_open, x.dn_close, x.eligible, {feats}
        FROM read_parquet('{RAW.as_posix()}') r
        JOIN read_parquet('{EXTRA.as_posix()}') x
          ON r.ts_code = x.ts_code AND r.trade_date = CAST(x.trade_date AS TIMESTAMP)
        ORDER BY r.ts_code, r.trade_date
    ) TO '{OUT.as_posix()}' (FORMAT PARQUET)""")
    stats = con.execute(f"SELECT count(*), count(DISTINCT ts_code), count(DISTINCT trade_date), min(trade_date), "
                        f"max(trade_date), sum(CAST(eligible AS INTEGER)) FROM read_parquet('{OUT.as_posix()}')").fetchone()
    days = [pd.Timestamp(d) for (d,) in con.execute(
        f"SELECT DISTINCT trade_date FROM read_parquet('{OUT.as_posix()}') ORDER BY 1").fetchall()]
    con.close()
    window = [d for d in cal if days[0] <= d <= days[-1]]
    meta = {"rows": stats[0], "codes": stats[1], "dates": stats[2], "first": str(stats[3]), "last": str(stats[4]),
            "eligible_rows": int(stats[5]), "calendar_matches": window == days,
            "source": str(DB), "seconds": round(time.time() - start)}
    (HERE / "panel.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    RAW.unlink()
    EXTRA.unlink()
    print(meta, flush=True)
    if not meta["calendar_matches"]:
        raise SystemExit("面板的日期和 SSE 开市日不一致")


if __name__ == "__main__":
    main()
