"""Generate the source fingerprint and one feature slice from a supplied database."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date

from paths import PROJECT, MODEL_REPO, DATABASE, PROFILE, FEATURE_CACHE, FEATURE_STAMP

sys.path[:0] = [str(PROJECT / "work/runtime"), str(MODEL_REPO)]

import duckdb
from rebound.features import feature_panel
from rebound.store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", type=date.fromisoformat)
    args = parser.parse_args()
    if not DATABASE.is_file():
        raise SystemExit(f"未找到数据库：{DATABASE}")
    fingerprint = hashlib.sha256()
    with DATABASE.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            fingerprint.update(chunk)
    profile = {"provenance": {"sha256": fingerprint.hexdigest(), "bytes": DATABASE.stat().st_size}, "tables": {}}
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        con.execute("SET threads=2")
        for (table,) in con.execute("SHOW TABLES").fetchall():
            quoted = '"' + table.replace('"', '""') + '"'
            columns = con.execute(f"DESCRIBE {quoted}").fetchall()
            record = {"rows": con.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]}
            names = {column[0] for column in columns}
            if "trade_date" in names:
                record["date_min"], record["date_max"] = con.execute(
                    f"SELECT min(trade_date),max(trade_date) FROM {quoted}"
                ).fetchone()
            if "ts_code" in names:
                record["stocks"] = con.execute(f"SELECT count(DISTINCT ts_code) FROM {quoted}").fetchone()[0]
            profile["tables"][table] = record
        profile["daily_quality"] = con.execute("""
            SELECT count(*)-count(DISTINCT (ts_code,trade_date)) duplicate_keys,
                   count(*) FILTER (WHERE close IS NULL) missing_close,
                   count(*) FILTER (WHERE amount IS NULL) missing_amount,
                   count(*) FILTER (WHERE close<=0) nonpositive_close
            FROM daily
        """).fetchdf().to_dict(orient="records")[0]
    target = args.as_of or profile["tables"]["daily"]["date_max"]
    if target > profile["tables"]["daily"]["date_max"]:
        raise SystemExit("特征日期超出本地历史库。")
    with Store(DATABASE, read_only=True, visible_through=target) as store:
        store.connection.execute("SET threads=2")
        frame = feature_panel(store, target, target)
    PROFILE.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PROFILE.write_text(json.dumps(profile, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    frame.to_parquet(FEATURE_CACHE, index=False)
    FEATURE_STAMP.write_text(
        json.dumps({"as_of": str(target), "sha256": fingerprint.hexdigest()}), encoding="utf-8",
    )
    print(f"数据指纹和 {target} 特征截面已生成：{len(frame)} 只股票。")


if __name__ == "__main__":
    main()
