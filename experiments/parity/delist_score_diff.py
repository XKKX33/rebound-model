"""排除退市整理前后，两版正式打分的对比（只读）。

不限价、每月重训这条线：每天全池秩相关、前 6 名重合、分数完全相同的天数。
旧打分在写库前的备份里，旧模型在改名后的旧模型目录里。两个库都只读打开。
"""
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
NEW_MODELS = ROOT / "data/market.models/return_model"
OLD_MODELS = ROOT / "data/market-before-delist-202611010047.models/return_model"
OLD_DB = ROOT / "data/market.duckdb.bak-before-return-model-run-202610311928"


def manifests(root: Path, variant: str) -> dict[str, dict]:
    return {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / variant).glob("*/manifest.json"))}


for variant in ("nocap", "cap70"):
    new, old = manifests(NEW_MODELS, variant), manifests(OLD_MODELS, variant)
    gaps = {fit: old[fit]["train"]["rows"] - new[fit]["train"]["rows"] for fit in new}
    changed = [fit for fit in new if new[fit]["files"] != old[fit]["files"]]
    print(variant, "训练行少了：", sorted(set(gaps.values())), "最早变的重训日", changed[0], "模型文件变了的", len(changed), "/", len(new))

con = duckdb.connect(str(ROOT / "data/market.duckdb"), read_only=True)
con.execute(f"ATTACH '{OLD_DB.as_posix()}' AS old (READ_ONLY)")
# model_score 每个版本、每种重训频率各存一份（同一个模型可能同时是每季度和每月的那个），只取不限价、每月的。
for name, table in (("n", "model_score"), ("o", "old.model_score")):
    con.execute(f"CREATE TEMP TABLE {name} AS SELECT trade_date, ts_code, score FROM {table} WHERE variant = 'nocap' AND cadence = 'M'")
    print(name, con.execute(f"SELECT count(*), count(DISTINCT trade_date), count(*) - count(DISTINCT (trade_date, ts_code)) FROM {name}").fetchone())
rows = con.execute("""
WITH j AS (SELECT trade_date, n.score AS sn, o.score AS so FROM n JOIN o USING (trade_date, ts_code)),
r AS (SELECT trade_date, rank() OVER (PARTITION BY trade_date ORDER BY sn) AS rn, rank() OVER (PARTITION BY trade_date ORDER BY so) AS ro, abs(sn - so) AS d FROM j),
day AS (SELECT trade_date, corr(rn, ro) AS rho, max(d) AS maxd FROM r GROUP BY 1),
tn AS (SELECT trade_date, ts_code FROM n QUALIFY row_number() OVER (PARTITION BY trade_date ORDER BY score DESC) <= 6),
tt AS (SELECT trade_date, ts_code FROM o QUALIFY row_number() OVER (PARTITION BY trade_date ORDER BY score DESC) <= 6),
ov AS (SELECT trade_date, count(*) AS k FROM tn JOIN tt USING (trade_date, ts_code) GROUP BY 1),
d AS (SELECT CASE WHEN day.trade_date < DATE '2015-07-01' THEN '2015上半年' ELSE CAST(year(day.trade_date) AS VARCHAR) END AS period,
             rho, coalesce(k, 0) AS k, maxd FROM day LEFT JOIN ov USING (trade_date))
SELECT period, count(*), round(avg(rho), 4), round(min(rho), 4), round(avg(k), 2), sum(CASE WHEN maxd = 0 THEN 1 ELSE 0 END)
FROM d GROUP BY ROLLUP (period) ORDER BY period NULLS LAST
""").fetchall()
print("时段 天数 秩相关均值 最低 前6重合 分数完全相同的天数")
for row in rows:
    print(row)
for name in ("o", "n"):
    print(name, con.execute(f"SELECT ts_code FROM {name} WHERE trade_date = DATE '2026-10-30' ORDER BY score DESC LIMIT 8").fetchall())
