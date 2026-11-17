"""正式库上试算纸面跟踪（只读库；跟踪文件写到临时目录，不在正式库旁边建 `.tracking/`）。

两条核对，都对着 `capital_delist_20000_nocap.json`（同一份打分算的本金测算，2 万、每月 DB-K2-b8）：
1. 下一个开市日起步（11-02）：清单 = 本金测算里"今天开新账户明天买"的那份。
2. 2023-01-04 起步：期末倍数 = 本金测算换起点体检 2023 年那一格（同一天收盘起步）。
"""

import json
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rebound import track
from rebound.backtest import capital  # noqa: E402

HERE = Path(__file__).resolve().parent
DB = ROOT / "data" / "market.duckdb"
tmp = Path(tempfile.mkdtemp(prefix="track-dryrun-"))
track.tracking_root = lambda _db: tmp  # 不在正式库旁边建目录

ref = json.loads((HERE / "capital_delist_20000_nocap.json").read_text(encoding="utf-8"))
row = next(r for r in ref["rows"] if r["cadence"] == "M" and r["name"] == "DB-K2-b8")
request = capital.CapitalRequest(capital=20_000.0)


def tracker(start: date) -> track.Tracker:
    with track.open_store(DB) as store:
        first_buy, signal = track.resolve_start(store, "nocap", "M", start)
    return track.Tracker(track_id=track._new_id(), rule=row["rule"], cadence="M", variant="nocap", capital=20_000.0,
                         commission=request.commission, min_commission=request.min_commission, stamp=request.stamp,
                         transfer=request.transfer, start=first_buy.isoformat(), signal_day=signal.isoformat(),
                         created_at=track._now())


t0 = time.time()
waiting = track.compute(DB, tracker(date(2026, 11, 2)))
print("1. 起步", waiting["start"], "状态", waiting["state"], "打分到", waiting["scored_through"], "下一个开市日", waiting["next_open"],
      f"（{time.time() - t0:.0f}s）")
same = {k: waiting["plan"][k] == row["new"][k] for k in ("buys", "alternates", "sells", "cash", "signal_day")}
print("   清单和本金测算的新账户逐项相同：", same)
for b in waiting["plan"]["buys"]:
    print("   买", b["ts_code"], b["name"], b["shares"], "股", b["amount"], "元")

t0 = time.time()
running = track.compute(DB, tracker(date(2023, 1, 4)))
s, fresh = running["summary"], row["fresh"]["starts"]["2023"]
print("2. 起步", running["start"], "（信号", running["signal_day"], "）状态", running["state"], f"（{time.time() - t0:.0f}s）")
print("   账户", s["value"], "元，累计", s["return_pct"], "%，回撤", s["max_dd_pct"], "%，卖掉", s["trades"], "笔，费用", s["fees"], "元")
print("   期末倍数", round(s["value"] / 20_000.0, 2), "对 本金测算换起点 2023：", fresh["final_x"], "；回撤", fresh["max_dd"])
print("   清单：买", [b["ts_code"] for b in running["plan"]["buys"]], "卖", [x["ts_code"] for x in running["plan"]["sells"]],
      "；持仓", [h["ts_code"] for h in running["holdings"]])
print("   JSON 严格可读：", bool(json.dumps(running, allow_nan=False)), "；临时目录", tmp)
