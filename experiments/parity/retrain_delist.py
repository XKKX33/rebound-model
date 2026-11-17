"""2026-10-31 夜：准入池排除退市整理（`ST_PATTERN` 加"退"）之后，重训、重打分、重算本金测算。一次性脚本，可以重跑（接着做）。

步骤写进 retrain_delist.log，状态写进 retrain_delist_status.json；哪一步失败就停在那里，不往下做。
1. 全量测试：除了已知的 test_coverage（正式库 T+5 覆盖率），有别的失败就停，不重训（--skip-tests 跳过）。
   测试是轻活，和别的会话的任务一起跑没关系。
1.5 等别的会话的重任务（--wait-pid，可以给几个）结束再训练：计划里定的，另一个会话在跑重任务时不跑训练（内存紧）。
   等了 --wait-hours 还没完就停在这里，不硬上。
2. 旧模型目录改名留着（data/market-before-delist-<时间>.models/），重训 284 次。重跑时目录已是新口径的就接着训。
3. 删掉旧模型打的分（旧分在 data/market.duckdb.bak-before-return-model-run-202610311928 里），重打分。
4. 和十年实验比分数（scores_check.py；旧结果先改名留着）。
5. 本金测算 2 万、20 万（不限价、最低 5 元），存进缓存，页面直接能读；再和十年实验比（compare_capital.py）。
跑的时候：低优先级；插着电时不让系统空闲睡眠（进程退出就放开）。
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

LAB = Path(__file__).resolve().parent
ROOT = LAB.parents[1]
sys.path.insert(0, str(ROOT))
DB = ROOT / "data" / "market.duckdb"
LOG = LAB / "retrain_delist.log"
STATUS = LAB / "retrain_delist_status.json"
PY = sys.executable
KNOWN_FAIL = "tests/test_coverage.py::test_coverage_on_the_real_database_meets_the_blueprint_target"
OLD_SCORES = "data/market.duckdb.bak-before-return-model-run-202610311928"
STATE: dict = {"started": datetime.now().isoformat(timespec="seconds"), "steps": {}}


def log(line: str) -> None:
    text = f"{datetime.now():%m-%d %H:%M:%S} {line}"
    print(text, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def save(step: str, **info) -> None:
    STATE["steps"][step] = {"at": datetime.now().isoformat(timespec="seconds"), **info}
    STATUS.write_text(json.dumps(STATE, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def stop(step: str, why: str) -> None:
    log(f"停在「{step}」：{why}")
    STATE["stopped"] = {"step": step, "why": why}
    save(step, ok=False, why=why)
    sys.exit(1)


def low_priority_and_awake() -> bool:
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)  # 子进程（测试、本金测算）跟着低优先级
    battery = psutil.sensors_battery()
    if battery is not None and not battery.power_plugged:
        return False  # 用电池时不拦睡眠
    es_continuous, es_system_required = 0x80000000, 0x00000001
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required))


def wait_pids(pids: list[int], hours: float) -> None:
    alive = [p for p in pids if psutil.pid_exists(p)]
    if not alive:
        return
    log(f"等别的会话的重任务结束再训练：进程 {alive}（最多 {hours:g} 小时）")
    save("wait", ok=None, pids=alive, since=datetime.now().isoformat(timespec="seconds"))
    t0 = time.time()
    while any(psutil.pid_exists(p) for p in pids) and time.time() - t0 < hours * 3600:
        time.sleep(60)
    still = [p for p in pids if psutil.pid_exists(p)]
    if still:
        stop("wait", f"等了 {hours:g} 小时，进程 {still} 还在跑：不和它抢内存，没训练。它结束后加 --skip-tests 重跑")
    save("wait", ok=True, pids=pids, waited_seconds=round(time.time() - t0))
    log(f"别的会话的重任务结束了（等了 {time.time() - t0:.0f}s），开始训练")


def run_tests() -> None:
    out, t0 = LAB / "pytest_delist.log", time.time()
    with out.open("w", encoding="utf-8") as f:
        code = subprocess.run([PY, "-m", "pytest", "-p", "no:cacheprovider", "-q"], cwd=ROOT, stdout=f,
                              stderr=subprocess.STDOUT).returncode
    failed = [line.split(" ", 2)[1] for line in out.read_text(encoding="utf-8", errors="replace").splitlines()
              if line.startswith(("FAILED ", "ERROR "))]
    other = [t for t in failed if t != KNOWN_FAIL]
    save("tests", ok=not other, exit=code, failed=failed, seconds=round(time.time() - t0))
    log(f"全量测试：退出码 {code}，没过 {failed}（{time.time() - t0:.0f}s）")
    if other:
        stop("tests", f"有已知之外的失败：{other}，不重训")


def train_models() -> None:
    from rebound.features import fingerprint
    from rebound.model.artifacts import list_blocks, models_root
    from rebound.model.train import train

    want, root = fingerprint(), models_root(DB).parent
    blocks = [b for v in ("nocap", "cap70") for b in list_blocks(DB, v)]
    old = [b for b in blocks if b.manifest["features"]["fingerprint"] != want]
    if old:
        if len(old) != len(blocks):
            stop("train", f"模型目录里新旧口径混着（旧 {len(old)}、新 {len(blocks) - len(old)}）：先手工看一下")
        target = root.with_name(f"market-before-delist-{datetime.now():%Y%m%d%H%M}.models")
        root.rename(target)
        log(f"旧模型目录（{len(old)} 次重训，指纹 {old[0].manifest['features']['fingerprint']}）改名留着：{target}")
        save("models_backup", path=str(target), blocks=len(old))
    t0 = time.time()
    report = train(DB, log=log)
    save("train", ok=True, trained=len(report.trained), skipped=len(report.skipped), fingerprint=want,
         seconds=round(time.time() - t0))
    del report
    gc.collect()


def rescore() -> None:
    import duckdb

    from rebound.data.db import WriteStore
    from rebound.model.artifacts import list_blocks
    from rebound.model.score import fill_scores

    current = sorted({b.model_id for v in ("nocap", "cap70") for b in list_blocks(DB, v)})
    for attempt in range(60):
        try:
            with WriteStore(DB) as store:
                before = store.row_count("model_score")
                store.connection.execute("DELETE FROM model_score WHERE model_id NOT IN (SELECT unnest(?))", [current])
                after = store.row_count("model_score")
            break
        except duckdb.IOException as exc:
            log(f"库被占着（{exc}），一分钟后再试（第 {attempt + 1} 次）")
            time.sleep(60)
    else:
        stop("rescore", "库一直被别的进程占着，没删旧分")
    log(f"删掉旧模型打的分 {before - after} 行，留下 {after} 行（旧分在 {OLD_SCORES}）")
    t0, days, rows = time.time(), 0, 0
    for attempt in range(30):
        try:
            report = fill_scores(DB, log=log)
            days, rows = days + report.days, rows + report.rows
            break
        except duckdb.IOException as exc:
            log(f"打分时库被占着（{exc}），一分钟后接着补（已写的日子不重写）")
            time.sleep(60)
    else:
        stop("rescore", "打分时库一直被别的进程占着")
    save("rescore", ok=True, deleted=before - after, days=days, rows=rows, seconds=round(time.time() - t0))


def compare_scores() -> None:
    kept = LAB / "scores_check.before-delist.json"
    if (LAB / "scores_check.json").exists() and not kept.exists():
        (LAB / "scores_check.json").rename(kept)
    with (LAB / "scores_check_delist.log").open("w", encoding="utf-8") as f:
        code = subprocess.run([PY, "-X", "utf8", "-B", "scores_check.py"], cwd=LAB, stdout=f,
                              stderr=subprocess.STDOUT).returncode
    if code:
        stop("scores_check", f"退出码 {code}，见 scores_check_delist.log")
    got = json.loads((LAB / "scores_check.json").read_text(encoding="utf-8"))
    save("scores_check", ok=True, series={k: {f: v[f] for f in ("days", "pool_days_differ", "only_research_rows",
                                                               "rho_mean", "top6_mean", "top30_mean")}
                                         for k, v in got.items()})


def capital_runs() -> None:
    for cap in (20000, 200000):
        out = LAB / f"capital_delist_{cap}_nocap.json"
        with out.open("w", encoding="utf-8") as f, (LAB / f"capital_delist_{cap}.log").open("w", encoding="utf-8") as e:
            code = subprocess.run([PY, "-X", "utf8", "-B", "-m", "rebound", "capital", "--capital", str(cap), "--json"],
                                  cwd=ROOT, stdout=f, stderr=e).returncode
        if code:
            stop("capital", f"{cap} 退出码 {code}，见 capital_delist_{cap}.log")
        with (LAB / f"capital_delist_{cap}.txt").open("w", encoding="utf-8") as f:  # 缓存里读，几秒
            subprocess.run([PY, "-X", "utf8", "-B", "-m", "rebound", "capital", "--capital", str(cap)], cwd=ROOT,
                           stdout=f, stderr=subprocess.DEVNULL)
        prev, keep = LAB / f"compare_capital_{cap}_nocap.json", LAB / f"compare_capital_{cap}_nocap.before-delist.json"
        if prev.exists() and not keep.exists():
            prev.rename(keep)
        compared = subprocess.run([PY, "-X", "utf8", "-B", "compare_capital.py", out.name, str(cap), "nocap"], cwd=LAB,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        log(f"和十年实验比本金测算 {cap}：退出码 {compared}（compare_capital_{cap}_nocap.json）")
        d = json.loads(out.read_text(encoding="utf-8"))
        rec = d["rows"][d["recommended"]] if d["recommended"] is not None else None
        walk = d["walk_forward"][d["picker"]]
        save(f"capital_{cap}", ok=True, key=d["key"], basis=d["basis"],
             walk_ir={k: walk[k] for k in ("excess", "ci", "cagr", "max_dd")},
             recommended=None if rec is None else {
                 **{k: rec[k] for k in ("cadence", "rule", "cagr", "excess", "max_dd")},
                 "fresh": {y: v["cagr"] for y, v in rec["fresh"]["starts"].items()} if rec["fresh"] else None,
                 "new_buys": [(b["ts_code"], b["name"], b["shares"]) for b in rec["new"]["buys"]]})
        log(f"本金测算 {cap}：推荐 {rec and (rec['cadence'], rec['rule'], rec['cagr'])}（{d['basis']}），编号 {d['key']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, action="append", default=[])
    ap.add_argument("--wait-hours", type=float, default=10.0)
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    awake = low_priority_and_awake()
    log(f"开始：低优先级；{'插着电，跑完之前不让系统空闲睡眠' if awake else '在用电池（或设不了），不拦睡眠'}")
    try:
        if not args.skip_tests:
            run_tests()
        wait_pids(args.wait_pid, args.wait_hours)
        train_models()
        rescore()
        compare_scores()
        capital_runs()
    except KeyboardInterrupt:
        stop("interrupted", "手工停了（Ctrl-C）；重跑接着做，测试过了就加 --skip-tests")
    STATE["finished"] = datetime.now().isoformat(timespec="seconds")
    save("done", ok=True)
    log("全部做完")


if __name__ == "__main__":
    main()
