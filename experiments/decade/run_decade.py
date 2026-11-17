"""十年实验一条龙：依次建面板、重训、模拟、出报告（十年库要先建好，见 common.py 的 DB）。

先只跑按年：建面板 → 按年重训 → 模拟 → 报告（先出一版只有按年的报告）；再补半年、季、月 → 模拟 → 报告。
任何一步失败就停，写 FAILED 和出错的那一步；全部跑完写 DONE。日志 run_decade.log。
用法：python -X utf8 run_decade.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
LOG, DONE, FAILED = HERE / "run_decade.log", HERE / "DONE", HERE / "FAILED"
#: 这些脚本还在跑就先等（很占内存）。原来要等补取行情和上一轮实验的后台任务；本仓库里没有要等的。
WAIT_FOR: tuple[str, ...] = ()
MIN_FREE_GB = 1.0  # 训练改成分批喂数据以后，峰值约 0.9 GB
STEPS = [
    ("panel10.py",),
    ("train_cadence.py", "--cadences", "Y"),
    ("decade_sim.py",),
    ("decade_report.py",),
    ("train_cadence.py", "--cadences", "Y", "H", "Q", "M"),
    ("decade_sim.py",),
    ("decade_report.py",),
]


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def busy() -> list[str]:
    found = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if "python" not in (p.info["name"] or "").lower():
                continue
            cmd = " ".join(p.info["cmdline"] or [])
            found += [s for s in WAIT_FOR if s in cmd]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(set(found))


def wait_until_free() -> None:
    """两次相隔一分钟都没有要等的进程、而且空闲内存够，才开始（中间换下一步的那几秒不算空闲）。

    要等的进程都结束以后，内存最多再等 30 分钟（别的程序占着内存不放时，不能一直等下去）。
    """
    clear, idle_since = 0, None
    while clear < 2:
        running, free = busy(), psutil.virtual_memory().available / 2**30
        idle_since = None if running else idle_since or time.time()
        enough = free >= MIN_FREE_GB or (idle_since is not None and time.time() - idle_since > 1800)
        clear = clear + 1 if not running and enough else 0
        if clear < 2:
            log(f"waiting: running={running} free={free:.2f}GB")
            time.sleep(60 if clear or not running else 300)


def run(step: tuple[str, ...]) -> None:
    log("start " + " ".join(step))
    t0 = time.time()
    with (HERE / f"run_{step[0].removesuffix('.py')}.log").open("a", encoding="utf-8") as out:
        code = subprocess.call([sys.executable, "-X", "utf8", "-u", *step], cwd=HERE, stdout=out, stderr=subprocess.STDOUT)
    if code != 0:
        FAILED.write_text(" ".join(step) + f" exit {code}\n", encoding="utf-8")
        log(f"FAILED {' '.join(step)} exit {code}")
        raise SystemExit(code)
    log(f"done {' '.join(step)} ({time.time() - t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="first", type=int, default=0, help="从第几步（0 起）接着跑，前面的已经跑完")
    args = ap.parse_args()
    DONE.unlink(missing_ok=True)
    FAILED.unlink(missing_ok=True)
    wait_until_free()
    for step in STEPS[args.first:]:
        run(step)
    DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    log("ALL DONE")


if __name__ == "__main__":
    main()
