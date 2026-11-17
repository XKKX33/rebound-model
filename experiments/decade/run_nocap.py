"""不限价版一条龙（准入池去掉 70 元上限，其余和原版一样），结果都在 nocap/ 下。
等原版十年实验的 DONE（避开两边同时训练），然后：面板 → 每年、每半年重训 → 十年模拟 → 20 万规则搜索 → 和限价版对比（初步）
→ 每季度、每月重训 → 十年模拟 → 20 万补齐 → 2 万规则搜索 → 对比（终版）。
任何一步失败就停，写 nocap/NOCAP_FAILED；全部跑完写 nocap/NOCAP_DONE。日志 nocap/run_nocap.log。
用法：python -X utf8 run_nocap.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "nocap"
LOG, DONE, FAILED = OUT / "run_nocap.log", OUT / "NOCAP_DONE", OUT / "NOCAP_FAILED"
PY = [sys.executable, "-X", "utf8"]
ENV = {**os.environ, "DECADE_DATA": str(OUT)}
PLAIN = {k: v for k, v in os.environ.items() if k != "DECADE_DATA"}  # 对比脚本读两版，按原版目录跑


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


#: 低于正常的优先级：和原版的规则搜索（单线程）同时跑时，让原版先用 CPU
LOW = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)


def run(args: list[str], env: dict = ENV) -> None:
    log("start " + " ".join(args))
    with LOG.open("a", encoding="utf-8") as f:
        code = subprocess.call(PY + args, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT, creationflags=LOW)
    if code:
        FAILED.write_text(" ".join(args) + f" exit {code}\n", encoding="utf-8")
        log(f"FAILED {' '.join(args)} exit {code}")
        raise SystemExit(code)
    log("ok " + " ".join(args))


def rules(capital: str, cadences: list[str], finalists: bool) -> None:
    run(["rule_search.py", "--capital", capital, "--tier", "real", "--cadences", *cadences])
    run(["rule_select.py", "--capital", capital])
    if finalists:
        run(["rule_search.py", "--capital", capital, "--tier", "hist", "--finalists"])
    run(["rule_report.py", "--capital", capital])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    DONE.unlink(missing_ok=True)
    FAILED.unlink(missing_ok=True)
    log("等原版十年实验的 DONE")
    while not (HERE / "DONE").is_file():
        if (HERE / "FAILED").is_file():
            FAILED.write_text("原版十年实验失败，没开始\n", encoding="utf-8")
            raise SystemExit(1)
        time.sleep(60)
    run(["nocap_panel.py"])
    run(["train_cadence.py", "--cadences", "Y", "H"])
    run(["decade_sim.py"])
    run(["decade_report.py"])
    rules("200000", ["Y", "H"], finalists=False)
    run(["compare_cap.py"], env=PLAIN)
    run(["train_cadence.py", "--cadences", "Y", "H", "Q", "M"])
    run(["decade_sim.py"])
    run(["decade_report.py"])
    rules("200000", ["Q", "M"], finalists=True)
    rules("20000", list("YHQM"), finalists=True)
    run(["compare_cap.py"], env=PLAIN)
    DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    log("NOCAP DONE")


if __name__ == "__main__":
    main()
