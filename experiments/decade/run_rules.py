"""持股规则搜索一条龙：等十年实验的 DONE（训练和 decade_sim 都跑完），然后
20 万：补算每季度、每月 → 选优 → 入围规则补 hist 档 → 报告；2 万：同样一遍。
任何一步失败就停，写 RULES_FAILED 和出错的那一步；全部跑完写 RULES_DONE。日志 run_rules.log。
2 万的每年、每半年另一个窗口在算，这里等它两份结果都满 150 组再选优。
用法：python -X utf8 run_rules.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG, DONE, FAILED = HERE / "run_rules.log", HERE / "RULES_DONE", HERE / "RULES_FAILED"
N_RULES = 150
PY = [sys.executable, "-X", "utf8"]


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args: list[str]) -> None:
    log("start " + " ".join(args))
    with LOG.open("a", encoding="utf-8") as f:
        code = subprocess.call(PY + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    if code:
        FAILED.write_text(" ".join(args) + f" exit {code}\n", encoding="utf-8")
        log(f"FAILED {' '.join(args)} exit {code}")
        raise SystemExit(code)
    log("ok " + " ".join(args))


def complete(capital: int, cadences: tuple[str, ...]) -> bool:
    for c in cadences:
        path = HERE / "rules" / str(capital) / f"real_{c}.json"
        if not path.is_file() or len(json.loads(path.read_text(encoding="utf-8"))) < N_RULES:
            return False
    return True


def wait_for(what: str, ok) -> None:
    log(f"等 {what}")
    while not ok():
        if (HERE / "FAILED").is_file():
            FAILED.write_text(f"十年实验失败，没等到 {what}\n", encoding="utf-8")
            log("十年实验 FAILED，停")
            raise SystemExit(1)
        time.sleep(60)
    log(f"等到了 {what}")


def chain(capital: int) -> None:
    cap = ["--capital", str(capital)]
    run(["rule_search.py", *cap, "--tier", "real", "--cadences", "Q", "M"])
    wait_for(f"{capital} 元每年、每半年的 {N_RULES} 组", lambda: complete(capital, ("Y", "H")))
    run(["rule_select.py", *cap])
    run(["rule_search.py", *cap, "--tier", "hist", "--finalists"])
    run(["rule_report.py", *cap])


def main() -> None:
    DONE.unlink(missing_ok=True)
    FAILED.unlink(missing_ok=True)
    wait_for("十年实验的 DONE", lambda: (HERE / "DONE").is_file())
    for capital in (200_000, 20_000):
        chain(capital)
    DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    log("RULES DONE")


if __name__ == "__main__":
    main()
