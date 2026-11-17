"""按顺序跑完最少持有天数实验（PREREG.md）：核对 → 主规则 → 敏感性 → 90 条 DB 规则全网格。

每步一个子进程（DECADE_DATA 指到本目录的 nocap/ 或 cap70/），日志在 logs/<步骤>.log；任何一步失败就停，写 FAILED。
全部跑完写 DONE。中断后 `--from 步骤名` 接着跑（算过的组合子进程自己会跳过）。
用法：python -X utf8 run_all.py [--from grid-nocap-200000]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECADE = HERE.parent / "decade"
SOURCES = {"nocap": DECADE / "nocap", "cap70": DECADE}
LINKED = ("panel10.parquet", "axes.json", "scores10_M.parquet")
ALL_N = ["1", "4", "2", "3", "5", "6", "8", "10"]

PLAN = [
    ("check-nocap", "nocap", ["--check"]),
    ("check-cap70", "cap70", ["--check"]),
    ("primary-nocap", "nocap", ["--capital", "20000", "200000", "--rules", "primary", "--fresh"]),
    ("free5-nocap", "nocap", ["--capital", "20000", "--rules", "primary", "--fresh", "--fee", "free5"]),
    ("primary-cap70", "cap70", ["--capital", "20000", "200000", "--rules", "primary", "--fresh"]),
    ("grid-nocap-200000", "nocap", ["--capital", "200000", "--rules", "grid", "--fresh"]),
    ("grid-nocap-20000", "nocap", ["--capital", "20000", "--rules", "grid", "--fresh"]),
]


def link(variant: str) -> Path:
    """本目录的变体子目录里放十年实验的面板、坐标轴、每月重训打分（硬链接，同一个盘不占地方；链不上就复制）。"""
    dst = HERE / variant
    dst.mkdir(exist_ok=True)
    for name in LINKED:
        target = dst / name
        if target.exists():
            continue
        try:
            os.link(SOURCES[variant] / name, target)
        except OSError:
            shutil.copy2(SOURCES[variant] / name, target)
    return dst


def log(line: str) -> None:
    stamp = time.strftime("%m-%d %H:%M:%S")
    with (HERE / "run_all.log").open("a", encoding="utf-8") as f:
        f.write(f"{stamp} {line}\n")
    print(f"{stamp} {line}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=PLAN[0][0], choices=[p[0] for p in PLAN])
    args = ap.parse_args()
    for marker in ("DONE", "FAILED"):
        (HERE / marker).unlink(missing_ok=True)
    (HERE / "logs").mkdir(exist_ok=True)
    first = [p[0] for p in PLAN].index(args.start)
    for step, variant, extra in PLAN[first:]:
        env = {**os.environ, "DECADE_DATA": str(link(variant)), "PYTHONUTF8": "1"}
        cmd = [sys.executable, "-X", "utf8", "-u", "run_minhold.py", *extra]
        if "--check" not in extra:
            cmd += ["--ns", *ALL_N]
        log(f"start {step}: {' '.join(cmd[3:])}")
        with (HERE / "logs" / f"{step}.log").open("w", encoding="utf-8") as out:
            code = subprocess.run(cmd, cwd=HERE, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
        if code != 0:
            log(f"FAILED {step} (exit {code})")
            (HERE / "FAILED").write_text(step, encoding="utf-8")
            sys.exit(code)
        log(f"ok {step}")
    (HERE / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log("ALL DONE")


if __name__ == "__main__":
    main()
