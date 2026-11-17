"""补充登记八一条龙（结果都在 win/ 下）：
链接原版面板和标签 → 冒烟自检（窗口版读原版打分时，同频率的逐日差必须是 0）→ 每年重训 W2 / W3 → 每年版比较、
每年这一档的全网格 → 中间报告（写 win/WINDOW_Y_DONE）→ 每月重训（顺带每半年、每季度）→ 终版比较 → 四种频率的全网格和
滚动选法 → 终版报告。任何一步失败就停，写 win/WINDOW_FAILED；全部跑完写 win/WINDOW_DONE。日志 win/run_window.log。
优先级低于正常：别的会话的任务先用 CPU。
用法：python -X utf8 run_window.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "win"
LOG, MID, DONE, FAILED = OUT / "run_window.log", OUT / "WINDOW_Y_DONE", OUT / "WINDOW_DONE", OUT / "WINDOW_FAILED"
YEARS = (2, 3)
PY = [sys.executable, "-X", "utf8"]
BASE = {k: v for k, v in os.environ.items() if k != "DECADE_DATA"}  # 比较和报告脚本读原版目录
LOW = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args: list[str], data: Path | None = None) -> None:
    env = {**BASE, "DECADE_DATA": str(data)} if data else BASE
    tag = f" [DECADE_DATA={data.relative_to(HERE)}]" if data else ""
    log("start " + " ".join(args) + tag)
    with LOG.open("a", encoding="utf-8") as f:
        code = subprocess.call(PY + args, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT, creationflags=LOW)
    if code:
        FAILED.write_text(" ".join(args) + tag + f" exit {code}\n", encoding="utf-8")
        log(f"FAILED {' '.join(args)}{tag} exit {code}")
        raise SystemExit(code)
    log("ok " + " ".join(args) + tag)


def link_all(d: Path) -> None:
    """窗口版目录里放原版面板、标签的硬链接（同一个盘不占地方），链不上就复制。已经有了就不动。"""
    labels = sorted(HERE.glob("labels10_o1o5_*.parquet"))
    if len(labels) != 1:
        raise SystemExit(f"本目录应该正好有一份标签文件，现在是 {[p.name for p in labels]}")
    d.mkdir(parents=True, exist_ok=True)
    for src in (HERE / "panel10.parquet", HERE / "panel10.meta.json", HERE / "axes.json", labels[0]):
        dst = d / src.name
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for marker in (MID, DONE, FAILED):
        marker.unlink(missing_ok=True)
    dirs = {n: OUT / f"w{n}" for n in YEARS}
    for d in dirs.values():
        link_all(d)
    run(["window_eval.py", "--stage", "y", "--self-check"])
    for n, d in dirs.items():
        run(["train_cadence.py", "--cadences", "Y", "--window-years", str(n)], d)
    run(["window_eval.py", "--stage", "y"])
    for d in dirs.values():
        for capital in ("200000", "20000"):
            run(["rule_search.py", "--capital", capital, "--tier", "real", "--cadences", "Y"], d)
    run(["window_report.py"])
    MID.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    log("每年重训的中间结果已出")
    for n, d in dirs.items():  # 每月的重训日包含每年、每半年、每季度的，先训每月，其余直接用缓存拼
        run(["train_cadence.py", "--cadences", "M", "Y", "H", "Q", "--window-years", str(n)], d)
    run(["window_eval.py", "--stage", "m"])
    for d in dirs.values():
        for capital in ("200000", "20000"):
            run(["rule_search.py", "--capital", capital, "--tier", "real", "--cadences", "Y", "H", "Q", "M"], d)
            run(["rule_select.py", "--capital", capital], d)
    run(["window_report.py"])
    DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    log("WINDOW DONE")


if __name__ == "__main__":
    main()
