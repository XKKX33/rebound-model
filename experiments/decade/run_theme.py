"""补充登记七一条龙（结果都在 ind/ 下）：
行业特征 → 冒烟自检（V1 = V0 时逐日差必须是 0）→ 每季度重训（顺带每年、每半年）→ 中间检查和止损
→ 每月重训 → 终版比较 → V1 全网格（2 个本金 × 4 种频率）和滚动选法 → 报告。
任何一步失败就停，写 ind/THEME_FAILED；全部跑完写 ind/THEME_DONE。日志 ind/run_theme.log。
优先级低于正常：另一个会话的正式训练先用 CPU。
用法：python -X utf8 run_theme.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
OUT = HERE / "ind"
LOG, DONE, FAILED, STOPPED = OUT / "run_theme.log", OUT / "THEME_DONE", OUT / "THEME_FAILED", OUT / "THEME_STOPPED"
PY = [sys.executable, "-X", "utf8"]
ENV = {**os.environ, "DECADE_DATA": str(OUT)}
LOW = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args: list[str]) -> None:
    log("start " + " ".join(args))
    with LOG.open("a", encoding="utf-8") as f:
        code = subprocess.call(PY + args, cwd=HERE, env=ENV, stdout=f, stderr=subprocess.STDOUT, creationflags=LOW)
    if code:
        FAILED.write_text(" ".join(args) + f" exit {code}\n", encoding="utf-8")
        log(f"FAILED {' '.join(args)} exit {code}")
        raise SystemExit(code)
    log("ok " + " ".join(args))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for marker in (DONE, FAILED, STOPPED):
        marker.unlink(missing_ok=True)
    run(["ind_features.py"])
    extra = [c for c in pq.read_schema(OUT / "ind_features.parquet").names if c not in ("di", "ji")]
    run(["theme_eval.py", "--stage", "q", "--self-check"])
    run(["train_cadence.py", "--cadences", "Q", "Y", "H", "--extra", *extra])
    run(["theme_eval.py", "--stage", "q"])
    import json
    if json.loads((OUT / "theme_eval_q.json").read_text(encoding="utf-8"))["stop"]:
        STOPPED.write_text("每季度版主规则配对差的区间上限 < 0，按预先登记不训练每月版\n", encoding="utf-8")
        log("止损：不训练每月版")
    else:
        run(["train_cadence.py", "--cadences", "M", "--extra", *extra])
        run(["theme_eval.py", "--stage", "m"])
        for capital in ("200000", "20000"):
            run(["rule_search.py", "--capital", capital, "--tier", "real", "--cadences", "Y", "H", "Q", "M"])
            run(["rule_select.py", "--capital", capital])
    run(["theme_report.py"])
    DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    log("THEME DONE")


if __name__ == "__main__":
    main()
