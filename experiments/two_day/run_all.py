"""两日反弹模型实验一条龙（结果都在本目录）：
链接输入 → 两个变体各自核对（这里算的 o1o5 标签和十年实验逐行一样；不剔退市整理期时几条规则的模拟和十年实验逐项一样）→
GPU 上依次训练（不限价两日 → 不限价五日重训 → 限 70 元两日 → 不限价三日 → 限 70 元五日重训）→
每个模型训完就跑 150 条规则的 2 万账户模拟（CPU，最多同时两个，内存不够就等）→ 不限价三个模型跑完写中间报告 →
全部跑完挑入围规则，补免五、按当时印花税 → 终版报告 REPORT.md。
训练、模拟任一步失败就停（写 FAILED）；报告失败只记一笔（结果都缓存了，改好 report.py 重跑即可）。全部跑完写 DONE。
优先级低于正常：别的会话的任务先用 CPU。日志 run_all.log、logs/<任务>.log。
用法：python -X utf8 run_all.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import psutil

from horizon import HERE, VARIANTS, link_inputs

PY = [sys.executable, "-X", "utf8"]
LOG, LOGS = HERE / "run_all.log", HERE / "logs"
DONE, FAILED, INTERIM = HERE / "DONE", HERE / "FAILED", HERE / "INTERIM_DONE"
FINALISTS = HERE / "finalists.json"
LOW = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
CPU_SLOTS = 2
#: 已经有模拟在跑时，再起一个要的可用内存（GB）
MEM_GB = 1.5
POLL = 10
#: GPU 上训练的顺序（变体, 买卖时点）
TRAIN = (("nocap", "o1o2"), ("nocap", "o1o5"), ("cap70", "o1o2"), ("nocap", "o1o3"), ("cap70", "o1o5"))
CORE = ("sim-nocap-o1o5-cpu", "sim-nocap-o1o2-cuda", "sim-nocap-o1o5-cuda")


@dataclass
class Job:
    name: str
    args: list[str]
    variant: str | None = None
    deps: tuple[str, ...] = ()
    lane: str = "cpu"   # gpu：按顺序一个一个跑；cpu：最多 CPU_SLOTS 个
    soft: bool = False  # 失败只记一笔，不停（报告）
    state: str = "wait"  # wait / run / ok / failed
    proc: subprocess.Popen | None = None
    out: IO[str] | None = None


def log(msg: str) -> None:
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def plan() -> list[Job]:
    jobs = []
    for v in VARIANTS:
        jobs.append(Job(f"decade-{v}", ["sim_horizon.py", "--model", "o1o5-cpu", "--check-decade"], v))
        jobs.append(Job(f"sim-{v}-o1o5-cpu", ["sim_horizon.py", "--model", "o1o5-cpu"], v, deps=(f"decade-{v}",)))
    checked: set[str] = set()
    for v, cell in TRAIN:
        if v not in checked:  # 标签核对排在这个变体第一次训练之前（GPU 那条线按列表顺序跑）
            jobs.append(Job(f"labels-{v}", ["train_horizon.py", "--check-labels"], v, lane="gpu"))
            checked.add(v)
        jobs.append(Job(f"train-{v}-{cell}", ["train_horizon.py", "--cell", cell, "--device", "cuda"], v,
                        deps=(f"labels-{v}",), lane="gpu"))
        jobs.append(Job(f"sim-{v}-{cell}-cuda", ["sim_horizon.py", "--model", f"{cell}-cuda"], v,
                        deps=(f"train-{v}-{cell}", f"decade-{v}")))
    sims = tuple(j.name for j in jobs if j.name.startswith("sim-"))
    jobs.append(Job("report-interim", ["report.py"], deps=CORE, soft=True))
    jobs.append(Job("report-finalists", ["report.py", "--finalists"], deps=(*sims, "report-interim"), soft=True))
    return jobs


def sensitivity() -> list[Job]:
    """入围规则补两档：免五（没有最低佣金）、印花税按当时税率。报告挑入围时写 finalists.json。"""
    if not FINALISTS.is_file():
        log("没有 finalists.json：报告没写成，跳过免五和按当时印花税（改好 report.py 后重跑 run_all.py）")
        return []
    jobs = []
    for item in json.loads(FINALISTS.read_text(encoding="utf-8")):
        v, model, rules = item["variant"], item["model"], item["rules"]
        for fee, tier in (("free5", "real"), ("min5", "hist")):
            jobs.append(Job(f"sens-{v}-{model}-{fee}-{tier}",
                            ["sim_horizon.py", "--model", model, "--fee", fee, "--tier", tier, "--rules", *rules], v,
                            deps=("report-finalists",)))
    jobs.append(Job("report-final", ["report.py"], deps=tuple(j.name for j in jobs), soft=True))
    return jobs


def ready(job: Job, by_name: dict[str, Job]) -> bool:
    return all(by_name[d].state == "ok" or (by_name[d].state == "failed" and by_name[d].soft) for d in job.deps)


def start(job: Job) -> None:
    env = {k: v for k, v in os.environ.items() if k != "DECADE_DATA"}
    if job.variant:
        env["DECADE_DATA"] = str(HERE / job.variant)
    job.out = (LOGS / f"{job.name}.log").open("a", encoding="utf-8")
    job.proc = subprocess.Popen(PY + job.args, cwd=HERE, env=env, stdout=job.out, stderr=subprocess.STDOUT,
                                creationflags=LOW)
    job.state = "run"
    log(f"start {job.name}: {' '.join(job.args)}" + (f" [{job.variant}]" if job.variant else ""))


def launch(jobs: list[Job]) -> None:
    by_name = {j.name: j for j in jobs}
    for lane, slots in (("gpu", 1), ("cpu", CPU_SLOTS)):
        busy = sum(j.lane == lane and j.state == "run" for j in jobs)
        for job in jobs:
            if busy >= slots:
                break
            if job.lane != lane or job.state != "wait" or not ready(job, by_name):
                continue
            if lane == "cpu" and busy and psutil.virtual_memory().available < MEM_GB * 2 ** 30:
                break  # 内存不够：等正在跑的那个
            start(job)
            busy += 1


def stop_all(jobs: list[Job], why: str) -> None:
    for job in jobs:
        if job.state == "run" and job.proc is not None:
            job.proc.kill()
    FAILED.write_text(why + "\n", encoding="utf-8")
    log(f"FAILED {why}")
    raise SystemExit(1)


def poll(jobs: list[Job]) -> None:
    for job in jobs:
        if job.state != "run" or job.proc is None or job.proc.poll() is None:
            continue
        code = job.proc.returncode
        job.out.close()
        job.state = "ok" if code == 0 else "failed"
        if code == 0:
            log(f"ok {job.name}")
            if job.name == "report-interim":
                INTERIM.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
        elif job.soft:
            log(f"报告失败（不影响训练和模拟）：{job.name} exit {code}，见 logs/{job.name}.log")
        else:
            stop_all(jobs, f"{job.name} exit {code}（见 logs/{job.name}.log）")


def main() -> None:
    for marker in (DONE, FAILED, INTERIM):
        marker.unlink(missing_ok=True)
    LOGS.mkdir(exist_ok=True)
    for v in VARIANTS:
        link_inputs(v)
    jobs, added = plan(), False
    log(f"plan: {len(jobs)} jobs")
    while True:
        poll(jobs)
        finalists = next(j for j in jobs if j.name == "report-finalists")
        if not added and finalists.state in ("ok", "failed"):
            jobs += sensitivity() if finalists.state == "ok" else []
            added = True
        if added and all(j.state in ("ok", "failed") for j in jobs):
            break
        launch(jobs)
        time.sleep(POLL)
    bad = [j.name for j in jobs if j.state == "failed"]
    DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + (f" 报告失败：{bad}" if bad else "") + "\n", encoding="utf-8")
    log("DONE" + (f"（报告失败：{bad}）" if bad else ""))


if __name__ == "__main__":
    main()
