"""任务 3.11：所有选择冻结后，一次性跑全部模型的测试段（按年重训，PREREG 补充登记十），再生成总排行榜。

顺序（一次只跑一个，避免内存不够）：
1. 树模型和参照（`train_eval --final`）：先 F28 的，再 F69 的（同一套特征只读一次）；
2. LSTM（`p3_6_lstm --final`，在 dl-study 环境训练）；
3. 两个集成（`p3_10_ensemble --final`，不训练）；
4. 总排行榜（`p3_11_leaderboard`）。
可续跑：已经有 `ml/results/p3_models/<model_id>/final.json` 的模型跳过。必须先在 PREREG 写好冻结记录，否则拒绝运行。

运行：python -X utf8 -m ml.p3_11_run_final
"""

from __future__ import annotations

import subprocess
import sys
import time

from ml.common import RESULTS, ROOT

TREES = {"F28": ["random", "rev_r5", "xgb_repro_F28", "rf_F28", "xgb_F28"],
         "F69": ["rf_F69", "xgb_F69", "xgb_rank_F69", "xgb_clf_F69"]}
LSTM = ["lstm_S", "lstm_SF69"]


def _done(mid: str) -> bool:
    return (RESULTS / "p3_models" / mid / "final.json").exists()


def _log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    (RESULTS / "p3_11_run_final.log").open("a", encoding="utf-8").write(line + "\n")


def main() -> None:
    if "## 冻结记录" not in (ROOT / "ml" / "PREREG.md").read_text(encoding="utf-8"):
        raise SystemExit("PREREG.md 里还没有冻结记录，不能跑测试段")
    from ml import train_eval as te
    for fs, ids in TREES.items():
        todo = [m for m in ids if not _done(m)]
        if not todo:
            continue
        keys, x = te.load(fs)
        ref = te.keys_fingerprint(keys)
        fresh = te.keys_fingerprint(te.ev.load_keys())          # 再独立读一次核对：读进来的数据本身完整
        if ref != fresh:
            raise SystemExit(f"{fs}：两次读到的 keys 不一致，先排查数据文件")
        for mid in todo:
            te.check_keys(keys, ref, f"{mid} 开始之前")
            t0 = time.time()
            r = te.run(mid, "final", keys=keys, x=x)
            _log(f"{mid}：测试段（按年重训）完成，{time.time() - t0:.0f}s，种子数 {len(r['seed_ic'])}")
        del keys, x
    if not all(_done(m) for m in LSTM):
        t0 = time.time()
        subprocess.run([sys.executable, "-X", "utf8", "-W", "ignore", "-m", "ml.p3_6_lstm", "--final"], cwd=ROOT, check=True)
        _log(f"LSTM：测试段完成，{time.time() - t0:.0f}s")
    if not all(_done(m) for m in ("ens_top3", "ens_family")):
        subprocess.run([sys.executable, "-X", "utf8", "-W", "ignore", "-m", "ml.p3_10_ensemble", "--final"], cwd=ROOT, check=True)
        _log("集成：测试段完成")
    subprocess.run([sys.executable, "-X", "utf8", "-W", "ignore", "-m", "ml.p3_11_leaderboard"], cwd=ROOT, check=True)
    _log("总排行榜已生成：ml/results/leaderboard.md")


if __name__ == "__main__":
    main()
