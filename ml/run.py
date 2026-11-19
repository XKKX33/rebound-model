"""任务 6.5 一键复现：按顺序调用各步脚本（每步都是独立的模块，也可以单独运行）。

    python -X utf8 -m ml.run panel        # P1.1 ~ 1.3：数据质量、主面板、探索性分析               约 5 分钟
    python -X utf8 -m ml.run features     # P1.4 ~ 1.6、P2：新特征、预处理检查、序列数据、协议检查   约 10 分钟
    python -X utf8 -m ml.run train        # P3.1 ~ 3.10：参照、复现、三个模型调参、任务形式、集成（验证段）约 4 ~ 5 小时（含 GPU）
    python -X utf8 -m ml.run evaluate     # P3.11 + P4：测试段（按年重训）、排行榜、单因子、消融、解释性 约 5 ~ 6 小时
    python -X utf8 -m ml.run backtest     # P5：组合回测                                             约 2 分钟
    python -X utf8 -m ml.run history      # P7：2015 ~ 2026 长期回测（补充分析）                     约 1.5 小时
    python -X utf8 -m ml.run deliver      # P8：交付范围的三个模型（LSTM 长期回测、解释、模型文件、交付版结果）约 1.5 小时
    python -X utf8 -m ml.run report       # P6.3：报告用图                                           约 1 分钟
    python -X utf8 -m ml.run test         # 单元测试                                                 约 10 秒
    python -X utf8 -m ml.run all          # 以上全部，按顺序
    python -X utf8 -m ml.run all --dry-run   # 只列出要执行的命令，不运行

环境：主环境 `D:\\Anaconda\\python.exe`（pandas、pyarrow、duckdb、xgboost、scikit-learn、matplotlib）；
LSTM 相关步骤自动调用 dl-study 环境 `D:\\Anaconda\\envs\\dl-study\\python.exe`（torch + CUDA）。见 `ml/ENV.md`。

注意：
- 调参脚本可续跑：已有 `ml/results/<步骤>/tuning_*.csv` 的组会跳过。要完全从头重跑，先删掉对应的结果目录。
- `evaluate` 会跑测试段：按 PREREG，只有在 `ml/PREREG.md` 写了冻结记录之后才允许（`p3_11_run_final` 会检查）；
  已有 `ml/results/p3_models/<模型>/final.json` 的模型会跳过。
- 一次只跑一步重活：测试段第一次运行时，几个进程同时读写大文件出现过偶发错误（PREREG“冻结后的运行记录”）。
"""

from __future__ import annotations

import subprocess
import sys
import time

from ml.common import ROOT

PY = sys.executable
DL_PY = r"D:\Anaconda\envs\dl-study\python.exe"


def _m(module: str, *args: str) -> list[str]:
    return [PY, "-X", "utf8", "-W", "ignore", "-m", module, *args]


STEPS: dict[str, list[tuple[str, list[str]]]] = {
    "panel": [
        ("1.1 数据质量", _m("ml.p1_1_data_quality")),
        ("1.2 主面板", _m("ml.p1_2_panel", "--rebuild")),
        ("1.3 探索性分析", _m("ml.p1_3_eda")),
    ],
    "features": [
        ("1.4 新特征", _m("ml.p1_4_build_features", "--rebuild")),
        ("1.5 预处理检查", _m("ml.p1_5_preprocess_check")),
        ("1.6 序列数据", _m("ml.p1_6_build_seq", "--rebuild")),
        ("1.6 LSTM 计算量摸底（GPU）", [DL_PY, "-X", "utf8", "ml/lstm_probe.py"]),
        ("P2 协议检查", _m("ml.p2_protocol_check")),
    ],
    "train": [
        ("3.1 ~ 3.3 参照与复现", _m("ml.train_eval", "random", "rev_r5", "xgb_repro_F28")),
        ("3.1 ~ 3.3 报告", _m("ml.p3_baselines")),
        ("3.4 随机森林", _m("ml.p3_4_random_forest")),
        ("3.5 XGBoost 调参版", _m("ml.p3_5_xgboost")),
        ("3.6 LSTM（GPU）", _m("ml.p3_6_lstm")),
        ("3.9 任务形式对比", _m("ml.p3_9_task_form")),
        ("3.10 简单集成", _m("ml.p3_10_ensemble")),
    ],
    "evaluate": [
        ("3.11 测试段（按年重训）+ 排行榜", _m("ml.p3_11_run_final")),
        ("4.1 单因子检验", _m("ml.p4_1_single_factor")),
        ("4.2 按组消融", _m("ml.p4_2_ablation")),
        ("4.3 解释性（树模型）", _m("ml.p4_3_explain")),
        ("4.3 LSTM 积分梯度（GPU）", [DL_PY, "-X", "utf8", "ml/lstm_ig.py"]),
        ("4.3 报告", _m("ml.p4_3_explain", "--report")),
        ("4.4 重要性的时间稳定性", _m("ml.p4_4_shap_stability")),
    ],
    "backtest": [
        ("P5 组合回测", _m("ml.p5_backtest")),
        ("风格归因（补充）", _m("ml.p5_style")),
    ],
    "history": [
        ("P7 长期回测：逐年重训 2015 ~ 2026", _m("ml.p7_history", "train")),
        ("P7 长期回测：回测与分析", _m("ml.p7_history", "analyze")),
        ("P7 补充：换起点", _m("ml.p7_starts")),
    ],
    "deliver": [
        ("P8 LSTM 长期回测：逐年重训（GPU）", _m("ml.p7_history_lstm")),
        ("P8 随机森林解释（路径分解）", _m("ml.p8_explain_rf")),
        ("P8 LSTM 解释（积分梯度，GPU）", _m("ml.p8_explain_lstm")),
        ("P8 交付模型文件", _m("ml.p8_deploy")),
        ("P8 交付模型流程核对", _m("ml.predict", "--check")),
        ("P8 交付版结果与报告", _m("ml.p8_delivery")),
        ("P8 换起点净值曲线与图", _m("ml.p8_starts_nav")),
        ("P8 报告（加入换起点图）", _m("ml.p8_delivery", "--report")),
    ],
    "report": [
        ("6.3 报告用图", _m("ml.p6_figures")),
        ("交接：导出接口数据包", _m("ml.p6_export")),
    ],
    "test": [
        ("单元测试", _m("pytest", "-p", "no:cacheprovider", "-o", "addopts=", "-q", "ml/tests")),
    ],
}
ORDER = ["panel", "features", "train", "evaluate", "backtest", "history", "deliver", "report", "test"]


def main(argv: list[str]) -> None:
    targets = [a for a in argv if not a.startswith("--")]
    dry = "--dry-run" in argv
    if not targets or any(t not in STEPS and t != "all" for t in targets):
        raise SystemExit(__doc__)
    names = ORDER if "all" in targets else targets
    t_all = time.time()
    for name in names:
        for label, cmd in STEPS[name]:
            print(f"\n=== [{name}] {label}：{' '.join(cmd[1:] if cmd[0] == PY else cmd)}", flush=True)
            if dry:
                continue
            t0 = time.time()
            r = subprocess.run(cmd, cwd=ROOT)
            if r.returncode != 0:
                raise SystemExit(f"[{name}] {label} 失败（退出码 {r.returncode}），停止。修好后可以从这一步重新开始：python -m ml.run {name}")
            print(f"=== [{name}] {label} 完成，{time.time() - t0:.0f}s", flush=True)
    print(f"\n全部完成，{time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
