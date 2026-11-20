"""P8：LSTM S+F69 的逐只股票解释（积分梯度，PREREG 补充登记十三第 3 条），主环境调度。

测试段每一年（2023 ~ 2026）：用和正式分数相同的训练数据（`final_<年>`）、冻结配置、3 个种子重训并保存权重
（GPU 训练不是逐位可复现的，所以核对重训分数和正式分数 `lstm_SF69__final` 的相关系数），
再在 dl-study 环境里对这一年的全部测试行算积分梯度（`ml/lstm_explain.py`），拼成 `data/ml/explain/lstm_SF69.parquet`，
格式与 `xgb_F69.parquet` 相同。解释项名称：序列通道记作 `seq_<通道>`（60 天归因之和，数值取信号日当天），手工特征用 dense 预处理后的名称。

实际结果：重训分数与正式分数最大差 4e-8（这套训练在同一台机器上可以复现），见 `data/ml/explain/lstm_SF69_checks.json`。

运行：python -X utf8 -m ml.p8_explain_lstm        # 约 30 ~ 60 分钟（GPU）
"""

from __future__ import annotations

import json
import subprocess
import time

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import RESULTS, ROOT, data_dir, register

TOP = 8
CHANNELS = ["ret", "ovn", "intra", "hl", "clv", "ln_amt", "ln_turn", "mf", "big", "limit", "traded"]


def main() -> None:
    from ml.p3_6_lstm import DL_PY, train
    keys = ev.load_keys()
    plan = ev.protocol_a(keys)
    test_pos = np.flatnonzero(plan.final.eval_rows)
    final_sc = pd.read_parquet(data_dir() / "scores" / "lstm_SF69__final.parquet")["score"].to_numpy()
    ch = json.loads((RESULTS / "p3_6_lstm" / "chosen.json").read_text(encoding="utf-8"))["SF"]
    lstm = data_dir() / "lstm"
    parts, checks, t0 = [], [], time.time()
    for st in plan.refits:
        y = st.name
        mine = st.eval_rows[test_pos]
        np.save(lstm / f"final_{y}" / f"explain_mask_{y}.npy", mine)
        if not (lstm / "runs" / f"lstm_SF69__explain_{y}_seed2.pt").exists():
            train("final", "SF", ch["cfg"], "0,1,2", f"lstm_SF69__explain_{y}", epochs=ch["最终阶段 epoch"], data=f"final_{y}", save_model=True)
        S = np.load(lstm / "runs" / f"lstm_SF69__explain_{y}.npy").mean(axis=1)[mine]
        r = subprocess.run([DL_PY, "-X", "utf8", str(ROOT / "ml" / "lstm_explain.py"), "--year", y], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-2000:])
        print(r.stdout.strip(), flush=True)
        e = lstm / f"explain_{y}"
        score, order, vals, xs = (np.load(e / f"{k}.npy") for k in ("score", "order", "vals", "xs"))
        names = ["seq_" + c for c in CHANNELS] + json.loads((lstm / f"final_{y}" / "meta.json").read_text(encoding="utf-8"))["f_names"]
        corr = float(np.corrcoef(S, final_sc[mine])[0, 1])
        same = float(np.max(np.abs(score - S)))
        checks.append({"year": y, "corr_with_official": round(corr, 4), "max_diff_ig_vs_train": same,
                       "gap_median": float(np.median(np.load(e / "gap.npy")))})
        k = keys[st.eval_rows]
        out = pd.DataFrame({"trade_date": k["trade_date"].to_numpy(), "ts_code": k["ts_code"].to_numpy(),
                            "score": score, "base": np.float32(np.load(e / "base_score.npy")[0])})
        for j in range(TOP):
            out[f"f{j + 1}"] = pd.Categorical.from_codes(order[:, j].astype(np.int32), categories=names)
            out[f"x{j + 1}"] = xs[:, j]
            out[f"v{j + 1}"] = vals[:, j]
        parts.append(out)
        print(f"{y}：{len(out):,} 行，重训分数与正式分数相关 {corr:.4f}，{time.time() - t0:.0f}s", flush=True)
    exp_dir = data_dir() / "explain"
    exp_dir.mkdir(exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(exp_dir / "lstm_SF69.parquet", index=False)
    (exp_dir / "lstm_SF69_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1), encoding="utf-8")
    register("P8_explain_lstm", checks=checks, seconds=round(time.time() - t0, 1))
    print(f"写出 {exp_dir / 'lstm_SF69.parquet'}")


if __name__ == "__main__":
    main()
