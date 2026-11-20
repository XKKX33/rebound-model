"""P8：交付用的模型文件（PREREG 补充登记十三第 4 条）。

三个模型（XGBoost F69、随机森林 F69、LSTM S+F69）各用“截至数据最后一天已经卖出、知道结果的全部样本”训练一次，
参数全部沿用冻结记录（和测试段、长期回测相同），3 个种子；连同预处理参数一起存到 `data/ml/deploy/<模型>/`。
这一份没有样本外可评估，不产生任何评估数字。打分用 `python -m ml.predict`。

保存的内容：
- `xgb_F69/`：`booster_seed{0,1,2}.json`（xgboost 自己的格式，跨版本可读）、`preprocess.joblib`、`meta.json`（树数等）；
- `rf_F69/`：`forests.joblib`（scikit-learn 对象，读的时候 scikit-learn 版本要相同，见 meta）、`preprocess.joblib`、`meta.json`；
- `lstm_SF69/`：`net_seed{0,1,2}.pt`（PyTorch 权重）、`preprocess.joblib`（dense 预处理）、`meta.json`（网络结构）；
- 每个目录还有 `check.parquet`：训练完立刻在内存里给最后 5 个交易日打的分，`ml.predict --check` 用它核对“从文件读回来再打分”结果相同。

运行：python -X utf8 -m ml.p8_deploy           # 约 20 分钟（随机森林最慢；LSTM 用 GPU）
以后有了新数据，重建面板后再跑一次就是“用最新数据重训”。
"""

from __future__ import annotations

import json
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import RESULTS, data_dir, register

MODELS = ("xgb_F69", "rf_F69", "lstm_SF69")


def deploy_dir():
    d = data_dir() / "deploy"
    d.mkdir(exist_ok=True)
    return d


def _versions() -> dict:
    import sklearn
    import xgboost
    return {"xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}


def main() -> None:
    from ml.p7_history import make_model
    from ml.preprocess import Preprocessor
    from ml.train_eval import load
    t0 = time.time()
    keys, x = load("F69")
    day = keys["di"].to_numpy()
    last = int(day.max())
    fit = ev.fit_rows(keys, last + 1)
    chk = day > last - 5
    y = ev.train_labels(keys, fit)
    base = {"trained_on": f"{keys.loc[fit, 'trade_date'].min().date()} ~ {keys.loc[fit, 'trade_date'].max().date()} 的信号日（卖出日不晚于 "
                          f"{keys['trade_date'].max().date()}）", "fit_rows": int(fit.sum()), "data_last_day": str(keys["trade_date"].max().date()),
            "features": list(x.columns), "versions": _versions(), "prereg": "补充登记十三第 4 条"}
    k_chk = keys.loc[chk, ["trade_date", "ts_code"]].reset_index(drop=True)
    for mid in ("xgb_F69", "rf_F69"):
        out = deploy_dir() / mid
        if (out / "meta.json").exists():
            continue
        out.mkdir(exist_ok=True)
        m = make_model(mid)
        pre = Preprocessor(m.preprocess).fit(x[fit], day[fit])
        X_fit, names = pre.transform(x[fit], day[fit])
        m.fit(X_fit, y)
        del X_fit
        X_chk, _ = pre.transform(x[chk], day[chk])
        k_chk.assign(score=m.predict(X_chk)).to_parquet(out / "check.parquet", index=False)
        joblib.dump(pre, out / "preprocess.joblib")
        meta = {**base, "model_id": mid, "inputs": names, "info": m.info()}
        if mid == "xgb_F69":
            for s, b in zip(m.seeds, m.boosters):
                b.save_model(str(out / f"booster_seed{s}.json"))
            meta["best_rounds"] = m.best_rounds
        else:
            joblib.dump(m.forests, out / "forests.joblib", compress=3)
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"{mid}：训练 {int(fit.sum()):,} 行，{time.time() - t0:.0f}s", flush=True)
    del x
    out = deploy_dir() / "lstm_SF69"
    if not (out / "meta.json").exists():
        from ml.lstm_prep import prepare
        from ml.p3_6_lstm import ROUND2, train
        ch = json.loads((RESULTS / "p3_6_lstm" / "chosen.json").read_text(encoding="utf-8"))["SF"]
        meta_p = prepare("deploy")
        if meta_p["fit_rows"] != int(fit.sum()) or meta_p["eval_rows"] != int(chk.sum()):
            raise RuntimeError("LSTM 准备的行数和树模型不一致")
        S, info = train("final", "SF", ch["cfg"], "0,1,2", "lstm_SF69__deploy", epochs=ch["最终阶段 epoch"], data="deploy", save_model=True)
        out.mkdir(exist_ok=True)
        runs = data_dir() / "lstm" / "runs"
        for s in (0, 1, 2):
            shutil.copy(runs / f"lstm_SF69__deploy_seed{s}.pt", out / f"net_seed{s}.pt")
        shutil.copy(data_dir() / "lstm" / "deploy" / "preprocess.joblib", out / "preprocess.joblib")
        k_chk.assign(score=S.mean(axis=1)).to_parquet(out / "check.parquet", index=False)
        meta = {**base, "model_id": "lstm_SF69", "inputs": meta_p["f_names"], "cfg": {**ch["cfg"], **ROUND2},
                "epochs": ch["最终阶段 epoch"], "window": 60, "n_channels": 11, "torch": info.get("torch"), "gpu": info.get("gpu")}
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        shutil.rmtree(data_dir() / "lstm" / "deploy")
        print(f"lstm_SF69：训练 {meta_p['fit_rows']:,} 行，{time.time() - t0:.0f}s", flush=True)
    register("P8_deploy", seconds=round(time.time() - t0, 1), fit_rows=int(fit.sum()))


if __name__ == "__main__":
    main()
