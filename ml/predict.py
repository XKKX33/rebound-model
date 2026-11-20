"""用交付的模型文件给某一天打分、出名单（PREREG 补充登记十三第 4 条）。

前提：`data/ml/deploy/` 里有模型文件（`python -m ml.p8_deploy` 训练，或从数据包解压），
并且主面板、特征、序列数据里已经有要打分的那一天。研究库更新到新的交易日后，先加 `--rebuild` 重建这三样（约 15 分钟，需要研究库）：
它会依次运行 1.2 主面板、1.4 新特征、1.6 序列数据（这三步也会改写 `ml/results/p1_*` 里的构建记录）。

    python -X utf8 -m ml.predict                          # 数据里最后一天，三个模型，每个模型前 30 名（每个行业最多 2 只）
    python -X utf8 -m ml.predict --date 2026-10-23 --models xgb_F69 --top 6
    python -X utf8 -m ml.predict --rebuild                # 研究库更新后：重建面板、特征、序列，再打最后一天
    python -X utf8 -m ml.predict --check                  # 流程核对：从文件读回模型再打分，和训练时内存里的分数比

输出：`data/ml/deploy/out/scores_<日期>.csv`（全池：代码、名称、行业、每个模型的分数、名次、百分位）
和 `picks_<日期>.csv`（每个模型的名单）。信号日 T 收盘后出名单，按仓库口径 T+1 开盘买入。
LSTM 打分会调用 dl-study 环境（PyTorch）；没有的话用 `--models xgb_F69,rf_F69`。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import ROOT, data_dir

DL_PY = r"D:\Anaconda\envs\dl-study\python.exe"
MODELS = ("xgb_F69", "rf_F69", "lstm_SF69")
NAMES = {"xgb_F69": "XGBoost F69", "rf_F69": "随机森林 F69", "lstm_SF69": "LSTM S+F69"}


def deploy_dir() -> Path:
    return data_dir() / "deploy"


def score_rows(mid: str, keys: pd.DataFrame, x: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """用交付模型给 keys[rows] 打分。x 是 F69 特征（和 keys 同行序）。"""
    d = deploy_dir() / mid
    if not (d / "meta.json").exists():
        raise SystemExit(f"没有 {mid} 的模型文件（{d}）：先运行 python -m ml.p8_deploy，或解压数据包")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    pre = joblib.load(d / "preprocess.joblib")
    day = keys["di"].to_numpy()[rows]
    X, names = pre.transform(x[rows], day)
    if list(names) != meta["inputs"]:
        raise RuntimeError(f"{mid}：特征和训练时不一致")
    if mid == "xgb_F69":
        import xgboost as xgb
        dm = xgb.DMatrix(X)
        out = []
        for s, n in enumerate(meta["best_rounds"]):
            b = xgb.Booster()
            b.load_model(str(d / f"booster_seed{s}.json"))
            out.append(b.predict(dm, iteration_range=(0, n)))
        return np.mean(out, axis=0)
    if mid == "rf_F69":
        return np.mean([f.predict(X) for f in joblib.load(d / "forests.joblib")], axis=0)
    from ml.seq_data import load_X
    _, _, _, panel_j = load_X()
    with tempfile.TemporaryDirectory() as tmp:
        np.save(Path(tmp) / "j.npy", panel_j[rows])
        np.save(Path(tmp) / "d.npy", day.astype(np.int64))
        np.save(Path(tmp) / "f.npy", X.astype(np.float32))
        r = subprocess.run([DL_PY, "-X", "utf8", str(ROOT / "ml" / "lstm_predict.py"), tmp], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"LSTM 打分失败：{r.stderr[-2000:]}")
        return np.load(Path(tmp) / "score.npy")


def rebuild() -> None:
    for mod, args in (("ml.p1_2_panel", ["--rebuild"]), ("ml.p1_4_build_features", ["--rebuild"]), ("ml.p1_6_build_seq", ["--rebuild"])):
        print(f"重建：{mod}", flush=True)
        subprocess.run([sys.executable, "-X", "utf8", "-m", mod, *args], cwd=ROOT, check=True)


def check(models) -> None:
    from ml.train_eval import load
    keys, x = load("F69")
    for mid in models:
        ref = pd.read_parquet(deploy_dir() / mid / "check.parquet")
        idx = keys.reset_index().set_index(["trade_date", "ts_code"]).loc[pd.MultiIndex.from_frame(ref[["trade_date", "ts_code"]]), "index"]
        rows = np.zeros(len(keys), bool)
        rows[idx.to_numpy()] = True
        got = pd.Series(score_rows(mid, keys, x, rows), index=pd.MultiIndex.from_frame(keys.loc[rows, ["trade_date", "ts_code"]]))
        diff = np.abs(got.reindex(pd.MultiIndex.from_frame(ref[["trade_date", "ts_code"]])).to_numpy() - ref["score"].to_numpy())
        print(f"{mid}：{len(ref):,} 行，从文件读回再打分与训练时的分数最大差 {diff.max():.2e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None, help="信号日（默认：数据里最后一天）")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--max-per-industry", type=int, default=2)
    ap.add_argument("--rebuild", action="store_true", help="先用研究库重建主面板、特征、序列数据")
    ap.add_argument("--check", action="store_true", help="流程核对，不打分")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    if bad := [m for m in models if m not in MODELS]:
        raise SystemExit(f"没有模型 {bad}；可选：{', '.join(MODELS)}")
    if a.rebuild:
        rebuild()
    if a.check:
        check(models)
        return
    from ml.api import _stocks
    from ml.train_eval import load
    from rebound.select import dedup_top
    keys, x = load("F69")
    day = pd.Timestamp(a.date) if a.date else keys["trade_date"].max()
    rows = (keys["trade_date"] == day).to_numpy()
    if not rows.any():
        raise SystemExit(f"{day.date()} 不在数据里（数据范围 {keys['trade_date'].min().date()} ~ {keys['trade_date'].max().date()}）；"
                         "新的交易日要先更新研究库再加 --rebuild")
    info = _stocks()
    out = keys.loc[rows, ["ts_code"]].reset_index(drop=True)
    out["name"] = info["name"].reindex(out["ts_code"]).to_numpy()
    out["industry"] = info["industry"].reindex(out["ts_code"]).fillna("").to_numpy()
    picks = []
    for mid in models:
        s = score_rows(mid, keys, x, rows)
        out[f"{mid}_score"] = s
        rank = pd.Series(-s).rank(method="first").astype(int).to_numpy()
        out[f"{mid}_rank"] = rank
        out[f"{mid}_pct"] = 1 - (rank - 1) / len(rank)
        srt = out.sort_values(f"{mid}_rank").reset_index(drop=True)
        keep = dedup_top(srt["industry"].tolist(), max_per_industry=a.max_per_industry, top_n=a.top)
        p = srt.iloc[keep][["ts_code", "name", "industry", f"{mid}_score"]].rename(columns={f"{mid}_score": "score"})
        picks.append(p.assign(model_id=mid, 名次=np.arange(1, len(p) + 1)))
        print(f"\n{NAMES[mid]}：{day.date()} 前 {a.top} 名（每个行业最多 {a.max_per_industry} 只）\n"
              f"{p.head(10).to_string(index=False)}{'  …' if len(p) > 10 else ''}", flush=True)
    od = deploy_dir() / "out"
    od.mkdir(parents=True, exist_ok=True)
    out.to_csv(od / f"scores_{day.date()}.csv", index=False, encoding="utf-8-sig")
    pd.concat(picks, ignore_index=True).to_csv(od / f"picks_{day.date()}.csv", index=False, encoding="utf-8-sig")
    print(f"\n写出 {od / f'scores_{day.date()}.csv'}、picks_{day.date()}.csv")


if __name__ == "__main__":
    main()
