"""任务 3.1 统一入口：按名字跑任意一个模型 → 训练 → 打分 → 评估 → 落盘 → 登记。

运行：
    python -X utf8 -m ml.train_eval <model_id> [<model_id> ...]          # 选择阶段：训练段 → 验证段
    python -X utf8 -m ml.train_eval <model_id> --final                   # 测试段：按年重训（所有选择冻结后才用）

每个模型的产物：
- `data/ml/scores/<model_id>__<阶段>.parquet`：`trade_date, ts_code, score`（各种子平均）+ 每个种子的分数 `s0, s1, ...`
- `ml/results/p3_models/<model_id>/<阶段>.json`：配置、耗时、各段指标、各种子 IC
- `ml/results/p3_models/<model_id>/<阶段>_daily.csv`：逐日 IC、Top-K 超额、多空（模型两两比较用）
- `ml/results/registry.jsonl` 追加一条

模型配置在本文件的 `CONFIGS` 里登记：model_id → (构造函数, 特征集)。
"""

from __future__ import annotations

import json
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import data_dir, register, result_dir, rss_gb
from ml.dataset import F28, load_features
from ml.models import XGB, Model, RandomScore, SingleFactor
from ml.preprocess import Preprocessor

STAGES = ("select", "final")
STAGE_NAMES = {"select": "验证段", "final": "测试段（按年重训）", "final_once": "测试段（只训练一次，对照）"}

#: model_id → (构造函数, 特征集)。构造函数接收阶段（"select" / "final"），有的模型两个阶段设置不同（如早停树数）。
CONFIGS: dict[str, tuple[Callable[[str], Model], str]] = {
    # 3.2 参照
    "random": (lambda stage: RandomScore(), "F28"),
    "rev_r5": (lambda stage: SingleFactor(column=F28.index("r5"), sign=-1.0, label="-r5", name="rev_r5"), "F28"),
    # 3.3 XGBoost 复现基准：仓库原参数、28 个原特征、3 个种子
    "xgb_repro_F28": (lambda stage: XGB(name="xgb_repro"), "F28"),
}


def _rf(fs: str) -> Model:
    """3.4 随机森林：参数取调参选定的（ml/results/p3_4_random_forest/chosen.json），3 个种子。"""
    from ml.models import RandomForest
    from ml.p3_4_random_forest import chosen_params
    return RandomForest(params=chosen_params(fs), seeds=(0, 1, 2), name=f"rf_{fs}")


CONFIGS.update({f"rf_{fs}": ((lambda stage, fs=fs: _rf(fs)), fs) for fs in ("F28", "F69")})


def _xgb(fs: str, stage: str) -> Model:
    """3.5 XGBoost 调参版：选择阶段各种子早停；最终阶段用早停树数的平均（ml/results/p3_5_xgboost/chosen.json）。"""
    from ml.p3_5_xgboost import model_for
    return model_for(fs, stage)


CONFIGS.update({f"xgb_{fs}": ((lambda stage, fs=fs: _xgb(fs, stage)), fs) for fs in ("F28", "F69")})


def _xgb_form(obj: str, stage: str) -> Model:
    """3.9 任务形式对比：xgb_F69 的参数，目标换成二分类 / 排序（ml/results/p3_9_task_form/chosen.json）。"""
    from ml.p3_9_task_form import model_for
    return model_for(obj, stage)


CONFIGS.update({f"xgb_{obj}_F69": ((lambda stage, obj=obj: _xgb_form(obj, stage)), "F69") for obj in ("clf", "rank")})


def load(feature_set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ev.load_keys()
    x = load_features(feature_set)
    if len(x) != len(keys):
        raise RuntimeError("特征和主面板行数不一致")
    return keys, x


def run(model_id: str, stage: str = "select", *, model: Model | None = None, feature_set: str | None = None,
        keys: pd.DataFrame | None = None, x: pd.DataFrame | None = None, save: bool = True) -> dict:
    """跑一个模型的一个阶段。model / feature_set 不给就按 CONFIGS 取；keys / x 可以复用已读好的数据。"""
    if stage not in STAGES:
        raise ValueError(f"阶段只能是 {STAGES}")
    make = None
    if model is None:
        make, feature_set = CONFIGS[model_id]
        model = make(stage)
    if keys is None or x is None:
        keys, x = load(feature_set)
    plan = ev.protocol_a(keys)
    if stage == "final":
        return run_final(model_id, plan, model, make, feature_set, keys, x, save)
    st = plan.select
    day = keys["di"].to_numpy()
    fit, evr = st.fit_rows, st.eval_rows

    t0 = time.time()
    y = ev.train_labels(keys, fit)
    pre = Preprocessor(model.preprocess).fit(x[fit], day[fit])
    X_fit, names = pre.transform(x[fit], day[fit])
    X_eval, _ = pre.transform(x[evr], day[evr])
    prep_secs = time.time() - t0

    t0 = time.time()
    kw = {}
    if stage == "select":                         # 早停只在选择阶段用验证段；最终阶段不看任何评估段数据
        kw = {"X_valid": X_eval, "y_valid": keys.loc[evr, "label_pct"].to_numpy(np.float32), "day_valid": day[evr]}
    if getattr(model, "objective", None) == "rank":
        kw["day_train"] = day[fit]
    model.fit(X_fit, y, **kw)
    fit_secs = time.time() - t0
    del X_fit

    t0 = time.time()
    S = model.predict_seeds(X_eval)
    score = S.mean(axis=1)
    pred_secs = time.time() - t0

    meta = {"feature_set": feature_set, "n_features": len(names), "model": model.info(), "complexity": model.complexity,
            "seconds": {"prep": round(prep_secs, 1), "fit": round(fit_secs, 1), "predict": round(pred_secs, 1)}}
    return score_and_save(model_id, stage, st, keys, S, meta, objective=getattr(model, "objective", "reg"), save=save)


KEY_CHECK_COLS = ["trade_date", "ts_code", "di", "ret_o1o5", "exit_di", "label_pct", "excess_o1o5"]


def keys_fingerprint(keys: pd.DataFrame) -> dict[str, int]:
    """keys 各关键列的指纹（逐行哈希之和）。3.11 第一次运行时出现过两次无法复现的错误，像是内存里的 keys 被改坏了；
    测试段运行中反复核对它，一旦变了就停下，绝不保存可疑的结果。"""
    return {c: int(pd.util.hash_pandas_object(keys[c], index=False).sum()) for c in KEY_CHECK_COLS if c in keys}


def check_keys(keys: pd.DataFrame, ref: dict[str, int], where: str) -> None:
    now = keys_fingerprint(keys)
    bad = [c for c in ref if now.get(c) != ref[c]]
    if bad:
        raise RuntimeError(f"{where}：keys 的列 {bad} 和开始时不一致（内存里的数据被改动或损坏），停止，不保存结果")


def run_final(model_id: str, plan: ev.ProtocolA, model: Model, make, feature_set: str, keys: pd.DataFrame, x: pd.DataFrame,
              save: bool = True) -> dict:
    """测试段：按年重训（PREREG 补充登记十）。

    每年第一个交易日用此前全部已卖出的行重训一次（参数、树数都固定，不早停，不看任何测试数据），模型给全部测试行打分：
    - 正式分数（阶段 `final`）：每个模型只取它这一年的行，四段拼起来；
    - 对照（阶段 `final_once`）：第一次（2023 年）重训的模型给整个测试段的分数，即“只训练一次”。
    """
    import copy
    ref = keys_fingerprint(keys)
    dates = keys.drop_duplicates("di").set_index("di")["trade_date"]
    day = keys["di"].to_numpy().copy()
    test = plan.final.eval_rows
    test_pos = np.flatnonzero(test)
    S_final, S_once, per_year = None, None, []
    t_all = time.time()
    for k, st in enumerate(plan.refits):
        m = copy.deepcopy(model) if make is None else make("final")       # 每年一个全新的模型
        t0 = time.time()
        fit = st.fit_rows
        y = ev.train_labels(keys, fit)
        pre = Preprocessor(m.preprocess).fit(x[fit], day[fit])
        X_fit, names = pre.transform(x[fit], day[fit])
        X_test, _ = pre.transform(x[test], day[test])
        prep = time.time() - t0
        t0 = time.time()
        kw = {"day_train": day[fit]} if getattr(m, "objective", None) == "rank" else {}
        m.fit(X_fit, y, **kw)
        fit_secs = time.time() - t0
        del X_fit
        t0 = time.time()
        S = m.predict_seeds(X_test)                                      # [全部测试行, 种子]
        pred = time.time() - t0
        if S_final is None:
            S_final = np.full(S.shape, np.nan)
            S_once = S.copy()
        mine = st.eval_rows[test_pos]                                    # 这一年的行在测试行里的位置
        S_final[mine] = S[mine]
        if not np.isfinite(S).all():
            raise RuntimeError(f"{model_id} {st.name} 年重训：分数里有空值或无穷")
        check_keys(keys, ref, f"{model_id} {st.name} 年重训之后")
        per_year.append({"year": st.name, "fit_date": str(dates[st.fit_di].date()),
                         "fit_rows": int(fit.sum()), "eval_rows": int(mine.sum()), "model": m.info(),
                         "seconds": {"prep": round(prep, 1), "fit": round(fit_secs, 1), "predict": round(pred, 1)}})
        print(f"  [{model_id}] {st.name} 年重训：训练 {int(fit.sum()):,} 行，{fit_secs:.0f}s", flush=True)
    if np.isnan(S_final).any():
        raise RuntimeError("按年重训的分数没有覆盖全部测试行")
    first = per_year[0]
    base = {"feature_set": feature_set, "n_features": len(names), "complexity": model.complexity}
    meta = {**base, "protocol": "测试段按年重训（PREREG 补充登记十）", "model": first["model"], "refits": per_year,
            "seconds": {"fit": round(sum(r["seconds"]["fit"] for r in per_year), 1), "total": round(time.time() - t_all, 1)}}
    meta_once = {**base, "protocol": "只训练一次（2023-01 重训的模型给整个测试段打分；对照，补充登记十）",
                 "model": first["model"], "seconds": first["seconds"]}
    obj = getattr(model, "objective", "reg")
    check_keys(keys, ref, f"{model_id} 保存之前")
    score_and_save(model_id, "final_once", plan.final, keys, S_once, meta_once, objective=obj, save=save)
    return score_and_save(model_id, "final", plan.final, keys, S_final, meta, objective=obj, save=save)


def _evaluate_twice(model_id, stage, score, k_eval, parts, objective, S, tries: int = 3) -> dict:
    """评估算两遍，两遍完全一致才采用；报错或不一致就重算（最多 tries 次），每次异常都记到 data/ml/debug/anomalies.log。

    原因：3.11 跑测试段时，长时间运行的进程里出现过几次无法复现的计算错误（同样的数据单独重算总是正常）。
    这道检查保证偶发的计算错误不会被悄悄保存。
    """
    dump = data_dir() / "debug"
    kw = {"parts": parts, "auc": objective == "clf", "ndcg": objective == "rank"}
    for k in range(tries):
        try:
            a = ev.evaluate(score, k_eval, **kw)
            b = ev.evaluate(score, k_eval, **kw)
            same = (json.dumps(a["summary"], sort_keys=True) == json.dumps(b["summary"], sort_keys=True)
                    and a["daily"].equals(b["daily"]))
            if same:
                return a
            msg = "两次评估结果不一致"
        except Exception as e:                 # noqa: BLE001
            msg = f"评估报错：{e!r}"
        dump.mkdir(exist_ok=True)
        np.save(dump / f"{model_id}__{stage}_S.npy", S)
        with open(dump / "anomalies.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {model_id} {stage} 第 {k + 1} 次：{msg}\n")
        print(f"  [{model_id} {stage}] 第 {k + 1} 次评估异常（{msg}），重算", flush=True)
    raise RuntimeError(f"{model_id} {stage}：评估连续 {tries} 次异常，停止，不保存")


def score_and_save(model_id: str, stage: str, st: ev.Stage, keys: pd.DataFrame, S: np.ndarray, meta: dict, *,
                   objective: str = "reg", save: bool = True) -> dict:
    """评估一组分数（[评估行, 种子]，与 keys[st.eval_rows] 逐行对齐）并落盘。

    `run` 训练完调用它；在别的环境里训练的模型（LSTM，dl-study 环境）把分数读回来后也调用它，
    保证所有模型用同一套评估和同样的文件格式。
    """
    fit, evr = st.fit_rows, st.eval_rows
    S = np.asarray(S, dtype=np.float64)
    if S.ndim == 1:
        S = S[:, None]
    score = S.mean(axis=1)
    k_eval = keys[evr].reset_index(drop=True)
    parts = {name: m[evr] for name, m in st.parts.items()} if st.parts else None
    res = _evaluate_twice(model_id, stage, score, k_eval, parts, objective, S)
    f = ev._frame(score, k_eval)
    seed_ic = []
    for j in range(S.shape[1]):
        f["score"] = ev._frame(S[:, j], k_eval)["score"].to_numpy()
        seed_ic.append(float(ev.daily_rank_ic(f).mean()))

    out = {"model_id": model_id, "stage": stage, **meta,
           "fit_rows": int(fit.sum()), "eval_rows": int(evr.sum()),
           "mem_gb": round(rss_gb(), 2), "seed_ic": seed_ic, "seed_spread": ev.seed_spread(seed_ic),
           "summary": res["summary"], "groups": {k: v.round(6).tolist() for k, v in res["groups"].items()}}
    if save:
        sc = pd.DataFrame({"trade_date": k_eval["trade_date"], "ts_code": k_eval["ts_code"], "score": score.astype(np.float32)})
        for j in range(S.shape[1]):
            sc[f"s{j}"] = S[:, j].astype(np.float32)
        (data_dir() / "scores").mkdir(exist_ok=True)
        sc.to_parquet(data_dir() / "scores" / f"{model_id}__{stage}.parquet", index=False)
        d = result_dir(f"p3_models/{model_id}")
        res["daily"].to_csv(d / f"{stage}_daily.csv", encoding="utf-8-sig")
        (d / f"{stage}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        register("3_train_eval", **{k: v for k, v in out.items() if k != "groups"})
    out["daily"] = res["daily"]
    out["scores"] = S
    return out


def main(argv: list[str]) -> None:
    stage = "final" if "--final" in argv else "select"
    ids = [a for a in argv if not a.startswith("--")]
    unknown = [i for i in ids if i not in CONFIGS]
    if unknown or not ids:
        raise SystemExit(f"用法：python -m ml.train_eval <model_id> ... [--final]；可选：{', '.join(CONFIGS)}；未知：{unknown}")
    cache: dict[str, tuple] = {}
    for model_id in ids:
        fs = CONFIGS[model_id][1]
        if fs not in cache:
            cache.clear()
            cache[fs] = load(fs)
        keys, x = cache[fs]
        r = run(model_id, stage, keys=keys, x=x)
        s = r["summary"]["全部" if stage == "select" else "全段"]
        print(f"{model_id} [{STAGE_NAMES[stage]}] IC {s['ic_mean']:.4f}  NW t {s['nw_t']:.1f}  "
              f"前6 {s['top6'] * 100:+.2f}%  前30 {s['top30'] * 100:+.2f}%  种子 IC {np.round(r['seed_ic'], 4).tolist()}  "
              f"训练 {r['seconds']['fit']:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
