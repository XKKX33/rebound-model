"""P7 长期回测补 LSTM S+F69（PREREG 补充登记十三第 2 条，设置同补充登记十一）。

每年第一个交易日（2015 ~ 2026）用此前已卖出的全部样本重训（冻结配置、固定 epoch、3 个种子），给当年准入池全部行打分。
主环境准备数据（`ml/lstm_prep.py hist_<年>`），dl-study 环境训练（`ml/p3_6_lstm.train`）；每年训练完删掉这一年的训练数据，省磁盘。
可续跑：每年的分数存在 data/ml/history/lstm_parts/，有了就跳过。最后拼成 data/ml/history/scores_lstm_SF69.parquet（格式同树模型）。

运行：python -X utf8 -m ml.p7_history_lstm        # 约 1 小时
"""

from __future__ import annotations

import json
import shutil
import time

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.common import RESULTS, data_dir, register
from ml.p7_history import START_YEAR, _log, hist_dir

MID = "lstm_SF69"


def main() -> None:
    from ml.lstm_prep import prepare
    from ml.p3_6_lstm import train
    from ml.train_eval import check_keys, keys_fingerprint
    keys = ev.load_keys()
    ref = keys_fingerprint(keys)
    ch = json.loads((RESULTS / "p3_6_lstm" / "chosen.json").read_text(encoding="utf-8"))["SF"]
    parts_dir = hist_dir() / "lstm_parts"
    parts_dir.mkdir(exist_ok=True)
    t_all = time.time()
    for st in ev.history_refits(keys, START_YEAR):
        part = parts_dir / f"{st.name}.parquet"
        if part.exists():
            continue
        t0 = time.time()
        stage = f"hist_{st.name}"
        meta = prepare(stage)
        if meta["eval_rows"] != int(st.eval_rows.sum()) or meta["fit_rows"] != int(st.fit_rows.sum()):
            raise RuntimeError(f"{stage}：准备的行数和划分不一致")
        S, info = train("final", "SF", ch["cfg"], "0,1,2", f"{MID}__{stage}", epochs=ch["最终阶段 epoch"], data=stage)
        if S.shape[0] != meta["eval_rows"] or not np.isfinite(S).all():
            raise RuntimeError(f"{stage}：分数行数不对或有空值")
        check_keys(keys, ref, f"{MID} {stage} 之后")
        k = keys.loc[st.eval_rows, ["trade_date", "ts_code"]].reset_index(drop=True)
        k.assign(score=S.mean(axis=1).astype(np.float32), **{f"s{j}": S[:, j].astype(np.float32) for j in range(S.shape[1])}).to_parquet(part, index=False)
        shutil.rmtree(data_dir() / "lstm" / stage)
        _log(f"{MID} {st.name} 年：训练 {meta['fit_rows']:,} 行，打分 {meta['eval_rows']:,} 行，"
             f"{time.time() - t0:.0f}s（GPU {sum(r['seconds'] for r in info['runs']):.0f}s，峰值 {info['peak_mem_gb']} GB）")
    years = [st.name for st in ev.history_refits(keys, START_YEAR)]
    df = pd.concat([pd.read_parquet(parts_dir / f"{y}.parquet") for y in years], ignore_index=True)
    df.to_parquet(hist_dir() / f"scores_{MID}.parquet", index=False)
    _log(f"{MID} 完成，{len(df):,} 行，本次 {time.time() - t_all:.0f}s")
    register("P7_history_train", model=MID, seconds=round(time.time() - t_all, 1))


if __name__ == "__main__":
    main()
