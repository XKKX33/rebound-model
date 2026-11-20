"""交接：导出 `ml/api.py` 需要的精简数据包（`data/ml/api_export/`），并打成 zip 交给队友。

`data/` 不进 Git，所以队友拉代码后还需要这份数据包才能用接口。只保留接口真正读的列：
- scores/<model_id>__<select|final>.parquet：trade_date、ts_code、score（3 个种子平均，float32）；
- explain_<model_id>.parquet：测试段每行贡献最大的 8 项（XGBoost：4.4；随机森林、LSTM：P8），原样复制；
- outcomes.parquet：trade_date、ts_code、ret_o1o5、excess_o1o5（`with_outcome=True` 用）；
- stocks.parquet：ts_code、name、industry（研究库 stock_basic 的快照，队友不需要研究库）；
- data_summary.json：`api.data_summary()` 的结果；
- manifest.json：文件清单、行数、导出时间、数据库指纹。
另把交付用的模型文件 `data/ml/deploy/<模型>/`（P8，`ml/p8_deploy.py`）一起打进 zip（zip 里的 `deploy/`），给 `ml/predict.py` 用。
交付范围只有三个模型（PREREG 补充登记十三），所以只导出 `api.MODELS` 里的三个。
有了这个数据包（解压到 `rebound-model/data/ml/api_export/`），接口不再读研究库、主面板和原始分数文件。

运行：python -X utf8 -m ml.p6_export
"""

from __future__ import annotations

import json
import shutil
import time
import zipfile

import pandas as pd

from ml.common import DB_PATH, connect, data_dir, file_fingerprint

EXPORT = "api_export"


def main() -> None:
    from ml import api
    out = data_dir() / EXPORT
    if out.exists():
        shutil.rmtree(out)
    (out / "scores").mkdir(parents=True)
    files = {}
    for mid in api.MODELS:
        for stage in api.STAGES:
            src = data_dir() / "scores" / f"{mid}__{stage}.parquet"
            sc = pd.read_parquet(src, columns=["trade_date", "ts_code", "score"])
            sc["score"] = sc["score"].astype("float32")
            dst = out / "scores" / f"{mid}__{stage}.parquet"
            sc.to_parquet(dst, index=False, compression="zstd")
            files[f"scores/{dst.name}"] = len(sc)
    for mid, fname in api.EXPLAINED.items():
        ex = pd.read_parquet(data_dir() / "explain" / fname)
        ex.to_parquet(out / f"explain_{fname}", index=False, compression="zstd")
        files[f"explain_{fname}"] = len(ex)
    from ml.dataset import load_panel
    oc = load_panel(["trade_date", "ts_code", "ret_o1o5", "excess_o1o5"])
    oc = oc[oc["trade_date"] >= pd.Timestamp("2021-01-01")]           # 只有有分数的区间（验证段起）需要
    oc[["ret_o1o5", "excess_o1o5"]] = oc[["ret_o1o5", "excess_o1o5"]].astype("float32")
    oc.to_parquet(out / "outcomes.parquet", index=False, compression="zstd")
    files["outcomes.parquet"] = len(oc)
    with connect() as con:
        st = con.execute("SELECT ts_code, name, industry FROM stock_basic").df()
    st.to_parquet(out / "stocks.parquet", index=False, compression="zstd")
    files["stocks.parquet"] = len(st)
    (out / "data_summary.json").write_text(json.dumps(api.data_summary(), ensure_ascii=False, indent=1), encoding="utf-8")
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": files,
                "db_fingerprint": file_fingerprint(DB_PATH), "models": list(api.MODELS),
                "说明": "解压到 rebound-model/data/ml/（得到 api_export/ 和 deploy/）；ml/api.py 优先读 api_export/，ml/predict.py 读 deploy/；见 ml/INTERFACE.md",
                "deploy": {mid: json.loads((data_dir() / "deploy" / mid / "meta.json").read_text(encoding="utf-8")).get("versions")
                           for mid in api.MODELS if (data_dir() / "deploy" / mid / "meta.json").exists()}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    zpath = data_dir() / f"{EXPORT}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:           # parquet 已压缩，zip 只打包
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out.parent))
        dep = data_dir() / "deploy"
        for mid in api.MODELS:
            for f in sorted((dep / mid).rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(dep.parent))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"写出 {out}（{size:.0f} MB）和 {zpath}（{zpath.stat().st_size / 1e6:.0f} MB）")


if __name__ == "__main__":
    main()
