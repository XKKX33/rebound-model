"""LSTM S+F69 交付模型的打分（dl-study 环境，numpy + torch），由 `ml/predict.py` 调用。

输入目录里有：j.npy（股票在序列张量里的下标）、d.npy（交易日下标）、f.npy（dense 预处理后的手工特征）；
权重和结构在 `data/ml/deploy/lstm_SF69/`。输出同目录 score.npy（3 个种子的平均分数）。

运行：D:/Anaconda/envs/dl-study/python.exe -X utf8 ml/lstm_predict.py <输入目录>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from lstm_train import SEQ, WINDOW, Net  # noqa: E402

DEPLOY = ROOT / "data" / "ml" / "deploy" / "lstm_SF69"


def main() -> None:
    inp = Path(sys.argv[1])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((DEPLOY / "meta.json").read_text(encoding="utf-8"))
    cfg = meta["cfg"]
    X = np.load(SEQ, mmap_mode="r")
    j, d = np.load(inp / "j.npy"), np.load(inp / "d.npy")
    f = torch.from_numpy(np.load(inp / "f.npy")).float().to(dev)
    xb = torch.from_numpy(np.stack([X[jj, dd - WINDOW + 1:dd + 1] for jj, dd in zip(j, d)]).astype(np.float32)).to(dev)
    preds = []
    for s in (0, 1, 2):
        net = Net(X.shape[2], cfg["hidden"], cfg["layers"], cfg["dropout"], f.shape[1]).to(dev)
        net.load_state_dict(torch.load(DEPLOY / f"net_seed{s}.pt", map_location=dev))
        net.eval()
        with torch.no_grad():
            preds.append(torch.cat([net(xb[i:i + 4096], f[i:i + 4096]) for i in range(0, len(xb), 4096)]).cpu().numpy())
    np.save(inp / "score.npy", np.mean(preds, axis=0).astype(np.float32))


if __name__ == "__main__":
    main()
