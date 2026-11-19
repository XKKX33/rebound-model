"""任务 1.6 的计算量摸底：LSTM 在 RTX 5060 上训练一步多久、占多少显存，推算一个 epoch 的耗时。

只依赖 numpy 和 torch（在 dl-study 环境运行，不导入 ml 包里其他依赖 pandas / duckdb 的模块）：
    D:/Anaconda/envs/dl-study/python.exe -X utf8 ml/lstm_probe.py
读 data/ml/seq/X.npy，随机取有交易的 (股票, 日) 作样本，结果写到 ml/results/p1_6_seq/probe.json。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "data" / "ml" / "seq"
OUT = ROOT / "ml" / "results" / "p1_6_seq" / "probe.json"
WINDOW = 60
TRAIN_ROWS = 1_924_614          # 选择阶段的训练行数（P2 报告）


class Net(nn.Module):
    def __init__(self, n_in: int, hidden: int, layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1]).squeeze(-1)


def main() -> None:
    dev = torch.device("cuda")
    X = torch.from_numpy(np.load(SEQ / "X.npy")).to(dev)            # [股票, 日, 通道] float16
    n_stock, n_day, n_ch = X.shape
    traded = X[:, :, -1] == 1
    jj, dd = torch.nonzero(traded[:, WINDOW:], as_tuple=True)
    dd = dd + WINDOW
    offs = torch.arange(-WINDOW + 1, 1, device=dev)
    runs = []
    for hidden, layers, batch in ((64, 1, 4096), (64, 2, 4096), (128, 2, 4096), (128, 2, 8192)):
        torch.manual_seed(0)
        net = Net(n_ch, hidden, layers).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for step in range(60):
            idx = torch.randint(0, len(jj), (batch,), device=dev)
            xb = X[jj[idx][:, None], dd[idx][:, None] + offs[None, :]].float()   # [batch, 60, 11]，在显卡上切
            yb = torch.rand(batch, device=dev) - 0.5
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = ((net(xb) - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            if step >= 10:
                times.append(time.perf_counter() - t0)
        step_s = float(np.mean(times))
        runs.append({"隐藏维度": hidden, "层数": layers, "批大小": batch,
                     "参数个数": sum(p.numel() for p in net.parameters()),
                     "每步毫秒": round(step_s * 1000, 1),
                     "全量一个 epoch 秒": round(step_s * TRAIN_ROWS / batch, 1),
                     "显存峰值 GB": round(torch.cuda.max_memory_allocated() / 1024**3, 2)})
        print(runs[-1], flush=True)
    info = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "tensor_gb": round(X.numel() * 2 / 1024**3, 2),
            "note": "每步含前向、反向、参数更新；不含验证段打分。全量 epoch = 选择阶段训练行数 / 批大小 × 每步耗时。", "runs": runs}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {OUT}")


if __name__ == "__main__":
    main()
