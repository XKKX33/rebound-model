"""P8：LSTM S+F69 的逐只股票解释（积分梯度，PREREG 补充登记十三第 3 条），在 dl-study 环境运行（numpy + torch）。

由主环境的 `ml/p8_explain_lstm.py` 调度。对某一年（测试段按年重训的那一年）：
1. 读 `data/ml/lstm/runs/lstm_SF69__explain_<年>_seed{0,1,2}.pt`（同配置、同训练行重训并保存的权重）；
2. 对 `final_<年>` 评估行里属于这一年的行（`explain_mask_<年>.npy`）算积分梯度，基线与 4.3（`ml/lstm_ig.py`）相同：
   序列取 0（截面中位数），“有交易”通道取 1，手工特征取 0；从基线到实际输入分 32 步，梯形近似；3 个种子的归因取平均
  （积分梯度对模型是线性的，平均归因 = 三个种子平均分数的归因）；
3. 解释项 = 11 个序列通道（每个通道 60 天的归因加总）+ 78 个手工特征（dense 预处理后的值）；每行保存贡献最大的 8 项。

LSTM S+F69 只有 1 层、dropout = 0，训练模式和推理模式输出相同，所以用训练模式让 cuDNN 也能反向传播（比关掉 cuDNN 快很多）。

运行：D:/Anaconda/envs/dl-study/python.exe -X utf8 ml/lstm_explain.py --year 2023
输出：data/ml/lstm/explain_<年>/{score,base_score,order,vals,xs,gap}.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from lstm_train import SEQ, WINDOW, Net  # noqa: E402

LSTM_DIR = ROOT / "data" / "ml" / "lstm"
CHANNELS = ["ret", "ovn", "intra", "hl", "clv", "ln_amt", "ln_turn", "mf", "big", "limit", "traded"]
STEPS, BATCH, TOP = 32, 2000, 8


def ig_batch(nets, xb, fb, x0, f0):
    """3 个种子平均的积分梯度：返回 (通道归因 [n,11], 特征归因 [n,F], 平均分数, 平均基线分数)。"""
    alphas = torch.linspace(0, 1, STEPS + 1, device=xb.device)
    ax = torch.zeros(xb.shape[0], xb.shape[2], device=xb.device)
    af = torch.zeros_like(fb)
    for net in nets:
        gx, gf = torch.zeros_like(xb), torch.zeros_like(fb)
        for k, a in enumerate(alphas):
            w = 0.5 if k in (0, STEPS) else 1.0
            xi = (x0 + a * (xb - x0)).requires_grad_(True)
            fi = (f0 + a * (fb - f0)).requires_grad_(True)
            g = torch.autograd.grad(net(xi, fi).sum(), [xi, fi])
            gx += w * g[0]
            gf += w * g[1]
        ax += ((xb - x0) * gx / STEPS).sum(dim=1)
        af += (fb - f0) * gf / STEPS
    with torch.no_grad():
        s = torch.stack([n(xb, fb) for n in nets]).mean(0)
        s0 = torch.stack([n(x0, f0) for n in nets]).mean(0)
    k = len(nets)
    return ax / k, af / k, s, s0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ch = json.loads((ROOT / "ml" / "results" / "p3_6_lstm" / "chosen.json").read_text(encoding="utf-8"))["SF"]
    cfg = ch["cfg"]
    if cfg["dropout"] != 0.0 or cfg["layers"] != 1:
        raise SystemExit("训练模式 = 推理模式的前提不成立（dropout 不是 0 或多层），要改成关掉 cuDNN 的做法")
    d = LSTM_DIR / f"final_{a.year}"
    mask = np.load(d / f"explain_mask_{a.year}.npy")
    ej = np.load(d / "eval_j.npy")[mask]
    ed = np.load(d / "eval_d.npy")[mask]
    F = torch.from_numpy(np.load(d / "eval_f.npy")[mask]).to(dev)
    X = torch.from_numpy(np.load(SEQ)).to(dev)
    nets = []
    for s in (0, 1, 2):
        net = Net(X.shape[2], cfg["hidden"], cfg["layers"], cfg["dropout"], F.shape[1]).to(dev)
        net.load_state_dict(torch.load(LSTM_DIR / "runs" / f"lstm_SF69__explain_{a.year}_seed{s}.pt", map_location=dev))
        net.train()                               # dropout = 0，输出与 eval 相同；cuDNN 的 LSTM 只能在训练模式下反向传播
        nets.append(net)
    offs = torch.arange(-WINDOW + 1, 1, device=dev)
    n = len(ej)
    score, order, vals, xs, gap = (np.empty(n, np.float32), np.empty((n, TOP), np.int16), np.empty((n, TOP), np.float32),
                                   np.empty((n, TOP), np.float32), np.empty(n, np.float32))
    base = None
    for lo in range(0, n, BATCH):
        sl = slice(lo, lo + BATCH)
        jj = torch.from_numpy(ej[sl]).to(dev).long()
        dd = torch.from_numpy(ed[sl]).to(dev).long()
        xb = X[jj[:, None], dd[:, None] + offs[None, :]].float()
        x0 = torch.zeros_like(xb)
        x0[..., CHANNELS.index("traded")] = 1.0
        fb = F[sl].float()
        f0 = torch.zeros_like(fb)
        ax, af, s, s0 = ig_batch(nets, xb, fb, x0, f0)
        A = torch.cat([ax, af], dim=1)                                    # [n, 11 + F]
        V = torch.cat([xb[:, -1, :], fb], dim=1)                          # 数值：通道取信号日 T 当天的值
        o = torch.argsort(-A.abs(), dim=1)[:, :TOP]
        score[sl] = s.detach().cpu().numpy()
        order[sl] = o.cpu().numpy()
        vals[sl] = torch.gather(A, 1, o).detach().cpu().numpy()
        xs[sl] = torch.gather(V, 1, o).cpu().numpy()
        gap[sl] = (A.sum(1) - (s - s0)).abs().detach().cpu().numpy()
        base = float(s0[0])
    out = LSTM_DIR / f"explain_{a.year}"
    out.mkdir(exist_ok=True)
    for k, v in {"score": score, "order": order, "vals": vals, "xs": xs, "gap": gap, "base_score": np.array([base], np.float32)}.items():
        np.save(out / f"{k}.npy", v)
    print(f"{a.year}：{n:,} 行，完备性误差中位数 {np.median(gap):.2e}（分数尺度约 {np.std(score):.3f}）", flush=True)


if __name__ == "__main__":
    main()
