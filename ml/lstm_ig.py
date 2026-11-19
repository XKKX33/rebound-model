"""任务 4.3：LSTM 的积分梯度（Integrated Gradients），在 dl-study 环境运行（只依赖 numpy、torch、matplotlib）。

看 LSTM 的分数主要由 60 天里的哪几天、11 个通道里的哪几个决定；S+F69 版本再看序列和 69 个手工特征各占多少。

做法：
1. 用冻结配置在选择阶段训练（训练 2013 ~ 2020，固定 epoch 数，种子 0，和测试段的训练方式一样），
   由 `ml/lstm_train.py --stage final --data select --save-model` 训练并保存权重（本脚本会自动调用）；
2. 在验证段随机抽 N 行，对每行算积分梯度：基线 = 截面中位数的股票（排名通道取 0，涨跌停取 0，“有交易”取 1；手工特征取 0），
   从基线到实际输入分 STEPS 步，累计梯度 × (输入 − 基线)。归因之和 ≈ 分数 − 基线分数（完备性，本脚本会核对）；
3. 汇总：每个通道、每个滞后天（T、T−1、…、T−59）的平均 |归因|，以及按通道的平均带符号归因。

运行：D:/Anaconda/envs/dl-study/python.exe -X utf8 ml/lstm_ig.py
输出：ml/results/p4_3_explain/lstm_ig.md、lstm_ig_*.png、lstm_ig.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from lstm_train import WINDOW, Net, load_stage  # noqa: E402

OUT = ROOT / "ml" / "results" / "p4_3_explain"
RUNS = ROOT / "data" / "ml" / "lstm" / "ig"
CHANNELS = ["ret", "ovn", "intra", "hl", "clv", "ln_amt", "ln_turn", "mf", "big", "limit", "traded"]
CN = {"ret": "日收益", "ovn": "隔夜收益", "intra": "日内收益", "hl": "日内振幅", "clv": "收盘位置", "ln_amt": "成交额",
      "ln_turn": "换手率", "mf": "资金净流入", "big": "大单净买", "limit": "涨跌停", "traded": "有交易"}
N_ROWS, STEPS, BATCH = 20000, 32, 1000
ROUND2 = {"ema": 0.999, "tmax": 20}


def trained(variant: str, mid: str, cfg: dict, epochs: int) -> Path:
    path = RUNS / f"{mid}_seed0.pt"
    if not path.exists():
        RUNS.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "ml" / "lstm_train.py"), "--stage", "final", "--data", "select",
                        "--variant", variant, "--cfg", json.dumps({**cfg, **ROUND2}), "--seeds", "0", "--epochs", str(epochs),
                        "--out", str(RUNS / mid), "--save-model"], cwd=ROOT, check=True)
    return path


def ig(net, xb, fb, x0, f0):
    """一批样本的积分梯度（梯形近似）。返回 (序列归因 [n,60,11], 特征归因 [n,F] 或 None, 分数, 基线分数)。"""
    alphas = torch.linspace(0, 1, STEPS + 1, device=xb.device)
    gx = torch.zeros_like(xb)
    gf = torch.zeros_like(fb) if fb is not None else None
    for k, a in enumerate(alphas):
        w = 0.5 if k in (0, STEPS) else 1.0
        xi = (x0 + a * (xb - x0)).requires_grad_(True)
        fi = (f0 + a * (fb - f0)).requires_grad_(True) if fb is not None else None
        out = net(xi, fi).sum()
        grads = torch.autograd.grad(out, [xi] + ([fi] if fi is not None else []))
        gx += w * grads[0]
        if fi is not None:
            gf += w * grads[1]
    gx, attr_x = gx / STEPS, None
    attr_x = (xb - x0) * gx
    attr_f = (fb - f0) * (gf / STEPS) if fb is not None else None
    with torch.no_grad():
        s, s0 = net(xb, fb), net(x0, f0)
    return attr_x, attr_f, s, s0


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    torch.backends.cudnn.enabled = False          # cuDNN 的 LSTM 只能在训练模式下反向传播；关掉它，用普通实现
    dev = torch.device("cuda")
    ch = json.loads((ROOT / "ml" / "results" / "p3_6_lstm" / "chosen.json").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "data" / "ml" / "lstm" / "select" / "meta.json").read_text(encoding="utf-8"))
    X = torch.from_numpy(np.load(ROOT / "data" / "ml" / "seq" / "X.npy")).to(dev)
    offs = torch.arange(-WINDOW + 1, 1, device=dev)
    rng = np.random.default_rng(20261119)
    res = {}
    for variant, mid in (("S", "lstm_S"), ("SF", "lstm_SF69")):
        cfg = ch[variant]["cfg"]
        path = trained(variant, mid, cfg, ch[variant]["最终阶段 epoch"])
        data = load_stage("select", dev, with_f=variant == "SF")
        n_f = data["eval_f"].shape[1] if variant == "SF" else 0
        net = Net(X.shape[2], cfg["hidden"], cfg["layers"], cfg["dropout"], n_f).to(dev)
        net.load_state_dict(torch.load(path, map_location=dev))
        net.eval()
        pick = np.sort(rng.choice(len(data["eval_j"]), N_ROWS, replace=False))
        ej = torch.from_numpy(data["eval_j"][pick]).to(dev).long()
        ed = torch.from_numpy(data["eval_d"][pick]).to(dev).long()
        sum_abs = torch.zeros(WINDOW, X.shape[2], device=dev, dtype=torch.float64)
        sum_sgn = torch.zeros(WINDOW, X.shape[2], device=dev, dtype=torch.float64)
        f_abs = torch.zeros(n_f, device=dev, dtype=torch.float64) if n_f else None
        gaps, seq_share = [], []
        for lo in range(0, N_ROWS, BATCH):
            sl = slice(lo, lo + BATCH)
            xb = X[ej[sl][:, None], ed[sl][:, None] + offs[None, :]].float()
            x0 = torch.zeros_like(xb)
            x0[..., CHANNELS.index("traded")] = 1.0
            fb = data["eval_f"][torch.from_numpy(pick[sl]).to(dev)].float() if n_f else None
            f0 = torch.zeros_like(fb) if n_f else None
            ax, af, s, s0 = ig(net, xb, fb, x0, f0)
            total = ax.sum(dim=(1, 2)) + (af.sum(dim=1) if af is not None else 0)
            gaps.append(((total - (s - s0)).abs() / (s - s0).abs().clamp_min(1e-6)).cpu().numpy())
            sum_abs += ax.abs().sum(0).double()
            sum_sgn += ax.sum(0).double()
            if n_f:
                f_abs += af.abs().sum(0).double()
                seq_share.append((ax.abs().sum(dim=(1, 2)) / (ax.abs().sum(dim=(1, 2)) + af.abs().sum(1))).cpu().numpy())
        A = (sum_abs / N_ROWS).cpu().numpy()            # [60 天, 11 通道]，下标 59 = 信号日 T
        Sg = (sum_sgn / N_ROWS).cpu().numpy()
        r = {"channel_abs": dict(zip(CHANNELS, A.sum(0).tolist())), "channel_signed": dict(zip(CHANNELS, Sg.sum(0).tolist())),
             "lag_abs": A.sum(1)[::-1].tolist(),          # 下标 0 = T，1 = T−1，…
             "completeness_median_rel_err": float(np.median(np.concatenate(gaps)))}
        if n_f:
            fa = (f_abs / N_ROWS).cpu().numpy()
            names = meta["f_names"]
            order = np.argsort(-fa)[:15]
            r["seq_share_median"] = float(np.median(np.concatenate(seq_share)))
            r["top_features"] = [(names[i], float(fa[i])) for i in order]
        res[mid] = r
        # 图：通道 × 滞后热力图（只画最近 20 天，越靠右越近）
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [3, 1]})
        im = axes[0].imshow(A[-20:].T, aspect="auto", cmap="magma")
        axes[0].set_yticks(range(len(CHANNELS)))
        axes[0].set_yticklabels([f"{c}（{CN[c]}）" for c in CHANNELS], fontsize=8)
        axes[0].set_xticks(range(20))
        axes[0].set_xticklabels([f"T−{19 - k}" if k < 19 else "T" for k in range(20)], fontsize=7, rotation=90)
        axes[0].set_title(f"{mid}：平均 |归因|（最近 20 个交易日 × 11 个通道）")
        fig.colorbar(im, ax=axes[0])
        lag = np.array(r["lag_abs"])
        axes[1].bar(range(WINDOW), lag / lag.sum(), color="#2a5d9f")
        axes[1].set_xlabel("距信号日的天数（0 = T）")
        axes[1].set_title("各天的归因占比（60 天）")
        fig.tight_layout()
        fig.savefig(OUT / f"lstm_ig_{mid}.png", dpi=130)
        plt.close(fig)
        print(f"{mid}：完备性相对误差中位数 {r['completeness_median_rel_err']:.3f}", flush=True)
    (OUT / "lstm_ig.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(res)


def write_md(res: dict) -> None:
    lines = [f"- 模型：冻结配置在选择阶段训练（2013 ~ 2020，固定 epoch，种子 0）；验证段随机抽 {N_ROWS:,} 行；积分梯度 {STEPS} 步，",
             "  基线 = 截面中位数的股票（排名通道 0、涨跌停 0、有交易 1、手工特征 0）。",
             "  完备性核对（归因之和 vs 分数 − 基线分数，相对误差中位数）：" +
             "、".join(f"{m} {r['completeness_median_rel_err']:.3f}" for m, r in res.items()), ""]
    for mid, r in res.items():
        tot = sum(r["channel_abs"].values())
        lag = np.array(r["lag_abs"])
        lines += [f"**{mid}：各通道的平均 |归因| 占比，以及带符号的平均归因（正 = 让分数变高）**", "",
                  "| 通道 | 中文 | |归因| 占比 | 平均带符号归因 |", "|---|---|---:|---:|"]
        for c in sorted(r["channel_abs"], key=lambda k: -r["channel_abs"][k]):
            lines.append(f"| {c} | {CN[c]} | {r['channel_abs'][c] / tot * 100:.1f}% | {r['channel_signed'][c]:+.5f} |")
        lines += ["", f"时间上：最近 1 天（T）占 {lag[0] / lag.sum() * 100:.1f}%，最近 5 天占 {lag[:5].sum() / lag.sum() * 100:.1f}%，"
                  f"最近 20 天占 {lag[:20].sum() / lag.sum() * 100:.1f}%，21 ~ 60 天前占 {lag[20:].sum() / lag.sum() * 100:.1f}%。", ""]
        if "seq_share_median" in r:
            lines += [f"序列 vs 手工特征：每行 |归因| 中序列所占比例的中位数是 {r['seq_share_median'] * 100:.1f}%。"
                      "手工特征里归因最大的 15 个：" + "、".join(f"{n}（{v:.4f}）" for n, v in r["top_features"]), ""]
        lines += [f"![{mid}](lstm_ig_{mid}.png)", ""]
    (OUT / "lstm_ig.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
