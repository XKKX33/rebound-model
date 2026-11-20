"""任务 3.6 LSTM 的训练与打分（在 dl-study 环境运行，只依赖 numpy 和 torch）。

输入（由主环境的 `ml/lstm_prep.py` 准备，全部是 .npy）：
- data/ml/seq/X.npy                         序列张量 [股票, 交易日, 11]，float16，整份放进显卡
- data/ml/lstm/<阶段>/fit_*.npy、eval_*.npy  训练行 / 评估行的股票下标 j、信号日 d、标签 y、交易日 day，
                                             以及 F69 手工特征（dense 预处理后，float16，只给 S+F69 版本用）
输出：<out>.npy（[评估行, 种子] 的分数）和 <out>.json（每个 epoch 的验证 IC、最佳 epoch、耗时）。
评估行的顺序和主环境 `keys[stage.eval_rows]` 完全一致，分数回到主环境用 `ml/evaluate.py` 统一评估。

规则（PREREG 补充登记三；第二轮另加学习率余弦衰减和权重滑动平均，补充登记九，cfg 里给 tmax、ema 时启用）：损失 = 均方误差（目标 = 训练标签 − 0.5）；Adam；每个 epoch 随机抽一部分训练日，
每批由若干个完整的交易日组成（不超过 4096 行）；选择阶段每个 epoch 末算验证段逐日 IC，连续 5 个 epoch 不提高就停，
取 IC 最高的 epoch；最终阶段不看评估段，直接训练指定的 epoch 数。

运行示例：
  D:/Anaconda/envs/dl-study/python.exe -X utf8 ml/lstm_train.py --stage select --variant S \
      --cfg '{"hidden":64,"layers":1,"dropout":0.0,"lr":0.001,"wd":0.0,"frac":0.5}' --seeds 0 --out data/ml/lstm/runs/S_t1
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "data" / "ml" / "seq" / "X.npy"
LSTM_DIR = ROOT / "data" / "ml" / "lstm"
WINDOW = 60
BATCH_ROWS = 4096          # 1.6 摸底：8192 会超过 8 GB 显存
MICRO_ROWS = 2048           # 反向传播时每块的行数上限（梯度累加，见 train_one）
EVAL_BATCH = 4096           # 打分时的批大小；16384 在隐藏 128 时要 10 GB 显存（第一次运行在第 3 组中断）


class Net(nn.Module):
    """LSTM 读 60 天 × 11 通道的序列，取最后一步的隐状态；S+F69 版本再拼上手工特征；两层全连接输出一个分数。"""

    def __init__(self, n_ch: int, hidden: int, layers: int, dropout: float, n_f: int = 0) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_ch, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden + n_f, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self.lstm(x)
        z = self.drop(h[:, -1])
        if f is not None:
            z = torch.cat([z, f], dim=1)
        return self.head(z).squeeze(-1)


def daily_ic(pred: np.ndarray, y: np.ndarray, day: np.ndarray) -> float:
    """逐日“分数名次与收益百分位”的相关系数再平均（与 ml/models.daily_ic_metric 相同的算法，只用于早停）。"""
    codes = np.searchsorted(np.unique(day), day)
    n = np.bincount(codes).astype(np.float64)
    starts = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64)
    order = np.lexsort((pred, codes))
    r = np.empty(len(pred))
    r[order] = np.arange(len(pred)) - starts[codes[order]]
    my = np.bincount(codes, y) / n
    vy = np.bincount(codes, y * y) / n - my ** 2
    mr, vr = (n - 1) / 2, (n * n - 1) / 12
    cov = np.bincount(codes, r * y) / n - mr * my
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmean(cov / np.sqrt(vr * vy)))


def day_batches(day: np.ndarray, frac: float, rng: np.random.Generator) -> list[np.ndarray]:
    """随机抽 frac 比例的交易日，打乱后把完整的交易日装进批，每批不超过 BATCH_ROWS 行。行已按交易日排好。"""
    uniq, start, count = np.unique(day, return_index=True, return_counts=True)
    pick = rng.permutation(len(uniq))[: max(1, int(round(len(uniq) * frac)))]
    batches, cur, size = [], [], 0
    for k in pick:
        if size + count[k] > BATCH_ROWS and cur:
            batches.append(np.concatenate(cur))
            cur, size = [], 0
        cur.append(np.arange(start[k], start[k] + count[k]))
        size += count[k]
    if cur:
        batches.append(np.concatenate(cur))
    return batches


def load_stage(stage: str, dev: torch.device, with_f: bool) -> dict:
    d = LSTM_DIR / stage
    out = {}
    for part in ("fit", "eval"):
        for k in ("j", "d", "y", "day"):
            out[f"{part}_{k}"] = np.load(d / f"{part}_{k}.npy")
        if with_f:
            out[f"{part}_f"] = torch.from_numpy(np.load(d / f"{part}_f.npy")).to(dev)
    return out


def train_one(X, data, cfg: dict, seed: int, stage: str, epochs: int | None, max_epochs: int, patience: int,
              dev: torch.device, log) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    with_f = "fit_f" in data
    n_f = data["fit_f"].shape[1] if with_f else 0
    net = Net(X.shape[2], cfg["hidden"], cfg["layers"], cfg["dropout"], n_f).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    # 第二轮（PREREG 补充登记九）的两项稳定化做法；cfg 里没有这两个键时与第一轮完全相同
    ema_decay, tmax = cfg.get("ema"), cfg.get("tmax")
    avg = copy.deepcopy(net) if ema_decay else None      # 权重的指数滑动平均；验证、早停、输出都用它
    if avg is not None:
        avg.lstm.flatten_parameters()                     # 复制出来的权重要重新排成连续的一块，否则 cuDNN 每次调用都要整理
    out_net = avg if avg is not None else net
    n_step = 0
    offs = torch.arange(-WINDOW + 1, 1, device=dev)
    fj, fd = torch.from_numpy(data["fit_j"]).to(dev).long(), torch.from_numpy(data["fit_d"]).to(dev).long()
    fy = torch.from_numpy(data["fit_y"]).to(dev).float() - 0.5
    ej, ed = torch.from_numpy(data["eval_j"]).to(dev).long(), torch.from_numpy(data["eval_d"]).to(dev).long()

    def predict() -> np.ndarray:
        out_net.eval()
        out = []
        with torch.no_grad():
            for lo in range(0, len(ej), EVAL_BATCH):
                sl = slice(lo, lo + EVAL_BATCH)
                xb = X[ej[sl][:, None], ed[sl][:, None] + offs[None, :]].float()
                fb = data["eval_f"][sl].float() if with_f else None
                out.append(out_net(xb, fb).float().cpu().numpy())
        net.train()
        return np.concatenate(out)

    n_epochs = epochs if epochs is not None else max_epochs
    curve, best_ic, best_epoch, best_state, bad = [], -np.inf, 0, None, 0
    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        if tmax:                                          # 余弦衰减：第 1 个 epoch 用设定值，第 tmax 个降到接近 0
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * 0.5 * (1 + math.cos(math.pi * (ep - 1) / tmax))
        losses = []
        for b in day_batches(data["fit_day"], cfg["frac"], rng):
            idx_all = torch.from_numpy(b).to(dev)
            opt.zero_grad()
            loss = 0.0
            # 一批（一个交易日）再切成 ≤ MICRO_ROWS 行的小块，逐块反向传播、梯度累加后更新一次：
            # 与整批一起算在数学上相同（损失按整批行数平均），只是显存峰值低（隐藏 128 两层整批会用满 8 GB）
            for lo in range(0, len(idx_all), MICRO_ROWS):
                idx = idx_all[lo:lo + MICRO_ROWS]
                xb = X[fj[idx][:, None], fd[idx][:, None] + offs[None, :]].float()
                fb = data["fit_f"][idx].float() if with_f else None
                part = ((net(xb, fb) - fy[idx]) ** 2).sum() / len(idx_all)
                part.backward()
                loss = loss + part.detach()
            opt.step()
            if avg is not None:
                n_step += 1
                d = min(ema_decay, (1 + n_step) / (10 + n_step))   # 预热：开头几百步不让随机初始权重占太大比重
                with torch.no_grad():
                    for pa, p in zip(avg.parameters(), net.parameters()):
                        pa.mul_(d).add_(p, alpha=1 - d)
            losses.append(loss.item())
        row = {"epoch": ep, "train_mse": float(np.mean(losses)), "lr": opt.param_groups[0]["lr"],
               "seconds": round(time.time() - t0, 1)}
        if stage == "select":
            ic = daily_ic(predict(), data["eval_y"].astype(np.float64), data["eval_day"])
            row["valid_ic"] = ic
            if ic > best_ic:
                best_ic, best_epoch, bad = ic, ep, 0
                best_state = {k: v.detach().clone() for k, v in out_net.state_dict().items()}
            else:
                bad += 1
        curve.append(row)
        log(json.dumps(row, ensure_ascii=False))
        if stage == "select" and bad >= patience:
            break
    if stage == "select":
        out_net.load_state_dict(best_state)
    else:
        best_epoch = n_epochs
    pred = predict()
    info = {"seed": seed, "best_epoch": best_epoch, "best_valid_ic": None if stage == "final" else best_ic,
            "epochs_run": len(curve), "seconds": round(time.time() - t0, 1), "curve": curve,
            "params": sum(p.numel() for p in net.parameters())}
    info["_net"] = out_net
    return pred, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("select", "final"), required=True)
    ap.add_argument("--variant", choices=("S", "SF"), required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=None, help="最终阶段：训练的 epoch 数（不早停）")
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-model", action="store_true", help="把每个种子训练好的权重存成 <out>_seed<种子>.pt（4.3 解释性用）")
    ap.add_argument("--data", default=None, help="data/ml/lstm/ 下的数据子目录（默认 = 阶段名；测试段按年重训用 final_<年>）")
    a = ap.parse_args()
    if a.stage == "final" and a.epochs is None:
        raise SystemExit("最终阶段必须给 --epochs（来自选择阶段的早停结果）")
    cfg = json.loads(a.cfg)
    dev = torch.device("cuda")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(str(out) + ".log", "w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    X = torch.from_numpy(np.load(SEQ)).to(dev)
    data = load_stage(a.data or a.stage, dev, with_f=a.variant == "SF")
    preds, infos = [], []
    for s in [int(v) for v in a.seeds.split(",")]:
        log(f"# 种子 {s}")
        p, info = train_one(X, data, cfg, s, a.stage, a.epochs, a.max_epochs, a.patience, dev, log)
        net = info.pop("_net")
        if a.save_model:
            torch.save(net.state_dict(), f"{out}_seed{s}.pt")
        preds.append(p)
        infos.append(info)
    np.save(str(out) + ".npy", np.column_stack(preds).astype(np.float32))
    meta = {"stage": a.stage, "variant": a.variant, "cfg": cfg, "seeds": a.seeds, "runs": infos,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
    Path(str(out) + ".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log_f.close()


if __name__ == "__main__":
    main()
