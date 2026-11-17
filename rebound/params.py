"""反弹模型训练用的固定口径：准入池的两个版本、xgboost 参数、种子、线程数、重训日。

这些写在代码里而不是策略 yaml 里：改其中任何一项，都要重新训练模型，否则打分用的
模型和准入池对不上。`config/settings.yaml` 只选择股票池版本和重训频率；
训练与打分始终使用这里对应版本的固定条件。

数值全部和十年实验相同（`experiments/decade/`：`train_cadence.py`、`nocap_panel.py`、
`PREREG.md`），2026-10-31 按用户的决定对齐。
"""

from __future__ import annotations

from datetime import date
from typing import Any

from rebound.config import Universe

#: 准入池的两个版本：主板、非 ST 和退市整理（按当时的名称）、成交额 ≥ 0.8 亿、收盘 ≥ 2 元，
#: 限价版再加"收盘 < 70 元"。再加"库里至少 MIN_BARS 根日线"。训练、打分、百分位都在各自的池里。
VARIANTS: dict[str, Universe] = {
    "nocap": Universe(board="mainboard", exclude_st=True, price_max=None, min_amount_yi=0.8, price_min=2.0),
    "cap70": Universe(board="mainboard", exclude_st=True, price_max=70.0, min_amount_yi=0.8, price_min=2.0),
}
VARIANT_NAMES = {"nocap": "不限价", "cap70": "限 70 元"}
#: 每天选股默认用的版本（用户 2026-10-31 定：不限价）。
DEFAULT_VARIANT = "nocap"
MIN_BARS = 150

#: 重训频率 -> 在哪几个月的第一个开市日重训。每年、每季度的重训日都是每月重训日的一部分，
#: 所以只按月训，三种频率共用同一批模型文件。
CADENCES: dict[str, tuple[int, ...]] = {"Y": (1,), "Q": (1, 4, 7, 10), "M": tuple(range(1, 13))}
CADENCE_NAMES = {"Y": "每年", "Q": "每季度", "M": "每月"}
DEFAULT_CADENCE = "M"
#: 第一次重训（和十年实验的信号起点相同）。训练数据从库里第一天（2013-01）起。
FIRST_FIT = date(2015, 1, 1)

#: xgboost 参数（与研究版 A_base 相同）。
PARAMS: dict[str, Any] = {
    "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8,
    "colsample_bytree": 0.8, "tree_method": "hist",
}
#: 3 个种子各训一棵，打分取平均。
SEEDS: tuple[int, ...] = (0, 1, 2)
#: 线程数固定：xgboost 的 hist 算法换线程数，同样的数据训出来的树不一样
#: （2026-10-31 实测，4 线程和 8 线程的预测差到 0.07）。16 和十年实验相同。
THREADS = 16
