"""真实账户的费用：本金按元计，买入按一手取整，佣金有最低额，卖出收印花税（十年实验 `sim.CostModel` 原样）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: 2023-08-28 印花税减半（0.1% → 0.05%）。
STAMP_CUT = pd.Timestamp("2023-08-28")


@dataclass(frozen=True)
class CostModel:
    """默认：佣金万五（每笔最低 5 元）、印花税万五（卖出）、过户费十万分之一（双边）、一手 100 股。

    `stamp_before`：2023-08-28 印花税减半之前的税率（None = 全期按现行税率，十年实验的 real 档）。
    """

    capital: float = 20_000.0
    commission: float = 0.0005
    min_commission: float = 5.0
    stamp: float = 0.0005
    transfer: float = 0.00001
    lot: int = 100
    stamp_before: float | None = None

    def buy_fee(self, value: float) -> float:
        return max(self.min_commission, value * self.commission) + value * self.transfer if value > 0 else 0.0

    def sell_fee(self, value: float, when: pd.Timestamp) -> float:
        """不超过成交额本身：券商从卖出所得里扣费，扣不出负数（否则本金很小时现金会被扣成负的）。"""
        if value <= 0:
            return 0.0
        stamp = self.stamp_before if self.stamp_before is not None and when < STAMP_CUT else self.stamp
        return min(value, max(self.min_commission, value * self.commission) + value * (stamp + self.transfer))

    def lot_need(self, raw_price: np.ndarray) -> np.ndarray:
        """买一手要多少钱（含费），逐只股票。"""
        value = raw_price * self.lot
        return value + np.maximum(self.min_commission, value * self.commission) + value * self.transfer
