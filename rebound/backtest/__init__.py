"""回测：反弹模型的逐日打分 → 按持股规则逐日模拟真实账户（一手取整、最低佣金、印花税、过户费）。

    rules.py    持股规则网格（固定持有 FH、带缓冲的每日调仓 DB）和规则名的解析
    sim.py      逐日模拟器：开盘买、尾盘或开盘卖，买不进顺延下一名，卖不出就拿着
    costs.py    费率表
    market.py   行情矩阵与候选（从库里读）
    stats.py    年化、回撤、超额和按月重抽的区间、信息比率、滚动样本外选规则
    capital.py  本金测算：规则 × 重训频率 × 起点，给出推荐
    plan.py     下一个开市日的买卖清单

模拟器、规则网格、选法和十年实验（`experiments/decade/`、`experiments/shared/sim.py`）是同一套，
喂研究版的打分时逐项相同（`experiments/parity/capital_check.py`）。
"""
