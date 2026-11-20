"""期末项目：数据与模型部分（多模型比较）。说明见 ml/README.md。"""

# 关掉 pandas 的 numexpr 加速（2026-11-19 查明）：本机 numexpr 2.14.1（16 线程）对大数组做算术、比较时，
# 偶尔整列返回 0（240 次里 7 次；关掉后 0 次）。这就是本项目一直“原因不明”的偶发计算错误的根源：
# 市场状态特征整列为 0、评估时 KeyError(10)、截面排名整列变 −0.5、空结果等。见 PREREG 冻结后的运行记录。
import pandas as _pd

_pd.set_option("compute.use_numexpr", False)
