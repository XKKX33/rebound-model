# 反弹模型

用历史行情给 A 股主板股票排序，生成收盘后的候选名单，并模拟不同本金、持仓规则和交易费用下的账户表现。
项目从“股票 v2”的反弹模型拆出，Python 包名是 `rebound`，内部模型键保留 `return_model`。

## 工作流程

```mermaid
flowchart LR
    A["行情数据"] --> B["股票池与 28 个量价特征"]
    B --> C["收益标签与滚动训练"]
    C --> D["每天打分与选股"]
    D --> E["账户回测与本金测算"]
    E --> F["前向纸面跟踪"]
```

- **股票池**：主板，按当时名称排除 ST 和退市整理；成交额至少 0.8 亿元、收盘至少 2 元、至少 150 根日线。默认不设股价上限，另有收盘低于 70 元的版本。
- **模型**：XGBoost（树模型），用 28 个量价特征预测“信号后第 1 个交易日开盘买、第 5 个交易日开盘卖”的收益在当天股票池里的百分位。3 个种子分别训练，分数取平均；默认每月重训。
- **每日名单**：按分数从高到低选 6 只，同一行业最多 2 只。
- **账户模拟**：考虑一手取整、现金、最低佣金、印花税、过户费、停牌和涨跌停。正式持股规则用前一晚的信息定单，开盘买、尾盘卖，卖出资金在下一交易日开盘再投入。
- **纸面跟踪**：记录模拟账户和下一交易日的买卖清单，每个信号日的清单冻结保存。

模型的标签期限与账户的实际持有天数是两件事：固定持有规则按期限卖出，每日调仓规则按排名是否掉出缓冲区决定卖出。
完整口径见 [模型说明](docs/model.md)，评估方法见 [方法说明](docs/methodology.md)。

## 技术架构

Python 负责流程，Tushare 提供行情，DuckDB 保存数据与每日分数，pandas / NumPy 计算特征，XGBoost 训练模型。
命令行把数据、模型、选股、回测和跟踪各层串起来；文件职责和调用关系见 [ARCHITECTURE.md](ARCHITECTURE.md)。

| 目录 | 内容 |
|---|---|
| `rebound/` | 正式程序与命令行 |
| `config/settings.yaml` | 数据位置、同步参数、默认模型版本和选股数量 |
| `tests/` | 正式程序的隔离测试 |
| `experiments/` | 历史实验、测试、报告及小型结果文件 |
| `docs/` | 模型口径、评估方法和数据维护说明 |

## 本地运行

需要 Python 3.11 或以上。依赖版本固定在 `pyproject.toml`，模型文件的校验信息也记录训练版本。
本机已经有版本匹配的 Anaconda 环境，可直接复用。

在仓库根目录打开 PowerShell：

```powershell
$reboundPython = 'C:\Users\xuan\anaconda3\python.exe'
& $reboundPython -X utf8 -m rebound --help
```

已有的其他环境首次安装依赖时：

```powershell
& $reboundPython -m pip install -e '.[dev,research]'
```

`dev` 是测试依赖，`research` 是实验脚本使用的额外依赖。只运行正式程序时可安装 `-e .`。

### 首次准备真实数据

仓库交付源码、测试和历史报告。行情库、训练好的模型、账户记录与凭据由使用者在本地准备。
Tushare 的接口权限必须覆盖交易日历、股票列表、日线、复权因子、涨跌停价和名称变更。

```powershell
Copy-Item -LiteralPath '.env.example' -Destination '.env'
```

将 `.env` 中的 `TUSHARE_TOKEN` 改为自己的凭据。已有同名环境变量优先于 `.env`。
确认目标库位置和写入范围后，首次回补历史并训练：

```powershell
& $reboundPython -X utf8 -m rebound sync --days 3500
& $reboundPython -X utf8 -m rebound model train
& $reboundPython -X utf8 -m rebound model score
```

这些命令会写入行情库、模型和每日分数。首次全量任务比日常增量任务更长。
已有兼容的旧库时，先做导入演练：

```powershell
& $reboundPython -X utf8 -m rebound data import-history --source 'D:\行情\old.duckdb' --dry-run
```

确认演练的范围和结果后去掉 `--dry-run` 才会正式导入；目标已有数据时先备份，重叠数据不一致会拒绝写入。
失败和备份处理见 [数据维护说明](docs/RUNBOOK.md)。

### 日常使用

行情定稿时间默认是北京时间 15:30。日常命令补行情、到期模型和缺失分数，再选股并刷新纸面跟踪。

```powershell
& $reboundPython -X utf8 -m rebound daily
& $reboundPython -X utf8 -m rebound pick
& $reboundPython -X utf8 -m rebound pick --as-of 2024-03-15
& $reboundPython -X utf8 -m rebound capital --capital 20000
& $reboundPython -X utf8 -m rebound track start --rule DB-K2-b6-o1c5 --capital 20000
& $reboundPython -X utf8 -m rebound track list
& $reboundPython -X utf8 -m rebound track show
```

`DB-K2-b6-o1c5` 表示最多拿 2 只，排名掉出前 12 名才按规则卖出；这里用于说明命令格式。
`capital` 会按输入本金重新比较规则。每日 6 只候选名单与账户持仓数量分别由选股配置和持股规则决定。

主要文件位置：

```text
data/
├─ market.duckdb                              行情与每日分数
├─ market.models/return_model/<版本>/<重训日>/  模型文件及校验信息
├─ market.capital/                            本金测算缓存
└─ market.tracking/                           纸面账户与冻结清单
```

## 安装与部署

部署单位是“Python 环境 + 项目目录 + 本地数据”。普通安装时：

```powershell
& $reboundPython -m pip install .
& $reboundPython -X utf8 -m rebound --config 'C:\项目\rebound-model\config\settings.yaml' track list
```

普通安装后，`--config` 必须指向项目的 `config/settings.yaml`，放在子命令前。
相对数据库路径按该文件上两级的项目根目录解析，凭据从同一项目根的 `.env` 读取。
在源码仓库内运行或使用可编辑安装时，可以省略 `--config`。

复制部署目录时保留 `config/settings.yaml`，再单独准备 `.env` 和 `data/`。

## 测试

正式测试使用临时合成行情和小模型，覆盖训练、打分、选股、账户模拟和命令行，不需要真实行情或联网。

```powershell
& $reboundPython -X utf8 -m pytest -p no:cacheprovider tests
& $reboundPython -X utf8 -m pytest -p no:cacheprovider experiments
```

第二条验证实验的算法与报告逻辑；需要研究面板的检查会按测试说明跳过。
它不等于重跑十年行情、训练和回测。

2026-11-16 的仓库收尾验证结果记录在 [progress.md](progress.md)。

## 历史实验与主要结果

下面是随代码迁入的 2026-10 / 11 月研究记录，本次收尾没有重跑真实数据。

| 实验 | 研究内容与记录的结论 |
|---|---|
| [十年实验](experiments/decade/CONCLUSION.md) | 比较重训频率、持仓规则、股价上限和本金；小本金受最低佣金与一手取整影响很大 |
| [买卖时点](experiments/execgap/REPORT.md) | 同一个开盘价先卖后买会高估满仓换仓收益，正式规则改为前一晚决定、尾盘卖 |
| [最短持有天数](experiments/minhold/REPORT.md) | 强制多持有几天没有按预注册标准证实改善 |
| [两日标签](experiments/two_day/REPORT.md) | 两日模型没有找到四个起点都赚钱的规则，保留五日标签 |
| [研究与正式程序核对](experiments/parity/RESULTS.md) | 核对数据、特征、标签、模型分数与模拟器；记录了浮点误差和退市整理排除带来的差异 |

`docs/model.md` 记录的尾盘卖历史版本中，2 万本金选中的规则全期年化为 29.0%、最大回撤为 −70.6%；
按历史数据逐年选规则并拼接的滚动检验年化为 5.5%。这两个数字的选法不同，不能互换。
开发期最初要求的超额置信区间下限大于零没有通过，后来也多次查看验证期，历史结果不能当作新账户的收益承诺。

### 重跑研究

`experiments/decade/` 接受 `DECADE_DB` 指定研究库、`DECADE_DATA` 指定生成目录。
原十年库默认名为 `data/research-2013-2026.duckdb`；其他早期脚本使用 `data/research.duckdb`。
十年报告另外读取 `index_daily`（指数日线表）；正式行情同步只生成七张正式表，不会生成这张研究用表。

研究面板、标签、各频率分数和历史备份没有随仓库交付。
两日标签实验的完整启动器使用 CUDA（GPU 训练）；正式程序和隔离测试使用 CPU。
各实验的先决文件、参数和判定标准见对应脚本开头及 `PREREG.md`，应在独立的实验生成目录运行。

## 搜索记录与来源

此次是现有模型的拆出与收尾，直接复用本地“股票 v2”的实现和实验，没有新增算法模块，因此没有重复搜索 skills.sh 或 GitHub。
原方法已经参考 [Qlib](https://github.com/microsoft/qlib)，具体借用内容与文献列在 [方法说明](docs/methodology.md)。
源版本和导入范围见 [导入清单](导入清单.md)。

## 已完成与待办

- 已迁入数据、特征、标签、模型、选股、回测、跟踪和实验代码。
- 已修正数据模块被 Git 忽略的问题，补齐安装后的显式配置入口。
- 已补齐运行说明、架构、数据维护与验证记录。
- 待使用者配置真实数据与凭据，并完成真实数据上的独立运行验收。
