# 数据维护

命令中的 Python 路径应使用现有环境。本机为 `C:/Users/xuan/anaconda3/python.exe`。
普通安装需在子命令前加 `--config <项目>/config/settings.yaml`；源码目录运行可省略。

## 新项目准备

1. 在项目根准备 `.env`，设置具备所需接口权限的 `TUSHARE_TOKEN`。
2. 核对 `config/settings.yaml` 的 `data.db_path`。相对路径属于项目根。
3. 确认目标和写入范围后运行 `rebound sync --days 3500`，再运行 `rebound model train` 和 `rebound model score`。

正式库包含交易日历、股票列表、日线、涨跌停价、复权因子、名称变更和每日模型分数。
现有研究库可能另有指数等表，正式同步不会生成这些研究用表。

## 导入旧库

先运行：

```powershell
& 'C:\Users\xuan\anaconda3\python.exe' -X utf8 -m rebound data import-history --source 'D:\行情\old.duckdb' --dry-run
```

演练只读核对，不创建目标库、不备份、不写入；目标库需先存在。报告给出可补数据、重叠差异和拒绝原因。
确认导入范围后去掉 `--dry-run`。目标已有内容时，程序先创建带时间戳的备份，再按年份事务提交。
已存在的每日模型分数不从旧库导入，需按本项目模型口径训练和打分。

## 错误与恢复

- **重叠数据不同**：核对源库和目标库的版本、字段与日期。程序拒绝自动覆盖已有行情。
- **导入中断**：先看错误中列出的已提交年份与备份路径。未提交的一批会整批回滚，原样重跑会补剩余数据。
- **要恢复备份**：先结束使用该目标库的同步、训练和查询；确认目标绝对路径、备份路径以及模型和分数对应的日期。保留当前库与同名 `.wal`，再由使用者确认恢复范围后替换数据库文件。
- **模型口径不同**：错误会列出变化项。先核对实际改过的特征、股票池或标签，再决定重新生成哪些模型和分数；不要直接覆盖历史计划。
- **行情不新或不完整**：检查固定行情入口、接口权限和同步报告，补齐实际缺少的表与交易日后再打分。

## AI 辩论常见故障

- **Node 未安装或版本过低**：侧车要求 Node.js `>= 22.19.0`。先升级 Node，再运行 `rebound debate check` 核对。
- **侧车依赖未安装**：`debate_agent/node_modules` 缺失时在 `debate_agent/` 目录执行 `npm install`。
- **缺少大模型密钥**：确认 `.env` 中已设置 `WORKBENCH_AI_API_KEY`（或 `debate.llm.api_key_env` 指定的变量）；`check` 只报有没有，不会打印密钥值。
- **上游限流或单只失败**：检查 `config/settings.yaml` 的 `debate.concurrency`（`step-5-preview` 上限 5 路并发）；单只失败不写检查点，其余已辩完股票保留。
- **辩完不足或中途中断（退出码 5）**：辩完少于 `debate.min_debated` 或按下 `Ctrl+C` 时不出 `final.json`。直接重跑同一命令即可基于 `stocks/*.json` 检查点续跑，仅发送未完成股票并重新统一终选。
- **侧车卡死被看门狗杀死**：超过 `call_timeout_seconds * 1.5 + 60` 秒无任何输出会被强杀，检查报错末尾的侧车 stderr 日志与网络连通性后重跑续上。

退出码：`0` 成功，`1` 导入演练未通过，`2` 参数错误，`3` 数据不新，`4` 数据、配置或模型错误，`5` 辩论未出名单。
