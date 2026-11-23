# A 股量化选股与智能研判

将市场行情、量价特征、三模型回测、资金配置和多角色辩论放在同一工作台，支持从个股走势追溯到模型评分依据，再检查行业与预算约束下的持仓选择。

## 页面与技术路线

| 栏目 | 展示内容 | 数据与计算 |
| --- | --- | --- |
| 市场行情 | 指数、市场涨跌、股票搜索、日周月 K 线、成交额和资金流 | Tushare Pro 历史库；周月线先聚合，再计算均线 |
| 特征工程 | 6 项量价观察、69 项完整特征目录、60×11 日线序列、评分归因 | 28 项基础量价与 41 项扩展特征；沿用模型输入口径 |
| 模型与策略 | XGBoost、随机森林、LSTM 净值、收益风险指标、持仓策略 | 成员已交付的评分与回测结果，统一规则和区间比较 |
| 投资组合案 | 预算、现金留存、行业上限、整手股数、费用与选择依据 | 按同一信号日的模型前 20 名实时重算配置 |
| AI辩论台 | 工作流、候选队列、角色论证、思考过程、终选、净值和投研对话 | 复用成员辩论后端，支持历史回放与实时 SSE |

前端采用 HTML、CSS 与 SVG 图表；后端采用 FastAPI、DuckDB 和成员的 `ml.api`。主页面、辩论子页面、静态资源和流式接口通过同一个网址访问，浏览器不依赖访问者电脑上的额外端口。

## 安装与启动

已在 Python 3.13.9 环境验证。查看历史行情、回测和已保存辩论不需要模型 API 密钥；重新辩论或发送投研追问需要配置可用的大模型接口。

```powershell
python -m venv work\.venv
.\work\.venv\Scripts\python.exe -m pip install -r requirements.txt
git clone https://github.com/XKKX33/rebound-model.git github_clone\rebound-model
```

成员模型接口对应提交 `9c4339c18f573b143d5330e7732f0c4237ed2d27`。建议先在独立克隆中检出该提交，再验证更新后的兼容性。

本项目的数据目录结构：

```text
项目根目录/
├── data/
│   ├── research-2013-2026.duckdb
│   ├── research-2013-2026.debate/     已保存的辩论与对话
│   └── incremental/                 独立增量数据库
├── 特征与模型/
│   └── api_export/                  成员导出的评分、归因和净值
├── github_clone/
│   └── rebound-model/
└── work/
    ├── ui_data_profile.json
    └── dashboard/
        ├── server.py
        ├── feature_snapshot.parquet
        └── feature_snapshot.json
```

数据库和模型导出包通过小组原有渠道提供。复制源码后，可继续引用已有项目目录，避免重复复制数据库：

```powershell
$env:DASHBOARD_PROJECT_ROOT = 'D:\Codex\统计机器学习小组展示'
.\work\dashboard\start.ps1
```

在新电脑上按照上述目录放置数据后，运行一次下面的命令生成数据指纹与观察截面。它只读数据库，不训练模型。

```powershell
.\work\.venv\Scripts\python.exe work\dashboard\prepare_data.py --as-of 2026-10-23
.\work\dashboard\start.ps1
```

在小组的 `rebound-model` 仓库内，直接运行 `work/dashboard/` 下的相同入口即可。看板自动使用同仓库的 `rebound/`、`ml/` 和 `data/ml/api_export/`，不需要再克隆一份模型仓库。数据指纹与观察截面仍由 `prepare_data.py` 生成。

统一网址为 `http://127.0.0.1:8875/`。启动脚本先核对并关闭本项目旧服务，再启动这一入口；`DASHBOARD_PYTHON` 可指定 Python。数据库、模型仓库、导出包和截面缓存分别支持 `DASHBOARD_DATABASE`、`REBOUND_REPO`、`DASHBOARD_MODEL_ARTIFACTS`、`DASHBOARD_FEATURE_CACHE` 等路径配置，定义见 `paths.py`。

真实模型评分截至 2026-10-16，行情截至 2026-10-23。特征观察截面与模型信号日分别显示；已有回测和历史辩论沿用对应信号日，不因后来行情更新而重新标注日期。

实时辩论使用成员仓库的 `debate_agent`，需要 Node.js ≥22.19 并在其目录执行 `npm ci`。模型服务地址与模型名通过成员 `config/settings.yaml` 配置，密钥仅放在环境变量或本机 `.env`，不要加入 Git。

## 小组共享与课堂演示

本机访问用 `127.0.0.1` 即可；队员远程访问需要公网入口或同一网络下的局域网地址。`share.py` 提供一个带密码的转发网关，保留 SSE 流式传输。Windows 上先启动看板，再双击 `work/dashboard/开启共享.cmd`，或执行：

```powershell
.\work\dashboard\share.ps1
```

网关默认监听本机 8880，转发同一个 8875 看板。首次运行自动在脚本旁生成 `work/dashboard/share_access.json`，共享用户名为 `team`；用记事本打开该文件，`password` 字段就是共享密码。密码不进入仓库。启动脚本使用 Windows 自带的 SSH 连接 localhost.run，在健康检查通过后显示可分享的 HTTPS 地址；`tunnel.py` 将地址和连接状态更新到 `work/sharing/public_state.json`，域名轮换后重新检查。保持脚本窗口和电脑运行，按 Ctrl+C 结束隧道。已在公网浏览器逐栏核验五个栏目，并检查 SSE 首个事件；普通 HTML、脚本与 JSON 采用 gzip 压缩，实时流保留直接传输。

免费隧道的域名会轮换，加载较多内容也可能触发免费流量限制。它适合临时给队员看效果。课堂固定网址应另用托管后端或固定域名隧道，并保留本机演示和录像。

`重启看板.cmd` 先由 `service_control.py` 核对 8765、8767、8875 的进程确属本项目，等待进程真正退出后清理仅含生成网页的旧目录，最后启动 8875。不会删除数据库、模型结果和辩论记录。只有 Windows 实际拒绝读取或终止进程时，脚本才提示“以管理员身份运行”；等待超时、启动失败会分别显示原因。`service_state.json` 登记当前辩论生成目录，重启锁避免多个窗口同时重启；部分旧目录清理权限不足也不会阻止新服务启动。

GitHub Pages 只能托管静态页面，无法直接运行当前的 DuckDB、Python 后端和 AI 流式接口。固定公网网址需要部署完整后端，或配置固定域名隧道；上传源码到 GitHub 后仍需单独完成这一步。

固定网址可使用小组已有域名的子域名，配合 Cloudflare 正式 Tunnel 将请求转发到本机 8880；原有密码与 SSE 流保持不变。域名需要由其管理员完成授权和 DNS 配置，电脑与本机服务仍须运行。临时 `*.app.lhr.life` 地址由服务商分配，不能在页面代码中改成自选名称；添加网址跳转也不会解决隧道断开或限流的问题。

## 源码交付

小组交付目标为 [XKKX33/rebound-model](https://github.com/XKKX33/rebound-model)；看板源码位于该仓库的 `work/dashboard/`。在 GitHub Desktop 的 **Current repository** 选择 `rebound-model`，核对路径和远程地址，再建立 `ui-dashboard` 分支。提交看板改动后选择 **Publish branch / Push origin**，通过 Pull request 合并到小组 main。只把个人仓库的远程地址改成小组地址会遇到两份独立提交历史，不采用这种方式。

`build_release.py` 保留已有个人源码目录，`build_team_delivery.py` 按明确清单更新小组克隆里的看板文件。数据库、模型导出包、密钥、辩论账本、聊天记录、日志和检查截图均不随源码上传。成员模型与辩论源文件不覆盖；顶层 README 增加看板入口。

`Publish repository` 用于在自己的账号下创建新仓库，不能向另一位成员已有的仓库提交代码。公共仓库阅读与克隆不需要邀请；向小组推送分支需要仓主通过 **Settings → Collaborators → Add people** 邀请你的 GitHub 账号，并由你接受邀请。合并完成后，小组 main 才会显示看板新增目录。

数据来源为 [Tushare Pro](https://tushare.pro)，日线及股票资料口径分别见[日线接口](https://tushare.pro/document/2?doc_id=27)与[每日指标](https://tushare.pro/document/2?doc_id=32)。模型、回测与辩论工作流来自小组成员的 [rebound-model](https://github.com/XKKX33/rebound-model)。看板负责数据适配、栏目联动、交互图表、用户表达、版式与共享入口。
