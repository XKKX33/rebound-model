"""命令行入口。

    python -m rebound sync --days 3500             # 首次建库：从 Tushare 回补约 14 年的开市日（幂等，已有的跳过）
    python -m rebound sync                         # 平时：补最近 10 个开市日
    python -m rebound data import-history --source old.duckdb --dry-run   # 从别的库搬历史：先演练
    python -m rebound model train                  # 2015-01 起每月第一个开市日重训一次（两个股票池版本），训过的跳过
    python -m rebound model score                  # 补齐每天的全池打分（两个版本 × 每年/季/月重训），已有的不改写
    python -m rebound pick                         # 最新已收盘交易日的名单（数据不新就拒绝）
    python -m rebound pick --as-of 2024-03-15      # 回放某一天的名单
    python -m rebound daily                        # 收盘后一条命令：同步 → 补训补分 → 选股 → 更新纸面跟踪
    python -m rebound capital --capital 20000      # 本金测算：150 条持股规则 × 三种重训频率，逐日模拟真实账户
    python -m rebound track start --rule DB-K2-b6-o1c5 --capital 20000   # 开一个纸面跟踪
    python -m rebound track list | show | refresh | stop

这个模块只解析参数、调各层的函数、把结果印成能读的字；不含业务判断。
退出码：0 成功；1 演练没通过；2 参数错；3 数据不新（拒绝选股）；4 数据或模型出错。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from rebound import config
from rebound import params as model_params
from rebound.clock import local_is_beijing
from rebound.data import freshness as fresh
from rebound.data.db import WriteStore
from rebound.data.errors import DataError
from rebound.data.sync import AuditReport, DateSync, SkippedDate, SyncReport

EXIT_REFUSED = 1
EXIT_STALE = 3
EXIT_DATA = 4


def _parse_date(raw: str) -> date:
    """接受 `2026-10-14` 与 `20261014` 两种写法。"""
    text = raw.strip().replace("/", "-")
    if len(text) == 8 and text.isdigit():
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD 或 YYYYMMDD，收到 {raw!r}") from exc


def _say(line: str) -> None:
    print(line, flush=True)


def _dump(value: Any) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=1, default=str)
    sys.stdout.write("\n")


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #


def _add_cost_flags(target: argparse.ArgumentParser) -> None:
    target.add_argument("--capital", type=float, default=20_000.0, help="本金（元，默认 2 万）")
    target.add_argument("--variant", choices=tuple(model_params.VARIANTS), default=model_params.DEFAULT_VARIANT,
                        help="股票池版本（默认不限价）")
    target.add_argument("--min-commission", type=float, default=5.0, help="每笔最低佣金（元，默认 5；免五填 0）")
    target.add_argument("--commission", type=float, default=0.0005, help="佣金率（默认万五）")
    target.add_argument("--stamp", type=float, default=0.0005, help="印花税率（卖出，默认万五）")
    target.add_argument("--transfer", type=float, default=0.00001, help="过户费率（默认十万分之一）")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rebound", description="反弹模型：A 股主板短线选股的机器学习全流程")
    parser.add_argument("--config", type=Path, default=None,
                        help="项目的 config/settings.yaml（放在子命令前；缺省取源码仓库配置）")
    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="增量同步行情数据（幂等）")
    sync.add_argument("--days", type=int, default=None, help="向前回补多少个开市日（缺省取 settings.sync.days）")
    sync.add_argument("--db", type=Path, default=None, help="数据库路径")
    sync.add_argument("--no-audit", action="store_true", help="跳过同步后的缺口审计")
    sync.add_argument("--json", action="store_true", help="输出 JSON")

    data = sub.add_parser("data", help="一次性的数据维护")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    history = data_sub.add_parser("import-history", help="从别的库补本库缺的历史（先备份；重叠日逐行核对，对不上就拒绝）")
    history.add_argument("--source", type=Path, required=True, help="源库（只读挂上）")
    history.add_argument("--db", type=Path, default=None, help="本库（缺省取 settings 里的库）")
    history.add_argument("--dry-run", action="store_true", help="只核对、只报计划，不备份不写")
    history.add_argument("--json", action="store_true", help="输出 JSON")

    model = sub.add_parser("model", help="训练、逐日打分")
    model_sub = model.add_subparsers(dest="model_command", required=True)
    for name, text in (("train", "2015-01 起每月第一个开市日重训一次（不限价、限 70 元各一套），模型存在库旁边；训过的跳过"),
                       ("score", "补齐 2015-01 以来每天的打分（两个版本 × 每年/季/月重训）；已有的日子不改写")):
        cmd = model_sub.add_parser(name, help=text)
        cmd.add_argument("--db", type=Path, default=None, help="数据库路径（模型目录挨着它）")
        cmd.add_argument("--json", action="store_true", help="输出 JSON")

    pick = sub.add_parser("pick", help="某个交易日收盘后的名单（只读）")
    pick.add_argument("--as-of", type=_parse_date, default=None,
                      help="交易日 YYYY-MM-DD（缺省 = 最新已收盘交易日，数据不新就拒绝）")
    pick.add_argument("--variant", choices=tuple(model_params.VARIANTS), default=None, help="股票池版本（缺省取配置）")
    pick.add_argument("--cadence", choices=tuple(model_params.CADENCES), default=None, help="重训频率（缺省取配置）")
    pick.add_argument("--refresh", action="store_true", help="先补训到期的模型、补没打分的日子（会写库和模型目录）")
    pick.add_argument("--db", type=Path, default=None, help="数据库路径")
    pick.add_argument("--json", action="store_true", help="输出 JSON")

    daily = sub.add_parser("daily", help="收盘后一条命令：同步 → 补训补分 → 选股 → 更新纸面跟踪")
    daily.add_argument("--no-sync", action="store_true", help="不同步，数据不新就直接拒绝")
    daily.add_argument("--db", type=Path, default=None, help="数据库路径")
    daily.add_argument("--json", action="store_true", help="输出 JSON")

    capital = sub.add_parser("capital", help="本金测算：150 条持股规则 × 三种重训频率，逐日模拟真实账户")
    _add_cost_flags(capital)
    capital.add_argument("--refresh", action="store_true", help="同样的输入算过也重算")
    capital.add_argument("--top", type=int, default=12, help="文本输出列前几名（默认 12）")
    capital.add_argument("--db", type=Path, default=None, help="数据库路径")
    capital.add_argument("--json", action="store_true", help="输出 JSON（全部 450 条规则）")

    track = sub.add_parser("track", help="纸面跟踪：照一条持股规则从某天起做的账户")
    track_sub = track.add_subparsers(dest="track_command", required=True)
    start = track_sub.add_parser("start", help="开一个跟踪（并设为当前显示项）")
    start.add_argument("--rule", required=True, help="持股规则全名，如 DB-K2-b6-o1c5")
    start.add_argument("--cadence", choices=tuple(model_params.CADENCES), default=model_params.DEFAULT_CADENCE,
                       help="重训频率（默认每月）")
    _add_cost_flags(start)
    start.add_argument("--start", type=_parse_date, default=None,
                       help="第一次开盘买的那天（缺省 = 最后一个打分日之后的开市日）")
    for name, text in (("list", "列出全部跟踪"), ("refresh", "重算全部未结束的跟踪")):
        track_sub.add_parser(name, help=text).add_argument("--db", type=Path, default=None, help="数据库路径")
    show = track_sub.add_parser("show", help="看一个跟踪：账户、持仓、下一个开市日的买卖清单")
    show.add_argument("track_id", nargs="?", default=None, help="跟踪编号（缺省 = 当前显示项）")
    show.add_argument("--json", action="store_true", help="输出 JSON")
    stop = track_sub.add_parser("stop", help="结束一个跟踪")
    stop.add_argument("track_id", nargs="?", default=None, help="跟踪编号（缺省 = 当前显示项）")
    for cmd in (start, show, stop):
        cmd.add_argument("--db", type=Path, default=None, help="数据库路径")
    return parser


# --------------------------------------------------------------------------- #
# 同步
# --------------------------------------------------------------------------- #


def _target(settings: config.Settings, db: Path | None) -> Path:
    return Path(db) if db is not None else settings.data.db_path


def _check_freshness(store: WriteStore, settings: config.Settings) -> fresh.Freshness:
    return fresh.check_freshness(store, exchange=settings.sync.exchange, min_rows=settings.data.min_daily_rows,
                                 windows={"daily": settings.data.history_bars}, close_time=settings.data.close_time)


def _progress_detail(record: DateSync | SkippedDate | object) -> str:
    if isinstance(record, SkippedDate):
        return f"{record.trade_date} 跳过（{record.reason}）：{record.detail}"
    if isinstance(record, DateSync):
        return (f"{record.trade_date} daily {record.daily_rows} 行、涨跌停价 {record.limit_rows} 行、"
                f"复权因子 {record.adj_rows} 行")
    return str(record)


def run_sync(settings: config.Settings, target: Path, *, days: int, with_audit: bool = True,
             log: Any = _say) -> tuple[SyncReport, AuditReport | None, fresh.Freshness]:
    """拉一次数据：日历和股票列表 → 历史名称 → 行情，然后体检和新鲜度闸门。

    名称必须在行情之前：名称取不全就整趟停下，新的一天的行情还没写。要是先写了行情，下次新鲜度闸门
    看到行情已齐、不再同步，模型就会拿旧名称（新戴帽的 ST 还不是 ST）给这一天打分，而分写下就不再改。
    """
    from dataclasses import replace

    from rebound.data.sync import audit, sync_market, sync_metadata, sync_names
    from rebound.data.tushare_client import tushare_source

    log(f"[同步] 准备回补 {days} 个开市日。")
    with WriteStore(target) as store:
        source = tushare_source(settings)
        calendar_rows, basic_rows = sync_metadata(
            store, source, exchange=settings.sync.exchange, days=days,
            calendar_lookahead_days=settings.sync.calendar_lookahead_days)
        try:
            name_rows = sync_names(store, source)
        except DataError:
            log("[同步] 历史名称没取全，这一趟没写新的行情（免得拿旧名称给新的一天打分），过几分钟再同步一次。")
            raise
        report = sync_market(
            store, source, exchange=settings.sync.exchange, days=days, min_rows=settings.data.min_daily_rows,
            with_limits=settings.sync.with_limits,
            on_progress=lambda index, total, record: log(f"[{index:>4}/{total}] {_progress_detail(record)}"))
        report = replace(report, calendar_rows=calendar_rows, stock_basic_rows=basic_rows, name_rows=name_rows)
        audited = audit(store, exchange=settings.sync.exchange, min_rows=settings.data.min_daily_rows) \
            if with_audit else None
        gate = _check_freshness(store, settings)
    return report, audited, gate


def _print_sync_text(report: SyncReport, audited: AuditReport | None) -> None:
    print(f"同步完成：补齐 {len(report.synced)} 个交易日，跳过 {len(report.skipped)} 个，"
          f"写入 {report.written_rows} 行（其中历史名称 {report.name_rows} 行）")
    print("\n覆盖范围")
    for item in report.coverage:
        span = f"{item.first} … {item.last}" if item.rows else "（空）"
        print(f"  {item.table:<14} {item.rows:>10} 行   {span}")
    if audited is None:
        return
    print()
    if audited.calendar is None:
        print("缺口审计：本地日历是空的 —— 先跑 sync 把日历拉下来")
        return
    print(f"缺口审计（{audited.today}，日历 {audited.calendar[0]}~{audited.calendar[1]}）")
    if audited.empty_tables:
        print(f"  一行数据都没有：{'、'.join(audited.empty_tables)}")
    for table, days in audited.holes.items():
        if days:
            more = "…" if len(days) > 8 else ""
            print(f"  {table} 缺 {len(days)} 天 —— {'、'.join(str(d) for d in days[:8])}{more}")
    if audited.backlog:
        more = "…" if len(audited.backlog) > 8 else ""
        print(f"  落后 {len(audited.backlog)} 天 —— {'、'.join(str(d) for d in audited.backlog[:8])}{more}")
    if audited.healthy:
        print("  结论：健康")


def _run_sync(args: argparse.Namespace, settings: config.Settings) -> int:
    days = int(args.days if args.days is not None else settings.sync.days)
    report, audited, gate = run_sync(settings, _target(settings, args.db), days=days, with_audit=not args.no_audit,
                                     log=(lambda _line: None) if args.json else _say)
    if args.json:
        _dump({"report": report.as_dict(), "audit": audited.as_dict() if audited else None,
               "freshness": gate.as_dict()})
    else:
        _print_sync_text(report, audited)
        print("\n" + ("数据是最新的：可以选股" if gate.ok else fresh.describe(gate)))
    return 0


def _run_import_history(args: argparse.Namespace, settings: config.Settings) -> int:
    from rebound.data.history_import import import_history

    report = import_history(args.source, _target(settings, args.db), dry_run=args.dry_run,
                            log=(lambda _line: None) if args.json else _say)
    if args.json:
        _dump(report.as_dict())
        return EXIT_REFUSED if report.refused else 0
    if report.dry_run:
        head = "演练（没备份、没写）"
    elif report.written:
        head = f"已导入，备份在 {report.backup}"
    else:
        head = "没有要补的，本库没动"
    print(f"{head}\n  源库 {report.source}\n  本库 {report.target}")
    for plan in report.tables.values():
        span = f"{plan.first} … {plan.last}" if plan.first else "—"
        print(f"  {plan.table:<12} 补 {plan.insert:>9} 行 / {plan.insert_days:>5} 天  {span}  "
              f"重叠 {plan.overlap} 行，对不上 {plan.mismatch}，坏值 {plan.invalid}，"
              f"那几天只本库有 {plan.target_only} / 只源库有 {plan.source_only}")
        for sample in plan.samples:
            print(f"      {sample}")
    if report.refused:
        print("拒绝导入：重叠日有对不上的行，或要搬的行里有坏值。本库一行没动。")
    return EXIT_REFUSED if report.refused else 0


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #


def _run_model(args: argparse.Namespace, settings: config.Settings) -> int:
    target = _target(settings, args.db)
    log = (lambda _line: None) if args.json else _say
    if args.model_command == "train":
        from rebound.model.train import train

        report = train(target, log=log)
        summary = f"训练完成：新训 {len(report.trained)} 次重训，已有 {len(report.skipped)} 次（每个版本每个重训日算一次）"
    else:
        from rebound.model.score import fill_scores

        report = fill_scores(target, log=log)
        summary = f"打分完成：补了 {report.days} 个交易日，写入 {report.rows} 行"
    if args.json:
        _dump(report.as_dict())
    else:
        print(summary)
    return 0


# --------------------------------------------------------------------------- #
# 选股
# --------------------------------------------------------------------------- #


def _fresh_day(settings: config.Settings, target: Path) -> date | None:
    """数据新就返回最新已收盘交易日；不新就打印原因、返回 None。"""
    with WriteStore(target) as store:
        gate = _check_freshness(store, settings)
    if gate.ok and gate.expected is not None:
        return gate.expected
    print(fresh.describe(gate), file=sys.stderr)
    return None


def _pick(settings: config.Settings, target: Path, day: date, *, variant: str, cadence: str) -> Any:
    from rebound.select import pick_day

    return pick_day(target, as_of=day, variant=variant, cadence=cadence, top_n=settings.pick.top_n,
                    max_per_industry=settings.pick.max_per_industry, candidate_limit=settings.pick.candidate_limit)


def _pick_payload(result: Any) -> dict[str, Any]:
    cols = ["rank", "ts_code", "name", "industry", "close", "score", "pct"]
    return {"day": result.day.isoformat(), "variant": result.variant, "cadence": result.cadence,
            "fit_date": result.fit_date.isoformat(), "model_id": result.model_id, "stored": result.stored,
            "pool_size": len(result.pool), "differ": result.differ,
            "picks": result.picks.loc[:, cols].to_dict(orient="records")}


def _print_pick_text(result: Any) -> None:
    source = "库里存好的分数" if result.stored else "当场算的分数（库里还没有这天的分）"
    print(f"{result.day} 收盘后的名单（{model_params.VARIANT_NAMES[result.variant]}股票池，"
          f"{model_params.CADENCE_NAMES[result.cadence]}重训，重训日 {result.fit_date} 的模型；"
          f"全池 {len(result.pool)} 只，{source}）")
    if result.differ:
        print(f"  今天重算的股票池和当时存的差 {result.differ} 只（多半是后来补了改名记录），按当时存的。")
    for row in result.picks.to_dict(orient="records"):
        name = "" if row["name"] is None else str(row["name"])
        industry = "" if row["industry"] is None else str(row["industry"])
        print(f"  第 {row['rank']:>4} 名  {row['ts_code']}  {name:<8}  {industry:<8}  收盘 {row['close']:>8.2f}  "
              f"分位 {row['pct']:.3f}")
    print("下一个开市日开盘买；持股规则和卖点见 rebound capital / rebound track。")


def _run_pick(args: argparse.Namespace, settings: config.Settings) -> int:
    target = _target(settings, args.db)
    variant = args.variant or settings.model.variant
    cadence = args.cadence or settings.model.cadence
    day = args.as_of if args.as_of is not None else _fresh_day(settings, target)
    if day is None:
        return EXIT_STALE
    if args.refresh:
        from rebound.model.refresh import refresh

        refresh(target, log=(lambda _line: None) if args.json else _say)
    result = _pick(settings, target, day, variant=variant, cadence=cadence)
    if args.json:
        _dump(_pick_payload(result))
    else:
        _print_pick_text(result)
    return 0


def _run_daily(args: argparse.Namespace, settings: config.Settings) -> int:
    """同步 → 闸门 → 补训补分 → 选股 → 更新全部纸面跟踪。哪一步失败就停在那一步。"""
    from rebound import track
    from rebound.model.refresh import refresh

    target = _target(settings, args.db)
    log = (lambda _line: None) if args.json else _say
    if not args.no_sync:
        run_sync(settings, target, days=settings.sync.days, with_audit=False, log=log)
    day = _fresh_day(settings, target)
    if day is None:
        return EXIT_STALE
    refresh(target, log=log)
    result = _pick(settings, target, day, variant=settings.model.variant, cadence=settings.model.cadence)
    tracked = track.refresh_all(target, log=log)
    if args.json:
        _dump({"pick": _pick_payload(result), "tracking": {"updated": tracked["updated"],
                                                          "track_ids": tracked["track_ids"]}})
    else:
        _print_pick_text(result)
        print(f"纸面跟踪：更新了 {tracked['updated']} 个")
    return 0


# --------------------------------------------------------------------------- #
# 回测：本金测算
# --------------------------------------------------------------------------- #


def _request(args: argparse.Namespace) -> Any:
    from rebound.backtest.capital import CapitalRequest

    return CapitalRequest(capital=args.capital, variant=args.variant, commission=args.commission,
                          min_commission=args.min_commission, stamp=args.stamp, transfer=args.transfer)


def _run_capital(args: argparse.Namespace, settings: config.Settings) -> int:
    """本金测算只读库，结果存在库旁边的 `<库名>.capital/`（同样的输入再跑直接读）。"""
    from rebound.backtest import capital as capital_module

    out = capital_module.run(_target(settings, args.db), _request(args),
                             log=(lambda _line: None) if args.json else _say, refresh=args.refresh)
    if args.json:
        _dump(out)
        return 0
    _print_capital_text(out, max(1, int(args.top)))
    return 0


def _print_capital_text(out: dict[str, Any], top: int) -> None:
    from rebound.backtest.capital import PICKER_NAMES

    rows, req = out["rows"], out["request"]
    fee = "免五" if req["min_commission"] == 0 else f"最低 {req['min_commission']:g} 元"
    print(f"本金测算：{req['capital']:,.0f} 元，{out['variant_name']}，佣金万{req['commission'] * 1e4:g}（{fee}），"
          f"{out['first']} ~ {out['last']}（编号 {out['key']}）")
    pool = out["pool"]
    print(f"股票池等权：年化 {pool['cagr']:+.1f}%，最大回撤 {pool['max_dd']:.1f}%")
    walked, picker = out["walk_forward"], out["picker"]
    w = walked[picker]
    print(f"滚动样本外（2017 起每年只用之前的数据选，按{PICKER_NAMES[picker]}）：超额 {w['excess']:+.1f} "
          f"[{w['ci'][0]:+.1f}, {w['ci'][1]:+.1f}]，年化 {w['cagr']:+.1f}%，最大回撤 {w['max_dd']:.1f}%")
    if out["best_picker"] != picker:
        b = walked[out["best_picker"]]
        print(f"  （这一次滚动超额最高的是{PICKER_NAMES[out['best_picker']]}：{b['excess']:+.1f} "
              f"[{b['ci'][0]:+.1f}, {b['ci'][1]:+.1f}]，最大回撤 {b['max_dd']:.1f}%；只作参考，推荐按十年实验定的信息比率）")

    def line(r: dict[str, Any]) -> str:
        f = r["fresh"]
        starts = "/".join(f"{v['cagr']:+.1f}" for v in f["starts"].values()) if f else "—"
        return (f"{r['cadence_name']} {r['name']:<12} 年化 {r['cagr']:+6.1f}%  超额 {r['excess']:+6.1f}  "
                f"信息比率 {r['ir'] if r['ir'] is not None else '—'}  回撤 {r['max_dd']:6.1f}%  "
                f"期末 {r['final'] / 10000:,.1f} 万  换起点 {starts}")

    basis = ("按换起点体检：四个起点都赚、最差起点年化最高" if out["basis"] == "fresh"
             else f"按{PICKER_NAMES[picker]}今天会选的那条")
    print(f"\n推荐（{basis}）：{line(rows[out['recommended']])}" if out["recommended"] is not None
          else "\n推荐：没有四个起点都赚的规则")
    if out["recommended"] is None:
        print(f"  （{PICKER_NAMES[picker]}今天会选 {line(rows[out['picker_choice']])}，但它换起点会亏；不给新账户的买单）")
    elif out["recommended"] != out["picker_choice"]:
        why = ("小本金有最低佣金，十年实验以换起点体检为准" if out["small_capital"] else "它换起点会亏")
        print(f"  （{PICKER_NAMES[picker]}今天会选 {line(rows[out['picker_choice']])}；没推它：{why}）")
    if out["recommended"] is not None:
        new = rows[out["recommended"]]["new"]
        print(f"\n今天开新账户，{new['signal_day']} 之后的第一个开市日开盘买（股数按今天收盘价估）：")
        for b in new["buys"]:
            print(f"  {b['ts_code']} {b['name']:<8} 第 {b['rank']} 名  {b['shares']} 股 ≈ {b['amount']:,.0f} 元")
        if new["alternates"]:
            print("  备选（前面的买不进时按顺序顶上）：" + "、".join(f"{a['ts_code']} {a['name']}" for a in new["alternates"]))
    print(f"\n按全期年化排前 {top}：")
    for r in sorted(rows, key=lambda r: -r["cagr"])[:top]:
        print("  " + line(r))


# --------------------------------------------------------------------------- #
# 纸面跟踪
# --------------------------------------------------------------------------- #


def _print_track_text(out: dict[str, Any]) -> None:
    about = out["tracker"]
    print(f"跟踪 {out['track_id']}：{about['cadence_name']}重训 + {about['name']}（{about['plain']}），"
          f"{about['variant_name']}，本金 {about['capital']:,.0f} 元，{out['start']} 起步；打分到 {out['scored_through']}")
    if out["state"] == "pending":
        print(f"  {out['signal_day']} 收盘后才有第一份清单。")
        return
    summary = out["summary"]
    print(f"  账户 {summary['value']} 元，累计 {summary['return_pct']}%")
    for row in out["holdings"]:
        shares = "停牌" if row.get("halted") else f"{row.get('shares')} 股"
        print(f"  持有 {row['ts_code']} {row['name']}  {shares}  市值 {row['value']:,.0f}  "
              f"浮动 {row['return_pct']:+.2f}%  {row['bought']} 买入")
    plan = out["plan"]
    print(f"  下一个开市日（{out['next_open']}）：开盘买 {len(plan['buys'])} 只、尾盘卖 {len(plan['sells'])} 只")
    for item in plan["buys"]:
        print(f"    买 {item['ts_code']} {item['name']}  {item.get('shares')} 股 ≈ {item.get('amount', 0):,.0f} 元")
    for item in plan["sells"]:
        reason = f"（{item['reason']}）" if item.get("reason") else ""
        print(f"    卖 {item['ts_code']} {item['name']}  {item.get('shares')} 股{reason}")


def _run_track(args: argparse.Namespace, settings: config.Settings) -> int:
    from rebound import track

    target = _target(settings, args.db)
    command = args.track_command
    if command == "start":
        tracker = track.create(target, rule=args.rule, cadence=args.cadence, request=_request(args), start=args.start)
        print(f"开始跟踪 {tracker.track_id}：{tracker.rule}，{tracker.start} 第一次开盘买")
        return 0
    if command == "list":
        trackers, problems = track.list_trackers(target)
        current = track.selected(target)
        for t in trackers:
            mark = "*" if current is not None and t.track_id == current.track_id else " "
            state = "已结束" if t.closed_at else "进行中"
            print(f"{mark} {t.track_id}  {t.cadence} {t.rule:<16} 本金 {t.capital:>10,.0f}  {t.start} 起  {state}")
        for problem in problems:
            print(f"  读不了：{problem}", file=sys.stderr)
        return 0
    if command == "refresh":
        out = track.refresh_all(target, log=_say)
        print(f"更新了 {out['updated']} 个跟踪")
        return 0
    if command == "stop":
        tracker = track.stop(target, args.track_id)
        print(f"已结束跟踪 {tracker.track_id}")
        return 0
    tracker = track.active(target) if args.track_id is None else track.load(target, args.track_id)
    if tracker is None:
        print("还没有跟踪：先 rebound track start", file=sys.stderr)
        return EXIT_DATA
    out = track.compute(target, tracker)
    if args.json:
        _dump(out)
    else:
        _print_track_text(out)
    return 0


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

_COMMANDS = {
    "sync": _run_sync,
    "model": _run_model,
    "pick": _run_pick,
    "daily": _run_daily,
    "capital": _run_capital,
    "track": _run_track,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not local_is_beijing():
        print("提醒：本进程没能切成北京时间；新鲜度闸门和同步仍按北京时间判断，其余时间显示可能是本机时区。",
              file=sys.stderr)
    try:
        settings = config.load_settings(args.config)
        if args.command == "data":
            return _run_import_history(args, settings)
        return _COMMANDS[args.command](args, settings)
    except fresh.StaleDataError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_STALE
    except (DataError, config.ConfigError) as exc:
        print(f"出错：{exc}", file=sys.stderr)
        return EXIT_DATA
