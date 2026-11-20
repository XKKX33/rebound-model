"""`rebound debate` 子命令解析与文本展示。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable

from rebound import config
from rebound import params as model_params
from rebound.debate import ml_bridge, runs, service
from rebound.debate.contracts import AGENT_ROLES, ContractError
from rebound.debate.methodology import load_methodology
from rebound.debate.sidecar import Sidecar, SidecarError, resolve_api_key

EXIT_DEBATE = 5


def register_debate_parser(sub: argparse._SubParsersAction, parse_date: Callable[[str], date]) -> None:
    debate = sub.add_parser("debate", help="AI 多角色辩论复核（可选，不改模型与跟踪口径）")
    debate_sub = debate.add_subparsers(dest="debate_command", required=True)

    run = debate_sub.add_parser("run", help="运行当日或指定交易日的 AI 辩论（支持断点续跑）")
    run.add_argument("--as-of", type=parse_date, default=None, help="交易日 YYYY-MM-DD（缺省过新鲜度闸门）")
    run.add_argument("--variant", choices=tuple(model_params.VARIANTS), default=None, help="股票池版本（缺省取配置）")
    run.add_argument("--cadence", choices=tuple(model_params.CADENCES), default=None, help="重训频率（缺省取配置）")
    run.add_argument(
        "--model",
        dest="model_source",
        choices=ml_bridge.ML_MODEL_IDS,
        default=None,
        help="选股模型（xgb_F69 / rf_F69 / lstm_SF69）",
    )
    run.add_argument("--dry-run", action="store_true", help="只组装请求并跑侧车 --validate，不调模型不落盘")
    run.add_argument("--db", type=Path, default=None, help="数据库路径")
    run.add_argument("--json", action="store_true", help="输出 JSON")

    show = debate_sub.add_parser("show", help="查看某一天的全部辩论结果与正式名单")
    show.add_argument("--as-of", type=parse_date, default=None, help="交易日 YYYY-MM-DD（缺省取最近运行日）")
    show.add_argument("--variant", choices=tuple(model_params.VARIANTS), default=None, help="股票池版本（缺省取配置）")
    show.add_argument("--cadence", choices=tuple(model_params.CADENCES), default=None, help="重训频率（缺省取配置）")
    show.add_argument(
        "--model",
        dest="model_source",
        choices=ml_bridge.ML_MODEL_IDS,
        default=None,
        help="选股模型（xgb_F69 / rf_F69 / lstm_SF69）",
    )
    show.add_argument("--db", type=Path, default=None, help="数据库路径")
    show.add_argument("--json", action="store_true", help="输出 JSON")

    check = debate_sub.add_parser("check", help="检查 Node 版本、侧车依赖、密钥、方法论与配置")
    check.add_argument("--db", type=Path, default=None, help="数据库路径（预留参数，保持入口一致）")

    ui = debate_sub.add_parser("ui", help="启动本地 Web UI，实时查看辩论工作流、流式思考与最终名单")
    ui.add_argument("--db", type=Path, default=None, help="默认数据库路径")
    ui.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1，仅本机可访问）")
    ui.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    ui.add_argument("--no-open", action="store_true", help="启动后不自动打开系统浏览器")


def _dump(value: Any) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=1, default=str)
    sys.stdout.write("\n")


def _print_dry_run(snap: Any) -> None:
    req = snap.request
    print(f"演练（--validate 通过，未调模型、未写运行目录）：信号日 {req.as_of}")
    print(f"送辩候选（共 {len(req.candidates)} 只，{model_params.VARIANT_NAMES[req.context.variant]}，"
          f"{model_params.CADENCE_NAMES[req.context.cadence]}）：")
    for c in req.candidates:
        flags = "、".join(f.code for f in c.flags) if c.flags else "—"
        print(f"  第 {c.model.rank:>2} 名  {c.ts_code}  {c.name:<8}  {c.industry:<8}  flags={flags}")
    print("\n模型原名单：")
    for item in req.model_picks:
        tag = "" if item.debated else "（送辩范围外）"
        print(f"  第 {item.rank:>2} 名  {item.ts_code}  {item.name:<8}  {item.industry:<8}{tag}")
    print(f"\n角色（{len(AGENT_ROLES)} 个）：{', '.join(AGENT_ROLES)}")
    print(f"预计调用次数：{len(req.candidates) * 6 + 1} 次（{len(req.candidates)} × 6 + 1）")
    print(f"请求字符数：{len(req.to_line())}")


def print_final_summary(final: dict[str, Any], *, cached: bool = False) -> None:
    tag = "（命中已冻结运行，未重调模型）" if cached else ""
    print(
        f"{final['as_of']} AI 辩论名单（{model_params.VARIANT_NAMES[final['variant']]}，"
        f"{model_params.CADENCE_NAMES[final['cadence']]}，run_id={final['run_id']}）{tag}"
    )
    if final.get("shortage_reason"):
        print(f"  名额说明：{final['shortage_reason']}")
    for row in final["picks"]:
        score = row["score"]
        score_txt = str(int(score)) if score == int(score) else f"{score:g}"
        print(
            f"  AI 第 {row['rank']} 名  {row['ts_code']}  {row['name']:<8}  {row['industry']:<8}  "
            f"模型第 {row['model_rank']} 名  {row['verdict']} {score_txt}  {row['reason']}"
        )
    print(f"整体理由：{final['reason']}")

    ai_codes = [row["ts_code"] for row in final["picks"]]
    model_codes = [row["ts_code"] for row in final["model_picks"]]
    model_set = set(model_codes)
    ai_set = set(ai_codes)
    kept = [row["ts_code"] for row in final["picks"] if row["ts_code"] in model_set]
    added = [c for c in ai_codes if c not in model_set]
    dropped = [c for c in model_codes if c not in ai_set]
    print(
        f"与模型名单对照（模型 {len(model_codes)} 只，重合 {final['overlap_count']} 只）："
        f"保留 {kept or '无'}；新进 {added or '无'}；换出 {dropped or '无'}"
    )
    if final.get("failed_stocks"):
        print(f"未辩完（{final['failed_count']} 只）：" + "、".join(
            f"{f['ts_code']}({f['role']}: {f['error']})" for f in final["failed_stocks"]
        ))
    usage = final["usage"]
    print(
        f"用量：成功 {usage['calls']} 次，失败 {usage['failed_calls']} 次，"
        f"输入 {usage['input_tokens']} / 输出 {usage['output_tokens']} tokens"
    )


def _print_show_text(day_str: str, entries: list[dict[str, Any]]) -> None:
    print(f"{day_str} AI 辩论运行列表（共 {len(entries)} 条）：")
    for idx, entry in enumerate(entries, start=1):
        if not entry["completed"]:
            print(
                f"\n[{idx}] 未完成（{entry['hash_prefix']}）：已辩完 "
                f"{entry['debated_count']}/{entry['submitted_count']} 只"
            )
            continue
        official = " [当日正式结果]" if entry["official"] else ""
        print(f"\n[{idx}] 运行 {entry['hash_prefix']}{official}  完成于 {entry['completed_at']}")
        print_final_summary(entry["final"])
        result = entry.get("result")
        if result and result.get("verdicts"):
            print("  风控详情：")
            for v in result["verdicts"]:
                s = v["score"]
                s_txt = str(int(s)) if s == int(s) else f"{s:g}"
                print(f"    {v['ts_code']}  {v['verdict']} {s_txt}  {v['thesis']}")


def _run_check(settings: config.Settings) -> int:
    ok = True
    checks: list[tuple[str, bool, str]] = []
    checks.append(("配置加载与校验", True, f"模型 {settings.debate.llm.model}，送辩 {settings.debate.candidates} 只"))

    try:
        meth = load_methodology(root=settings.root)
        checks.append(("方法论加载", True, f"共 {len(meth.role_briefs)} 个角色"))
    except ContractError as exc:
        ok = False
        checks.append(("方法论加载", False, str(exc)))

    try:
        health = Sidecar(settings).health()
        checks.append(("Node 与侧车健康", True, f"Node {health.get('node')}，协议 v{health.get('protocol_version')}"))
    except (SidecarError, ContractError) as exc:
        ok = False
        checks.append(("Node 与侧车健康", False, str(exc)))

    try:
        resolve_api_key(settings)
        checks.append(("密钥环境变量", True, f"{settings.debate.llm.api_key_env} 已设置"))
    except SidecarError as exc:
        ok = False
        checks.append(("密钥环境变量", False, str(exc)))

    for label, passed, detail in checks:
        mark = "通过" if passed else "失败"
        print(f"[{mark}] {label}：{detail}")
    return 0 if ok else EXIT_DEBATE


def run_debate_command(
    args: argparse.Namespace,
    settings: config.Settings,
    *,
    target: Path,
    fresh_day: Callable[[config.Settings, Path], date | None],
    exit_stale: int,
    say: Callable[[str], None],
) -> int:
    command = args.debate_command
    if command == "check":
        return _run_check(settings)
    if command == "ui":
        from rebound.debate.web import serve_ui

        return serve_ui(
            settings,
            default_db=args.db,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )

    variant = args.variant or settings.model.variant
    cadence = args.cadence or settings.model.cadence
    model_source = getattr(args, "model_source", None)
    if command == "show":
        day_str, entries = runs.list_runs(
            target, variant=variant, cadence=cadence, as_of=args.as_of, model_source=model_source
        )
        if not entries:
            series = runs.series_name(variant, cadence, model_source)
            msg = f"{args.as_of or '当前'} 没有 {series} 的辩论运行记录"
            print(msg, file=sys.stderr)
            return EXIT_DEBATE
        if args.json:
            _dump({"as_of": day_str, "variant": variant, "cadence": cadence, "model_source": model_source, "runs": entries})
        else:
            assert day_str is not None
            _print_show_text(day_str, entries)
        return 0

    day = args.as_of if args.as_of is not None else (None if ml_bridge.is_ml_model(model_source) else fresh_day(settings, target))
    if day is None and not ml_bridge.is_ml_model(model_source):
        return exit_stale

    if args.dry_run:
        snap, _receipt = service.dry_run(
            target, settings=settings, as_of=day, variant=variant, cadence=cadence, model_source=model_source
        )
        if args.json:
            _dump(snap.request.payload())
        else:
            _print_dry_run(snap)
        return 0

    log = (lambda _line: None) if args.json else say
    out = service.run_debate(
        target, settings=settings, as_of=day, variant=variant, cadence=cadence, model_source=model_source, log=log
    )
    if args.json:
        _dump(out["final"])
    else:
        print_final_summary(out["final"], cached=bool(out["cached"]))
    return 0
