"""配置：`config/settings.yaml` 的加载与校验。

只有一份配置文件，分五段：

    data     库在哪、新鲜度闸门往前查多少天、一天至少多少行算完整、几点算收盘
    sync     每次回补多少个开市日、日历往后拉多少天、要不要同步涨跌停价
    tushare  token 从哪个环境变量读、超时、重试、限流间隔
    model    用哪一版股票池（nocap / cap70）、哪种重训频率（M / Q / Y）
    pick     每天选几只、每个行业最多几只

特征、标签、XGBoost 参数写在代码里（`rebound/params.py`、`rebound/features.py`、`rebound/labels.py`），
不放进配置：改其中任何一项都要重新训练，否则打分用的模型和股票池对不上。

校验在加载时做完，拼错的键、类型不对、取值越界都直接报 `ConfigError`，不留到运行时才发现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "ROOT",
    "ConfigError",
    "DataConfig",
    "DebateConfig",
    "DebateLlmConfig",
    "ModelConfig",
    "PickConfig",
    "Settings",
    "SyncConfig",
    "Universe",
    "load_settings",
]

# 仓库根 = 本文件上两级（rebound/config.py -> rebound -> 仓库根）
ROOT = Path(__file__).resolve().parents[1]

# 可省略的键的默认值。写成模块常量而不是去读数据类的类属性：`slots=True` 的数据类上，
# 类属性是槽描述符，不是默认值。
DEFAULT_EXCHANGE = "SSE"
DEFAULT_VARIANT = "nocap"
DEFAULT_CADENCE = "M"
DEFAULT_TOP_N = 6
DEFAULT_MAX_PER_INDUSTRY = 2
DEFAULT_CANDIDATE_LIMIT = 260

DEFAULT_DEBATE_CANDIDATES = 20
DEFAULT_DEBATE_MIN_DEBATED = 12
DEFAULT_DEBATE_BARS = 20
DEFAULT_DEBATE_CONCURRENCY = 5
DEFAULT_DEBATE_TIMEOUT = 300
DEFAULT_LLM_BASE_URL = "https://api.stepfun.com/step_plan/v1"
DEFAULT_LLM_MODEL = "step-5-preview"
DEFAULT_LLM_API_KEY_ENV = "WORKBENCH_AI_API_KEY"
DEFAULT_LLM_REASONING_EFFORT = "high"
DEFAULT_LLM_TEMPERATURE = 0.2
DEFAULT_LLM_MAX_TOKENS = 16384
REASONING_EFFORTS = frozenset({"off", "minimal", "low", "medium", "high"})


class ConfigError(RuntimeError):
    """配置缺失、类型不符或自相矛盾。一律在加载期抛，不留到运行期。"""


@dataclass(frozen=True, slots=True)
class Universe:
    """股票池的硬筛条件（两个版本见 `rebound.params.VARIANTS`）。"""

    board: str = "mainboard"
    exclude_st: bool = True
    #: 单价上限（元，收盘价 < 它才进池）。`None` = 不设上限（不限价版）。
    price_max: float | None = 70.0
    min_amount_yi: float = 0.8
    #: 单价下限（元，收盘价 >= 它才进池）。`None` = 不设下限。
    price_min: float | None = None

    @property
    def min_amount_qian(self) -> float:
        """Tushare 的 amount 单位是千元。0.8 亿元 = 80000 千元。"""
        return self.min_amount_yi * 100_000.0


@dataclass(frozen=True, slots=True)
class DataConfig:
    db_path: Path
    #: 新鲜度闸门往前检查多少个开市日的日线（特征要看前 120 根，股票池要求至少 150 根）。
    history_bars: int
    #: 一个交易日的日线少于这么多行，算没收盘或残缺。
    min_daily_rows: int
    #: 当天日线算定稿的时刻（北京时间，"HH:MM"）。
    session_close_time: str

    @property
    def close_time(self) -> time:
        hour, _, minute = self.session_close_time.strip().partition(":")
        return time(int(hour), int(minute))


@dataclass(frozen=True, slots=True)
class SyncConfig:
    """增量同步的窗口。`days` 是"每次向前回补多少个开市日"。"""

    days: int
    calendar_lookahead_days: int
    with_limits: bool
    exchange: str = DEFAULT_EXCHANGE


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """用哪一套打分：股票池版本 × 重训频率。"""

    variant: str = DEFAULT_VARIANT
    cadence: str = DEFAULT_CADENCE


@dataclass(frozen=True, slots=True)
class PickConfig:
    """每天怎么从打分里取名单：分数从高到低，每个行业最多 `max_per_industry` 只，取前 `top_n`。"""

    top_n: int = DEFAULT_TOP_N
    max_per_industry: int = DEFAULT_MAX_PER_INDUSTRY
    #: 只在分数前这么多只里做行业去重（取够 top_n 就停，平时用不到这么多）。
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT


@dataclass(frozen=True, slots=True)
class DebateLlmConfig:
    """AI 辩论使用的大模型接口参数（密钥不放进配置，只写环境变量名）。"""

    base_url: str = DEFAULT_LLM_BASE_URL
    model: str = DEFAULT_LLM_MODEL
    api_key_env: str = DEFAULT_LLM_API_KEY_ENV
    reasoning_effort: str = DEFAULT_LLM_REASONING_EFFORT
    temperature: float = DEFAULT_LLM_TEMPERATURE
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS


@dataclass(frozen=True, slots=True)
class DebateConfig:
    """AI 辩论流水线参数（终选只数与行业上限复用 `pick` 段）。"""

    candidates: int = DEFAULT_DEBATE_CANDIDATES
    min_debated: int = DEFAULT_DEBATE_MIN_DEBATED
    bars: int = DEFAULT_DEBATE_BARS
    concurrency: int = DEFAULT_DEBATE_CONCURRENCY
    call_timeout_seconds: int = DEFAULT_DEBATE_TIMEOUT
    llm: DebateLlmConfig = field(default_factory=DebateLlmConfig)


@dataclass(frozen=True, slots=True)
class Settings:
    data: DataConfig
    sync: SyncConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    pick: PickConfig = field(default_factory=PickConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    root: Path = ROOT
    #: 原样保留的段落（目前只有 `tushare`），按名字取：`settings.section("tushare")`。
    sections: Mapping[str, Any] = field(default_factory=dict)

    def section(self, name: str) -> Mapping[str, Any]:
        try:
            return self.sections[name]
        except KeyError:
            raise ConfigError(f"settings.yaml 缺少段落 {name!r}") from None


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} 的顶层必须是映射，实际是 {type(raw).__name__}")
    return raw


def _section(raw: Mapping[str, Any], name: str, path: Path, *, required: bool = True) -> dict[str, Any]:
    value = raw.get(name)
    if value is None:
        if required:
            raise ConfigError(f"{path} 缺少段落 {name!r}")
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{path} 的 {name!r} 必须是映射")
    return value


def _require(section: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ConfigError(f"{where} 缺少键 {key!r}")
    value = section[key]
    if value is None:
        raise ConfigError(f"{where} 的 {key!r} 不能为空")
    return value


def _check_keys(section: Mapping[str, Any], allowed: set[str], where: str) -> None:
    """拼错的键直接报错：不然它会被静默忽略，配置看起来改了、实际没生效。"""
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigError(f"{where} 有不认识的键 {unknown}；认识的键：{sorted(allowed)}")


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{where} 必须是正整数，收到 {value!r}")
    return value


def _int_in_range(value: Any, where: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low or value > high:
        raise ConfigError(f"{where} 必须是 {low}~{high} 之间的整数，收到 {value!r}")
    return value


def _non_empty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} 必须是非空字符串，收到 {value!r}")
    return value.strip()


def _parse_debate(raw: Mapping[str, Any], picks: PickConfig, path: Path) -> DebateConfig:
    import math

    debate = _section(raw, "debate", path, required=False)
    _check_keys(
        debate,
        {"candidates", "min_debated", "bars", "concurrency", "call_timeout_seconds", "llm"},
        "settings.debate",
    )
    candidates = _int_in_range(
        debate.get("candidates", DEFAULT_DEBATE_CANDIDATES), "settings.debate.candidates", 1, 50
    )
    min_debated = _int_in_range(
        debate.get("min_debated", DEFAULT_DEBATE_MIN_DEBATED), "settings.debate.min_debated", 1, 50
    )
    if min_debated < picks.top_n:
        raise ConfigError(f"settings.debate.min_debated（{min_debated}）不能小于 pick.top_n（{picks.top_n}）")
    if min_debated > candidates:
        raise ConfigError(
            f"settings.debate.min_debated（{min_debated}）不能大于 candidates（{candidates}）"
        )
    bars = _int_in_range(debate.get("bars", DEFAULT_DEBATE_BARS), "settings.debate.bars", 5, 60)
    concurrency = _int_in_range(
        debate.get("concurrency", DEFAULT_DEBATE_CONCURRENCY), "settings.debate.concurrency", 1, 16
    )
    timeout = _int_in_range(
        debate.get("call_timeout_seconds", DEFAULT_DEBATE_TIMEOUT),
        "settings.debate.call_timeout_seconds",
        1,
        10_800,
    )

    llm_raw = debate.get("llm")
    if llm_raw is None:
        if "llm" in debate:
            raise ConfigError("settings.debate.llm 不能为空")
        llm_raw = {}
    elif not isinstance(llm_raw, dict):
        raise ConfigError("settings.debate.llm 必须是映射")
    _check_keys(
        llm_raw,
        {"base_url", "model", "api_key_env", "reasoning_effort", "temperature", "max_tokens"},
        "settings.debate.llm",
    )
    effort = _non_empty_str(
        llm_raw.get("reasoning_effort", DEFAULT_LLM_REASONING_EFFORT),
        "settings.debate.llm.reasoning_effort",
    )
    if effort not in REASONING_EFFORTS:
        raise ConfigError(
            f"settings.debate.llm.reasoning_effort 只能是 {sorted(REASONING_EFFORTS)}，收到 {effort!r}"
        )
    temp_raw = llm_raw.get("temperature", DEFAULT_LLM_TEMPERATURE)
    if isinstance(temp_raw, bool) or not isinstance(temp_raw, (int, float)) or not math.isfinite(float(temp_raw)):
        raise ConfigError(f"settings.debate.llm.temperature 必须是有限数字，收到 {temp_raw!r}")
    temperature = float(temp_raw)
    if not 0.0 <= temperature <= 2.0:
        raise ConfigError(f"settings.debate.llm.temperature 必须在 0~2 之间，收到 {temperature!r}")

    llm = DebateLlmConfig(
        base_url=_non_empty_str(llm_raw.get("base_url", DEFAULT_LLM_BASE_URL), "settings.debate.llm.base_url"),
        model=_non_empty_str(llm_raw.get("model", DEFAULT_LLM_MODEL), "settings.debate.llm.model"),
        api_key_env=_non_empty_str(
            llm_raw.get("api_key_env", DEFAULT_LLM_API_KEY_ENV), "settings.debate.llm.api_key_env"
        ),
        reasoning_effort=effort,
        temperature=temperature,
        max_tokens=_int_in_range(
            llm_raw.get("max_tokens", DEFAULT_LLM_MAX_TOKENS), "settings.debate.llm.max_tokens", 1, 32768
        ),
    )
    return DebateConfig(
        candidates=candidates,
        min_debated=min_debated,
        bars=bars,
        concurrency=concurrency,
        call_timeout_seconds=timeout,
        llm=llm,
    )


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_settings(path: Path | None = None, *, root: Path | None = None) -> Settings:
    """读 settings.yaml。显式配置位于 <项目>/config/，相对数据路径从项目根解析。"""
    from rebound import params  # 延迟导入：params 依赖本模块的 Universe

    if path is None:
        root = ROOT if root is None else Path(root).resolve()
        path = root / "config" / "settings.yaml"
    else:
        path = Path(path).resolve()
        root = path.parent.parent if root is None else Path(root).resolve()
    raw = _read_yaml(path)
    _check_keys(raw, {"data", "sync", "tushare", "model", "pick", "debate"}, str(path))

    data = _section(raw, "data", path)
    sync = _section(raw, "sync", path)
    tushare = _section(raw, "tushare", path)
    model = _section(raw, "model", path, required=False)
    pick = _section(raw, "pick", path, required=False)
    _check_keys(data, {"db_path", "history_bars", "min_daily_rows", "session_close_time"}, "settings.data")
    _check_keys(sync, {"days", "calendar_lookahead_days", "with_limits", "exchange"}, "settings.sync")
    _check_keys(model, {"variant", "cadence"}, "settings.model")
    _check_keys(pick, {"top_n", "max_per_industry", "candidate_limit"}, "settings.pick")

    close = str(_require(data, "session_close_time", "settings.data"))
    try:
        hour, sep, minute = close.strip().partition(":")
        time(int(hour), int(minute))
        if not sep:
            raise ValueError
    except ValueError:
        raise ConfigError(f"settings.data.session_close_time 应写成 \"HH:MM\"，收到 {close!r}") from None

    variant = str(model.get("variant", DEFAULT_VARIANT))
    cadence = str(model.get("cadence", DEFAULT_CADENCE))
    if variant not in params.VARIANTS:
        raise ConfigError(f"settings.model.variant 只能是 {sorted(params.VARIANTS)}，收到 {variant!r}")
    if cadence not in params.CADENCES:
        raise ConfigError(f"settings.model.cadence 只能是 {sorted(params.CADENCES)}，收到 {cadence!r}")

    with_limits = _require(sync, "with_limits", "settings.sync")
    if not isinstance(with_limits, bool):
        raise ConfigError(f"settings.sync.with_limits 必须是布尔值，收到 {with_limits!r}")

    picks = PickConfig(
        top_n=_positive_int(pick.get("top_n", DEFAULT_TOP_N), "settings.pick.top_n"),
        max_per_industry=_positive_int(pick.get("max_per_industry", DEFAULT_MAX_PER_INDUSTRY),
                                       "settings.pick.max_per_industry"),
        candidate_limit=_positive_int(pick.get("candidate_limit", DEFAULT_CANDIDATE_LIMIT),
                                      "settings.pick.candidate_limit"),
    )
    if picks.candidate_limit < picks.top_n:
        raise ConfigError(f"settings.pick.candidate_limit（{picks.candidate_limit}）不能小于 top_n（{picks.top_n}）")

    debate_cfg = _parse_debate(raw, picks, path)

    return Settings(
        data=DataConfig(
            db_path=_resolve(root, _require(data, "db_path", "settings.data")),
            history_bars=_positive_int(_require(data, "history_bars", "settings.data"), "settings.data.history_bars"),
            min_daily_rows=_positive_int(_require(data, "min_daily_rows", "settings.data"),
                                         "settings.data.min_daily_rows"),
            session_close_time=close,
        ),
        sync=SyncConfig(
            days=_positive_int(_require(sync, "days", "settings.sync"), "settings.sync.days"),
            calendar_lookahead_days=_positive_int(_require(sync, "calendar_lookahead_days", "settings.sync"),
                                                  "settings.sync.calendar_lookahead_days"),
            with_limits=with_limits,
            exchange=str(sync.get("exchange", DEFAULT_EXCHANGE)),
        ),
        model=ModelConfig(variant=variant, cadence=cadence),
        pick=picks,
        debate=debate_cfg,
        root=root,
        sections={"tushare": dict(tushare)},
    )
