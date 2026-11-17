"""行情来源：唯一碰网络的地方。

对外只暴露 `MarketDataSource` 这个 Protocol 与一个真实实现 `TushareClient`。
测试注入假来源（见 `tests/test_data_sync.py`），所以同步逻辑本身不需要网络。

行情类端点一律按 `trade_date` 一次拿全市场（一个交易日一次调用），不逐票拉。
"""

from __future__ import annotations

import os
import time
from datetime import date
from typing import Any, Protocol

import pandas as pd

from rebound.config import Settings
from rebound.data import schema
from rebound.data.errors import DataError, NetworkError, ValidationError
from rebound.data.validate import validated_snapshot

# 每个端点索要的字段 = 表结构本身。多要字段上游会报错，少要字段校验会失败，
# 所以不另抄一份清单。
SNAPSHOT_FIELDS: dict[str, str] = {
    table: ",".join(fields) for table, fields in schema.TABLE_FIELDS.items()
}


class MarketDataSource(Protocol):
    """同步逻辑依赖的全部外部能力。"""

    def trade_cal(self, *, start: date, end: date, exchange: str = "SSE") -> pd.DataFrame: ...

    def stock_basic(self) -> pd.DataFrame: ...

    def daily(self, *, trade_date: date) -> pd.DataFrame: ...

    def daily_limit(self, *, trade_date: date) -> pd.DataFrame: ...

    def adj_factor(self, *, trade_date: date) -> pd.DataFrame: ...

    def namechange(self) -> pd.DataFrame: ...


class TushareClient:
    """Tushare 的薄封装：重试、限速、字段选择、**以及校验**。

    校验放在这一层而不是同步层，是为了保证"从网络进来的 DataFrame"与
    "从假来源进来的 DataFrame"走**同一条**校验路径。
    """

    #: 上游地址。**必须显式指定**，不能依赖 tushare 库的默认值。
    #:
    #: 实测 2026-10-20：库 1.4.29 的 `ts.pro_api()` 默认打到
    #: `http://api.waditu.com/dataapi`，而那个域名当时返回 **502**，于是每个
    #: 端点都"成功"返回一个空 DataFrame —— 同步报的是
    #: "Tushare trade_cal 连续 6 次返回空表（超出每分钟调用上限，或这天的数据
    #: 还没发布）"，看起来像限流，实际是域名挂了、数据一条都没请求到。
    #:
    #: 上游搬到了 `api.tushare.pro`，但**路径结构没变**：数据接口仍在
    #: `{host}/dataapi/{api_name}` 下面。只换主机名（`http://api.tushare.pro`）
    #: 会 404 —— 库拼 URL 的方式是 `f"{http_url}/{api_name}"`，所以 `/dataapi`
    #: 必须留在 `http_url` 里。实测三种写法的实际响应：
    #:
    #:     http://api.tushare.pro/dataapi/trade_cal   200  有数据
    #:     https://api.tushare.pro/dataapi/trade_cal  200  有数据
    #:     http://api.tushare.pro/trade_cal           404  page not found
    #:
    #: 用 https：同样的数据，但 token 走的是请求体而非 URL，https 额外防的是
    #: 路上被改写。
    #:
    #: 写成常量而不是配置项：换域名是"上游改了"，不该让用户自己试出来。
    ENDPOINT = "https://api.tushare.pro/dataapi"

    def __init__(
        self,
        token: str,
        *,
        retry: int = 3,
        sleep_seconds: float = 0.02,
        timeout_seconds: int = 120,
    ) -> None:
        if not token or not token.strip():
            raise DataError("Tushare token 为空")
        self.token = token.strip()
        self.retry = max(1, int(retry))
        self.sleep_seconds = max(0.0, float(sleep_seconds))
        self.timeout_seconds = int(timeout_seconds)
        self._pro: Any = None

    # ------------------------------------------------------------- 请求底座
    def _api(self) -> Any:
        if self._pro is None:
            import tushare.pro.client as tushare_client

            # 库把地址写成类私有属性 `__http_url`（`_DataApi__http_url`），
            # 构造参数里没有它，`pro_api(token, timeout=)` 也不接受。
            # 子类覆盖是唯一不改 site-packages 的办法 —— 直接改库文件会在下次
            # `pip install` 时被覆盖掉，而且报错现场会指向别人的代码。
            class _Pinned(tushare_client.DataApi):
                _DataApi__http_url = self.ENDPOINT

            self._pro = _Pinned(self.token, timeout=self.timeout_seconds)
        return self._pro

    # 空结果的重试节奏。Tushare 在**超出每分钟调用上限时不报错，而是返回空表** ——
    # 实测 `trade_cal` 大约只允许 2 次/分钟，且同一请求时好时坏（正是限流的特征）。
    # 所以"空"要按限流处理，退避得长到能跨过一个窗口；5 次共约 135 秒。
    _EMPTY_BACKOFF_SECONDS: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 60.0)
    # 元数据端点（trade_cal / stock_basic）是整天的地基，限得也最紧：退避表用尽才算失败。
    # 行情端点一天要连拉三张表，同样会撞上限，给 3 次退避（约 35 秒）再抛。
    _METADATA_EMPTY_RETRIES: int = len(_EMPTY_BACKOFF_SECONDS)
    _MARKET_EMPTY_RETRIES: int = 3

    def _attempt(self, endpoint: str, **kwargs: Any) -> pd.DataFrame:
        """一次调用，只重试**抛异常**的情况（短退避）。空结果由 `_call` 处理。"""
        last: Exception | None = None
        for attempt in range(self.retry):
            try:
                frame = getattr(self._api(), endpoint)(**kwargs)
            except Exception as exc:  # noqa: BLE001 - 上游异常类型不可枚举
                last = exc
                if attempt + 1 < self.retry:
                    time.sleep(min(2.0, 0.3 * (attempt + 1)))
                continue
            time.sleep(self.sleep_seconds)
            return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        raise NetworkError(
            f"Tushare {endpoint} 请求连续失败(重试 {self.retry} 次): "
            f"{type(last).__name__}: {last}"
        )

    def _call(
        self,
        endpoint: str,
        *,
        empty_retries: int = 0,
        allow_empty: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """调端点。空结果按限流处理：退避重试 `empty_retries` 次，用尽即抛。

        退避用尽仍然是空表，说明这不是"这天没有数据"而是**上游没给我们数据**
        （超出每分钟调用上限，或这天的数据还没发布）。这时必须抛 `NetworkError`：
        它是暂时的、重试有意义的失败，与 `ValidationError`（同样的输入重试也
        不会变好）是两件事。调用方据此决定"跳过这一天继续"还是"整趟中止"。
        """
        frame = pd.DataFrame()
        for empty_attempt in range(max(0, empty_retries) + 1):
            frame = self._attempt(endpoint, **kwargs)
            if not frame.empty:
                return frame
            if empty_attempt < empty_retries:
                time.sleep(
                    self._EMPTY_BACKOFF_SECONDS[
                        min(empty_attempt, len(self._EMPTY_BACKOFF_SECONDS) - 1)
                    ]
                )
        if allow_empty:  # 翻页：退避用尽仍是空页，就当作取完了（由调用方另外核对总数）
            return frame
        raise NetworkError(
            f"Tushare {endpoint} 连续 {empty_retries + 1} 次返回空表"
            "（超出每分钟调用上限，或这天的数据还没发布）"
        )

    @staticmethod
    def _stamp(value: date) -> str:
        return value.strftime("%Y%m%d")

    # --------------------------------------------------------------- 端点
    def trade_cal(self, *, start: date, end: date, exchange: str = "SSE") -> pd.DataFrame:
        frame = self._call(
            "trade_cal",
            empty_retries=self._METADATA_EMPTY_RETRIES,
            exchange=exchange,
            start_date=self._stamp(start),
            end_date=self._stamp(end),
            fields=SNAPSHOT_FIELDS["trade_cal"],
        )
        return validated_snapshot(
            frame,
            endpoint="trade_cal",
            required=schema.TABLE_FIELDS["trade_cal"],
            business_keys=schema.TABLE_PRIMARY_KEYS["trade_cal"],
            date_column="cal_date",
            numeric=("is_open",),
            allowed_values={"is_open": (0, 1)},
        )

    def stock_basic(self) -> pd.DataFrame:
        frame = self._call(
            "stock_basic",
            empty_retries=self._METADATA_EMPTY_RETRIES,
            exchange="",
            list_status="L",
            fields=SNAPSHOT_FIELDS["stock_basic"],
        )
        return validated_snapshot(
            frame,
            endpoint="stock_basic",
            required=schema.TABLE_FIELDS["stock_basic"],
            business_keys=schema.TABLE_PRIMARY_KEYS["stock_basic"],
            # `list_date` 允许为空：不从上游造日期。
            date_column="list_date",
        )

    def daily(self, *, trade_date: date) -> pd.DataFrame:
        frame = self._call(
            "daily",
            empty_retries=self._MARKET_EMPTY_RETRIES,
            trade_date=self._stamp(trade_date),
            fields=SNAPSHOT_FIELDS["daily"],
        )
        return validated_snapshot(
            frame,
            endpoint="daily",
            required=schema.TABLE_FIELDS["daily"],
            business_keys=schema.TABLE_PRIMARY_KEYS["daily"],
            date_column="trade_date",
            expected_date=trade_date,
            numeric=("open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"),
            nonnegative_numeric=("vol", "amount"),
        )

    def daily_limit(self, *, trade_date: date) -> pd.DataFrame:
        frame = self._call(
            "stk_limit",
            empty_retries=self._MARKET_EMPTY_RETRIES,
            trade_date=self._stamp(trade_date),
            fields=SNAPSHOT_FIELDS["daily_limit"],
        )
        return validated_snapshot(
            frame,
            endpoint="daily_limit",
            required=schema.TABLE_FIELDS["daily_limit"],
            business_keys=schema.TABLE_PRIMARY_KEYS["daily_limit"],
            date_column="trade_date",
            expected_date=trade_date,
            numeric=("up_limit", "down_limit"),
            nullable_numeric=("up_limit", "down_limit"),
            nonnegative_numeric=("up_limit", "down_limit"),
        )

    def adj_factor(self, *, trade_date: date) -> pd.DataFrame:
        frame = self._call(
            "adj_factor",
            empty_retries=self._MARKET_EMPTY_RETRIES,
            trade_date=self._stamp(trade_date),
            fields=SNAPSHOT_FIELDS["adj_factor"],
        )
        frame = validated_snapshot(
            frame,
            endpoint="adj_factor",
            required=schema.TABLE_FIELDS["adj_factor"],
            business_keys=schema.TABLE_PRIMARY_KEYS["adj_factor"],
            date_column="trade_date",
            expected_date=trade_date,
            # 不许缺：反弹模型的复权价 = 价 × 因子，缺一天就算不出那段收益
            numeric=("adj_factor",),
            nonnegative_numeric=("adj_factor",),
        )
        zero = int((frame["adj_factor"] == 0).sum())
        if zero:
            raise ValidationError(f"adj_factor {trade_date} 有 {zero} 行因子为 0：复权价会变成 0")
        return frame

    #: `namechange` 每页行数（上游单次上限 5000）。
    _NAME_PAGE = 5000
    #: 最多翻几页（2026-10 全部历史名称 13,925 行，按 5000 行一页是 3 页）。
    _NAME_MAX_PAGES = 100

    def namechange(self) -> pd.DataFrame:
        """全部历史名称（按 offset 翻页，和十年实验 `fetch_history.fetch_table` 同法）。

        第一页是空的就抛（上游没给数据，不是"没有名称"）。和十年实验一样**取到空页才算完**：
        上游每页给的可能比要的少，不能拿"不满一页"当取完。空页紧跟在满页（到目前最大的一页）后面，
        可能是撞了限流，按元数据的退避等足再认；紧跟在不满的一页后面，多半真取完了，只确认一次不等。
        少取的一截再由 `sync_names` 兜住：库里的键这次没取到太多、或者最新交易日的票大多没名称，就拒绝。
        同一个生效日有几行时只留公告日最晚的（再并列取名称最小的，和十年实验的 SQL 相同）。
        """
        fields = SNAPSHOT_FIELDS["namechange"]
        parts: list[pd.DataFrame] = []
        while True:
            if len(parts) >= self._NAME_MAX_PAGES:  # 上游一直不给空页（坏了或被改了）：不能无限翻下去
                raise ValidationError(f"namechange 翻了 {len(parts)} 页还没到头（已取 {sum(len(p) for p in parts)} 行，"
                                      f"正常 3 页左右）：上游不对，这次不写")
            offset = sum(len(p) for p in parts)
            full = not parts or len(parts[-1]) >= max(len(p) for p in parts)
            part = self._call("namechange", empty_retries=self._METADATA_EMPTY_RETRIES if full else 0,
                              allow_empty=bool(parts), fields=fields, limit=self._NAME_PAGE, offset=offset)
            if part.empty:
                break
            if parts and part.iloc[0].equals(parts[-1].iloc[0]):
                raise ValidationError(f"namechange 翻页没生效（offset={offset} 返回了上一页）")
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True).drop_duplicates()
        # 名称会原样打到终端上：去掉控制字符（真实名称里没有），免得上游给的转义序列在命令行里生效
        frame["name"] = frame["name"].where(
            frame["name"].isna(), frame["name"].astype(str).str.replace(r"[\x00-\x1f\x7f]", "", regex=True))
        for column in ("end_date", "ann_date"):
            frame[column] = pd.to_datetime(frame[column], format="%Y%m%d", errors="coerce").dt.date
            frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
        frame = (frame.sort_values(["ts_code", "start_date", "ann_date", "name"], ascending=[True, True, True, False],
                                   na_position="first", kind="mergesort")
                 .drop_duplicates(["ts_code", "start_date"], keep="last"))
        return validated_snapshot(
            frame,
            endpoint="namechange",
            required=schema.TABLE_FIELDS["namechange"],
            business_keys=schema.TABLE_PRIMARY_KEYS["namechange"],
            date_column="start_date",
        )


def resolve_token(settings: Settings) -> str:
    """从环境变量取 token；`.env` 是唯一被读取的凭据文件（不入库）。"""
    import dotenv

    section = settings.section("tushare")
    name = str(section.get("token_env", "TUSHARE_TOKEN"))
    dotenv.load_dotenv(settings.root / ".env", override=False)
    token = os.environ.get(name, "").strip()
    if not token:
        raise DataError(
            f"环境变量 {name} 未设置：把 token 写进 {settings.root / '.env'}（该文件已在 .gitignore 里）"
        )
    return token


def tushare_source(settings: Settings) -> TushareClient:
    section = settings.section("tushare")
    return TushareClient(
        resolve_token(settings),
        retry=int(section.get("retry", 3)),
        sleep_seconds=float(section.get("sleep_seconds", 0.02)),
        timeout_seconds=int(section.get("request_timeout_seconds", 120)),
    )
