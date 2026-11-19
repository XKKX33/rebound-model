"""辩论命令行（`rebound debate` 与 `rebound daily --debate`）测试，不碰网络与真实大模型。"""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path

import pytest
from conftest import MODEL_SESSIONS, TEST_FIRST_FIT
from test_debate_service import FakeSidecar

from rebound import cli
from rebound import config as app_config
from rebound.debate import runs, sidecar as sidecar_mod
from rebound.model.artifacts import models_root
from rebound.model.score import fill_scores

DAY = date(2023, 3, 15)


def _copy_world(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    shutil.copytree(models_root(src), models_root(dst))
    return dst


@pytest.fixture(scope="module")
def scored_pristine(trained_world_pristine: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = _copy_world(trained_world_pristine, tmp_path_factory.mktemp("debate-cli") / "market.duckdb")
    fill_scores(path, first_fit=TEST_FIRST_FIT)
    return path


@pytest.fixture
def world(scored_pristine: Path, tmp_path: Path) -> Path:
    return _copy_world(scored_pristine, tmp_path / "market.duckdb")


def test_debate_dry_run_and_json_contain_no_secret(
    world: Path, tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.setenv("WORKBENCH_AI_API_KEY", "super-secret-value-never-leak")
    monkeypatch.setattr("rebound.debate.service.Sidecar", lambda _settings: FakeSidecar())

    assert cli.main(["debate", "run", "--db", str(world), "--as-of", DAY.isoformat(), "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "信号日 2023-03-15" in text
    assert "预计调用次数：121 次" in text
    assert "请求字符数：" in text
    assert "super-secret-value-never-leak" not in text

    assert cli.main(["debate", "run", "--db", str(world), "--as-of", DAY.isoformat(), "--dry-run", "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["protocol_version"] == 4
    assert len(body["candidates"]) == 20
    assert "super-secret-value-never-leak" not in json.dumps(body, ensure_ascii=False)
    assert not runs.debate_root(world).exists()


def test_debate_run_and_show_commands(world: Path, monkeypatch, capsys) -> None:
    # 没有任何记录时 show 返回退出码 5
    assert cli.main(["debate", "show", "--db", str(world), "--as-of", DAY.isoformat()]) == 5
    assert "没有" in capsys.readouterr().err

    monkeypatch.setattr("rebound.debate.service.Sidecar", lambda _settings: FakeSidecar())
    assert cli.main(["debate", "run", "--db", str(world), "--as-of", DAY.isoformat()]) == 0
    out = capsys.readouterr().out
    assert "AI 辩论名单" in out and "与模型名单对照" in out

    assert cli.main(["debate", "show", "--db", str(world), "--as-of", DAY.isoformat()]) == 0
    shown = capsys.readouterr().out
    assert "[当日正式结果]" in shown and "风控详情" in shown

    assert cli.main(["debate", "show", "--db", str(world), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["as_of"] == DAY.isoformat() and payload["runs"][0]["official"] is True


def test_debate_run_without_as_of_checks_freshness_and_returns_exit_3(world: Path, capsys) -> None:
    assert cli.main(["debate", "run", "--db", str(world)]) == 3
    err = capsys.readouterr().err
    assert err.strip() != ""


def test_debate_run_returns_exit_5_when_min_debated_not_met(world: Path, monkeypatch, capsys) -> None:
    fake = FakeSidecar(fail_codes=frozenset({f"60{k:04d}.SH" for k in range(33)}))
    monkeypatch.setattr("rebound.debate.service.Sidecar", lambda _settings: fake)
    assert cli.main(["debate", "run", "--db", str(world), "--as-of", DAY.isoformat()]) == 5
    assert "少于 min_debated" in capsys.readouterr().err


def test_debate_check_checks_key_without_leaking_value(tmp_path: Path, monkeypatch, capsys) -> None:
    cfg = tmp_path / "config" / "settings.yaml"
    cfg.parent.mkdir()
    cfg.write_text((Path(app_config.ROOT) / "config" / "settings.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    # 缺密钥时返回退出码 5
    monkeypatch.delenv("WORKBENCH_AI_API_KEY", raising=False)
    monkeypatch.setattr(
        "rebound.debate.cli.Sidecar.health",
        lambda _self: {"ok": True, "node": "v24.14.0", "protocol_version": 4},
    )
    assert cli.main(["--config", str(cfg), "debate", "check"]) == 5
    out_missing = capsys.readouterr().out
    assert "[失败] 密钥环境变量" in out_missing

    # 配置密钥后通过且不打印密钥值
    monkeypatch.setenv("WORKBENCH_AI_API_KEY", "secret-token-xyz-123")
    assert cli.main(["--config", str(cfg), "debate", "check"]) == 0
    out_ok = capsys.readouterr().out
    assert "[通过] 密钥环境变量" in out_ok
    assert "secret-token-xyz-123" not in out_ok


def test_daily_with_debate_and_failure_exit_5(world: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    cfg = tmp_path / "config" / "settings.yaml"
    cfg.parent.mkdir()
    base = (Path(app_config.ROOT) / "config" / "settings.yaml").read_text(encoding="utf-8")
    cfg.write_text(base.replace("min_daily_rows: 1000", "min_daily_rows: 10"), encoding="utf-8")

    latest = MODEL_SESSIONS[-1]
    monkeypatch.setattr("rebound.data.freshness.beijing_now", lambda: datetime(2023, 5, 31, 16, 0))
    monkeypatch.setattr("rebound.model.refresh.refresh", lambda *_a, **_kw: None)

    # 先测成功路径
    monkeypatch.setattr("rebound.debate.service.Sidecar", lambda _settings: FakeSidecar())
    code = cli.main(["--config", str(cfg), "daily", "--no-sync", "--debate", "--db", str(world)])
    assert code == 0
    out = capsys.readouterr().out
    assert f"{latest} 收盘后的名单" in out
    assert "纸面跟踪：更新了 0 个" in out
    assert "AI 辩论名单" in out

    # 再测辩论失败时：前面的选股与跟踪结果照常打印，整体返回退出码 5
    monkeypatch.setattr(
        "rebound.debate.service.Sidecar",
        lambda _settings: FakeSidecar(fail_codes=frozenset({f"60{k:04d}.SH" for k in range(33)})),
    )
    shutil.rmtree(runs.debate_root(world))
    code_fail = cli.main(["--config", str(cfg), "daily", "--no-sync", "--debate", "--db", str(world)])
    assert code_fail == 5
    captured = capsys.readouterr()
    assert f"{latest} 收盘后的名单" in captured.out
    assert "纸面跟踪：更新了 0 个" in captured.out
    assert "辩论未出名单" in captured.err
