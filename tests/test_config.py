"""配置加载（`rebound/config.py`）：正式配置能读，拼错的键、不认识的取值在加载时就报错。"""

from __future__ import annotations

from pathlib import Path

import pytest

from rebound import config

ROOT = Path(__file__).resolve().parents[1]
BASE = (ROOT / "config" / "settings.yaml").read_text(encoding="utf-8")


def _load(tmp_path: Path, text: str) -> config.Settings:
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    return config.load_settings(path, root=tmp_path)


def test_the_shipped_settings_load() -> None:
    settings = config.load_settings()
    assert settings.data.db_path == ROOT / "data" / "market.duckdb"
    assert (settings.model.variant, settings.model.cadence) == ("nocap", "M")
    assert (settings.pick.top_n, settings.pick.max_per_industry) == (6, 2)
    assert settings.section("tushare")["token_env"] == "TUSHARE_TOKEN"
    assert settings.data.close_time.hour == 15


@pytest.mark.parametrize(("old", "new", "message"), [
    ("  variant: nocap", "  variant: all", "variant 只能是"),
    ("  cadence: M", "  cadence: W", "cadence 只能是"),
    ("  top_n: 6", "  top_n: 0", "top_n 必须是正整数"),
    ("  top_n: 6", "  top_n: 6\n  top: 3", "不认识的键 \\['top'\\]"),
    ("  with_limits: true", "  with_limits: yes please", "with_limits 必须是布尔值"),
    ('  session_close_time: "15:30"', '  session_close_time: "1530"', "HH:MM"),
    ("  candidate_limit: 260", "  candidate_limit: 3", "不能小于 top_n"),
])
def test_bad_values_are_refused_when_loading(tmp_path: Path, old: str, new: str, message: str) -> None:
    assert old in BASE
    with pytest.raises(config.ConfigError, match=message):
        _load(tmp_path, BASE.replace(old, new, 1))


def test_an_unknown_section_and_a_missing_one_are_refused(tmp_path: Path) -> None:
    with pytest.raises(config.ConfigError, match="不认识的键 \\['llm'\\]"):
        _load(tmp_path, BASE + "\nllm:\n  model: x\n")
    with pytest.raises(config.ConfigError, match="缺少段落 'tushare'"):
        _load(tmp_path, BASE.replace("tushare:", "tushare_off:", 1).replace("tushare_off:", "# off:", 1)
              .replace("  token_env: TUSHARE_TOKEN\n  request_timeout_seconds: 120\n  retry: 5\n  sleep_seconds: 0.02\n",
                       "", 1))


def test_model_and_pick_sections_are_optional(tmp_path: Path) -> None:
    start = BASE.index("model:")
    settings = _load(tmp_path, BASE[:start])
    assert (settings.model.variant, settings.model.cadence, settings.pick.top_n) == ("nocap", "M", 6)
    assert settings.data.db_path == tmp_path / "data" / "market.duckdb"


def test_an_explicit_config_resolves_data_from_its_project(tmp_path: Path, monkeypatch) -> None:
    """安装后在别的目录运行，配置仍指向它所属项目的数据。"""
    project = tmp_path / "项目"
    path = project / "config" / "settings.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(BASE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = config.load_settings(path.relative_to(tmp_path))

    assert settings.root == project
    assert settings.data.db_path == project / "data" / "market.duckdb"


def test_cli_accepts_an_explicit_project_config(tmp_path: Path, capsys) -> None:
    """列出空项目的纸面跟踪不需要真实数据库。"""
    from rebound.cli import main

    path = tmp_path / "config" / "settings.yaml"
    path.parent.mkdir()
    path.write_text(BASE, encoding="utf-8")

    assert main(["--config", str(path), "track", "list"]) == 0
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "data").exists()


def test_token_comes_from_the_selected_project(tmp_path: Path, monkeypatch) -> None:
    """凭据和数据必须属于同一个项目，不能读取安装位置的凭据。"""
    from rebound.data.tushare_client import resolve_token

    token_name = "REBOUND_TEST_TUSHARE_TOKEN"
    monkeypatch.setenv(token_name, "")
    monkeypatch.delenv(token_name)
    settings = _load(tmp_path, BASE.replace("TUSHARE_TOKEN", token_name))
    (tmp_path / ".env").write_text(f"{token_name}=local-test-token\n", encoding="utf-8")

    assert resolve_token(settings) == "local-test-token"


def test_debate_defaults_when_omitted(tmp_path: Path) -> None:
    stripped = BASE.split("\ndebate:")[0]
    settings = _load(tmp_path, stripped)
    assert settings.debate.candidates == 20
    assert settings.debate.min_debated == 12
    assert settings.debate.bars == 20
    assert settings.debate.concurrency == 5
    assert settings.debate.call_timeout_seconds == 300
    assert settings.debate.llm.model == "step-5-preview"
    assert settings.debate.llm.api_key_env == "WORKBENCH_AI_API_KEY"
    assert settings.debate.llm.reasoning_effort == "high"
    assert settings.debate.llm.max_tokens == 16384


@pytest.mark.parametrize(("extra", "message"), [
    ("debate:\n  unknown_key: 1\n", "settings.debate 有不认识的键"),
    ("debate:\n  candidates: 51\n", "candidates 必须是 1~50"),
    ("debate:\n  min_debated: 4\n", "不能小于 pick.top_n"),
    ("debate:\n  candidates: 10\n  min_debated: 12\n", "不能大于 candidates"),
    ("debate:\n  bars: 4\n", "bars 必须是 5~60"),
    ("debate:\n  concurrency: 17\n", "concurrency 必须是 1~16"),
    ("debate:\n  call_timeout_seconds: 0\n", "call_timeout_seconds 必须是 1~10800"),
    ("debate:\n  llm:\n    bad: 1\n", "settings.debate.llm 有不认识的键"),
    ("debate:\n  llm:\n    reasoning_effort: extreme\n", "reasoning_effort 只能是"),
    ("debate:\n  llm:\n    temperature: 2.5\n", "temperature 必须在 0~2"),
    ("debate:\n  llm:\n    max_tokens: 40000\n", "max_tokens 必须是 1~32768"),
])
def test_debate_bad_values_are_refused(tmp_path: Path, extra: str, message: str) -> None:
    stripped = BASE.split("\ndebate:")[0]
    with pytest.raises(config.ConfigError, match=message):
        _load(tmp_path, stripped + "\n" + extra)
