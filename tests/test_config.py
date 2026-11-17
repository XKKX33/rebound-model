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
