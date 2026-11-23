"""Resolve project inputs without copying data or changing member repositories."""
from __future__ import annotations

import os
from pathlib import Path


def _configured(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


PROJECT = _configured("DASHBOARD_PROJECT_ROOT", Path(__file__).resolve().parents[2])
DASHBOARD = Path(__file__).resolve().parent
_member_default = PROJECT if (PROJECT / "ml/api.py").is_file() else PROJECT / "github_clone/rebound-model"
MODEL_REPO = _configured("REBOUND_REPO", _member_default)
AGORA_REPO = _configured("AGORA_REPO", PROJECT / "github_clone/Agroe")
DATABASE = _configured("DASHBOARD_DATABASE", PROJECT / "data/research-2013-2026.duckdb")
_artifacts_default = PROJECT / "data/ml" if _member_default == PROJECT else PROJECT / "特征与模型"
if not (_artifacts_default / "api_export/manifest.json").is_file() and (
    PROJECT / "data/ml/api_export/manifest.json"
).is_file():
    _artifacts_default = PROJECT / "data/ml"
MODEL_ARTIFACTS = _configured("DASHBOARD_MODEL_ARTIFACTS", _artifacts_default)
PROFILE = _configured("DASHBOARD_DATA_PROFILE", PROJECT / "work/ui_data_profile.json")
FEATURE_CACHE = _configured("DASHBOARD_FEATURE_CACHE", PROJECT / "work/dashboard/feature_snapshot.parquet")
FEATURE_STAMP = _configured("DASHBOARD_FEATURE_STAMP", PROJECT / "work/dashboard/feature_snapshot.json")
INCREMENTAL_DIRECTORY = _configured("DASHBOARD_INCREMENTAL_DIR", PROJECT / "data/incremental")
