"""YAML config loader with automatic path resolution relative to project root."""

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(domain: str) -> dict[str, Any]:
    """
    Load the YAML config for a given domain.

    Args:
        domain: "credit_risk" or "network_intrusion".

    Returns:
        Config dictionary with all path values resolved to absolute paths.
    """
    config_path = PROJECT_ROOT / "configs" / f"{domain}_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Expected one of: credit_risk_config.yaml, network_intrusion_config.yaml"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = _resolve_paths(config)
    return config


def _resolve_paths(obj: Any) -> Any:
    """Recursively resolve relative path strings to absolute paths from PROJECT_ROOT."""
    path_extensions = {".csv", ".parquet", ".pkl", ".joblib", ".txt", ".json"}

    if isinstance(obj, dict):
        return {k: _resolve_paths(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_resolve_paths(item) for item in obj]

    if isinstance(obj, str):
        p = Path(obj)
        if p.suffix in path_extensions or ("/" in obj and not obj.startswith("http")):
            return str(PROJECT_ROOT / obj)

    return obj


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT
