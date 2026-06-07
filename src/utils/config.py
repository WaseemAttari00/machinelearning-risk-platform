"""
Config loader utility.

Why this exists:
  Every script in the project needs to read the YAML config for its domain.
  Rather than duplicating yaml.safe_load(...) everywhere, we centralize it here.
  This also resolves paths relative to the project root automatically, so the
  project works regardless of which directory you launch it from.
"""

from pathlib import Path
from typing import Any

import yaml


# The project root is two levels up from this file:
#   src/utils/config.py → src/ → project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(domain: str) -> dict[str, Any]:
    """
    Load a YAML config file for the given domain.

    Args:
        domain: One of "credit_risk" or "network_intrusion".
                Maps to configs/<domain>_config.yaml.

    Returns:
        A plain Python dictionary with the full config.

    Example:
        cfg = load_config("credit_risk")
        target_col = cfg["data"]["target_column"]   # "SeriousDlqin2yrs"
    """
    config_path = PROJECT_ROOT / "configs" / f"{domain}_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Expected one of: credit_risk_config.yaml, network_intrusion_config.yaml"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve all path values relative to project root.
    # This means configs can store paths like "data/raw/credit_risk/cs-training.csv"
    # and they will resolve correctly no matter where Python is launched from.
    config = _resolve_paths(config)
    return config


def _resolve_paths(obj: Any) -> Any:
    """
    Recursively walk the config dictionary and convert any string value that
    looks like a file path (contains "/" or ends with a known extension) into
    an absolute Path object.

    We convert to string at the end so callers always get str, not Path — this
    avoids surprises when passing paths to libraries that don't accept Path objects.
    """
    path_extensions = {".csv", ".parquet", ".pkl", ".joblib", ".txt", ".json"}

    if isinstance(obj, dict):
        return {k: _resolve_paths(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_resolve_paths(item) for item in obj]

    if isinstance(obj, str):
        # Heuristic: if the string contains a "/" and ends with a known extension,
        # treat it as a relative path and resolve it from PROJECT_ROOT.
        p = Path(obj)
        if p.suffix in path_extensions or (
            "/" in obj and not obj.startswith("http")
        ):
            return str(PROJECT_ROOT / obj)

    return obj


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT
