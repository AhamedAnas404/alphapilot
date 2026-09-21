"""Central config loader — reads config.yaml once and exposes a typed dict."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]  # repo root


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the YAML config as a plain dict."""
    cfg_path = Path(path) if path else _ROOT / "config.yaml"
    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh)
    logger.debug("Config loaded from %s", cfg_path)
    return cfg
