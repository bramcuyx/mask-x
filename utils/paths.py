from __future__ import annotations

import os
import pathlib
from typing import Any

import yaml


# Project root:
# TianyuXie/utils/paths.py -> parents[1] = TianyuXie/
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "mask_config.yaml"
EXAMPLE_CONFIG_PATH = CONFIG_DIR / "mask_config.example.yaml"


def load_config(config_path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """
    Load project config.

    By default, this reads:
        config/mask_config.yaml

    This file should be local only and should not be committed.
    """
    path = pathlib.Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Copy {EXAMPLE_CONFIG_PATH} to {DEFAULT_CONFIG_PATH} "
            f"and edit the local paths."
        )

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    return config


def resolve_path(path_value: str | pathlib.Path | None) -> pathlib.Path | None:
    """
    Resolve a path from config.

    Rules:
    - None or empty string -> None
    - absolute path -> return as-is
    - relative path -> resolve relative to PROJECT_ROOT
    """
    if path_value is None:
        return None

    path_str = str(path_value).strip()
    if path_str == "":
        return None

    path = pathlib.Path(path_str).expanduser()

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_config_path(
    config: dict[str, Any],
    key: str,
    required: bool = True,
) -> pathlib.Path | None:
    """
    Read paths.<key> from config and resolve it.

    Example:
        models_dir = get_config_path(config, "models_folder")
    """
    paths = config.get("paths", {})
    value = paths.get(key)

    resolved = resolve_path(value)

    if required and resolved is None:
        raise KeyError(f"Missing required config value: paths.{key}")

    return resolved


def get_fuss_root(config: dict[str, Any] | None = None) -> pathlib.Path:
    """
    Get FUSS root.

    Priority:
    1. Environment variable FUSS_ROOT
    2. config paths.fuss_root
    """
    env_value = os.environ.get("FUSS_ROOT")

    if env_value:
        root = pathlib.Path(env_value).expanduser()
    else:
        if config is None:
            config = load_config()
        root = get_config_path(config, "fuss_root", required=True)

    if root is None:
        raise FileNotFoundError(
            "FUSS root is not set. Set FUSS_ROOT or paths.fuss_root in config/mask_config.yaml."
        )

    if not root.exists():
        raise FileNotFoundError(f"FUSS root does not exist: {root}")

    return root


def ensure_dir(path: pathlib.Path | None) -> pathlib.Path:
    """
    Create a directory if needed and return it.
    """
    if path is None:
        raise ValueError("Cannot create directory from None.")

    path.mkdir(parents=True, exist_ok=True)
    return path