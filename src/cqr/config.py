from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Mapping


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def merge_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(common: Path, dataset: Path, overrides: tuple[Path, ...] = ()) -> dict[str, Any]:
    config = merge_config(load_toml(common), load_toml(dataset))
    for path in overrides:
        config = merge_config(config, load_toml(path))
    return config


def require(config: Mapping[str, Any], *path: str) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(f"Missing configuration value: {'.'.join(path)}")
        value = value[key]
    return value
