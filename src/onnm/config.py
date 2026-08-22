"""Configuration loading: YAML -> attribute-accessible nested config.

Deliberately a thin wrapper rather than a deep dataclass tree. The config is
nested five levels in places, and mirroring that in dataclasses buys type
checking at the cost of a lot of boilerplate that has to be edited every time a
knob is added. What actually matters here is that overrides *merge* rather than
replace, so a profile can change ``loader.num_workers`` without silently
dropping ``loader.batch_size``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    """Raised when a config is missing a key or is internally inconsistent."""


class Config:
    """Attribute- and item-accessible view over a nested dict."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    # -- access ------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(
                f"config has no key {name!r}; available: {sorted(self._data)}"
            ) from exc
        return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._data:
            return default
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def resolve_path(self, dotted: str) -> Path:
        """Resolve a ``paths.*`` value against the repo root when relative.

        Absolute paths (Kaggle's ``/kaggle/input/...``) are passed through, so
        the same config works locally and in a notebook.
        """
        raw = self.lookup(dotted)
        path = Path(raw).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()

    def lookup(self, dotted: str, default: Any = ...) -> Any:
        """Fetch a value by dotted path, e.g. ``lookup("loader.batch_size")``."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is ...:
                    raise ConfigError(f"missing config key: {dotted!r}")
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Lists are replaced wholesale, not concatenated -- half-overriding a list of
    augmentation ranges is never what anyone means.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(
    path: str | Path = "configs/base.yaml",
    overrides: list[str | Path] | None = None,
    profile: str | None = None,
) -> Config:
    """Load a YAML config, apply override files, then apply a named profile.

    Args:
        path: Base YAML file, relative to the repo root or absolute.
        overrides: Additional YAML files merged in order.
        profile: Key from the config's ``profiles:`` block (e.g. ``kaggle``).
    """
    base_path = Path(path)
    if not base_path.is_absolute():
        base_path = REPO_ROOT / base_path
    if not base_path.is_file():
        raise ConfigError(f"config file not found: {base_path}")

    with base_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    for override in overrides or []:
        ov_path = Path(override)
        if not ov_path.is_absolute():
            ov_path = REPO_ROOT / ov_path
        if not ov_path.is_file():
            raise ConfigError(f"override config not found: {ov_path}")
        with ov_path.open("r", encoding="utf-8") as fh:
            data = _deep_merge(data, yaml.safe_load(fh) or {})

    profiles = data.pop("profiles", {}) or {}
    if profile:
        if profile not in profiles:
            raise ConfigError(
                f"unknown profile {profile!r}; available: {sorted(profiles)}"
            )
        data = _deep_merge(data, profiles[profile])

    return Config(data)
