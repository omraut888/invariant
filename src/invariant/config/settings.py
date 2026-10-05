"""Environment settings (.env) and layered YAML pipeline config.

Paths are never resolved against the current working directory: relative paths (defaults,
env vars, ``.env`` values, and config file arguments) are resolved against the project root,
the nearest ancestor of this module that contains ``pyproject.toml``. This makes behaviour
identical whether code runs from the repo root, a notebook in notebooks/, or a test tmpdir.
It requires an editable install (``uv sync`` installs the project editable by default).

YAML layering: ``configs/base.yaml`` plus an optional override file merged on top with
``deep_merge`` (nested mappings merge recursively; lists and scalars replace). The merged
mapping is validated into a frozen ``PipelineConfig`` with ``extra="forbid"``, so a mistyped key
fails loudly instead of being silently ignored — the config feeds the run hash.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictInt,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from invariant.core.types import Frequency

__all__ = [
    "ConfigError",
    "DateRange",
    "PipelineConfig",
    "Settings",
    "UniverseConfig",
    "deep_merge",
    "find_project_root",
    "load_config",
    "load_settings",
]

DEFAULT_BASE_CONFIG = Path("configs/base.yaml")


class ConfigError(Exception):
    """Configuration could not be located or parsed."""


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest ancestor of ``start`` (default: this module) containing pyproject.toml."""
    origin = (start if start is not None else Path(__file__)).resolve()
    candidates = [origin, *origin.parents] if origin.is_dir() else list(origin.parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigError(
        f"no pyproject.toml found above {origin}; INVARIANT must be installed in editable "
        "mode (`uv sync`) so the project root can be located",
    )


def _resolve_against_root(path: Path) -> Path:
    return path if path.is_absolute() else find_project_root() / path


class Settings(BaseSettings):
    """Secrets and filesystem locations, from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        frozen=True,
        extra="ignore",
        env_ignore_empty=True,
        populate_by_name=True,
    )

    fred_api_key: SecretStr | None = Field(default=None, validation_alias="FRED_API_KEY")
    data_dir: Path = Field(
        default=Path("data"),
        validation_alias="INVARIANT_DATA_DIR",
        validate_default=True,
    )
    runs_dir: Path = Field(
        default=Path("runs"),
        validation_alias="INVARIANT_RUNS_DIR",
        validate_default=True,
    )

    @field_validator("data_dir", "runs_dir")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        return _resolve_against_root(value)


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings. ``env_file`` defaults to ``<project root>/.env``; a missing file is fine."""
    path = _resolve_against_root(env_file) if env_file is not None else find_project_root() / ".env"
    return Settings(_env_file=path)


_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class UniverseConfig(_Frozen):
    tickers: tuple[str, ...] = Field(min_length=1)

    @field_validator("tickers")
    @classmethod
    def _check_tickers(cls, tickers: tuple[str, ...]) -> tuple[str, ...]:
        bad = [t for t in tickers if not _TICKER_RE.fullmatch(t)]
        if bad:
            raise ValueError(f"tickers must be uppercase symbols, got {bad}")
        dupes = sorted({t for t in tickers if tickers.count(t) > 1})
        if dupes:
            raise ValueError(f"duplicate tickers: {dupes}")
        return tickers


class DateRange(_Frozen):
    """Closed sample window. Fixed dates only: a moving end date would break reproducibility."""

    start: date
    end: date

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.start >= self.end:
            raise ValueError(f"start {self.start} must be before end {self.end}")
        return self


class PipelineConfig(_Frozen):
    universe: UniverseConfig
    dates: DateRange
    frequency: Frequency
    seed: StrictInt = Field(ge=0, lt=2**32)


def deep_merge(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    """Merge ``override`` onto ``base`` without mutating either; the result shares no objects.

    Mappings present on both sides merge recursively; any other override value (scalar, list,
    None) replaces the base value.
    """
    merged: dict[str, object] = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError(
            f"config file {path} must contain a YAML mapping, got {type(data).__name__}",
        )
    return data


def load_config(
    base_path: Path | None = None,
    override_path: Path | None = None,
) -> PipelineConfig:
    """Load ``base_path`` (default ``configs/base.yaml``), merge ``override_path`` on top, validate.

    Relative paths resolve against the project root.
    """
    merged = _read_yaml_mapping(_resolve_against_root(base_path or DEFAULT_BASE_CONFIG))
    if override_path is not None:
        merged = deep_merge(merged, _read_yaml_mapping(_resolve_against_root(override_path)))
    return PipelineConfig.model_validate(merged)
