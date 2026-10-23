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
from typing import Literal, Self

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
    "AlfredConfig",
    "CausalBootstrapConfig",
    "CausalConfig",
    "ConfigError",
    "DataConfig",
    "DateRange",
    "DecayConfig",
    "FrenchConfig",
    "HttpConfig",
    "PipelineConfig",
    "PitConfig",
    "PublicationLags",
    "Settings",
    "TargetConfig",
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


_FRED_SERIES_RE = re.compile(r"^[A-Z0-9_]{1,30}$")
_CALENDAR_RE = re.compile(r"^[A-Z0-9]{2,8}$")


class HttpConfig(_Frozen):
    """Timeouts and retry policy shared by every HTTP source."""

    timeout_s: float = Field(gt=0)
    max_retries: StrictInt = Field(ge=0, le=10)
    backoff_s: float = Field(ge=0)


class FrenchConfig(_Frozen):
    """Ken French Data Library zip URLs. Tunable because the library has moved before."""

    five_factor_url: str = Field(pattern=r"^https://")
    momentum_url: str = Field(pattern=r"^https://")


class AlfredConfig(_Frozen):
    """FRED/ALFRED series fetched with all vintages."""

    series: tuple[str, ...] = Field(min_length=1)
    page_size: StrictInt = Field(ge=1, le=100_000)

    @field_validator("series")
    @classmethod
    def _check_series(cls, series: tuple[str, ...]) -> tuple[str, ...]:
        bad = [s for s in series if not _FRED_SERIES_RE.fullmatch(s)]
        if bad:
            raise ValueError(f"FRED series ids must be uppercase alphanumeric, got {bad}")
        dupes = sorted({s for s in series if series.count(s) > 1})
        if dupes:
            raise ValueError(f"duplicate series: {dupes}")
        return series


class DataConfig(_Frozen):
    """Data-source settings. ``calendar`` is an exchange-calendars code (e.g. XNYS)."""

    calendar: str
    http: HttpConfig
    french: FrenchConfig
    alfred: AlfredConfig

    @field_validator("calendar")
    @classmethod
    def _check_calendar(cls, value: str) -> str:
        if not _CALENDAR_RE.fullmatch(value):
            raise ValueError(f"calendar must be an exchange code like 'XNYS', got {value!r}")
        return value


class TargetConfig(_Frozen):
    """Forward-return target timing: ``y_t = r_{t+1+e} + ... + r_{t+e+h}``, ``e = execution_lag``.

    ``execution_lag`` is the number of sessions between the close at which features are known
    and the close at which the position is entered. It has no default on purpose: real data
    uses 1 (a close-t signal cannot trade at the close-t auction), the planted-signal benchmark
    uses 0 because its expected values assume ``S_t -> R_{t+1}``. See docs/adr/0006.
    """

    execution_lag: StrictInt = Field(ge=0, le=21)
    horizon: StrictInt = Field(ge=1, le=252)


class PublicationLags(_Frozen):
    """Sessions between a row's date and its first use, for sources without release dates."""

    french: StrictInt = Field(ge=0, le=252)


class PitConfig(_Frozen):
    """Point-in-time panel assembly (``invariant.data.pit``)."""

    targets: TargetConfig
    publication_lag_sessions: PublicationLags


class CausalBootstrapConfig(_Frozen):
    """Circular block bootstrap of the partial correlation at a signal's best (asset, lag)."""

    n_boot: StrictInt = Field(ge=99, le=100_000)
    block_length: StrictInt | None = Field(ge=1)
    """``None``: ``ceil(n ** (1/3))``, the MSE-optimal rate for a block-bootstrap variance."""
    confidence: float = Field(gt=0.5, lt=1.0)


class CausalConfig(_Frozen):
    """Stage 1 (``invariant.stages.causal.stage``; design and rationale in docs/adr/0010).

    ``ci_test`` admits only the test selected in docs/adr/0009. Another test needs a size study
    first: tigramite's ``ParCorrWLS`` and the classes derived from it are not even
    location-invariant. ``decision``: ``hybrid`` decides; ``mci`` is the admissible but
    low-power alternative. ``parents_r`` is not offered: it falsely passes a confounded signal.
    ``signals``: the candidate signal ids (a ``syn_`` column's id is its name without the
    prefix). ``None`` means every non-return feature column. Features that are not candidates
    stay in the graph as conditioning variables.
    """

    ci_test: Literal["parcorr_hc3"]
    decision: Literal["hybrid", "mci"]
    tau_max: StrictInt = Field(ge=1, le=20)
    pc_alpha: float = Field(gt=0.0, lt=1.0)
    alpha: float = Field(gt=0.0, lt=0.5)
    fdr: Literal["bh", "by"]
    min_obs: StrictInt = Field(ge=50)
    signals: tuple[str, ...] | None
    bootstrap: CausalBootstrapConfig

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.signals is not None:
            if not self.signals:
                raise ValueError("signals must be None (all features) or a non-empty list")
            dupes = sorted({s for s in self.signals if self.signals.count(s) > 1})
            if dupes:
                raise ValueError(f"duplicate signals: {dupes}")
        if self.min_obs <= 4 * self.tau_max:
            raise ValueError(
                f"min_obs {self.min_obs} leaves too few rows for tau_max {self.tau_max}"
            )
        return self


class DecayConfig(_Frozen):
    """Stage 2 (``invariant.stages.decay.stage``; docs/adr/0011 for the test, 0012 for the stage).

    A signal Stage 1 passed is killed when the one-sided sup-split decline test of its best
    link gives ``p < alpha``. ``n_boot`` sets the replicates of the fixed-regressor wild
    bootstrap, and ``trim`` and ``break_step`` set the break grid. ``min_obs`` is the minimum
    number of design rows. The test's size and power were measured only at T = 6000
    (ADR 0011), so a shorter sample is killed with ``general.insufficient_data`` rather than
    decided by an unvalidated test. Stage 2 reports no half-life: no estimator was admissible
    (ADR 0011).
    """

    alpha: float = Field(gt=0.0, lt=0.5)
    n_boot: StrictInt = Field(ge=99, le=100_000)
    trim: float = Field(gt=0.0, lt=0.5)
    break_step: StrictInt = Field(ge=1)
    min_obs: StrictInt = Field(ge=100)


class PipelineConfig(_Frozen):
    universe: UniverseConfig
    dates: DateRange
    frequency: Frequency
    seed: StrictInt = Field(ge=0, lt=2**32)
    data: DataConfig
    pit: PitConfig
    causal: CausalConfig
    decay: DecayConfig


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
