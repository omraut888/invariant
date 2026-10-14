"""Planted-signal datasets written as snapshots, and a detector for oracle-latent leaks.

``latent_leaks`` is the check behind "oracle-only latents never enter the snapshot or panel".
It looks for latents by *name* (every table column, every params key and string, every panel
column with its source prefix stripped) and by *value* (any panel feature column or synthetic
feature column numerically equal to a latent column, which catches a renamed leak). ``z`` is a
latent and also a legitimate observed feature when ``observe_confounder`` is set, so it is
exempt exactly when GroundTruth marks it observed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import JsonValue

from invariant.benchmark.config import load_benchmark_config, load_benchmark_target_timing
from invariant.benchmark.payload import (
    observable_feature_names,
    synthetic_etf_payload,
    synthetic_feature_payload,
    synthetic_tickers,
)
from invariant.benchmark.synthetic import SyntheticDataset, generate_market
from invariant.config.settings import PipelineConfig, load_config
from invariant.data.pit import PitPanel
from invariant.data.snapshot import Snapshot, write_snapshot
from invariant.data.sources.base import SourcePayload

PRESETS: tuple[str | None, ...] = (None, "hidden_confounder", "multi_asset")
_PREFIXES = ("ret_", "ff_", "macro_", "syn_", "fwd_")


def small_dataset(preset: str | None = None, seed: int = 5, n_obs: int = 300) -> SyntheticDataset:
    cfg = load_benchmark_config(preset).model_copy(update={"n_obs": n_obs, "burn_in": 10})
    return generate_market(cfg, seed)


def benchmark_config(dataset: SyntheticDataset) -> PipelineConfig:
    """The base pipeline config pointed at ``dataset`` with the benchmark target timing."""
    truth = dataset.ground_truth
    base = load_config()
    return base.model_copy(
        update={
            "universe": base.universe.model_copy(
                update={"tickers": synthetic_tickers(truth.asset_names)}
            ),
            "dates": base.dates.model_copy(
                update={"start": truth.first_date, "end": truth.last_date}
            ),
            "pit": base.pit.model_copy(update={"targets": load_benchmark_target_timing()}),
        }
    )


def write_benchmark_snapshot(
    dataset: SyntheticDataset,
    snapshots_dir: Path,
    *,
    features: SourcePayload | None = None,
) -> Snapshot:
    """Write the ETF and feature payloads (``features`` replaces the genuine feature payload)."""
    feature_payload = features if features is not None else synthetic_feature_payload(dataset)
    return write_snapshot([synthetic_etf_payload(dataset), feature_payload], snapshots_dir)


def with_feature_table(payload: SourcePayload, table: pd.DataFrame) -> SourcePayload:
    """``payload`` with its ``features`` table replaced (bypasses the payload's allowlist)."""
    return dataclasses.replace(payload, tables={"features": table})


def forbidden_latents(dataset: SyntheticDataset) -> frozenset[str]:
    return frozenset(map(str, dataset.latents.columns)) - observable_feature_names(
        dataset.ground_truth
    )


def _json_strings(value: JsonValue) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _json_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _json_strings(v)


def _strip(column: str) -> str:
    for prefix in _PREFIXES:
        if column.startswith(prefix):
            return column[len(prefix) :]
    return column


def _matches(values: np.ndarray, latent: np.ndarray) -> bool:
    return values.shape == latent.shape and bool(
        np.allclose(values, latent, rtol=0.0, atol=1e-12, equal_nan=True)
    )


def latent_leaks(snapshot: Snapshot, panel: PitPanel, dataset: SyntheticDataset) -> list[str]:
    """Every way an oracle-only latent reaches ``snapshot`` or ``panel`` (empty if none)."""
    forbidden = forbidden_latents(dataset)
    leaks: list[str] = []
    for entry in snapshot.manifest.sources:
        for name in set(_json_strings(dict(entry.params))) & forbidden:
            leaks.append(f"{entry.source} params mention {name!r}")
        for f in entry.files:
            for col in set(f.columns) & forbidden:
                leaks.append(f"{f.path} has column {col!r}")

    for col in panel.features.columns:
        if _strip(str(col)) in forbidden:
            leaks.append(f"panel has column {col!r}")

    latents = dataset.latents.reindex(panel.features.index)
    candidates: dict[str, np.ndarray] = {
        f"panel:{c}": panel.features[c].to_numpy(dtype="float64") for c in panel.features.columns
    }
    if any(e.source == "synthetic" for e in snapshot.manifest.sources):
        table = snapshot.synthetic_features().reindex(panel.features.index)
        candidates |= {f"snapshot:{c}": table[c].to_numpy(dtype="float64") for c in table.columns}
    for latent in sorted(forbidden):
        target = latents[latent].to_numpy(dtype="float64")
        for where, values in candidates.items():
            if _matches(values, target):
                leaks.append(f"{where} equals latent {latent!r}")
    return leaks
