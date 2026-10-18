"""Stage 1 helpers: a planted-signal PIT panel through the real snapshot path, and a fast config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.fixtures.synthetic_snapshot import benchmark_config, write_benchmark_snapshot

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.synthetic import SyntheticDataset, generate_market
from invariant.config.settings import CausalConfig, PipelineConfig
from invariant.data.pit import PitPanel, build_pit_panel
from invariant.data.snapshot import Snapshot


@dataclass(frozen=True)
class BenchmarkPanel:
    dataset: SyntheticDataset
    snapshot: Snapshot
    config: PipelineConfig
    panel: PitPanel


def benchmark_panel(
    tmp_path: Path, preset: str | None = None, *, n_obs: int = 1500, seed: int = 7
) -> BenchmarkPanel:
    cfg = load_benchmark_config(preset).model_copy(update={"n_obs": n_obs, "burn_in": 200})
    dataset = generate_market(cfg, seed)
    snapshot = write_benchmark_snapshot(dataset, tmp_path / "snapshots")
    config = benchmark_config(dataset)
    config = config.model_copy(update={"causal": fast_causal(config.causal, dataset)})
    return BenchmarkPanel(dataset, snapshot, config, build_pit_panel(snapshot, config))


def fast_causal(base: CausalConfig, dataset: SyntheticDataset | None = None) -> CausalConfig:
    """``base`` with a small bootstrap, and the benchmark's candidates as the signals."""
    update: dict[str, object] = {
        "bootstrap": base.bootstrap.model_copy(update={"n_boot": 199}),
    }
    if dataset is not None:
        update["signals"] = tuple(s.signal_id for s in dataset.ground_truth.signals)
    return CausalConfig.model_validate({**base.model_dump(), **update})
