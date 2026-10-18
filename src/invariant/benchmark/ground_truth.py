"""Ground truth exported with every planted-signal dataset.

``GroundTruth`` is the answer key that every pipeline stage is scored against: the true lagged
causal graph, each candidate signal's expected verdict and killing stage, the true half-life,
the population moments the data were drawn from, all generation parameters, and the seed. It is
a frozen pydantic model that round-trips through JSON exactly.

Graph semantics
---------------
Nodes are variables; an edge ``(source, target, lag)`` means ``source_{t-lag}`` appears in the
structural equation of ``target_t``. ``mechanism`` says how:

- ``linear``: additive term ``coefficient * source_{t-lag}``.
- ``linear_time_varying``: as linear, but ``coefficient`` is the *initial* value; the path
  multiplier is the ``decay_multiplier`` column of the dataset's latents.
- ``variance``: ``source`` enters only the conditional variance of ``target`` (GARCH: the
  residual ``a_{t-1}`` enters ``h_t`` with loading ``coefficient = alpha``). It has zero linear
  effect on the conditional mean, so a linear (partial-correlation) test should *not* find it;
  a nonlinear CI test may. Stages must be scored with that in mind.

Expected outcomes
-----------------
``expected_verdict`` / ``expected_kill_stage`` are the **oracle** outcomes: what a correct method
decides with unlimited data. They are not a promise that a finite-sample test reaches them. Each
``SignalTruth`` therefore also carries population ICs and an approximate expected Stage-1
t-statistic, so a stage test can distinguish "the stage is wrong" from "the stage had no power".
``identifiable=False`` marks a signal whose true status cannot be recovered from the observed
variables at all (S_conf when Z is hidden); stage tests should measure, not assert, those.

Target timing
-------------
Every expected value above is a moment of ``(S_t, R_{t+1})``. ``target_timing`` records that as
the ``TargetConfig`` (execution lag 0, horizon 1) a stage's targets must use to be scored against
this answer key. Scoring with execution lag 1 would compare a stage against
``Corr(S_t, R_{t+2})``, a decayed version of the planted effect, and silently pass or fail it
for the wrong reason. ``require_target_timing`` refuses any other timing.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from enum import StrEnum
from typing import Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from invariant.benchmark.config import SyntheticMarketConfig
from invariant.config.settings import TargetConfig
from invariant.core.types import StageName, Verdict, make_signal_id

__all__ = [
    "EXPECTED_VALUES_TARGET_TIMING",
    "GROUND_TRUTH_SCHEMA_VERSION",
    "TIMING_CONVENTION",
    "CausalEdge",
    "GroundTruth",
    "Mechanism",
    "Node",
    "NodeRole",
    "RealizedLoadings",
    "SignalTruth",
    "TargetTimingMismatchError",
]

GROUND_TRUTH_SCHEMA_VERSION = 3  # 2: added target_timing; 3: added expected_t_hybrid

TIMING_CONVENTION = (
    "A value dated t is known at the close of trading day t. returns[t] is the log return from "
    "close(t-1) to close(t). A signal dated t is scored against returns[t+1]. All mean edges have "
    "lag >= 1 except the latent market factor (lag 0)."
)

EXPECTED_VALUES_TARGET_TIMING = TargetConfig(execution_lag=0, horizon=1)
"""The timing every expected value in ``SignalTruth`` is computed for: ``(S_t, R_{t+1})``."""


class TargetTimingMismatchError(ValueError):
    """A benchmark run's target timing differs from the one ``GroundTruth`` was computed for."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NodeRole(StrEnum):
    TARGET = "target"
    CONFOUNDER = "confounder"
    CANDIDATE = "candidate"
    LATENT = "latent"


class Mechanism(StrEnum):
    LINEAR = "linear"
    LINEAR_TIME_VARYING = "linear_time_varying"
    VARIANCE = "variance"


class Node(_Frozen):
    name: str
    role: NodeRole
    observed: bool


class CausalEdge(_Frozen):
    """``source_{t-lag}`` enters the equation of ``target_t`` (see module docstring)."""

    source: str
    target: str
    lag: int = Field(ge=0)
    mechanism: Mechanism
    coefficient: float


class RealizedLoadings(_Frozen):
    """Per-asset return loadings actually used, after dispersion draws. Index = asset order."""

    beta_z: tuple[float, ...]
    beta_real: tuple[float, ...]
    beta_decay_initial: tuple[float, ...]
    market_beta: tuple[float, ...]

    @model_validator(mode="after")
    def _same_length(self) -> Self:
        lengths = {
            len(self.beta_z),
            len(self.beta_real),
            len(self.beta_decay_initial),
            len(self.market_beta),
        }
        if len(lengths) != 1:
            raise ValueError("all loading vectors must have one entry per asset")
        return self


class SignalTruth(_Frozen):
    """Answer key for one candidate signal. Per-asset tuples follow ``GroundTruth.asset_names``.

    ``marginal_ic``: population ``Corr(S_t, R_{t+1})`` with every coefficient at its initial value.
    ``partial_ic``: population partial correlation given the other *observed* parents of R
    (so for S_conf with Z hidden it is the confounding bias an oracle cannot remove).
    ``effective_partial_ic``: as ``partial_ic`` with S_decay's coefficient replaced by its
    full-sample average (stationary approximation; equal to ``partial_ic`` when nothing decays).
    ``expected_t``: ``effective_partial_ic * sqrt(n_obs - 1)``, the approximate full-sample t-stat
    of a correctly specified linear test. Low-SNR approximation; ignores estimation of the
    conditioning coefficients.
    ``expected_t_hybrid``: the same for the test Stage 1 decides with (``hybrid``, ADR 0009) at
    lag 1: the population partial correlation of ``S_{t-1}`` and ``R_t`` given the *true*
    observed parents of R other than S, and S's observed parents other than its own lag shifted
    by one (``Z_{t-2}`` for a Z-loading signal when Z is observed), with S_decay at its average
    coefficient, times ``sqrt(n_obs - 1)``. Stage 1 conditions on PC1's estimated parents, not
    the true ones, and decides over several lags with Bonferroni, so this is a power proxy,
    not an exact prediction. For S_conf with Z hidden it is nonzero: the bias no observed
    adjustment set removes.
    """

    signal_id: str
    has_causal_edge: bool
    expected_verdict: Verdict
    expected_kill_stage: StageName | None
    identifiable: bool
    marginal_ic: tuple[float, ...]
    partial_ic: tuple[float, ...]
    effective_partial_ic: tuple[float, ...]
    expected_t: tuple[float, ...]
    expected_t_hybrid: tuple[float, ...]
    coef_half_life: float | None = None
    te_half_life_approx: float | None = None
    decay_start: date | None = None
    final_coef_fraction: float | None = None
    note: str = ""

    @field_validator("signal_id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return make_signal_id(value)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if (self.expected_verdict is Verdict.KILL) != (self.expected_kill_stage is not None):
            raise ValueError("expected_kill_stage must be set exactly when the verdict is KILL")
        decay_fields = (
            self.coef_half_life,
            self.te_half_life_approx,
            self.decay_start,
            self.final_coef_fraction,
        )
        if len({f is None for f in decay_fields}) != 1:
            raise ValueError("decay fields must be all set or all unset")
        return self


class GroundTruth(_Frozen):
    """The answer key exported with a synthetic dataset (see module docstring)."""

    schema_version: int = GROUND_TRUTH_SCHEMA_VERSION
    seed: int = Field(ge=0, lt=2**32)
    config: SyntheticMarketConfig
    timing_convention: str = TIMING_CONVENTION
    target_timing: TargetConfig
    nodes: tuple[Node, ...]
    edges: tuple[CausalEdge, ...]
    signals: tuple[SignalTruth, ...]
    asset_names: tuple[str, ...]
    loadings: RealizedLoadings
    state_order: tuple[str, ...]
    stationary_covariance: tuple[tuple[float, ...], ...]
    first_date: date
    last_date: date
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _check_graph(self) -> Self:
        names = [n.name for n in self.nodes]
        dupes = sorted(n for n, c in Counter(names).items() if c > 1)
        if dupes:
            raise ValueError(f"duplicate node names: {dupes}")
        known = set(names)
        keys = [(e.source, e.target, e.lag) for e in self.edges]
        for source, target, lag in keys:
            if source not in known or target not in known:
                raise ValueError(f"edge {source}->{target} (lag {lag}) references unknown node")
        dup_edges = sorted(k for k, c in Counter(keys).items() if c > 1)
        if dup_edges:
            raise ValueError(f"duplicate edges: {dup_edges}")
        candidates = {n.name for n in self.nodes if n.role is NodeRole.CANDIDATE}
        signal_ids = [s.signal_id for s in self.signals]
        if len(set(signal_ids)) != len(signal_ids) or set(signal_ids) != candidates:
            raise ValueError("signals must list each candidate node exactly once")
        targets = [n.name for n in self.nodes if n.role is NodeRole.TARGET]
        if tuple(targets) != self.asset_names:
            raise ValueError("asset_names must equal the target nodes, in order")
        n_assets = len(self.asset_names)
        if len(self.loadings.beta_z) != n_assets:
            raise ValueError("loadings must have one entry per asset")
        for s in self.signals:
            for field in (
                s.marginal_ic,
                s.partial_ic,
                s.effective_partial_ic,
                s.expected_t,
                s.expected_t_hybrid,
            ):
                if len(field) != n_assets:
                    raise ValueError(f"{s.signal_id}: per-asset fields need {n_assets} entries")
        k = len(self.state_order)
        cov = self.stationary_covariance
        if len(cov) != k or any(len(row) != k for row in cov):
            raise ValueError("stationary_covariance must be square over state_order")
        return self

    # -- queries -----------------------------------------------------------------------------

    def require_target_timing(self, timing: TargetConfig) -> None:
        """Raise ``TargetTimingMismatchError`` unless ``timing`` equals ``target_timing``."""
        if timing != self.target_timing:
            raise TargetTimingMismatchError(
                f"benchmark targets built with execution_lag={timing.execution_lag}, "
                f"horizon={timing.horizon}, but GroundTruth expected values assume "
                f"execution_lag={self.target_timing.execution_lag}, "
                f"horizon={self.target_timing.horizon}; scores would not be comparable"
            )

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(n.name for n in self.nodes)

    @property
    def max_lag(self) -> int:
        return max((e.lag for e in self.edges), default=0)

    @property
    def snapshot_id(self) -> str:
        """Content-addressed id usable as ``RunInputs.snapshot_id``."""
        return f"synthetic-{self.dataset_sha256[:16]}"

    def node(self, name: str) -> Node:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def signal(self, signal_id: str) -> SignalTruth:
        for s in self.signals:
            if s.signal_id == signal_id:
                return s
        raise KeyError(signal_id)

    def adjacency(
        self,
        lag: int,
        *,
        mechanisms: frozenset[Mechanism] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Boolean matrix ``A[i, j]`` = edge ``node_i -> node_j`` at ``lag`` (order: node_names).

        ``mechanisms`` restricts to those edge types (e.g. exclude ``VARIANCE`` when scoring a
        linear method).
        """
        index = {name: i for i, name in enumerate(self.node_names)}
        out = np.zeros((len(index), len(index)), dtype=np.bool_)
        for e in self.edges:
            if e.lag == lag and (mechanisms is None or e.mechanism in mechanisms):
                out[index[e.source], index[e.target]] = True
        return out

    def parents(
        self,
        target: str,
        *,
        mechanisms: frozenset[Mechanism] | None = None,
    ) -> frozenset[tuple[str, int]]:
        """``{(source, lag)}`` of edges into ``target``, optionally filtered by mechanism."""
        if target not in self.node_names:
            raise KeyError(target)
        return frozenset(
            (e.source, e.lag)
            for e in self.edges
            if e.target == target and (mechanisms is None or e.mechanism in mechanisms)
        )
