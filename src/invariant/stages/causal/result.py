"""Stage 1 result contracts. Frozen, JSON-serialisable, and the only thing Stage 1 hands on.

Every per-link number keeps its uncertainty next to it: p-values for every test, the family
and FDR-adjusted p-values the verdict is decided on, and a block-bootstrap interval for the
effect size at the best link. See docs/adr/0010 for how each number is produced.
"""

from __future__ import annotations

from datetime import date
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from invariant.config.settings import CausalConfig
from invariant.core.types import KillReason, StageName, Verdict, make_signal_id
from invariant.stages.causal.bootstrap import BootstrapCI

__all__ = ["CausalResult", "LagTest", "LaggedLink", "SignalCausalLink", "SignalEvidence"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


_P = Field(ge=0.0, le=1.0)
_R = Field(ge=-1.0, le=1.0)


class LagTest(_Frozen):
    """The tests of ``signal_{t-lag} -> asset_t``: the deciding test and MCI, same CI test."""

    asset: str
    lag: int = Field(ge=1)
    p: float = _P
    """Deciding test's p-value (``CausalConfig.decision``)."""
    r: float = _R
    """Deciding test's partial correlation."""
    p_mci: float = _P
    """MCI p-value. Diagnostic only (ADR 0009: low power on persistent signals)."""
    r_mci: float = _R


class LaggedLink(_Frozen):
    """One MCI link of the PCMCI graph with ``p <= alpha``. Uncorrected, descriptive only."""

    source: str
    target: str
    lag: int = Field(ge=1)
    p: float = _P
    r: float = _R


class SignalEvidence(_Frozen):
    """The numbers behind one signal's verdict, present whenever PCMCI ran.

    ``best_asset``/``best_lag`` are the test with the smallest p-value. The lag is **not
    identified**: when PC1 misses the true lag, an adjacent lag stands in for it
    (docs/adr/0009, finding 5). Read it as "where the evidence is strongest", never as the
    signal's true horizon. ``r_best_ci`` holds the conditioning set and the link fixed at their
    selected values, so it is not selection-adjusted.
    """

    tests: tuple[LagTest, ...] = Field(min_length=1)
    n_family_tests: int = Field(ge=1)
    family_p: float = _P
    """Bonferroni over the signal's (asset, lag) tests: ``min(1, K min p)``."""
    mci_family_p: float = _P
    """The same Bonferroni over the MCI p-values. Diagnostic only."""
    q: float = _P
    """``family_p`` adjusted for the FDR across the signals of this run."""
    best_asset: str
    best_lag: int = Field(ge=1)
    best_conditions: tuple[tuple[str, int], ...]
    """Conditioning set ``(column, lag)`` of the deciding test at the best link."""
    r_best_ci: BootstrapCI

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.n_family_tests != len(self.tests):
            raise ValueError("n_family_tests must equal the number of tests")
        if self.q < self.family_p:
            raise ValueError("an FDR-adjusted p-value is never below the unadjusted one")
        return self


class SignalCausalLink(_Frozen):
    """Stage 1's decision on one candidate signal. ``evidence`` is ``None`` only when the stage
    could not test (insufficient or invalid data), and then the reason is a ``general.*`` one."""

    signal_id: str
    column: str
    verdict: Verdict
    reason: KillReason | None
    evidence: SignalEvidence | None

    @field_validator("signal_id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return make_signal_id(value)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.verdict is Verdict.KILL) != (self.reason is not None):
            raise ValueError("a kill needs a reason and a pass must not have one")
        if self.reason is not None and self.reason.stage not in (None, StageName.CAUSAL):
            raise ValueError(f"reason {self.reason} does not belong to Stage 1")
        if self.evidence is None and (self.reason is None or self.reason.stage is not None):
            raise ValueError("only a general.* kill may come without evidence")
        return self


class CausalResult(_Frozen):
    """Everything Stage 1 decided and measured on one panel."""

    config: CausalConfig
    tigramite_version: str
    n_obs: int = Field(ge=0)
    """Rows of the input after trimming leading missing values (0 if none are complete)."""
    n_rows_tested: int | None
    """Rows each test uses: ``n_obs - 2 tau_max``. ``None`` when PCMCI did not run."""
    first_date: date | None
    last_date: date | None
    var_names: tuple[str, ...]
    signal_columns: tuple[str, ...]
    return_columns: tuple[str, ...]
    parents: dict[str, tuple[tuple[str, int], ...]]
    """PC1 parents ``(column, lag)`` of every variable. Empty when PCMCI did not run."""
    graph: tuple[LaggedLink, ...]
    signals: tuple[SignalCausalLink, ...]
    data_issue: str | None = None
    """Why the stage could not test, when it could not."""

    @property
    def tested(self) -> bool:
        return self.n_rows_tested is not None
