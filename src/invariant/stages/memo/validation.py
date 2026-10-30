"""Validation of the memo verifier on the planted-signal benchmark (docs/adr/0017).

Method
------
**Runs.** For each run preset (``RunPreset``: a benchmark preset, optionally at another
``n_obs``) and replication ``i``, the dataset and pipeline seeds come from
``RngFactory(root_seed).child("seed/i")`` (``dataset``, ``pipeline``), as in the earlier
validations. Stages 1-3 run through the production path (``benchmark.run_benchmark_pipeline``).
Each run then:
- builds its template memo twice and checks the two are byte-identical (G4);
- writes the run's GroundTruth into the run directory and checks the fact sheet and the memo are
  unchanged (G4): the memo cannot read the answer key;
- verifies the template memo (G1: it must be accepted);
- draws ``instances_per_class`` corruptions of every class K1-K9 (``corruption.generate``), each
  from its own stream ``RngFactory(root_seed).child("corrupt/<preset>/<i>").generator(
  "<class>/<j>")``, and verifies each (G2: each must be rejected).

**Coverage rule.** After ``n_seeds`` replications per preset, if fewer than ``coverage_min``
runs fall in an outcome category (``CATEGORIES``), replication ``n_seeds``, ``n_seeds + 1``, ...
is added to every preset that ``COVERAGE_PRESETS`` maps that category to, one round at a time,
until every category is covered or those presets reach ``max_seeds``. Seeds are never chosen by
hand, and the rule reads pipeline outcomes only, never a verification result.

**Hand-written sets** (G3) are applied to the fixture memos (``tests/fixtures/memo_runs``) by
``corruption.apply_ops`` and verified; every corruption entry must be rejected. Benign entries
are reported (false rejections), not gated.

GroundTruth is read here only to write it into the run directory for the G4 check and to report
how the recommendation relates to the planted truth; no memo or verifier code reads it.

Reference: Clopper & Pearson (1934) for the zero-miss bound, (1 - p)^n.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import yaml
from joblib import Parallel, delayed, parallel_config
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from scipy.stats import beta

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.synthetic import generate_market
from invariant.config.settings import ConfigError, find_project_root
from invariant.core.logging import get_logger
from invariant.core.run_hash import GitState, RngFactory, canonical_json
from invariant.stages.memo.benchmark import run_benchmark_pipeline
from invariant.stages.memo.corruption import CLASSES, CorruptionError, apply_ops, generate
from invariant.stages.memo.document import render
from invariant.stages.memo.facts import FactSheet, build_fact_sheet
from invariant.stages.memo.template import build_template_memo
from invariant.stages.memo.verify import verify

__all__ = [
    "CATEGORIES",
    "COVERAGE_PRESETS",
    "MemoValidationConfig",
    "RunPreset",
    "RunRecord",
    "load_memo_validation_config",
    "run_hand_sets",
    "run_task",
    "run_validation",
]

DEFAULT_CONFIG: Final = Path("configs/memo_validation.yaml")

CATEGORIES: Final = (
    "stage1_kill",
    "stage2_kill",
    "stage3_kill",
    "multi_member_pass",
    "candidate",
    "against",
    "not_decided",
)
COVERAGE_PRESETS: Final[Mapping[str, tuple[str, ...]]] = {
    "stage1_kill": ("base", "realistic"),
    "stage2_kill": ("base", "multi_asset"),
    "stage3_kill": ("hidden_confounder", "multi_asset"),
    "multi_member_pass": ("multi_asset", "volfrag"),
    "candidate": ("multi_asset", "volfrag"),
    "against": ("realistic", "hidden_confounder"),
    "not_decided": ("base_short",),
}
"""For each outcome category, the run presets the coverage rule adds seeds to."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunPreset(_Frozen):
    name: str
    preset: str
    """A benchmark preset (``configs/benchmark.yaml``), or ``base`` for the default market."""
    n_obs: StrictInt | None = Field(default=None, ge=200)


class MemoValidationConfig(_Frozen):
    root_seed: StrictInt = Field(ge=0, lt=2**32)
    presets: tuple[RunPreset, ...] = Field(min_length=1)
    n_seeds: StrictInt = Field(ge=1)
    max_seeds: StrictInt = Field(ge=1)
    coverage_min: StrictInt = Field(ge=0)
    instances_per_class: StrictInt = Field(ge=1)
    min_instances: StrictInt = Field(ge=1)
    """G2: each class needs at least this many verified corruptions."""
    hand_sets: tuple[str, ...]
    """Set files, relative to the project root."""


def load_memo_validation_config(path: Path | None = None) -> MemoValidationConfig:
    p = path if path is not None else DEFAULT_CONFIG
    p = p if p.is_absolute() else find_project_root() / p
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"memo validation config not found: {p}") from None
    return MemoValidationConfig.model_validate(data)


class ClassRecord(_Frozen):
    cls: str
    n: int
    misses: int
    channels: dict[str, int]
    checks: dict[str, int]
    """How often each check (V1-V9) fired among the instances."""
    missed: tuple[str, ...]
    """Channel and tampered files of each miss, for audit."""


class RunRecord(_Frozen):
    preset: str
    index: int
    run_id: str
    categories: tuple[str, ...]
    recommendation: str
    members: tuple[str, ...]
    passed: tuple[str, ...]
    passed_with_true_edge: tuple[bool, ...]
    """For each passed signal, whether GroundTruth plants an edge (reported only)."""
    template_accepted: bool
    template_failures: tuple[str, ...]
    deterministic: bool
    ground_truth_independent: bool
    classes: tuple[ClassRecord, ...]
    seconds: float


def _u32(factory: RngFactory, name: str) -> int:
    return int(factory.seed_sequence(name).generate_state(1, np.uint32)[0])


def _categories(sheet: FactSheet) -> tuple[str, ...]:
    out: list[str] = []
    killed = {(str(f.stage), f.status.value) for f in sheet.fates}
    for stage, cat in (
        ("causal", "stage1_kill"),
        ("decay", "stage2_kill"),
        ("portfolio", "stage3_kill"),
    ):
        if (stage, "killed") in killed:
            out.append(cat)
    if len(sheet.members) >= 2 and all(
        sheet.fate(m).status.value == "passed" for m in sheet.members
    ):
        out.append("multi_member_pass")
    rec = {"candidate for paper trading": "candidate", "against trading": "against"}
    out.append(rec.get(str(sheet.recommendation), "not_decided"))
    return tuple(out)


def run_task(
    preset: RunPreset,
    index: int,
    cfg: MemoValidationConfig,
    *,
    git_state: GitState,
    patch: bytes | None = None,
    work_dir: Path | None = None,
    keep_dir: Path | None = None,
) -> RunRecord:
    """One run (module docstring). Pure in all but ``seconds``."""
    t0 = time.perf_counter()
    factory = RngFactory(cfg.root_seed).child(f"seed/{index}")
    ds_seed, pl_seed = _u32(factory, "dataset"), _u32(factory, "pipeline")
    market = load_benchmark_config(None if preset.preset == "base" else preset.preset)
    if preset.n_obs is not None:
        market = market.model_copy(update={"n_obs": preset.n_obs})
    dataset = generate_market(market, ds_seed)
    corrupt = RngFactory(cfg.root_seed).child(f"corrupt/{preset.name}/{index}")
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        run = run_benchmark_pipeline(dataset, pl_seed, Path(tmp), git_state=git_state, patch=patch)
        sheet = build_fact_sheet(run)
        memo = build_template_memo(sheet)
        md = render(memo, sheet).text
        memo_json = canonical_json(memo.model_dump(mode="json"))
        again = build_template_memo(build_fact_sheet(run))
        deterministic = again == memo and render(again, sheet).text == md
        report = verify(memo_json, md, run)
        classes: list[ClassRecord] = []
        for cls in CLASSES:
            channels: Counter[str] = Counter()
            checks: Counter[str] = Counter()
            missed: list[str] = []
            for j in range(cfg.instances_per_class):
                inst = generate(cls, memo, md, sheet, corrupt.generator(f"{cls}/{j}"))
                r = verify(inst.tampered.memo_json, inst.tampered.md, run)
                channels[inst.channel] += 1
                checks.update(r.checks_failed())
                if r.accepted:
                    missed.append(
                        json.dumps(
                            {
                                "channel": inst.channel,
                                "memo_json": inst.tampered.memo_json,
                                "md": inst.tampered.md,
                            }
                        )
                    )
            classes.append(
                ClassRecord(
                    cls=cls,
                    n=cfg.instances_per_class,
                    misses=len(missed),
                    channels=dict(sorted(channels.items())),
                    checks=dict(sorted(checks.items())),
                    missed=tuple(missed),
                )
            )
        truth = dataset.ground_truth
        (run / "ground_truth.json").write_text(truth.model_dump_json(), encoding="utf-8")
        sheet_gt = build_fact_sheet(run)
        independent = (
            sheet_gt == sheet and render(build_template_memo(sheet_gt), sheet_gt).text == md
        )
        if keep_dir is not None and (not report.accepted or any(c.misses for c in classes)):
            shutil.copytree(run, keep_dir / f"{preset.name}-{index}", dirs_exist_ok=True)
    passed = tuple(f.signal_id for f in sheet.fates if f.status.value == "passed")
    edges = {s.signal_id: s.has_causal_edge for s in truth.signals}
    return RunRecord(
        preset=preset.name,
        index=index,
        run_id=sheet.run_id,
        categories=_categories(sheet),
        recommendation=str(sheet.recommendation),
        members=sheet.members,
        passed=passed,
        passed_with_true_edge=tuple(bool(edges.get(s, False)) for s in passed),
        template_accepted=report.accepted,
        template_failures=tuple(f"{f.check}: {f.detail}" for f in report.failures),
        deterministic=deterministic,
        ground_truth_independent=independent,
        classes=tuple(classes),
        seconds=time.perf_counter() - t0,
    )


def _short_categories(records: Sequence[RunRecord], cfg: MemoValidationConfig) -> list[str]:
    counts = Counter(c for r in records for c in r.categories)
    return [c for c in CATEGORIES if counts[c] < cfg.coverage_min]


def run_validation(
    cfg: MemoValidationConfig,
    *,
    git_state: GitState,
    patch: bytes | None = None,
    n_jobs: int = 1,
    checkpoint: Path | None = None,
    keep_dir: Path | None = None,
) -> list[RunRecord]:
    """Every run of the validation, with the coverage rule (module docstring). Resumes from
    ``checkpoint`` (JSONL); results are independent of ``n_jobs``."""
    done: dict[tuple[str, int], RunRecord] = {}
    if checkpoint is not None and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = RunRecord.model_validate_json(line)
                done[(rec.preset, rec.index)] = rec
    by_name = {p.name: p for p in cfg.presets}
    seeds = {p.name: cfg.n_seeds for p in cfg.presets}
    log = get_logger(__name__)
    while True:
        todo = [(n, i) for n in by_name for i in range(seeds[n]) if (n, i) not in done]
        log.info("memo_validation_round", n_tasks=len(todo), n_reused=len(done), n_jobs=n_jobs)
        with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
            recs: list[RunRecord] = Parallel()(
                delayed(run_task)(
                    by_name[n], i, cfg, git_state=git_state, patch=patch, keep_dir=keep_dir
                )
                for n, i in todo
            )
        for rec in recs:
            done[(rec.preset, rec.index)] = rec
        if checkpoint is not None and recs:
            with checkpoint.open("a", encoding="utf-8") as fh:
                fh.writelines(r.model_dump_json() + "\n" for r in recs)
        records = [done[(n, i)] for n in by_name for i in range(seeds[n])]
        grow = {
            p
            for c in _short_categories(records, cfg)
            for p in COVERAGE_PRESETS[c]
            if p in seeds and seeds[p] < cfg.max_seeds
        }
        if not grow:
            return records
        for p in grow:
            seeds[p] += 1


class HandEntryResult(_Frozen):
    set: str
    id: str
    cls: str
    expect: str
    accepted: bool
    checks: tuple[str, ...]


def run_hand_sets(cfg: MemoValidationConfig, fixtures: Path | None = None) -> list[HandEntryResult]:
    """Every entry of the hand-written sets, verified against its fixture run."""
    root = find_project_root()
    runs = fixtures if fixtures is not None else root / "tests" / "fixtures" / "memo_runs"
    out: list[HandEntryResult] = []
    for rel in cfg.hand_sets:
        doc = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
        for e in doc["entries"]:
            sheet = build_fact_sheet(runs / e["case"])
            memo = build_template_memo(sheet)
            try:
                t = apply_ops(memo, render(memo, sheet).text, e["ops"], sheet)
            except CorruptionError as exc:
                raise CorruptionError(f"set {doc['set']} entry {e['id']}: {exc}") from exc
            r = verify(t.memo_json, t.md, runs / e["case"])
            out.append(
                HandEntryResult(
                    set=str(doc["set"]),
                    id=str(e["id"]),
                    cls=str(e["class"]),
                    expect=str(e["expect"]),
                    accepted=r.accepted,
                    checks=r.checks_failed(),
                )
            )
    return out


# ---------------------------------------------------------------------------------------------
# Summary and gates
# ---------------------------------------------------------------------------------------------


class GateResult(_Frozen):
    name: str
    passed: bool
    detail: str


class ClassSummary(_Frozen):
    cls: str
    n: int
    misses: int
    miss_rate_upper: float
    """Exact one-sided 95% upper bound on the miss rate: 1 - 0.05^(1/n) at zero misses."""
    channels: dict[str, int]
    checks: dict[str, int]


class HandSetSummary(_Frozen):
    set: str
    n_corruptions: int
    missed: tuple[str, ...]
    n_benign: int
    benign_rejected: tuple[str, ...]


class MemoValidationReport(_Frozen):
    n_runs: int
    runs_per_preset: dict[str, int]
    categories: dict[str, int]
    coverage_met: bool
    classes: tuple[ClassSummary, ...]
    hand_sets: tuple[HandSetSummary, ...]
    gates: tuple[GateResult, ...]
    accepted: bool
    recommendations: dict[str, int]
    candidate_with_edgeless_pass: int
    """Candidate memos in which some passed signal has no planted edge (reported only)."""
    median_seconds: float
    git_sha: str
    git_dirty: bool


def _upper(n: int, misses: int, confidence: float = 0.95) -> float:
    if n == 0:
        return 1.0
    if misses == 0:
        return float(1.0 - (1.0 - confidence) ** (1.0 / n))
    return float(beta.ppf(confidence, misses + 1, n - misses))


def summarize_validation(
    records: Sequence[RunRecord],
    hand: Sequence[HandEntryResult],
    cfg: MemoValidationConfig,
    *,
    git_sha: str,
    git_dirty: bool,
) -> MemoValidationReport:
    """The report and gates G1-G4 (docs/adr/0017)."""
    cats = Counter(c for r in records for c in r.categories)
    classes = []
    for cls in CLASSES:
        rs = [c for r in records for c in r.classes if c.cls == cls]
        n, misses = sum(c.n for c in rs), sum(c.misses for c in rs)
        channels: Counter[str] = Counter()
        checks: Counter[str] = Counter()
        for c in rs:
            channels.update(c.channels)
            checks.update(c.checks)
        classes.append(
            ClassSummary(
                cls=cls,
                n=n,
                misses=misses,
                miss_rate_upper=_upper(n, misses),
                channels=dict(sorted(channels.items())),
                checks=dict(sorted(checks.items())),
            )
        )
    sets = []
    for name in sorted({h.set for h in hand}):
        hs = [h for h in hand if h.set == name]
        bad = [h for h in hs if h.expect == "reject"]
        ok = [h for h in hs if h.expect == "accept"]
        sets.append(
            HandSetSummary(
                set=name,
                n_corruptions=len(bad),
                missed=tuple(h.id for h in bad if h.accepted),
                n_benign=len(ok),
                benign_rejected=tuple(h.id for h in ok if not h.accepted),
            )
        )
    false_rej = [f"{r.preset}/{r.index}" for r in records if not r.template_accepted]
    g2_bad = [c.cls for c in classes if c.misses or c.n < cfg.min_instances]
    g3_bad = [f"{s.set}:{m}" for s in sets for m in s.missed]
    g4_bad = [
        f"{r.preset}/{r.index}"
        for r in records
        if not (r.deterministic and r.ground_truth_independent)
    ]
    gates = (
        GateResult(
            name="G1",
            passed=not false_rej and bool(records),
            detail=f"{len(records) - len(false_rej)} of {len(records)} template memos accepted",
        ),
        GateResult(
            name="G2",
            passed=not g2_bad,
            detail="; ".join(f"{c.cls} {c.misses}/{c.n}" for c in classes),
        ),
        GateResult(
            name="G3",
            passed=not g3_bad and len(sets) == len(cfg.hand_sets),
            detail="; ".join(
                f"set {s.set}: {len(s.missed)}/{s.n_corruptions} missed" for s in sets
            ),
        ),
        GateResult(
            name="G4",
            passed=not g4_bad and bool(records),
            detail=f"{len(records) - len(g4_bad)} of {len(records)} deterministic and "
            "GroundTruth-independent",
        ),
    )
    secs = sorted(r.seconds for r in records)
    return MemoValidationReport(
        n_runs=len(records),
        runs_per_preset=dict(Counter(r.preset for r in records)),
        categories={c: cats[c] for c in CATEGORIES},
        coverage_met=all(cats[c] >= cfg.coverage_min for c in CATEGORIES),
        classes=tuple(classes),
        hand_sets=tuple(sets),
        gates=gates,
        accepted=all(g.passed for g in gates),
        recommendations=dict(Counter(r.recommendation for r in records)),
        candidate_with_edgeless_pass=sum(
            1
            for r in records
            if r.recommendation == "candidate for paper trading"
            and not all(r.passed_with_true_edge)
        ),
        median_seconds=secs[len(secs) // 2] if secs else 0.0,
        git_sha=git_sha,
        git_dirty=git_dirty,
    )
