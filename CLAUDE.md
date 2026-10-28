# INVARIANT

## What this is
An autonomous quant research pipeline. Five stages, each capable of killing a candidate signal:
1. Causal discovery: is the signal structurally upstream of returns, or confounded?
2. Decay validation: transfer-entropy half-life; is the edge being arbitraged away?
3. Portfolio construction: Wasserstein DRO was implemented, verified and evaluated (ADR 0013);
   ridge regularization (Ridge-CV) was selected. The memo must say this, never "DRO portfolio".
4. Adversarial regime certification: worst plausible market path, with a realism constraint
5. Research memo: citation-backed investment memo arguing for or against the strategy

Full spec: docs/invariant-spec.html

## The bar
Every claim this system makes must be measured, reproducible, and able to survive a hostile
reviewer. Rigor beats feature count. A smaller component that is provably correct beats a
bigger one that only looks impressive. If something in the spec is statistically unsound,
say so instead of implementing it as written.

## Non-negotiables
1. **No look-ahead.** A value used at time t uses only information available at t. Macro data
   uses ALFRED vintages and release dates, never final revised values. Every data transform
   gets a test in tests/leakage/.
2. **Ground truth first.** Every stage is validated on the planted-signal benchmark
   (src/invariant/benchmark/) before it touches real data. A stage that can't recover the
   planted truth is not done.
3. **Uncertainty, not point estimates.** Every estimated quantity (transfer entropy, half-life,
   Sharpe, effect sizes) ships with a CI, bootstrap distribution, or surrogate significance test.
4. **Multiple-testing awareness.** When many candidates are tested, control the FDR and report
   the Deflated Sharpe Ratio, not the raw Sharpe.
5. **Reproducibility.** Every run is keyed by a hash of (data snapshot id, config, git SHA, seed).
   All randomness flows through explicitly seeded np.random.Generator objects. Never use
   global np.random or unseeded randomness.
6. **Every rejection is recorded** in the kill ledger: stage, reason code, and the numbers
   behind the decision.
7. **Honest limitations.** When market data violates a method's assumptions (stationarity,
   causal sufficiency, i.i.d.), document it in the docstring and in docs/.

## Layout
- src/invariant/config/        settings (.env) + YAML config loading
- src/invariant/core/          typed contracts, run hashing/reproducibility, logging
- src/invariant/data/          sources (ETF, French factors, ALFRED), snapshots, point-in-time assembly
- src/invariant/benchmark/     synthetic planted-signal market + ground truth
- src/invariant/stages/        causal/ decay/ dro/ (evaluated DRO code) portfolio/ adversarial/ memo/
- src/invariant/ledger/        kill ledger
- src/invariant/orchestration/ pipeline DAG (later)
- src/invariant/api/           FastAPI service (later)
- tests/unit, tests/integration, tests/benchmark, tests/leakage
- docs/adr/                    architecture decision records
- data/, runs/                 gitignored, local only

## Conventions
- Python 3.11 managed with uv. Add deps with `uv add` (never pip) and say why each is needed.
- Full type hints; mypy strict must pass. No `Any` without a comment explaining why.
- Pydantic v2 models for configs and cross-stage contracts, frozen where possible.
- Pure functions for math, I/O at the edges. Stages exchange typed result objects, not dicts.
- Time series: sorted, unique, tz-naive DatetimeIndex, validated at module boundaries.
- Logging via structlog. No print() in library code.
- Every non-trivial algorithm's docstring states the method, its assumptions, and a paper reference.
- Secrets only from .env. Tunables in configs/*.yaml.
- Tests: unit tests for all logic, hypothesis property tests for mathematical invariants.
  Network calls are mocked in unit tests; real calls are marked @pytest.mark.slow.

## Commands
- uv run pytest -m "not slow"     # fast suite
- uv run pytest                   # everything
- uv run ruff check . --fix && uv run ruff format .
- uv run mypy
- uv run pre-commit install        # once per clone: installs the pre-commit and pre-push hooks

The pre-push hook runs ruff check, ruff format --check, mypy and the fast suite on the whole
tree, so a failing check cannot be pushed. Never bypass it with `--no-verify`.

## How to work
- For any non-trivial task: propose a plan and wait for approval before writing code.
- Write tests alongside the implementation, not after.
- A task is done only when pytest, ruff, and mypy all pass. Show the output.
- Record significant design decisions as a short ADR at docs/adr/NNNN-title.md
  (context, decision, alternatives considered, consequences).
- Don't silently change public interfaces.
- End each task with a suggested conventional commit message.

### Method choices
- Any choice between methods is made by a pre-registered study. The ADR states the rule, the
  gates, how M is derived and the tie-break, and it is committed and pushed before any full
  run. Results are committed afterwards, in a separate section, and never edited into the
  pre-registration.
- Size M so that a correctly behaving method passes each gate with high probability, and
  state the pass probabilities (at the expected rate and at the failure rate) in the ADR.
- A departure from a rule's outcome is recorded as a deviation, with its reasons, and
  confirmed on fresh seeds where possible.
- Every arm of a study runs on one tiny seed in the fast suite, so a smoke run can't miss a
  crashing code path.
- Long runs use a git worktree, so edits in the main tree can't change the code the workers
  import.
- Post-launch bug fixes are documented in the ADR: the cause, the fix commit, and whether the
  run resumed from checkpoints or restarted, with the reason.

## Current phase
- Phase 0 (foundations) is done.
- Stage 1 (causal, `hc3|hybrid`; ADRs 0009–0010) is accepted on the benchmark.
- Stage 2 (decay: a one-sided sup-split decline test, no half-life reported; ADRs 0011–0012)
  is accepted on the benchmark.
- Stage 3 method study (ADR 0013) is done: all size gates pass; Ridge-CV selected by the
  pre-registered gross-Sharpe rule (DRO-FC is best net of 5 bp; a net-of-cost confirmation
  study is in the backlog).
- Stage 3 stage (portfolio construction on Ridge-CV, kill rule A, diagnostic B; ADR 0014) is
  accepted on the benchmark. Stage 4 (adversarial regime certification) is next.
