# ADR 0007: Kill ledger and benchmark scoring

Status: accepted (Phase 0)

## Context
Non-negotiable #6 requires every rejection to be recorded with its stage, reason code and the
numbers behind it. Non-negotiable #2 requires every stage to be validated on the planted-signal
benchmark. The kill ledger is the record. The benchmark scorer is how the whole pipeline will be
graded, so it must not be easy to fool, either by a ledger from the wrong run or by treating an
undecided signal as a pass.

## Decisions

**Format.** `runs/<run_id>/kill_ledger.jsonl` holds one `LedgerEntry` per line in canonical JSON:
`seq, run_id, attempt, signal_id, stage, verdict, reason, metrics, placeholder, recorded_at,
prev_sha256`. Every decision is recorded, PASS as well as KILL, so a signal's full path through
the stages can be reconstructed.

**Append-only and tamper-evident.** There is no update or delete API. Lines form a hash chain
(`prev_sha256`), and `kill_ledger.head.json` (entry count plus last-line hash) is replaced
atomically after each append. The head matters because a chain alone cannot see a valid edit to
the last line or a truncation at a line boundary. Every read verifies the chain, `seq`, the head,
the run id, canonical form and the decision rules, and fails with `LedgerIntegrityError` rather
than repairing anything. This detects accidents and casual edits, not a determined forger who
recomputes every hash (that would need signing or an external anchor).

**Decision rules.** Per attempt, a signal's decisions follow `StageName` order with no gaps,
each stage at most once, and nothing after a KILL. A KILL needs a `KillReason` and non-empty
metrics. A PASS has no reason. A stage-specific reason must match the stage. Metrics are finite
scalars.

**Placeholders.** `placeholder: bool` (default False) is part of every hashed line. PASS entries
carry no reason, so a reason code alone could not mark them. `general.placeholder` is accepted
only on entries with `placeholder=True`. The scorer's `contains_placeholders` checks the field,
so a placeholder PASS is flagged as well.

**Attempts.** A resumed run (ADR 0002) appends its entries under a new `attempt` number, which
keeps the file append-only. Queries default to the latest attempt, so a crashed attempt's partial
decisions are never mixed in. Writing requires run status `created` and the attempt the ledger
was opened for.

**Scoring.** Fates are computed over an evaluated stage *prefix*. If the expected kill stage is
outside the prefix, the expected outcome is PASS, so Phases 1–2 can be graded on causal, then
causal plus decay. Each identifiable signal is a correct kill, wrong kill, correct pass or missed
kill, and every correct kill records `stage_match`. Signals that GroundTruth marks
`identifiable=False` are reported per signal but kept out of the matrix (ADR 0003). Undecided
signals are listed and make the score `complete=False`; they are never passes. Each row carries
`low_power`. `score_run` refuses a run unless all of the following hold: its snapshot is the
given one; that snapshot is synthetic with the GroundTruth's `dataset_sha256`; and its target
timing equals `GroundTruth.target_timing` (ADR 0006).

**Aggregation.** One run's matrix is a count over four signals. Pipeline grading pools runs over
independent seeds with Wilson 95% intervals. Per-signal rates are valid binomial proportions
across seeds. Pooled precision and recall mix signals that share a dataset, so their intervals
are optimistic, and the code documents this.

**Synthetic data as a snapshot.** `benchmark/payload.py` writes the dataset as an `etf` payload
(tickers `SYN00…`, prices `100·exp(cumsum r)`, `provider: synthetic`, `dataset_sha256` in the
params). Benchmark runs then take the real data path (snapshot, run id, PIT panel), and the
scorer can link a run to its dataset.

## Open question: per-signal verdicts in stages 3 and 4
**Stage 3: answered in ADR 0014** (2026-10-27). The portfolio's verdict is recorded on every
member's entry (stage `portfolio`, formerly `dro`). Stage 4 remains open.

Causal discovery and decay validation judge each signal on its own, so a per-signal PASS/KILL is
well-defined there. DRO builds **one portfolio** from the surviving signals, and the adversarial
stage certifies **that portfolio**. Neither naturally produces a verdict per signal. Possible
answers: portfolio-level ledger entries (a pseudo-signal for the portfolio), or a mapping such
as "zero DRO weight = kill at dro" together with "the adversary decertifies the portfolio = kill
every member". The choice changes what a "correct kill at dro" means. It is deferred until those
stages are built. Until then, the stage-prefix rule lets Phases 1–2 be graded without settling
it. The current GroundTruth expects no kills at dro or adversarial, so nothing depends on the
answer yet.

## Alternatives considered
- *SQLite ledger*: queries and locking for free, but it is a binary file and harder to diff,
  audit or hash. JSONL plus a hash chain is enough at this scale.
- *Truncating the ledger on resume*: simpler queries, but it breaks append-only and erases the
  evidence of a crashed attempt. Rejected.
- *Treating undecided signals as passes*: an unfinished pipeline would score as if it kept every
  true signal. Rejected.
- *A new `SourceName.SYNTHETIC`*: cleaner typing, but it would need new snapshot accessors and
  PIT plumbing for no behavioural difference. Rejected for now; `provider` in the params marks
  the data instead. *Superseded by ADR 0008*, which adds the source for the observed features
  (prices stay in `etf`).

## Consequences
- `KillReason` gained `general.placeholder`.
- mypy now also checks `scripts/`.
- Every future stage records through `KillLedger.record` and inherits the rules above. Stage
  benchmark tests should grade through `score_run`, so that the dataset and timing guards apply.
