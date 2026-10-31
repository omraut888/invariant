# ADR 0017: Stage 5, the research memo (fact sheet, template memo, verifier)

Status: **accepted** (2026-10-31): the template-only verifier passed every gate on the rerun of
deviation 1. Pre-registered 2026-10-30 (approved by the user the same day); the validation was
fixed and pushed before its run. Results are in separate sections at the end, after the
pre-registration, and are not edited into it. Run 1 failed G3; deviation 1 cut the LLM narrative
and made the verifier template-only.

## Context
Stages 1–3 decide; Stage 4 was evaluated and not adopted as a kill stage (ADR 0016). Stage 5
turns one run's kill ledger and stage results into a memo that argues for or against trading
the strategy. Its failure mode is the unsupported claim: a wrong number, a verdict the ledger
does not hold, a missing caveat, an uncited appeal to the literature, or a claim the pipeline
forbids ("distributionally robust", "certified"). The design makes these structurally hard and
a verifier rejects what remains. A memo that fails verification is **rejected, not patched**.

## Decisions (user, 2026-10-30)
- **The template memo is the product.** An LLM narrative is optional and clearly marked; if time
  runs short it is cut first. On a rejected narrative: up to 2 fresh attempts, then the
  template-only memo; every attempt is logged.
- **Positive recommendation wording:** "candidate for paper trading".
- **Benchmark validation rates are quoted**, loaded from the archived reports with tests. They
  are stated as natural sentences that name the planted-signal benchmark, the preset and the
  ADR ("On the planted-signal benchmark (multi-asset pipeline, ADR 0014), a signal with no edge
  passed all three kill stages in 0.015 [0.010, 0.023] of runs (95% interval), almost always as
  a passenger of a true signal"). On real data the memo says explicitly that they are benchmark
  validation rates, not properties of the strategy.
- **The recommendation rule** (revised from the plan, approved): candidate if any signal passed
  every kill stage; otherwise **not decided** if any signal was killed without a test decision
  (a `general.*` reason: it could have survived); otherwise against. The plan's "not decided
  only if every kill is general" said "against" when one signal was untested.
- The "not detectable" statement keeps the realistic-against-base power comparison.
- **Entailment entries of the hand-written sets are reported, not gated** (G3), consistent with
  V7's limit as stated in the plan before set (b) existed. The exemption is defined by class,
  not by entry id (below).
- **The LLM narrative is kept, time-boxed to one day after the validation run.** If it overruns,
  it is cut to the backlog and Stage 5 ships template-only.

## Decision

### Fact sheet (`stages/memo/facts.py`, `reference.py`)
Built from the run directory only: the ledger (hash chain verified on read), the three stage
result files and the manifest, plus the archived validation reports for quoted rates. Each
stage's verdict and reason, and the shared numbers, are cross-checked between the ledger and the
result files; a disagreement refuses the memo. A run that is not completed, a ledger with
placeholders and a signal with no final verdict are refused. **GroundTruth is never read**
(`tests/leakage/test_memo_inputs.py`). Every fact has a stable id, a kind that fixes its
rendering (p-values `< 0.001` or 3 decimals, Sharpe ratios 2, rates 3, coefficients `.2e`), a
digit-free label, a subject and its **sources** (`ledger:…`, `result:…`, `manifest:…`,
`reference:…`, `const:…`). Quoted rates check the preset, scenario and signal at their report
pointer, so a re-ordered report fails instead of quoting the wrong row.

### The memo document (`document.py`, `template.py`)
`memo.json` is the document; `memo.md` its rendering. Paragraphs are segments: text, or an
**atom** rendering one fact-sheet item in a fixed self-describing form (`{fact:id}` "subject's
label = value", `{fact:id|value}` in tables, `{fate:s}` from the ledger, `{stage:…}`,
`{cite:key}`, `{adr:n}`, `{term:…}`, `{recommendation}`). **Every digit of a memo lies inside an
atom** (tested), so no number enters except through the fact sheet. Sections in fixed order:
recommendation, signals, causal, decay, portfolio, stress, limits, methods, reproducibility,
references.

### Mandatory statements (`mandatory.py`)
The limits section holds exactly the statements whose trigger fires, in registry order:

| id | trigger |
|---|---|
| data_source | always (synthetic and real variants; real: ADR 0004's limits and "benchmark validation rate … not a property of this strategy") |
| hidden_confounder | a signal passed Stage 1 |
| not_detectable | a kill for `causal.no_significant_link`, `causal.not_significant_after_correction` or `portfolio.no_edge_over_equal_weight` |
| not_decided | a `general.*` kill |
| stage1_descriptive | a Stage 1 partial correlation is reported |
| no_half_life | a signal reached Stage 2 |
| decay_power | a signal passed Stage 2 |
| portfolio_method | Stage 3 built a portfolio ("Wasserstein DRO … evaluated … not selected"; "not distributionally robust") |
| gross_sharpe | Stage 3 built a portfolio (stronger wording when Ridge-CV's net Sharpe ≤ 0) |
| deflated_sharpe | Ridge-CV's DSR is reported |
| passenger | Stage 3 passed with ≥ 2 members (names members whose B interval contains 0) |
| collateral | Stage 3 killed ≥ 2 members |
| stage4_not_adopted | always |
| stress_description | the stress diagnostic is reported ("a description, not a certification") |

### Literature (`literature.yaml`)
A curated corpus, not Meridian: 26 entries cited by the ADRs, with bibliographic metadata and a
summary of at most 200 characters in our own words (no text from the papers). Every DOI was
checked against Crossref (`scripts/check_literature_dois.py`); all 26 resolve to the named
paper (Politis & Romano 1992 has none). Bai & Perron and Mohajerin Esfahani & Kuhn show their
online-first years there; the volume years the ADRs cite are kept. Meridian was not reused: a
separate repo, Docker Qdrant and two API keys (no offline mode), an off-domain corpus
(distributed systems), full-text chunks, and a PolyForm Strict licence.

### Verifier (`verify.py`)
`verify(memo.json, memo.md, run_dir)` rebuilds the fact sheet and reads the raw ledger lines
and result files by its own code path. Any failure rejects; every failure is reported.
- **V1 structure:** run id, title, recommendation field and sections are the template's; every
  template block equals the re-render; narrative paragraphs only end the narrative sections
  (recommendation, causal, decay, portfolio, stress), at most 2 per section, with no statement
  id, and only text, full-form fact, fate and cite segments; the narrative flag matches; the
  bibliography is exactly the cited keys.
- **V2 atoms:** every atom names an existing fact, signal, corpus key or term; every fact value
  equals its sources resolved from the raw files.
- **V3 numbers:** every digit of the rendering lies inside an atom; every number in an atom
  equals, at its displayed precision, the raw value its source names.
- **V4 narrative text:** no digit (any script), number or quantity word, frequency or
  probability word, signal id, stage name, verdict word, causal or performance claim,
  recommendation word or Markdown control character.
- **V5 verdicts:** the recommendation equals the rule recomputed from the raw ledger; every
  clause of `memo.md` that joins a signal id to a verdict word has a canonical form and agrees
  with the raw ledger.
- **V6 mandatory statements:** the statement ids present are exactly the triggered set, which is
  also recomputed from the raw ledger; each is a template paragraph of the limits section.
- **V7 literature:** cite keys are in the corpus; a narrative sentence with a literature cue (an
  author's surname, "et al", "literature", "studies", "research", "found", "shown", "according
  to", "well known", …) carries a cite atom.
- **V8 banned phrases** in narrative text ("distributionally robust", "DRO portfolio",
  "certif*", "stress-test*", "guarantee*", "prove*", "causal effect", "half-life", "robust*",
  "-proof", "immune", …).
- **V9 rendering:** `memo.md` equals `render(memo.json)`.

`write_memo` writes a memo only after the verifier accepts the exact bytes written.

**Limits, stated before the run.** The verifier checks that a literature claim is cited, not
that the cited paper supports it (entailment is semantic; the bibliography shows our summary so
a reader can check). Template memos assign citations deterministically, a fixed key to a fixed
statement, so the gap applies only to the optional LLM narrative. It checks narrative prose by lexicon, so a claim phrased in words it does
not list can pass V4–V8; it trades false rejections of harmless prose for catch rate, on
purpose.

### Offline mode and the LLM narrative (optional, not yet built)
`memo.llm.enabled: false` by default: the memo is template-only, deterministic and needs no key.
The planned narrative path: `anthropic` as an optional extra, `claude-opus-5-5` with structured
outputs whose segment enums are this run's fact ids, signal ids and corpus keys; server-side
refusal fallbacks; every request and response saved to `memo/llm_attempts/`; up to 2 fresh
attempts, each verified, then the template memo. The API takes no sampling parameters, so a
narrative is reproduced from its saved artifact, not by rerunning. Tests use a fake client with
recorded responses. The narrative is validated by those tests and this verifier; it is not part
of the gates below.

## Validation (pre-registered 2026-10-30, before the run)
**Harness.** `invariant.stages.memo.validation`, `scripts/memo_validation.py`, config
`configs/memo_validation.yaml`. Root seed **20261105** (fresh). Replication i of a preset takes
its dataset and pipeline seeds from `RngFactory(20261105).child("seed/i")`; Stages 1–3 run
through the production path (`stages.memo.benchmark`).

**Run set.** Presets base, realistic, hidden_confounder, multi_asset, volfrag, and base at
`n_obs` 4000 (`base_short`: below Stage 2's `min_obs`, so its Stage 1 survivors are killed
without a test decision), 8 seeds each: 48 runs.
**Coverage rule.** Categories: a Stage 1 kill, a Stage 2 kill, a Stage 3 kill, a multi-member
Stage 3 pass, and each recommendation. If fewer than 3 runs fall in a category, one more seed
(8, 9, …) is added to each preset that `COVERAGE_PRESETS` maps it to, round by round, up to 40
per preset. The rule reads pipeline outcomes only, never a verification result. If a category is
still short at 40, that is reported and the gates are evaluated on the runs made.

**Per run:** the template memo is built twice (byte-identical), verified, and rebuilt with the
run's GroundTruth written into the run directory (unchanged). Then **12 seeded corruptions of
each class** (`corruption.generate`, stream `RngFactory(20261105).child("corrupt/<preset>/<i>")
.generator("<class>/<j>")`) are verified:

| class | channels (one drawn at random per instance) |
|---|---|
| K1 wrong number | a number in a fact atom nudged in its last digit, sign-flipped or swapped with another, in `memo.md`; two template fact atoms of one form swapped in `memo.json` |
| K2 number in prose | a narrative sentence with a number (digits, %, or words); a number put into template text; "(n = …)" put into `memo.md` |
| K3 atom repointed | a fact atom's id changed, `memo.md` kept or re-rendered |
| K4 wrong verdict | fate atoms swapped; a fate's text replaced by the opposite outcome in `memo.md`; a narrative verdict sentence |
| K5 recommendation | the field flipped; the line flipped in `memo.md`; a narrative recommendation |
| K6 mandatory | a statement deleted, a spurious one added, relabelled, handed to the narrative, or reworded (a negation or a word removed) |
| K7 literature | a citation removed, a non-corpus key, an uncited literature sentence, a bibliography entry added or removed |
| K8 banned claim | a narrative sentence with a banned claim |
| K9 `memo.md` drift | a character substituted, inserted or deleted, or a line duplicated |

Narrative channels draw from fixed sentence banks, so G2 measures the verifier's mechanism on
those banks; how far its lexicons generalise is measured by the hand-written sets.

**Hand-written sets** (`tests/fixtures/memo_adversarial/`), applied to the fixture memos:
- **(a)**, 96 corruptions and 12 benign sentences, written by the verifier's author before the
  verifier existed (commit 545f0f8, pushed before the verifier commit ec5b879). The verifier
  rejected every corruption on its first run; three verifier bugs that caused false rejections
  of template memos and benign sentences were then fixed (clause punctuation, a block-quote
  marker, "paper trading" read as a literature cue). No lexicon entry was added for a set (a)
  entry.
- **(b)**, 128 corruptions and 13 benign sentences, written by an independent agent given only
  the memo format, the materials of the fixture memos and the class definitions, never the
  verifier code or set (a) (commit 1661e14). **The verifier has not been run on set (b).** Its
  author's report named ten entries it judged hardest, so the verifier's author knows those
  ten; no verifier change was made after it except the one below.
- **One lexicon change after set (b) was written**: decline words (decline, weaken, fade,
  erode) became narrative verdict words. The cause was a miss in a smoke run of the K4
  generator on the fixtures ("Its link to returns did not decline."), not set (b) (commit
  90c3b32).

**Gates (the verifier is accepted only if all hold):**
- **G1.** Every run's template memo is accepted: zero false rejections.
- **G2.** For each class K1–K9: at least 500 verified instances and zero misses. At 576
  instances and zero misses the exact one-sided 95% upper bound on the miss rate is .0052.
- **G3.** Zero misses on the corruption entries of sets (a) and (b), **except entailment
  entries**: a real corpus key attached to a claim the paper does not make. They test
  entailment, which V7 states it does not check; they are reported, not gated. The class is
  defined by an entry's operations (`validation.entailment_only`), never by its id: a K7 entry
  whose operations are only narrative insertions in which every sentence carries a cite atom
  of a corpus key, plus bibliography additions of keys those insertions cite. An entry that also
  drops a citation, uses a non-corpus key, leaves a sentence uncited or does anything else is
  gated. Classified before the verifier is run on set (b), the class holds no set (a) entry and
  two set (b) entries, b_k7_06 and b_k7_07, the two its author flagged as needing a semantic
  check (`tests/unit/test_memo_validation.py` pins this).
- **G4.** Every run's template memo is deterministic and its fact sheet ignores GroundTruth;
  every run completes.

**Pass probabilities.** The checks are deterministic, so a correct verifier passes every gate
with probability 1. A verifier with true per-instance miss rate p in a class passes G2 for that
class with probability (1 − p)^576:

| p | 0 | .0005 | .001 | .002 | .005 | .01 |
|---|---|---|---|---|---|---|
| P(G2 passes, one class) | 1 | .750 | .562 | .316 | .056 | .003 |

G1 with a per-memo false-rejection rate r over 48 memos: (1 − r)^48 = .617 at r = .01 and
.085 at r = .05. G3 is a fixed set: it passes if and only if no gated entry is missed.

**Reported, not gated:** false rejections of the benign sentences of each set; which checks
fired per class and channel; the category and recommendation counts; how many candidate memos
include a passed signal with no planted edge (the passenger risk the memo states); runtime.

**Compute.** One task per preset on root seed 1 (not this run's; only timings were looked at):
base 15 s, realistic 2 s, hidden_confounder 17 s, multi_asset 25 s, volfrag 128 s, base_short
2 s, about 190 s per seed of all six. 48 runs ≈ 25 min serial, a few minutes on 8 workers; at
the coverage rule's maximum (40 seeds of every preset) about 2 h on 8 workers allowing 2× for
load. No reported-only fallback is needed.

**The gate's limit.** It measures the verifier against the defined classes and two hand-written
sets, not against every way an LLM can fail. A miss is recorded with its cause and fix; the
fixed verifier is then rerun on fresh corruption seeds (and, for a hand-set miss, a new
independent set) as a recorded deviation. Thresholds never move.

If any gate fails, Stage 5 is not done and the design returns to review.

## Process
- The validation runs from a git worktree at the pushed commit of this ADR.
- Post-launch fixes are recorded here with their cause, the fix commit, and whether the run
  resumed from its checkpoint or restarted.
- The unit tests verify set (a) only; set (b) is verified only by this validation.
- Two look-ahead property tests (`tests/leakage/test_data_transforms.py`: log returns, the
  adjusted-close panel) had hypothesis's default 200 ms deadline and failed intermittently under
  load; they are correctness tests, not performance tests, and now run with `deadline=None`.
  Test-only; no code path changes.
- After the run: the LLM narrative, time-boxed to one day; on an overrun it goes to the backlog.

## Alternatives considered
- **Meridian for literature retrieval:** rejected (above). An optional search that proposes
  entries for a human to curate is in the backlog.
- **An LLM judge of entailment:** a reported-only backlog item; a judge's accept is not a check.
- **Letting the LLM write numbers and verifying them by string match:** rejected; a correct
  number attached to the wrong quantity passes a string match. Atoms carry their subject and
  label.
- **Repairing a rejected memo:** rejected by the requirement; a repaired memo is a new memo
  with no verification history.

## Results, run 1 (recorded 2026-10-31, after the run)
Run from a worktree at the pushed pre-registration commit 01ffdf8, clean tree, root seed
20261105, 8 workers, about 20 minutes. No post-launch fix. Report and hand-set results:
`docs/results/0017-memo-validation-run1/`.

48 runs (8 per preset; the coverage rule added none). Categories: Stage 1 kill 48, Stage 2 kill
27, Stage 3 kill 9, multi-member pass 9; recommendations: candidate 21, against 20, not decided 7.

| gate | result |
|---|---|
| G1 | **pass**: 48 of 48 template memos accepted |
| G2 | **pass**: K1–K9 each 0 misses in 576 (upper bound .0052) |
| G3 | **fail**: set (a) 0 of 96 missed; set (b) 7 of 126 gated entries missed |
| G4 | **pass**: 48 of 48 deterministic and GroundTruth-independent |

**The verifier was not accepted: G3 failed.** It is recorded as it stands; no entry is exempted
after the fact. All seven misses are narrative sentences on which no check fired:
- **Character evasions (3):** b_k8_08 ("rоbust" with a Cyrillic о), b_k8_09 ("certified" with a
  zero-width space), b_s_11 (a narrative paragraph of only a zero-width space). V4 and V8 matched
  raw characters, with no Unicode normalisation and no rejection of format characters.
- **Claims worded outside the lexicons (4):** b_k4_05 (a kill reframed as "for practical
  purposes as good as clearing the bar"), b_k4_06 ("none of these signals is genuinely
  predictive", read as proof of no edge), b_k5_06 ("still merits a modest live position" in an
  against-trading run), b_k6_14 ("the volatility proxy tracks the true stress regime closely",
  undercutting the stress caveat). This is the limit stated before the run ("a claim phrased in
  words it does not list can pass V4–V8"), and G3 gated these entries anyway.

**Reported, not gated.** Benign sentences falsely rejected: set (a) 0 of 12, **set (b) 8 of 13**
(mostly V4, also V5 and V7; harmless signposting such as "Each stage's evidence follows in
order."). Entailment entries: b_k7_06 missed, as expected. Candidate memos with a passed
edgeless signal: in the report. Median run time 27 s.

**b_k7_07 was rejected by V1 by accident.** The template orders the bibliography by key, and V1
requires the references section to equal the bibliography rebuilt from the cited keys, in that
order. The entry cites bollerslev1986 in a narrative sentence and adds the reference with
`add_reference`, which appended it after wilson1927. The cited set and the bibliography matched;
only the order differed. With the reference inserted at its sorted position the verifier
accepts the entry. So in substance both entailment entries passed the verifier, as V7's stated
limit predicts. The fix is to the corruption tool, not the verifier: `add_reference` now
inserts at the sorted position, so an added entry is caught for what it adds, not for where it
sits (`tests/unit/test_memo_corruption.py`).

**Reading.** The template memo, the product, met every gate and all of set (a). Every miss and
every benign false rejection was free narrative prose. Against an independent author the
lexicon checks failed in both directions: they passed claims worded in new ways and rejected
most harmless sentences. At that false-rejection rate most LLM attempts would fall back to the
template anyway.

## Deviation 1: template-only verifier and V10 (user, 2026-10-31; fixed and pushed before the rerun)
Decisions (user, 2026-10-31):
1. Record the G3 failure as it stands (above); no post-hoc exemptions.
2. **The LLM narrative is cut to the backlog**, superseding the one-day time-box. The verifier
   is template-only: V1 rejects any narrative paragraph outright, and any `narrative` flag
   other than `none`. V4, V7 and V8 still run on a narrative paragraph so a rejection says what
   the prose did, but they no longer decide anything.
3. **The character-evasion class is fixed generally**, with no lexicon change: a new check
   **V10** requires `memo.md` and every string of `memo.json` to be NFKC-normal (Unicode UAX
   #15), with no format character (category Cf) and no non-ASCII letter, mark or digit
   (categories L, M, N), which covers confusables. Template memos and the corpus are pure ASCII
   (tested), so V10 costs no false rejection. **No word was added for the four claim misses.**
4. b_k7_07 is explained and recorded (above); `add_reference` inserts at the sorted position.
5. The validation is rerun under the unchanged gates, on fresh seeds, with a new independent set.

**Rerun.** Config `configs/memo_validation_deviation1.yaml`: the run set, coverage rule,
`instances_per_class`, `min_instances` and gates of the pre-registration, unchanged; **root seed
20261107** (fresh, used nowhere else), so both the runs and the corruption streams are new; hand
sets (a), (b) and **(c)**. Run from a worktree at the pushed commit of this deviation. A test
pins that the config differs from the original only in the seed and the added set.

**Set (c)** (`tests/fixtures/memo_adversarial/set_c.yaml`): 146 corruptions (K1 13, K2 14, K3 14, K4 13, K5 14,
K6 18, K7 14, K8 14, K9 16, S 16) over all eight fixture cases, and no benign entry (its author
found no change a correct template-only verifier must accept). Six are narrative insertions; the
rest tamper with the template memo itself (`memo.json`, `memo.md`, or both kept consistent).
It was written by a new agent given only a materials folder: the template-only memo format
(`document.py`), the eight fixture memos, the mandatory registry, the corpus, the edit
operations, the class definitions and a checker that applies entries without verifying them;
never the verifier, sets (a) or (b), the tests or this ADR. It was frozen as written, all 146
entries, before the verifier was run on it. Its author's report named the entries it judged
hardest, so the verifier's author knows those; **no verifier change was made after set (c) was
written.** Its entailment-class entries, if any, are exempt by the same class rule.

**Gates, unchanged.** G1–G4 as pre-registered, with G3 over sets (a), (b) and (c) and the same
by-class entailment exemption. Pass probabilities are as stated: (1 − p)^576 per class for G2;
G1, G3 and G4 are deterministic.

**What the rerun can and cannot show.**
- In template-only mode, G2's narrative channels (in K2, K4, K5, K7 and K8) and every hand-set
  narrative entry are rejected by V1's structural rule. For them the rerun measures that rule,
  not the lexicons; the per-check counts in the report show which.
- Set (b) has been seen and its misses fixed by design, so on the rerun it is a regression
  check, not an independent one. Set (c) is the independent measure.
- Benign narrative sentences in sets (a) and (b) are now rejected by design. They are still
  reported; their rejection rate no longer means anything.
- The gate still measures the verifier against the defined classes and three hand-written sets,
  not against every possible tampering of a template memo.

If any gate fails, Stage 5 is not done and the design returns to review; thresholds never move.

## Results, deviation 1 rerun (recorded 2026-10-31, after the run)
Run from a worktree at the pushed deviation commit de8a7de, clean tree, root seed 20261107, 8
workers, about 20 minutes (median run 32 s). No post-launch fix. Report and hand-set results:
`docs/results/0017-memo-validation-deviation1/`.

48 runs (8 per preset; the coverage rule added none). Categories: Stage 1 kill 48, Stage 2 kill
32, Stage 3 kill 7, multi-member pass 16; recommendations: candidate 27, against 13, not
decided 8.

| gate | result |
|---|---|
| G1 | **pass**: 48 of 48 template memos accepted |
| G2 | **pass**: K1–K9 each 0 misses in 576 (upper bound .0052) |
| G3 | **pass**: set (a) 0 of 96, set (b) 0 of 126, set (c) 0 of 146 missed |
| G4 | **pass**: 48 of 48 deterministic and GroundTruth-independent |

**The template-only verifier is accepted, and Stage 5 is accepted on the benchmark with it.**

**Reported, not gated.** Every benign narrative sentence of sets (a) and (b) (25) is rejected, by
design in template-only mode. Set (b)'s entailment entries: 0 of 2 missed (V1's template-only
rule). Candidate memos with a passed edgeless signal: 5 of 27, the passenger risk the memo
states. Set (c)'s author's hardest-ranked entries were all rejected; c_k5_01 (the
recommendation field flipped with `memo.md` unchanged) by V5.

**Which checks caught set (c)** (an entry can fire several): V1 alone 66 of 146, V9 alone 18,
V3 with V9 14, V10 on 12 (including fullwidth digits, c_k2_12), V6 on 9, V5 on 11.

**Limitations of the result.**
- **Concentration on V1.** In template-only mode the only acceptable memo is the template memo,
  so V1's equality of every template block with the verifier's re-render is the right check:
  it needs no lexicon and so generalises. But it alone caught 66 of set (c)'s 146, including
  the misattributed citations (c_k7_03–06: a corpus key on the right topic, the wrong paper), an
  atom repointed to a fact that renders identically (c_k3_14) and a swapped run id (c_s_01).
  Most template tampering therefore rests on one equality. A bug shared by the template and the
  re-render the verifier uses would pass it; the raw-file checks (V2, V3, V5, V6) are the
  independent second line, and they did not fire on those 66.
- **"-0.00" against "0.00".** V3 compares a displayed number with its raw value at the displayed
  precision, so it treats "-0.00" and "0.00" as equal. c_k1_02 changed one to the other and
  was caught by V9 only because `memo.md` then differed from the rendering; a consistent change
  in both files rests on V1 alone. Not changed: the sign of a zero rounded to the display is
  not a claim about the run, but a sign shown for a value that rounds to zero is a reporting
  defect a reader could misread.
- **Scope.** The pass covers the template memo only. It says nothing about checking free prose:
  that surface was removed by deviation 1, not solved. Set (b) was a regression check on this
  run, not an independent one.

## Backlog (added 2026-10-31)
- **LLM narrative** (cut by deviation 1). A free-prose narrative failed against an independent
  set; it is not retried in that form.
- **Structured LLM narrative:** the model chooses and orders among verified atoms and fixed
  template sentences, writing no free text, so V1–V3 and V5 still cover everything it emits.
  It would need its own pre-registered validation.
- **A raw-path check for template text:** an independent second check on the template blocks
  that V1 alone now covers (above), and a sign check for values displayed as zero.
