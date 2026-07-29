# Continual Self-Monitored Industrial AI Program

Tracking document for the program that turns the deployment stage of SciRust's
structural-intelligence line into a *loop*. Program 4
([SRCC Robust Structural Intelligence](SRCC_ROBUST_STRUCTURAL_INTELLIGENCE_PROGRAM.md))
closed with two governance primitives in phase 4E.10: a pre-deployment gate
(`decide_deployment`, consuming a `CertifiedPipelineReport` from the 4E.7
certified pipeline) and a live rollback latch (`RollbackMonitor`). Program 5
([Causal and Experimental Structural Intelligence](CAUSAL_EXPERIMENTAL_STRUCTURAL_INTELLIGENCE_PROGRAM.md))
built the orthogonal causal-claims discipline. What neither owns is the loop
those primitives imply: a service that notices its own degradation, demands a
re-evaluation, promotes only on a fresh certificate, and falls back — then
repeats, indefinitely, with every step on the record.

The direction, as set when this program opened: *the 4E.10 gate-plus-rollback
loop is the skeleton of a safe continual-learning system — periodic
retraining, promotion only on certificate, automatic fallback. What is missing
is the orchestration and the streaming monitoring, not the foundations.* This
program builds exactly that missing layer, and nothing statistical: every
piece of statistical judgement stays where Programs 3–4 put it.

This file is maintained incrementally, in the same spirit as its
predecessors: each phase appends its design summary, merge commit, and known
limitations. It modifies neither Program 4's nor Program 5's closing
syntheses — those programs are finished; this one composes what they left.

## The mandate this program exists to enforce

**The orchestration layer must never manufacture statistical authority.**
Every entry of a model into service is a `DeploymentDecision` produced by the
existing certified pipeline (4E.7) and deployment gate (4E.10), on evidence
supplied at a stated tick. The orchestrator decides *when* to demand that
evaluation and *what to do* with its outcome; the judgement stays with the
certificate. Concretely:

- No code in this program retrains, evaluates, or scores a model.
- No code path promotes a model without a certified `DeployChallenger`
  decision, and no such decision is accepted unless the orchestrator itself
  demanded an evaluation (`UnsolicitedCandidate` is a typed error, even when
  the unsolicited certificate is flawless).
- `Serving` means "no monitor objects" — never "the model is correct".

## Program-wide invariants

Inherited from Programs 4 and 5, re-checked in each PR's self-review, with one
addition specific to orchestration:

- **Pure Rust.** No FFI, no network access in library code or tests;
  `#![forbid(unsafe_code)]` in every crate this program touches.
- **Deterministic — and now replayable.** Orchestration code runs on
  **logical time**: ticks are caller-supplied `u64`s, and there is no wall
  clock, no thread, no I/O and no RNG anywhere in the loop. The same event
  sequence therefore reproduces byte-identical state, status and audit log.
  "Real time" is deliberately the caller's business: feed events as they
  happen and the loop sequences the decisions.
- **Typed errors** with manual `Display` + `Error` impls, never a
  stringly-typed catch-all; invalid inputs leave state untouched.
- **Backward-compatible by default.** The 4E primitives are composed, not
  modified; existing public APIs stay source-compatible.
- **MSRV 1.89.**
- **Safe to abstain.** A hold, a block, a demand that stays open, and an
  explicit `Degraded` phase are ordinary, tested outcomes — never swallowed
  into a false "healthy".

## Honesty rules

1. **Serving is not a correctness claim.** A `Serving` phase coexists with an
   alarmed drift monitor (tested); it asserts only that no monitor has
   objected yet.
2. **An alarm never says why.** The drift monitor reports that the recent
   residual scale sits above its calibrated reference by the configured
   factor — drift, sensor fault, regime change and workload mix all look
   alike at that distance. Attribution belongs to the certified evaluation
   the alarm demands, not to the monitor.
3. **A hold does not certify recovery.** While rolled back or degraded, a
   `HoldIncumbent` or `BlockDeployment` records the attempt and keeps the
   demand open; only a certified promotion exits an unhealthy phase.
4. **A failed model is retired, not recycled.** A model that latched a
   rollback never becomes the fallback — it must not sit one crash away from
   serving again.
5. **Degradation is said out loud.** A rollback latch with no fallback
   available moves the service to an explicit `Degraded` phase that keeps
   serving the failed model *and says so*, rather than pretending a safe
   option exists.
6. **Fresh monitors mean a stated blind spot.** Promotion and rollback reset
   the live monitors, because carrying a failed model's evidence into its
   successor's record would be stale authority. The price — a warm-up window
   in which the monitors are not yet actionable — is deliberate and
   documented, not hidden.

## Protocol

- Each phase is a separate PR, branched from a **newly-merged** `master`. A
  later phase never starts from an unmerged earlier one.
- No PR auto-merges without explicit authorization and green CI.
- Every phase ships a deterministic benchmark example (run-twice plus release
  SHA-256 fingerprint) and reverifies every prior fingerprint in the
  workspace's registry unchanged.
- If `master` has advanced past what a phase assumed, or a planned mechanism
  would overclaim (e.g. an alarm that pretends to attribute cause), report
  and adjust scope rather than silently weakening the design.

## Phase 6.1 — Streaming drift monitoring and the governed continual loop

**Status: Done.** Branch `claude/scirust-srcc-robust-stats-6ue9xc`, restarted
from `origin/master` at `7468b232`. Draft PR opened from this branch; the
merge commit is recorded by the next phase, per the incremental discipline.

### What was missing, precisely

Phase 4E.10 ended with `decide_deployment` (pre-deployment: certificate in,
action out) and `RollbackMonitor` (live: rolling-window coverage with a
one-way latch), and its tracker section closed on the observation that the
reset after a latch is a human decision. Three things were left unowned:

1. **Nothing watches the residual scale.** Coverage is the *late* signal — by
   the time intervals stop covering, the damage is realized. The scale of
   recent residuals typically moves first.
2. **Nothing owns the demand.** Who decides a retrain evaluation is due —
   on a schedule, on an alarm, on a latch — and what stops a well-meaning
   caller from promoting a model nobody asked for?
3. **Nothing owns the aftermath.** What serves after a latch, what happens
   when the fallback fails too, and what, exactly, may claim the system is
   healthy again?

Phase 6.1 adds two modules to `scirust-srcc-bench` answering exactly these,
and nothing more.

### `drift_monitor` — the early warning

`DriftMonitor` watches one non-negative **residual score** per observation
(typically `|y − ŷ|`, or any monotone proxy the caller trusts) and compares
the rolling **median** of the last `window` scores against a **frozen**
`reference_scale` fixed at calibration time. Design decisions worth stating:

- **The reference is never re-estimated.** A reference that adapted to the
  stream would eventually absorb the very drift it exists to expose.
- **Median, not mean** — the robust choice this line has used since phase
  721: a minority of wild scores cannot fake or mask a drift signal on their
  own. Sorting is by `f64::total_cmp`; even windows use `f64::midpoint`.
- **Hysteresis in both directions, no latch.** A breach is
  `window median ≥ alarm_ratio × reference_scale` (evaluated once
  `minimum_samples` is met); `Quiet → Alarmed` takes
  `consecutive_breaches_to_alarm` breaches in a row, `Alarmed → Quiet` takes
  `consecutive_normals_to_clear` non-breaches in a row. One odd window cannot
  flip the state in either direction.
- **Deliberately unlatched, unlike the rollback monitor.** The two monitors
  encode two kinds of signal: a coverage collapse is a *safety event* with no
  honest way back short of a new certificate (latch); a scale excursion is an
  *early warning* that can subside (hysteresis). Collapsing that distinction
  either turns every wobble into a rollback or lets a real collapse clear
  itself — both wrong.
- Invalid scores (`NaN`, `∞`, negative) are typed errors that leave the
  monitor untouched.

### `continual` — the orchestrator

`ContinualOrchestrator` is a deterministic state machine over logical time
composing the drift monitor with the 4E.10 primitives. Its full semantics are
in the module documentation; the load-bearing rules:

- **Demands are ranked** `CoverageRollback > DriftAlarm > Scheduled`. A
  higher-ranked reason upgrades an already-open demand; a lower-ranked one
  never downgrades. Scheduled demands respect the cadence (anchored at start,
  promotion, or demand close); drift and scheduled demands respect a cooldown
  after the previous close; a coverage rollback respects neither — it is a
  safety event.
- **The latch edge executes the rollback.** With a fallback: the failed model
  is retired, the fallback serves under fresh monitors, the phase becomes
  `RolledBack`, and the demand stays open — serving the fallback is a
  mitigation, not a recovery. Without one: the phase becomes `Degraded`, the
  failed model keeps serving because nothing safer exists, and the monitors
  stay latched. If the fallback's own fresh monitor latches again, the system
  cascades to `Degraded` — by design; drift that broke the champion may break
  the fallback too.
- **Only a certified promotion exits `RolledBack` or `Degraded`.** This is
  deliberately stricter than the bare 4E.10 monitor, which leaves the reset
  to a human: here the exit criterion is explicit and machine-checkable — a
  new certificate. Promotion out of `Degraded` retires the failed model (the
  fallback slot stays empty); promotion out of `Serving`/`RolledBack` keeps
  the previous server as fallback.
- **The demand discipline is enforced, not advised.** A candidate evaluation
  with no open demand is `UnsolicitedCandidate` — a typed error — even when
  the decision it carries is a flawless certified promotion.
- **Everything is on the record.** Every transition (demand opened/upgraded,
  demand closed, promotion, rollback, degradation) appends a
  `TransitionRecord { tick, phase_before, phase_after, reason }` to an audit
  log that replays byte-identically.

Ticks may repeat (bursts are legitimate) but never regress; a regression is a
typed error that leaves state untouched, as does an invalid drift score
(validated before any state mutates).

### Composition with the real pipeline

The integration tests do not hand-build decisions: they run the actual
4E.7 `run_certified_pipeline` → 4E.10 `decide_deployment` chain (deterministic
tournament, seed `0x00D0_9111`) and feed its output to the orchestrator — a
certified tie holds and closes the demand, a certified winner promotes, and
the same fixture demonstrates the full arc: scheduled demand → certified
promotion → drift alarm *while window coverage still reads 1.0* → coverage
latch → rollback that retires the failure → certified re-promotion. Unit
tests cover each rule in isolation (13 for the orchestrator, 8 for the drift
monitor); the workspace suite for the crate stands at 152 library tests plus
6 lifecycle integration tests, all passing.

### Deterministic benchmark

`continual_lifecycle_benchmark` walks one full life of a governed service
through nine oracle-checked scenarios, on real certified decisions:

| # | Scenario | Oracle highlights |
|---|---|---|
| 1 | `scheduled_demand_opens` | cadence 200 opens the demand at exactly tick 200 |
| 2 | `certified_hold_closes_demand` | a real certified tie holds and closes |
| 3 | `certified_promotion` | v2 promoted at tick 412, v1 retained as fallback |
| 4 | `drift_alarm_before_damage` | alarm at tick 435 with window coverage still 1.0 |
| 5 | `recovery_attempt_held` | a tie while `Serving` closes the demand |
| 6 | `coverage_latch_rolls_back` | latch at tick 440: v1 serves, v2 retired, demand `CoverageRollback` |
| 7 | `hold_cannot_exit_rollback` | hold keeps the demand open |
| 8 | `certified_promotion_recovers` | only the certified v3 exits the rollback |
| 9 | `healthy_again` | `Serving`, v3, fallback v1, coverage 1.0, drift `Quiet` |

Scenario 4 is the phase's thesis in one line: the drift demand opens while
live coverage still reads 100% — the early warning arrives before the damage,
which is the entire point of watching residual scale separately from
coverage.

Fingerprint (run twice in debug, once in release; all three byte-identical):

    2bac7e099ca7143a95c383ef4aab1b743af32999bc53aa3f6db0e0fccab9caa7

### Prior fingerprints reverified

All ten standing determinism witnesses rerun unchanged at this phase:

| Example | Fingerprint |
|---|---|
| `industrial_protocol_demo` (Program 3/4, longest-standing) | `167c13de…` |
| `conditional_independence_benchmark` | `c1449177…` |
| `pc_stable_benchmark` | `79e57e69…` |
| `effect_estimation_benchmark` | `7ac0dc76…` |
| `sensitivity_benchmark` | `1bc59a1d…` |
| `invariance_benchmark` | `e1f0b99f…` |
| `counterfactual_benchmark` | `f34e1cfa…` |
| `experiment_design_benchmark` | `c368798e…` |
| `theory_revision_benchmark` | `51413a06…` |
| `claim_audit_benchmark` | `9d7e6d55…` |

### Known limitations

- **Logical time only.** The orchestrator provides replayability, not
  real-time guarantees: no wall-clock scheduling, no deadlines, no
  concurrency story. Wiring ticks to a clock, a stream, or a job queue is the
  caller's integration work, outside this crate's determinism boundary.
- **The loop demands retraining; it does not perform it.** What "run the
  certified pipeline on fresh evidence" means operationally — data
  collection, training, candidate construction — is upstream of
  `candidate_evaluated` and out of scope by mandate.
- **One scalar drift score, chosen by the caller.** The monitor sees whatever
  residual proxy it is fed; a proxy blind to the failure mode leaves the
  alarm blind too. Multivariate or per-group drift monitoring is future work.
- **No cause attribution, anywhere.** Rule 2 above; nothing in this phase
  distinguishes drift from sensor fault from workload change.
- **Warm-up blindness is real.** After every promotion or rollback the fresh
  monitors need `minimum_samples` observations before they are actionable; a
  model that fails instantly is caught only after the warm-up (honesty rule
  6). Shrinking that window tightens the trade against false latches; the
  policy owns the choice.
- **Model names are labels.** The orchestrator tracks identity, not
  artifacts; storage, versioning and registry integration are deliberately
  absent.
