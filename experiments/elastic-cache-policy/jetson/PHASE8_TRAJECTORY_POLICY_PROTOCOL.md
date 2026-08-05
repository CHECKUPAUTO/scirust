# Phase 8 — Causal and sequential trajectory-safety policy

## Motivation

The frozen phase-7 guard failed independent GSM8K validation. Its local
per-decision saving remained positive, but two correct answers became wrong and
all three prediction divergences occurred after exactly three skips, before the
configured cap of four. The failure therefore cannot be repaired by merely
lowering the cap or tightening the previous attention-output threshold.

Phase 8 changes the scientific target. A cache skip is no longer considered safe
because its local attention output is close to a refreshed output. It is safe
only when an atomic single-skip intervention preserves the complete generation
trajectory under the strict development label below.

## Development data

The phase uses only the official GSM8K **train** split. The 120 GSM8K test
questions consumed by phases 4, 5, and 7 are evaluation history and are never
read during phase-8 collection, fitting, calibration, or internal holdout
assessment.

For every selected training prompt:

1. run an `always_refresh` baseline while enumerating up to four skip candidates;
2. for each candidate, run a separate deterministic branch that permits exactly
   that one skip and forces every other decision to refresh;
3. record the candidate features, baseline and branch predictions, response
   hashes, correctness, decision counts, latency, and normalized refresh cost.

Default collection:

- seed: `20260809`;
- prompts: `40`;
- warm-up prompts: `2`;
- maximum candidates per prompt: `4`;
- maximum measured generations: approximately `200`;
- model: `Dream-org/Dream-v0-Instruct-7B` in BF16;
- hardware: NVIDIA Jetson AGX Thor;
- split: GSM8K train only.

## Strict unsafe label

An atomic skip is labelled `strict_unsafe` when any of the following changes
relative to the always-refresh baseline:

- final response hash;
- normalized numeric prediction;
- correctness;
- number of generation decisions.

This deliberately treats stylistic response changes and trajectory-length
changes as unsafe during development. The policy may abstain frequently; safety
takes precedence over coverage.

## SciRust components

### `scirust-causal`

The collector produces paired environments:

- observational `always_refresh`;
- atomic `single_skip_intervention`.

A typed `CausalDataset` and Invariant Causal Prediction diagnostic examine which
runtime variables remain associated with the strict unsafe outcome across the
two environments. This diagnostic is hypothesis-generating and cannot override
the fail-closed safety gate.

### `scirust-sequential`

A two-state linear-chain CRF models the candidate sequence for each prompt:

- `safe`;
- `strict_unsafe`.

The model uses emission features from each cache decision plus transition
features between consecutive candidates. This captures candidate order and
history instead of treating every decision as an independent row.

### `scirust-gp`

An exact deterministic Matérn-5/2 Gaussian process predicts strict-unsafe risk
and posterior variance. Runtime eligibility uses an upper confidence bound:

```text
gp_mean + uncertainty_multiplier × gp_standard_deviation
```

A high posterior variance therefore forces abstention even when the mean risk is
small.

### `scirust-evo`

NSGA-II searches a Pareto front over three separate objectives:

1. minimize strict-unsafe candidates incorrectly permitted;
2. maximize captured positive refresh-cost saving among safe candidates;
3. maximize safe-candidate coverage.

A policy is selectable only when it permits zero strict-unsafe candidates on the
validation prompt split. If no such policy exists, the serialized result is a
finite `deny_all` rule and the runtime action remains always refresh.

### `scirust-symreg`

Symbolic regression searches an interpretable surrogate for the strict unsafe
indicator. It is reported as a diagnostic Pareto front and does not bypass the
CRF, GP uncertainty, or zero-false-safe constraints.

## Prompt-level development split

The candidate corpus is separated deterministically by prompt identity:

- 60% fitting;
- 20% validation and NSGA-II selection;
- 20% internal development holdout.

No candidate from one prompt may appear in more than one partition.

## Fail-closed development criteria

`fail_closed_development_success` is true only when all conditions hold:

1. zero strict-unsafe skips permitted on validation;
2. zero strict-unsafe skips permitted on the internal GSM8K-train holdout;
3. zero quality regressions permitted on both held-out development partitions;
4. at least one skip permitted on validation and holdout;
5. internal holdout coverage at least `2%`;
6. positive net refresh-cost saving on the internal holdout.

A false result is scientifically valid. It means the evidence does not support
any cache reuse rule and always-refresh remains mandatory.

## Execution

```bash
bash experiments/elastic-cache-policy/jetson/run_trajectory_branch_development.sh
```

The runner performs both stages:

1. GPU collection of atomic trajectory branches;
2. offline Rust discovery with causal ICP, CRF, GP uncertainty, NSGA-II, and
   symbolic regression.

For an already collected JSONL corpus, the offline stage can be repeated without
GPU work:

```bash
bash experiments/elastic-cache-policy/jetson/finalize_trajectory_policy_discovery.sh
```

## Outputs

The timestamped result directory contains:

- `trajectory_development_manifest.json`;
- `dream_single_skip_trajectory_candidates.jsonl`;
- `dream_single_skip_trajectory_report.json`;
- `dream_trajectory_policy_development_report.json`.

The final report includes the source SHA-256, prompt splits, label prevalence,
feature standardization, causal diagnostic, CRF weights, GP configuration,
NSGA-II parameters and metrics, symbolic candidates, and the fail-closed verdict.

## Evidence boundary

This phase is development, not independent confirmation. A positive internal
result would authorize only freezing a candidate policy. It would not establish
task-level quality or performance. Any later confirmation must use a new,
untouched task dataset and a pre-registered protocol. A negative result must not
be converted into a positive claim by lowering the safety criteria after seeing
the holdout outcomes.
