# Phase 7 — Frozen guarded policy on independent GSM8K prompts

## Protocol

- model: `Dream-org/Dream-v0-Instruct-7B` in BF16;
- hardware: NVIDIA Jetson AGX Thor;
- frozen five-policy ensemble, vote threshold `3/5`;
- frozen trajectory-stability guard selected without GSM8K outcomes;
- guard parameters:
  - minimum skip margin: `0.000017300677036841128`;
  - minimum normalized refresh cost: `27/28`;
  - maximum skips per attention layer: `4`;
  - cooldown decisions: `0`;
- 60 GSM8K test questions selected with seed `20260808`;
- all evaluation indices disjoint from the 60 phase-4/5 indices;
- two repeats per mode and question;
- counterbalanced `ABBA/BAAB` execution;
- same generation seed within each question;
- 10,000 paired bootstrap samples;
- no policy or guard fitting on the evaluation prompts.

Primary report SHA-256:

```text
e36c72942805c8b19321bf6f66a0fb4bafc77d74e1dc76a67aef513d02780049
```

## Pre-registered criteria

Success required all of the following:

1. exact deterministic reproducibility within each mode;
2. exact-match non-inferiority within five percentage points, including the
   lower 95% bootstrap bound;
3. mean wall-clock improvement of at least `0.5%`, with a lower 95% bound above
   zero;
4. mean total refresh-cost improvement of at least `0.5%`, with a lower 95%
   bound above zero.

## Result

- within-mode determinism: **pass** (`60/60`, zero mismatches);
- mean accuracy delta: **-3.3333 percentage points**;
- accuracy-delta 95% interval: **[-8.3333, 0] percentage points**;
- same prediction rate: `95%`;
- exact response match rate: `63.3333%`;
- quality non-inferiority: **fail**;
- mean wall-clock improvement: **0.5690%**;
- wall-clock 95% interval: **[-1.1705%, 2.2961%]**;
- wall-clock criterion: **fail**;
- mean total refresh-cost improvement: **0.8266%**;
- refresh-cost 95% interval: **[-0.9422%, 2.5110%]**;
- refresh-cost criterion: **fail**;
- `independent_guard_validation_success`: **false**.

Secondary mechanism metrics:

- conditional refresh-cost saving: **6.5477%**, 95% interval
  `[6.0384%, 7.1639%]`;
- refresh-cost-per-decision improvement: **6.7718%**, 95% interval
  `[6.2492%, 7.4046%]`;
- latency-per-decision improvement: **6.5210%**, 95% interval
  `[5.9433%, 7.1805%]`;
- guarded/always decision-count ratio: **1.06435**, 95% interval
  `[1.04521, 1.08396]`.

## Scientific conclusion

The frozen guard improves work per cache decision, but it does not establish
end-to-end quality non-inferiority or statistically supported latency and total
refresh-cost gains. Approximately `6.5%` to `6.8%` less work per decision is
again offset by approximately `6.4%` more generation decisions.

The quality result is especially important: the point estimate is two net fewer
correct answers out of 60, and the lower bootstrap bound exceeds the registered
five-point non-inferiority margin in the adverse direction. This does not prove
a statistically significant quality degradation versus zero, because the
interval includes zero. It does mean non-inferiority was not demonstrated.

The local attention-output divergence metric is therefore insufficient as the
sole task-safety objective for this Dream decoding configuration. Future policy
design must model generation-trajectory stability directly, such as token reveal
order, unresolved-token count, stopping behavior, or final-logit invariance.

## Evidence boundary

This is an independent task-level negative result on one model, one decoding
configuration, one hardware platform, and 60 GSM8K prompts. The policy and guard
must not be retuned using these prompts while continuing to describe a later run
as independent confirmation. These prompts are now evaluation history.
