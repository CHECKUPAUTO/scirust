# TSA → ATRA → CANR → ANEE — program archive

**Status: ARCHIVED**, 2026-07-18, by principal decision, after Phase D
returned every remaining avenue with a verdict and the program's last open
novelty candidate was closed (negative, empirically, by its own
pre-registered criterion). The program is **closed with zero open claims**.
This file is the index and the seal; it adds no new results.

Scope note: this archive covers only the numerical-representation program
below. Other documents in `docs/research/` (HYPERCRYPTO, HYPERMEMORY, …)
belong to separate programs and are not covered here.

## Reading order

| # | Document | One-line fate | Delivered in |
|---|---|---|---|
| 1 | [`TSA_TRANSFORMED_SCALAR_ALGORITHMS_2026-07-16.md`](TSA_TRANSFORMED_SCALAR_ALGORITHMS_2026-07-16.md) | The conjugation principle is classical — ~24 fields own it; no new field | #627 |
| 2 | [`ATRA_ADAPTIVE_TRANSFORMED_REPRESENTATIONS_2026-07-16.md`](ATRA_ADAPTIVE_TRANSFORMED_REPRESENTATIONS_2026-07-16.md) | Unrestricted adaptivity is vacuous; every useful restriction is owned | #627 |
| 3 | [`CANR_CERTIFIED_ADAPTIVE_REPRESENTATIONS_2026-07-16.md`](CANR_CERTIFIED_ADAPTIVE_REPRESENTATIONS_2026-07-16.md) | One level-6 tool niche; the engineering shipped, the research program rejected | #627 |
| 4 | [`ANEE_ADAPTIVE_NUMERICAL_EXECUTION_ENGINE_2026-07-17.md`](ANEE_ADAPTIVE_NUMERICAL_EXECUTION_ENGINE_2026-07-17.md) | The 7-axis formulation is Rice 1976; the architecture is FFTW/ATLAS/SPIRAL/AutoTVM; composition is equality saturation. Phase C: kernel 1 wins 3/3, kernel 2 fails 1/3, the boundary heuristic validates 13/15 with a floor guard, the "graph" deflates to dictionary densification | #635, #653, #662 |
| 5 | [`ANEE_PROGRAM_SYNTHESIS_2026-07-18.md`](ANEE_PROGRAM_SYNTHESIS_2026-07-18.md) | The closing synthesis: what survives, what is closed, what remained | #666 |
| 6 | [`ANEE_PHASE_D_PREREGISTRATION_2026-07-18.md`](ANEE_PHASE_D_PREREGISTRATION_2026-07-18.md) | Phase D opened by principal decision; six criteria and priors locked before any run | #702 |
| 7 | [`ANEE_PHASE_D_RESULTS_2026-07-18.md`](ANEE_PHASE_D_RESULTS_2026-07-18.md) | D1 attack succeeds · D2 determinism cheap 15/15 · D3 bounds sound, coverage gap 6e13× · **D4 the last candidate dies** · D5 composition ~8× tighter than compounding · D6 transfer 13/15 — three author priors failed visibly and stand recorded | #702 |

Experiment sources: `tsa_experiments/`, `atra_experiments/`,
`canr_experiments/`, `anee_experiments/` (committed alongside their phases).

## What remains live in `master` (ordinary engineering, no research frame)

- `scirust-core::certified_numerics` — κ_rt certificates (measured in Phase
  D: median slack 4.3×, sound on all observed data; round-trip scope only).
- `scirust-core::representation_graph` — dictionary, composed
  representations (certify the composition, never multiply per-hop bounds —
  D5), joint/sequential search, `ablation_first_advice` with the
  **mandatory floor guard** (the program's single most load-bearing
  lesson), `current_hardware_key()`, and `PlanCache` — whose
  distribution-keying D1/D4 showed to be a liability as committed; kept as
  the honest negative exhibit its docs now are part of.
- `scirust-core::autotune_accumulate`, `transform_autotune`,
  `transform_search`, `compute_capability`, `representation_graph_quaternion`.
- `scirust-bench-schema` — the CANR §9 record as a type; adopters:
  `vst_bench`, the dose-response and Phase D binaries, `tdi-holdout`
  (with real CIs), plus the criterion converter.
- Reproducible benchmark binaries: `anee_phase_c_{prototype,kernel2_quaternion,dose_response,two_hop}`,
  `anee_phase_d_{cache,determinism_transfer,certificates}`.

## Reopening conditions (written; none currently met)

1. **Prior art surfaces** for distribution-keyed autotuning caches,
   falsifying ANEE §12.2's absence-of-evidence claim → record it in an
   addendum (historical honesty only — D4 already closed the candidate's
   practical value independently).
2. **A real workload class** where the guarded ablation-first rule
   systematically mispredicts → fix as engineering; only a structural
   failure of the guard itself would warrant more.
3. **A genuinely new candidate** — falsifiable, pre-registrable, and not
   owned anywhere on the program's ~25-system map → a new phase, under the
   same discipline: criteria committed before runs, priors on the record,
   negative results kept.

Anything else is maintenance. The program ends as it worked: closed by its
own criteria, with the record complete enough that every claim — including
the authors' failed predictions — can be checked against committed code.
