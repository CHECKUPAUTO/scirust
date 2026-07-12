# SciRust finmigrate — GnuCOBOL compiler-derived validation

## Environment

- Compiler: GnuCOBOL 3.1.2
- Source revision recorded in `metadata/source-commit.txt`
- System information recorded in `metadata/system.txt`

## Source normalization required for GnuCOBOL

The repository sources' **arithmetic and business algorithms were not changed by
normalization.** Temporary executable copies were produced by:

1. Converting fixed-format source layout to free-format layout.
2. Converting fixed-format comments to `*>`.
3. Replacing invalid `PIC S V9(n)` declarations with `PIC SV9(n)`.
4. Splitting the two same-line `MOVE` statements in `BRKTCALC.cbl`.

These are source-format and syntax-portability corrections only.

One deliberate **algorithm** change is recorded separately from normalization:
the CURRCVT Gap-R reconciliation (2026-07-12), applied identically to the
canonical `cobol/CURRCVT.cbl`, the normalized source, and the `-RUN` wrapper —
see "CURRCVT Gap-R reconciliation" below. It is not a normalization artifact; it
is a documented migration decision.

## Compiler-derived results

The **Model-vs-compiler differences** column reports comparison **(B)** below
(committed baselines vs the semantic model). Re-running GnuCOBOL against the
committed baselines (comparison (A)) also yields 0 mismatches for every unit.

| Unit | Executed cases/rows | Compared fields | Model-vs-compiler differences |
|---|---:|---|---:|
| INTACCR | 75 scenarios | principal, rate, rounded interest, truncated interest, new balance | 0 |
| AMORTSCH | 94 schedule rows | period, interest, principal, payment, balance | 0 |
| PAYCALC | 8 scenarios | principal, rate, periods, factor, payment | 0 |
| DAYCOUNT | 10 scenarios | NASD days, interest | 0 |
| BRKTCALC | 9 scenarios | base, marginal tax | 0 |
| CURRCVT | 10 scenarios | amount, source, target, triangulated result, euro intermediate | 0 (Gap-R reconciled) |

## Two distinct comparisons — do not conflate them

This evidence set involves **two separate comparisons**, each answering a
different question. They use different reference data. Both now report zero
differences, but they measure different things.

**(A) Faithfulness / reproducibility — committed baselines vs live GnuCOBOL.**
`tools/run_baselines.py verify` recompiles and re-runs every `*-RUN.cbl` wrapper
and asserts that the committed `baselines/*.csv` reproduce, cell-for-cell, what a
fresh GnuCOBOL 3.1.2 run emits: **0 mismatches over 1179 field comparisons**.
This proves the CSVs are an honest transcript of the compiler; it says nothing on
its own about the semantic model.

**(B) Equivalence — committed baselines vs the semantic model.**
`tools/run_baselines.py check` compares the same committed `baselines/*.csv`
against the independent semantic-model baselines under `../../sandbox`:
**0 unexpected mismatches + 0 documented divergences**. Prior to the Gap-R
reconciliation this check reported 3 documented CURRCVT divergences; the wrapper
now implements the target minor unit, so COBOL and model agree on every emitted
field.

In short: **(A)** measures the baselines against the compiler (reproduction
fidelity); **(B)** measures the semantic model against GnuCOBOL (real
COBOL-vs-model equivalence). Keeping them separate is what let the earlier Gap-R
divergence be seen at all — (A) stayed at 0 (the baselines faithfully recorded
whatever the wrapper emitted) while (B) surfaced the 3-cell model/COBOL gap. That
gap is now closed in the COBOL, so both are 0.

## Derived audit columns not emitted by the COBOL programs

The following model-baseline columns were intentionally excluded from direct
compiler comparison because they are audit calculations, not COBOL outputs:

- DAYCOUNT: `excel_days`
- BRKTCALC: `flat_tax`, `effective_pct`
- CURRCVT: `direct`

## CURRCVT Gap-R reconciliation (historical trace)

**What Gap-R is.** The final conversion result must be rounded to the *target*
currency's minor unit — 2 dp for DEM/FRF/IEP but **0 dp for ITL/ESP** (audit
`SEMANTICS_CURR.md`, Gap-R; EC Regulation 1103/97). The semantic model and the
Rust port always implemented this.

**The divergence the compiler evidence exposed.** The original COBOL wrapper
stored its result in a fixed 2-decimal field (`WS-RESULT PIC S9(11)V99`) and did
**not** implement the target minor unit. For the three 0-decimal-currency targets
the raw GnuCOBOL result therefore carried two decimals and diverged from the model
by a minor unit:

| Scenario   | Old wrapper (fixed 2-dp) | Model / Rust (target minor unit) | Reconciled wrapper |
|------------|-------------------------:|---------------------------------:|-------------------:|
| frf_to_itl | 295182.43                | 295182                           | 295182             |
| dem_to_esp | 21267.96                 | 21268                            | 21268              |
| esp_to_itl | 581860.75                | 581861                           | 581861             |

**The fix (2026-07-12).** `CURRCVT-RUN.cbl` (and the canonical `cobol/CURRCVT.cbl`
and its normalized copy) now carry the target minor unit as `WS-MINOR-UNIT`
(0 for ITL/ESP, 2 otherwise — the currency-master value) and round into a result
field of the matching scale: `WS-RESULT-0 PIC S9(13)` for 0-dp targets,
`WS-RESULT-2 PIC S9(11)V99` otherwise. The `-RUN` wrapper ACCEPTs `WS-MINOR-UNIT`
after the two rates and DISPLAYs the correctly-scaled field. This mirrors the
model's per-target `minor(to)` lookup exactly.

The committed `CURRCVT-compiler-baseline.csv` now records those three cells as
`295182 / 21268 / 581861` — the raw output of the reconciled wrapper, which equals
the model. `tools/run_baselines.py check` keeps an explicit **empty** Gap-R
exception set, so any regression would surface as an *unexpected* mismatch rather
than a silent pass. The euro intermediate and the seven same-scale rows were
unchanged by the fix.

This is a regulation-compliant correction of a genuine legacy defect, made as a
recorded migration decision (audit trail 2026-07-12), not a silent edit; the
historical values are preserved in the table above and in `audit_trail.log`.

## Reproducibility

The baselines and compiler logs are regenerated from scratch by the deterministic,
standard-library driver `tools/run_baselines.py` (`generate` / `verify` /
`check`). The exact toolchain, compile flags, run commands, normalization
procedure, and manifest regeneration are documented in `REPRODUCE.md`. `verify`
is comparison **(A)** above (committed CSVs equal live GnuCOBOL output — 0
mismatches over 1179 field comparisons); `check` is comparison **(B)** (baselines
vs the semantic model — 0 unexpected, 0 divergences). The integrity manifest
`SHA256SUMS` covers every committed file including the per-program
`compiler-logs/`.

## Representation-only difference

INTACCR produced numeric zero for the negative truncated-zero case, while the
model CSV serialized it as `-0.00`. Both parse to the same decimal value.

The model CSV also uses CRLF line endings while generated compiler baselines
use LF. These are representation differences, not numerical differences.

## Conclusion

All six finmigrate units have now been executed through a real GnuCOBOL
compiler on every committed scenario.

For every field actually emitted by the COBOL workload, the compiler-derived
results are numerically identical to the committed semantic-model baselines,
including the three CURRCVT 0-decimal-currency `result` cells that were reconciled
on 2026-07-12 (the wrapper now implements Gap-R). Both the reproduction check (A)
and the model-equivalence check (B) report zero differences.

This upgrades all six units (INTACCR, AMORTSCH, PAYCALC, DAYCOUNT, BRKTCALC,
CURRCVT) from model-only validation to GnuCOBOL-validated semantic equivalence for
the tested datasets, and records the one place (CURRCVT Gap-R) where the committed
COBOL initially diverged from the model and how it was closed.

It does not prove equivalence with IBM Enterprise COBOL, z/OS runtime options,
or any unavailable original production environment.
