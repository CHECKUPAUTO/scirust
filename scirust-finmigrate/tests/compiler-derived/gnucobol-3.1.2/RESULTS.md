# SciRust finmigrate — GnuCOBOL compiler-derived validation

## Environment

- Compiler: GnuCOBOL 3.1.2
- Source revision recorded in `metadata/source-commit.txt`
- System information recorded in `metadata/system.txt`

## Source normalization required for GnuCOBOL

The repository sources were not modified.

Temporary executable copies were produced by:

1. Converting fixed-format source layout to free-format layout.
2. Converting fixed-format comments to `*>`.
3. Replacing invalid `PIC S V9(n)` declarations with `PIC SV9(n)`.
4. Splitting the two same-line `MOVE` statements in `BRKTCALC.cbl`.

These are source-format and syntax-portability corrections. The arithmetic
statements and business algorithms were not changed.

## Compiler-derived results

The **Numerical differences** column reports comparison **(B)** below —
committed baselines vs the semantic model. It is *not* the reproduction check:
re-running GnuCOBOL against the committed baselines yields 0 mismatches for every
unit (comparison (A)). See "Two distinct comparisons" immediately below.

| Unit | Executed cases/rows | Compared fields | Model-vs-compiler differences |
|---|---:|---|---:|
| INTACCR | 75 scenarios | principal, rate, rounded interest, truncated interest, new balance | 0 |
| AMORTSCH | 94 schedule rows | period, interest, principal, payment, balance | 0 |
| PAYCALC | 8 scenarios | principal, rate, periods, factor, payment | 0 |
| DAYCOUNT | 10 scenarios | NASD days, interest | 0 |
| BRKTCALC | 9 scenarios | base, marginal tax | 0 |
| CURRCVT | 10 scenarios | amount, source, target, triangulated result, euro intermediate | 3 (documented, Gap-R) |

## Two distinct comparisons — do not conflate them

This evidence set involves **two separate comparisons**, each answering a
different question. They use different reference data and therefore report
different numbers. Both statements below are simultaneously true.

**(A) Faithfulness / reproducibility — committed baselines vs live GnuCOBOL.**
`tools/run_baselines.py verify` recompiles and re-runs every `*-RUN.cbl` wrapper
and asserts that the committed `baselines/*.csv` reproduce, cell-for-cell, what a
fresh GnuCOBOL 3.1.2 run emits: **0 mismatches over 1179 field comparisons**.
There are 0 mismatches *because the committed baselines record raw GnuCOBOL
output* — including the three CURRCVT 0-decimal cells, which were corrected to
their raw compiler values (295182.43 / 21267.96 / 581860.75). This comparison
proves the CSVs are an honest transcript of the compiler; it says nothing about
the semantic model.

**(B) Equivalence — committed baselines vs the semantic model.**
`tools/run_baselines.py check` compares the same committed `baselines/*.csv`
against the independent semantic-model baselines under `../../sandbox`:
**0 unexpected mismatches + 3 documented divergences**. The three divergences are
exactly the CURRCVT 0-decimal-currency `result` cells (audit Gap-R, detailed
below), where the committed COBOL wrapper's fixed 2-dp result field differs from
the model's target-minor-unit rounding.

In short: **0 mismatches in (A)** measures the baselines against the compiler
(reproduction fidelity); **3 divergences in (B)** measure the initial semantic
model against GnuCOBOL (a real COBOL-vs-model scale difference). The same three
`result` cells that are "correct" in (A) — because they faithfully record raw
GnuCOBOL — are the three "divergences" flagged in (B) against the model. The
"Numerical differences" column in the results table above reports comparison
**(B)** (model vs compiler); comparison (A) has no mismatches by construction.

## Derived audit columns not emitted by the COBOL programs

The following model-baseline columns were intentionally excluded from direct
compiler comparison because they are audit calculations, not COBOL outputs:

- DAYCOUNT: `excel_days`
- BRKTCALC: `flat_tax`, `effective_pct`
- CURRCVT: `direct`

## Documented numerical divergence — CURRCVT 0-decimal currencies (audit Gap-R)

`CURRCVT-RUN.cbl` stores its converted result in a fixed 2-decimal field
(`WS-RESULT PIC S9(11)V99`) and does **not** implement the target-currency minor
unit. For the three scenarios whose target is a 0-decimal currency (ITL/ESP), the
raw GnuCOBOL result therefore carries two decimals and differs from the
model/Rust baseline, which rounds to the target minor unit (audit Gap-R):

| Scenario   | Raw GnuCOBOL `result` | Model / Rust (0-dp) |
|------------|----------------------:|--------------------:|
| frf_to_itl | 295182.43             | 295182              |
| dem_to_esp | 21267.96              | 21268               |
| esp_to_itl | 581860.75             | 581861              |

The committed `CURRCVT-compiler-baseline.csv` records the **raw GnuCOBOL value**
(faithful to the committed wrapper), not the model value. This is a genuine
COBOL-vs-model scale divergence, not a representation artifact: it means the
committed COBOL wrapper does not yet implement Gap-R for 0-decimal currencies. The
model and the Rust port do; the euro intermediate and all same-scale (2-dp target)
rows are identical. `tools/run_baselines.py check` treats exactly these three
`result` cells as expected divergences and fails on any other mismatch.

## Reproducibility

The baselines and compiler logs are regenerated from scratch by the deterministic,
standard-library driver `tools/run_baselines.py` (`generate` / `verify` /
`check`). The exact toolchain, compile flags, run commands, normalization
procedure, and manifest regeneration are documented in `REPRODUCE.md`. `verify`
is comparison **(A)** above: it confirms the committed CSVs equal live GnuCOBOL
output (0 mismatches over 1179 field comparisons) — 0 precisely because the
baselines record raw GnuCOBOL, including the 3 corrected CURRCVT cells. `check`
is comparison **(B)**: baselines vs the semantic model, 0 unexpected + 3
documented Gap-R divergences. The integrity manifest `SHA256SUMS` covers every
committed file including the per-program `compiler-logs/`.

## Representation-only difference

INTACCR produced numeric zero for the negative truncated-zero case, while the
model CSV serialized it as `-0.00`. Both parse to the same decimal value.

The model CSV also uses CRLF line endings while generated compiler baselines
use LF. These are representation differences, not numerical differences.

## Conclusion

All six finmigrate units have now been executed through a real GnuCOBOL
compiler on every committed scenario.

For every field actually emitted by the COBOL workload, the compiler-derived
results are numerically identical to the committed semantic-model baselines —
**except** the three CURRCVT 0-decimal-currency `result` cells documented above,
where the committed COBOL wrapper's fixed 2-dp result field diverges from the
model's target-minor-unit rounding (Gap-R). Those baselines record the raw
GnuCOBOL value.

This evidence upgrades five units (INTACCR, AMORTSCH, PAYCALC, DAYCOUNT, BRKTCALC)
and the seven same-scale CURRCVT rows from model-only validation to
GnuCOBOL-validated semantic equivalence for the tested datasets, and precisely
localizes the one place (CURRCVT Gap-R) where the committed COBOL wrapper and the
model still differ.

It does not prove equivalence with IBM Enterprise COBOL, z/OS runtime options,
or any unavailable original production environment.
