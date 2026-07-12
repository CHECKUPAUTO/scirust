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

| Unit | Executed cases/rows | Compared fields | Numerical differences |
|---|---:|---|---:|
| INTACCR | 75 scenarios | principal, rate, rounded interest, truncated interest, new balance | 0 |
| AMORTSCH | 94 schedule rows | period, interest, principal, payment, balance | 0 |
| PAYCALC | 8 scenarios | principal, rate, periods, factor, payment | 0 |
| DAYCOUNT | 10 scenarios | NASD days, interest | 0 |
| BRKTCALC | 9 scenarios | base, marginal tax | 0 |
| CURRCVT | 10 scenarios | amount, source, target, triangulated result, euro intermediate | 0 |

## Derived audit columns not emitted by the COBOL programs

The following model-baseline columns were intentionally excluded from direct
compiler comparison because they are audit calculations, not COBOL outputs:

- DAYCOUNT: `excel_days`
- BRKTCALC: `flat_tax`, `effective_pct`
- CURRCVT: `direct`

## Representation-only difference

INTACCR produced numeric zero for the negative truncated-zero case, while the
model CSV serialized it as `-0.00`. Both parse to the same decimal value.

The model CSV also uses CRLF line endings while generated compiler baselines
use LF. These are representation differences, not numerical differences.

## Conclusion

All six finmigrate units have now been executed through a real GnuCOBOL
compiler on every committed scenario.

For every field actually emitted by the COBOL workload, the compiler-derived
results are numerically identical to the committed semantic-model baselines.

This evidence upgrades the six units from model-only validation to
GnuCOBOL-validated semantic equivalence for the tested datasets.

It does not prove equivalence with IBM Enterprise COBOL, z/OS runtime options,
or any unavailable original production environment.
