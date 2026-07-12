# GnuCOBOL compiler-derived baselines

This directory contains compiler-derived evidence for the six
`scirust-finmigrate` COBOL migration units.

Compiler:

- GnuCOBOL 3.1.2

Validated units:

- INTACCR
- AMORTSCH
- PAYCALC
- DAYCOUNT
- BRKTCALC
- CURRCVT

The committed semantic-model baselines and the GnuCOBOL executions are
numerically identical for every tested field directly emitted by the COBOL
programs.

The normalized sources are preserved because the original repository files
required syntax and source-format portability corrections before GnuCOBOL
could compile them:

- fixed-format source converted to free format;
- fixed-format comments converted to `*>`;
- `PIC S V9(n)` corrected to `PIC SV9(n)`;
- same-line `MOVE` statements in BRKTCALC separated.

The arithmetic and business algorithms were not changed.

`*-RUN.cbl` files are instrumented executable wrappers that add only input
and output operations around the corresponding arithmetic routines.

This evidence validates the tested scenarios with GnuCOBOL. It does not prove
equivalence with IBM Enterprise COBOL, z/OS compiler options, or an unavailable
original production environment.
