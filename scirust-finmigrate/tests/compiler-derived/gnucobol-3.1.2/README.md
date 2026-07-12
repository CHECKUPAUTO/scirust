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
programs, with **one documented exception**: for CURRCVT's three
0-decimal-currency targets (ITL/ESP), the COBOL wrapper stores its result in a
fixed 2-dp field and so diverges from the model, which rounds to the target
currency's minor unit (audit Gap-R). These baselines record the **raw GnuCOBOL
value**; see `RESULTS.md` and `REPRODUCE.md`.

## Reproducing this evidence

Everything here is regenerated from scratch by a deterministic, standard-library
driver:

```sh
python3 tools/run_baselines.py generate   # compile + run -> baselines/ + compiler-logs/
python3 tools/run_baselines.py verify     # committed CSVs == live GnuCOBOL (0 mismatches)
python3 tools/run_baselines.py check      # vs model: 0 unexpected + 3 documented Gap-R rows
sha256sum -c SHA256SUMS                    # integrity manifest
```

See `REPRODUCE.md` for the exact toolchain, compile/run commands, and the manifest
regeneration step.

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
