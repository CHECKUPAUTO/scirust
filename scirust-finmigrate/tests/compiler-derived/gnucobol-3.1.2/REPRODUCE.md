# Reproducing the GnuCOBOL compiler-derived baselines

This document lets an independent reviewer regenerate every file under
`baselines/` and `compiler-logs/` from scratch and confirm the results, with no
manual editing.

## Toolchain

| Component | Version |
|-----------|---------|
| GnuCOBOL  | `cobc (GnuCOBOL) 3.1.2.0` (see `metadata/compiler-version.txt`) |
| Python    | 3.11 (standard library only; no third-party packages) |
| OS (original run) | aarch64 GNU/Linux — see `metadata/system.txt` |
| OS (independent re-run) | also reproduced bit-for-bit on `x86_64 GNU/Linux` |

GnuCOBOL 3.1.2 performs decimal (`COMP-3`) arithmetic via libcob/GMP, which is
platform-independent; the baselines were originally generated on the aarch64
system recorded in `metadata/system.txt` and independently reproduced on x86_64
with the same GnuCOBOL version, yielding identical results.

Install GnuCOBOL 3.1.2 (Debian/Ubuntu):

```sh
sudo apt-get install -y gnucobol        # provides cobc (GnuCOBOL) 3.1.2.x
cobc --version                          # must report 3.1.2.0
```

## Inputs

- **Programs:** `normalized-sources/<UNIT>-RUN.cbl` — instrumented, executable
  wrappers around `normalized-sources/<UNIT>.cbl`. They add only `ACCEPT`
  (stdin) and `DISPLAY` (stdout); the arithmetic is unchanged (see below).
- **Scenarios:** the committed model scenario files under
  `scirust-finmigrate/tests/sandbox/` — `dataset.csv` (INTACCR),
  `amort_scenarios.csv`, `pay_scenarios.csv`, `day_scenarios.csv`,
  `brkt_scenarios.csv`, `curr_scenarios.csv`. For CURRCVT the driver maps each
  ISO currency code to its fixed EC-1103/97 rate (DEM 1.95583, FRF 6.55957,
  ITL 1936.27, ESP 166.386, IEP 0.787564) and to its minor unit / decimal places
  (ITL 0, ESP 0, all others 2), which it passes as the wrapper's fourth input
  (Gap-R).

## Normalization procedure (already applied; recorded for provenance)

The repository sources in `scirust-finmigrate/cobol/` were **not** modified.
The committed `normalized-sources/*.cbl` are format/syntax-portability copies:

1. fixed-format layout → free format;
2. fixed-format comments → `*>`;
3. `PIC S V9(n)` → `PIC SV9(n)`;
4. the two same-line `MOVE` statements in `BRKTCALC.cbl` split onto separate lines.

These are whitespace/comment-level only. You can confirm no arithmetic changed:
stripping comments + all whitespace + upper-casing makes each normalized source
character-identical to its `scirust-finmigrate/cobol/<UNIT>.cbl` original.

## Compile command

```sh
# executable used to run the scenarios:
cobc -x -free -O2 -o <UNIT>-RUN normalized-sources/<UNIT>-RUN.cbl
# pure-cobc acceptance diagnostics captured as compiler-logs/<UNIT>.log:
cobc -fsyntax-only -free -Wall normalized-sources/<UNIT>-RUN.cbl
```

## Run command

Each wrapper reads its inputs from stdin (in `ACCEPT` order) and writes its
outputs to stdout (in `DISPLAY` order); raw `DISPLAY` is `±`-signed and
zero-padded, e.g. `+000000100.01`. The driver strips the sign/leading zeros to
the CSV form `100.01`, echoes the scenario input columns verbatim, and orders the
columns per the committed CSV header. Example (INTACCR, one scenario):

```sh
printf '100.00\n0.00060\n' | ./INTACCR-RUN
# +000000000.01  (monthly_int)  +000000000.00  (monthly_trunc)  +000000100.01  (new_balance)
```

## One-command regeneration

The whole pipeline is `tools/run_baselines.py` (stdlib only, deterministic):

```sh
cd scirust-finmigrate/tests/compiler-derived/gnucobol-3.1.2

# 1. compile + run every unit; (re)write baselines/*.csv and compiler-logs/*.log
python3 tools/run_baselines.py generate

# 2. recompile + run and assert the committed CSVs match live GnuCOBOL exactly
python3 tools/run_baselines.py verify         # -> 0 mismatches

# 3. no-compiler consistency check vs the model baselines in ../../sandbox
python3 tools/run_baselines.py check          # -> 0 unexpected, 0 divergences (Gap-R reconciled)

# 4. regenerate and verify the integrity manifest
LC_ALL=C; find . -type f ! -name SHA256SUMS ! -path './tools/_build/*' \
  | sort | xargs sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS                        # -> all OK
```

`generate` writes only into `baselines/`, `compiler-logs/`, and the transient
`tools/_build/` (git-ignored). A clean `generate` reproduces every CSV
byte-for-byte.

## CURRCVT Gap-R reconciliation (2026-07-12)

`CURRCVT-RUN.cbl` now implements the target-currency minor unit (audit Gap-R). It
ACCEPTs a fourth input, `WS-MINOR-UNIT` (0 for ITL/ESP, 2 otherwise — the
currency-master value the driver supplies from its `MINOR` table), and rounds into
a result field of the matching scale: `WS-RESULT-0 PIC S9(13)` for 0-dp targets,
`WS-RESULT-2 PIC S9(11)V99` otherwise, DISPLAYing the correctly-scaled field.

The earlier wrapper stored every result in a fixed 2-dp field and so diverged from
the model by a minor unit on the three 0-decimal-currency targets:

| Scenario   | Old wrapper (2-dp) | Model / Rust & reconciled wrapper (minor unit) |
|------------|-------------------:|-----------------------------------------------:|
| frf_to_itl | 295182.43          | 295182                                         |
| dem_to_esp | 21267.96           | 21268                                          |
| esp_to_itl | 581860.75          | 581861                                         |

The committed baselines now record the reconciled whole-number values, equal to
the model. `tools/run_baselines.py check` keeps an explicit **empty** Gap-R
exception set, so any regression surfaces as an *unexpected* mismatch. See
`RESULTS.md` for the full historical trace.
