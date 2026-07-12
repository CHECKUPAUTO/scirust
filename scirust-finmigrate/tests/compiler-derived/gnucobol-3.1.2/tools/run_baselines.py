#!/usr/bin/env python3
"""Deterministic driver for the GnuCOBOL compiler-derived baselines.

Regenerates every `baselines/<UNIT>-compiler-baseline.csv` **from scratch** by
compiling the instrumented `normalized-sources/<UNIT>-RUN.cbl` wrappers with
GnuCOBOL and executing them once per committed scenario, capturing the raw
`DISPLAY` output. No value is invented, rounded, or hand-edited: each CSV cell is
either echoed verbatim from the scenario input file or is the faithful,
lexically-formatted raw output of the COBOL program.

Standard library only. Deterministic (GnuCOBOL compilation and the ACCEPT/DISPLAY
arithmetic are deterministic; scenarios are read in file order).

Modes
-----
  generate   compile + run + (re)write baselines/*.csv and compiler-logs/*.log
             (requires `cobc` on PATH)
  verify     recompile + run and assert the committed baselines/*.csv match the
             live GnuCOBOL output field-for-field as decimals (requires `cobc`)
  check      no compiler: assert committed baselines are numerically consistent
             with the model baselines in ../../sandbox for the shared emitted
             fields (this is the divergence-aware equivalence check)

Faithfulness note (CURRCVT — Gap-R reconciled 2026-07-12)
---------------------------------------------------------
`CURRCVT-RUN.cbl` now implements the target-currency minor unit (audit Gap-R):
it ACCEPTs `WS-MINOR-UNIT` (0 for ITL/ESP, 2 otherwise) after the two rates and
rounds into a result field of the matching scale. The driver supplies that minor
unit from the `MINOR` table below. For the three 0-decimal-currency targets the
raw COBOL result is therefore a whole number (e.g. 295182) and equals the
model/Rust baseline. The earlier wrapper used a fixed 2-dp field and diverged by
a minor unit on those rows; that history is recorded in RESULTS.md. `check` now
expects zero divergences.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # .../gnucobol-3.1.2
SRC = os.path.join(ROOT, "normalized-sources")
BASELINES = os.path.join(ROOT, "baselines")
LOGS = os.path.join(ROOT, "compiler-logs")
SANDBOX = os.path.normpath(os.path.join(ROOT, "..", "..", "sandbox"))
BUILD = os.path.join(HERE, "_build")             # transient; never committed

# The exact compile command (also recorded in REPRODUCE.md). `-free` because the
# normalized sources are free-format; `-x` builds an executable to run.
COMPILE = ["cobc", "-x", "-free", "-O2"]
# Pure-cobc acceptance diagnostics captured as the committed compiler log (no C
# backend, so no host-gcc noise).
SYNTAX = ["cobc", "-fsyntax-only", "-free", "-Wall"]

# 1 EUR = X national currency, six significant figures (EC 1103/97).
RATE = {
    "DEM": "1.95583", "FRF": "6.55957", "ITL": "1936.27",
    "ESP": "166.386", "IEP": "0.787564",
}

# Target currency minor unit (decimal places) — the currency-master value the
# CURRCVT wrapper rounds to (Gap-R). 0 for ITL/ESP, 2 otherwise.
MINOR = {
    "DEM": "2", "FRF": "2", "ITL": "0", "ESP": "0", "IEP": "2",
}


def fmt(raw: str) -> str:
    """Format one raw COBOL DISPLAY token into its CSV representation.

    Purely lexical: DISPLAY already emits the field's declared number of decimal
    places, so we only drop the leading `+`, strip leading integer zeros (keeping
    at least one digit), and keep a leading `-` for negatives. The fractional part
    (including trailing zeros) is preserved verbatim.
    """
    s = raw.strip()
    neg = s.startswith("-")
    if s and s[0] in "+-":
        s = s[1:]
    if "." in s:
        intp, frac = s.split(".", 1)
    else:
        intp, frac = s, None
    intp = intp.lstrip("0") or "0"
    val = intp if frac is None else f"{intp}.{frac}"
    # A zero magnitude never carries a sign (COBOL emits +0 for -0 on store).
    if neg and set(intp + (frac or "")) != {"0"}:
        val = "-" + val
    return val


def compile_unit(unit: str) -> tuple[str, str]:
    """Compile <unit>-RUN.cbl; return (exe_path, syntax_diagnostics)."""
    os.makedirs(BUILD, exist_ok=True)
    src = os.path.join(SRC, f"{unit}-RUN.cbl")
    exe = os.path.join(BUILD, f"{unit}-RUN")
    r = subprocess.run(COMPILE + ["-o", exe, src], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{unit}: compile failed (rc={r.returncode})\n{r.stderr}")
    syn = subprocess.run(SYNTAX + [src], capture_output=True, text=True)
    return exe, syn.stderr


def run(exe: str, inputs: list[str]) -> list[str]:
    """Run the wrapper feeding `inputs` on stdin; return non-empty DISPLAY lines."""
    r = subprocess.run([exe], input="\n".join(inputs) + "\n", capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"run failed (rc={r.returncode})\n{r.stderr}")
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def scenarios(name: str) -> list[dict]:
    with open(os.path.join(SANDBOX, name), newline="") as f:
        return list(csv.DictReader(f))


# Per-unit specification. `emit(exe, scenario_row) -> list[csv_row_dict]`.
# `header` is the exact committed column order.

def emit_intaccr(exe, s):
    o = run(exe, [s["principal"], s["annual_rate"]])
    return [{"case_id": s["case_id"], "principal": s["principal"], "annual_rate": s["annual_rate"],
             "monthly_int": fmt(o[0]), "monthly_trunc": fmt(o[1]), "new_balance": fmt(o[2])}]


def emit_amortsch(exe, s):
    o = run(exe, [s["principal"], s["monthly_rate"], s["payment"], s["num_periods"]])
    rows = []
    for i in range(0, len(o), 5):
        p, itr, prin, pay, bal = o[i:i + 5]
        rows.append({"case_id": s["case_id"], "period": fmt(p), "interest": fmt(itr),
                     "principal": fmt(prin), "payment": fmt(pay), "balance": fmt(bal)})
    return rows


def emit_paycalc(exe, s):
    o = run(exe, [s["principal"], s["monthly_rate"], s["num_periods"]])
    return [{"case_id": s["case_id"], "principal": s["principal"], "monthly_rate": s["monthly_rate"],
             "num_periods": s["num_periods"], "factor": fmt(o[0]), "payment": fmt(o[1])}]


def emit_daycount(exe, s):
    o = run(exe, [s["principal"], s["annual_rate"], s["y1"], s["m1"], s["d1"], s["y2"], s["m2"], s["d2"]])
    return [{"case_id": s["case_id"], "nasd_days": fmt(o[0]), "interest": fmt(o[1])}]


def emit_brktcalc(exe, s):
    o = run(exe, [s["base"]])
    return [{"case_id": s["case_id"], "base": s["base"], "tax": fmt(o[0])}]


def emit_currcvt(exe, s):
    # inputs: amount, rate_from, rate_to, target minor unit (Gap-R)
    o = run(exe, [s["amount"], RATE[s["from_ccy"]], RATE[s["to_ccy"]], MINOR[s["to_ccy"]]])  # DISPLAY: euro, result
    return [{"case_id": s["case_id"], "amount": s["amount"], "from_ccy": s["from_ccy"],
             "to_ccy": s["to_ccy"], "result": fmt(o[1]), "euro": fmt(o[0])}]


UNITS = {
    "INTACCR": ("dataset.csv", "case_id,principal,annual_rate,monthly_int,monthly_trunc,new_balance", emit_intaccr),
    "AMORTSCH": ("amort_scenarios.csv", "case_id,period,interest,principal,payment,balance", emit_amortsch),
    "PAYCALC": ("pay_scenarios.csv", "case_id,principal,monthly_rate,num_periods,factor,payment", emit_paycalc),
    "DAYCOUNT": ("day_scenarios.csv", "case_id,nasd_days,interest", emit_daycount),
    "BRKTCALC": ("brkt_scenarios.csv", "case_id,base,tax", emit_brktcalc),
    "CURRCVT": ("curr_scenarios.csv", "case_id,amount,from_ccy,to_ccy,result,euro", emit_currcvt),
}


def build_rows(unit):
    scen_file, header, emit = UNITS[unit]
    exe, syn = compile_unit(unit)
    cols = header.split(",")
    rows = []
    for s in scenarios(scen_file):
        rows.extend(emit(exe, s))
    return header, cols, rows, syn


def write_csv(path, header, cols, rows):
    # LF line endings, trailing newline — matches the committed baselines.
    with open(path, "w", newline="\n") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(r[c] for c in cols) + "\n")


def cmd_generate():
    os.makedirs(LOGS, exist_ok=True)
    for unit in UNITS:
        header, cols, rows, syn = build_rows(unit)
        write_csv(os.path.join(BASELINES, f"{unit}-compiler-baseline.csv"), header, cols, rows)
        with open(os.path.join(LOGS, f"{unit}.log"), "w", newline="\n") as f:
            f.write(f"$ {' '.join(SYNTAX)} normalized-sources/{unit}-RUN.cbl\n")
            f.write(syn if syn else "(no diagnostics: 0 warnings, 0 errors)\n")
        print(f"generate {unit}: {len(rows)} rows")
    print("done. regenerate SHA256SUMS next (see REPRODUCE.md).")


def cmd_verify():
    total = mism = 0
    for unit in UNITS:
        header, cols, rows, _ = build_rows(unit)
        committed = list(csv.DictReader(open(os.path.join(BASELINES, f"{unit}-compiler-baseline.csv"))))
        assert len(rows) == len(committed), f"{unit}: row count {len(rows)} != {len(committed)}"
        for got, want in zip(rows, committed):
            for c in cols:
                total += 1
                a, b = got[c], want[c]
                try:
                    eq = Decimal(a) == Decimal(b)
                except Exception:
                    eq = a == b
                if not eq:
                    mism += 1
                    print(f"  {unit} {got.get('case_id')} {c}: live={a} committed={b}")
        print(f"verify {unit}: {len(rows)} rows OK")
    print(f"\nverify: {total} field comparisons, {mism} mismatches vs live GnuCOBOL")
    return 0 if mism == 0 else 1


def cmd_check():
    """No compiler: committed compiler CSV vs model baseline (../../sandbox)."""
    model = {
        "INTACCR": ("golden_baseline.csv", ["principal", "annual_rate", "monthly_int", "monthly_trunc", "new_balance"], ["case_id"]),
        "AMORTSCH": ("amort_baseline.csv", ["interest", "principal", "payment", "balance"], ["case_id", "period"]),
        "PAYCALC": ("pay_baseline.csv", ["principal", "monthly_rate", "num_periods", "factor", "payment"], ["case_id"]),
        "DAYCOUNT": ("day_baseline.csv", ["nasd_days", "interest"], ["case_id"]),
        "BRKTCALC": ("brkt_baseline.csv", ["base", "tax"], ["case_id"]),
        "CURRCVT": ("curr_baseline.csv", ["amount", "result", "euro"], ["case_id"]),
    }
    # Gap-R was reconciled 2026-07-12 (CURRCVT-RUN now rounds to the target minor
    # unit); no COBOL-vs-model divergences remain. Kept as an explicit empty set
    # so a regression would surface as an UNEXPECTED mismatch, not a silent pass.
    KNOWN: set[tuple[str, str, str]] = set()
    total = mism = known = 0
    for unit, (mf, cols, keys) in model.items():
        comp = list(csv.DictReader(open(os.path.join(BASELINES, f"{unit}-compiler-baseline.csv"))))
        mrows = {tuple(r[k] for k in keys): r for r in csv.DictReader(open(os.path.join(SANDBOX, mf)))}
        for r in comp:
            k = tuple(r[k2] for k2 in keys)
            m = mrows[k]
            for c in cols:
                total += 1
                try:
                    eq = Decimal(r[c]) == Decimal(m[c])
                except Exception:
                    eq = r[c] == m[c]
                if not eq:
                    if (unit, r["case_id"], c) in KNOWN:
                        known += 1
                    else:
                        mism += 1
                        print(f"  UNEXPECTED {unit} {r['case_id']} {c}: compiler={r[c]} model={m[c]}")
    print(f"check: {total} comparisons, {mism} unexpected mismatches, {known} documented Gap-R divergences")
    return 0 if mism == 0 else 1


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "generate":
        cmd_generate()
    elif mode == "verify":
        sys.exit(cmd_verify())
    elif mode == "check":
        sys.exit(cmd_check())
    else:
        raise SystemExit(f"unknown mode {mode!r}; use generate | verify | check")


if __name__ == "__main__":
    main()
