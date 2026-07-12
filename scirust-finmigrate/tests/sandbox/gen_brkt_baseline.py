#!/usr/bin/env python3
"""Golden Baseline generator for the BRKTCALC migration unit.

Encodes cobol/SEMANTICS_BRKT.md in Python decimal (no binary float) and records,
alongside the correct MARGINAL tax, the WRONG flat-top-rate tax so the
marginal-vs-flat divergence is visible audit evidence. Emits:

  * ``brkt_scenarios.csv`` — inputs (base).
  * ``brkt_baseline.csv``  — tax (marginal), flat_tax (top rate * base), effective %.
  * ``brkt_baseline.sha256`` — tamper-evidence digest.

Deterministic. Production gate: swap ``marginal_tax`` for a live cobc run.
"""

from __future__ import annotations

import csv
import hashlib
from decimal import Decimal, getcontext, ROUND_HALF_UP
from pathlib import Path

getcontext().prec = 38

HERE = Path(__file__).resolve().parent
Q_MONEY = Decimal("0.01")

# (lower inclusive floor, marginal rate) — must match BRKTCALC.cbl 1000-LOAD-TABLE.
BRACKETS = [
    (Decimal("0.00"), Decimal("0.00000")),
    (Decimal("10000.00"), Decimal("0.10000")),
    (Decimal("40000.00"), Decimal("0.22000")),
    (Decimal("85000.00"), Decimal("0.24000")),
    (Decimal("165000.00"), Decimal("0.32000")),
]
TOP_RATE = BRACKETS[-1][1]


def marginal_tax(base: Decimal) -> Decimal:
    """Progressive marginal tax — cobol/SEMANTICS_BRKT.md. Single rounding event."""
    base = base.quantize(Q_MONEY)
    accum = Decimal(0)
    n = len(BRACKETS)
    for i in range(n):
        lower = BRACKETS[i][0]
        rate = BRACKETS[i][1]
        upper = BRACKETS[i + 1][0] if i + 1 < n else base
        if upper > base:
            upper = base
        portion = upper - lower
        if portion < 0:
            portion = Decimal(0)
        accum += portion * rate  # full precision, no rounding
    return accum.quantize(Q_MONEY, rounding=ROUND_HALF_UP)


def flat_tax(base: Decimal) -> Decimal:
    """The WRONG flat computation: top rate on the whole base. Evidence only."""
    return (base.quantize(Q_MONEY) * TOP_RATE).quantize(Q_MONEY, rounding=ROUND_HALF_UP)


def scenarios():
    """(case_id, base). Each pins a documented property."""
    d = Decimal
    return [
        ("zero_base", d("0.00")),
        ("in_zero_bracket", d("5000.00")),          # below 10k -> 0 tax
        ("exactly_on_boundary_10k", d("10000.00")), # fills bracket 1, bracket 2 empty
        ("exactly_on_boundary_40k", d("40000.00")), # 30000*0.10 = 3000.00
        ("mid_third_bracket", d("60000.00")),        # 3000 + 20000*0.22 = 7400
        ("typical_100k", d("100000.00")),            # 16500.00 (marginal) vs 32000 flat
        ("into_top_bracket", d("200000.00")),        # includes 35000*0.32
        ("rounding_sensitive", d("12345.67")),       # 2345.67*0.10 = 234.567 -> 234.57
        ("near_max_base", d("900000000.00")),        # large; stays in field
    ]


def main() -> None:
    scen = scenarios()

    with (HERE / "brkt_scenarios.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "base"])
        for cid, base in scen:
            w.writerow([cid, f"{base:.2f}"])

    baseline_path = HERE / "brkt_baseline.csv"
    with baseline_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "base", "tax", "flat_tax", "effective_pct"])
        for cid, base in scen:
            tax = marginal_tax(base)
            flat = flat_tax(base)
            eff = (tax / base * Decimal(100)).quantize(Decimal("0.0001")) if base != 0 else Decimal("0.0000")
            w.writerow([cid, f"{base:.2f}", f"{tax:.2f}", f"{flat:.2f}", f"{eff:.4f}"])

    digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    (HERE / "brkt_baseline.sha256").write_text(f"{digest}  brkt_baseline.csv\n")
    print(f"wrote brkt_scenarios.csv, brkt_baseline.csv ({len(scen)} rows), sha256={digest}")


if __name__ == "__main__":
    main()
