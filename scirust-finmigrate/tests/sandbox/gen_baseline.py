#!/usr/bin/env python3
"""Golden Baseline generator for the INTACCR migration unit.

Encodes the COBOL semantics of ``cobol/INTACCR.cbl`` (see ``cobol/SEMANTICS.md``)
using Python's ``decimal`` module — NO binary floating point anywhere — and
emits two committed artifacts under this directory:

  * ``dataset.csv``          — synthetic inputs (hand-picked edge cases +
                               deterministic pseudo-random fill, fixed seed).
  * ``golden_baseline.csv``  — the expected outputs for every input row.
  * ``golden_baseline.sha256`` — tamper-evidence digest of the baseline.

The generator is fully deterministic: re-running it must reproduce byte-identical
files. That property is what lets the Rust equivalence test treat the baseline as
an immutable oracle.

PRODUCTION GATE: when a real COBOL compiler is available, replace the body of
``cobol_accrue`` with the parsed output of:

    cobc -x -free cobol/INTACCR.cbl && ./INTACCR   # driven per input row

The output column layout (principal, annual_rate, monthly_int, monthly_trunc,
new_balance) is intentionally the shape a DISPLAY of the WORKING-STORAGE fields
produces, so the swap is mechanical.
"""

from __future__ import annotations

import csv
import hashlib
import random
from decimal import Decimal, getcontext, ROUND_HALF_UP, ROUND_DOWN
from pathlib import Path

# 38 significant digits models the high-precision GMP-backed intermediates that
# GnuCOBOL uses; the destination scale (2 dp) can never observe more than this.
getcontext().prec = 38

HERE = Path(__file__).resolve().parent

# Fixed-scale quantizers matching the PICTURE clauses in INTACCR.cbl.
Q_MONEY = Decimal("0.01")     # V99  -> 2 dp
Q_RATE = Decimal("0.00001")   # V9(5) -> 5 dp

# PIC S9(9)V99 magnitude bound.
MONEY_MAX = Decimal("999999999.99")


def q_money(value: Decimal) -> Decimal:
    """Store into a V99 field with NEAREST-AWAY-FROM-ZERO (COBOL ROUNDED)."""
    return value.quantize(Q_MONEY, rounding=ROUND_HALF_UP)


def q_money_trunc(value: Decimal) -> Decimal:
    """Store into a V99 field with truncation toward zero (COBOL default)."""
    return value.quantize(Q_MONEY, rounding=ROUND_DOWN)


def cobol_accrue(principal: Decimal, annual_rate: Decimal) -> dict[str, Decimal]:
    """Reference implementation of INTACCR's PROCEDURE DIVISION.

    Mirrors, clause for clause, ``cobol/SEMANTICS.md``:
      * product is exact (no rounding),
      * division carried at 38-digit intermediate precision,
      * a SINGLE rounding event at the store into each 2 dp field,
      * ROUNDED = ROUND_HALF_UP, default = ROUND_DOWN,
      * posting is an exact 2 dp + 2 dp sum.
    """
    # Fields carry their declared fixed scale.
    principal = principal.quantize(Q_MONEY)
    annual_rate = annual_rate.quantize(Q_RATE)

    # principal * rate is exact (2 dp * 5 dp = 7 dp); / 12 at high precision.
    intermediate = (principal * annual_rate) / Decimal(12)

    monthly_int = q_money(intermediate)          # ROUNDED
    monthly_trunc = q_money_trunc(intermediate)  # default truncation
    new_balance = (principal + monthly_int).quantize(Q_MONEY)  # exact sum

    return {
        "monthly_int": monthly_int,
        "monthly_trunc": monthly_trunc,
        "new_balance": new_balance,
    }


def edge_cases() -> list[tuple[str, Decimal, Decimal]]:
    """Hand-picked rows that discriminate the semantic contract.

    Each tuple is (case_id, principal, annual_rate). ``case_id`` documents WHY
    the row exists so a future auditor can read intent, not just numbers.
    """
    d = Decimal
    return [
        # --- rounding-mode discriminators (the money question) ---------------
        # 100.00 * 0.00060 / 12 = 0.005 EXACTLY. ROUNDED must give 0.01 (up),
        # truncation must give 0.00. Banker's rounding would (wrongly) give 0.00.
        ("half_cent_positive", d("100.00"), d("0.00060")),
        # Mirror with a negative principal: -0.005 -> ROUNDED -0.01 (away from 0),
        # truncation -0.00. Proves sign-aware half-away-from-zero.
        ("half_cent_negative", d("-100.00"), d("0.00060")),
        # 100.00 * 0.00180 / 12 = 0.015 EXACTLY -> ROUNDED 0.02, banker's 0.02
        # (even) — pairs with the 0.005 case to distinguish HALF_UP from EVEN.
        ("half_cent_up_to_even", d("100.00"), d("0.00180")),
        # 100.00 * 0.00300 / 12 = 0.025 EXACTLY -> ROUNDED 0.03, banker's 0.02.
        ("half_cent_up_from_even", d("100.00"), d("0.00300")),

        # --- truncation vs rounding on a non-tie fraction --------------------
        # 1200.40 * 0.02999 / 12 = 2.99966... -> trunc 2.99, ROUNDED 3.00.
        ("just_below_next_cent", d("1200.40"), d("0.02999")),

        # --- boundaries ------------------------------------------------------
        ("zero_principal", d("0.00"), d("0.03500")),
        # Principal exactly at the PIC ceiling with zero rate: new_balance lands
        # on 999999999.99 EXACTLY — the largest representable value, no overflow.
        ("max_principal_zero_rate", d("999999999.99"), d("0.00000")),
        ("tiny_principal", d("0.01"), d("0.10000")),
        # Large magnitude with real interest; new_balance stays strictly in range
        # (990000000.00 * 0.01 / 12 = 825000.00 -> 990825000.00 < ceiling).
        ("large_principal_in_range", d("990000000.00"), d("0.01000")),
        ("negative_rate", d("5000.00"), d("-0.04250")),

        # --- an everyday posting ---------------------------------------------
        ("typical_savings", d("2500.75"), d("0.03500")),
    ]


def random_fill(n: int, seed: int) -> list[tuple[str, Decimal, Decimal]]:
    """Deterministic pseudo-random rows to widen coverage. Fixed seed => stable."""
    rng = random.Random(seed)
    rows: list[tuple[str, Decimal, Decimal]] = []
    # Bound magnitude to 900_000_000.00 so that, even at the maximum rate,
    # new_balance = principal + interest cannot overflow PIC S9(9)V99. This keeps
    # the equivalence set strictly free of size-errors (audit_report.md Gap-5);
    # the size-error branch is proven separately in the Rust unit tests.
    for i in range(n):
        cents = rng.randint(-90000000000, 90000000000)  # +/- 900,000,000.00
        principal = (Decimal(cents) / Decimal(100)).quantize(Q_MONEY)
        rate = (Decimal(rng.randint(0, 99999)) / Decimal(100000)).quantize(Q_RATE)
        rows.append((f"rand_{i:03d}", principal, rate))
    return rows


def main() -> None:
    rows = edge_cases() + random_fill(n=64, seed=20260711)

    dataset_path = HERE / "dataset.csv"
    baseline_path = HERE / "golden_baseline.csv"

    with dataset_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "principal", "annual_rate"])
        for case_id, principal, rate in rows:
            w.writerow([case_id, f"{principal:.2f}", f"{rate:.5f}"])

    with baseline_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["case_id", "principal", "annual_rate",
             "monthly_int", "monthly_trunc", "new_balance"]
        )
        for case_id, principal, rate in rows:
            if abs(principal) > MONEY_MAX:
                raise ValueError(f"{case_id}: principal out of PIC range")
            out = cobol_accrue(principal, rate)
            # Guarantee the equivalence set is size-error-free: every stored
            # money field must fit PIC S9(9)V99. If this ever trips, tighten the
            # random-fill bound rather than shipping a row the Rust port rejects.
            for name in ("monthly_int", "monthly_trunc", "new_balance"):
                if abs(out[name]) > MONEY_MAX:
                    raise ValueError(f"{case_id}: {name}={out[name]} overflows PIC range")
            w.writerow([
                case_id,
                f"{principal:.2f}",
                f"{rate:.5f}",
                f"{out['monthly_int']:.2f}",
                f"{out['monthly_trunc']:.2f}",
                f"{out['new_balance']:.2f}",
            ])

    digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    (HERE / "golden_baseline.sha256").write_text(
        f"{digest}  golden_baseline.csv\n"
    )

    print(f"wrote {dataset_path.name}, {baseline_path.name} "
          f"({len(rows)} rows), sha256={digest}")


if __name__ == "__main__":
    main()
