#!/usr/bin/env python3
"""Golden Baseline generator for the PAYCALC migration unit.

Encodes cobol/SEMANTICS_PAY.md in Python ``decimal`` (the shipped path is
decimal-only) and ALSO computes the payment the legacy-float way to prove the
arithmetic swap (float negative-exponent form -> decimal positive-integer form)
does not change the customer's payment at the cent. Emits:

  * ``pay_scenarios.csv`` — inputs (principal, monthly_rate, num_periods).
  * ``pay_baseline.csv``  — expected factor (9 dp) and payment (2 dp).
  * ``pay_baseline.sha256`` — tamper-evidence digest.

Deterministic. Production gate: swap ``decimal_paycalc`` for a live ``cobc`` run.
"""

from __future__ import annotations

import csv
import hashlib
from decimal import Decimal, getcontext, ROUND_HALF_UP
from pathlib import Path

getcontext().prec = 38

HERE = Path(__file__).resolve().parent

Q_MONEY = Decimal("0.01")        # V99   -> 2 dp
Q_RATE = Decimal("0.0000001")    # V9(7) -> 7 dp
Q_FACTOR = Decimal("0.000000001")  # V9(9) -> 9 dp
MONEY_MAX = Decimal("999999999.99")
PAYMENT_MAX = Decimal("9999999.99")  # S9(7)V99


def q(value: Decimal, quantum: Decimal) -> Decimal:
    """Store into a fixed-scale field, NEAREST-AWAY-FROM-ZERO (COBOL ROUNDED)."""
    return value.quantize(quantum, rounding=ROUND_HALF_UP)


def decimal_paycalc(principal: Decimal, rate: Decimal, n: int) -> dict[str, Decimal]:
    """Reference for PAYCALC's decimal (shipped) path — cobol/SEMANTICS_PAY.md."""
    principal = principal.quantize(Q_MONEY)
    rate = rate.quantize(Q_RATE)

    if rate == 0:
        factor = q(Decimal(1), Q_FACTOR)
        payment = q(principal / Decimal(n), Q_MONEY)
        return {"factor": factor, "payment": payment}

    # f = (1+i)^n as a succession of fixed-point multiplications; stored at 9 dp.
    one_plus = Decimal(1) + rate
    f = Decimal(1)
    for _ in range(n):
        f = f * one_plus
    factor = q(f, Q_FACTOR)  # single rounding event into the 9-dp field
    payment = q((principal * rate * factor) / (factor - Decimal(1)), Q_MONEY)
    return {"factor": factor, "payment": payment}


def float_payment(principal: Decimal, rate: Decimal, n: int) -> Decimal:
    """Legacy-float path: the IEEE-754 arithmetic the negative-exponent annuity
    form would have used. Used ONLY to cross-check the decimal payment."""
    if rate == 0:
        return q(principal / Decimal(n), Q_MONEY)
    p = float(principal)
    i = float(rate)
    pay = p * i / (1.0 - (1.0 + i) ** (-n))
    return q(Decimal(repr(pay)), Q_MONEY)


def scenarios() -> list[tuple[str, Decimal, Decimal, int]]:
    """(case_id, principal, monthly_rate, num_periods). Bounds: n<=120, i<=0.05."""
    d = Decimal
    return [
        ("mortgage_5pct_5yr", d("10000.00"), d("0.0041667"), 60),
        ("auto_3pct_4yr", d("25000.00"), d("0.0025000"), 48),
        ("card_18pct_2yr", d("3000.00"), d("0.0150000"), 24),
        ("high_5pct_mo_1yr", d("1000.00"), d("0.0500000"), 12),
        ("zero_rate_10", d("1200.00"), d("0.0000000"), 10),
        ("single_period", d("5000.00"), d("0.0041667"), 1),
        ("long_120", d("50000.00"), d("0.0033333"), 120),
        ("small_principal", d("100.00"), d("0.0100000"), 36),
    ]


def main() -> None:
    scen = scenarios()

    with (HERE / "pay_scenarios.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "principal", "monthly_rate", "num_periods"])
        for cid, p, r, n in scen:
            w.writerow([cid, f"{p:.2f}", f"{r:.7f}", n])

    baseline_path = HERE / "pay_baseline.csv"
    with baseline_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "principal", "monthly_rate", "num_periods", "factor", "payment"])
        for cid, p, r, n in scen:
            out = decimal_paycalc(p, r, n)
            # Audit cross-check: the decimal payment must equal the legacy-float
            # payment at the cent. If this ever trips, it is a real finding —
            # annotate the scenario rather than silently shipping a divergence.
            fpay = float_payment(p, r, n)
            if out["payment"] != fpay:
                raise ValueError(
                    f"{cid}: decimal payment {out['payment']} != float payment {fpay} "
                    f"(arithmetic swap changed the cent — investigate)"
                )
            if abs(out["payment"]) > PAYMENT_MAX:
                raise ValueError(f"{cid}: payment {out['payment']} overflows S9(7)V99")
            w.writerow([
                cid, f"{p:.2f}", f"{r:.7f}", n,
                f"{out['factor']:.9f}", f"{out['payment']:.2f}",
            ])

    digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    (HERE / "pay_baseline.sha256").write_text(f"{digest}  pay_baseline.csv\n")
    print(f"wrote pay_scenarios.csv, pay_baseline.csv ({len(scen)} rows), "
          f"sha256={digest}  [decimal==float at the cent for all rows]")


if __name__ == "__main__":
    main()
