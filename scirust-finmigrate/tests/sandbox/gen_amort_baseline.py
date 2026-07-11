#!/usr/bin/env python3
"""Golden Baseline generator for the AMORTSCH migration unit.

Encodes the COBOL semantics of ``cobol/AMORTSCH.cbl`` (see
``cobol/SEMANTICS_AMORT.md``) in Python ``decimal`` — no binary float — and emits:

  * ``amort_scenarios.csv`` — the inputs (one row per loan scenario).
  * ``amort_baseline.csv``  — the expected schedule (one row per period).
  * ``amort_baseline.sha256`` — tamper-evidence digest of the baseline.

Deterministic: re-running reproduces byte-identical files. Same production gate
as INTACCR — swap ``cobol_amortize`` for a live ``cobc`` run before shipping.
"""

from __future__ import annotations

import csv
import hashlib
from decimal import Decimal, getcontext, ROUND_HALF_UP
from pathlib import Path

getcontext().prec = 38

HERE = Path(__file__).resolve().parent

Q_MONEY = Decimal("0.01")      # V99  -> 2 dp
Q_RATE = Decimal("0.0000001")  # V9(7) -> 7 dp
MONEY_MAX = Decimal("999999999.99")


def q_money_rounded(value: Decimal) -> Decimal:
    """Store into a V99 field, NEAREST-AWAY-FROM-ZERO (COBOL ROUNDED)."""
    return value.quantize(Q_MONEY, rounding=ROUND_HALF_UP)


def cobol_amortize(principal: Decimal, rate: Decimal, payment: Decimal,
                   num_periods: int) -> list[dict[str, object]]:
    """Reference implementation of AMORTSCH's PROCEDURE DIVISION.

    Mirrors cobol/SEMANTICS_AMORT.md clause for clause: one rounding event per
    period on the current balance, exact principal split, final-payment
    reconciliation at the last period or as soon as princ >= balance, and a
    running balance that closes to exactly 0.00.
    """
    principal = principal.quantize(Q_MONEY)
    rate = rate.quantize(Q_RATE)
    payment = payment.quantize(Q_MONEY)

    balance = principal
    rows: list[dict[str, object]] = []
    period = 1
    while period <= num_periods and balance != 0:
        interest = q_money_rounded(balance * rate)  # single rounding event
        princ = (payment - interest).quantize(Q_MONEY)  # exact, may be negative

        if period == num_periods or princ >= balance:
            princ = balance
            actual_payment = (interest + princ).quantize(Q_MONEY)
        else:
            actual_payment = payment

        balance = (balance - princ).quantize(Q_MONEY)

        for name, val in (("interest", interest), ("princ", princ),
                          ("actual_payment", actual_payment), ("balance", balance)):
            if abs(val) > MONEY_MAX:
                raise ValueError(f"period {period}: {name}={val} overflows PIC range")

        rows.append({
            "period": period,
            "interest": interest,
            "principal": princ,
            "payment": actual_payment,
            "balance": balance,
        })
        period += 1
    return rows


def scenarios() -> list[tuple[str, Decimal, Decimal, Decimal, int]]:
    """(case_id, principal, monthly_rate, payment, num_periods).

    Each case documents the semantic feature it pins.
    """
    d = Decimal
    return [
        # Ordinary amortization: 5%/12 monthly, closes to 0.00 at the last row
        # after per-period rounding drift is absorbed by reconciliation.
        ("typical_5pct", d("10000.00"), d("0.0041667"), d("200.00"), 60),
        # Zero rate: every period principal == payment, exact payoff at N.
        ("zero_rate", d("1200.00"), d("0.0000000"), d("100.00"), 12),
        # Early payoff: large payment clears the loan before N; fewer rows than N.
        ("early_payoff", d("500.00"), d("0.0041667"), d("300.00"), 12),
        # Heavy half-cent drift: rate chosen so interest keeps landing on ties,
        # stress-testing the accumulated-rounding reconciliation.
        ("rounding_drift", d("1000.00"), d("0.0033333"), d("90.00"), 12),
        # Negative amortization: payment < interest, balance GROWS until the last
        # period forces payoff (princ := balance).
        ("neg_amortization", d("10000.00"), d("0.0100000"), d("50.00"), 6),
        # Single period: one row, principal == whole balance.
        ("single_period", d("1000.00"), d("0.0050000"), d("250.00"), 1),
        # Balance not divisible by payment: last scheduled period leaves a stub
        # that reconciliation must clear exactly.
        ("uneven_tail", d("1000.00"), d("0.0041667"), d("333.33"), 4),
    ]


def main() -> None:
    scen = scenarios()

    with (HERE / "amort_scenarios.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "principal", "monthly_rate", "payment", "num_periods"])
        for cid, p, r, pay, n in scen:
            w.writerow([cid, f"{p:.2f}", f"{r:.7f}", f"{pay:.2f}", n])

    baseline_path = HERE / "amort_baseline.csv"
    with baseline_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "period", "interest", "principal", "payment", "balance"])
        for cid, p, r, pay, n in scen:
            rows = cobol_amortize(p, r, pay, n)
            for row in rows:
                w.writerow([
                    cid, row["period"],
                    f"{row['interest']:.2f}", f"{row['principal']:.2f}",
                    f"{row['payment']:.2f}", f"{row['balance']:.2f}",
                ])

    digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    (HERE / "amort_baseline.sha256").write_text(f"{digest}  amort_baseline.csv\n")
    print(f"wrote amort_scenarios.csv, amort_baseline.csv, sha256={digest}")


if __name__ == "__main__":
    main()
