#!/usr/bin/env python3
"""Golden Baseline generator for the DAYCOUNT migration unit.

Encodes cobol/SEMANTICS_DAY.md in Python (integer day count + decimal interest,
no binary float in the money path) and ALSO records the Excel DAYS360 (US) day
count so every NASD-vs-Excel divergence is visible. Emits:

  * ``day_scenarios.csv`` — inputs (principal, rate, date1, date2).
  * ``day_baseline.csv``  — nasd_days, excel_days, interest.
  * ``day_baseline.sha256`` — tamper-evidence digest.

Deterministic. Production gate: swap ``nasd_days``/``interest`` for a live cobc run.
"""

from __future__ import annotations

import csv
import hashlib
from decimal import Decimal, getcontext, ROUND_HALF_UP
from pathlib import Path

getcontext().prec = 38

HERE = Path(__file__).resolve().parent
Q_MONEY = Decimal("0.01")
Q_RATE = Decimal("0.0000001")


def is_leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def last_day_of_feb(y: int) -> int:
    return 29 if is_leap(y) else 28


def is_last_of_feb(y: int, m: int, d: int) -> bool:
    return m == 2 and d == last_day_of_feb(y)


def nasd_days(d1: tuple[int, int, int], d2: tuple[int, int, int]) -> int:
    """30/360 US (NASD bond basis) — cobol/SEMANTICS_DAY.md, rules in order."""
    y1, m1, dd1 = d1
    y2, m2, dd2 = d2
    ad1, ad2 = dd1, dd2
    feb1 = is_last_of_feb(y1, m1, dd1)
    feb2 = is_last_of_feb(y2, m2, dd2)
    if feb1 and feb2:            # rule 1
        ad2 = 30
    if feb1:                     # rule 2
        ad1 = 30
    if ad2 == 31 and ad1 in (30, 31):  # rule 3
        ad2 = 30
    if ad1 == 31:                # rule 4
        ad1 = 30
    return 360 * (y2 - y1) + 30 * (m2 - m1) + (ad2 - ad1)


def excel_days360_us(d1: tuple[int, int, int], d2: tuple[int, int, int]) -> int:
    """Excel DAYS360(...,FALSE): US method WITHOUT the February EOM rules.

    Recorded only to expose the divergence from the NASD basis; NOT shipped.
    """
    y1, m1, dd1 = d1
    y2, m2, dd2 = d2
    ad1, ad2 = dd1, dd2
    if ad1 == 31:
        ad1 = 30
    if ad2 == 31 and ad1 == 30:
        ad2 = 30
    return 360 * (y2 - y1) + 30 * (m2 - m1) + (ad2 - ad1)


def interest(principal: Decimal, rate: Decimal, days: int) -> Decimal:
    """principal * rate * days / 360, single ROUNDED (NEAREST-AWAY-FROM-ZERO)."""
    principal = principal.quantize(Q_MONEY)
    rate = rate.quantize(Q_RATE)
    raw = (principal * rate * Decimal(days)) / Decimal(360)
    return raw.quantize(Q_MONEY, rounding=ROUND_HALF_UP)


def scenarios():
    """(case_id, principal, rate, date1, date2). Dates as (Y, M, D)."""
    d = Decimal
    P = d("100000.00")
    R = d("0.0500000")  # 5% annual
    return [
        # --- NASD-vs-Excel discriminators (the money question) ---------------
        # 28-Feb (non-leap) -> 31-Aug: NASD 180 vs Excel 183.
        ("feb_eom_nonleap", P, R, (2023, 2, 28), (2023, 8, 31)),
        # 29-Feb (leap) -> 31-Aug: NASD 180 vs Excel 182.
        ("feb_eom_leap", P, R, (2024, 2, 29), (2024, 8, 31)),
        # Both last-day-of-Feb across a year: exactly 360 (rule 1).
        ("feb_to_feb_year", P, R, (2024, 2, 29), (2025, 2, 28)),
        # --- 31st rules ------------------------------------------------------
        ("both_31", P, R, (2023, 1, 31), (2023, 3, 31)),        # 60
        ("d1_31_only", P, R, (2023, 1, 31), (2023, 4, 30)),      # 90
        ("d2_31_d1_mid", P, R, (2023, 1, 15), (2023, 3, 31)),    # 76 (no reduction)
        # --- boundaries ------------------------------------------------------
        ("same_date", P, R, (2023, 6, 15), (2023, 6, 15)),       # 0
        ("full_year", P, R, (2023, 1, 1), (2024, 1, 1)),         # 360
        ("half_year", P, R, (2023, 1, 10), (2023, 7, 10)),       # 180
        # A rounding-sensitive interest amount: 100000*0.05*76/360 = 1055.5555..
        ("interest_rounding", P, R, (2023, 1, 15), (2023, 3, 31)),
    ]


def main() -> None:
    scen = scenarios()

    with (HERE / "day_scenarios.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "principal", "annual_rate",
                    "y1", "m1", "d1", "y2", "m2", "d2"])
        for cid, p, r, a, b in scen:
            w.writerow([cid, f"{p:.2f}", f"{r:.7f}", a[0], a[1], a[2], b[0], b[1], b[2]])

    baseline_path = HERE / "day_baseline.csv"
    diverge = 0
    with baseline_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "nasd_days", "excel_days", "interest"])
        for cid, p, r, a, b in scen:
            nd = nasd_days(a, b)
            ed = excel_days360_us(a, b)
            if nd != ed:
                diverge += 1
            w.writerow([cid, nd, ed, f"{interest(p, r, nd):.2f}"])

    digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    (HERE / "day_baseline.sha256").write_text(f"{digest}  day_baseline.csv\n")
    print(f"wrote day_scenarios.csv, day_baseline.csv ({len(scen)} rows), "
          f"{diverge} NASD!=Excel divergences recorded, sha256={digest}")


if __name__ == "__main__":
    main()
