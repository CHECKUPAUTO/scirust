#!/usr/bin/env python3
"""Golden Baseline generator for the CURRCVT migration unit.

Encodes cobol/SEMANTICS_CURR.md in Python decimal (no binary float): euro
triangulation per EC 1103/97. Records BOTH the lawful triangulated result and
the (unlawful) direct cross-rate result so the divergence is visible audit
evidence. Emits:

  * ``curr_scenarios.csv`` — inputs (amount, from, to).
  * ``curr_baseline.csv``  — triangulated result, direct result, euro intermediate.
  * ``curr_baseline.sha256`` — tamper-evidence digest.

Deterministic. Production gate: swap ``triangulate`` for a live cobc run AND
confirm the target's intermediate-euro precision (Gap-Q).
"""

from __future__ import annotations

import csv
import hashlib
from decimal import Decimal, getcontext, ROUND_HALF_UP
from pathlib import Path

getcontext().prec = 38

HERE = Path(__file__).resolve().parent

# code -> (rate: 1 EUR = X national, six significant figures; minor unit dp)
CCY = {
    "DEM": (Decimal("1.95583"), 2),
    "FRF": (Decimal("6.55957"), 2),
    "ITL": (Decimal("1936.27"), 0),
    "ESP": (Decimal("166.386"), 0),
    "IEP": (Decimal("0.787564"), 2),
}
Q_EURO = Decimal("0.001")  # >= 3 dp intermediate


def quant(scale: int) -> Decimal:
    return Decimal(1).scaleb(-scale) if scale > 0 else Decimal(1)


def triangulate(amount: Decimal, src: str, dst: str) -> tuple[Decimal, Decimal]:
    """(result, euro_intermediate). cobol/SEMANTICS_CURR.md, both ROUND half-up."""
    rate_from, minor_from = CCY[src]
    rate_to, minor_to = CCY[dst]
    amount = amount.quantize(quant(minor_from))
    euro = (amount / rate_from).quantize(Q_EURO, rounding=ROUND_HALF_UP)  # >=3 dp
    result = (euro * rate_to).quantize(quant(minor_to), rounding=ROUND_HALF_UP)
    return result, euro


def direct(amount: Decimal, src: str, dst: str) -> Decimal:
    """The UNLAWFUL shortcut: amount * rate_to / rate_from, no euro rounding."""
    rate_from, minor_from = CCY[src]
    rate_to, minor_to = CCY[dst]
    amount = amount.quantize(quant(minor_from))
    return (amount * rate_to / rate_from).quantize(quant(minor_to), rounding=ROUND_HALF_UP)


def scenarios():
    """(case_id, amount, from, to). Each pins a documented property."""
    d = Decimal
    return [
        ("dem_to_frf", d("100.00"), "DEM", "FRF"),
        ("frf_to_dem", d("100.00"), "FRF", "DEM"),
        ("frf_to_itl", d("1000.00"), "FRF", "ITL"),   # target 0 dp (whole lira)
        ("dem_to_esp", d("250.00"), "DEM", "ESP"),    # target 0 dp
        ("itl_to_frf", d("100000"), "ITL", "FRF"),    # source 0 dp
        ("iep_to_dem", d("100.00"), "IEP", "DEM"),    # rate < 1 source
        ("dem_to_iep", d("100.00"), "DEM", "IEP"),    # rate < 1 target
        ("esp_to_itl", d("50000"), "ESP", "ITL"),     # both 0 dp
        ("small_dem_to_frf", d("1.00"), "DEM", "FRF"),
        ("large_frf_to_dem", d("9999999.00"), "FRF", "DEM"),
    ]


def main() -> None:
    scen = scenarios()

    with (HERE / "curr_scenarios.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "amount", "from_ccy", "to_ccy"])
        for cid, amt, src, dst in scen:
            w.writerow([cid, f"{amt.quantize(quant(CCY[src][1]))}", src, dst])

    baseline_path = HERE / "curr_baseline.csv"
    diverge = 0
    with baseline_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "amount", "from_ccy", "to_ccy",
                    "result", "direct", "euro"])
        for cid, amt, src, dst in scen:
            result, euro = triangulate(amt, src, dst)
            drct = direct(amt, src, dst)
            if result != drct:
                diverge += 1
            minor_to = CCY[dst][1]
            w.writerow([
                cid, f"{amt.quantize(quant(CCY[src][1]))}", src, dst,
                f"{result.quantize(quant(minor_to))}",
                f"{drct.quantize(quant(minor_to))}",
                f"{euro:.3f}",
            ])

    digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    (HERE / "curr_baseline.sha256").write_text(f"{digest}  curr_baseline.csv\n")
    print(f"wrote curr_scenarios.csv, curr_baseline.csv ({len(scen)} rows), "
          f"{diverge} triangulated!=direct divergences, sha256={digest}")


if __name__ == "__main__":
    main()
