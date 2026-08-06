#!/usr/bin/env python3
"""Generate `envelope_oracle.rs` from the ITD reference implementation.

The reference is the shift-aware OOD detector and three-state abstention policy
of the ITD research simulator (`itd_research.ood_shift.detector` /
`itd_research.ood_shift.policy`, standardization statistics from
`itd_research.ood.reference.fit_reference`).  Every value emitted here is
produced by the ORIGINAL Python functions -- never re-derived in this script --
so the Rust `envelope` module is validated against the reference implementation
as an oracle.

Usage (needs the itd-simulator repository on PYTHONPATH):

    PYTHONPATH=/path/to/itd-simulator python3 gen_envelope_oracle.py envelope_oracle.rs

Determinism: `numpy.random.default_rng(42)` only.
"""

from __future__ import annotations

import sys

import numpy as np

from itd_research.ood_shift.detector import fit_shift_reference, monotone_separation
from itd_research.ood_shift.policy import confidence_discount, three_state_policy

N_ROWS, N_FEATURES = 12, 4
SEVERITY_K = 3
S_LOW, S_HIGH = 1.5, 3.0

STATE_CODE = {"accept": 0, "accept_with_reduced_confidence": 1, "abstain": 2}


def fmt_f64(values) -> str:
    return ", ".join(repr(float(v)) for v in np.asarray(values, dtype=np.float64).ravel())


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "envelope_oracle.rs"

    rng = np.random.default_rng(42)
    calibration = rng.normal(size=(N_ROWS, N_FEATURES))
    calibration[:, 0] = calibration[:, 0] * 2.0 + 5.0   # shifted/scaled axis
    calibration[:, 3] = 7.25                            # constant axis: exercises the scale floor

    names = tuple(f"f{i}" for i in range(N_FEATURES))
    reference = fit_shift_reference(calibration, names, severity_k=SEVERITY_K)

    probes = np.array(
        [
            calibration[0],                                  # in-domain row
            calibration.mean(axis=0) + [0.0, 8.0, 0.0, 0.0],  # single-axis shift (axis 1)
            calibration.mean(axis=0) + [4.0, -6.0, 5.0, 0.0],  # multi-axis shift
        ]
    )
    deviations = reference.per_axis_deviation(probes)
    severities = reference.severity(probes)
    attributions = reference.attribution(probes)

    sweep = np.array([0.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    discounts = confidence_discount(sweep, S_LOW, S_HIGH)
    decision = three_state_policy(sweep, S_LOW, S_HIGH)
    degenerate = three_state_policy(sweep, 2.0, 2.0)

    sep_scores = np.array([0.1, 0.4, 0.3, 0.9, 0.8, 1.5])
    sep_levels = np.array([0, 0, 1, 1, 2, 2])
    separation = monotone_separation(sep_scores, sep_levels)

    lines: list[str] = []
    push = lines.append
    push("// AUTO-GENERATED from the ITD reference (itd_research.ood_shift + ood.reference).")
    push("// Do not edit by hand. Regenerate with:")
    push("//   PYTHONPATH=/path/to/itd-simulator python3 gen_envelope_oracle.py <out>")
    push("// Included via `mod envelope_oracle { include!(...) }` in tests/envelope.rs.")
    push("")
    push(f"pub const N_ROWS: usize = {N_ROWS};")
    push(f"pub const N_FEATURES: usize = {N_FEATURES};")
    push(f"pub const SEVERITY_K: usize = {SEVERITY_K};")
    push(f"pub const S_LOW: f64 = {S_LOW!r};")
    push(f"pub const S_HIGH: f64 = {S_HIGH!r};")
    push("")
    push("// ---- calibration matrix (row-major) and fitted per-axis statistics ----")
    push(f"pub static CALIBRATION: &[f64] = &[{fmt_f64(calibration)}];")
    push(f"pub static FITTED_MEAN: &[f64] = &[{fmt_f64(reference.base.mean)}];")
    push("/// Population std with the reference floor: `std < 1e-9` is replaced by `1.0`.")
    push(f"pub static FITTED_SCALE: &[f64] = &[{fmt_f64(reference.base.std)}];")
    push("")
    push("// ---- probes: per-axis |z|, top-k severity, arg-max attribution ----")
    push(f"pub static PROBES: &[f64] = &[{fmt_f64(probes)}];")
    push(f"pub static PROBE_DEVIATIONS: &[f64] = &[{fmt_f64(deviations)}];")
    push(f"pub static PROBE_SEVERITIES: &[f64] = &[{fmt_f64(severities)}];")
    attrib = ", ".join(str(int(a)) for a in attributions)
    push(f"pub static PROBE_ATTRIBUTIONS: &[usize] = &[{attrib}];")
    push("")
    push("// ---- three-state policy over a severity sweep ----")
    push(f"pub static SWEEP: &[f64] = &[{fmt_f64(sweep)}];")
    push(f"pub static SWEEP_DISCOUNTS: &[f64] = &[{fmt_f64(discounts)}];")
    states = ", ".join(str(STATE_CODE[s]) for s in decision.states)
    push("/// State codes: 0 accept, 1 accept-with-reduced-confidence, 2 abstain.")
    push(f"pub static SWEEP_STATES: &[usize] = &[{states}];")
    push(f"pub static SWEEP_CONFIDENCES: &[f64] = &[{fmt_f64(decision.confidence)}];")
    degenerate_states = ", ".join(str(STATE_CODE[s]) for s in degenerate.states)
    push("/// Degenerate band `s_high <= s_low` collapses to a hard step at `s_low`.")
    push(f"pub static DEGENERATE_STATES: &[usize] = &[{degenerate_states}];")
    push(f"pub static DEGENERATE_CONFIDENCES: &[f64] = &[{fmt_f64(degenerate.confidence)}];")
    push("")
    push("// ---- monotone-separation rank agreement ----")
    push(f"pub static SEP_SCORES: &[f64] = &[{fmt_f64(sep_scores)}];")
    sep_levels_s = ", ".join(str(int(v)) for v in sep_levels)
    push(f"pub static SEP_LEVELS: &[usize] = &[{sep_levels_s}];")
    push(f"pub const SEPARATION: f64 = {separation!r};")
    push("")

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"wrote {out_path}")
    print(f"  fitted scale: {reference.base.std}")
    print(f"  severities: {severities}")
    print(f"  attributions: {attributions}")
    print(f"  sweep states: {decision.states}")
    print(f"  separation: {separation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
