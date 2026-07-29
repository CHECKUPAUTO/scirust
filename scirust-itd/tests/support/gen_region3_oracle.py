#!/usr/bin/env python3
"""Generate `region3_oracle.rs` from the ITD reference implementation.

The reference is the Python `itd_research.mission8.vortex_regions` module of the
ITD research simulator (the same repository whose V29 engine generated
`oracle_data.rs`).  Every value emitted here is produced by the ORIGINAL Python
functions -- never re-derived in this script -- so the Rust `region3` module is
validated against the reference implementation as an oracle, matching the
crate's standing acceptance discipline.

Usage (needs the itd-simulator repository on PYTHONPATH):

    PYTHONPATH=/path/to/itd-simulator python3 gen_region3_oracle.py region3_oracle.rs

Determinism: fixed literal fields only -- no RNG anywhere.
"""

from __future__ import annotations

import sys

import numpy as np

from itd_research.mission8.vortex_regions import (
    core_count_series,
    detect_regions,
    detect_topology_events,
    iou,
    dice,
    label_components_3d,
    max_nearest_distance,
)

# Distinct dims on purpose: any axis-order transposition in the port changes the
# flat layout and every downstream value (the "axis permutation" failure mode
# the ITD Mission 8 oracles exist to catch).
NZ, NY, NX = 4, 5, 6

THRESHOLD = 0.3
MIN_CELLS = 2


def gaussian_blob(cz: float, cy: float, cx: float, width: float) -> np.ndarray:
    z, y, x = np.meshgrid(
        np.arange(NZ, dtype=np.float64),
        np.arange(NY, dtype=np.float64),
        np.arange(NX, dtype=np.float64),
        indexing="ij",
    )
    r2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
    return np.exp(-r2 / (2.0 * width * width))


def merger_frame(separation: float) -> np.ndarray:
    half = separation / 2.0
    return gaussian_blob(1.5, 2.0, 2.5 - half, 0.55) + gaussian_blob(1.5, 2.0, 2.5 + half, 0.55)


def fmt_f64_slice(values: np.ndarray) -> str:
    return ", ".join(repr(float(v)) for v in values.ravel())


def fmt_usize_slice(values) -> str:
    return ", ".join(str(int(v)) for v in np.asarray(values).ravel())


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "region3_oracle.rs"

    # Two well-separated blobs: the primary labelling/records case.
    field_two = merger_frame(3.0)
    labels_two, records_two = detect_regions(
        field_two, threshold=THRESHOLD, comparison="greater", min_cells=MIN_CELLS
    )

    # A blob straddling the periodic x seam: bounded sees two components,
    # periodic sees one (the Mission 8 wraparound correctness case).
    field_seam = gaussian_blob(1.5, 2.0, 0.0, 0.8) + gaussian_blob(1.5, 2.0, float(NX), 0.8)
    seam_mask = field_seam > THRESHOLD
    _, sizes_seam_bounded = label_components_3d(seam_mask, periodic=False)
    _, sizes_seam_periodic = label_components_3d(seam_mask, periodic=True)

    # Mask metrics between the two-blob mask and a one-cell-shifted variant.
    mask_a = field_two > THRESHOLD
    field_shift = np.roll(field_two, shift=1, axis=2)
    mask_b = field_shift > THRESHOLD
    metric_iou = iou(mask_a, mask_b)
    metric_dice = dice(mask_a, mask_b)
    metric_mnd = max_nearest_distance(mask_a, mask_b)

    # A five-frame merger sequence: counts then persistence-gated events.
    seps = [3.0, 3.0, 0.0, 0.0, 0.0]
    frames = [merger_frame(s) for s in seps]
    counts_seq = core_count_series(
        frames, threshold=THRESHOLD, comparison="greater", min_cells=MIN_CELLS
    )
    events_seq = detect_topology_events(counts_seq, persistence=2)

    # The transient-blip regression case (a recovery must not fire an event) and
    # a sustained change (must fire exactly one merger).
    blip = [2, 2, 1, 2, 2]
    events_blip = detect_topology_events(blip, persistence=2)
    sustained = [2, 2, 2, 1, 1, 1]
    events_sustained = detect_topology_events(sustained, persistence=2)

    kind_code = {"core_merger": 0, "core_split": 1}

    def emit_events(name: str, events) -> str:
        rows = ", ".join(
            f"({kind_code[e.event_type]}, {e.frame_index}, {e.from_count}, {e.to_count})"
            for e in events
        )
        return (
            f"/// `(kind, frame_index, from_count, to_count)`; kind 0 = merger, 1 = split.\n"
            f"pub static {name}: &[(usize, usize, usize, usize)] = &[{rows}];\n"
        )

    lines: list[str] = []
    push = lines.append
    push("// AUTO-GENERATED from the ITD reference (itd_research.mission8.vortex_regions).")
    push("// Do not edit by hand. Regenerate with:")
    push("//   PYTHONPATH=/path/to/itd-simulator python3 gen_region3_oracle.py <out>")
    push("// Included via `mod region3_oracle { include!(...) }` in tests/region3.rs.")
    push("")
    push(f"pub const NZ: usize = {NZ};")
    push(f"pub const NY: usize = {NY};")
    push(f"pub const NX: usize = {NX};")
    push(f"pub const THRESHOLD: f64 = {THRESHOLD!r};")
    push(f"pub const MIN_CELLS: usize = {MIN_CELLS};")
    push("")
    push("// ---- two separated blobs: labelling + region records ----")
    push(f"pub static FIELD_TWO_BLOBS: &[f64] = &[{fmt_f64_slice(field_two)}];")
    push(f"pub static LABELS_TWO_BLOBS: &[usize] = &[{fmt_usize_slice(labels_two)}];")
    push(
        "/// `(label, volume_cells, centroid_z, centroid_y, centroid_x)` per retained region."
    )
    records_rows = ", ".join(
        f"({r.label}, {r.volume_cells}, {r.centroid[0]!r}, {r.centroid[1]!r}, {r.centroid[2]!r})"
        for r in records_two
    )
    push(
        "pub static RECORDS_TWO_BLOBS: &[(usize, usize, f64, f64, f64)] = "
        f"&[{records_rows}];"
    )
    push("")
    push("// ---- periodic-seam wraparound: bounded splits what periodic joins ----")
    push(f"pub static FIELD_SEAM: &[f64] = &[{fmt_f64_slice(field_seam)}];")
    push(f"pub static SEAM_BOUNDED_SIZES: &[usize] = &[{fmt_usize_slice(sizes_seam_bounded)}];")
    push(
        f"pub static SEAM_PERIODIC_SIZES: &[usize] = &[{fmt_usize_slice(sizes_seam_periodic)}];"
    )
    push("")
    push("// ---- mask-overlap metrics (mask B = mask of FIELD_TWO_BLOBS rolled by +1 in x) ----")
    push(f"pub static FIELD_TWO_BLOBS_ROLLED_X: &[f64] = &[{fmt_f64_slice(field_shift)}];")
    push(f"pub const IOU_A_B: f64 = {metric_iou!r};")
    push(f"pub const DICE_A_B: f64 = {metric_dice!r};")
    push(f"pub const MAX_NEAREST_A_B: f64 = {metric_mnd!r};")
    push("")
    push("// ---- five-frame merger sequence: counts + persistence-gated events ----")
    for i, frame in enumerate(frames):
        push(f"pub static FRAME_{i}: &[f64] = &[{fmt_f64_slice(frame)}];")
    push(f"pub static COUNTS_SEQUENCE: &[usize] = &[{fmt_usize_slice(counts_seq)}];")
    push(emit_events("EVENTS_SEQUENCE", events_seq).rstrip())
    push("")
    push("// ---- persistence-rule regression cases ----")
    push(f"pub static COUNTS_BLIP: &[usize] = &[{fmt_usize_slice(blip)}];")
    push(emit_events("EVENTS_BLIP", events_blip).rstrip())
    push(f"pub static COUNTS_SUSTAINED: &[usize] = &[{fmt_usize_slice(sustained)}];")
    push(emit_events("EVENTS_SUSTAINED", events_sustained).rstrip())
    push("")

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"wrote {out_path}")
    print(f"  two-blob records: {len(records_two)}")
    print(f"  seam bounded/periodic component counts: "
          f"{len(sizes_seam_bounded)}/{len(sizes_seam_periodic)}")
    print(f"  sequence counts: {counts_seq}")
    print(f"  sequence events: {[(e.event_type, e.frame_index) for e in events_seq]}")
    print(f"  blip events: {len(events_blip)}  sustained events: {len(events_sustained)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
