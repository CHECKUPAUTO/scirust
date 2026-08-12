# SciRust — Industrial verticals: implementation plan

Technical complement to `INDUSTRIAL_ROADMAP.md` (which covers adoption /
go-to-market). Here: the **industrial domains** to deepen (present)
and to open (absent), each built on the project's DNA — **guarantees**,
not just accuracy.

## Non-negotiables (= Definition of Done per item)

1. **Pure Rust, zero FFI**; **bit-exact determinism** (seeded PRNG, fixed order).
2. **No claim without a test**: honest oracle or property test — no
   stub. The test measures the claimed guarantee.
3. The differentiator is **always a guarantee**: determinism, conformal
   coverage (without distributional assumptions), certified bound (IBP/CROWN),
   verifiable inference (Freivalds), hash-chained audit, ASIL/SIL safety.
4. **CLI/demo** when relevant; **docs** when a CLI command is added.
5. **8 green gates** (fmt, clippy `-D warnings --all-targets`, build, test, simd,
   aarch64, doc, deny) + `--features wgpu` if relevant. **Commit + push** per item.

---

## Phase 1 — Deepen PdM (quick wins, link the existing)

- **I1 · RUL with conformal intervals** — `scirust-pdm` (`rul` × `conformal_guard`).
  Remaining useful life with a **guaranteed-coverage interval** `[t_low, t_high]`
  without distributional assumptions. Oracle: empirical coverage ≥ 1−α on
  seeded simulated degradation trajectories.

- **I2 · ISO 10816/20816 vibration severity** — `scirust-pdm`.
  Normalized A/B/C/D zones per machine class → compliance verdict.
  Oracle: standard thresholds verified by table cases.

- **I3 · MCSA — motor current signature analysis** — `scirust-signal`/`pdm`.
  Rotor bars / eccentricity / stator faults via sidebands
  `(1±2ks)·f` around the fundamental. Oracle: synthetic signals with known
  faults → sidebands at the correct offset.

## Phase 2 — Shared estimation infrastructure

- **I4 · Deterministic Kalman/EKF/UKF with certified bounds** —
  new `scirust-estimation`. Bit-exact state estimation + **set-membership
  filtering** with proven error envelope. Unlocks I7.
  Oracle: convergence vs known linear system; the certified set always
  contains the true state.

## Phase 3 — OT safety & security (build on the guarantees)

- **I5 · Certified "Simplex" monitor** — `scirust-func-safety` ×
  `scirust-core::nn::ibp` (CROWN). Simple verified controller in fallback, activated
  as soon as the NN output leaves the proven safe envelope. Oracle: over an L∞
  box, the monitor never lets an out-of-envelope output through.

- **I6 · IDS for OT/ICS protocols** — `scirust-ids` × `opcua`/`mqtt`.
  Modbus/DNP3/OPC-UA anomalies with **guaranteed false-alarm rate** (conformal).
  Oracle: empirical FAR ≤ α on normal traffic; injection detected.

## Phase 4 — New verticals (new crates, same DNA)

- **I7 · BMS — battery management** — new `scirust-bms` (uses I4).
  SoC/SoH via EKF, early thermal runaway alert, **conformal SoH
  bounds**. Oracle: SoC tracked on simulated cell model; SoH coverage.

- **I8 · Power grids / smart grid** — new `scirust-grid`.
  Frequency/RoCoF, synchronized phasors, islanding, THD/harmonics. Oracle:
  synthetic grid signals with known frequency/THD.

- **I9 · SHM — structural health monitoring** — new `scirust-shm`.
  Modal analysis (natural frequencies, damping), damage by frequency
  drift, fatigue (Paris law) + conformal RUL. Oracle: known
  mass-spring → exact natural frequencies.

- **I10 · Medical ECG/PPG (IEC 62304)** — new `scirust-biomed`.
  Arrhythmia with **conformal prediction sets** + audit trail. Oracle:
  R-peaks on synthetic ECG; conformal set coverage.

## Phase 5 — Certification proof

- **I11 · DO-178C / rail SIL demonstrator** — `scirust-func-safety`
  + `scirust-runtime`. Bit-exact determinism + verifiable inference +
  hash-chained attestation + ASIL/SIL in a single **reproducible evidence pack**.
  Oracle: bit-identical replay + verified chain + counterexample on falsification.

---

## Execution order

I1→I3 (PdM) → I4 (estimation) → I5,I6 (safety/security) → I7→I10 (verticals)
→ I11 (certification). Each item delivered complete (code + oracle + gates +
commit/push) before the next. Status tracked in `CHANGELOG.md`.
