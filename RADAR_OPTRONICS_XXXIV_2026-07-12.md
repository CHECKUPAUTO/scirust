# SciRust — Radar & Optronics, Block 34 (2026-07-12)

Completing the single-dwell angle-estimation pair. Block 32 delivered
*amplitude-comparison* monopulse, which reads a target's off-boresight angle from
the **ratio** of two squinted beams. This block adds its complement — a **phase
interferometer**, which reads the same angle from the **phase difference** between
two antenna elements on a baseline `d`. The two are the classic tracking-radar
angle sensors; together they let the suite estimate angle from either amplitude or
phase, single-dwell, well below the beamwidth.

## What shipped — `scirust-signal::radar::interferometer`

A plane wave from angle `θ` reaches the far element with an extra path `d·sin θ`,
i.e. a phase lead `Δφ = 2π·d·sin θ/λ`; measuring `Δφ` and inverting gives the
angle.

- **`phase_difference(θ, d, λ)`** = `2π·d·sin θ/λ` — the interferometric phase
  between the two elements (may exceed `±π` on a wide baseline).
- **`angle_from_phase(Δφ, d, λ)`** = `arcsin(Δφ·λ/(2π·d))` — the inversion, with
  the `arcsin` argument clamped to `[−1, 1]`.
- **`phase_from_signals(near, far)`** = `arg(far·conj(near))` — the phase a receiver
  actually observes from the two element voltages, which aliases when the true
  `Δφ` exceeds `±π`.
- **`unambiguous_angle(d, λ)`** = `arcsin(λ/2d)` — the largest off-boresight angle
  whose phase stays within `±π`. A half-wavelength baseline (`d = λ/2`) is
  unambiguous over the full `±90°`; a wider baseline sharpens the measurement but
  shrinks this field — the interferometer's resolution/ambiguity trade-off.
- **`wrap_phase(φ)`** — wraps a phase into the principal interval `(−π, π]`.

Built directly on the crate's `Complex`; dependency-free.

## The oracles

- **Phase is zero on boresight and odd in `θ`** — the sign of `Δφ` gives the side.
- **Estimate inverts the phase within the unambiguous field** — for a `d = λ/2`
  baseline, `angle_from_phase(phase_difference(θ)) = θ` across the whole `±90°`
  field to 1e-9.
- **Phase recovered from element voltages** — `phase_from_signals` returns a known
  injected phase difference, and a target inside the unambiguous field inverts
  back to its angle.
- **Wide baseline narrows the unambiguous field** — `d = λ/2 ⇒ ±90°`,
  `d = λ ⇒ ±30°`, and a still wider baseline is narrower again.
- **Aliasing outside the field** — a target beyond the unambiguous field produces a
  true phase past `±π`; the wrapped measurement the receiver sees maps to a
  different (aliased) angle — the ambiguity the trade-off is about.
- **Guards** — a degenerate baseline/wavelength returns a safe zero (or the full
  `±90°` field), never a NaN.

## Verification

- `cargo test -p scirust-signal` — **245 tests green** (+7).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The single-dwell angle-estimation repertoire is now complete on both axes:
amplitude-comparison monopulse (block 32) and phase-comparison interferometry
(this block), alongside the beamformer / MVDR-Capon / MUSIC / ESPRIT array-DOA
family. Together with the waveform/ranging chain (LFM, Barker, FMCW,
stepped-frequency), the full detection–tracking stack, and the complete EO/IR
optronics chain, the 34-block program remains a physically-grounded,
closed-form-oracle-tested detect–track–classify suite across both the radar and
optronics modalities.
