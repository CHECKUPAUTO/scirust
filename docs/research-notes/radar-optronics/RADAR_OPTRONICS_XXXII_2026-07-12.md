# SciRust — Radar & Optronics, Block 32 (2026-07-12)

Follow-up deepening on the radar angle-estimation side. The DOA methods so far
(beamforming, MVDR, MUSIC, ESPRIT) estimate angles from an array of snapshots;
this block adds the **monopulse** technique a tracking radar uses to measure a
target's off-boresight angle from a *single* dwell — the workhorse precision
angle measurement of a tracking radar.

## What shipped — `scirust-signal::radar::monopulse`

- **`beam_voltage(θ, θ₀, σ)`** — the voltage gain of a Gaussian beam of width `σ`
  pointed at `θ₀`.
- **`monopulse_ratio(θ, squint, σ)`** — the ratio `Δ/Σ = (A − B)/(A + B)` of the
  difference and sum channels formed from two beams squinted to `±squint`. For
  Gaussian beams this is *exactly* `tanh(θ·squint/σ²)` — zero on boresight, odd,
  and bounded to `(−1, 1)`.
- **`monopulse_slope(squint, σ)`** = `squint/σ²` — the discriminator gain at
  boresight, where the ratio is locally `k_m·θ`.
- **`estimate_angle(ratio, squint, σ)`** = `atanh(ratio)·σ²/squint` — inverts the
  ratio to recover the off-boresight angle.

## The oracles

- **Zero on boresight, and odd** — the difference channel nulls on boresight, and
  the ratio's sign gives the side of boresight.
- **Monotonic and bounded** — the ratio increases with angle and stays inside
  `(−1, 1)`, so it is an unambiguous error signal.
- **Equals the `tanh` closed form** — the forward sum/difference model matches
  `tanh(θ·squint/σ²)` to machine precision.
- **Estimate inverts the ratio exactly** — the headline: forming the ratio for a
  known angle and inverting it recovers the angle, far finer than the beamwidth.
- **Slope is the boresight linearisation** — near boresight `ratio ≈ k_m·θ` and
  the linear estimate `ratio/k_m` recovers a small angle.
- **Degenerate-input guards.**

## Verification

- `cargo test -p scirust-signal` — **233 tests green** (+6).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The radar angle chain now spans array DOA (beamforming, MVDR, MUSIC, ESPRIT) and
single-dwell monopulse, feeding the tracking layer. Together with the full
waveform / detection / classification stack and the complete EO/IR optronics
chain, the 32-block program is a physically-grounded, closed-form-oracle-tested
detect–track–classify suite across both modalities.
