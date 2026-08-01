# SciRust — Radar & Optronics, Block 6 (2026-07-11)

Follow-up to block 5 (conventional / delay-and-sum beamforming). This block adds
**high-resolution** direction finding — the MVDR (Capon) beamformer, which
resolves sources closer together than the array beamwidth, the property that
makes multi-channel radar worth the extra hardware.

## What shipped — `scirust-signal::radar::doa`

The conventional beamformer's resolution is limited by the array beamwidth
(≈ 2/M). MVDR instead picks, per look direction, the weights that minimise total
output power subject to unit gain toward that direction — nulling everything
else — giving a far sharper spatial spectrum.

- **`covariance(snapshots)`** — the `M × M` Hermitian sample covariance
  `R = (1/T)·Σ_t x[t]·x[t]ᴴ`.
- **`mvdr_spectrum(snapshots, spacing, angles, loading)`** — the Capon spectrum
  `P(θ) = 1 / (aᴴ(θ)·R⁻¹·a(θ))`, with diagonal loading for stability. Built on a
  from-scratch **complex matrix inverse** (Gauss–Jordan with partial pivoting) —
  no new dependency.

## The oracles

- **Peak at the source** — a single source is located by the MVDR peak.
- **Sub-beamwidth resolution** — two sources 6° apart, inside a 10-element
  array's ≈ 11° beamwidth, are *resolved* by MVDR (the midpoint between them is a
  valley below both peaks) while the conventional Bartlett beamformer *merges*
  them (the midpoint is not a valley). This is the headline high-resolution
  property.
- **Hermitian covariance** — `R[i][j] = conj(R[j][i])`.

## Verification

- `cargo test -p scirust-signal` — **141 tests + 1 doctest green** (+3).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.

## Where the radar track stands

- Single-channel pulse-Doppler chain: **complete** (blocks 1–4).
- Array / angle processing: conventional beamformer (block 5) + **MVDR/Capon
  high-resolution DOA** (this block). Remaining in this track: **MUSIC/ESPRIT**
  (subspace methods needing an eigendecomposition of `R` — a small local
  Hermitian eigensolver, or reuse `scirust-solvers`), on the same covariance.
- Then the **FMCW** track, **detection → track** (clustering + existing
  `scirust-estimation` Kalman/IMM), and the wider program's optronics /
  optical-imaging / optoelectronic-device pieces.
