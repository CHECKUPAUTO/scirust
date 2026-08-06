# SciRust — Radar & Optronics, Block 3 (2026-07-11)

Follow-up to blocks 1 (pulse-compression waveforms + matched filtering) and 2
(CFAR detection). This block adds the **range-Doppler map** — the 2-D surface on
which CFAR detects, and the piece both reference projects (OpenRadar;
AERIS/plfm_radar) build their detection around.

## What shipped — `scirust-signal::radar::doppler`

After pulse compression each transmitted pulse is a range profile. Stacking `M`
pulses and taking an FFT along **slow-time** (the pulse index) resolves radial
velocity: a target's Doppler shift places it in a Doppler bin, and stationary
clutter collapses to zero Doppler.

- **`doppler_spectrum(slow_time)`** — the FFT of one range bin's slow-time
  sequence (one complex sample per pulse). Bin 0 is zero-Doppler.
- **`range_doppler_map(pulses)`** — from a stack of `M` range-compressed pulses,
  a Doppler FFT per range bin, giving the `N × M` magnitude map
  `map[range][doppler]`.

Both reuse the crate's existing radix-2 FFT (so `M` must be a power of two — an
enforced precondition), no new dependency.

## The oracles

- **Stationary target → zero Doppler.** A target constant across pulses lands in
  Doppler bin 0, with coherent integration gain (magnitude `M`), and empty range
  bins stay empty.
- **Moving target → the matching Doppler bin.** A phase ramp of `k₀` cycles over
  the `M` pulses lands in Doppler bin `k₀` (up to the FFT sign convention),
  sharp and coherent (magnitude `M`), and *not* at zero Doppler — the
  moving-vs-stationary separation that motivates Doppler processing.
- **Preconditions.** Non-power-of-two lengths and ragged pulse stacks are
  rejected (empty result) rather than panicking.

## Verification

- `cargo test -p scirust-signal` — **126 tests + 1 doctest green** (+3).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.

## The detection chain, now complete end-to-end

Blocks 1–3 form the classic pulse-Doppler processing chain:
**LFM/Barker waveform → matched-filter pulse compression → range-Doppler map →
CFAR detection**. Next blocks: the ambiguity function and MTI cancellers; then
beamforming / DOA (MUSIC/ESPRIT, reusing `scirust-solvers` eigendecomposition)
for angle; an FMCW track (OpenRadar's dechirp/beat-frequency style); and the
optronics / optical-imaging / optoelectronic-device pieces of the wider program.
