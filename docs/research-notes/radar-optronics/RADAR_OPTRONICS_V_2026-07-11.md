# SciRust — Radar & Optronics, Block 5 (2026-07-11)

Follow-up to blocks 1–4 (the single-channel pulse-Doppler chain: waveform →
pulse compression → ambiguity → range-Doppler → MTI → CFAR). This block opens
the **multi-channel** track — array processing for **direction of arrival
(DOA)**, the angle stage that both reference projects add (OpenRadar's AoA;
AERIS's phased array).

## What shipped — `scirust-signal::radar::beamform`

Direction finding with a uniform linear array (ULA). A plane wave from angle `θ`
(from broadside) reaches element `m` at phase `2π·d·m·sin θ`, the **steering
vector** `a(θ)`.

- **`steering_vector(num_sensors, spacing, angle)`** — `a(θ)`; unit magnitude,
  all ones at broadside.
- **`beamform_spectrum(snapshots, spacing, angles)`** — the conventional
  (delay-and-sum / Bartlett) beamformer: the average output power
  `mean_t |aᴴ(θ)·x[t]|²` scanned over a steering-angle grid. Its peaks are the
  source directions.
- **`estimate_doa(spectrum, angles)`** — the peak angle (single-source DOA).

Dependency-free (built on `Complex`).

## The oracles

- **Peak at the source.** A single plane wave from `θ0` yields a beamformer
  spectrum whose maximum is at `θ0` (within the angle-grid resolution).
- **Two sources seen.** Two well-separated sources both stand well above an
  empty steering direction — the array separates them in angle.
- **Steering vector.** All ones at broadside, unit magnitude everywhere.

## Verification

- `cargo test -p scirust-signal` — **138 tests + 1 doctest green** (+4).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.

## Where the radar track stands

- Single-channel pulse-Doppler chain: **complete** (blocks 1–4).
- Angle / array processing: **started** (this block — the conventional
  beamformer). Next in this track: **high-resolution DOA** — MVDR/Capon (needs a
  Hermitian matrix inverse) and MUSIC/ESPRIT (needs an eigendecomposition, which
  `scirust-solvers` already provides), reusing these steering vectors.
- Then the **FMCW** track (OpenRadar's dechirp/beat-frequency + range-Doppler
  cube), **detection → track** (clustering + the existing `scirust-estimation`
  Kalman/IMM), and the wider program's optronics / optical-imaging /
  optoelectronic-device pieces.
