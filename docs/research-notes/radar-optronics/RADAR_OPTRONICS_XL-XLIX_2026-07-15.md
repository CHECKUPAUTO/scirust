# SciRust — Radar & Optronics, Blocks 40–49 (2026-07-15)

A massive parallel push: ten self-contained radar and EO/IR capability modules,
each built and oracle-verified in an isolated worktree by a dedicated agent, then
integrated centrally with a single full `test` / `clippy -D warnings` / `fmt` pass
over both crates. This is the "archive-cleanly" batch — broadening coverage across
the whole detect → track → classify → measure chain in one sweep.

## Radar — `scirust-signal::radar`

- **`cfar_variants`** — greatest-of / smallest-of / trimmed-mean CFAR (`go_cfar`,
  `so_cfar`, `tm_cfar`). GO holds the threshold up at a clutter edge (suppressing the
  false alarms cell-averaging emits); SO ignores an interferer in one half-window so a
  weak neighbour survives; trimmed-mean censors a bounded number of interferers.
- **`binary_integration`** — M-of-N binary integration (`binomial_pmf`,
  `binomial_sf_ge`, `integrated_pfa`, `integrated_pd`, `optimal_m`). Closed-form
  detection/false-alarm probabilities via the binomial; M>1 slashes the false-alarm
  rate. `optimal_m ≈ 1.5·√N`.
- **`crt_prf`** — multi-PRF range disambiguation by the Chinese Remainder Theorem
  (`egcd`, `mod_inverse`, `crt_pair`, `resolve_range`, `combined_ambiguity`).
  Reconstructs a range that folds differently under coprime PRFs; combined unambiguous
  span = the modulus product.
- **`costas`** — Costas frequency-hop arrays (`welch_costas`, `is_costas`,
  `max_coincidence`, `primitive_root`). Welch construction from a primitive root; the
  displacement-vector test proves the ideal-thumbtack (≤1 coincidence) property.
- **`propagation`** — two-ray (flat-earth) multipath: `propagation_factor` =
  `2|sin(2π·h_a·h_t/(λR))|` (oscillating 0…2), `power_factor = F⁴`, `first_null_range`,
  `path_length_difference`, and the multipath phase difference (re-exported as
  `multipath_phase_difference` to avoid clashing with the interferometer's).
- **`dbs`** — Doppler beam sharpening (`azimuth_doppler = (2v/λ)cosθ`,
  `doppler_gradient`, `dbs_azimuth_resolution = λ/(2vT|sinθ|)`, `sharpening_ratio`).
  Cross-range resolution from the in-beam Doppler gradient; +∞ at boresight, finest
  broadside.

## Optronics — `scirust-vision`

- **`nuc`** — two-point non-uniformity correction for an IR focal plane
  (`two_point_coeffs`, `apply_nuc`, `fixed_pattern_noise`). Per-pixel gain/offset from
  two calibration scenes; a third scene reads uniform after correction.
- **`lidar`** — laser time-of-flight and CW phase ranging (`range_from_time_of_flight`,
  `time_of_flight`, `range_resolution`, `range_from_phase`, and pulsed/CW
  unambiguous-range limits).
- **`centroid`** — sub-pixel spot/star centroiding (`center_of_gravity`,
  `thresholded_centroid`, `windowed_centroid`) for EO/IR pointing: exact on a lone
  pixel, weighted between split pixels, pedestal-robust when thresholded.
- **`zernike`** — Noll-normalized Zernike wavefront aberrations (`defocus`,
  `astigmatism`, `coma`, `spherical`), `rms_wavefront_error` (quadrature sum), and the
  Maréchal Strehl `strehl_marechal = exp(−(2π·σ)²)` (σ in waves) — with the numerically
  verified unit-RMS / orthogonality oracles and the λ/14 → Strehl≈0.8 criterion.

## Verification

- `cargo test -p scirust-signal` — **317 tests green** (+44).
- `cargo test -p scirust-vision` — **95 tests green** (+28); +72 oracle tests total.
- `cargo clippy -p scirust-signal -p scirust-vision --all-targets -- -D warnings` — clean.
- `cargo fmt -- --check` — clean.

## Method

Each module was implemented by an independent agent in its own git worktree, verified
there (`test` + `clippy` + `fmt`), and returned as a complete file plus wiring
metadata. Integration was done centrally in one pass: files written, modules wired
(`pub mod` + re-exports, one name aliased to avoid a clash), then the whole workspace
re-verified — catching and fixing the two `identity_op` lints that a per-`--lib`
agent check had missed. The result is a coherent, uniformly-styled, oracle-tested
expansion landed as a single batch.
