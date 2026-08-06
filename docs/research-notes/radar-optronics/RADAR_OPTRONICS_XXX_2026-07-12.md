# SciRust — Radar & Optronics, Block 30 (2026-07-12)

Follow-up deepening on the radar detection side. CFAR (block 9) and the Swerling
statistics (block 28) both assume a model of the *clutter amplitude
distribution* to set thresholds and predict false-alarm rates. This block
supplies those distributions — and shows why a Rayleigh threshold fails over
spiky clutter.

## What shipped — `scirust-signal::radar::clutter`

- **Rayleigh** — `rayleigh_pdf` / `rayleigh_cdf` / `rayleigh_quantile`: the
  homogeneous, noise-like clutter of the complex-Gaussian return envelope, where
  cell-averaging CFAR is optimal.
- **Weibull** — `weibull_pdf` / `weibull_cdf` / `weibull_quantile`: the workhorse
  spiky-clutter model; its shape `c` tunes the tail (`c = 2` is Rayleigh with
  `b = σ√2`, `c = 1` exponential, `c < 2` spikier — a heavier tail a Rayleigh
  threshold badly under-estimates).
- **Log-normal** — `lognormal_pdf` / `lognormal_cdf`: very spiky clutter with a
  long high-amplitude tail, built on a self-contained **error function** `erf`
  (Abramowitz & Stegun 7.1.26, error < 1.5·10⁻⁷).

## The oracles

- **`erf` matches known values** — `erf(0)=0`, `erf(±∞)=±1`, `erf(0.5)≈0.5205`,
  `erf(1)≈0.8427`, odd symmetry.
- **Rayleigh consistency** — CDF monotone `0 → 1`, the quantile inverts the CDF,
  and the PDF integrates to 1.
- **Weibull(shape 2) recovers Rayleigh** — the headline cross-check: PDF and CDF
  equal `Rayleigh(σ)` for `b = σ√2` to machine precision.
- **Weibull quantile inverts, and lower shape is spikier** — a smaller shape
  gives a heavier tail and a larger 99.9 % quantile; the PDF integrates to 1.
- **Log-normal is a valid distribution** — CDF monotone `0 → 1` with the median
  at `e^μ` (CDF = 0.5), PDF integrates to 1.
- **Negative-support guards.**

## Verification

- `cargo test -p scirust-signal` — **221 tests green** (+6).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The radar detection theory is now complete: the clutter amplitude distribution
(this block) feeds the CFAR threshold (block 9) that holds a false-alarm rate,
the Swerling statistics (block 28) give the detection probability against a
fluctuating target, and the range equation (block 29) turns the required SNR into
a detection range. With the front-end, tracking, DOA, and classification on the
radar side, and the full EO/IR optronics chain, the 30-block program is a
physically-grounded, closed-form-oracle-tested detect–track–classify suite across
both modalities.
