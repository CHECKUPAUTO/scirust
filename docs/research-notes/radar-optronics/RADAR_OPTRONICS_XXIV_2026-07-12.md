# SciRust — Radar & Optronics, Block 24 (2026-07-12)

Follow-up deepening on the optronics side. The `optics` module characterises an
EO/IR imager's *spatial* response (PSF, MTF); this block adds its *radiometric*
and *sensitivity* counterparts — the physics that decides how small a
temperature difference a thermal sensor can actually see, ending in the two
canonical IR-imager specs, **NETD** and **MRTD**.

## What shipped — `scirust-vision::radiometry`

**Radiometry**

- **`planck_radiance(λ, T)`** — blackbody spectral radiance (Planck's law) and
  **`planck_radiance_dt`** its analytic temperature derivative
  `∂L/∂T = L·x·eˣ/(T(eˣ−1))`.
- **`radiant_exitance(T) = σT⁴`** (Stefan–Boltzmann) and
  **`exitance_derivative(T) = 4σT³`**.
- **`peak_wavelength(T) = b/T`** (Wien's displacement law).
- **`band_radiance`** / **`thermal_contrast`** — in-band radiance and its
  temperature derivative `∫∂L/∂T dλ` by quadrature (the contrast that drives IR
  sensitivity).

**Sensitivity**

- **`netd(...)`** — noise-equivalent temperature difference, the ΔT giving a
  signal equal to the detector noise:
  `NETD = 4F²√Δf / (π·√A_d·τ_o·D*·(∂L/∂T)_band)`, from f-number, detector area,
  noise bandwidth, specific detectivity `D*`, optical transmission, and in-band
  thermal contrast.
- **`mrtd(netd, mtf, k)`** — minimum resolvable temperature difference
  `MRTD = k·NETD/MTF`, the thermal-sensitivity/resolution trade-off that folds
  the NETD together with the `optics` MTF — the headline thermal-imager spec.

## The oracles

- **Planck integral recovers Stefan–Boltzmann** — the headline cross-check:
  integrating spectral radiance over the whole spectrum and multiplying by π
  reproduces `σT⁴` to 1e-3.
- **Exitance and its derivative** match `σT⁴` / `4σT³` and a finite difference.
- **Wien peak shifts as 1/T** — halves when T doubles, and the Planck curve
  peaks there.
- **`∂L/∂T` matches a finite difference** and is positive.
- **Thermal contrast is positive and grows with temperature** (LWIR band).
- **NETD obeys its scaling laws** — `∝ F²`, `∝ 1/D*`, `∝ 1/contrast`,
  `∝ 1/√A_d`, and diverges when contrast → 0.
- **MRTD rises as the MTF falls** — `k·NETD` at full MTF, doubling as the MTF
  halves, diverging at the MTF cutoff.

## Verification

- `cargo test -p scirust-vision` — **54 tests green** (+7).
- `cargo clippy -p scirust-vision --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-vision -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-vision --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The optronics module now spans the full EO/IR imaging chain: **radiometry**
(Planck/Stefan–Boltzmann/Wien) → **sensitivity** (NETD/MRTD) → **optics**
(PSF/MTF, deconvolution) → **detection** (image-domain CFAR) → the shared
tracking stack. Together with the radar front-end and the tracking deepenings
(blocks 1–23), the program is a complete sensor-to-track suite across the radar
and EO/IR modalities, grounded end to end in closed-form physics oracles.
