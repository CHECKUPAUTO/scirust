# SciRust — Radar & Optronics, Block 38 (2026-07-12)

Back to the optronics side. The EO/IR chain already had PSF/MTF/deconvolution
(image quality), radiometry and NETD/MRTD (sensitivity), and Beer–Lambert
atmosphere (path *attenuation*). This block adds the other half of the atmosphere's
effect: **turbulence**, which even on a perfectly transmitting path *blurs* a
long-range image and *fades* a laser beam. It is the physics that decides whether a
big EO/IR telescope actually resolves better than a small one, and whether an
adaptive-optics system is needed.

## What shipped — `scirust-vision::turbulence`

Refractive-index turbulence, its strength set by the structure constant `Cn²`,
randomises the wavefront. The standard closed-form descriptors:

- **`fried_parameter(Cn², λ, L)`** = `(0.423·k²·Cn²·L)^(−3/5)` — the coherence
  length `r₀`, the aperture over which the wavefront stays coherent (`k = 2π/λ`).
- **`seeing_angle(λ, r₀)`** ≈ `0.98·λ/r₀` — the long-exposure blur that replaces
  the diffraction limit `λ/D` once `D > r₀`.
- **`strehl_ratio(D, r₀)`** = `[1 + (D/r₀)^(5/3)]^(−6/5)` — the peak-intensity
  fraction an uncorrected aperture keeps.
- **`greenwood_frequency(v, r₀)`** = `0.426·v/r₀` — the temporal bandwidth an
  adaptive-optics loop must run at.
- **`degrees_of_freedom(D, r₀)`** = `(D/r₀)²` — the count of turbulence cells /
  corrector actuators across the aperture.
- **`rytov_variance(Cn², λ, L)`** = `1.23·Cn²·k^(7/6)·L^(11/6)` — the
  weak-turbulence scintillation (intensity twinkling) that fades a laser beam.

Dependency-free.

## The oracles

- **Fried parameter** matches the closed form (a realistic ~2 cm `r₀` at visible
  wavelengths, 1 km path) and scales as **λ^(6/5)**, shrinking with stronger `Cn²`
  and longer path; no turbulence ⇒ infinite coherence.
- **Seeing vs diffraction** — `seeing = 0.98·λ/r₀`, and for `D ≫ r₀` it exceeds the
  diffraction limit `λ/D` (turbulence-limited); it collapses to 0 as `r₀ → ∞`.
- **Strehl ratio** — → 1 for `D ≪ r₀`, exactly `2^(−6/5)` at `D = r₀`,
  monotonically decreasing in `D/r₀`, bounded in (0, 1]; 1 with no turbulence.
- **Greenwood frequency** — ∝ wind speed and ∝ 1/r₀.
- **Rytov variance** — ∝ `Cn²`, ∝ `L^(11/6)`, ∝ `k^(7/6)` (shorter wavelengths
  scintillate more).
- **Degrees of freedom** `(D/r₀)²` and degenerate-input guards (no NaN).

## Verification

- `cargo test -p scirust-vision` — **67 tests green** (+7).
- `cargo clippy -p scirust-vision --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-vision -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-vision --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The EO/IR optronics chain is now complete from focal plane to atmosphere: image
quality (PSF/MTF/deconvolution), sensitivity (radiometry, NETD/MRTD), small-target
detection (image CFAR), path attenuation (Beer–Lambert), and now path turbulence
(Fried/seeing/Strehl/Greenwood/Rytov). Paired with the full radar suite — search,
track, angle estimation, GMTI/STAP, SAR imaging, and the polyphase/CAZAC waveform
library (block 37) — the 38-block program remains a physically-grounded,
closed-form-oracle-tested capability across both the radar and optronics
modalities.
