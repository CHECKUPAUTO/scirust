# SciRust — Radar & Optronics, Block 18 (2026-07-12)

Follow-up deepening. Block 8 added MUSIC, a spectral subspace direction finder;
this block adds **ESPRIT**, its *gridless* sibling — the angle estimates fall
out of an eigenvalue computation with no spectral scan and no grid-resolution
trade-off.

## What shipped — `scirust-signal::radar::esprit`

- **`esprit_doa(snapshots, spacing, num_sources)`** — ESPRIT direction of
  arrival for a uniform linear array. It exploits the array's *rotational
  invariance*: the first `M−1` and last `M−1` elements are the same array
  shifted by one spacing, so each source appears in the second subarray phased
  by `e^{jμ}`, with `μ = 2π·spacing·sin θ` — exactly the steering vector's
  inter-element phase step. The signal subspace inherits that structure, so the
  `d×d` matrix `Ψ` solving the least-squares relation `E₁·Ψ = E₂` between the
  two subarrays' subspaces is similar to `diag(e^{jμ₁},…,e^{jμ_d})`. Its
  eigenvalues therefore carry the angles directly: `sin θ_k = arg(λ_k) /
  (2π·spacing)`. Angles are returned sorted ascending.

Two pieces are reused/added underneath:

- The **Hermitian eigensolver** shared with `radar::music` (now `pub(super)`)
  supplies the signal subspace from the sample covariance.
- A small **from-scratch complex eigensolver** for the non-Hermitian `Ψ`:
  upper-**Hessenberg reduction** by Givens similarities, then the **shifted QR
  algorithm** with a Wilkinson shift (trailing-2×2 eigenvalue nearest the bottom
  corner) and bottom-corner deflation. A 2×2 complex eigenpair (`eig2`, via the
  characteristic quadratic and a principal complex square root) closes each
  deflation and supplies the shift. The subspace least-squares system is solved
  by complex Gauss–Jordan with partial pivoting. All dependency-free.

## The oracles

- **Eigensolver, triangular** — the eigenvalues of an upper-triangular complex
  matrix are exactly its diagonal; recovered to 1e-9. This pins the QR /
  Hessenberg machinery against a closed form independent of ESPRIT.
- **Eigensolver, rotation** — a matrix whose diagonal is `e^{iθ_k}` (plus a
  strictly-upper perturbation, so eigenvalues stay the diagonal) yields
  eigenvalues on the unit circle to 1e-9.
- **Single source** — ESPRIT recovers one angle to well under a degree.
- **Two sources, off grid** — the headline test: two sources at `−7.3°` and
  `+11.8°`, deliberately between any integer-degree grid, are both recovered to
  better than 0.5°, demonstrating the gridless advantage over a scanned
  spectrum.
- **Guards** — an empty covariance or a single-element array returns an empty
  vector.

## Verification

- `cargo test -p scirust-signal` — **168 tests green** (+5).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The full radar (1–10) + optronics (11–17) program is merged; this is another
optional deepening on the signal side. The DOA family now spans conventional
beamforming, MVDR/Capon, MUSIC, and ESPRIT — spectral and gridless subspace
methods both. The remaining optional piece flagged earlier is an IMM/Kalman
tracker upgrade on the tracking side.
