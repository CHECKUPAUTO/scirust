# SciRust — Radar & Optronics, Block 8 (2026-07-11)

Follow-up to block 7 (FMCW / mmWave). This block adds **MUSIC** subspace
direction finding, which **closes the angle-processing track**: conventional
beamformer (block 5) → MVDR/Capon (block 6) → MUSIC (this block), the standard
low-to-high-resolution DOA progression.

## What shipped — `scirust-signal::radar::music`

MVDR sharpens the conventional beamformer but its resolution still degrades as
sources approach. MUSIC is a *subspace* method: it eigendecomposes the array
covariance, splits the eigenvectors into a **signal subspace** (the `d` largest
eigenvalues — signal plus noise) and a **noise subspace** (the rest), and
exploits that every source steering vector is orthogonal to the noise subspace.

- **`music_spectrum(snapshots, spacing, angles, num_sources)`** — the MUSIC
  spatial spectrum `P(θ) = 1 / ‖Eₙᴴ·a(θ)‖²`, where `Eₙ` is the noise subspace
  (eigenvectors of the `M − num_sources` smallest covariance eigenvalues). It
  spikes — in the noise-free limit, to infinity — exactly at the source
  directions, with resolution set by snapshot count and SNR rather than the
  array aperture. `num_sources` is clamped to `1..=M-1`.

The engine is a from-scratch **complex-Hermitian eigensolver** (private
`hermitian_eig`): cyclic Jacobi rotations that first rotate away the phase of
each off-diagonal element — leaving a real symmetric 2×2 block — then apply the
standard real Jacobi rotation. Dependency-free, and reusable for ESPRIT later.
Built on `radar::doa::covariance`.

## The oracles

- **Eigensolver correctness** — on a fixed 3×3 Hermitian matrix: `V·diag(λ)·Vᴴ`
  reconstructs `A`; the eigenvectors are orthonormal (`Vᴴ V = I`); the
  eigenvalues are real and sum to the trace.
- **Single-source peak** — MUSIC locates one source at its direction.
- **Sub-beamwidth resolution** — two sources 6° apart, inside a 10-element
  array's ≈ 11° beamwidth, are resolved (the midpoint between them is a valley
  below both peaks).
- **Degenerate input** — an empty array, or a single-element array (no noise
  subspace), yields an all-zero spectrum.

## Verification

- `cargo test -p scirust-signal` — **151 tests + 1 doctest green** (+4).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.

## Where the radar track stands

- Single-channel pulse-Doppler chain: **complete** (blocks 1–4).
- Array / angle processing: **complete** — conventional beamformer (5),
  MVDR/Capon (6), MUSIC subspace (this block). A local complex-Hermitian
  eigensolver now exists in-crate, so **ESPRIT** (rotational-invariance DOA)
  could follow cheaply if wanted.
- FMCW / mmWave ranging + range-Doppler cube (block 7).
- Remaining: **detection → track** (cluster CFAR detections, then reuse
  `scirust-estimation`'s Kalman/IMM), then the wider program's optronics /
  optical-imaging / optoelectronic-device pieces.
