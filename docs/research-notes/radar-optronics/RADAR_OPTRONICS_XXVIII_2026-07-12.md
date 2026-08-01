# SciRust — Radar & Optronics, Block 28 (2026-07-12)

Follow-up deepening on the radar detection side. CFAR (block 9) sets the
detection *threshold* to hold a chosen false-alarm rate; this block answers the
complementary question — given that threshold and a target's SNR, what is the
**probability of detection** `P_d`? — and how it depends on the target's RCS
fluctuation (the classic **Swerling** cases).

## What shipped — `scirust-signal::radar::swerling`

- **`single_pulse_threshold(pfa)`** — the square-law single-pulse detection
  threshold `V_T = −ln(P_fa)`.
- **`swerling1_pd(snr, pfa) = P_fa^{1/(1+SNR)}`** — the Swerling I (slow,
  Rayleigh-fluctuating target) single-pulse `P_d`, with
  **`swerling1_required_snr`** its inverse (the linear SNR needed for a target
  `P_d`).
- **`albersheim_snr(pd, pfa, n_pulses)`** — **Albersheim's equation** for a
  non-fluctuating (steady) target: the SNR (dB) required after non-coherent
  integration of `N` pulses,
  `−5·log₁₀N + (6.2 + 4.54/√(N+0.44))·log₁₀(A + 0.12·A·B + 1.7·B)`, with
  `A = ln(0.62/P_fa)`, `B = ln(P_d/(1−P_d))`; and **`albersheim_pd`**, its
  inverse (`P_d` from a given SNR).

## The oracles

- **Threshold matches the false-alarm law** — `V_T = −ln P_fa`, higher for a
  tighter `P_fa`.
- **Swerling I limits and monotonicity** — `P_d = P_fa` with no signal, rising
  monotonically to 1; the inversion round-trips.
- **Albersheim forward/inverse round-trip** — `P_d → SNR → P_d` recovers `P_d` to
  1e-6 across four `(P_d, P_fa, N)` points.
- **`P_d` rises with SNR and with `P_fa`**, and **integration lowers the required
  SNR** (more pulses → less SNR for the same `P_d`).
- **Swerling fluctuation loss** — the headline test: a Swerling I target needs
  several dB more SNR than a steady one for `P_d = 0.9`, the classic fluctuation
  penalty.

## Verification

- `cargo test -p scirust-signal` — **209 tests green** (+5).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The radar detection chain is now complete on both ends: CFAR sets the threshold
for a chosen false-alarm rate, and the Swerling / Albersheim statistics predict
the resulting detection probability against steady or fluctuating targets. With
the front-end (blocks 1–10), tracking toolkit (10, 19–22, 26), DOA (18),
classification (27), and the full EO/IR optronics chain (11–17, 23–25), the
program is a physically-grounded, closed-form-oracle-tested detect–track–classify
suite across the radar and EO/IR modalities.
