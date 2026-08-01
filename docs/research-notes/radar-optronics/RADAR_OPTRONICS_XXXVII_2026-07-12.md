# SciRust — Radar & Optronics, Block 37 (2026-07-12)

Back to the waveform end of the chain. Block 2 gave the radar its two classic
pulse-compression codes — the linear-FM chirp and the Barker binary codes — but
Barker codes are optimal only up to length 13. This block adds the **polyphase and
CAZAC** families that lift that ceiling: codes of any length with perfect periodic
autocorrelation, and the low-probability-of-intercept (LPI) waveforms of modern
radar.

## What shipped — `scirust-signal::radar::polyphase`

- **`frank_code(n)`** — the Frank code, length `N²`, element `(i, k)` at phase
  `2π·i·k/n`; perfect periodic autocorrelation.
- **`p3_code(L)`** = `exp(j·π·n²/L)` and **`p4_code(L)`** = `exp(j·(π·n²/L − π·n))`
  — the LFM-derived codes, any length, Doppler-tolerant LPI waveforms.
- **`zadoff_chu(L, u)`** — a constant-amplitude zero-autocorrelation (CAZAC)
  sequence, perfect at *any* length for a root `u` coprime to `L` (also the LTE/5G
  reference sequence).
- **`periodic_autocorrelation(code)`** — `R[τ] = Σ code[n]·conj(code[(n+τ) mod L])`,
  the tool the perfect-code property is defined against; a reusable primitive
  complementing the aperiodic matched filter.

Built on the crate's `Complex`; dependency-free.

## The oracles

- **Frank structure** — length `N²`, unit magnitude, phase exactly `2π·i·k/n`.
- **Frank has perfect periodic autocorrelation** — `R[0] = N²`, every other lag
  zero to machine precision (checked for `n = 2..5`).
- **Zadoff-Chu is CAZAC** — unit magnitude and perfect periodic autocorrelation at
  even (12, 16) and prime (7) lengths; a root sharing a factor with the length is
  rejected.
- **P3/P4 are sampled LFM phases** — unit magnitude and the exact quadratic-phase
  formulas.
- **Aperiodic autocorrelation peak is the code length** — the matched-filter
  energy, for all four codes, via the existing `cross_correlate`.
- **Polyphase beats Barker periodically** — Frank's periodic autocorrelation is a
  clean impulse (zero sidelobes) where even Barker-13, the best binary code, carries
  a real periodic sidelobe — the reason perfect polyphase codes are used.
- **Guards** — zero length or a non-coprime/out-of-range Zadoff-Chu root return
  empty.

## Verification

- `cargo test -p scirust-signal` — **266 tests green** (+7).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The waveform library now spans binary (Barker), linear-FM (chirp, FMCW,
stepped-frequency), and polyphase/CAZAC (Frank, P3, P4, Zadoff-Chu) codes —
covering classical pulse compression, wideband synthesis, and the LPI /
perfect-autocorrelation regimes. Together with the full detection, tracking,
angle-estimation, STAP, and SAR-imaging stack and the complete EO/IR optronics
chain, the 37-block program remains a physically-grounded,
closed-form-oracle-tested capability across both the radar and optronics
modalities.
