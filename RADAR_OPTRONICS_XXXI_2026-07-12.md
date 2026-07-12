# SciRust — Radar & Optronics, Block 31 (2026-07-12)

Follow-up deepening on the radar waveform/processing side. A pulse-Doppler radar
samples the scene once per pulse, at the pulse repetition frequency (PRF); that
sampling imposes two hard, coupled limits — one in range, one in velocity — that
this block makes explicit, along with the MTI blind speeds and the range/velocity
folding an ambiguous target undergoes.

## What shipped — `scirust-signal::radar::prf`

- **`unambiguous_range(prf)`** = `c/(2·PRF)` — the maximum range whose echo
  returns before the next pulse.
- **`unambiguous_velocity(λ, prf)`** = `λ·PRF/4` — the maximum radial speed
  sampled without Doppler aliasing; **`max_doppler(prf)`** = `PRF/2` (Nyquist).
- **`blind_speed(n, λ, prf)`** = `n·λ·PRF/2` — the speeds whose Doppler lands on
  `n·PRF`, nulled by an MTI / pulse-Doppler canceller along with the clutter;
  **`velocity_from_doppler(f_d, λ)`** = `λ·f_d/2`.
- **`fold_range(range, prf)`** and **`fold_velocity(v, λ, prf)`** — fold a true
  range or velocity into its measured (aliased) value.

## The oracles

- **Unambiguous range ∝ 1/PRF** — a higher PRF shortens it; infinite at PRF 0.
- **Range–velocity ambiguity product is invariant** — the headline: `R_ua·v_ua =
  cλ/8`, independent of the PRF. Raising the PRF widens the velocity window
  exactly as much as it shrinks the range window — the pulse-Doppler dilemma.
- **Blind speeds are evenly spaced multiples** — `v_blind(0)=0` (clutter DC),
  `v_blind(1)=2·v_ua`, spaced by `λ·PRF/2`.
- **Nyquist Doppler maps to `v_ua`.**
- **Range folding wraps beyond `R_ua`**; **velocity folding aliases beyond
  `±v_ua`** and always lands within the unambiguous interval.

## Verification

- `cargo test -p scirust-signal` — **227 tests green** (+6).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The radar side now spans the full waveform/processing/detection/tracking/
classification stack, with the pulse-Doppler ambiguity limits (this block)
sitting alongside the ambiguity function, range-Doppler, and MTI it constrains.
Together with the complete EO/IR optronics chain, the 31-block program is a
physically-grounded, closed-form-oracle-tested detect–track–classify suite across
both modalities.
