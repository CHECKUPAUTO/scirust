# SciRust — Radar & Optronics, Block 29 (2026-07-12)

Follow-up deepening on the radar side — the link budget that pairs with the
detection statistics. Block 28 gives the SNR a target *needs* for a required
detection probability; this block supplies the other half — the SNR a radar
*delivers* on a target of a given RCS at a given range, and the **maximum
detection range**. It is the radar analog of the EO/IR range budget (block 25),
completing the symmetry between the two modalities.

## What shipped — `scirust-signal::radar::range_equation`

- **`RadarLink`** — a monostatic radar link budget bundling the transmitter,
  antenna, and receiver-noise parameters (peak power, gain, wavelength,
  bandwidth, noise figure, temperature, losses) in SI / linear units.
- **`noise_power`** = `k_B·T·B·F`.
- **`received_power(rcs, range)`** — the monostatic radar equation
  `P_r = P_t·G²·λ²·σ / ((4π)³·R⁴·L)`.
- **`snr_at_range(rcs, range)`** — the delivered single-pulse SNR.
- **`max_range(rcs, snr_min)`** — inverting the radar equation,
  `R_max = [P_t·G²·λ²·σ / ((4π)³·N·L·SNR_min)]^{1/4}`, where `snr_min` is the
  required SNR from [`swerling`](../scirust-signal/src/radar/swerling.rs).

## The oracles

- **Inverse-fourth-power law** — received power drops by `2⁴ = 16` when the range
  doubles; zero at zero range.
- **Noise power is `k_B·T·B·F`.**
- **SNR scales with RCS and falls as `1/R⁴`.**
- **Max-range ↔ SNR consistency** — the headline check: the SNR delivered at the
  computed maximum range equals the required minimum.
- **Range scales as the fourth root of RCS** — 16× the RCS doubles the range.
- **Integrates with Swerling** — the required SNR for a target `P_d`/`P_fa` (via
  Albersheim, block 28) sets the detection range, and demanding a higher `P_d`
  shortens it.

## Verification

- `cargo test -p scirust-signal` — **215 tests green** (+6).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

Both modalities now carry a full range budget: the EO/IR side (radiometry →
atmosphere → NETD → required ΔT at range) and the radar side (range equation →
delivered SNR → Swerling required SNR → detection range). With the front-end,
tracking toolkit, DOA, micro-Doppler classification, detection statistics, and
the complete optics/optronics chain (blocks 1–28), the program is a
physically-grounded, closed-form-oracle-tested detect–track–classify suite —
end to end, across radar and EO/IR — with quantitative range prediction on both.
