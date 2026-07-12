//! Radar detection statistics — Swerling target fluctuation and detection
//! probability.
//!
//! CFAR ([`super::cfar`]) sets the detection *threshold* to hold a chosen
//! false-alarm rate; this module answers the complementary question — given that
//! threshold and a target's signal-to-noise ratio, what is the **probability of
//! detection** `P_d`? The answer depends on how the target's radar cross-section
//! *fluctuates* scan to scan, captured by the classic **Swerling** cases. A
//! steady (non-fluctuating) target needs the least SNR; a Rayleigh-fluctuating
//! **Swerling I** target needs more — the *fluctuation loss* — because a
//! deep fade can drop it below threshold.
//!
//! Provided in closed form: the single-pulse detection threshold, the Swerling I
//! `P_d`, and **Albersheim's equation** (the standard closed-form approximation
//! for the non-fluctuating case) both forward (required SNR for a target `P_d`)
//! and inverted (`P_d` from a given SNR). Dependency-free.

/// The single-pulse square-law **detection threshold** (normalised to the noise
/// power) that yields false-alarm probability `pfa`: `V_T = −ln(P_fa)`, since a
/// square-law envelope's noise exceeds `V_T` with probability `e^{−V_T}`. `pfa`
/// is clamped to `(0, 1)`.
pub fn single_pulse_threshold(pfa: f64) -> f64 {
    -pfa.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON).ln()
}

/// The **Swerling I** probability of detection for a single pulse:
/// `P_d = P_fa^{1/(1+SNR)}`, for a slowly Rayleigh-fluctuating target at linear
/// signal-to-noise ratio `snr`. At `snr = 0` this is `P_fa` (no signal), rising
/// to 1 as the SNR grows.
pub fn swerling1_pd(snr: f64, pfa: f64) -> f64 {
    let pfa = pfa.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    pfa.powf(1.0 / (1.0 + snr.max(0.0)))
}

/// The linear single-pulse SNR a **Swerling I** target needs to reach detection
/// probability `pd` at false-alarm rate `pfa`, inverting [`swerling1_pd`]:
/// `SNR = ln(P_fa)/ln(P_d) − 1`.
pub fn swerling1_required_snr(pd: f64, pfa: f64) -> f64 {
    let pfa = pfa.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    let pd = pd.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    pfa.ln() / pd.ln() - 1.0
}

/// **Albersheim's equation**: the single-pulse SNR (in **dB**) a non-fluctuating
/// (steady) target needs for detection probability `pd` at false-alarm rate
/// `pfa` after non-coherent integration of `n_pulses`:
///
/// `SNR_dB = −5·log₁₀N + (6.2 + 4.54/√(N+0.44))·log₁₀(A + 0.12·A·B + 1.7·B)`,
///
/// with `A = ln(0.62/P_fa)` and `B = ln(P_d/(1−P_d))`. Accurate to a few tenths
/// of a dB for `10⁻⁷ ≤ P_fa ≤ 10⁻³`, `0.1 ≤ P_d ≤ 0.9`, and `1 ≤ N ≤ 8096`.
pub fn albersheim_snr(pd: f64, pfa: f64, n_pulses: usize) -> f64 {
    let pfa = pfa.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    let pd = pd.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    let n = n_pulses.max(1) as f64;
    let a = (0.62 / pfa).ln();
    let b = (pd / (1.0 - pd)).ln();
    let inner = a + 0.12 * a * b + 1.7 * b;
    -5.0 * n.log10() + (6.2 + 4.54 / (n + 0.44).sqrt()) * inner.log10()
}

/// The non-fluctuating probability of detection from a given SNR (in **dB**)
/// after integrating `n_pulses`, inverting [`albersheim_snr`]: recover the inner
/// term, solve `inner = A + (0.12·A + 1.7)·B` for `B`, then `P_d = 1/(1 + e^{−B})`.
pub fn albersheim_pd(snr_db: f64, pfa: f64, n_pulses: usize) -> f64 {
    let pfa = pfa.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    let n = n_pulses.max(1) as f64;
    let a = (0.62 / pfa).ln();
    let coeff = 6.2 + 4.54 / (n + 0.44).sqrt();
    let inner = 10.0_f64.powf((snr_db + 5.0 * n.log10()) / coeff);
    let b = (inner - a) / (0.12 * a + 1.7);
    1.0 / (1.0 + (-b).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db_to_lin(db: f64) -> f64 {
        10.0_f64.powf(db / 10.0)
    }

    #[test]
    fn single_pulse_threshold_matches_the_false_alarm_law() {
        // P_fa = e^{−V_T} ⇒ V_T = −ln P_fa; a tighter P_fa needs a higher
        // threshold.
        let pfa = 1e-6;
        let vt = single_pulse_threshold(pfa);
        assert!((vt - (-pfa.ln())).abs() < 1e-12);
        assert!((-vt.exp().recip().ln() - vt).abs() < 1e-9); // round-trip sanity
        assert!(single_pulse_threshold(1e-8) > single_pulse_threshold(1e-4));
    }

    #[test]
    fn swerling1_pd_limits_and_monotonicity() {
        let pfa = 1e-6;
        // No signal ⇒ P_d = P_fa.
        assert!((swerling1_pd(0.0, pfa) - pfa).abs() < 1e-12);
        // Rises monotonically with SNR toward 1.
        let (a, b, c) = (
            swerling1_pd(10.0, pfa),
            swerling1_pd(100.0, pfa),
            swerling1_pd(1e6, pfa),
        );
        assert!(a < b && b < c && c < 1.0 && c > 0.99);
        // Inversion round-trips.
        let snr = swerling1_required_snr(0.8, pfa);
        assert!(
            (swerling1_pd(snr, pfa) - 0.8).abs() < 1e-9,
            "{}",
            swerling1_pd(snr, pfa)
        );
    }

    #[test]
    fn albersheim_forward_and_inverse_round_trip() {
        for &(pd, pfa, n) in &[
            (0.9, 1e-6, 1),
            (0.5, 1e-4, 10),
            (0.8, 1e-7, 4),
            (0.3, 1e-3, 1),
        ]
        {
            let snr_db = albersheim_snr(pd, pfa, n);
            let recovered = albersheim_pd(snr_db, pfa, n);
            assert!(
                (recovered - pd).abs() < 1e-6,
                "pd {pd} -> {snr_db} dB -> {recovered}"
            );
        }
    }

    #[test]
    fn albersheim_pd_rises_with_snr_and_integration_lowers_the_required_snr() {
        let pfa = 1e-6;
        assert!(albersheim_pd(5.0, pfa, 1) < albersheim_pd(15.0, pfa, 1));
        // A looser false-alarm rate raises P_d at fixed SNR.
        assert!(albersheim_pd(10.0, 1e-6, 1) < albersheim_pd(10.0, 1e-3, 1));
        // Integrating more pulses lowers the SNR needed for the same P_d.
        assert!(albersheim_snr(0.9, pfa, 10) < albersheim_snr(0.9, pfa, 1));
        for snr in [2.0, 8.0, 14.0]
        {
            let pd = albersheim_pd(snr, pfa, 1);
            assert!((0.0..=1.0).contains(&pd));
        }
    }

    #[test]
    fn swerling1_fluctuation_loss_exceeds_the_steady_target() {
        // The classic fluctuation loss: a Swerling I target needs more SNR than a
        // steady one to reach the same high P_d.
        let (pd, pfa) = (0.9, 1e-6);
        let steady_lin = db_to_lin(albersheim_snr(pd, pfa, 1));
        let swer1_lin = swerling1_required_snr(pd, pfa);
        assert!(
            swer1_lin > steady_lin,
            "swerling {swer1_lin} vs steady {steady_lin}"
        );
        // And the gap (fluctuation loss) is several dB, as expected at P_d = 0.9.
        let loss_db = 10.0 * (swer1_lin / steady_lin).log10();
        assert!(loss_db > 3.0, "fluctuation loss {loss_db} dB");
    }
}
