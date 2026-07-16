//! Multi-stage **detect → treat → re-detect** denoising cascade for mixed noise.
//!
//! Real records rarely carry a single noise process: an industrial acquisition may
//! suffer sensor spikes *and* mains hum *and* a broadband thermal floor at once. A
//! single-family denoiser then faces an impossible trade-off — a median filter wide
//! enough to kill the spikes blurs the signal, a notch leaves the spikes, a Wiener
//! gain treats the spikes and the hum as "signal" because they dominate their bins.
//! The classical engineering answer is a **cascade**: remove the most salient
//! disturbance with the family built for it, look at what is left, and repeat —
//! exactly how a human operator works through a dirty record.
//!
//! [`denoise_cascade`] automates that loop. Each stage runs [`super::classify`] on
//! the *current* signal and applies the family matching the dominant [`NoiseType`]:
//! a detect→treat mapping close to [`super::denoise_auto`]'s, but tuned for
//! re-detection — broadband stages use the short-time Wiener ([`super::stft_wiener_auto`])
//! and level-dependent wavelet variants suited to being re-classified, and periodic
//! interference gets the same multi-line harmonic-aware notching. Three safeguards
//! bound the loop:
//!
//! 1. a hard stage budget (`max_stages`; 4 is a sensible default, see
//!    [`denoise_cascade_auto`]);
//! 2. **loop protection** — if the classifier returns the same [`NoiseType`] twice
//!    in a row the cascade stops rather than re-treating a disturbance it evidently
//!    cannot remove;
//! 3. an **accept-or-roll-back guard** on broadband stages: a Gaussian/colored stage
//!    is committed only if what it removed is noise-like (flat spectrum). If the
//!    increment it removed is *tonal*, the stage was eating a signal tone, so it is
//!    rejected and the pre-stage signal kept. Structured-noise stages (impulsive,
//!    periodic, baseline) remove correlated content by design and are always
//!    committed — the classification gate is what steers them. (A cumulative-whiteness
//!    progress test cannot serve here: removing colored *noise* lowers whiteness just
//!    as removing a *tone* does, so whiteness alone cannot separate a good broadband
//!    stage from a destructive one; the tonality of the increment can.)
//!
//! On pure white noise the loop ends within two stages: the first treats it
//! (Gaussian → short-time Wiener, whose removed increment is flat → committed), and
//! the second classification either repeats (loop protection) or reports
//! [`NoiseType::LowNoise`]. Every committed stage is recorded in
//! [`CascadeResult::stages`], so the full decision trail can be audited after the
//! fact.

use serde::{Deserialize, Serialize};

use super::detect::{NoiseProfile, NoiseType, classify, whiteness_of};
use super::rank::hampel_filter;
use super::stft::stft_wiener_auto;
use super::transform::{ThresholdMode, Wavelet, wavelet_denoise_leveldep};
use super::variational::tikhonov_smooth;

/// One recorded stage of a [`denoise_cascade`] run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeStage {
    /// Which method the stage applied (human-readable, includes the parameters).
    pub method: String,
    /// The dominant [`NoiseType`] the classifier saw *before* this stage ran.
    pub noise_type: NoiseType,
    /// Robust noise level (Donoho MAD) of the signal entering this stage.
    pub noise_std_before: f64,
    /// Whiteness of the cumulative residual (`original − current`) after the stage,
    /// computed like [`super::separate`]'s self-check.
    pub residual_whiteness_after: f64,
}

/// The outcome of a [`denoise_cascade`] run: the final signal plus the full audit
/// trail of what was detected and removed at every stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeResult {
    /// The denoised signal (same length as the input).
    pub output: Vec<f64>,
    /// Every stage that ran, in order.
    pub stages: Vec<CascadeStage>,
    /// The noise characterization of the *final* output — what is left.
    pub final_profile: NoiseProfile,
    /// Whiteness of the total removed component (`original − output`), in [0, 1].
    pub residual_whiteness: f64,
}

/// Apply the single-family treatment matching a classification verdict, on one
/// stage's signal. The mapping is close to [`super::denoise_auto`]'s but tuned for
/// re-detection: Gaussian → short-time Wiener and colored → level-dependent wavelet
/// shrinkage (the plain, un-cycle-spun variant), so a re-classification sees the
/// residual cleanly. Impulsive, periodic and baseline share `denoise_auto`'s
/// treatments exactly.
fn apply_stage(current: &[f64], sample_rate: f64, profile: &NoiseProfile) -> (String, Vec<f64>) {
    match profile.dominant
    {
        NoiseType::Impulsive => (
            "hampel_filter(3, 3.0)".into(),
            hampel_filter(current, 3, 3.0),
        ),
        NoiseType::Periodic =>
        {
            super::notch_detected_lines(current, sample_rate, profile.dominant_freq_hz)
        },
        NoiseType::Baseline =>
        {
            let base = tikhonov_smooth(current, 1.0e4);
            let out: Vec<f64> = current
                .iter()
                .zip(base.iter())
                .map(|(s, b)| s - b)
                .collect();
            ("baseline_removal (signal − tikhonov trend)".into(), out)
        },
        NoiseType::Colored => (
            "wavelet_denoise_leveldep(auto, soft, db4)".into(),
            wavelet_denoise_leveldep(current, 0, ThresholdMode::Soft, Wavelet::Db4),
        ),
        NoiseType::Gaussian => ("stft_wiener_auto".into(), stft_wiener_auto(current)),
        // Unreachable through denoise_cascade (the loop breaks on LowNoise first),
        // but per module convention nothing here may panic.
        NoiseType::LowNoise => ("identity".into(), current.to_vec()),
    }
}

/// **Mixed-noise cascade**: repeatedly classify the current signal and remove its
/// dominant disturbance, until the record is clean, the classifier stalls, or the
/// stage budget runs out. See the module docs for the loop and its safeguards.
///
/// The stage treatments are: impulsive → Hampel(3, 3σ); periodic → harmonic-aware
/// zero-phase IIR notching of up to five [`super::detect_lines`] lines (a
/// [`super::harmonic_stack`] of ≥ 2 lines becomes one
/// [`super::remove_mains_hum_iir`] cascade); baseline → subtract the Tikhonov(1e4)
/// trend; colored → level-dependent wavelet shrinkage; Gaussian → automatic
/// short-time Wiener.
///
/// Degrades gracefully: an empty, too-short or already-clean signal comes back as
/// an unchanged copy with an empty stage list (the classifier reports
/// [`NoiseType::LowNoise`] and the loop never starts). `sample_rate` is used for
/// classification and to place the notch filters; pass `1.0` for normalized units.
pub fn denoise_cascade(signal: &[f64], sample_rate: f64, max_stages: usize) -> CascadeResult {
    let mut current = signal.to_vec();
    let mut stages: Vec<CascadeStage> = Vec::new();
    let mut last_type: Option<NoiseType> = None;

    for _ in 0..max_stages
    {
        let profile = classify(&current, sample_rate);
        if profile.dominant == NoiseType::LowNoise
        {
            break;
        }
        // Loop protection: the same verdict twice in a row means the previous
        // stage did not shift the classification — re-treating would not either.
        if last_type == Some(profile.dominant)
        {
            break;
        }
        let (method, next) = apply_stage(&current, sample_rate, &profile);

        // Accept-or-roll-back guard. Structured-noise stages (impulsive, periodic,
        // baseline) remove *correlated* content by design, so a non-white removed
        // component is expected — accept them. A broadband stage (Gaussian, colored)
        // must remove noise-like content; if what it removed is *tonal*, it is eating
        // a signal tone (the failure the level-dependent wavelet's own docs warn of),
        // so the stage is rejected and the pre-stage signal kept. This is what a
        // cumulative-whiteness progress test cannot do: removing colored noise
        // legitimately lowers whiteness too, so whiteness alone cannot tell a good
        // broadband stage from a destructive one — the tonality of the *increment*
        // can.
        let is_broadband = matches!(profile.dominant, NoiseType::Gaussian | NoiseType::Colored);
        if is_broadband
        {
            let increment: Vec<f64> = current
                .iter()
                .zip(next.iter())
                .map(|(c, x)| c - x)
                .collect();
            if super::detect::is_tonal(&increment, sample_rate)
            {
                break;
            }
        }
        current = next;

        let removed: Vec<f64> = signal
            .iter()
            .zip(current.iter())
            .map(|(s, c)| s - c)
            .collect();
        let w = whiteness_of(&removed);
        stages.push(CascadeStage {
            method,
            noise_type: profile.dominant,
            noise_std_before: profile.noise_std,
            residual_whiteness_after: w,
        });
        last_type = Some(profile.dominant);
    }

    let removed: Vec<f64> = signal
        .iter()
        .zip(current.iter())
        .map(|(s, c)| s - c)
        .collect();
    let residual_whiteness = whiteness_of(&removed);
    let final_profile = classify(&current, sample_rate);
    CascadeResult {
        output: current,
        stages,
        final_profile,
        residual_whiteness,
    }
}

/// [`denoise_cascade`] with the recommended default budget of **4 stages** — enough
/// for the worst realistic mix (spikes + hum + drift + broadband floor) while the
/// safeguards keep typical runs at one or two stages.
pub fn denoise_cascade_auto(signal: &[f64], sample_rate: f64) -> CascadeResult {
    denoise_cascade(signal, sample_rate, 4)
}

#[cfg(test)]
mod tests {
    use super::super::iir::notch_iir;
    use super::super::testutil::{Lcg, snr_db};
    use super::*;
    use core::f64::consts::PI;

    /// The canonical mixed-noise record: a slow sine (the signal) plus aperiodic
    /// impulses (+8, Bernoulli-placed — genuine impulsive noise; a periodic train
    /// would read as a legitimate feature), a strong 50 Hz tone and a white floor, at
    /// fs = 1000 Hz over 2048 samples.
    fn mixed_fixture() -> (Vec<f64>, Vec<f64>) {
        let n = 2048;
        let fs = 1000.0;
        let mut rng = Lcg::new(401);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / fs).sin())
            .collect();
        let mut obs: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &c)| c + 1.0 * (2.0 * PI * 50.0 * i as f64 / fs).sin() + 0.2 * rng.gauss())
            .collect();
        for v in obs.iter_mut()
        {
            if rng.uniform() < 1.0 / 37.0
            {
                *v += 8.0;
            }
        }
        (clean, obs)
    }

    #[test]
    fn cascade_beats_every_single_family_method_on_mixed_noise() {
        let (clean, obs) = mixed_fixture();
        let fs = 1000.0;
        let res = denoise_cascade(&obs, fs, 4);
        assert_eq!(res.output.len(), obs.len());
        assert!(res.output.iter().all(|v| v.is_finite()));
        assert!(
            res.stages.len() >= 2,
            "expected at least 2 stages, got {:?}",
            res.stages
        );
        let s_cascade = snr_db(&clean, &res.output);

        // No single family can treat all three disturbances at once.
        let bin = fs / 2048.0;
        let singles = [
            ("hampel alone", hampel_filter(&obs, 3, 3.0)),
            (
                "notch alone",
                notch_iir(&obs, fs, 50.0, (50.0_f64 * 0.05).max(4.0 * bin)),
            ),
            (
                "wavelet alone",
                wavelet_denoise_leveldep(&obs, 0, ThresholdMode::Soft, Wavelet::Db4),
            ),
        ];
        for (name, out) in singles
        {
            let s_single = snr_db(&clean, &out);
            assert!(
                s_cascade >= s_single + 1.0,
                "cascade {s_cascade:.2} dB must beat {name} {s_single:.2} dB by ≥ 1 dB"
            );
        }
    }

    #[test]
    fn cascade_records_a_meaningful_audit_trail() {
        let (_, obs) = mixed_fixture();
        let res = denoise_cascade(&obs, 1000.0, 4);
        // The first disturbance treated must be the impulses (most salient), and
        // some later stage must treat the 50 Hz tone.
        assert_eq!(res.stages[0].noise_type, NoiseType::Impulsive);
        assert!(
            res.stages
                .iter()
                .any(|s| s.noise_type == NoiseType::Periodic),
            "no periodic stage in {:?}",
            res.stages
        );
        for s in &res.stages
        {
            assert!(s.noise_std_before.is_finite());
            assert!((0.0..=1.0).contains(&s.residual_whiteness_after));
            assert!(!s.method.is_empty());
        }
        assert!((0.0..=1.0).contains(&res.residual_whiteness));
    }

    #[test]
    fn cascade_terminates_on_pure_white_noise_within_two_stages() {
        let mut rng = Lcg::new(403);
        let x: Vec<f64> = (0..1024).map(|_| 0.5 * rng.gauss()).collect();
        // The budget is deliberately generous: the *safeguards*, not the budget,
        // must end the loop.
        let res = denoise_cascade(&x, 1000.0, 8);
        assert!(
            res.stages.len() <= 2,
            "pure white noise took {} stages: {:?}",
            res.stages.len(),
            res.stages
        );
        assert_eq!(res.output.len(), x.len());
        assert!(res.output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cascade_max_stages_is_live() {
        let (_, obs) = mixed_fixture();
        let zero = denoise_cascade(&obs, 1000.0, 0);
        assert_eq!(zero.output, obs, "0 stages must be the identity");
        assert!(zero.stages.is_empty());
        let one = denoise_cascade(&obs, 1000.0, 1);
        assert_eq!(one.stages.len(), 1);
        let four = denoise_cascade(&obs, 1000.0, 4);
        assert!(four.stages.len() > 1);
        // And the auto wrapper is exactly the 4-stage budget.
        let auto = denoise_cascade_auto(&obs, 1000.0);
        assert_eq!(auto.output, four.output);
        assert_eq!(auto.stages.len(), four.stages.len());
    }

    #[test]
    fn cascade_degrades_gracefully_on_edge_cases() {
        let empty: [f64; 0] = [];
        let res = denoise_cascade(&empty, 1000.0, 4);
        assert!(res.output.is_empty());
        assert!(res.stages.is_empty());
        for len in 1..4_usize
        {
            let x: Vec<f64> = (0..len).map(|i| i as f64 - 1.0).collect();
            let res = denoise_cascade(&x, 1000.0, 4);
            assert_eq!(res.output, x, "len {len} must pass through");
            assert!(res.stages.is_empty());
        }
        // A constant signal is LowNoise: untouched, zero stages.
        let c = vec![3.5; 64];
        let res = denoise_cascade(&c, 1000.0, 4);
        assert_eq!(res.output, c);
        assert!(res.stages.is_empty());
        assert_eq!(res.final_profile.dominant, NoiseType::LowNoise);
    }
}
