//! # Débruitage & détection de bruit — a broad, extensible noise-removal toolkit
//!
//! This module answers two questions at once:
//!
//! 1. **How do we make noise removal "exhaustive"?** Not by enumerating every
//!    algorithm ever published — that set is open-ended — but by fixing a small,
//!    closed **taxonomy of families** ([`DenoiserFamily`]) and a single uniform
//!    interface ([`Denoiser`]). Adding a new method is then a mechanical act:
//!    pick its family, implement the trait. The families below cover the standard
//!    signal-processing literature end to end:
//!
//!    | Family | Idea | Best against |
//!    |--------|------|--------------|
//!    | [`DenoiserFamily::Linear`] | LTI convolution (moving average, Gaussian, Savitzky-Golay, EMA) | broadband Gaussian, gentle smoothing |
//!    | [`DenoiserFamily::Rank`] | order statistics (median, Hampel, α-trimmed mean) | impulsive / salt-and-pepper spikes |
//!    | [`DenoiserFamily::Transform`] | Fourier / wavelet shrinkage (low-pass, notch, Wiener, wavelet threshold) | tonal interference, white & colored noise |
//!    | [`DenoiserFamily::Variational`] | penalized least squares (Tikhonov, Total Variation) | edge-preserving smoothing, baseline drift |
//!    | [`DenoiserFamily::Adaptive`] | model / data-driven (Kalman RTS smoother, LMS/RLS line enhancers) | non-stationary noise, drifting tones |
//!
//! 2. **How do we detect "any" noise on "any" signal?** By *characterizing* the
//!    noise with a fixed feature set rather than trying to recognize it by name.
//!    [`detect::classify`] estimates the noise level (robust MAD), impulsivity
//!    (kurtosis / crest factor), spectral flatness, periodicity (dominant spectral
//!    line prominence), low-frequency trend strength, and the `1/f` color slope,
//!    then a small decision tree maps that [`NoiseProfile`] onto a [`NoiseType`].
//!    [`denoise_auto`] closes the loop: detect → pick the matching family → apply.
//!
//! Everything is pure Rust over `f64` slices, no external dependencies, and every
//! routine is validated by a signal-to-noise-ratio improvement test on a synthetic
//! signal with a *known* clean reference.
//!
//! ## Example
//!
//! ```
//! use scirust_signal::denoise::{classify, denoise_auto, separate};
//!
//! // A clean tone plus deterministic broadband disturbance.
//! let clean: Vec<f64> = (0..256).map(|i| (i as f64 * 0.2).sin()).collect();
//! let noisy: Vec<f64> = clean
//!     .iter()
//!     .enumerate()
//!     .map(|(i, &c)| c + 0.3 * ((i.wrapping_mul(2654435761)) as f64).sin())
//!     .collect();
//!
//! // 1. Characterize the noise without naming it up front.
//! let profile = classify(&noisy, 256.0);
//! println!("noise looks like {:?} (σ ≈ {:.3})", profile.dominant, profile.noise_std);
//!
//! // 2. One call detects the noise character and removes it.
//! let result = denoise_auto(&noisy, 256.0);
//! assert_eq!(result.output.len(), noisy.len());
//!
//! // 3. Or split into information + noise, with a whiteness self-check that flags
//! //    whether structure leaked into the "noise".
//! let sep = separate(&noisy, 256.0);
//! assert_eq!(sep.signal_estimate.len(), noisy.len());
//! assert!(sep.residual_whiteness >= 0.0 && sep.residual_whiteness <= 1.0);
//! ```

pub mod adaptive;
pub mod detect;
pub mod linear;
pub mod pipeline;
pub mod rank;
pub mod transform;
pub mod variational;

pub use adaptive::{
    KalmanFit, kalman_smooth, kalman_smooth_auto, kalman_trend_smooth, lms_line_enhancer,
    rls_line_enhancer,
};
pub use detect::{
    NoiseProfile, NoiseType, Separation, classify, estimate_noise_std, estimate_snr_db, separate,
};
pub use linear::{exp_moving_average, gaussian_smooth, moving_average, savitzky_golay};
pub use pipeline::{WaveletRlsRtsParams, wavelet_rls_rts_smooth, wavelet_rls_rts_smooth_1d};
pub use rank::{alpha_trimmed_mean, hampel_filter, impulse_mask, median_filter};
pub use transform::{
    ThresholdMode, Wavelet, fft_highpass, fft_lowpass, notch_filter, remove_mains_hum,
    spectral_subtraction, wavelet_denoise, wavelet_denoise_sure, wavelet_denoise_with,
    wiener_white,
};
pub use variational::{
    tikhonov_smooth, total_variation, total_variation_exact, total_variation_norm,
};

/// The family a denoiser belongs to — the taxonomy axis along which this toolkit
/// is meant to be *exhaustive*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenoiserFamily {
    /// Linear time-invariant convolution smoothers.
    Linear,
    /// Order-statistic / robust rank filters.
    Rank,
    /// Fourier- or wavelet-domain shrinkage.
    Transform,
    /// Penalized-least-squares / total-variation optimization.
    Variational,
    /// Model-based / adaptive: Kalman RTS smoother (auto-tuned by innovation
    /// whiteness), LMS/RLS adaptive line enhancers.
    Adaptive,
}

/// A uniform interface every denoiser implements, so callers — and the automatic
/// selector — can treat them interchangeably, and users can register their own
/// without touching the core.
pub trait Denoiser {
    /// Human-readable method name.
    fn name(&self) -> &str;
    /// Which [`DenoiserFamily`] this method belongs to.
    fn family(&self) -> DenoiserFamily;
    /// Apply the denoiser, returning a signal of the same length as the input.
    fn apply(&self, signal: &[f64]) -> Vec<f64>;
}

macro_rules! denoiser {
    ($ty:ident { $($field:ident : $fty:ty),* $(,)? }, $name:literal, $family:expr, $call:expr) => {
        /// Configured denoiser wrapper implementing [`Denoiser`].
        #[derive(Debug, Clone)]
        pub struct $ty {
            $(pub $field: $fty),*
        }
        impl Denoiser for $ty {
            fn name(&self) -> &str { $name }
            fn family(&self) -> DenoiserFamily { $family }
            fn apply(&self, signal: &[f64]) -> Vec<f64> {
                let f: &dyn Fn(&$ty, &[f64]) -> Vec<f64> = &$call;
                f(self, signal)
            }
        }
    };
}

denoiser!(
    MovingAverage { window: usize },
    "moving_average",
    DenoiserFamily::Linear,
    |s: &MovingAverage, x: &[f64]| moving_average(x, s.window)
);
denoiser!(
    GaussianSmooth { sigma: f64 },
    "gaussian_smooth",
    DenoiserFamily::Linear,
    |s: &GaussianSmooth, x: &[f64]| gaussian_smooth(x, s.sigma)
);
denoiser!(
    SavitzkyGolay {
        poly_order: usize,
        half_window: usize
    },
    "savitzky_golay",
    DenoiserFamily::Linear,
    |s: &SavitzkyGolay, x: &[f64]| savitzky_golay(x, s.poly_order, s.half_window)
);
denoiser!(
    Median { half_window: usize },
    "median_filter",
    DenoiserFamily::Rank,
    |s: &Median, x: &[f64]| median_filter(x, s.half_window)
);
denoiser!(
    Hampel {
        half_window: usize,
        n_sigma: f64
    },
    "hampel_filter",
    DenoiserFamily::Rank,
    |s: &Hampel, x: &[f64]| hampel_filter(x, s.half_window, s.n_sigma)
);
denoiser!(
    WaveletShrink {
        levels: usize,
        mode: ThresholdMode
    },
    "wavelet_denoise",
    DenoiserFamily::Transform,
    |s: &WaveletShrink, x: &[f64]| wavelet_denoise(x, s.levels, s.mode)
);
denoiser!(
    TotalVariation {
        lambda: f64,
        iters: usize
    },
    "total_variation",
    DenoiserFamily::Variational,
    |s: &TotalVariation, x: &[f64]| total_variation(x, s.lambda, s.iters)
);
denoiser!(
    Tikhonov { lambda: f64 },
    "tikhonov_smooth",
    DenoiserFamily::Variational,
    |s: &Tikhonov, x: &[f64]| tikhonov_smooth(x, s.lambda)
);
denoiser!(
    KalmanAuto {},
    "kalman_smooth_auto",
    DenoiserFamily::Adaptive,
    |_s: &KalmanAuto, x: &[f64]| kalman_smooth_auto(x).output
);
denoiser!(
    AdaptiveLine {
        taps: usize,
        delay: usize,
        mu: f64
    },
    "lms_line_enhancer",
    DenoiserFamily::Adaptive,
    |s: &AdaptiveLine, x: &[f64]| lms_line_enhancer(x, s.taps, s.delay, s.mu)
);

/// A reasonable default catalog spanning every family — a starting point that
/// callers can extend with their own [`Denoiser`] implementations.
pub fn catalog() -> Vec<Box<dyn Denoiser>> {
    vec![
        Box::new(MovingAverage { window: 5 }),
        Box::new(GaussianSmooth { sigma: 1.5 }),
        Box::new(SavitzkyGolay {
            poly_order: 2,
            half_window: 5,
        }),
        Box::new(Median { half_window: 3 }),
        Box::new(Hampel {
            half_window: 3,
            n_sigma: 3.0,
        }),
        Box::new(WaveletShrink {
            levels: 0,
            mode: ThresholdMode::Soft,
        }),
        Box::new(TotalVariation {
            lambda: 1.0,
            iters: 8,
        }),
        Box::new(Tikhonov { lambda: 10.0 }),
        Box::new(KalmanAuto {}),
    ]
}

/// Result of the automatic detect-then-denoise pipeline.
#[derive(Debug, Clone)]
pub struct AutoResult {
    /// The noise characterization that drove the choice.
    pub profile: NoiseProfile,
    /// Which method was selected (human-readable).
    pub method: String,
    /// The denoised signal.
    pub output: Vec<f64>,
}

/// **Universal denoiser**: characterize the noise, pick the matching family, apply.
///
/// This is the "one call that handles any signal" entry point. It never fails:
/// for an unrecognized or clean signal it falls back to a light Savitzky-Golay
/// smooth. `sample_rate` is used only to report physical frequencies and to size
/// the notch filter; pass `1.0` if you work in normalized units.
pub fn denoise_auto(signal: &[f64], sample_rate: f64) -> AutoResult {
    let profile = detect::classify(signal, sample_rate);
    let (method, output): (String, Vec<f64>) = match profile.dominant
    {
        NoiseType::LowNoise => ("savitzky_golay(2, 5)".into(), savitzky_golay(signal, 2, 5)),
        NoiseType::Impulsive => (
            "hampel_filter(3, 3.0)".into(),
            hampel_filter(signal, 3, 3.0),
        ),
        NoiseType::Periodic =>
        {
            let bin = sample_rate / next_pow2(signal.len().max(2)) as f64;
            let bw = (profile.dominant_freq_hz * 0.05).max(bin * 4.0);
            let out = notch_filter(signal, sample_rate, profile.dominant_freq_hz, bw);
            (
                format!("notch_filter @ {:.3} Hz", profile.dominant_freq_hz),
                out,
            )
        },
        NoiseType::Baseline =>
        {
            let base = tikhonov_smooth(signal, 1.0e4);
            let out: Vec<f64> = signal.iter().zip(base.iter()).map(|(s, b)| s - b).collect();
            ("baseline_removal (signal − tikhonov trend)".into(), out)
        },
        NoiseType::Gaussian =>
        {
            // Stationary broadband noise: the Wiener gain leaves the whitest residual.
            (
                "wiener_white".into(),
                wiener_white(signal, profile.noise_std),
            )
        },
        NoiseType::Colored => (
            "wavelet_denoise(auto, soft)".into(),
            wavelet_denoise(signal, 0, ThresholdMode::Soft),
        ),
    };
    AutoResult {
        profile,
        method,
        output,
    }
}

// ---------------------------------------------------------------------------
// Shared helpers used across the submodules.
// ---------------------------------------------------------------------------

/// Reflect an out-of-range index back into `0..n` (symmetric / edge-repeated
/// mirror), so windowed filters can run over signal borders without shrinking.
pub(crate) fn mirror_index(i: isize, n: usize) -> usize {
    if n <= 1
    {
        return 0;
    }
    let n_i = n as isize;
    let period = 2 * n_i;
    let mut m = i.rem_euclid(period);
    if m >= n_i
    {
        m = period - 1 - m;
    }
    m as usize
}

/// Median of a slice (clones and sorts). Returns 0.0 for an empty slice.
pub(crate) fn median(values: &[f64]) -> f64 {
    if values.is_empty()
    {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1
    {
        v[n / 2]
    }
    else
    {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Median absolute deviation: `median(|x − median(x)|)`. A robust, breakdown-heavy
/// scale estimator immune to a minority of outliers.
pub(crate) fn mad(values: &[f64]) -> f64 {
    if values.is_empty()
    {
        return 0.0;
    }
    let med = median(values);
    let dev: Vec<f64> = values.iter().map(|&x| (x - med).abs()).collect();
    median(&dev)
}

/// Robust noise-σ estimate (Donoho MAD on the finest Haar detail band). Shared by
/// [`detect::estimate_noise_std`] and any denoiser that needs a noise scale.
pub(crate) fn estimate_noise_std_helper(signal: &[f64]) -> f64 {
    let n = signal.len();
    if n < 2
    {
        return 0.0;
    }
    let half = n / 2;
    let mut detail = Vec::with_capacity(half);
    for i in 0..half
    {
        detail.push((signal[2 * i] - signal[2 * i + 1]) / core::f64::consts::SQRT_2);
    }
    mad(&detail) / 0.6745
}

/// Smallest power of two `>= n` (at least 1).
pub(crate) fn next_pow2(n: usize) -> usize {
    if n <= 1
    {
        return 1;
    }
    n.next_power_of_two()
}

/// Right-pad a signal by symmetric reflection up to the next power of two, so the
/// power-of-two FFT and Haar transform can process arbitrary-length inputs with a
/// smooth (non-discontinuous) boundary. Returns the original slice unchanged when
/// it is already a power of two.
pub(crate) fn pad_reflect_pow2(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let target = next_pow2(n);
    if target == n
    {
        return signal.to_vec();
    }
    let mut out = Vec::with_capacity(target);
    out.extend_from_slice(signal);
    for i in 0..(target - n)
    {
        let idx = mirror_index((n + i) as isize, n);
        out.push(signal[idx]);
    }
    out
}

#[cfg(test)]
pub(crate) mod testutil {
    use core::f64::consts::PI;

    /// Deterministic 64-bit LCG so noise tests are reproducible without a `rand`
    /// dependency.
    pub(crate) struct Lcg(u64);

    impl Lcg {
        pub(crate) fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        /// Uniform in [0, 1).
        pub(crate) fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        /// Standard normal via Box-Muller.
        pub(crate) fn gauss(&mut self) -> f64 {
            let u1 = self.uniform().max(1.0e-12);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        }
    }

    /// Signal-to-noise ratio in dB of an estimate `est` against a clean reference.
    pub(crate) fn snr_db(clean: &[f64], est: &[f64]) -> f64 {
        let sig: f64 = clean.iter().map(|&x| x * x).sum();
        let err: f64 = clean
            .iter()
            .zip(est.iter())
            .map(|(&c, &e)| (c - e) * (c - e))
            .sum();
        10.0 * (sig / err.max(1.0e-30)).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::Lcg;
    use super::*;
    use core::f64::consts::PI;

    fn noisy_sine(n: usize) -> Vec<f64> {
        let mut rng = Lcg::new(79);
        (0..n)
            .map(|i| (2.0 * PI * 4.0 * i as f64 / n as f64).sin() + 0.3 * rng.gauss())
            .collect()
    }

    #[test]
    fn catalog_spans_every_family_and_every_entry_runs() {
        let obs = noisy_sine(256);
        let cat = catalog();
        let mut families = Vec::new();
        for d in cat.iter()
        {
            let out = d.apply(&obs);
            assert_eq!(out.len(), obs.len(), "{} changed the length", d.name());
            assert!(
                out.iter().all(|v| v.is_finite()),
                "{} produced non-finite output",
                d.name()
            );
            assert!(!d.name().is_empty());
            if !families.contains(&d.family())
            {
                families.push(d.family());
            }
        }
        for fam in [
            DenoiserFamily::Linear,
            DenoiserFamily::Rank,
            DenoiserFamily::Transform,
            DenoiserFamily::Variational,
            DenoiserFamily::Adaptive,
        ]
        {
            assert!(families.contains(&fam), "catalog misses family {fam:?}");
        }
    }

    #[test]
    fn denoiser_wrappers_match_their_functions() {
        // The trait wrappers must plumb their parameters through in the right
        // order — a taps/delay transposition would compile silently.
        let obs = noisy_sine(512);
        let wrapper = AdaptiveLine {
            taps: 12,
            delay: 2,
            mu: 0.3,
        };
        assert_eq!(wrapper.apply(&obs), lms_line_enhancer(&obs, 12, 2, 0.3));
        assert_eq!(wrapper.family(), DenoiserFamily::Adaptive);
        let kalman = KalmanAuto {};
        assert_eq!(kalman.apply(&obs), kalman_smooth_auto(&obs).output);
        let sg = SavitzkyGolay {
            poly_order: 2,
            half_window: 5,
        };
        assert_eq!(sg.apply(&obs), savitzky_golay(&obs, 2, 5));
        let hampel = Hampel {
            half_window: 3,
            n_sigma: 3.0,
        };
        assert_eq!(hampel.apply(&obs), hampel_filter(&obs, 3, 3.0));
    }
}
