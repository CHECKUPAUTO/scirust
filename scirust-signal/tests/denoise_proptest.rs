//! Property-based (randomized) invariants for `scirust_signal::denoise`.
//!
//! The example-based tests pin *specific* behaviors; these check the invariants
//! that must hold for **every** input. Randomized testing is what catches the class
//! of bug the module's own review surfaced — a single NaN silently corrupting a
//! streaming rank filter's window — because that failure only shows up on inputs no
//! hand-written example happened to try. Each property is deliberately cheap so the
//! whole file stays well within a normal `cargo test` budget.
//!
//! Invariants covered:
//! * every denoiser in the representative pool below preserves length and never
//!   invents non-finite values from finite input, and never panics on adversarial
//!   input (empty, tiny, constant, or NaN/±∞ laced);
//! * streaming denoisers agree bit-for-bit with their batch counterparts on the
//!   window interior, restore state exactly on `reset`, and recover to finite output
//!   once a NaN has left the window;
//! * `separate` reconstructs the observation exactly (`signal + noise == observed`);
//! * `classify` returns a finite, non-negative noise scale on any input;
//! * the zero-phase IIR notch preserves length and finiteness, and an identity
//!   section is a true pass-through.

use proptest::prelude::*;
use scirust_signal::denoise::streaming::{
    StreamingHampel, StreamingKalman, StreamingMedian, StreamingMovingAverage,
};
use scirust_signal::denoise::{
    ThresholdMode, Wavelet, classify, filtfilt_sos, gaussian_smooth, hampel_filter,
    kalman_smooth_auto, median_filter, moving_average, nlm1d_auto, notch_iir, rbj_notch,
    savitzky_golay, separate, stft_wiener_auto, tikhonov_smooth, total_variation, wavelet_denoise,
    wavelet_denoise_bayes, wavelet_denoise_leveldep, wavelet_denoise_neighblock,
    wavelet_denoise_sure, wavelet_denoise_ti,
};
use scirust_signal::filter::Biquad;

/// A single pool entry: a name paired with its denoising function.
type DenoiserEntry = (&'static str, fn(&[f64]) -> Vec<f64>);

/// A representative denoiser pool spanning every family (linear, rank, transform,
/// variational, adaptive) — one default configuration per method, mirroring the
/// candidate shortlists used by `denoise_auto` / `denoise_best`. The invariant
/// tests below run every entry.
fn denoiser_pool() -> Vec<DenoiserEntry> {
    vec![
        ("moving_average(5)", |x| moving_average(x, 5)),
        ("gaussian_smooth(1.5)", |x| gaussian_smooth(x, 1.5)),
        ("savitzky_golay(2, 5)", |x| savitzky_golay(x, 2, 5)),
        ("median_filter(3)", |x| median_filter(x, 3)),
        ("hampel_filter(3, 3.0)", |x| hampel_filter(x, 3, 3.0)),
        ("wavelet_denoise(auto, soft)", |x| {
            wavelet_denoise(x, 0, ThresholdMode::Soft)
        }),
        ("wavelet_denoise_ti(auto, soft, db4, 15)", |x| {
            wavelet_denoise_ti(x, 0, ThresholdMode::Soft, Wavelet::Db4, 15)
        }),
        ("wavelet_denoise_bayes(auto, db4)", |x| {
            wavelet_denoise_bayes(x, 0, Wavelet::Db4)
        }),
        ("wavelet_denoise_sure(auto, db4)", |x| {
            wavelet_denoise_sure(x, 0, Wavelet::Db4)
        }),
        ("wavelet_denoise_neighblock(auto, db4)", |x| {
            wavelet_denoise_neighblock(x, 0, Wavelet::Db4)
        }),
        ("nlm1d_auto", |x| nlm1d_auto(x)),
        ("wavelet_denoise_leveldep(auto, soft, db4)", |x| {
            wavelet_denoise_leveldep(x, 0, ThresholdMode::Soft, Wavelet::Db4)
        }),
        ("total_variation(1.0, 8)", |x| total_variation(x, 1.0, 8)),
        ("tikhonov_smooth(10.0)", |x| tikhonov_smooth(x, 10.0)),
        ("stft_wiener_auto", |x| stft_wiener_auto(x)),
        ("kalman_smooth_auto", |x| kalman_smooth_auto(x).output),
    ]
}

/// A finite signal of moderate length and bounded amplitude — the domain on which
/// the length/finiteness invariants are meaningful (unbounded input could overflow a
/// linear filter to ±∞ legitimately).
fn finite_signal() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-1.0e3f64..1.0e3, 8..256)
}

/// A signal that may contain NaN / ±∞ / subnormals — for no-panic robustness.
fn wild_signal() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(
        prop_oneof![
            Just(f64::NAN),
            Just(f64::INFINITY),
            Just(f64::NEG_INFINITY),
            Just(0.0f64),
            -1.0e12f64..1.0e12,
        ],
        0..64,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Every pooled denoiser is length-preserving and finite-in ⇒ finite-out.
    #[test]
    fn pool_preserves_length_and_finiteness(sig in finite_signal()) {
        for (name, denoise) in denoiser_pool()
        {
            let out = denoise(&sig);
            prop_assert_eq!(out.len(), sig.len(), "{} changed length", name);
            prop_assert!(
                out.iter().all(|v| v.is_finite()),
                "{} produced a non-finite value on finite input",
                name
            );
        }
    }

    /// No pooled denoiser panics on adversarial input, and length is still preserved
    /// (the module-wide graceful-degradation contract).
    #[test]
    fn pool_never_panics_on_wild_input(sig in wild_signal()) {
        for (name, denoise) in denoiser_pool()
        {
            let out = denoise(&sig);
            prop_assert_eq!(out.len(), sig.len(), "{} changed length", name);
        }
    }

    /// `separate` is a *reconstruction*: information + noise must equal the input
    /// exactly (the decomposition never loses or invents energy), whatever the
    /// sample rate.
    #[test]
    fn separate_reconstructs_the_observation(sig in finite_signal(), fs in 1.0f64..4096.0) {
        let sep = separate(&sig, fs);
        prop_assert_eq!(sep.signal_estimate.len(), sig.len());
        prop_assert_eq!(sep.noise_estimate.len(), sig.len());
        for (i, &o) in sig.iter().enumerate()
        {
            let recon = sep.signal_estimate[i] + sep.noise_estimate[i];
            prop_assert!(
                (recon - o).abs() <= 1.0e-9 * (1.0 + o.abs()),
                "reconstruction drift at {i}: {recon} vs {o}"
            );
        }
        prop_assert!((0.0..=1.0).contains(&sep.residual_whiteness));
    }

    /// `classify` always returns a finite, non-negative noise scale, on any input.
    #[test]
    fn classify_noise_scale_is_finite_and_nonnegative(sig in finite_signal(), fs in 1.0f64..4096.0) {
        let p = classify(&sig, fs);
        prop_assert!(p.noise_std.is_finite() && p.noise_std >= 0.0, "noise_std = {}", p.noise_std);
        prop_assert!(p.spectral_flatness.is_finite());
        prop_assert!(p.trend_strength.is_finite());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// Streaming moving average matches the batch filter bit-for-bit on the window
    /// interior (the region where the batch window touches no mirrored border).
    #[test]
    fn streaming_moving_average_matches_batch_interior(
        sig in finite_signal(),
        half in 1usize..6,
    ) {
        let window = 2 * half + 1;
        let batch = moving_average(&sig, window);
        let mut f = StreamingMovingAverage::new(window);
        prop_assert_eq!(f.delay(), half);
        let stream: Vec<f64> = sig.iter().map(|&x| f.push(x)).collect();
        for i in (2 * half)..sig.len()
        {
            prop_assert!(
                (stream[i] - batch[i - half]).abs() <= 1.0e-9 * (1.0 + batch[i - half].abs()),
                "moving_average mismatch at {i}: {} vs {}",
                stream[i],
                batch[i - half]
            );
        }
    }

    /// Streaming median matches the batch median filter exactly on the interior.
    #[test]
    fn streaming_median_matches_batch_interior(sig in finite_signal(), half in 1usize..6) {
        let batch = median_filter(&sig, half);
        let mut f = StreamingMedian::new(half);
        let stream: Vec<f64> = sig.iter().map(|&x| f.push(x)).collect();
        for i in (2 * half)..sig.len()
        {
            prop_assert_eq!(
                stream[i],
                batch[i - half],
                "median mismatch at {}",
                i
            );
        }
    }

    /// Streaming Hampel matches the batch Hampel filter exactly on the interior.
    #[test]
    fn streaming_hampel_matches_batch_interior(sig in finite_signal(), half in 1usize..6) {
        let batch = hampel_filter(&sig, half, 3.0);
        let mut f = StreamingHampel::new(half, 3.0);
        let stream: Vec<f64> = sig.iter().map(|&x| f.push(x)).collect();
        for i in (2 * half)..sig.len()
        {
            prop_assert_eq!(stream[i], batch[i - half], "hampel mismatch at {}", i);
        }
    }

    /// `reset` returns every streaming filter to its exact just-constructed state:
    /// re-pushing the same prefix reproduces the same outputs.
    #[test]
    fn streaming_reset_is_exact(sig in finite_signal(), half in 1usize..6) {
        macro_rules! check_reset {
            ($f:expr) => {{
                let mut f = $f;
                let first: Vec<f64> = sig.iter().map(|&x| f.push(x)).collect();
                f.reset();
                let second: Vec<f64> = sig.iter().map(|&x| f.push(x)).collect();
                prop_assert_eq!(first, second, "reset was not exact");
            }};
        }
        check_reset!(StreamingMovingAverage::new(2 * half + 1));
        check_reset!(StreamingMedian::new(half));
        check_reset!(StreamingHampel::new(half, 3.0));
        check_reset!(StreamingKalman::new(1.0e-2, 1.0));
    }

    /// A single NaN sample must not permanently corrupt a streaming rank filter:
    /// once the NaN has left the window, every output is finite again. (This is the
    /// randomized guard for the fixed `partial_cmp` window-corruption bug.)
    #[test]
    fn streaming_rank_filters_recover_after_a_nan(
        sig in prop::collection::vec(-1.0e3f64..1.0e3, 12..128),
        half in 1usize..5,
        idx in 0usize..128,
    ) {
        let nan_at = idx % sig.len();
        let mut poisoned = sig.clone();
        poisoned[nan_at] = f64::NAN;
        let window = 2 * half + 1;

        for kind in ["median", "hampel"]
        {
            let out: Vec<f64> = match kind
            {
                "median" =>
                {
                    let mut f = StreamingMedian::new(half);
                    poisoned.iter().map(|&x| f.push(x)).collect()
                },
                _ =>
                {
                    let mut f = StreamingHampel::new(half, 3.0);
                    poisoned.iter().map(|&x| f.push(x)).collect()
                },
            };
            // Outputs whose window no longer contains the NaN index must be finite.
            for (i, &v) in out.iter().enumerate()
            {
                if i >= nan_at + window
                {
                    prop_assert!(
                        v.is_finite(),
                        "{kind}: non-finite output at {i} long after the NaN at {nan_at}"
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// The zero-phase IIR notch preserves length and finiteness for any finite input
    /// and any interior notch frequency.
    #[test]
    fn notch_iir_preserves_length_and_finiteness(
        sig in finite_signal(),
        fs in 100.0f64..4000.0,
        frac in 0.02f64..0.48,
        bw in 0.5f64..40.0,
    ) {
        let center = frac * fs;
        let out = notch_iir(&sig, fs, center, bw);
        prop_assert_eq!(out.len(), sig.len());
        prop_assert!(out.iter().all(|v| v.is_finite()), "notch produced a non-finite value");
    }

    /// An identity biquad through `filtfilt_sos` is an exact pass-through (up to
    /// round-off) — the forward-backward machinery adds nothing on a unit section.
    #[test]
    fn filtfilt_identity_is_passthrough(sig in finite_signal()) {
        let identity = Biquad { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };
        let out = filtfilt_sos(&[identity], &sig);
        prop_assert_eq!(out.len(), sig.len());
        for (i, (&a, &b)) in sig.iter().zip(out.iter()).enumerate()
        {
            prop_assert!((a - b).abs() <= 1.0e-9 * (1.0 + a.abs()), "identity drift at {i}: {a} vs {b}");
        }
    }

    /// A degenerate `rbj_notch` design (frequency at/above Nyquist, or non-positive Q)
    /// falls back to the identity section rather than an unstable filter, so notching
    /// with it is a no-op — never a blow-up.
    #[test]
    fn rbj_notch_degrades_to_identity_out_of_domain(
        sig in finite_signal(),
        fs in 100.0f64..4000.0,
    ) {
        let bad = rbj_notch(fs, fs, 10.0); // center == fs (>= Nyquist) → identity
        let out = filtfilt_sos(&[bad], &sig);
        for (i, (&a, &b)) in sig.iter().zip(out.iter()).enumerate()
        {
            prop_assert!((a - b).abs() <= 1.0e-9 * (1.0 + a.abs()), "out-of-domain notch altered sample {i}");
        }
    }
}
