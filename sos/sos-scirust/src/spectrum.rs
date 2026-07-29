//! [`PeriodogramSimulator`] — a real [`Simulate`] backend for spectral
//! analysis, wrapping `scirust-signal`'s FFT. `L3`.
//!
//! The backends this crate shipped first all answer *"what does this system
//! do?"* — integrate an ODE, evaluate an integral, solve for a root. This one
//! answers a different kind of question: *"what is in this signal?"* The
//! observation is a **spectrum**, not a trajectory or a scalar, which is what
//! `scirust-signal`'s executor kind adds that the solver backends could not.
//!
//! ## `L3` means reproducible. It does not mean accurate.
//!
//! This is the distinction this backend exists to make explicit, because it
//! is the one a determinism taxonomy invites a reader to blur.
//!
//! A windowed periodogram is a fixed sequence of `f64` operations — FFT,
//! magnitude, scale — with no iteration, no tolerance and no sampling. Run it
//! twice on the same input and you get the same bits. That is exactly what
//! `L3` asserts, and it is all `L3` asserts.
//!
//! It is *not* a claim that the numbers are a good estimate of the signal's
//! true power spectral density. A single periodogram is a **biased,
//! high-variance** estimator: its variance does not shrink as you collect
//! more samples — longer records buy frequency resolution, not stability.
//! Reducing that variance needs averaging over segments (Welch's method) or
//! smoothing, neither of which is implemented here and neither of which this
//! module will pretend to. A caller who needs a stable PSD estimate should
//! know they do not have one.
//!
//! So: bit-reproducible, and honestly described. Those are different
//! properties, and an `L3` tag only carries the first.
//!
//! ## The window travels with the result
//!
//! Which window was applied changes the spectrum — its leakage, its
//! resolution, the height of every peak. [`Spectrum`] therefore carries
//! [`window`](Spectrum::window) in the **output**, not only in the
//! configuration that produced it, for the same reason
//! [`crate::root::CertifiedRoot`] carries its starting point: a stored result
//! travels without its config, and a spectrum that did not say which window
//! shaped it could be read as *the* spectrum of the signal.
//!
//! ## Validation is this adapter's job, because the primitive asserts
//!
//! `scirust_signal::fft_real` **panics** on a non-power-of-two input. A
//! [`Simulate`] backend must never panic on a configuration — the contract is
//! `Ok` or a structured [`SimError`]. So every precondition the FFT assumes is
//! checked here first and reported as [`SimError::InvalidConfig`]. The same
//! shape of care the rest of this crate applies to solver errors, for the same
//! reason.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use scirust_signal::features::spectral::{
    psd, spectral_centroid, spectral_flatness, spectral_spread,
};
use scirust_signal::{apply_window, blackman, fft_real, flattop, hamming, hanning};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Body, DeterminismLevel};
use sos_simulation::{Observation, Result, SimDescriptor, SimError, Simulate};

use crate::solver::{ExactF64Seq, encode_f64};

/// Which window to apply before transforming.
///
/// Named rather than supplied as coefficients so the choice is canonically
/// encodable — and so two studies that used the same window are recognisably
/// comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WindowKind {
    /// No shaping. Maximum frequency resolution, worst spectral leakage.
    Rectangular,
    /// Hann — the usual default: good leakage suppression, moderate width.
    Hann,
    /// Hamming.
    Hamming,
    /// Blackman — stronger leakage suppression, wider main lobe.
    Blackman,
    /// Flat-top — poor resolution, but the most accurate *amplitude* reading,
    /// which is what a calibration measurement wants.
    FlatTop,
}

impl WindowKind {
    /// A short, stable code used for canonical encoding and display.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self
        {
            Self::Rectangular => "rectangular",
            Self::Hann => "hann",
            Self::Hamming => "hamming",
            Self::Blackman => "blackman",
            Self::FlatTop => "flat-top",
        }
    }

    /// The window's coefficients for a record of `n` samples.
    #[must_use]
    pub fn coefficients(self, n: usize) -> Vec<f64> {
        match self
        {
            Self::Rectangular => vec![1.0; n],
            Self::Hann => hanning(n),
            Self::Hamming => hamming(n),
            Self::Blackman => blackman(n),
            Self::FlatTop => flattop(n),
        }
    }

    /// The window named by `code`, or `None` — the inverse of
    /// [`code`](Self::code), so a window survives a file.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        [
            Self::Rectangular,
            Self::Hann,
            Self::Hamming,
            Self::Blackman,
            Self::FlatTop,
        ]
        .into_iter()
        .find(|w| w.code() == code)
    }
}

impl std::fmt::Display for WindowKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

impl Canonical for WindowKind {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(self.code());
    }
}

impl Serialize for WindowKind {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(self.code())
    }
}

impl<'de> Deserialize<'de> for WindowKind {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let code = String::deserialize(d)?;
        // An unknown window is an error, never a default: a spectrum shaped by
        // a window this build does not have must not load as a different one.
        Self::from_code(&code).ok_or_else(|| D::Error::custom(format!("no window named {code:?}")))
    }
}

/// One spectral measurement: the record, how it was sampled, and how it was
/// shaped.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectrumConfig {
    /// The observed samples. Length must be a power of two and at least 2 —
    /// see the module docs on why this is checked here.
    pub signal: Vec<f64>,
    /// The sampling rate in hertz. Must be finite and positive.
    pub sample_rate_hz: f64,
    /// The window applied before transforming.
    pub window: WindowKind,
}

impl SpectrumConfig {
    /// Construct a spectral configuration. Validity is checked by
    /// [`PeriodogramSimulator::run`], not here — a config is just data.
    #[must_use]
    pub fn new(signal: Vec<f64>, sample_rate_hz: f64, window: WindowKind) -> Self {
        Self {
            signal,
            sample_rate_hz,
            window,
        }
    }
}

impl Canonical for SpectrumConfig {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&ExactF64Seq(&self.signal));
        encode_f64(enc, self.sample_rate_hz);
        enc.value(&self.window);
    }
}

/// The serialized shape of a [`SpectrumConfig`]: every sample an exact
/// shortest round-trip decimal string.
///
/// Same reasoning as [`crate::model::ModelRun`]'s — `serde_json`'s `f64`
/// round-trip is not bit-exact, and a sample that came back changed would move
/// the config's content address, so a manifest pinning it would stop
/// resolving. This is also what makes a spectral measurement *authorable*: a
/// `SpectrumConfig` in a file is the whole experiment.
#[derive(Serialize, Deserialize)]
struct ConfigRepr {
    signal: Vec<String>,
    sample_rate_hz: String,
    window: WindowKind,
}

impl Serialize for SpectrumConfig {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        ConfigRepr {
            signal: self
                .signal
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
            sample_rate_hz: self.sample_rate_hz.to_string(),
            window: self.window,
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for SpectrumConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let repr = ConfigRepr::deserialize(d)?;
        let parse = |s: &str| s.parse::<f64>().map_err(D::Error::custom);
        Ok(SpectrumConfig {
            signal: repr
                .signal
                .iter()
                .map(|v| parse(v))
                .collect::<std::result::Result<Vec<f64>, _>>()?,
            sample_rate_hz: parse(&repr.sample_rate_hz)?,
            window: repr.window,
        })
    }
}

/// A computed spectrum, with the context needed to read it correctly.
#[derive(Debug, Clone, PartialEq)]
pub struct Spectrum {
    /// Power spectral density over the positive half-spectrum, DC to Nyquist.
    pub psd: Vec<f64>,
    /// The bin spacing in hertz — `sample_rate / n`, so a caller can label
    /// the axis without re-deriving it from the config.
    pub bin_width_hz: f64,
    /// The window that shaped this spectrum. Carried here deliberately: see
    /// the module docs.
    pub window: WindowKind,
    /// Spectral centroid in hertz — the power-weighted mean frequency.
    pub centroid_hz: f64,
    /// Spectral spread in hertz about the centroid.
    pub spread_hz: f64,
    /// Spectral flatness in `[0, 1]`: near `1` is noise-like, near `0` is
    /// tonal.
    pub flatness: f64,
}

/// The serialized shape of a [`Spectrum`] — exact decimal text throughout,
/// for the same round-trip reason [`ConfigRepr`] uses it.
#[derive(Serialize, Deserialize)]
struct SpectrumRepr {
    psd: Vec<String>,
    bin_width_hz: String,
    window: WindowKind,
    centroid_hz: String,
    spread_hz: String,
    flatness: String,
}

impl Serialize for Spectrum {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        SpectrumRepr {
            psd: self
                .psd
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
            bin_width_hz: self.bin_width_hz.to_string(),
            window: self.window,
            centroid_hz: self.centroid_hz.to_string(),
            spread_hz: self.spread_hz.to_string(),
            flatness: self.flatness.to_string(),
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for Spectrum {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let r = SpectrumRepr::deserialize(d)?;
        let parse = |s: &str| s.parse::<f64>().map_err(D::Error::custom);
        Ok(Spectrum {
            psd: r
                .psd
                .iter()
                .map(|v| parse(v))
                .collect::<std::result::Result<Vec<f64>, _>>()?,
            bin_width_hz: parse(&r.bin_width_hz)?,
            window: r.window,
            centroid_hz: parse(&r.centroid_hz)?,
            spread_hz: parse(&r.spread_hz)?,
            flatness: parse(&r.flatness)?,
        })
    }
}

impl Canonical for Spectrum {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&ExactF64Seq(&self.psd));
        encode_f64(enc, self.bin_width_hz);
        enc.value(&self.window);
        encode_f64(enc, self.centroid_hz);
        encode_f64(enc, self.spread_hz);
        encode_f64(enc, self.flatness);
    }
}

/// The body of a stored spectral measurement: the spectrum, the level the
/// backend realized, and the seed it ran under.
///
/// Like [`crate::model::ModeledTrajectoryBody`], this makes the result a
/// first-class SOS object — and like [`Spectrum`] itself, it keeps the window
/// beside the numbers, so a stored spectrum can still say what shaped it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpectrumBody {
    /// The measured spectrum.
    pub spectrum: Spectrum,
    /// The determinism level the backend realized.
    pub level: DeterminismLevel,
    /// The seed the run used.
    pub seed: u64,
}

impl SpectrumBody {
    /// Flatten an observation into a storable body.
    #[must_use]
    pub fn from_observation(observation: Observation<Spectrum>) -> Self {
        let (level, seed) = (observation.level(), observation.seed);
        Self {
            spectrum: observation.output,
            level,
            seed,
        }
    }

    /// Rebuild the [`Observation`] this body was stored from.
    #[must_use]
    pub fn observation(&self) -> Observation<Spectrum> {
        Observation::new(self.spectrum.clone(), self.level, self.seed)
    }
}

impl Canonical for SpectrumBody {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.spectrum);
        enc.value(&self.level);
        enc.u64(self.seed);
    }
}

impl Body for SpectrumBody {
    const KIND: &'static str = "Spectrum";
    const SCHEMA_VERSION: u32 = 1;
}

/// A real [`Simulate`] backend: computes a windowed periodogram with
/// `scirust-signal`'s FFT.
///
/// Unlike the solver backends, this one closes over no model — the
/// "experiment" is fully described by its configuration, so `descriptor` is
/// still caller-supplied but only to name the measurement, not to
/// distinguish two different physical systems sharing one solver.
#[derive(Debug, Clone)]
pub struct PeriodogramSimulator {
    descriptor: SimDescriptor,
}

impl PeriodogramSimulator {
    /// A named, versioned periodogram backend.
    #[must_use]
    pub fn new(descriptor: SimDescriptor) -> Self {
        Self { descriptor }
    }
}

impl Simulate for PeriodogramSimulator {
    type Config = SpectrumConfig;
    type Output = Spectrum;

    fn descriptor(&self) -> SimDescriptor {
        self.descriptor.clone()
    }

    fn level(&self) -> DeterminismLevel {
        // Exact arithmetic, no iteration, no sampling. See the module docs on
        // what this does and does not assert.
        DeterminismLevel::L3
    }

    fn run(&self, config: &SpectrumConfig, seed: u64) -> Result<Observation<Spectrum>> {
        let n = config.signal.len();
        // Every precondition `fft_real` asserts, checked before it can panic.
        if n < 2
        {
            return Err(SimError::InvalidConfig(format!(
                "a spectrum needs at least 2 samples (got {n})"
            )));
        }
        if !n.is_power_of_two()
        {
            return Err(SimError::InvalidConfig(format!(
                "the FFT requires a power-of-two record length (got {n})"
            )));
        }
        if !(config.sample_rate_hz.is_finite() && config.sample_rate_hz > 0.0)
        {
            return Err(SimError::InvalidConfig(format!(
                "sample rate must be finite and positive (got {})",
                config.sample_rate_hz
            )));
        }
        if let Some(bad) = config.signal.iter().position(|x| !x.is_finite())
        {
            return Err(SimError::InvalidConfig(format!(
                "sample #{bad} is not finite ({})",
                config.signal[bad]
            )));
        }

        let mut shaped = config.signal.clone();
        apply_window(&mut shaped, &config.window.coefficients(n));
        let spectrum = fft_real(&shaped);

        #[allow(clippy::cast_precision_loss)] // record lengths are far below 2^53
        let bin_width_hz = config.sample_rate_hz / n as f64;
        let measured = Spectrum {
            psd: psd(&spectrum, n),
            bin_width_hz,
            window: config.window,
            centroid_hz: spectral_centroid(&spectrum, config.sample_rate_hz),
            spread_hz: spectral_spread(&spectrum, config.sample_rate_hz),
            flatness: spectral_flatness(&spectrum),
        };
        Ok(Observation::new(measured, self.level(), seed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_stats::SplitMix64;
    use sos_core::SemVer;
    use sos_simulation::Vcr;
    use std::f64::consts::TAU;

    fn descriptor(name: &str) -> SimDescriptor {
        SimDescriptor::new(name, SemVer::new(1, 0, 0))
    }

    /// `n` samples of a pure tone at `freq_hz`, sampled at `rate_hz`.
    fn tone(freq_hz: f64, rate_hz: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (TAU * freq_hz * (i as f64) / rate_hz).sin())
            .collect()
    }

    fn measure(config: &SpectrumConfig) -> Spectrum {
        PeriodogramSimulator::new(descriptor("test/spectrum"))
            .run(config, 0)
            .expect("a valid config must measure")
            .output
    }

    /// `n` samples of zero-mean white noise from a fixed seed — reproducible,
    /// so the statistical assertions below are deterministic rather than flaky.
    fn white_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = SplitMix64::new(seed);
        (0..n).map(|_| rng.next_f64() - 0.5).collect()
    }

    /// Relative dispersion (`std / mean`) of a PSD across its non-DC bins.
    ///
    /// For a periodogram of white noise each bin is approximately exponentially
    /// distributed, so this is ≈ 1 *whatever `n` is* — which is precisely the
    /// property the module docs decline to hide.
    fn relative_dispersion(psd: &[f64]) -> f64 {
        let bins = &psd[1..];
        let n = bins.len() as f64;
        let mean = bins.iter().sum::<f64>() / n;
        let var = bins.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n;
        var.sqrt() / mean
    }

    #[test]
    fn a_pure_tone_peaks_in_the_bin_that_contains_it() {
        // The check that this is a real transform: 64 Hz sampled at 1024 Hz
        // over 1024 samples puts the tone exactly in bin 64.
        let config = SpectrumConfig::new(tone(64.0, 1024.0, 1024), 1024.0, WindowKind::Rectangular);
        let s = measure(&config);
        let peak = s
            .psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, 64, "the tone must land in its own bin");
        assert!((s.bin_width_hz - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_centroid_tracks_the_tone_frequency() {
        let low = measure(&SpectrumConfig::new(
            tone(64.0, 1024.0, 1024),
            1024.0,
            WindowKind::Hann,
        ));
        let high = measure(&SpectrumConfig::new(
            tone(256.0, 1024.0, 1024),
            1024.0,
            WindowKind::Hann,
        ));
        assert!(
            high.centroid_hz > low.centroid_hz,
            "{} should exceed {}",
            high.centroid_hz,
            low.centroid_hz
        );
    }

    #[test]
    fn a_tone_is_tonal_and_an_impulse_is_not() {
        // Flatness separates the two by construction: near 0 tonal, near 1
        // noise-like. A real measurement, not a tautology about the code.
        let tonal = measure(&SpectrumConfig::new(
            tone(64.0, 1024.0, 1024),
            1024.0,
            WindowKind::Hann,
        ));
        // A single non-zero sample has a flat magnitude spectrum.
        let mut impulse = vec![0.0; 1024];
        impulse[0] = 1.0;
        let flat = measure(&SpectrumConfig::new(
            impulse,
            1024.0,
            WindowKind::Rectangular,
        ));
        assert!(
            tonal.flatness < flat.flatness,
            "{} vs {}",
            tonal.flatness,
            flat.flatness
        );
    }

    #[test]
    fn the_window_changes_the_spectrum_and_is_recorded_in_the_output() {
        // Two measurements of the *same* signal that are legitimately
        // different results — which is exactly why the window has to travel
        // with the spectrum rather than only with the config.
        let signal = tone(64.5, 1024.0, 1024); // off-bin, so leakage differs
        let rect = measure(&SpectrumConfig::new(
            signal.clone(),
            1024.0,
            WindowKind::Rectangular,
        ));
        let hann = measure(&SpectrumConfig::new(signal, 1024.0, WindowKind::Hann));

        assert_eq!(rect.window, WindowKind::Rectangular);
        assert_eq!(hann.window, WindowKind::Hann);
        assert_ne!(rect.psd, hann.psd);
        assert_ne!(rect.canonical_bytes(), hann.canonical_bytes());
    }

    #[test]
    fn a_hann_window_suppresses_leakage_from_an_off_bin_tone() {
        // The point of windowing, measured rather than asserted: for a tone
        // sitting between bins, a Hann window puts far less power far from
        // the peak than a rectangular one.
        let signal = tone(64.5, 1024.0, 1024);
        let far_power = |w: WindowKind| -> f64 {
            let s = measure(&SpectrumConfig::new(signal.clone(), 1024.0, w));
            s.psd
                .iter()
                .enumerate()
                .filter(|(i, _)| (*i as i64 - 64).abs() > 8)
                .map(|(_, p)| p)
                .sum()
        };
        assert!(
            far_power(WindowKind::Hann) < far_power(WindowKind::Rectangular),
            "the Hann window must leak less"
        );
    }

    // ---- "L3 is not accuracy", measured rather than merely documented -----

    #[test]
    fn a_longer_record_does_not_make_the_periodogram_less_noisy() {
        // The module docs' central caveat, checked instead of asserted:
        // sixteen times the data buys sixteen times the frequency resolution
        // and no stability whatsoever. Both dispersions sit near 1 because
        // each bin is ~exponentially distributed however long the record is.
        let dispersion_at = |n: usize| {
            relative_dispersion(
                &measure(&SpectrumConfig::new(
                    white_noise(n, 0x5EED),
                    1024.0,
                    WindowKind::Rectangular,
                ))
                .psd,
            )
        };
        let (short, long) = (dispersion_at(256), dispersion_at(4096));
        assert!(short > 0.7, "a periodogram of noise is noisy: {short}");
        assert!(
            long > 0.9 * short,
            "16x the samples must not shrink the variance: {short} -> {long}"
        );
    }

    #[test]
    fn averaging_segments_is_what_would_reduce_the_variance() {
        // The other half of the same claim, so "not implemented" is a real
        // omission rather than an excuse: the variance *is* reducible, by
        // Welch-style averaging over segments. Averaging 32 independent
        // 256-sample periodograms cuts the dispersion by roughly sqrt(32)
        // while keeping the 256-sample run's resolution — which is exactly
        // the machinery this backend would have to grow before anyone could
        // call it a stable PSD estimator.
        let single = measure(&SpectrumConfig::new(
            white_noise(256, 1),
            1024.0,
            WindowKind::Rectangular,
        ));
        let segments = 32_u64;
        let mut averaged = vec![0.0; single.psd.len()];
        for k in 1..=segments
        {
            let s = measure(&SpectrumConfig::new(
                white_noise(256, k),
                1024.0,
                WindowKind::Rectangular,
            ));
            for (acc, p) in averaged.iter_mut().zip(&s.psd)
            {
                *acc += p / segments as f64;
            }
        }
        let one = relative_dispersion(&single.psd);
        let many = relative_dispersion(&averaged);
        assert!(
            many < one / 3.0,
            "averaging {segments} segments must visibly stabilize the estimate: {one} -> {many}"
        );
    }

    // ---- the adapter's validation, because the primitive asserts ----------

    #[test]
    fn a_non_power_of_two_record_is_an_error_not_a_panic() {
        // `fft_real` asserts on this. A Simulate backend must not.
        let sim = PeriodogramSimulator::new(descriptor("test/guard"));
        for n in [3_usize, 100, 1000]
        {
            let config = SpectrumConfig::new(vec![0.5; n], 1024.0, WindowKind::Hann);
            assert!(
                matches!(sim.run(&config, 0), Err(SimError::InvalidConfig(_))),
                "n = {n} must be a structured error"
            );
        }
    }

    #[test]
    fn a_too_short_record_is_an_error() {
        let sim = PeriodogramSimulator::new(descriptor("test/short"));
        for signal in [vec![], vec![1.0]]
        {
            assert!(matches!(
                sim.run(&SpectrumConfig::new(signal, 1024.0, WindowKind::Hann), 0),
                Err(SimError::InvalidConfig(_))
            ));
        }
    }

    #[test]
    fn an_invalid_sample_rate_is_an_error() {
        let sim = PeriodogramSimulator::new(descriptor("test/rate"));
        for rate in [0.0, -1.0, f64::NAN, f64::INFINITY]
        {
            assert!(
                matches!(
                    sim.run(
                        &SpectrumConfig::new(vec![0.5; 8], rate, WindowKind::Hann),
                        0
                    ),
                    Err(SimError::InvalidConfig(_))
                ),
                "rate {rate} must be rejected"
            );
        }
    }

    #[test]
    fn a_non_finite_sample_is_rejected_and_named() {
        let sim = PeriodogramSimulator::new(descriptor("test/nan"));
        let mut signal = vec![0.5; 8];
        signal[5] = f64::NAN;
        let err = sim
            .run(&SpectrumConfig::new(signal, 1024.0, WindowKind::Hann), 0)
            .unwrap_err();
        assert!(matches!(err, SimError::InvalidConfig(_)));
        assert!(err.to_string().contains('5'), "{err}");
    }

    // ---- determinism and content addressing ------------------------------

    #[test]
    fn the_measurement_is_l3_and_bit_reproducible() {
        let sim = PeriodogramSimulator::new(descriptor("test/level"));
        let config = SpectrumConfig::new(tone(64.0, 1024.0, 1024), 1024.0, WindowKind::Hann);
        assert_eq!(sim.level(), DeterminismLevel::L3);
        let a = sim.run(&config, 3).unwrap();
        let b = sim.run(&config, 3).unwrap();
        assert_eq!(a.output.canonical_bytes(), b.output.canonical_bytes());
        assert_eq!(a.level(), DeterminismLevel::L3);
        assert_eq!(a.seed, 3);
    }

    #[test]
    fn canonical_encoding_reflects_every_config_field() {
        let base = SpectrumConfig::new(vec![1.0, 0.0, -1.0, 0.0], 1024.0, WindowKind::Hann);
        assert_eq!(base.canonical_bytes(), base.clone().canonical_bytes());
        for other in [
            SpectrumConfig::new(vec![1.0, 0.0, -1.0, 0.5], 1024.0, WindowKind::Hann),
            SpectrumConfig::new(vec![1.0, 0.0, -1.0, 0.0], 2048.0, WindowKind::Hann),
            SpectrumConfig::new(vec![1.0, 0.0, -1.0, 0.0], 1024.0, WindowKind::Blackman),
        ]
        {
            assert_ne!(base.canonical_bytes(), other.canonical_bytes(), "{other:?}");
        }
    }

    #[test]
    fn every_window_kind_encodes_distinctly() {
        let kinds = [
            WindowKind::Rectangular,
            WindowKind::Hann,
            WindowKind::Hamming,
            WindowKind::Blackman,
            WindowKind::FlatTop,
        ];
        for (i, a) in kinds.iter().enumerate()
        {
            for b in &kinds[i + 1..]
            {
                assert_ne!(a.canonical_bytes(), b.canonical_bytes(), "{a:?} vs {b:?}");
            }
            assert_eq!(a.coefficients(8).len(), 8);
        }
    }

    #[test]
    fn the_vcr_records_then_replays_a_real_measurement() {
        let sim = PeriodogramSimulator::new(descriptor("test/vcr"));
        let config = SpectrumConfig::new(tone(64.0, 1024.0, 1024), 1024.0, WindowKind::Hann);
        let mut vcr = Vcr::new();

        let first = vcr.observe(&sim, &config, 0).unwrap();
        assert!(!first.replayed);
        assert!(vcr.observe(&sim, &config, 0).unwrap().replayed);
        assert_eq!(vcr.len(), 1);

        // A different window is a different measurement, not a replay.
        let other = SpectrumConfig::new(tone(64.0, 1024.0, 1024), 1024.0, WindowKind::FlatTop);
        assert!(!vcr.observe(&sim, &other, 0).unwrap().replayed);
        assert_eq!(vcr.len(), 2);
    }
}
