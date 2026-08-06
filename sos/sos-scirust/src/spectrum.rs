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
    psd, psd_centroid, psd_flatness, psd_spread, spectral_centroid, spectral_flatness,
    spectral_spread,
};
use scirust_signal::{apply_window, blackman, fft_real, flattop, hamming, hanning, welch_psd};
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
    /// Spectral centroid in hertz — the **magnitude**-weighted mean
    /// frequency, which is what `scirust-signal`'s `spectral_centroid`
    /// computes.
    ///
    /// Not the same statistic as [`AveragedSpectrum::centroid_hz`], which is
    /// power-weighted; the two must not be compared. (This doc previously
    /// said "power-weighted" and was simply wrong about the function it
    /// describes.)
    pub centroid_hz: f64,
    /// Magnitude-weighted spectral spread in hertz about the centroid.
    pub spread_hz: f64,
    /// Spectral flatness in `[0, 1]` over **magnitudes**: near `1` is
    /// noise-like, near `0` is tonal. Not comparable with
    /// [`AveragedSpectrum::flatness`], which is taken over power.
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

/// One Welch measurement: the record, how it was sampled and shaped, and how
/// it is to be segmented.
#[derive(Debug, Clone, PartialEq)]
pub struct WelchConfig {
    /// The observed samples. Must be at least `segment_len` long.
    pub signal: Vec<f64>,
    /// The sampling rate in hertz. Must be finite and positive.
    pub sample_rate_hz: f64,
    /// The window applied to each segment.
    pub window: WindowKind,
    /// Samples per segment. Must be a power of two of at least 2 — the
    /// resolution/stability dial: smaller segments mean more of them.
    pub segment_len: usize,
    /// Samples shared between consecutive segments. Must be `< segment_len`.
    pub overlap: usize,
}

impl WelchConfig {
    /// Construct a Welch configuration. Validity is checked by
    /// [`WelchSimulator::run`], not here — a config is just data.
    #[must_use]
    pub fn new(
        signal: Vec<f64>,
        sample_rate_hz: f64,
        window: WindowKind,
        segment_len: usize,
        overlap: usize,
    ) -> Self {
        Self {
            signal,
            sample_rate_hz,
            window,
            segment_len,
            overlap,
        }
    }
}

impl Canonical for WelchConfig {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&ExactF64Seq(&self.signal));
        encode_f64(enc, self.sample_rate_hz);
        enc.value(&self.window);
        enc.u64(self.segment_len as u64);
        enc.u64(self.overlap as u64);
    }
}

/// Serialized form of a [`WelchConfig`] — exact decimal text for the floats,
/// same reasoning as [`ConfigRepr`].
#[derive(Serialize, Deserialize)]
struct WelchConfigRepr {
    signal: Vec<String>,
    sample_rate_hz: String,
    window: WindowKind,
    segment_len: usize,
    overlap: usize,
}

impl Serialize for WelchConfig {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        WelchConfigRepr {
            signal: self
                .signal
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
            sample_rate_hz: self.sample_rate_hz.to_string(),
            window: self.window,
            segment_len: self.segment_len,
            overlap: self.overlap,
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for WelchConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let r = WelchConfigRepr::deserialize(d)?;
        let parse = |s: &str| s.parse::<f64>().map_err(D::Error::custom);
        Ok(WelchConfig {
            signal: r
                .signal
                .iter()
                .map(|v| parse(v))
                .collect::<std::result::Result<Vec<f64>, _>>()?,
            sample_rate_hz: parse(&r.sample_rate_hz)?,
            window: r.window,
            segment_len: r.segment_len,
            overlap: r.overlap,
        })
    }
}

/// A Welch power-spectral-density estimate, with the descriptor that says how
/// much it is worth.
///
/// ## A third kind of metadata
///
/// This crate's `L2` backends carry a **tolerance certificate**: the answer is
/// within a declared ε ([`crate::ode::CertifiedTrajectory`],
/// [`crate::quadrature::CertifiedIntegral`]). Its `L1` estimator carries a
/// **standard error** from sampling ([`crate::nmc`]). This is neither. A Welch
/// estimate is `L3` — bit-reproducible, no tolerance, nothing sampled — and
/// yet its *quality as an estimate* still varies, with `1/segments`. So
/// [`segments`](Self::segments) is a **statistical quality descriptor on an
/// otherwise exactly-reproducible result**, and it belongs in the output for
/// the same reason the other two do: a number reported without it cannot be
/// read correctly.
///
/// [`dropped_samples`](Self::dropped_samples) is here for the adjacent
/// reason — a record that does not divide evenly into segments loses its tail,
/// and a result that quietly used less data than it was given overstates
/// itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "AveragedSpectrumRepr", into = "AveragedSpectrumRepr")]
pub struct AveragedSpectrum {
    /// Averaged power spectral density, DC to Nyquist, on the same scale as
    /// [`Spectrum::psd`].
    pub psd: Vec<f64>,
    /// Bin spacing in hertz — `sample_rate / segment_len`. Coarser than a
    /// whole-record periodogram's: that is the resolution traded for stability.
    pub bin_width_hz: f64,
    /// The window applied to each segment.
    pub window: WindowKind,
    /// How many segments were averaged. See the type docs.
    pub segments: usize,
    /// Trailing samples that did not fill a segment and were not used.
    pub dropped_samples: usize,
    /// Spectral centroid in hertz — the **power**-weighted mean frequency,
    /// from `scirust-signal`'s `psd_centroid`.
    ///
    /// Power-weighted rather than magnitude-weighted because a Welch estimate
    /// *is* a power spectral density: the magnitude-domain functions
    /// [`Spectrum`] uses take a complex spectrum, which this is not. The two
    /// are different statistics and must not be compared — see
    /// [`Spectrum::centroid_hz`].
    pub centroid_hz: f64,
    /// Power-weighted spectral spread in hertz about the centroid.
    pub spread_hz: f64,
    /// Spectral flatness in `[0, 1]` over **power** — the textbook Wiener
    /// entropy. Saturates at exactly `0` for a spectrum with any empty bin;
    /// `scirust-signal`'s `psd_flatness` documents why.
    pub flatness: f64,
}

/// Serialized form of an [`AveragedSpectrum`].
#[derive(Serialize, Deserialize, Clone)]
struct AveragedSpectrumRepr {
    psd: Vec<String>,
    bin_width_hz: String,
    window: WindowKind,
    segments: usize,
    dropped_samples: usize,
    centroid_hz: String,
    spread_hz: String,
    flatness: String,
}

impl From<AveragedSpectrum> for AveragedSpectrumRepr {
    fn from(a: AveragedSpectrum) -> Self {
        Self {
            psd: a.psd.iter().map(std::string::ToString::to_string).collect(),
            bin_width_hz: a.bin_width_hz.to_string(),
            window: a.window,
            segments: a.segments,
            dropped_samples: a.dropped_samples,
            centroid_hz: a.centroid_hz.to_string(),
            spread_hz: a.spread_hz.to_string(),
            flatness: a.flatness.to_string(),
        }
    }
}

impl From<AveragedSpectrumRepr> for AveragedSpectrum {
    fn from(r: AveragedSpectrumRepr) -> Self {
        // `serde(from = ...)` cannot fail, so an unparsable float would have to
        // become a silent default here. It cannot occur through this crate's
        // own writer, and a corrupted record is caught by the store's content
        // check on the way out — the id will not match.
        let parse = |s: &String| s.parse::<f64>().unwrap_or(f64::NAN);
        Self {
            psd: r.psd.iter().map(parse).collect(),
            bin_width_hz: parse(&r.bin_width_hz),
            window: r.window,
            segments: r.segments,
            dropped_samples: r.dropped_samples,
            centroid_hz: parse(&r.centroid_hz),
            spread_hz: parse(&r.spread_hz),
            flatness: parse(&r.flatness),
        }
    }
}

impl Canonical for AveragedSpectrum {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&ExactF64Seq(&self.psd));
        encode_f64(enc, self.bin_width_hz);
        enc.value(&self.window);
        enc.u64(self.segments as u64);
        enc.u64(self.dropped_samples as u64);
        encode_f64(enc, self.centroid_hz);
        encode_f64(enc, self.spread_hz);
        encode_f64(enc, self.flatness);
    }
}

/// A real [`Simulate`] backend computing Welch's averaged periodogram.
///
/// `L3`, exactly as [`PeriodogramSimulator`] is, and for the same reason: a
/// fixed sequence of `f64` operations with no iteration, tolerance or
/// sampling. What differs is not reproducibility but *estimator quality* —
/// this one's variance falls as `1/segments`, where a single periodogram's
/// does not fall at all. That is the whole reason to reach for it, and
/// [`AveragedSpectrum::segments`] is how a result says how much of it it got.
#[derive(Debug, Clone)]
pub struct WelchSimulator {
    descriptor: SimDescriptor,
}

impl WelchSimulator {
    /// A named, versioned Welch backend.
    #[must_use]
    pub fn new(descriptor: SimDescriptor) -> Self {
        Self { descriptor }
    }
}

impl Simulate for WelchSimulator {
    type Config = WelchConfig;
    type Output = AveragedSpectrum;

    fn descriptor(&self) -> SimDescriptor {
        self.descriptor.clone()
    }

    fn level(&self) -> DeterminismLevel {
        DeterminismLevel::L3
    }

    fn run(&self, config: &WelchConfig, seed: u64) -> Result<Observation<AveragedSpectrum>> {
        if !(config.sample_rate_hz.is_finite() && config.sample_rate_hz > 0.0)
        {
            return Err(SimError::InvalidConfig(format!(
                "sample rate must be finite and positive (got {})",
                config.sample_rate_hz
            )));
        }
        // Every remaining precondition is `welch_psd`'s own, and it reports
        // them as structured errors rather than asserting — so they are mapped
        // rather than re-checked here. That is the opposite of what
        // `PeriodogramSimulator` must do for `fft_real`, which panics.
        let coefficients = config.window.coefficients(config.segment_len);
        let estimate = welch_psd(
            &config.signal,
            config.segment_len,
            config.overlap,
            &coefficients,
        )
        .map_err(|e| SimError::InvalidConfig(e.to_string()))?;

        #[allow(clippy::cast_precision_loss)] // segment lengths are far below 2^53
        let bin_width_hz = config.sample_rate_hz / config.segment_len as f64;
        Ok(Observation::new(
            AveragedSpectrum {
                centroid_hz: psd_centroid(&estimate.psd, config.sample_rate_hz),
                spread_hz: psd_spread(&estimate.psd, config.sample_rate_hz),
                flatness: psd_flatness(&estimate.psd),
                psd: estimate.psd,
                bin_width_hz,
                window: config.window,
                segments: estimate.segments,
                dropped_samples: estimate.dropped_samples,
            },
            self.level(),
            seed,
        ))
    }
}

/// The body of a stored Welch measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AveragedSpectrumBody {
    /// The averaged spectrum, with its quality descriptor.
    pub spectrum: AveragedSpectrum,
    /// The determinism level the backend realized.
    pub level: DeterminismLevel,
    /// The seed the run used.
    pub seed: u64,
}

impl AveragedSpectrumBody {
    /// Flatten an observation into a storable body.
    #[must_use]
    pub fn from_observation(observation: Observation<AveragedSpectrum>) -> Self {
        let (level, seed) = (observation.level(), observation.seed);
        Self {
            spectrum: observation.output,
            level,
            seed,
        }
    }

    /// Rebuild the [`Observation`] this body was stored from.
    #[must_use]
    pub fn observation(&self) -> Observation<AveragedSpectrum> {
        Observation::new(self.spectrum.clone(), self.level, self.seed)
    }
}

impl Canonical for AveragedSpectrumBody {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.spectrum);
        enc.value(&self.level);
        enc.u64(self.seed);
    }
}

impl Body for AveragedSpectrumBody {
    const KIND: &'static str = "AveragedSpectrum";
    const SCHEMA_VERSION: u32 = 1;
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

    // ---- Welch: the estimator whose variance actually falls -------------

    fn welch(config: &WelchConfig) -> AveragedSpectrum {
        WelchSimulator::new(descriptor("test/welch"))
            .run(config, 0)
            .expect("a valid config must measure")
            .output
    }

    #[test]
    fn welch_averaging_delivers_the_stability_a_periodogram_cannot() {
        // The pair of claims `PeriodogramSimulator`'s docs make, now both
        // testable against real backends rather than a hand-rolled average.
        let noisy = white_noise(4096, 0x5EED);
        let single = measure(&SpectrumConfig::new(
            noisy[..256].to_vec(),
            1024.0,
            WindowKind::Rectangular,
        ));
        let averaged = welch(&WelchConfig::new(
            noisy,
            1024.0,
            WindowKind::Rectangular,
            256,
            0,
        ));

        assert_eq!(averaged.segments, 16);
        assert_eq!(averaged.dropped_samples, 0);
        let (one, many) = (
            relative_dispersion(&single.psd),
            relative_dispersion(&averaged.psd),
        );
        assert!(
            many < one / 2.5,
            "16 segments must visibly stabilize the estimate: {one} -> {many}"
        );
        // And the resolution paid for it: same bin width as one 256-sample
        // periodogram, not the 4096-sample one the record could have bought.
        assert!((averaged.bin_width_hz - 4.0).abs() < 1e-12);
    }

    #[test]
    fn a_single_rectangular_segment_agrees_with_the_plain_periodogram() {
        // The two backends compute the same quantity; Welch just averages it.
        let signal = tone(64.0, 1024.0, 1024);
        let direct = measure(&SpectrumConfig::new(
            signal.clone(),
            1024.0,
            WindowKind::Rectangular,
        ));
        let averaged = welch(&WelchConfig::new(
            signal,
            1024.0,
            WindowKind::Rectangular,
            1024,
            0,
        ));
        assert_eq!(averaged.segments, 1);
        assert_eq!(averaged.bin_width_hz, direct.bin_width_hz);
        for (i, (a, b)) in averaged.psd.iter().zip(&direct.psd).enumerate()
        {
            assert!((a - b).abs() < 1e-12, "bin {i}: {a} vs {b}");
        }
    }

    #[test]
    fn a_tone_survives_the_averaging() {
        let averaged = welch(&WelchConfig::new(
            tone(64.0, 1024.0, 4096),
            1024.0,
            WindowKind::Hann,
            1024,
            512,
        ));
        let peak = averaged
            .psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, 64);
        assert_eq!(averaged.segments, 7, "50% overlap over 4096 samples");
    }

    #[test]
    fn a_dropped_tail_is_reported_rather_than_hidden() {
        let averaged = welch(&WelchConfig::new(
            white_noise(300, 1),
            1024.0,
            WindowKind::Rectangular,
            256,
            0,
        ));
        assert_eq!(averaged.segments, 1);
        assert_eq!(averaged.dropped_samples, 44);
    }

    #[test]
    fn welch_rejects_bad_configurations_as_structured_errors() {
        let sim = WelchSimulator::new(descriptor("test/welch-guard"));
        let signal = white_noise(1024, 2);
        let bad = [
            // Sample rate is this adapter's own check.
            WelchConfig::new(signal.clone(), 0.0, WindowKind::Hann, 256, 0),
            WelchConfig::new(signal.clone(), f64::NAN, WindowKind::Hann, 256, 0),
            // The rest are `welch_psd`'s, mapped rather than re-checked.
            WelchConfig::new(signal.clone(), 1024.0, WindowKind::Hann, 300, 0),
            WelchConfig::new(signal.clone(), 1024.0, WindowKind::Hann, 256, 256),
            WelchConfig::new(vec![0.5; 100], 1024.0, WindowKind::Hann, 256, 0),
        ];
        for config in bad
        {
            assert!(
                matches!(sim.run(&config, 0), Err(SimError::InvalidConfig(_))),
                "{config:?} must be rejected"
            );
        }
    }

    #[test]
    fn welch_is_l3_and_bit_reproducible() {
        let sim = WelchSimulator::new(descriptor("test/welch-level"));
        let config = WelchConfig::new(white_noise(2048, 9), 1024.0, WindowKind::Hann, 256, 128);
        assert_eq!(sim.level(), DeterminismLevel::L3);
        let a = sim.run(&config, 4).unwrap();
        let b = sim.run(&config, 4).unwrap();
        assert_eq!(a.output.canonical_bytes(), b.output.canonical_bytes());
        assert_eq!(a.seed, 4);
    }

    #[test]
    fn the_segmentation_is_part_of_the_content_address() {
        // Two Welch estimates of the same record at different segmentations
        // are different measurements, and must not share a cache entry.
        let signal = white_noise(2048, 3);
        let base = WelchConfig::new(signal.clone(), 1024.0, WindowKind::Hann, 256, 0);
        for other in [
            WelchConfig::new(signal.clone(), 1024.0, WindowKind::Hann, 512, 0),
            WelchConfig::new(signal.clone(), 1024.0, WindowKind::Hann, 256, 128),
            WelchConfig::new(signal.clone(), 1024.0, WindowKind::Blackman, 256, 0),
            WelchConfig::new(signal, 2048.0, WindowKind::Hann, 256, 0),
        ]
        {
            assert_ne!(base.canonical_bytes(), other.canonical_bytes(), "{other:?}");
        }
    }

    #[test]
    fn the_averaged_spectrum_summarizes_itself_in_the_power_domain() {
        // The gap this closes: `Spectrum` carried features and
        // `AveragedSpectrum` did not, because `scirust-signal`'s
        // magnitude-domain functions cannot read a PSD at all.
        let averaged = welch(&WelchConfig::new(
            tone(64.0, 1024.0, 4096),
            1024.0,
            WindowKind::Hann,
            1024,
            512,
        ));
        assert!(
            (averaged.centroid_hz - 64.0).abs() < 5.0,
            "the centroid must find the tone: {}",
            averaged.centroid_hz
        );
        assert!(averaged.spread_hz > 0.0);
        assert!((0.0..=1.0).contains(&averaged.flatness));
        assert!(averaged.flatness < 0.5, "a tone is not flat");
    }

    #[test]
    fn the_two_backends_report_different_statistics_under_the_same_name() {
        // Deliberate, and the reason both types document it: `Spectrum`'s
        // features are magnitude-weighted (that is what
        // `scirust-signal::spectral_centroid` computes) and
        // `AveragedSpectrum`'s are power-weighted. On a signal that is not a
        // single tone they disagree, so the two must not be compared.
        let rate = 1024.0;
        let mixed: Vec<f64> = tone(64.0, rate, 1024)
            .iter()
            .zip(tone(320.0, rate, 1024))
            .map(|(lo, hi)| 4.0 * lo + hi)
            .collect();

        let magnitude = measure(&SpectrumConfig::new(
            mixed.clone(),
            rate,
            WindowKind::Rectangular,
        ));
        let power = welch(&WelchConfig::new(
            mixed,
            rate,
            WindowKind::Rectangular,
            1024,
            0,
        ));

        // Same record, same window, one segment — only the weighting differs.
        assert_eq!(power.segments, 1);
        assert!(
            power.centroid_hz < magnitude.centroid_hz - 5.0,
            "power weighting must favour the loud low tone: {} vs {}",
            power.centroid_hz,
            magnitude.centroid_hz
        );
    }

    #[test]
    fn a_welch_config_and_result_survive_a_file() {
        let config = WelchConfig::new(white_noise(1024, 7), 1024.0, WindowKind::FlatTop, 256, 64);
        let back: WelchConfig = serde_json::from_str(&serde_json::to_string(&config).unwrap())
            .expect("a config must round-trip");
        assert_eq!(back, config);
        assert_eq!(back.canonical_bytes(), config.canonical_bytes());

        let body = AveragedSpectrumBody::from_observation(
            WelchSimulator::new(descriptor("test/welch-file"))
                .run(&config, 2)
                .unwrap(),
        );
        let reloaded: AveragedSpectrumBody =
            serde_json::from_str(&serde_json::to_string(&body).unwrap()).unwrap();
        assert_eq!(reloaded.canonical_bytes(), body.canonical_bytes());
        assert_eq!(reloaded.spectrum.segments, body.spectrum.segments);
        for (a, b) in body.spectrum.psd.iter().zip(&reloaded.spectrum.psd)
        {
            assert_eq!(a.to_bits(), b.to_bits());
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
