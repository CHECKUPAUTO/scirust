//! Measuring the spectrum of a *simulated* signal — the first stage whose
//! input is another stage's output.
//!
//! Every other backend in this crate carries its own data: a
//! [`SpectrumConfig`](crate::spectrum::SpectrumConfig) holds literal samples,
//! a [`ModelRun`](crate::model::ModelRun) holds literal parameters. That is
//! what let them be written in a file and driven from `sos run`, and it is
//! also their limit — a study could simulate a population *or* analyse a
//! signal, never simulate a population **and then** analyse what it produced.
//!
//! `sos-workflow`'s `consumes` closed the plumbing half of that gap: a stage
//! can now be handed an upstream stage's output ids. This closes the other
//! half, because being handed an object is not the same as knowing what to do
//! with it. [`TrajectorySpectrumConfig`] carries no signal at all — it names a
//! *component* of whatever trajectory the stage consumed, and everything else
//! about the measurement is derived from that object.
//!
//! ## The sample rate is derived, and that is the point
//!
//! A [`Trajectory`] is a sequence of `(t, y(t))`
//! pairs: it carries its own time grid. So this configuration does **not**
//! take a sample rate — it reads one off the trajectory.
//!
//! That is not convenience, it is the only way to catch the error that
//! matters. An FFT assumes uniform sampling. Two of this crate's ODE backends
//! do not produce it: [`Dopri5OdeSimulator`](crate::ode::Dopri5OdeSimulator)
//! is *adaptive*, so it takes short steps where the solution moves fast and
//! long ones where it does not. Feeding such a trajectory to a periodogram
//! produces a spectrum that looks entirely plausible and means nothing. If the
//! rate were restated by hand in the config, nothing in the system could
//! notice. Deriving it forces the check: [`uniform_sample_rate`] verifies
//! every consecutive gap agrees before returning a rate, and refuses the run
//! otherwise.
//!
//! This is the same principle as [`crate::root::CertifiedRoot`] carrying its
//! starting point — a number is not trustworthy on its own, only together with
//! the thing that makes it meaningful.
//!
//! ## Why Welch and not the periodogram
//!
//! [`welch`](crate::spectrum::WelchSimulator) averages segments, so it accepts
//! a record of any length and reports the tail it could not use. A trajectory
//! is whatever length `(t_end - t0) / step` works out to — 12 000 points, say
//! — and essentially never a power of two, which is what
//! [`fft_real`](scirust_signal::fft_real) demands of a single periodogram.
//!
//! Truncating a record to the previous power of two would silently change the
//! frequency resolution the user asked for, so this backend does not offer the
//! single periodogram at all rather than inventing a convention for it. Welch
//! already has one: `dropped_samples`, which was there before this module
//! existed.
//!
//! ## What the result records
//!
//! [`TrajectorySpectrum`] carries the averaged spectrum *and* the component
//! index and derived rate. The provenance parent says which trajectory was
//! measured; without the component it would not say **what about it**. For a
//! three-compartment SIR model, "the spectrum of the trajectory" is not a
//! statement — susceptible, infected and recovered have different ones.

use serde::{Deserialize, Serialize};

use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Body, DeterminismLevel};
use sos_simulation::{Observation, Result, SimDescriptor, SimError, Simulate};

use crate::ode::Trajectory;
use crate::solver::encode_f64;
use crate::spectrum::{AveragedSpectrum, WelchConfig, WelchSimulator, WindowKind};

/// How closely consecutive time steps must agree to count as a uniform grid.
///
/// Relative to the first gap, so it scales with the step size rather than
/// assuming one. Loose enough to absorb the float error of accumulating `t +=
/// step` a few thousand times, tight enough that an adaptive solver's genuinely
/// varying step — which differs by factors, not parts per billion — never
/// passes.
const UNIFORMITY_TOLERANCE: f64 = 1e-9;

/// What to measure, given a trajectory this stage consumed.
///
/// Carries no signal and no sample rate: both come from the upstream object.
/// See the module docs on why the rate is derived rather than declared.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectorySpectrumConfig {
    /// Which component of `y(t)` to measure, 0-based.
    ///
    /// A trajectory's state is a vector — for `ecology/lotka-volterra` that is
    /// `[prey, predator]` — and each component has its own spectrum.
    pub component: usize,
    /// The window applied to each segment.
    pub window: WindowKind,
    /// Samples per segment. Must be a power of two of at least 2.
    pub segment_len: usize,
    /// Samples shared between consecutive segments. Must be `< segment_len`.
    pub overlap: usize,
}

impl TrajectorySpectrumConfig {
    /// Construct a configuration. Validity is checked at run time, not here —
    /// a config is just data, and half of what makes it valid (does the
    /// trajectory have this many components?) is not knowable until the
    /// upstream object is in hand.
    #[must_use]
    pub fn new(component: usize, window: WindowKind, segment_len: usize, overlap: usize) -> Self {
        Self {
            component,
            window,
            segment_len,
            overlap,
        }
    }
}

impl Canonical for TrajectorySpectrumConfig {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.component as u64);
        enc.value(&self.window);
        enc.u64(self.segment_len as u64);
        enc.u64(self.overlap as u64);
    }
}

/// A spectrum measured from a simulated trajectory.
///
/// The spectrum alone would not say what was measured: the provenance parent
/// names the trajectory, but a trajectory has as many spectra as it has
/// components. `component` and `sample_rate_hz` make the result readable
/// without re-deriving them from an object that may no longer be at hand.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectorySpectrum {
    /// The averaged spectrum, with its segment count and dropped tail.
    pub spectrum: AveragedSpectrum,
    /// Which component of `y(t)` this measures.
    pub component: usize,
    /// The sampling rate **derived** from the trajectory's time grid, in
    /// hertz. Present so a reader need not have the trajectory to interpret
    /// the frequency axis.
    pub sample_rate_hz: f64,
}

impl Canonical for TrajectorySpectrum {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.spectrum);
        enc.u64(self.component as u64);
        encode_f64(enc, self.sample_rate_hz);
    }
}

/// The storable form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectorySpectrumBody {
    /// The measurement.
    pub measured: TrajectorySpectrum,
    /// The determinism level the backend realized.
    pub level: DeterminismLevel,
    /// The seed the run used.
    pub seed: u64,
}

impl TrajectorySpectrumBody {
    /// Flatten an observation into a storable body.
    #[must_use]
    pub fn from_observation(observation: Observation<TrajectorySpectrum>) -> Self {
        Self {
            level: observation.level(),
            seed: observation.seed,
            measured: observation.output,
        }
    }
}

impl Canonical for TrajectorySpectrumBody {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.measured);
        enc.value(&self.level);
        enc.u64(self.seed);
    }
}

impl Body for TrajectorySpectrumBody {
    const KIND: &'static str = "TrajectorySpectrum";
    const SCHEMA_VERSION: u32 = 1;
}

/// The sampling rate implied by a trajectory's time grid, in hertz.
///
/// # Errors
/// [`SimError::InvalidConfig`] if the trajectory has fewer than two samples,
/// if its time steps are not uniform (see the module docs — this is the check
/// that stops an adaptive solver's output being spectrally analysed as though
/// it were evenly sampled), or if the step is not finite and positive.
pub fn uniform_sample_rate(trajectory: &Trajectory) -> Result<f64> {
    if trajectory.len() < 2
    {
        return Err(SimError::InvalidConfig(format!(
            "a sampling rate needs at least 2 samples, got {}",
            trajectory.len()
        )));
    }
    let step = trajectory[1].0 - trajectory[0].0;
    if !step.is_finite() || step <= 0.0
    {
        return Err(SimError::InvalidConfig(format!(
            "the trajectory's time step must be finite and positive, got {step}"
        )));
    }
    let tolerance = UNIFORMITY_TOLERANCE * step;
    for (i, pair) in trajectory.windows(2).enumerate()
    {
        let gap = pair[1].0 - pair[0].0;
        if (gap - step).abs() > tolerance
        {
            return Err(SimError::InvalidConfig(format!(
                "the trajectory is not uniformly sampled, so a spectrum of it would be \
                 meaningless: step {step} at the start but {gap} between samples {i} and {} \
                 (an adaptive solver produces trajectories like this; use a fixed-step one)",
                i + 1
            )));
        }
    }
    Ok(1.0 / step)
}

/// One component of a trajectory, as a signal.
///
/// # Errors
/// [`SimError::InvalidConfig`] if the trajectory is empty or any state is too
/// short to have this component.
pub fn component_of(trajectory: &Trajectory, component: usize) -> Result<Vec<f64>> {
    if trajectory.is_empty()
    {
        return Err(SimError::InvalidConfig(
            "cannot measure a component of an empty trajectory".to_owned(),
        ));
    }
    let mut signal = Vec::with_capacity(trajectory.len());
    for (i, (_, state)) in trajectory.iter().enumerate()
    {
        let value = state.get(component).ok_or_else(|| {
            SimError::InvalidConfig(format!(
                "component {component} was asked for but state {i} has only {} \
                 (components are 0-based)",
                state.len()
            ))
        })?;
        signal.push(*value);
    }
    Ok(signal)
}

/// A configuration paired with the trajectory it reads.
///
/// [`Simulate`] takes a single `Config`, and this backend genuinely needs
/// both: the answer depends on the trajectory as much as on the settings, so
/// pairing them is what makes the [`Vcr`](sos_simulation::Vcr) key and the
/// content address cover everything that determined the result. A stage
/// handler builds one of these from the store; a direct caller can skip it
/// and use [`TrajectorySpectrumSimulator::measure`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryMeasurement {
    /// What to measure.
    pub config: TrajectorySpectrumConfig,
    /// The trajectory to measure it on.
    pub trajectory: Trajectory,
}

impl TrajectoryMeasurement {
    /// Pair a configuration with its trajectory.
    #[must_use]
    pub fn new(config: TrajectorySpectrumConfig, trajectory: Trajectory) -> Self {
        Self { config, trajectory }
    }
}

impl Canonical for TrajectoryMeasurement {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.config);
        enc.u64(self.trajectory.len() as u64);
        for (t, state) in &self.trajectory
        {
            encode_f64(enc, *t);
            enc.u64(state.len() as u64);
            for v in state
            {
                encode_f64(enc, *v);
            }
        }
    }
}

/// Measures the spectrum of one component of a simulated trajectory.
///
/// Delegates the numerics to [`WelchSimulator`] rather than repeating them:
/// what this backend adds is *where the signal comes from* and the checks that
/// makes necessary, not a second implementation of Welch's method.
#[derive(Debug, Clone)]
pub struct TrajectorySpectrumSimulator {
    descriptor: SimDescriptor,
    welch: WelchSimulator,
}

impl TrajectorySpectrumSimulator {
    /// A simulator identified by `descriptor`.
    #[must_use]
    pub fn new(descriptor: SimDescriptor) -> Self {
        let welch = WelchSimulator::new(SimDescriptor::new(
            "scirust-signal/welch",
            descriptor.version,
        ));
        Self { descriptor, welch }
    }

    /// Measure `trajectory` under `config`.
    ///
    /// # Errors
    /// [`SimError::InvalidConfig`] if the trajectory is not uniformly sampled,
    /// lacks the requested component, or is too short for the segmentation.
    pub fn measure(
        &self,
        config: &TrajectorySpectrumConfig,
        trajectory: &Trajectory,
        seed: u64,
    ) -> Result<Observation<TrajectorySpectrum>> {
        let sample_rate_hz = uniform_sample_rate(trajectory)?;
        let signal = component_of(trajectory, config.component)?;
        let welch = WelchConfig::new(
            signal,
            sample_rate_hz,
            config.window,
            config.segment_len,
            config.overlap,
        );
        let observed = self.welch.run(&welch, seed)?;
        let level = observed.level();
        Ok(Observation::new(
            TrajectorySpectrum {
                spectrum: observed.output,
                component: config.component,
                sample_rate_hz,
            },
            level,
            seed,
        ))
    }
}

impl Simulate for TrajectorySpectrumSimulator {
    type Config = TrajectoryMeasurement;
    type Output = TrajectorySpectrum;

    fn descriptor(&self) -> SimDescriptor {
        self.descriptor.clone()
    }

    fn level(&self) -> DeterminismLevel {
        // Exactly the periodogram's reasoning: a fixed sequence of f64
        // operations, no iteration to a tolerance and no sampling. Deriving
        // the rate adds a division, not a source of variation.
        DeterminismLevel::L3
    }

    fn run(
        &self,
        config: &TrajectoryMeasurement,
        seed: u64,
    ) -> Result<Observation<TrajectorySpectrum>> {
        self.measure(&config.config, &config.trajectory, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    use sos_simulation::Vcr;

    fn descriptor() -> SimDescriptor {
        SimDescriptor::new(
            "sos-scirust/trajectory-spectrum",
            sos_core::SemVer::new(1, 0, 0),
        )
    }

    /// A uniformly sampled trajectory whose first component is a tone.
    fn tone_trajectory(freq_hz: f64, rate_hz: f64, n: usize) -> Trajectory {
        (0..n)
            .map(|i| {
                let t = f64::from(u32::try_from(i).unwrap()) / rate_hz;
                (t, vec![(TAU * freq_hz * t).sin(), 0.5])
            })
            .collect()
    }

    #[test]
    fn the_sample_rate_comes_from_the_trajectorys_own_time_grid() {
        // 256 Hz because the samples are 1/256 s apart — nobody had to say so.
        let rate = uniform_sample_rate(&tone_trajectory(8.0, 256.0, 64)).unwrap();
        assert!((rate - 256.0).abs() < 1e-9, "{rate}");
    }

    #[test]
    fn an_adaptive_solvers_trajectory_is_refused_rather_than_measured() {
        // The error this module exists to prevent. A varying step is exactly
        // what `Dopri5OdeSimulator` produces, and a spectrum computed over it
        // looks perfectly plausible while meaning nothing.
        let uneven: Trajectory = vec![
            (0.0, vec![0.0]),
            (0.1, vec![1.0]),
            (0.15, vec![0.5]), // the solver took a shorter step here
            (0.25, vec![0.2]),
        ];
        let err = uniform_sample_rate(&uneven).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("not uniformly sampled"), "{message}");
        assert!(
            message.contains("adaptive"),
            "the fix must be named: {message}"
        );
    }

    #[test]
    fn float_drift_in_an_accumulated_grid_is_not_mistaken_for_drift_in_the_solver() {
        // A fixed-step solver accumulating `t += step` a few thousand times
        // does not produce exactly equal gaps. Rejecting that would make the
        // check useless in practice, so the tolerance is relative to the step.
        let mut trajectory = Trajectory::new();
        let mut t = 0.0;
        let step = 0.01;
        for _ in 0..4096
        {
            trajectory.push((t, vec![t.sin()]));
            t += step;
        }
        let rate = uniform_sample_rate(&trajectory).unwrap();
        assert!((rate - 100.0).abs() < 1e-6, "{rate}");
    }

    #[test]
    fn a_component_that_does_not_exist_is_an_error_not_a_panic() {
        let trajectory = tone_trajectory(8.0, 256.0, 16);
        let err = component_of(&trajectory, 5).unwrap_err();
        assert!(err.to_string().contains("component 5"), "{err}");
        assert!(err.to_string().contains("0-based"), "{err}");
        // The components that do exist are readable.
        assert_eq!(component_of(&trajectory, 1).unwrap().len(), 16);
    }

    #[test]
    fn a_short_or_empty_trajectory_is_refused() {
        assert!(uniform_sample_rate(&Trajectory::new()).is_err());
        assert!(uniform_sample_rate(&vec![(0.0, vec![1.0])]).is_err());
        assert!(component_of(&Trajectory::new(), 0).is_err());
    }

    #[test]
    fn a_backwards_time_grid_is_refused() {
        let backwards: Trajectory = vec![(1.0, vec![0.0]), (0.0, vec![1.0])];
        let err = uniform_sample_rate(&backwards).unwrap_err();
        assert!(err.to_string().contains("finite and positive"), "{err}");
    }

    #[test]
    fn measuring_a_simulated_tone_finds_its_frequency() {
        // End to end: a trajectory in, a spectrum out, with the peak where the
        // tone is — and the frequency axis derived, not declared.
        let sim = TrajectorySpectrumSimulator::new(descriptor());
        let config = TrajectorySpectrumConfig::new(0, WindowKind::Hann, 1024, 512);
        let trajectory = tone_trajectory(64.0, 1024.0, 4096);

        let observed = sim.measure(&config, &trajectory, 7).unwrap();
        let measured = &observed.output;
        assert!((measured.sample_rate_hz - 1024.0).abs() < 1e-9);
        assert_eq!(measured.component, 0);
        assert_eq!(measured.spectrum.segments, 7);

        let peak = measured
            .spectrum
            .psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // bin * rate / (2 * (len - 1)) = 64 Hz
        let peak_hz = peak as f64 * measured.sample_rate_hz
            / (2.0 * (measured.spectrum.psd.len() - 1) as f64);
        assert!((peak_hz - 64.0).abs() < 2.0, "peak at {peak_hz} Hz");
    }

    #[test]
    fn different_components_have_different_spectra() {
        // Why the component index is recorded: "the spectrum of the
        // trajectory" is not a statement about a multi-component system.
        let sim = TrajectorySpectrumSimulator::new(descriptor());
        let trajectory = tone_trajectory(64.0, 1024.0, 4096);

        let first = sim
            .measure(
                &TrajectorySpectrumConfig::new(0, WindowKind::Hann, 1024, 512),
                &trajectory,
                7,
            )
            .unwrap();
        let second = sim
            .measure(
                &TrajectorySpectrumConfig::new(1, WindowKind::Hann, 1024, 512),
                &trajectory,
                7,
            )
            .unwrap();

        assert_ne!(first.output.spectrum.psd, second.output.spectrum.psd);
        assert_eq!(second.output.component, 1);
        // The second component is a constant, so it is all at DC and flat
        // nowhere else.
        assert!(second.output.spectrum.psd[0] > 0.0);
    }

    #[test]
    fn the_measurement_is_l3_and_bit_reproducible() {
        let sim = TrajectorySpectrumSimulator::new(descriptor());
        let config = TrajectorySpectrumConfig::new(0, WindowKind::Hamming, 512, 256);
        let trajectory = tone_trajectory(32.0, 512.0, 2048);

        let a = sim.measure(&config, &trajectory, 1).unwrap();
        let b = sim.measure(&config, &trajectory, 1).unwrap();
        assert_eq!(a.level(), DeterminismLevel::L3);
        assert_eq!(a.output, b.output, "L3 means bit-identical");
    }

    #[test]
    fn the_vcr_records_then_replays_a_real_measurement() {
        let sim = TrajectorySpectrumSimulator::new(descriptor());
        let config = TrajectoryMeasurement::new(
            TrajectorySpectrumConfig::new(0, WindowKind::Hann, 512, 0),
            tone_trajectory(32.0, 512.0, 2048),
        );
        let mut vcr = Vcr::new();

        assert!(!vcr.observe(&sim, &config, 3).unwrap().replayed);
        assert!(vcr.observe(&sim, &config, 3).unwrap().replayed);
        assert_eq!(vcr.len(), 1);

        // A different component is a different measurement, not a replay —
        // which only holds because the trajectory and the component are both
        // inside the key.
        let other = TrajectoryMeasurement::new(
            TrajectorySpectrumConfig::new(1, WindowKind::Hann, 512, 0),
            config.trajectory.clone(),
        );
        assert!(!vcr.observe(&sim, &other, 3).unwrap().replayed);
        assert_eq!(vcr.len(), 2);
    }

    #[test]
    fn a_config_and_a_result_survive_a_file() {
        let config = TrajectorySpectrumConfig::new(2, WindowKind::Blackman, 256, 128);
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(
            serde_json::from_str::<TrajectorySpectrumConfig>(&json).unwrap(),
            config
        );

        let sim = TrajectorySpectrumSimulator::new(descriptor());
        let observed = sim
            .measure(
                &TrajectorySpectrumConfig::new(0, WindowKind::Hann, 512, 256),
                &tone_trajectory(32.0, 512.0, 2048),
                4,
            )
            .unwrap();
        let body = TrajectorySpectrumBody::from_observation(observed);
        let json = serde_json::to_string(&body).unwrap();
        let back: TrajectorySpectrumBody = serde_json::from_str(&json).unwrap();
        assert_eq!(back.measured.component, body.measured.component);
        assert_eq!(back.measured.sample_rate_hz, body.measured.sample_rate_hz);
        assert_eq!(back.measured.spectrum.psd, body.measured.spectrum.psd);
    }

    #[test]
    fn the_component_is_part_of_the_content_address() {
        // Two measurements of the same trajectory that differ only in which
        // component they read must not share a cache entry.
        let first = TrajectorySpectrumConfig::new(0, WindowKind::Hann, 512, 256);
        let second = TrajectorySpectrumConfig::new(1, WindowKind::Hann, 512, 256);
        assert_ne!(first.canonical_bytes(), second.canonical_bytes());

        // And so is the trajectory: the same settings over different data is a
        // different measurement.
        let a = TrajectoryMeasurement::new(first.clone(), tone_trajectory(32.0, 512.0, 1024));
        let b = TrajectoryMeasurement::new(first, tone_trajectory(64.0, 512.0, 1024));
        assert_ne!(a.canonical_bytes(), b.canonical_bytes());
    }
}
