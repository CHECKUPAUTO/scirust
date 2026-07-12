//! Kalman and Interacting-Multiple-Model (IMM) track filtering.
//!
//! The α–β filter ([`super::track`]) is a fixed-gain steady-state Kalman filter:
//! cheap and unbiased on a constant-velocity track, but its gains never adapt,
//! so it lags a manoeuvring target and cannot report its own uncertainty. This
//! module adds the two filters that lift those limits:
//!
//! - [`KalmanCV`] — a full constant-velocity **Kalman filter** with a live
//!   covariance. It carries a continuous-white-noise-acceleration process model,
//!   so its gain adapts to the measurement/process-noise balance and it exposes
//!   both the state uncertainty and the measurement likelihood.
//! - [`Imm`] — the **Interacting Multiple Model** estimator: a bank of Kalman
//!   filters (typically a quiet, low-process-noise model and an agile,
//!   high-process-noise one) blended each frame by Markov mode probabilities.
//!   During steady flight the quiet model dominates (smooth, low variance); the
//!   instant the target manoeuvres the agile model's likelihood wins and takes
//!   over, so the estimate follows the manoeuvre with far less lag than any
//!   single fixed model. Dependency-free — the 2-D state keeps every matrix a
//!   `2×2` handled in closed form.

use std::f64::consts::PI;

/// A constant-velocity Kalman filter over the scalar state `x = (p, v)`
/// (position and velocity), with the `2×2` covariance carried explicitly.
///
/// The dynamics are `p ← p + v·dt`, `v ← v`, driven by continuous white-noise
/// acceleration of power-spectral density `q`; measurements observe position
/// with variance `r`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KalmanCV {
    p: f64,
    v: f64,
    // Symmetric covariance [[p00, p01], [p01, p11]].
    p00: f64,
    p01: f64,
    p11: f64,
    dt: f64,
    q: f64,
    r: f64,
}

impl KalmanCV {
    /// A filter at frame interval `dt`, process-noise density `q`, measurement
    /// variance `r`, initialised at position `x0` with zero velocity and an
    /// isotropic initial covariance `var0` on both position and velocity.
    pub fn new(dt: f64, q: f64, r: f64, x0: f64, var0: f64) -> Self {
        Self {
            p: x0,
            v: 0.0,
            p00: var0,
            p01: 0.0,
            p11: var0,
            dt,
            q,
            r,
        }
    }

    /// Time update: advance the state by one frame and grow the covariance by
    /// `F·P·Fᵀ + Q`, with `Q` the continuous-white-noise-acceleration model
    /// `q·[[dt³/3, dt²/2], [dt²/2, dt]]`.
    pub fn predict(&mut self) {
        let dt = self.dt;
        self.p += self.v * dt;
        let (p00, p01, p11) = (self.p00, self.p01, self.p11);
        // F·P·Fᵀ with F = [[1, dt], [0, 1]].
        let n00 = p00 + 2.0 * dt * p01 + dt * dt * p11;
        let n01 = p01 + dt * p11;
        let n11 = p11;
        // + Q.
        self.p00 = n00 + self.q * dt * dt * dt / 3.0;
        self.p01 = n01 + self.q * dt * dt / 2.0;
        self.p11 = n11 + self.q * dt;
    }

    /// Measurement update with position measurement `z`; returns the Gaussian
    /// likelihood `𝒩(residual; 0, S)` of the innovation, which the [`Imm`] uses
    /// to weight the model. Observation matrix `H = [1, 0]`.
    pub fn update(&mut self, z: f64) -> f64 {
        let y = z - self.p; // innovation
        let s = self.p00 + self.r; // innovation variance S = H·P·Hᵀ + R
        let k0 = self.p00 / s; // Kalman gain K = P·Hᵀ / S
        let k1 = self.p01 / s;
        self.p += k0 * y;
        self.v += k1 * y;
        // P ← (I − K·H)·P; symmetry is preserved since (1−k0)·p01 = p01 − k1·p00.
        let (p00, p01, p11) = (self.p00, self.p01, self.p11);
        self.p00 = (1.0 - k0) * p00;
        self.p01 = (1.0 - k0) * p01;
        self.p11 = p11 - k1 * p01;
        (-(y * y) / (2.0 * s)).exp() / (2.0 * PI * s).sqrt()
    }

    /// Predict then update in one frame; returns the innovation likelihood.
    pub fn step(&mut self, z: f64) -> f64 {
        self.predict();
        self.update(z)
    }

    /// The current filtered position.
    pub fn position(&self) -> f64 {
        self.p
    }

    /// The current filtered velocity.
    pub fn velocity(&self) -> f64 {
        self.v
    }

    /// The current position-estimate variance (the `(0,0)` covariance entry).
    pub fn position_variance(&self) -> f64 {
        self.p00
    }
}

/// The **Interacting Multiple Model** estimator: a bank of [`KalmanCV`] filters
/// blended each frame by Markov mode probabilities.
///
/// Each [`step`](Self::step) mixes the models' states in proportion to the
/// mode-transition-weighted probabilities, runs every model's predict/update on
/// the measurement, updates the mode probabilities from the model likelihoods,
/// and reports the probability-weighted combined estimate.
#[derive(Debug, Clone)]
pub struct Imm {
    models: Vec<KalmanCV>,
    mu: Vec<f64>,
    trans: Vec<Vec<f64>>,
}

impl Imm {
    /// An IMM over `models` with Markov mode-transition matrix `trans`
    /// (`trans[i][j]` = probability of switching from model `i` to `j`) and
    /// initial mode probabilities `mu0`. Both `trans` rows and `mu0` are
    /// normalised to sum to one.
    pub fn new(models: Vec<KalmanCV>, trans: Vec<Vec<f64>>, mu0: Vec<f64>) -> Self {
        let n = models.len();
        let mut mu = mu0;
        mu.resize(n, 0.0);
        normalise(&mut mu);
        let mut trans = trans;
        trans.resize(n, vec![0.0; n]);
        for row in &mut trans
        {
            row.resize(n, 0.0);
            normalise(row);
        }
        Self { models, mu, trans }
    }

    /// Advance one frame with position measurement `z`.
    #[allow(clippy::needless_range_loop)] // mode-mixing sweep — indices are the algorithm
    pub fn step(&mut self, z: f64) {
        let n = self.models.len();
        if n == 0
        {
            return;
        }
        // Predicted mode probabilities c̄_j = Σ_i trans[i][j]·μ_i.
        let cbar: Vec<f64> = (0..n)
            .map(|j| (0..n).map(|i| self.trans[i][j] * self.mu[i]).sum())
            .collect();
        let states: Vec<(f64, f64)> = self.models.iter().map(|m| (m.p, m.v)).collect();
        let covs: Vec<(f64, f64, f64)> =
            self.models.iter().map(|m| (m.p00, m.p01, m.p11)).collect();
        // Mixing: each model j starts from the mixed estimate over all i, with
        // weights w_{i|j} = trans[i][j]·μ_i / c̄_j.
        for j in 0..n
        {
            let cj = cbar[j].max(1e-300);
            let (mut mp, mut mv) = (0.0, 0.0);
            for i in 0..n
            {
                let w = self.trans[i][j] * self.mu[i] / cj;
                mp += w * states[i].0;
                mv += w * states[i].1;
            }
            let (mut c00, mut c01, mut c11) = (0.0, 0.0, 0.0);
            for i in 0..n
            {
                let w = self.trans[i][j] * self.mu[i] / cj;
                let dp = states[i].0 - mp;
                let dv = states[i].1 - mv;
                c00 += w * (covs[i].0 + dp * dp);
                c01 += w * (covs[i].1 + dp * dv);
                c11 += w * (covs[i].2 + dv * dv);
            }
            let m = &mut self.models[j];
            m.p = mp;
            m.v = mv;
            m.p00 = c00;
            m.p01 = c01;
            m.p11 = c11;
        }
        // Model-matched filtering; collect likelihoods.
        let like: Vec<f64> = self.models.iter_mut().map(|m| m.step(z)).collect();
        // Mode-probability update μ_j ∝ c̄_j·Λ_j.
        let mut newmu: Vec<f64> = (0..n).map(|j| cbar[j] * like[j]).collect();
        let norm: f64 = newmu.iter().sum();
        if norm <= 1e-300
        {
            // Likelihoods underflowed — fall back to the predicted modes.
            newmu = cbar;
        }
        normalise(&mut newmu);
        self.mu = newmu;
    }

    /// The combined position estimate `Σ_j μ_j·p_j`.
    pub fn position(&self) -> f64 {
        self.models
            .iter()
            .zip(&self.mu)
            .map(|(m, &w)| w * m.p)
            .sum()
    }

    /// The combined velocity estimate `Σ_j μ_j·v_j`.
    pub fn velocity(&self) -> f64 {
        self.models
            .iter()
            .zip(&self.mu)
            .map(|(m, &w)| w * m.v)
            .sum()
    }

    /// The current mode probabilities, one per model, summing to one.
    pub fn mode_probabilities(&self) -> &[f64] {
        &self.mu
    }
}

/// Scale a vector to sum to one; a degenerate all-zero vector becomes uniform.
fn normalise(v: &mut [f64]) {
    let s: f64 = v.iter().sum();
    if s > 0.0
    {
        for x in v.iter_mut()
        {
            *x /= s;
        }
    }
    else if !v.is_empty()
    {
        let u = 1.0 / v.len() as f64;
        for x in v.iter_mut()
        {
            *x = u;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kalman_recovers_constant_velocity() {
        // Noise-free constant-velocity truth: the filter's velocity converges to
        // the true slope and the position to truth.
        let (dt, v) = (0.5, 3.0);
        let mut f = KalmanCV::new(dt, 1e-6, 1.0, 0.0, 10.0);
        for k in 1..=200
        {
            f.step(v * k as f64 * dt);
        }
        let truth = v * 200.0 * dt;
        assert!(
            (f.velocity() - v).abs() < 1e-2,
            "vel {} vs {v}",
            f.velocity()
        );
        assert!(
            (f.position() - truth).abs() < 1e-1,
            "pos {} vs {truth}",
            f.position()
        );
    }

    #[test]
    fn kalman_update_reduces_position_variance_and_reaches_steady_state() {
        let mut f = KalmanCV::new(1.0, 0.5, 2.0, 0.0, 100.0);
        // Warm up to steady state on a constant-velocity ramp.
        for k in 1..=100
        {
            f.step(1.5 * k as f64);
        }
        let steady = f.position_variance();
        // One more predict grows the variance; the paired update shrinks it back
        // to (essentially) the same steady-state value.
        f.predict();
        let after_predict = f.position_variance();
        f.update(1.5 * 101.0);
        let after_update = f.position_variance();
        assert!(after_update < after_predict, "update must reduce variance");
        assert!(
            (after_update - steady).abs() < 1e-6,
            "should be at steady state"
        );
    }

    #[test]
    fn kalman_likelihood_peaks_at_the_prediction() {
        let mut f = KalmanCV::new(1.0, 0.1, 1.0, 0.0, 5.0);
        for k in 1..=30
        {
            f.step(2.0 * k as f64);
        }
        // From the same predicted state, a measurement at the prediction is more
        // likely than one five sigma away.
        let mut a = f;
        a.predict();
        let pred = a.position();
        let mut b = a;
        let on = a.update(pred);
        let off = b.update(pred + 5.0);
        assert!(on > off, "on-target {on} should beat off-target {off}");
    }

    #[test]
    fn imm_mode_probabilities_are_a_valid_distribution() {
        let m0 = KalmanCV::new(1.0, 1e-3, 1.0, 0.0, 10.0);
        let m1 = KalmanCV::new(1.0, 5.0, 1.0, 0.0, 10.0);
        let mut imm = Imm::new(
            vec![m0, m1],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            vec![0.5, 0.5],
        );
        for k in 1..=25
        {
            imm.step(1.0 * k as f64);
        }
        let mu = imm.mode_probabilities();
        assert_eq!(mu.len(), 2);
        assert!((mu.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(mu.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn imm_favours_the_quiet_model_on_a_steady_target() {
        // A constant-velocity target: the low-process-noise model explains it
        // with a tighter innovation variance, so it should dominate.
        let quiet = KalmanCV::new(1.0, 1e-4, 1.0, 0.0, 10.0);
        let agile = KalmanCV::new(1.0, 10.0, 1.0, 0.0, 10.0);
        let mut imm = Imm::new(
            vec![quiet, agile],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.5, 0.5],
        );
        for k in 1..=60
        {
            imm.step(2.0 * k as f64);
        }
        let mu = imm.mode_probabilities();
        assert!(mu[0] > mu[1], "quiet model should dominate: {mu:?}");
    }

    #[test]
    fn imm_beats_a_single_quiet_filter_through_a_manoeuvre() {
        // Truth: constant velocity +1/frame, then a sharp reversal to −2/frame at
        // frame K. A lone quiet filter lags the reversal; the IMM's agile model
        // takes over and tracks it with smaller error.
        let (dt, k_turn, n) = (1.0_f64, 25usize, 45usize);
        let truth = |k: usize| -> f64 {
            if k <= k_turn
            {
                k as f64
            }
            else
            {
                k_turn as f64 - 2.0 * (k - k_turn) as f64
            }
        };
        let quiet_params = (dt, 2e-3, 1.0, 0.0, 1.0);
        let mut lone = KalmanCV::new(
            quiet_params.0,
            quiet_params.1,
            quiet_params.2,
            quiet_params.3,
            quiet_params.4,
        );
        let mut imm = Imm::new(
            vec![
                KalmanCV::new(dt, 2e-3, 1.0, 0.0, 1.0),
                KalmanCV::new(dt, 5.0, 1.0, 0.0, 1.0),
            ],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            vec![0.5, 0.5],
        );
        let mut mu1_before = 0.0;
        let (mut lone_err, mut imm_err) = (0.0, 0.0);
        for k in 1..=n
        {
            let z = truth(k);
            lone.step(z);
            imm.step(z);
            if k == k_turn
            {
                mu1_before = imm.mode_probabilities()[1];
            }
            if k > k_turn && k <= k_turn + 10
            {
                lone_err += (lone.position() - z).abs();
                imm_err += (imm.position() - z).abs();
            }
        }
        let mu1_after = imm.mode_probabilities()[1];
        assert!(
            mu1_after > mu1_before,
            "agile-model probability should rise on the manoeuvre: {mu1_before} -> {mu1_after}"
        );
        assert!(
            imm_err < lone_err,
            "IMM error {imm_err} should beat the lone quiet filter {lone_err}"
        );
    }

    #[test]
    fn imm_empty_bank_is_inert() {
        let mut imm = Imm::new(Vec::new(), Vec::new(), Vec::new());
        imm.step(1.0);
        assert_eq!(imm.position(), 0.0);
        assert!(imm.mode_probabilities().is_empty());
    }
}
