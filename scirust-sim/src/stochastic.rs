//! Stochastic simulation: geometric Brownian motion and Ornstein–Uhlenbeck
//! paths by their *exact* transition laws (no discretization bias), and an
//! M/M/1 queue by discrete-event simulation, validated against the classic
//! queueing formulas.
//!
//! Every function takes an explicit `seed`; equal seeds give bit-identical
//! paths, so stochastic results are as reproducible as deterministic ones.

use crate::engine::SimError;
use crate::rng::SplitMix64;
use std::collections::VecDeque;

/// Hard cap on path lengths and event counts, so a malformed request cannot
/// allocate or spin without bound.
const MAX_STEPS: usize = 10_000_000;

fn check_finite(name: &str, value: f64) -> Result<(), SimError> {
    if value.is_finite()
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be finite"
        )))
    }
}

fn check_positive(name: &str, value: f64) -> Result<(), SimError> {
    check_finite(name, value)?;
    if value > 0.0
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be positive"
        )))
    }
}

fn check_steps(steps: usize) -> Result<(), SimError> {
    if steps == 0
    {
        return Err(SimError::BadInput(
            "at least one step is required".to_string(),
        ));
    }
    if steps > MAX_STEPS
    {
        return Err(SimError::BadInput(format!(
            "{steps} steps exceed the {MAX_STEPS} budget"
        )));
    }
    Ok(())
}

/// Sample a geometric Brownian motion path `dS = μ·S·dt + σ·S·dW` at times
/// `0, dt, …, steps·dt`, using the exact log-normal transition
/// `S_{k+1} = S_k·exp((μ - σ²/2)·dt + σ·√dt·Z)`.
///
/// Returns `steps + 1` values starting at `s0`. With `σ = 0` the path is the
/// exact exponential `s0·e^{μ·k·dt}`.
pub fn gbm_path(
    s0: f64,
    mu: f64,
    sigma: f64,
    dt: f64,
    steps: usize,
    seed: u64,
) -> Result<Vec<f64>, SimError> {
    check_positive("s0", s0)?;
    check_finite("mu", mu)?;
    check_finite("sigma", sigma)?;
    if sigma < 0.0
    {
        return Err(SimError::BadInput(format!(
            "sigma = {sigma} must be non-negative"
        )));
    }
    check_positive("dt", dt)?;
    check_steps(steps)?;

    let mut rng = SplitMix64::new(seed);
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let diffusion = sigma * dt.sqrt();
    let mut path = Vec::with_capacity(steps + 1);
    path.push(s0);
    let mut s = s0;
    for _ in 0..steps
    {
        s *= (drift + diffusion * rng.next_gaussian()).exp();
        path.push(s);
    }
    Ok(path)
}

/// Sample an Ornstein–Uhlenbeck path `dX = θ·(μ - X)·dt + σ·dW` at times
/// `0, dt, …, steps·dt`, using the exact Gaussian transition
/// `X_{k+1} = μ + (X_k - μ)·e^{-θ·dt} + σ·√((1 - e^{-2θ·dt})/(2θ))·Z`.
///
/// Returns `steps + 1` values starting at `x0`. The stationary distribution
/// is `N(μ, σ²/(2θ))`, the oracle used in the tests.
pub fn ou_path(
    x0: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
    dt: f64,
    steps: usize,
    seed: u64,
) -> Result<Vec<f64>, SimError> {
    check_finite("x0", x0)?;
    check_positive("theta", theta)?;
    check_finite("mu", mu)?;
    check_finite("sigma", sigma)?;
    if sigma < 0.0
    {
        return Err(SimError::BadInput(format!(
            "sigma = {sigma} must be non-negative"
        )));
    }
    check_positive("dt", dt)?;
    check_steps(steps)?;

    let mut rng = SplitMix64::new(seed);
    let decay = (-theta * dt).exp();
    let stddev = sigma * ((1.0 - decay * decay) / (2.0 * theta)).sqrt();
    let mut path = Vec::with_capacity(steps + 1);
    path.push(x0);
    let mut x = x0;
    for _ in 0..steps
    {
        x = mu + (x - mu) * decay + stddev * rng.next_gaussian();
        path.push(x);
    }
    Ok(path)
}

/// An M/M/1 queue: Poisson arrivals at rate `λ`, one server with exponential
/// service times at rate `μ`, infinite waiting room, FIFO discipline.
///
/// For `ρ = λ/μ < 1` the classic steady-state results are
/// `L = ρ/(1-ρ)` customers in the system, mean sojourn `W = 1/(μ-λ)` and
/// server utilization `ρ` — the oracles the discrete-event simulation is
/// tested against.
#[derive(Debug, Clone, PartialEq)]
pub struct MM1Queue {
    arrival_rate: f64,
    service_rate: f64,
}

/// Aggregate statistics from one M/M/1 simulation run.
#[derive(Debug, Clone, PartialEq)]
pub struct QueueStats {
    /// Time-average number of customers in the system (queue + in service).
    pub time_average_in_system: f64,
    /// Fraction of the horizon during which the server was busy.
    pub utilization: f64,
    /// Number of customers that completed service within the horizon.
    pub served: u64,
    /// Mean time in system (wait + service) of the served customers; `0.0`
    /// when no customer completed service.
    pub mean_sojourn: f64,
}

impl MM1Queue {
    /// Create the model; both rates must be finite and positive. `ρ ≥ 1`
    /// (an unstable queue) is allowed — the simulation stays well-defined,
    /// only the steady-state formulas stop applying.
    pub fn new(arrival_rate: f64, service_rate: f64) -> Result<Self, SimError> {
        check_positive("arrival_rate", arrival_rate)?;
        check_positive("service_rate", service_rate)?;
        Ok(MM1Queue {
            arrival_rate,
            service_rate,
        })
    }

    /// Traffic intensity `ρ = λ/μ`.
    pub fn traffic_intensity(&self) -> f64 {
        self.arrival_rate / self.service_rate
    }

    /// Run the queue for `horizon` time units by discrete-event simulation.
    ///
    /// Deterministic for a given `seed`. Returns an error when `horizon` is
    /// not finite and positive or the run would exceed the internal event
    /// budget.
    pub fn simulate(&self, horizon: f64, seed: u64) -> Result<QueueStats, SimError> {
        check_positive("horizon", horizon)?;

        let mut rng = SplitMix64::new(seed);
        // `new` guarantees the rates are valid, so sampling cannot fail.
        let draw = |rate: f64, rng: &mut SplitMix64| {
            rng.next_exponential(rate)
                .expect("rates validated by the constructor")
        };

        let mut now = 0.0;
        let mut in_system: u64 = 0;
        let mut next_arrival = draw(self.arrival_rate, &mut rng);
        let mut next_departure = f64::INFINITY;
        let mut arrivals: VecDeque<f64> = VecDeque::new();

        let mut area = 0.0; // ∫ N(t) dt
        let mut busy = 0.0; // ∫ 1{N(t) > 0} dt
        let mut served: u64 = 0;
        let mut sojourn_sum = 0.0;

        for _ in 0..MAX_STEPS
        {
            let next_event = next_arrival.min(next_departure);
            if next_event > horizon
            {
                let dt = horizon - now;
                area += in_system as f64 * dt;
                if in_system > 0
                {
                    busy += dt;
                }
                let mean_sojourn = if served > 0
                {
                    sojourn_sum / served as f64
                }
                else
                {
                    0.0
                };
                return Ok(QueueStats {
                    time_average_in_system: area / horizon,
                    utilization: busy / horizon,
                    served,
                    mean_sojourn,
                });
            }

            let dt = next_event - now;
            area += in_system as f64 * dt;
            if in_system > 0
            {
                busy += dt;
            }
            now = next_event;

            if next_arrival <= next_departure
            {
                in_system += 1;
                arrivals.push_back(now);
                next_arrival = now + draw(self.arrival_rate, &mut rng);
                if in_system == 1
                {
                    next_departure = now + draw(self.service_rate, &mut rng);
                }
            }
            else
            {
                in_system -= 1;
                // A departure implies a matching earlier arrival.
                if let Some(arrived) = arrivals.pop_front()
                {
                    sojourn_sum += now - arrived;
                    served += 1;
                }
                next_departure = if in_system > 0
                {
                    now + draw(self.service_rate, &mut rng)
                }
                else
                {
                    f64::INFINITY
                };
            }
        }
        Err(SimError::BadInput(format!(
            "simulation exceeded the {MAX_STEPS} event budget; shorten the horizon"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gbm_with_zero_volatility_is_the_exact_exponential() {
        let path = gbm_path(2.0, 0.5, 0.0, 0.1, 50, 1).unwrap();
        for (k, s) in path.iter().enumerate()
        {
            let exact = 2.0 * (0.5 * 0.1 * k as f64).exp();
            assert!((s - exact).abs() < 1e-12 * exact, "k = {k}");
        }
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn gbm_log_return_statistics_match_the_transition_law() {
        // Over M paths, log(S_T/S_0) is N((μ - σ²/2)·T, σ²·T).
        let (mu, sigma, dt, steps) = (0.08, 0.2, 0.02, 50);
        let t_end = dt * steps as f64;
        let m = 2_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for path_index in 0..m
        {
            let path = gbm_path(1.0, mu, sigma, dt, steps, 1_000 + path_index).unwrap();
            let log_return = path.last().unwrap().ln();
            sum += log_return;
            sum_sq += log_return * log_return;
        }
        let mean = sum / m as f64;
        let var = sum_sq / m as f64 - mean * mean;
        let exact_mean = (mu - 0.5 * sigma * sigma) * t_end;
        let exact_var = sigma * sigma * t_end;
        assert!(
            (mean - exact_mean).abs() < 0.015,
            "mean {mean} vs {exact_mean}"
        );
        assert!(
            (var - exact_var).abs() < 0.2 * exact_var,
            "var {var} vs {exact_var}"
        );
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn ou_relaxes_to_the_mean_and_reaches_the_stationary_variance() {
        // σ = 0: exact deterministic relaxation toward μ.
        let path = ou_path(5.0, 0.8, 1.0, 0.0, 0.1, 100, 7).unwrap();
        for (k, x) in path.iter().enumerate()
        {
            let exact = 1.0 + 4.0 * (-0.8 * 0.1 * k as f64).exp();
            assert!((x - exact).abs() < 1e-12, "k = {k}");
        }
        // Long stochastic path: sample variance ≈ σ²/(2θ).
        let (theta, sigma) = (0.5, 0.3);
        let path = ou_path(1.0, theta, 1.0, sigma, 0.05, 200_000, 42).unwrap();
        let n = path.len() as f64;
        let mean = path.iter().sum::<f64>() / n;
        let var = path.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        let exact_var = sigma * sigma / (2.0 * theta);
        assert!((mean - 1.0).abs() < 0.02, "mean {mean}");
        assert!(
            (var - exact_var).abs() < 0.1 * exact_var,
            "var {var} vs {exact_var}"
        );
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn mm1_statistics_match_the_classic_queueing_formulas() {
        // λ = 1, μ = 2: ρ = 0.5, L = ρ/(1-ρ) = 1, W = 1/(μ-λ) = 1.
        let queue = MM1Queue::new(1.0, 2.0).unwrap();
        assert!((queue.traffic_intensity() - 0.5).abs() < 1e-15);
        let stats = queue.simulate(200_000.0, 2024).unwrap();
        assert!(
            (stats.time_average_in_system - 1.0).abs() < 0.05,
            "L = {}",
            stats.time_average_in_system
        );
        assert!(
            (stats.utilization - 0.5).abs() < 0.01,
            "ρ = {}",
            stats.utilization
        );
        assert!(
            (stats.mean_sojourn - 1.0).abs() < 0.05,
            "W = {}",
            stats.mean_sojourn
        );
        // Little's law ties the three measurements together: L = λ_eff · W.
        let lambda_eff = stats.served as f64 / 200_000.0;
        assert!(
            (stats.time_average_in_system - lambda_eff * stats.mean_sojourn).abs() < 0.02,
            "Little's law violated"
        );
    }

    #[test]
    // Ignored under Miri: Miri deliberately perturbs the last bits of the
    // transcendental float intrinsics (exp/ln/sin/cos) to model their
    // platform freedom, so bit-identity across runs is not expected under
    // the interpreter. Native Build & Test jobs enforce it.
    #[cfg_attr(miri, ignore)]
    fn stochastic_runs_are_reproducible_and_seed_sensitive() {
        let a = gbm_path(1.0, 0.1, 0.3, 0.01, 100, 5).unwrap();
        let b = gbm_path(1.0, 0.1, 0.3, 0.01, 100, 5).unwrap();
        let c = gbm_path(1.0, 0.1, 0.3, 0.01, 100, 6).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        let queue = MM1Queue::new(1.0, 2.0).unwrap();
        assert_eq!(
            queue.simulate(1_000.0, 9).unwrap(),
            queue.simulate(1_000.0, 9).unwrap()
        );
    }

    #[test]
    fn malformed_requests_are_rejected() {
        assert!(gbm_path(0.0, 0.1, 0.2, 0.01, 10, 1).is_err());
        assert!(gbm_path(1.0, 0.1, -0.2, 0.01, 10, 1).is_err());
        assert!(gbm_path(1.0, 0.1, 0.2, 0.0, 10, 1).is_err());
        assert!(gbm_path(1.0, 0.1, 0.2, 0.01, 0, 1).is_err());
        assert!(gbm_path(1.0, f64::NAN, 0.2, 0.01, 10, 1).is_err());
        assert!(ou_path(1.0, 0.0, 1.0, 0.3, 0.05, 10, 1).is_err());
        assert!(ou_path(f64::INFINITY, 0.5, 1.0, 0.3, 0.05, 10, 1).is_err());
        assert!(MM1Queue::new(0.0, 2.0).is_err());
        assert!(MM1Queue::new(1.0, f64::NAN).is_err());
        let queue = MM1Queue::new(1.0, 2.0).unwrap();
        assert!(queue.simulate(0.0, 1).is_err());
        assert!(queue.simulate(f64::INFINITY, 1).is_err());
    }
}
