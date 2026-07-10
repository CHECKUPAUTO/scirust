//! Compartmental epidemic models: SIR and SEIR over population *fractions*,
//! validated against the exact final-size relation and the conservation of
//! the total population.

use crate::engine::{SimError, System};

fn check_rate(name: &str, value: f64) -> Result<(), SimError> {
    if value.is_finite() && value > 0.0
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be finite and positive"
        )))
    }
}

/// The Kermack–McKendrick SIR model over population fractions,
/// state `y = [s, i, r]`:
///
/// `s' = -β·s·i`, `i' = β·s·i - γ·i`, `r' = γ·i`.
///
/// `β` is the transmission rate and `γ` the recovery rate; the basic
/// reproduction number is `R₀ = β/γ`. The total `s + i + r` is a linear
/// invariant, which RK4 preserves to round-off.
#[derive(Debug, Clone, PartialEq)]
pub struct Sir {
    beta: f64,
    gamma: f64,
}

impl Sir {
    /// Create the model; both rates must be finite and positive.
    pub fn new(beta: f64, gamma: f64) -> Result<Self, SimError> {
        check_rate("beta", beta)?;
        check_rate("gamma", gamma)?;
        Ok(Sir { beta, gamma })
    }

    /// Basic reproduction number `R₀ = β/γ`.
    pub fn r0(&self) -> f64 {
        self.beta / self.gamma
    }
}

impl System for Sir {
    fn dim(&self) -> usize {
        3
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        let infection = self.beta * y[0] * y[1];
        let recovery = self.gamma * y[1];
        dydt[0] = -infection;
        dydt[1] = infection - recovery;
        dydt[2] = recovery;
    }
}

/// The SEIR model with a latent (exposed) compartment,
/// state `y = [s, e, i, r]`:
///
/// `s' = -β·s·i`, `e' = β·s·i - σ·e`, `i' = σ·e - γ·i`, `r' = γ·i`,
///
/// where `1/σ` is the mean latent period. `R₀ = β/γ` as for SIR.
#[derive(Debug, Clone, PartialEq)]
pub struct Seir {
    beta: f64,
    sigma: f64,
    gamma: f64,
}

impl Seir {
    /// Create the model; all three rates must be finite and positive.
    pub fn new(beta: f64, sigma: f64, gamma: f64) -> Result<Self, SimError> {
        check_rate("beta", beta)?;
        check_rate("sigma", sigma)?;
        check_rate("gamma", gamma)?;
        Ok(Seir { beta, sigma, gamma })
    }

    /// Basic reproduction number `R₀ = β/γ`.
    pub fn r0(&self) -> f64 {
        self.beta / self.gamma
    }
}

impl System for Seir {
    fn dim(&self) -> usize {
        4
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        let infection = self.beta * y[0] * y[2];
        let incubation = self.sigma * y[1];
        let recovery = self.gamma * y[2];
        dydt[0] = -infection;
        dydt[1] = infection - incubation;
        dydt[2] = incubation - recovery;
        dydt[3] = recovery;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::simulate;

    #[test]
    fn sir_population_is_conserved_to_round_off() {
        let sir = Sir::new(0.6, 0.2).unwrap();
        let traj = simulate(&sir, &[0.99, 0.01, 0.0], 0.0, 100.0, 0.05).unwrap();
        for row in &traj.y
        {
            assert!((row[0] + row[1] + row[2] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn epidemic_grows_above_threshold_and_dies_below() {
        // R0 = 3 > 1: the infected fraction must rise well above i0.
        let growing = Sir::new(0.6, 0.2).unwrap();
        assert!((growing.r0() - 3.0).abs() < 1e-12);
        let traj = simulate(&growing, &[0.999, 0.001, 0.0], 0.0, 60.0, 0.05).unwrap();
        let peak = traj.column(1).unwrap().iter().cloned().fold(0.0, f64::max);
        assert!(peak > 0.25, "peak {peak}");

        // R0 = 0.5 < 1: the infected fraction decays monotonically.
        let dying = Sir::new(0.1, 0.2).unwrap();
        let traj = simulate(&dying, &[0.999, 0.001, 0.0], 0.0, 60.0, 0.05).unwrap();
        let infected = traj.column(1).unwrap();
        assert!(infected.windows(2).all(|w| w[1] <= w[0] + 1e-15));
        assert!(infected.last().unwrap() < &1e-5);
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn sir_final_size_satisfies_the_exact_transcendental_relation() {
        // For i0 → 0, s∞ solves ln(s∞/s0) = -R0·(1 - s∞). Integrating far
        // past the epidemic must land on that curve.
        let r0 = 2.0;
        let sir = Sir::new(2.0 * 0.25, 0.25).unwrap();
        let traj = simulate(&sir, &[1.0 - 1e-6, 1e-6, 0.0], 0.0, 400.0, 0.05).unwrap();
        let s_inf = traj.last_state().unwrap()[0];
        let residual = (s_inf / (1.0 - 1e-6)).ln() + r0 * (1.0 - s_inf);
        assert!(
            residual.abs() < 1e-3,
            "final size s∞ = {s_inf}, residual {residual}"
        );
        // Sanity: the infection is over.
        assert!(traj.last_state().unwrap()[1] < 1e-8);
    }

    #[test]
    fn seir_conserves_population_and_shows_an_epidemic_when_r0_exceeds_one() {
        let seir = Seir::new(0.6, 0.3, 0.2).unwrap();
        assert!((seir.r0() - 3.0).abs() < 1e-12);
        let traj = simulate(&seir, &[0.999, 0.0, 0.001, 0.0], 0.0, 120.0, 0.05).unwrap();
        for row in &traj.y
        {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        let peak = traj.column(2).unwrap().iter().cloned().fold(0.0, f64::max);
        assert!(peak > 0.1, "peak {peak}");
        // The latent stage delays the SEIR peak relative to SIR with the
        // same rates.
        let sir = Sir::new(0.6, 0.2).unwrap();
        let sir_traj = simulate(&sir, &[0.999, 0.001, 0.0], 0.0, 120.0, 0.05).unwrap();
        let argmax = |xs: &[f64]| {
            xs.iter()
                .enumerate()
                .fold((0, 0.0), |m, (i, &v)| if v > m.1 { (i, v) } else { m })
                .0
        };
        let sir_peak_at = argmax(&sir_traj.column(1).unwrap());
        let seir_peak_at = argmax(&traj.column(2).unwrap());
        assert!(seir_peak_at > sir_peak_at);
    }

    #[test]
    fn constructors_reject_bad_rates() {
        assert!(Sir::new(0.0, 0.2).is_err());
        assert!(Sir::new(0.5, -0.2).is_err());
        assert!(Sir::new(f64::NAN, 0.2).is_err());
        assert!(Seir::new(0.5, 0.0, 0.2).is_err());
        assert!(Seir::new(0.5, f64::INFINITY, 0.2).is_err());
    }
}
