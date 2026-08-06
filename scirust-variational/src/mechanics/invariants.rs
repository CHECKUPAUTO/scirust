#[derive(Debug, Clone)]
pub struct ConservationReport {
    pub initial: f32,
    pub final_value: f32,
    pub max_abs_drift: f32,
    pub relative_drift: f32,
    pub rms_drift: f32,
}

impl ConservationReport {
    pub fn new(values: &[f32]) -> Self {
        let initial = values[0];
        let final_value = values[values.len() - 1];
        let max_abs_drift = values
            .iter()
            .map(|&v| (v - initial).abs())
            .fold(0.0, f32::max);
        let relative_drift = if initial.abs() > 1e-10
        {
            max_abs_drift / initial.abs()
        }
        else
        {
            max_abs_drift
        };
        let rms_drift = (values.iter().map(|&v| (v - initial).powi(2)).sum::<f32>()
            / values.len() as f32)
            .sqrt();

        Self {
            initial,
            final_value,
            max_abs_drift,
            relative_drift,
            rms_drift,
        }
    }

    pub fn is_conserved_within(&self, tolerance: f32) -> bool {
        self.max_abs_drift < tolerance
    }
}

pub fn compute_energy<F>(lagrangian: &F, q: &[f32], dq: &[f32], t: f32) -> f32
where
    F: Fn(&[f32], &[f32], f32) -> f32,
{
    let eps = 1e-4;
    let n = q.len();
    let L = lagrangian(q, dq, t);

    let mut energy = 0.0;
    for i in 0..n
    {
        let mut dq_plus = dq.to_vec();
        dq_plus[i] += eps;
        let L_plus = lagrangian(q, &dq_plus, t);

        let mut dq_minus = dq.to_vec();
        dq_minus[i] -= eps;
        let L_minus = lagrangian(q, &dq_minus, t);

        let dL_ddq_i = (L_plus - L_minus) / (2.0 * eps);
        energy += dL_ddq_i * dq[i];
    }
    energy - L
}

pub fn compute_energy_from_trajectory<F>(
    lagrangian: &F,
    trajectory: &[(f32, Vec<f32>)],
    ndim: usize,
) -> Vec<f32>
where
    F: Fn(&[f32], &[f32], f32) -> f32,
{
    trajectory
        .iter()
        .map(|&(t, ref state)| {
            let q = &state[..ndim];
            let dq = &state[ndim..];
            compute_energy(lagrangian, q, dq, t)
        })
        .collect()
}

pub fn compute_invariant_diagnostics<F>(
    lagrangian: &F,
    trajectory: &[(f32, Vec<f32>)],
    ndim: usize,
    label: &str,
) -> ConservationReport
where
    F: Fn(&[f32], &[f32], f32) -> f32,
{
    let values = compute_energy_from_trajectory(lagrangian, trajectory, ndim);
    let report = ConservationReport::new(&values);
    println!(
        "Conservation diagnostics for {label}: \
         initial={:.6}, final={:.6}, max_drift={:.2e}, rel_drift={:.2e}, rms_drift={:.2e}",
        report.initial,
        report.final_value,
        report.max_abs_drift,
        report.relative_drift,
        report.rms_drift
    );
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harmonic_oscillator_energy(q: &[f32], dq: &[f32], _t: f32) -> f32 {
        0.5 * dq[0] * dq[0] + 0.5 * q[0] * q[0]
    }

    #[test]
    fn test_energy_conservation_harmonic() {
        let traj = vec![
            (0.0, vec![1.0, 0.0]),
            (0.5, vec![0.87758, -0.47943]),
            (1.0, vec![0.54030, -0.84147]),
            (1.5, vec![0.07074, -0.99749]),
            (2.0, vec![-0.41615, -0.90930]),
        ];
        let report = compute_invariant_diagnostics(
            &harmonic_oscillator_energy,
            &traj,
            1,
            "harmonic_oscillator",
        );
        assert!(
            report.max_abs_drift < 1.0,
            "energy drift too large: {}",
            report.max_abs_drift
        );
    }

    #[test]
    fn test_conservation_report_all_constant() {
        let values = vec![10.0, 10.0, 10.0];
        let report = ConservationReport::new(&values);
        assert!(report.max_abs_drift < 1e-6);
        assert!(report.relative_drift < 1e-6);
        assert!(report.rms_drift < 1e-6);
    }

    #[test]
    fn test_conservation_report_with_drift() {
        let values = vec![10.0, 10.1, 10.2, 10.3];
        let report = ConservationReport::new(&values);
        assert!((report.max_abs_drift - 0.3).abs() < 1e-6);
    }
}
