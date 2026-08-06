use crate::mechanics::invariants::ConservationReport;

pub fn compare_conservation(trajectory: &[(f32, Vec<f32>)], ndim: usize) -> ConservationReport {
    let n_states = trajectory.len();
    let _phase_dim = if n_states > 0
    {
        trajectory[0].1.len()
    }
    else
    {
        0
    };

    let mut jacobi_integrals = Vec::with_capacity(n_states);

    for (_, state) in trajectory
    {
        let mut sum = 0.0;
        for i in 0..ndim.min(state.len())
        {
            let qi = state[i];
            let pi = if i + ndim < state.len()
            {
                state[i + ndim]
            }
            else
            {
                0.0
            };
            sum += pi * qi;
        }
        jacobi_integrals.push(sum);
    }

    if jacobi_integrals.is_empty()
    {
        return ConservationReport::new(&[0.0]);
    }

    ConservationReport::new(&jacobi_integrals)
}

pub fn detect_drift(values: &[f32], drift_threshold: f32) -> (bool, f32) {
    if values.len() < 2
    {
        return (false, 0.0);
    }
    let initial = values[0];
    let final_val = values[values.len() - 1];
    let drift = (final_val - initial).abs();
    (drift > drift_threshold, drift)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_detection() {
        let values = vec![1.0, 1.01, 1.02, 1.03];
        let (has_drift, drift) = detect_drift(&values, 0.01);
        assert!(has_drift);
        assert!((drift - 0.03).abs() < 1e-6);
    }

    #[test]
    fn test_no_drift() {
        let values = vec![1.0, 1.0, 1.0];
        let (has_drift, _) = detect_drift(&values, 0.01);
        assert!(!has_drift);
    }
}
