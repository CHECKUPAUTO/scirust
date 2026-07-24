use crate::error::Result;
use crate::learning::dataset::TrajectoryDataset;

#[derive(Debug, Clone)]
pub struct ModelRolloutReport {
    pub rollout_steps: usize,
    pub max_position_error: f32,
    pub rms_position_error: f32,
    pub max_velocity_error: f32,
    pub rms_velocity_error: f32,
    pub energy_drift: f32,
}

pub fn compute_rollout_errors(
    predicted_trajectory: &[(f32, Vec<f32>)],
    ground_truth: &TrajectoryDataset,
    ndim: usize,
) -> Result<ModelRolloutReport> {
    let n = predicted_trajectory.len().min(ground_truth.len());
    if n == 0
    {
        return Err(crate::error::VariationalError::TrainingFailure {
            details: "empty trajectory for rollout comparison".into(),
        });
    }

    let mut pos_errors = Vec::new();
    let mut vel_errors = Vec::new();

    for i in 0..n
    {
        let pred = &predicted_trajectory[i].1;
        let truth = ground_truth.get(i);
        for j in 0..ndim
        {
            pos_errors.push((pred[j] - truth.q[j]).abs());
            vel_errors.push((pred[ndim + j] - truth.dq[j]).abs());
        }
    }

    let max_pos = pos_errors.iter().copied().fold(0.0, f32::max);
    let max_vel = vel_errors.iter().copied().fold(0.0, f32::max);
    let rms_pos = (pos_errors.iter().map(|e| e * e).sum::<f32>() / pos_errors.len() as f32).sqrt();
    let rms_vel = (vel_errors.iter().map(|e| e * e).sum::<f32>() / vel_errors.len() as f32).sqrt();

    Ok(ModelRolloutReport {
        rollout_steps: n,
        max_position_error: max_pos,
        rms_position_error: rms_pos,
        max_velocity_error: max_vel,
        rms_velocity_error: rms_vel,
        energy_drift: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::dataset::trajectory_from_ode;

    fn dummy_dynamics(_t: f32, state: &[f32], deriv: &mut [f32]) {
        deriv[0] = state[1];
        deriv[1] = -state[0];
    }

    #[test]
    fn test_rollout_errors() {
        let t_span: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let dataset = trajectory_from_ode(dummy_dynamics, &[1.0], &[0.0], &t_span, 1).unwrap();
        let predicted: Vec<(f32, Vec<f32>)> = t_span
            .iter()
            .map(|&t| (t, vec![t.cos(), -t.sin()]))
            .collect();
        let report = compute_rollout_errors(&predicted, &dataset, 1).unwrap();
        assert_eq!(report.rollout_steps, 10);
    }
}
