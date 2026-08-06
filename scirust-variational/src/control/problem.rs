use crate::error::{Result, VariationalError};

#[derive(Debug, Clone)]
pub struct ControlBounds {
    pub lower: Vec<f32>,
    pub upper: Vec<f32>,
}

impl ControlBounds {
    pub fn new(lower: Vec<f32>, upper: Vec<f32>) -> Result<Self> {
        if lower.len() != upper.len()
        {
            return Err(VariationalError::DimensionMismatch {
                expected: lower.len(),
                got: upper.len(),
                context: "ControlBounds::new".into(),
            });
        }
        for i in 0..lower.len()
        {
            if lower[i] > upper[i]
            {
                return Err(VariationalError::InfeasibleControlProblem {
                    details: format!("control bound {i}: lower {} > upper {}", lower[i], upper[i]),
                });
            }
        }
        Ok(Self { lower, upper })
    }
}

#[derive(Debug, Clone)]
pub struct OptimalControlProblem<D, RC, TC> {
    pub state_dim: usize,
    pub control_dim: usize,
    pub dynamics: D,
    pub running_cost: RC,
    pub terminal_cost: TC,
    pub initial_state: Vec<f32>,
    pub horizon: (f32, f32),
    pub control_bounds: Option<ControlBounds>,
    pub state_bounds: Option<ControlBounds>,
    pub num_time_steps: usize,
}

impl<D, RC, TC> OptimalControlProblem<D, RC, TC>
where
    D: Fn(f32, &[f32], &[f32], &mut [f32]),
    RC: Fn(f32, &[f32], &[f32]) -> f32,
    TC: Fn(&[f32], f32) -> f32,
{
    pub fn new(
        state_dim: usize,
        control_dim: usize,
        dynamics: D,
        running_cost: RC,
        terminal_cost: TC,
        initial_state: Vec<f32>,
        horizon: (f32, f32),
    ) -> Result<Self> {
        if state_dim == 0
        {
            return Err(VariationalError::DimensionMismatch {
                expected: 1,
                got: 0,
                context: "OptimalControlProblem::new".into(),
            });
        }
        if initial_state.len() != state_dim
        {
            return Err(VariationalError::DimensionMismatch {
                expected: state_dim,
                got: initial_state.len(),
                context: "OptimalControlProblem::new".into(),
            });
        }
        if horizon.0 >= horizon.1
        {
            return Err(VariationalError::InvalidInterval {
                start: horizon.0,
                end: horizon.1,
            });
        }
        Ok(Self {
            state_dim,
            control_dim,
            dynamics,
            running_cost,
            terminal_cost,
            initial_state,
            horizon,
            control_bounds: None,
            state_bounds: None,
            num_time_steps: 100,
        })
    }

    pub fn with_control_bounds(mut self, bounds: ControlBounds) -> Self {
        self.control_bounds = Some(bounds);
        self
    }

    pub fn with_state_bounds(mut self, bounds: ControlBounds) -> Self {
        self.state_bounds = Some(bounds);
        self
    }

    pub fn with_time_steps(mut self, n: usize) -> Self {
        self.num_time_steps = n;
        self
    }
}

#[derive(Debug, Clone)]
pub struct ControlSolution {
    pub times: Vec<f32>,
    pub states: Vec<Vec<f32>>,
    pub controls: Vec<Vec<f32>>,
    pub objective: f32,
    pub feasibility_residual: f32,
    pub converged: bool,
    pub iterations: usize,
}

impl ControlSolution {
    pub fn new(times: Vec<f32>, states: Vec<Vec<f32>>, controls: Vec<Vec<f32>>) -> Self {
        Self {
            times,
            states,
            controls,
            objective: f32::NAN,
            feasibility_residual: f32::NAN,
            converged: false,
            iterations: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_bounds_validation() {
        assert!(ControlBounds::new(vec![0.0], vec![1.0]).is_ok());
        assert!(ControlBounds::new(vec![1.0], vec![0.0]).is_err());
    }
}
