use crate::error::{Result, VariationalError};

#[derive(Debug, Clone)]
pub struct Trajectory {
    pub times: Vec<f32>,
    pub states: Vec<Vec<f32>>,
    pub controls: Option<Vec<Vec<f32>>>,
    pub state_dim: usize,
}

impl Trajectory {
    pub fn new(times: Vec<f32>, states: Vec<Vec<f32>>) -> Result<Self> {
        if times.is_empty() || states.is_empty()
        {
            return Err(VariationalError::TrainingFailure {
                details: "empty trajectory".into(),
            });
        }
        if times.len() != states.len()
        {
            return Err(VariationalError::DimensionMismatch {
                expected: times.len(),
                got: states.len(),
                context: "Trajectory::new".into(),
            });
        }
        let state_dim = states[0].len();
        for s in &states
        {
            if s.len() != state_dim
            {
                return Err(VariationalError::DimensionMismatch {
                    expected: state_dim,
                    got: s.len(),
                    context: "Trajectory::new".into(),
                });
            }
        }
        Ok(Self {
            times,
            states,
            controls: None,
            state_dim,
        })
    }

    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    pub fn with_controls(mut self, controls: Vec<Vec<f32>>) -> Result<Self> {
        if controls.len() != self.times.len()
        {
            return Err(VariationalError::DimensionMismatch {
                expected: self.times.len(),
                got: controls.len(),
                context: "Trajectory::with_controls".into(),
            });
        }
        self.controls = Some(controls);
        Ok(self)
    }

    pub fn final_state(&self) -> &[f32] {
        &self.states[self.states.len() - 1]
    }
}
