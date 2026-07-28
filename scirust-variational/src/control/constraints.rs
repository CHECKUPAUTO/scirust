#[derive(Debug, Clone)]
pub enum ConstraintType {
    Equality,
    Inequality,
    BoxBound,
}

#[derive(Debug, Clone)]
pub struct PathConstraint {
    pub name: String,
    pub constraint_type: ConstraintType,
    pub function: fn(&[f32], &[f32], f32) -> f32,
    pub tolerance: f32,
}

impl PathConstraint {
    pub fn new(
        name: &str,
        constraint_type: ConstraintType,
        function: fn(&[f32], &[f32], f32) -> f32,
        tolerance: f32,
    ) -> Self {
        Self {
            name: name.to_string(),
            constraint_type,
            function,
            tolerance,
        }
    }

    pub fn violation(&self, state: &[f32], control: &[f32], time: f32) -> f32 {
        let val = (self.function)(state, control, time);
        match self.constraint_type
        {
            ConstraintType::Equality => val.abs(),
            ConstraintType::Inequality => (-val).max(0.0),
            ConstraintType::BoxBound => 0.0,
        }
    }
}

pub fn project_to_box_bounds(x: &mut [f32], lower: &[f32], upper: &[f32]) {
    for i in 0..x.len()
    {
        x[i] = x[i].clamp(lower[i], upper[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_to_box_bounds() {
        let mut x = vec![-0.5, 1.5, 0.5];
        project_to_box_bounds(&mut x, &[0.0; 3], &[1.0; 3]);
        assert!((x[0] - 0.0).abs() < 1e-6);
        assert!((x[1] - 1.0).abs() < 1e-6);
        assert!((x[2] - 0.5).abs() < 1e-6);
    }
}
