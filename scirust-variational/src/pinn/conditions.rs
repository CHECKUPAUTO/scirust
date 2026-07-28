use crate::error::{Result, VariationalError};

#[derive(Debug, Clone)]
pub enum ConditionKind {
    Initial,
    Dirichlet,
    Neumann,
    Robin,
    Periodic,
}

impl std::fmt::Display for ConditionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::Initial => write!(f, "initial"),
            Self::Dirichlet => write!(f, "Dirichlet"),
            Self::Neumann => write!(f, "Neumann"),
            Self::Robin => write!(f, "Robin"),
            Self::Periodic => write!(f, "periodic"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Condition {
    pub kind: ConditionKind,
    pub points: Vec<Vec<f32>>,
    pub target_fn: fn(&[f32]) -> f32,
    pub weight: f32,
    pub name: String,
}

impl Condition {
    pub fn new(
        kind: ConditionKind,
        points: Vec<Vec<f32>>,
        target_fn: fn(&[f32]) -> f32,
        weight: f32,
        name: &str,
    ) -> Result<Self> {
        if weight < 0.0
        {
            return Err(VariationalError::InvalidBoundaryCondition {
                details: format!("negative weight {weight}"),
            });
        }
        if points.is_empty()
        {
            return Err(VariationalError::InvalidBoundaryCondition {
                details: "no points provided".into(),
            });
        }
        Ok(Self {
            kind,
            points,
            target_fn,
            weight,
            name: name.to_string(),
        })
    }

    pub fn evaluate_targets(&self) -> Vec<f32> {
        self.points.iter().map(|p| (self.target_fn)(p)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct ConditionConfig {
    pub conditions: Vec<Condition>,
}

impl ConditionConfig {
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }

    pub fn with_condition(mut self, condition: Condition) -> Self {
        self.conditions.push(condition);
        self
    }

    pub fn total_points(&self) -> usize {
        self.conditions.iter().map(|c| c.points.len()).sum()
    }
}

impl Default for ConditionConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condition_creation() {
        let pts = vec![vec![0.0], vec![1.0]];
        let c = Condition::new(
            ConditionKind::Dirichlet,
            pts.clone(),
            |x| x[0].sin(),
            1.0,
            "test",
        )
        .unwrap();
        assert_eq!(c.points.len(), 2);
        let targets = c.evaluate_targets();
        assert!((targets[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_negative_weight_error() {
        let result = Condition::new(
            ConditionKind::Dirichlet,
            vec![vec![0.0]],
            |_| 0.0,
            -1.0,
            "bad",
        );
        assert!(result.is_err());
    }
}
