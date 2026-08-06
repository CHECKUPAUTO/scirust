use crate::error::{Result, VariationalError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeInterval {
    pub start: f32,
    pub end: f32,
}

impl TimeInterval {
    pub fn new(start: f32, end: f32) -> Result<Self> {
        if start >= end
        {
            return Err(VariationalError::InvalidInterval { start, end });
        }
        Ok(Self { start, end })
    }

    pub fn duration(&self) -> f32 {
        self.end - self.start
    }

    pub fn contains(&self, t: f32) -> bool {
        t >= self.start && t <= self.end
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryCondition {
    Fixed(f32),
    Free,
    Natural,
}

impl BoundaryCondition {
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }

    pub fn fixed_value(&self) -> Option<f32> {
        match self
        {
            Self::Fixed(v) => Some(*v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeneralizedCoordinate {
    pub name: String,
    pub index: usize,
}

impl GeneralizedCoordinate {
    pub fn new(name: &str, index: usize) -> Self {
        Self {
            name: name.to_string(),
            index,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeneralizedVelocity {
    pub name: String,
    pub coord_index: usize,
}

impl GeneralizedVelocity {
    pub fn new(coordinate: &GeneralizedCoordinate) -> Self {
        Self {
            name: format!("d{}", coordinate.name),
            coord_index: coordinate.index,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActionProblem {
    pub num_coordinates: usize,
    pub interval: TimeInterval,
    pub initial_conditions: Vec<BoundaryCondition>,
    pub terminal_conditions: Vec<BoundaryCondition>,
    pub has_explicit_time: bool,
}

impl ActionProblem {
    pub fn new(num_coordinates: usize, interval: TimeInterval) -> Result<Self> {
        if num_coordinates == 0
        {
            return Err(VariationalError::DimensionMismatch {
                expected: 1,
                got: 0,
                context: "ActionProblem::new".into(),
            });
        }
        Ok(Self {
            num_coordinates,
            interval,
            initial_conditions: vec![BoundaryCondition::Free; num_coordinates],
            terminal_conditions: vec![BoundaryCondition::Free; num_coordinates],
            has_explicit_time: false,
        })
    }

    pub fn with_initial_condition(
        mut self,
        index: usize,
        condition: BoundaryCondition,
    ) -> Result<Self> {
        if index >= self.num_coordinates
        {
            return Err(VariationalError::DimensionMismatch {
                expected: self.num_coordinates,
                got: index + 1,
                context: "ActionProblem::with_initial_condition".into(),
            });
        }
        self.initial_conditions[index] = condition;
        Ok(self)
    }

    pub fn with_terminal_condition(
        mut self,
        index: usize,
        condition: BoundaryCondition,
    ) -> Result<Self> {
        if index >= self.num_coordinates
        {
            return Err(VariationalError::DimensionMismatch {
                expected: self.num_coordinates,
                got: index + 1,
                context: "ActionProblem::with_terminal_condition".into(),
            });
        }
        self.terminal_conditions[index] = condition;
        Ok(self)
    }

    pub fn with_explicit_time(mut self) -> Self {
        self.has_explicit_time = true;
        self
    }
}
