use scirust_core::autodiff::reverse::Tensor;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ReasoningError {
    #[error("Logic error: {0}")]
    Logic(String),
    #[error("Solver error: {0}")]
    Solver(String),
    #[error("Constraint error: {0}")]
    Constraint(String),
    #[error("Symbolic error: {0}")]
    Symbolic(String),
    #[error("Neural error: {0}")]
    Neural(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, ReasoningError>;

/// Base trait for all reasoning engines.
pub trait Reasoner {
    fn name(&self) -> &str;
}

/// Interface for differentiable reasoning components.
pub trait DifferentiableReasoner: Reasoner {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor>;
}

/// Interface for symbolic knowledge bases.
pub trait KnowledgeBase: Reasoner {
    fn add_fact(&mut self, fact: &str) -> Result<()>;
    fn query(&self, query: &str) -> Result<bool>;
}
