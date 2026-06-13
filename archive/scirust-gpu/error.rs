// scirust-gpu/src/error.rs
//
// Erreurs locales au crate scirust-gpu (quantization, etc.).

use std::fmt;

#[derive(Debug, Clone)]
pub enum QuantError {
    Io(String),
    InvalidFormat(String),
    ShapeMismatch { expected: usize, got: usize },
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            QuantError::Io(msg) => write!(f, "IO error: {}", msg),
            QuantError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            QuantError::ShapeMismatch { expected, got } =>
            {
                write!(f, "Shape mismatch: expected {}, got {}", expected, got)
            },
        }
    }
}

impl std::error::Error for QuantError {}

pub type Result<T> = std::result::Result<T, QuantError>;
