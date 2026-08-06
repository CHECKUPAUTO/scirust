extern crate alloc;

use alloc::string::String;
use core::fmt;

use crate::DeviceKind;

pub type ComputeResult<T> = Result<T, ComputeError>;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ComputeError {
    InvalidArgument(&'static str),
    ShapeOverflow,
    ByteSizeOverflow,
    BackendUnavailable(DeviceKind),
    Unsupported(&'static str),
    Allocation(String),
    Transfer(String),
    Compilation(String),
    Launch(String),
    Synchronization(String),
}

impl fmt::Display for ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidArgument(message) =>
            {
                write!(f, "invalid argument: {message}")
            },
            Self::ShapeOverflow =>
            {
                write!(f, "tensor shape overflows usize")
            },
            Self::ByteSizeOverflow =>
            {
                write!(f, "tensor byte size overflows usize")
            },
            Self::BackendUnavailable(kind) =>
            {
                write!(f, "compute backend {kind:?} is unavailable")
            },
            Self::Unsupported(message) =>
            {
                write!(f, "unsupported operation: {message}")
            },
            Self::Allocation(message) =>
            {
                write!(f, "allocation failed: {message}")
            },
            Self::Transfer(message) =>
            {
                write!(f, "transfer failed: {message}")
            },
            Self::Compilation(message) =>
            {
                write!(f, "kernel compilation failed: {message}")
            },
            Self::Launch(message) =>
            {
                write!(f, "kernel launch failed: {message}")
            },
            Self::Synchronization(message) =>
            {
                write!(f, "synchronization failed: {message}")
            },
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ComputeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_have_stable_messages() {
        assert_eq!(
            ComputeError::ShapeOverflow.to_string(),
            "tensor shape overflows usize"
        );

        assert_eq!(
            ComputeError::BackendUnavailable(DeviceKind::Cuda).to_string(),
            "compute backend Cuda is unavailable"
        );

        assert_eq!(
            ComputeError::Compilation("invalid PTX".into()).to_string(),
            "kernel compilation failed: invalid PTX"
        );
    }
}
