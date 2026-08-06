use core::fmt;

use crate::NodeId;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphError {
    TooManyNodes,
    ArityMismatch { expected: usize, actual: usize },
    InvalidInput { node_index: usize, input: NodeId },
    InvalidOutput { output: NodeId },
}

impl fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::TooManyNodes => formatter.write_str("tensor graph exceeds u32 node capacity"),
            Self::ArityMismatch { expected, actual } =>
            {
                write!(
                    formatter,
                    "operation expects {expected} inputs but received {actual}"
                )
            },
            Self::InvalidInput { node_index, input } =>
            {
                write!(
                    formatter,
                    "node {node_index} references unavailable input {}",
                    input.get()
                )
            },
            Self::InvalidOutput { output } =>
            {
                write!(
                    formatter,
                    "graph references unavailable output {}",
                    output.get()
                )
            },
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GraphError {}
