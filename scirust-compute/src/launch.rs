use crate::{ComputeError, ComputeResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_memory_bytes: u32,
}

impl LaunchConfig {
    pub fn new(grid: [u32; 3], block: [u32; 3], shared_memory_bytes: u32) -> ComputeResult<Self> {
        if grid.contains(&0)
        {
            return Err(ComputeError::InvalidArgument(
                "grid dimensions must be non-zero",
            ));
        }

        if block.contains(&0)
        {
            return Err(ComputeError::InvalidArgument(
                "block dimensions must be non-zero",
            ));
        }

        Ok(Self {
            grid,
            block,
            shared_memory_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_launch_is_accepted() {
        assert!(LaunchConfig::new([8, 1, 1], [32, 1, 1], 0).is_ok());
    }

    #[test]
    fn zero_grid_is_rejected() {
        assert!(LaunchConfig::new([0, 1, 1], [32, 1, 1], 0).is_err());
    }

    #[test]
    fn zero_block_is_rejected() {
        assert!(LaunchConfig::new([1, 1, 1], [0, 1, 1], 0).is_err());
    }
}
