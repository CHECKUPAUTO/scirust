//! Training checkpoint — save & restore model parameters + optimizer state.
//!
//! Writes two files per checkpoint:
//! - `checkpoint.json` — metadata, optimizer state, epoch/step
//! - `weights.json` — model weights as key → flat f32 arrays
//!
//! # Example
//!
//! ```ignore
//! use scirust_core::checkpoint::{Checkpoint, OptimizerState, save_checkpoint, load_checkpoint};
//!
//! let opt_state = OptimizerState {
//!     learning_rate: 0.001, step: 100, epoch: 5,
//!     m: HashMap::new(), v: HashMap::new(),
//!     beta1_t: 0.9f32.powi(100), beta2_t: 0.999f32.powi(100),
//! };
//! let weights = HashMap::from([("fc1.weight".into(), vec![0.1f32; 256])]);
//! save_checkpoint("ckpt_epoch5", &weights, &opt_state, 5, 100).unwrap();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Optimizer state snapshot for exact training resumption.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OptimizerState {
    /// Current learning rate.
    pub learning_rate: f32,
    /// Global training step.
    pub step: usize,
    /// Current epoch.
    pub epoch: usize,
    /// Adam first moment (m) buffers: param_name → flat f32 values.
    pub m: HashMap<String, Vec<f32>>,
    /// Adam second moment (v) buffers: param_name → flat f32 values.
    pub v: HashMap<String, Vec<f32>>,
    /// beta1^t for Adam bias correction.
    pub beta1_t: f32,
    /// beta2^t for Adam bias correction.
    pub beta2_t: f32,
}

/// Complete checkpoint bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// SciRust version (from Cargo.toml).
    pub scirust_version: String,
    /// Epoch at checkpoint time.
    pub epoch: usize,
    /// Global training step.
    pub step: usize,
    /// Optimizer state.
    pub optimizer_state: OptimizerState,
    /// Model weights: param_name → flat f32 slice.
    pub weights: HashMap<String, Vec<f32>>,
    /// Free-form metadata (loss, config, etc.).
    pub metadata: HashMap<String, String>,
}

/// Save a checkpoint to a directory.
///
/// Creates `{dir}/checkpoint.json` with all data embedded.
pub fn save_checkpoint(
    dir: impl AsRef<Path>,
    weights: &HashMap<String, Vec<f32>>,
    opt_state: &OptimizerState,
    epoch: usize,
    step: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;

    let checkpoint = Checkpoint {
        scirust_version: env!("CARGO_PKG_VERSION").to_string(),
        epoch,
        step,
        optimizer_state: opt_state.clone(),
        weights: weights.clone(),
        metadata: HashMap::new(),
    };

    let json = serde_json::to_string_pretty(&checkpoint)?;
    fs::write(dir.join("checkpoint.json"), json)?;
    Ok(())
}

/// Load a checkpoint from a directory.
pub fn load_checkpoint(dir: impl AsRef<Path>) -> Result<Checkpoint, Box<dyn std::error::Error>> {
    let dir = dir.as_ref();
    let json = fs::read_to_string(dir.join("checkpoint.json"))?;
    let checkpoint: Checkpoint = serde_json::from_str(&json)?;
    Ok(checkpoint)
}

/// List all checkpoints in a directory, sorted by epoch.
pub fn list_checkpoints(
    parent_dir: impl AsRef<Path>,
) -> Result<Vec<(usize, std::path::PathBuf)>, Box<dyn std::error::Error>> {
    let parent = parent_dir.as_ref();
    let mut checkpoints = Vec::new();

    if !parent.exists()
    {
        return Ok(checkpoints);
    }

    for entry in fs::read_dir(parent)?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir()
        {
            let ckpt_file = path.join("checkpoint.json");
            if ckpt_file.exists()
            {
                if let Ok(json) = fs::read_to_string(&ckpt_file)
                {
                    if let Ok(ckpt) = serde_json::from_str::<Checkpoint>(&json)
                    {
                        checkpoints.push((ckpt.epoch, path));
                    }
                }
            }
        }
    }

    checkpoints.sort_by_key(|(epoch, _)| *epoch);
    Ok(checkpoints)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_roundtrip() {
        let dir = std::env::temp_dir().join("scirust_ckpt_test_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);

        let mut weights = HashMap::new();
        weights.insert("fc1.weight".into(), vec![0.1, 0.2, 0.3]);
        weights.insert("fc1.bias".into(), vec![0.01, 0.02]);

        let opt_state = OptimizerState {
            learning_rate: 0.001,
            step: 100,
            epoch: 5,
            m: HashMap::new(),
            v: HashMap::new(),
            beta1_t: 0.9f32.powi(100),
            beta2_t: 0.999f32.powi(100),
        };

        save_checkpoint(&dir, &weights, &opt_state, 5, 100).unwrap();

        let loaded = load_checkpoint(&dir).unwrap();
        assert_eq!(loaded.epoch, 5);
        assert_eq!(loaded.step, 100);
        assert_eq!(loaded.optimizer_state.learning_rate, 0.001);
        assert_eq!(
            loaded.weights.get("fc1.weight").unwrap(),
            &vec![0.1, 0.2, 0.3]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_list_checkpoints() {
        let parent = std::env::temp_dir().join("scirust_ckpt_test_list");
        let _ = std::fs::remove_dir_all(&parent);

        let mut weights = HashMap::new();
        weights.insert("w".into(), vec![1.0]);

        let opt = OptimizerState {
            learning_rate: 0.01,
            step: 0,
            epoch: 0,
            m: HashMap::new(),
            v: HashMap::new(),
            beta1_t: 0.9,
            beta2_t: 0.999,
        };

        save_checkpoint(parent.join("ckpt_epoch1"), &weights, &opt, 1, 10).unwrap();
        save_checkpoint(parent.join("ckpt_epoch3"), &weights, &opt, 3, 30).unwrap();
        save_checkpoint(parent.join("ckpt_epoch2"), &weights, &opt, 2, 20).unwrap();

        let list = list_checkpoints(&parent).unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].0, 1);
        assert_eq!(list[1].0, 2);
        assert_eq!(list[2].0, 3);

        let _ = std::fs::remove_dir_all(&parent);
    }
}
