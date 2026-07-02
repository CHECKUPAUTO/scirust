use std::fs;
use std::path::Path;

use scirust_core::autodiff::reverse::Tensor;
use scirust_core::error::Result;

use crate::config::SciAgentConfig;
use crate::model::SciAgentModel;

#[derive(Clone, Debug)]
pub struct CheckpointMeta {
    pub step: usize,
    pub loss: f32,
    pub lr: f32,
    pub config: SciAgentConfig,
}

pub fn save_checkpoint(
    model: &SciAgentModel,
    meta: &CheckpointMeta,
    path: &Path,
) -> Result<()> {
    fs::create_dir_all(path).map_err(|e| format!("Cannot create checkpoint dir: {e}"))?;

    let meta_path = path.join("meta.json");
    let meta_json = serde_json::json!({
        "step": meta.step,
        "loss": meta.loss,
        "lr": meta.lr,
        "config": {
            "vocab_size": meta.config.vocab_size,
            "d_model": meta.config.d_model,
            "n_layers": meta.config.n_layers,
            "n_heads": meta.config.n_heads,
            "n_kv_heads": meta.config.n_kv_heads,
            "d_ff": meta.config.d_ff,
            "max_seq_len": meta.config.max_seq_len,
            "rope_theta": meta.config.rope_theta,
            "tie_embeddings": meta.config.tie_embeddings,
            "use_bias": meta.config.use_bias,
            "eps": meta.config.eps,
        },
    });
    let meta_str = serde_json::to_string_pretty(&meta_json)
        .map_err(|e| format!("Cannot serialize meta: {e}"))?;
    fs::write(&meta_path, meta_str).map_err(|e| format!("Cannot write meta: {e}"))?;

    let state = model.state_dict();
    let mut tensors: Vec<(String, Tensor)> = state.into_iter().collect();
    tensors.sort_by(|a, b| a.0.cmp(&b.0)); // deterministic ordering
    scirust_core::io::safetensors::save_safetensors(&tensors, path.join("model.safetensors"))
        .map_err(|e| format!("Cannot save safetensors: {e}"))?;

    Ok(())
}

pub fn load_checkpoint(model: &mut SciAgentModel, path: &Path) -> Result<CheckpointMeta> {
    let meta_path = path.join("meta.json");
    let meta_str =
        fs::read_to_string(&meta_path).map_err(|e| format!("Cannot read meta: {e}"))?;
    let meta_json: serde_json::Value =
        serde_json::from_str(&meta_str).map_err(|e| format!("Cannot parse meta: {e}"))?;

    let cfg = &meta_json["config"];
    let config = SciAgentConfig {
        vocab_size: cfg["vocab_size"].as_u64().unwrap_or(32768) as usize,
        d_model: cfg["d_model"].as_u64().unwrap_or(1024) as usize,
        n_layers: cfg["n_layers"].as_u64().unwrap_or(24) as usize,
        n_heads: cfg["n_heads"].as_u64().unwrap_or(16) as usize,
        n_kv_heads: cfg["n_kv_heads"].as_u64().unwrap_or(4) as usize,
        d_ff: cfg["d_ff"].as_u64().unwrap_or(2816) as usize,
        max_seq_len: cfg["max_seq_len"].as_u64().unwrap_or(8192) as usize,
        rope_theta: cfg["rope_theta"].as_f64().unwrap_or(1_000_000.0) as f32,
        tie_embeddings: cfg["tie_embeddings"].as_bool().unwrap_or(true),
        use_bias: cfg["use_bias"].as_bool().unwrap_or(false),
        eps: cfg["eps"].as_f64().unwrap_or(1e-5) as f32,
    };

    let meta = CheckpointMeta {
        step: meta_json["step"].as_u64().unwrap_or(0) as usize,
        loss: meta_json["loss"].as_f64().unwrap_or(0.0) as f32,
        lr: meta_json["lr"].as_f64().unwrap_or(0.0) as f32,
        config,
    };

    let safetensors_path = path.join("model.safetensors");
    let state = scirust_core::io::safetensors::load_safetensors(&safetensors_path)
        .map_err(|e| format!("Cannot load safetensors: {e}"))?;

    model.load_state_dict(&state)?;
    Ok(meta)
}

pub fn latest_checkpoint(dir: &Path) -> Option<std::path::PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut dirs: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().ok().map_or(false, |t| t.is_dir()))
        .collect();
    dirs.sort_by_key(|d| d.file_name());

    let last = dirs.last()?;
    if last.path().join("meta.json").exists() {
        Some(last.path())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SciAgentConfig;
    use crate::model::SciAgentModel;
    use scirust_core::autodiff::reverse::Tape;
    use std::path::PathBuf;

    #[test]
    fn test_save_load_roundtrip() {
        let cfg = SciAgentConfig::debug();
        let mut model = SciAgentModel::new(&cfg);
        let tape = Tape::new();
        let input_ids = vec![4usize, 5, 6, 7];
        let _ = model.forward(&tape, &input_ids, 4);

        let dir = PathBuf::from("/tmp/scirust_test_ckpt");
        let _ = fs::remove_dir_all(&dir);
        let meta = CheckpointMeta {
            step: 42,
            loss: 1.234,
            lr: 0.001,
            config: cfg.clone(),
        };
        save_checkpoint(&model, &meta, &dir).expect("save should succeed");

        let mut loaded = SciAgentModel::new(&cfg);
        let loaded_meta = load_checkpoint(&mut loaded, &dir).expect("load should succeed");
        assert_eq!(loaded_meta.step, 42);
        assert!((loaded_meta.loss - 1.234).abs() < 1e-5);

        let tape2 = Tape::new();
        let logits_orig = model.forward(&tape2, &input_ids, 4);
        let tape3 = Tape::new();
        let logits_loaded = loaded.forward(&tape3, &input_ids, 4);
        let v_orig = tape2.value(logits_orig.idx());
        let v_loaded = tape3.value(logits_loaded.idx());
        assert_eq!(v_orig.data, v_loaded.data, "Weights should match after load");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_latest_checkpoint_none() {
        let dir = PathBuf::from("/tmp/scirust_test_nonexistent");
        assert!(latest_checkpoint(&dir).is_none());
    }
}
