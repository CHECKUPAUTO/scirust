#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"
EXAMPLE = ROOT / "scirust-sciagent/examples/cuda_pretrain.rs"
TEST = ROOT / "scirust-sciagent/tests/cuda_parity.rs"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:140]!r}")
    return text.replace(old, new, count)


IMPORTS_OLD = '''use scirust_core::autodiff::reverse::Tensor;\nuse scirust_core::autodiff::scheduler::LrSchedule;\nuse scirust_cuda::{CudaChain, CudaF32, CudaMatrix};\n'''
IMPORTS_NEW = '''use std::collections::HashMap;\nuse std::path::Path;\n\nuse scirust_core::autodiff::reverse::Tensor;\nuse scirust_core::autodiff::scheduler::LrSchedule;\nuse scirust_cuda::{CudaChain, CudaF32, CudaMatrix};\n'''

META_STRUCT = r'''
/// Persisted CUDA optimizer/schedule metadata. Unlike legacy checkpoints that only
/// saved model weights, B32 checkpoints restore AdamW moments and bias-correction
/// step exactly. Schedule fields let an interrupted run continue its original LR
/// curve unless the caller explicitly overrides it.
#[derive(Clone, Debug)]
pub struct CudaOptimizerResume {
    pub step: usize,
    pub base_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub betas: (f32, f32),
    pub adam_eps: f32,
    pub weight_decay: f32,
}

fn optimizer_tensor(
    chain: &CudaChain,
    x: &CudaF32,
    rows: usize,
    cols: usize,
) -> Tensor {
    Tensor::from_vec(chain.download_f32(x), rows, cols)
}

fn load_optimizer_tensor(
    chain: &CudaChain,
    state: &HashMap<String, Tensor>,
    name: &str,
    rows: usize,
    cols: usize,
) -> std::result::Result<CudaF32, String> {
    let t = state
        .get(name)
        .ok_or_else(|| format!("optimizer checkpoint missing tensor '{name}'"))?;
    if (t.rows, t.cols) != (rows, cols) {
        return Err(format!(
            "optimizer tensor '{name}' has shape {}x{}, expected {rows}x{cols}",
            t.rows, t.cols
        ));
    }
    Ok(chain.upload_f32(&t.data))
}
'''

OPT_METHODS = r'''
    /// Save AdamW moments + optimizer step next to a model checkpoint. Model fp32
    /// masters themselves are already represented by `model.safetensors` after
    /// `sync_to_model`; duplicating them here would add ~1.2 GB without information.
    pub fn save_optimizer_state(
        &self,
        cfg: &CudaPretrainConfig,
        path: &Path,
    ) -> std::result::Result<(), String> {
        let ch = &self.model.chain;
        let mut tensors: Vec<(String, Tensor)> = Vec::new();
        let (er, ec) = (self.model.embedding.rows(), self.model.embedding.cols());
        tensors.push((
            "embedding.m".into(),
            optimizer_tensor(ch, &self.m_embedding, er, ec),
        ));
        tensors.push((
            "embedding.v".into(),
            optimizer_tensor(ch, &self.v_embedding, er, ec),
        ));
        let (nr, nc) = (self.model.final_norm.rows(), self.model.final_norm.cols());
        tensors.push((
            "final_norm.m".into(),
            optimizer_tensor(ch, &self.m_final_norm, nr, nc),
        ));
        tensors.push((
            "final_norm.v".into(),
            optimizer_tensor(ch, &self.v_final_norm, nr, nc),
        ));
        for i in 0..self.model.blocks.len() {
            let b = &self.model.blocks[i];
            let mm = &self.m_blocks[i];
            let vv = &self.v_blocks[i];
            macro_rules! push_pair {
                ($field:ident) => {{
                    let rows = b.$field.rows();
                    let cols = b.$field.cols();
                    tensors.push((
                        format!("blocks.{i}.{}.m", stringify!($field)),
                        optimizer_tensor(ch, &mm.$field, rows, cols),
                    ));
                    tensors.push((
                        format!("blocks.{i}.{}.v", stringify!($field)),
                        optimizer_tensor(ch, &vv.$field, rows, cols),
                    ));
                }};
            }
            push_pair!(norm1);
            push_pair!(wq);
            push_pair!(wk);
            push_pair!(wv);
            push_pair!(wo);
            push_pair!(norm2);
            push_pair!(wg);
            push_pair!(wu);
            push_pair!(wd);
        }
        tensors.sort_by(|a, b| a.0.cmp(&b.0));
        scirust_core::io::safetensors::save_safetensors(
            &tensors,
            path.join("optimizer.safetensors"),
        )
        .map_err(|e| format!("cannot save optimizer.safetensors: {e}"))?;

        let meta = serde_json::json!({
            "version": 1,
            "step": self.step,
            "base_lr": cfg.base_lr,
            "min_lr": cfg.min_lr,
            "warmup_steps": cfg.warmup_steps,
            "total_steps": cfg.total_steps,
            "betas": [cfg.betas.0, cfg.betas.1],
            "adam_eps": cfg.adam_eps,
            "weight_decay": cfg.weight_decay,
        });
        let encoded = serde_json::to_string_pretty(&meta)
            .map_err(|e| format!("cannot serialize optimizer metadata: {e}"))?;
        std::fs::write(path.join("optimizer.json"), encoded)
            .map_err(|e| format!("cannot write optimizer metadata: {e}"))?;
        Ok(())
    }

    /// Restore AdamW state. `Ok(None)` means a legacy checkpoint with no optimizer
    /// sidecar; malformed/incomplete B32 state is an error and must not silently
    /// degrade to zero moments.
    pub fn load_optimizer_state(
        &mut self,
        path: &Path,
    ) -> std::result::Result<Option<CudaOptimizerResume>, String> {
        let state_path = path.join("optimizer.safetensors");
        let meta_path = path.join("optimizer.json");
        let has_state = state_path.exists();
        let has_meta = meta_path.exists();
        if !has_state && !has_meta {
            return Ok(None);
        }
        if has_state != has_meta {
            return Err(format!(
                "incomplete optimizer checkpoint at {} (need optimizer.safetensors + optimizer.json)",
                path.display()
            ));
        }
        let state = scirust_core::io::safetensors::load_safetensors(&state_path)
            .map_err(|e| format!("cannot load {}: {e}", state_path.display()))?;
        let raw = std::fs::read_to_string(&meta_path)
            .map_err(|e| format!("cannot read {}: {e}", meta_path.display()))?;
        let meta: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("cannot parse {}: {e}", meta_path.display()))?;
        if meta["version"].as_u64() != Some(1) {
            return Err(format!("unsupported optimizer checkpoint version in {}", meta_path.display()));
        }
        let step = meta["step"]
            .as_u64()
            .ok_or_else(|| "optimizer metadata missing step".to_string())? as usize;
        if step > u32::MAX as usize {
            return Err(format!("optimizer step {step} exceeds u32::MAX"));
        }
        let number = |key: &str| -> std::result::Result<f32, String> {
            meta[key]
                .as_f64()
                .map(|x| x as f32)
                .ok_or_else(|| format!("optimizer metadata missing {key}"))
        };
        let usize_field = |key: &str| -> std::result::Result<usize, String> {
            meta[key]
                .as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| format!("optimizer metadata missing {key}"))
        };
        let betas = meta["betas"]
            .as_array()
            .filter(|x| x.len() == 2)
            .ok_or_else(|| "optimizer metadata missing betas".to_string())?;
        let beta0 = betas[0]
            .as_f64()
            .ok_or_else(|| "optimizer beta0 invalid".to_string())? as f32;
        let beta1 = betas[1]
            .as_f64()
            .ok_or_else(|| "optimizer beta1 invalid".to_string())? as f32;

        let ch = &self.model.chain;
        let (er, ec) = (self.model.embedding.rows(), self.model.embedding.cols());
        self.m_embedding = load_optimizer_tensor(ch, &state, "embedding.m", er, ec)?;
        self.v_embedding = load_optimizer_tensor(ch, &state, "embedding.v", er, ec)?;
        let (nr, nc) = (self.model.final_norm.rows(), self.model.final_norm.cols());
        self.m_final_norm = load_optimizer_tensor(ch, &state, "final_norm.m", nr, nc)?;
        self.v_final_norm = load_optimizer_tensor(ch, &state, "final_norm.v", nr, nc)?;
        for i in 0..self.model.blocks.len() {
            let b = &self.model.blocks[i];
            macro_rules! load_pair {
                ($field:ident) => {{
                    let rows = b.$field.rows();
                    let cols = b.$field.cols();
                    self.m_blocks[i].$field = load_optimizer_tensor(
                        ch,
                        &state,
                        &format!("blocks.{i}.{}.m", stringify!($field)),
                        rows,
                        cols,
                    )?;
                    self.v_blocks[i].$field = load_optimizer_tensor(
                        ch,
                        &state,
                        &format!("blocks.{i}.{}.v", stringify!($field)),
                        rows,
                        cols,
                    )?;
                }};
            }
            load_pair!(norm1);
            load_pair!(wq);
            load_pair!(wk);
            load_pair!(wv);
            load_pair!(wo);
            load_pair!(norm2);
            load_pair!(wg);
            load_pair!(wu);
            load_pair!(wd);
        }
        self.step = step as u32;
        Ok(Some(CudaOptimizerResume {
            step,
            base_lr: number("base_lr")?,
            min_lr: number("min_lr")?,
            warmup_steps: usize_field("warmup_steps")?,
            total_steps: usize_field("total_steps")?,
            betas: (beta0, beta1),
            adam_eps: number("adam_eps")?,
            weight_decay: number("weight_decay")?,
        }))
    }

    /// Save model + optimizer into a hidden partial directory and atomically rename
    /// it only after both payloads are complete. `latest_checkpoint` therefore never
    /// selects a crash-torn training state.
    fn save_training_checkpoint(
        &self,
        model: &SciAgentModel,
        meta: &CheckpointMeta,
        cfg: &CudaPretrainConfig,
        final_dir: &Path,
    ) -> std::result::Result<(), String> {
        if self.step as usize != meta.step {
            return Err(format!(
                "optimizer step {} does not match checkpoint step {}",
                self.step, meta.step
            ));
        }
        let parent = final_dir.parent().unwrap_or_else(|| Path::new("."));
        let name = final_dir
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| format!("invalid checkpoint path {}", final_dir.display()))?;
        let partial = parent.join(format!(".{name}.partial"));
        if partial.exists() {
            std::fs::remove_dir_all(&partial)
                .map_err(|e| format!("cannot remove stale {}: {e}", partial.display()))?;
        }
        save_checkpoint(model, meta, &partial)
            .map_err(|e| format!("cannot save model checkpoint: {e}"))?;
        self.save_optimizer_state(cfg, &partial)?;
        if final_dir.exists() {
            return Err(format!("checkpoint already exists: {}", final_dir.display()));
        }
        std::fs::rename(&partial, final_dir).map_err(|e| {
            format!(
                "cannot atomically publish {} -> {}: {e}",
                partial.display(),
                final_dir.display()
            )
        })?;
        Ok(())
    }

'''


def patch_model(text: str) -> str:
    if "pub struct CudaOptimizerResume" in text:
        raise SystemExit("model already B32 patched")
    text = must_replace(text, IMPORTS_OLD, IMPORTS_NEW)
    # Put helpers immediately before the training diagnostics structure.
    marker = "/// Resident diagnostics for one optimizer step. Keeping these tiny buffers alive\n"
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing diagnostics marker")
    text = text[:pos] + META_STRUCT + "\n" + text[pos:]
    # Add methods just before eval_loss, after train-step methods.
    marker2 = "    /// Mean cross-entropy over up to `max_windows` non-overlapping `seq_len` windows\n"
    pos2 = text.find(marker2)
    if pos2 < 0:
        raise SystemExit("missing optimizer-method insertion point")
    text = text[:pos2] + OPT_METHODS + text[pos2:]
    old_save = '''                match save_checkpoint(model, &meta, &dir)\n                {\n                    Ok(()) =>\n'''
    new_save = '''                match self.save_training_checkpoint(model, &meta, cfg, &dir)\n                {\n                    Ok(()) =>\n'''
    text = must_replace(text, old_save, new_save)
    return text


OLD_RESUME = '''    let mut model = SciAgentModel::new(&config);\n    let mut start_step = 0usize;\n    if let Some((path, meta)) = &resume\n    {\n        match load_checkpoint(&mut model, path)\n        {\n            Ok(_) =>\n            {\n                start_step = meta.step;\n                println!(\n                    "resuming from {} (step {}, loss {:.4})",\n                    path.display(),\n                    meta.step,\n                    meta.loss\n                );\n            },\n            Err(e) => eprintln!("could not load {}: {e}; starting fresh", path.display()),\n        }\n    }\n\n    let Some(mut trainer) = CudaTrainer::from_model(&model)\n    else\n    {\n        eprintln!("no CUDA device available. Run on the Jetson Thor (needs the CUDA toolkit).");\n        std::process::exit(2);\n    };\n    trainer.reset_step(); // fresh AdamW moments; the LR schedule continues via start_step\n'''
NEW_RESUME = '''    let mut model = SciAgentModel::new(&config);\n    let mut start_step = 0usize;\n    let mut loaded_resume_path: Option<std::path::PathBuf> = None;\n    if let Some((path, meta)) = &resume\n    {\n        match load_checkpoint(&mut model, path)\n        {\n            Ok(_) =>\n            {\n                start_step = meta.step;\n                loaded_resume_path = Some(path.clone());\n                println!(\n                    "resuming model from {} (step {}, loss {:.4})",\n                    path.display(),\n                    meta.step,\n                    meta.loss\n                );\n            },\n            Err(e) => eprintln!("could not load {}: {e}; starting fresh", path.display()),\n        }\n    }\n\n    let Some(mut trainer) = CudaTrainer::from_model(&model)\n    else\n    {\n        eprintln!("no CUDA device available. Run on the Jetson Thor (needs the CUDA toolkit).");\n        std::process::exit(2);\n    };\n    let optimizer_resume = if let Some(path) = loaded_resume_path.as_deref()\n    {\n        match trainer.load_optimizer_state(path)\n        {\n            Ok(Some(state)) =>\n            {\n                if state.step != start_step\n                {\n                    eprintln!(\n                        "optimizer checkpoint step {} != model step {}; refusing mismatched resume",\n                        state.step, start_step\n                    );\n                    std::process::exit(1);\n                }\n                println!("optimizer state restored exactly at step {} (AdamW m/v + bias correction)", state.step);\n                Some(state)\n            },\n            Ok(None) =>\n            {\n                eprintln!(\n                    "legacy checkpoint has no optimizer state; AdamW moments restart once. \\\n                     The next B32 checkpoint will be exactly resumable."\n                );\n                trainer.reset_step();\n                None\n            },\n            Err(e) =>\n            {\n                eprintln!("optimizer checkpoint is present but invalid: {e}");\n                std::process::exit(1);\n            },\n        }\n    }\n    else\n    {\n        trainer.reset_step();\n        None\n    };\n'''

OLD_SCHED = '''    // Total-step target. `SCIAGENT_TOTAL_STEPS` sets it ABSOLUTELY — so resuming an\n    // interrupted run to its original target is just `SCIAGENT_TOTAL_STEPS=40000`\n    // (no arithmetic, no overshoot from the additive default). Otherwise it stays\n    // additive (`start_step + STEPS`), the historical behavior. Warmup is 10% of the\n    // ACTUAL remaining run either way, so a short resume re-warms briefly (cushioning\n    // the AdamW-moment reset) rather than re-ramping a full fresh-run warmup.\n    let steps_env = env_usize("SCIAGENT_STEPS", 300);\n    let total_steps = std::env::var("SCIAGENT_TOTAL_STEPS")\n        .ok()\n        .and_then(|v| v.parse::<usize>().ok())\n        .filter(|&t| t > start_step)\n        .unwrap_or(start_step + steps_env);\n    let run_len = total_steps.saturating_sub(start_step).max(1);\n    let warmup_extra = std::env::var("SCIAGENT_WARMUP")\n        .ok()\n        .and_then(|v| v.parse::<usize>().ok())\n        .unwrap_or((run_len / 10).max(1));\n    // LR must key off model *size*, not vocab: a 270M byte-level model (vocab 256)\n    // diverges at the 3e-3 that suits a tiny demo. Small trunks (d_model ≤ 256) can\n    // take the hot 3e-3; anything larger gets the standard 3e-4 (with warmup+cosine).\n    let default_lr = if config.d_model <= 256 { 3e-3 } else { 3e-4 };\n    let base_lr = std::env::var("SCIAGENT_LR")\n        .ok()\n        .and_then(|v| v.parse().ok())\n        .unwrap_or(default_lr);\n'''
NEW_SCHED = '''    // Exact B32 resumes inherit the saved optimizer/LR trajectory by default. An\n    // explicit environment override still wins. Legacy checkpoints (no AdamW sidecar)\n    // keep the historical one-time re-warm that cushions their zero-moment restart.\n    let steps_env = env_usize("SCIAGENT_STEPS", 300);\n    let total_steps = std::env::var("SCIAGENT_TOTAL_STEPS")\n        .ok()\n        .and_then(|v| v.parse::<usize>().ok())\n        .filter(|&t| t > start_step)\n        .or_else(|| optimizer_resume.as_ref().map(|s| s.total_steps).filter(|&t| t > start_step))\n        .unwrap_or(start_step + steps_env);\n    let run_len = total_steps.saturating_sub(start_step).max(1);\n    let explicit_warmup = std::env::var("SCIAGENT_WARMUP")\n        .ok()\n        .and_then(|v| v.parse::<usize>().ok());\n    let warmup_steps = if let Some(extra) = explicit_warmup\n    {\n        start_step + extra\n    }\n    else if let Some(saved) = optimizer_resume.as_ref()\n    {\n        saved.warmup_steps\n    }\n    else\n    {\n        start_step + (run_len / 10).max(1)\n    };\n    // LR must key off model size for fresh/legacy runs. Exact resumes inherit the\n    // saved schedule unless SCIAGENT_LR explicitly requests a new peak LR.\n    let size_default_lr = if config.d_model <= 256 { 3e-3 } else { 3e-4 };\n    let explicit_lr = std::env::var("SCIAGENT_LR")\n        .ok()\n        .and_then(|v| v.parse::<f32>().ok());\n    let base_lr = explicit_lr\n        .or_else(|| optimizer_resume.as_ref().map(|s| s.base_lr))\n        .unwrap_or(size_default_lr);\n    let min_lr = if explicit_lr.is_some()\n    {\n        base_lr * 0.1\n    }\n    else\n    {\n        optimizer_resume\n            .as_ref()\n            .map(|s| s.min_lr)\n            .unwrap_or(base_lr * 0.1)\n    };\n'''


def patch_example(text: str) -> str:
    if "optimizer state restored exactly" in text:
        raise SystemExit("example already B32 patched")
    text = must_replace(text, OLD_RESUME, NEW_RESUME)
    text = must_replace(text, OLD_SCHED, NEW_SCHED)
    old_eps = '''    let adam_eps = std::env::var("SCIAGENT_EPS")\n        .ok()\n        .and_then(|v| v.parse().ok())\n        .unwrap_or(1e-5f32);\n'''
    new_eps = '''    let adam_eps = std::env::var("SCIAGENT_EPS")\n        .ok()\n        .and_then(|v| v.parse().ok())\n        .or_else(|| optimizer_resume.as_ref().map(|s| s.adam_eps))\n        .unwrap_or(1e-5f32);\n'''
    text = must_replace(text, old_eps, new_eps)
    text = must_replace(text, "        min_lr: base_lr * 0.1,\n        warmup_steps: start_step + warmup_extra,\n", "        min_lr,\n        warmup_steps,\n")
    # Preserve the saved betas/weight decay on exact resume; current production path
    # otherwise uses the existing defaults/zero decay.
    text = must_replace(
        text,
        "    let cfg = CudaPretrainConfig {\n        base_lr,\n",
        "    let resume_betas = optimizer_resume.as_ref().map(|s| s.betas);\n    let resume_weight_decay = optimizer_resume.as_ref().map(|s| s.weight_decay);\n    let cfg = CudaPretrainConfig {\n        base_lr,\n",
    )
    text = must_replace(
        text,
        "        weight_decay: 0.0,\n        adam_eps,\n",
        "        betas: resume_betas.unwrap_or(CudaPretrainConfig::default().betas),\n        weight_decay: resume_weight_decay.unwrap_or(0.0),\n        adam_eps,\n",
    )
    return text


TEST_SRC = r'''

/// B32: an interrupted CUDA run restored from model + optimizer sidecars must take
/// the same next AdamW step as the uninterrupted trainer. This is the regression
/// test for moment/bias-correction loss on resume.
#[test]
fn cuda_optimizer_resume_matches_uninterrupted_next_step() {
    use scirust_sciagent::cuda_model::CudaPretrainConfig;
    use scirust_sciagent::train::checkpoint::{CheckpointMeta, load_checkpoint, save_checkpoint};
    use std::path::PathBuf;

    let config = tiny_tied();
    let mut model = SciAgentModel::new(&config);
    let Some(mut continuous) = CudaTrainer::from_model(&model) else {
        eprintln!("cuda: no device, skipping optimizer-resume parity");
        return;
    };
    let tokens: Vec<u32> = (0..8).map(|i| ((i * 7 + 3) % config.vocab_size) as u32).collect();
    let targets: Vec<u32> = (0..8).map(|i| ((i * 5 + 1) % config.vocab_size) as u32).collect();
    let (lr, betas, eps, wd) = (3e-3f32, (0.9f32, 0.95f32), 1e-5f32, 0.0f32);
    for _ in 0..3 {
        continuous.train_step(&tokens, &targets, lr, betas, eps, wd);
    }
    continuous.sync_to_model(&mut model);

    let dir = PathBuf::from("/tmp/scirust_cuda_optimizer_resume");
    let _ = std::fs::remove_dir_all(&dir);
    let meta = CheckpointMeta {
        step: 3,
        loss: 0.0,
        lr,
        config: config.clone(),
    };
    save_checkpoint(&model, &meta, &dir).expect("save model checkpoint");
    let cfg = CudaPretrainConfig {
        base_lr: lr,
        min_lr: lr * 0.1,
        warmup_steps: 1,
        total_steps: 10,
        betas,
        adam_eps: eps,
        weight_decay: wd,
        ..Default::default()
    };
    continuous
        .save_optimizer_state(&cfg, &dir)
        .expect("save optimizer state");

    let mut resumed_model = SciAgentModel::new(&config);
    load_checkpoint(&mut resumed_model, &dir).expect("reload model checkpoint");
    let mut resumed = CudaTrainer::from_model(&resumed_model).expect("CUDA trainer");
    let resume_meta = resumed
        .load_optimizer_state(&dir)
        .expect("load optimizer state")
        .expect("B32 optimizer sidecar");
    assert_eq!(resume_meta.step, 3);

    let loss_a = continuous.train_step(&tokens, &targets, lr, betas, eps, wd);
    let loss_b = resumed.train_step(&tokens, &targets, lr, betas, eps, wd);
    assert!((loss_a - loss_b).abs() < 1e-4, "resume loss mismatch: {loss_a} vs {loss_b}");

    let mut a = SciAgentModel::new(&config);
    let mut b = SciAgentModel::new(&config);
    continuous.sync_to_model(&mut a);
    resumed.sync_to_model(&mut b);
    let logits_a = CudaModel::from_model(&a).expect("CUDA model A").forward(&tokens);
    let logits_b = CudaModel::from_model(&b).expect("CUDA model B").forward(&tokens);
    let e = rel_err(&logits_a, &logits_b);
    assert!(e < 1e-4, "resumed optimizer diverged from uninterrupted step: rel_err {e}");
    let _ = std::fs::remove_dir_all(&dir);
}
'''


def patch_test(text: str) -> str:
    if "cuda_optimizer_resume_matches_uninterrupted_next_step" in text:
        raise SystemExit("test already B32 patched")
    return text.rstrip() + TEST_SRC + "\n"


MODEL.write_text(patch_model(MODEL.read_text()))
EXAMPLE.write_text(patch_example(EXAMPLE.read_text()))
TEST.write_text(patch_test(TEST.read_text()))
print("patched B32 exact optimizer resume")
