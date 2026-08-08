#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"
EXAMPLE = ROOT / "scirust-sciagent/examples/cuda_pretrain.rs"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:180]!r}")
    return text.replace(old, new, count)


def patch_model(text: str) -> str:
    if "pub save_interval_seconds:" in text:
        print("model already B41 patched")
        return text

    text = must_replace(
        text,
        '''    /// Write a checkpoint every this many steps (0 = never).\n    pub save_interval: usize,\n    /// Directory the `step_N/` checkpoints are written under.\n''',
        '''    /// Write an exact recovery checkpoint every this many optimizer steps\n    /// (`0` disables the step cadence).\n    pub save_interval: usize,\n    /// Optional wall-clock recovery cadence in seconds (`0` disables it). The\n    /// checkpoint is still published only after a completed optimizer step.\n    pub save_interval_seconds: u64,\n    /// Directory the `step_N/` checkpoints are written under.\n''',
    )
    text = must_replace(
        text,
        '''            save_interval: 500,\n            checkpoint_dir: "checkpoints".to_string(),\n''',
        '''            save_interval: 500,\n            save_interval_seconds: 0,\n            checkpoint_dir: "checkpoints".to_string(),\n''',
    )
    text = must_replace(
        text,
        '''        let t0 = std::time::Instant::now();\n        let mut packed_inputs = Vec::with_capacity(batch * s);\n''',
        '''        let t0 = std::time::Instant::now();\n        let mut last_checkpoint_at = t0;\n        let mut last_checkpoint_step: Option<usize> = None;\n        let checkpointing_enabled = cfg.save_interval > 0 || cfg.save_interval_seconds > 0;\n        let mut packed_inputs = Vec::with_capacity(batch * s);\n''',
    )
    text = must_replace(
        text,
        '''            let need_save = cfg.save_interval > 0 && step.is_multiple_of(cfg.save_interval);\n            let need_flush = pending.len() >= telemetry\n''',
        '''            let need_save_by_step =\n                cfg.save_interval > 0 && step.is_multiple_of(cfg.save_interval);\n            let need_save_by_time = cfg.save_interval_seconds > 0\n                && last_checkpoint_at.elapsed().as_secs() >= cfg.save_interval_seconds;\n            let need_save = need_save_by_step || need_save_by_time;\n            let need_flush = pending.len() >= telemetry\n''',
    )
    text = must_replace(
        text,
        '''                        println!("  exact checkpoint → {}", dir.display());\n                        if !val_windows.is_empty()\n''',
        '''                        println!("  exact checkpoint → {}", dir.display());\n                        last_checkpoint_step = Some(step);\n                        last_checkpoint_at = std::time::Instant::now();\n                        if !val_windows.is_empty()\n''',
    )
    text = must_replace(
        text,
        '''                    Err(e) => eprintln!("  checkpoint at step {step} failed: {e}"),\n''',
        '''                    Err(e) =>\n                    {\n                        eprintln!("  checkpoint at step {step} failed: {e}");\n                        // Back off after an I/O failure instead of retrying every step.\n                        last_checkpoint_at = std::time::Instant::now();\n                    },\n''',
    )
    text = must_replace(
        text,
        '''        if cfg.save_interval > 0 && step > cfg.start_step && !step.is_multiple_of(cfg.save_interval)\n        {\n''',
        '''        if checkpointing_enabled\n            && step > cfg.start_step\n            && last_checkpoint_step != Some(step)\n        {\n''',
    )
    return text


def patch_example(text: str) -> str:
    if "SCIAGENT_SAVE_HOURS" in text:
        print("cuda_pretrain already B41 patched")
        return text

    text = must_replace(
        text,
        '''//! - `SCIAGENT_CKPT` (default `checkpoints/cuda`), `SCIAGENT_STEPS` (300),\n//!   `SCIAGENT_SEQ` (128), `SCIAGENT_BATCH` (1), `SCIAGENT_TELEMETRY` (25),\n//!   `SCIAGENT_LR` — run knobs.\n''',
        '''//! - `SCIAGENT_CKPT` (default `checkpoints/cuda`), `SCIAGENT_STEPS` (300),\n//!   `SCIAGENT_SEQ`, `SCIAGENT_BATCH` (1), `SCIAGENT_TELEMETRY` (25), `SCIAGENT_LR`.\n//! - `SCIAGENT_SAVE=<steps>` explicitly selects a step cadence. When unset, exact\n//!   recovery checkpoints default to `SCIAGENT_SAVE_HOURS=6` wall-clock hours.\n''',
    )
    text = must_replace(
        text,
        '''    // Checkpoint/telemetry cadence does not alter model math.\n    let save_interval = env_usize("SCIAGENT_SAVE", 500);\n    let keep_last = env_usize("SCIAGENT_KEEP", 2);\n''',
        '''    // Exact recovery points contain model fp32 weights plus AdamW m/v. At the\n    // ~304M production shape, a fixed 500-step default can write hundreds of GB per\n    // corpus pass. Prefer a wall-clock cadence; explicit SCIAGENT_SAVE=<steps> opts\n    // back into a step cadence. SCIAGENT_SAVE_HOURS=0 disables periodic saves.\n    let explicit_save_steps = std::env::var("SCIAGENT_SAVE")\n        .ok()\n        .and_then(|v| v.parse::<usize>().ok());\n    let save_interval = explicit_save_steps.unwrap_or(0);\n    let save_interval_seconds = if explicit_save_steps.is_some()\n    {\n        0\n    }\n    else\n    {\n        let hours = std::env::var("SCIAGENT_SAVE_HOURS")\n            .ok()\n            .and_then(|v| v.parse::<f64>().ok())\n            .unwrap_or(6.0)\n            .max(0.0);\n        (hours * 3600.0).round().min(u64::MAX as f64) as u64\n    };\n    let keep_last = env_usize("SCIAGENT_KEEP", 2);\n''',
    )
    text = must_replace(
        text,
        '''        log_interval: 25,\n        save_interval,\n        checkpoint_dir: ckpt_dir.clone(),\n''',
        '''        log_interval: 25,\n        save_interval,\n        save_interval_seconds,\n        checkpoint_dir: ckpt_dir.clone(),\n''',
    )
    text = must_replace(
        text,
        '''         eps {adam_eps:.0e} | clip {max_grad_norm} | save/{save_interval} keep {keep_last} | \\\n         shuffle {shuffle} | ckpt → {ckpt_dir}\\n"\n''',
        '''         eps {adam_eps:.0e} | clip {max_grad_norm} | save_steps/{save_interval} save_seconds/{save_interval_seconds} keep {keep_last} | \\\n         shuffle {shuffle} | ckpt → {ckpt_dir}\\n"\n''',
    )
    return text


MODEL.write_text(patch_model(MODEL.read_text()))
EXAMPLE.write_text(patch_example(EXAMPLE.read_text()))
print("B41 patched: wall-clock recovery checkpoint cadence")
