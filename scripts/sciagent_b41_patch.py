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
    if "save_interval_seconds" in text:
        raise SystemExit("model already B41 patched")

    text = must_replace(
        text,
        '''    /// Write a checkpoint every this many steps (0 = never).\n    pub save_interval: usize,\n''',
        '''    /// Write an exact recovery checkpoint every this many optimizer steps\n    /// (`0` disables the step cadence).\n    pub save_interval: usize,\n    /// Optional wall-clock recovery cadence in seconds (`0` disables it). This is\n    /// operational only: checkpoints are still published on completed step boundaries.\n    pub save_interval_seconds: u64,\n''',
        1,
    )
    text = must_replace(
        text,
        '''            save_interval: 500,\n            checkpoint_dir: "checkpoints/cuda".into(),\n''',
        '''            save_interval: 500,\n            save_interval_seconds: 0,\n            checkpoint_dir: "checkpoints/cuda".into(),\n''',
        1,
    )

    text = must_replace(
        text,
        '''        let t0 = std::time::Instant::now();\n        let mut packed_inputs = Vec::with_capacity(batch * s);\n''',
        '''        let t0 = std::time::Instant::now();\n        let mut last_checkpoint_at = t0;\n        let mut last_checkpoint_step: Option<usize> = None;\n        let checkpointing_enabled = cfg.save_interval > 0 || cfg.save_interval_seconds > 0;\n        let mut packed_inputs = Vec::with_capacity(batch * s);\n''',
        1,
    )

    old_need = '''            let need_save = cfg.save_interval > 0 && step.is_multiple_of(cfg.save_interval);\n'''
    new_need = '''            let need_save_by_step =\n                cfg.save_interval > 0 && step.is_multiple_of(cfg.save_interval);\n            let need_save_by_time = cfg.save_interval_seconds > 0\n                && last_checkpoint_at.elapsed().as_secs() >= cfg.save_interval_seconds;\n            let need_save = need_save_by_step || need_save_by_time;\n'''
    text = must_replace(text, old_need, new_need, 1)

    # Record the successful exact checkpoint and reset wall-clock cadence after any
    # save attempt, so a transient disk error cannot hammer the filesystem every step.
    old_ok = '''                        println!("  exact checkpoint → {}", dir.display());\n                        if !val_windows.is_empty()\n'''
    new_ok = '''                        println!("  exact checkpoint → {}", dir.display());\n                        last_checkpoint_step = Some(step);\n                        last_checkpoint_at = std::time::Instant::now();\n                        if !val_windows.is_empty()\n'''
    text = must_replace(text, old_ok, new_ok, 1)
    old_err = '''                    Err(e) => eprintln!("  checkpoint at step {step} failed: {e}"),\n'''
    new_err = '''                    Err(e) =>\n                    {\n                        eprintln!("  checkpoint at step {step} failed: {e}");\n                        last_checkpoint_at = std::time::Instant::now();\n                    },\n'''
    text = must_replace(text, old_err, new_err, 1)

    old_final_cond = '''        if cfg.save_interval > 0\n            && step > cfg.start_step\n            && !step.is_multiple_of(cfg.save_interval)\n        {\n'''
    new_final_cond = '''        if checkpointing_enabled\n            && step > cfg.start_step\n            && last_checkpoint_step != Some(step)\n        {\n'''
    text = must_replace(text, old_final_cond, new_final_cond, 1)
    return text


def patch_example(text: str) -> str:
    if "SCIAGENT_SAVE_HOURS" in text:
        raise SystemExit("example already B41 patched")
    text = must_replace(
        text,
        '''//! - `SCIAGENT_CKPT` (default `checkpoints/cuda`), `SCIAGENT_STEPS` (300),\n//!   `SCIAGENT_SEQ` (128), `SCIAGENT_BATCH` (1), `SCIAGENT_TELEMETRY` (25),\n//!   `SCIAGENT_LR` — run knobs.\n''',
        '''//! - `SCIAGENT_CKPT` (default `checkpoints/cuda`), `SCIAGENT_STEPS` (300),\n//!   `SCIAGENT_SEQ`, `SCIAGENT_BATCH` (1), `SCIAGENT_TELEMETRY` (25), `SCIAGENT_LR`.\n//! - `SCIAGENT_SAVE=<steps>` explicitly selects a step cadence. When unset, production\n//!   recovery checkpoints default to `SCIAGENT_SAVE_HOURS=6` wall-clock hours.\n''',
        1,
    )

    old = '''    // Checkpoint cadence + retention. Saving every 100 steps and never pruning fills\n    // the disk on a long run (each 350M checkpoint is ~1.2 GB fp32) — so save less\n    // often and keep only the last few (SCIAGENT_KEEP) plus the best-val one.\n    let save_interval = env_usize("SCIAGENT_SAVE", 500);\n    let keep_last = env_usize("SCIAGENT_KEEP", 2);\n'''
    new = '''    // Exact B32+ recovery points include model fp32 weights plus AdamW m/v. At ~304M\n    // parameters a step-based default would generate terabytes of writes over a long\n    // corpus pass, especially after true batching. Production therefore defaults to\n    // a six-hour wall-clock cadence; SCIAGENT_SAVE=<steps> explicitly opts back into\n    // a step cadence, while SCIAGENT_SAVE_HOURS=0 disables periodic recovery saves.\n    let explicit_save_steps = std::env::var("SCIAGENT_SAVE")\n        .ok()\n        .and_then(|v| v.parse::<usize>().ok());\n    let save_interval = explicit_save_steps.unwrap_or(0);\n    let save_interval_seconds = if explicit_save_steps.is_some()\n    {\n        0\n    }\n    else\n    {\n        let hours = std::env::var("SCIAGENT_SAVE_HOURS")\n            .ok()\n            .and_then(|v| v.parse::<f64>().ok())\n            .unwrap_or(6.0)\n            .max(0.0);\n        (hours * 3600.0).round().min(u64::MAX as f64) as u64\n    };\n    let keep_last = env_usize("SCIAGENT_KEEP", 2);\n'''
    text = must_replace(text, old, new, 1)
    text = must_replace(
        text,
        '''        save_interval,\n        checkpoint_dir: ckpt_dir.clone(),\n''',
        '''        save_interval,\n        save_interval_seconds,\n        checkpoint_dir: ckpt_dir.clone(),\n''',
        1,
    )
    old_print = '''        "batch {batch_size} × seq_len {seq_len} | telemetry/{telemetry_interval} | steps {start_step}..{total_steps} | base_lr {base_lr:.1e} | \\\n         eps {adam_eps:.0e} | clip {max_grad_norm} | save/{save_interval} keep {keep_last} | \\\n         shuffle {shuffle} | ckpt → {ckpt_dir}\\n"\n'''
    new_print = '''        "batch {batch_size} × seq_len {seq_len} | telemetry/{telemetry_interval} | steps {start_step}..{total_steps} | base_lr {base_lr:.1e} | \\\n         eps {adam_eps:.0e} | clip {max_grad_norm} | save_steps/{save_interval} save_seconds/{save_interval_seconds} keep {keep_last} | \\\n         shuffle {shuffle} | ckpt → {ckpt_dir}\\n"\n'''
    text = must_replace(text, old_print, new_print, 1)
    return text


MODEL.write_text(patch_model(MODEL.read_text()))
EXAMPLE.write_text(patch_example(EXAMPLE.read_text()))
print("B41 patched: wall-clock recovery checkpoint cadence")
