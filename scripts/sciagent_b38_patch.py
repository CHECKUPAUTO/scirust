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


BEST_HELPERS = r'''
fn load_best_validation(dir: &str) -> Option<(usize, f32)> {
    let path = Path::new(dir).join("best").join("selection.json");
    let raw = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let step = value["step"].as_u64()? as usize;
    let val = value["val_loss"].as_f64()? as f32;
    val.is_finite().then_some((step, val))
}

/// Publish a model-only best-validation snapshot. Exact resume checkpoints keep
/// AdamW state in `step_N/`; the historical best does not need another ~2.4 GB of
/// moments. `best/` is deliberately ignored by `latest_checkpoint`.
fn save_best_validation_model(
    model: &SciAgentModel,
    meta: &CheckpointMeta,
    val_loss: f32,
    root: &str,
) -> std::result::Result<(), String> {
    let root = Path::new(root);
    std::fs::create_dir_all(root)
        .map_err(|e| format!("cannot create checkpoint root {}: {e}", root.display()))?;
    let partial = root.join(".best.partial");
    let old = root.join(".best.old");
    let best = root.join("best");
    if partial.exists() {
        std::fs::remove_dir_all(&partial)
            .map_err(|e| format!("cannot remove stale {}: {e}", partial.display()))?;
    }
    if old.exists() {
        std::fs::remove_dir_all(&old)
            .map_err(|e| format!("cannot remove stale {}: {e}", old.display()))?;
    }
    save_checkpoint(model, meta, &partial)
        .map_err(|e| format!("cannot save best model: {e}"))?;
    let selection = serde_json::json!({
        "version": 1,
        "step": meta.step,
        "val_loss": val_loss,
    });
    std::fs::write(
        partial.join("selection.json"),
        serde_json::to_string_pretty(&selection)
            .map_err(|e| format!("cannot encode best selection: {e}"))?,
    )
    .map_err(|e| format!("cannot write best selection: {e}"))?;

    if best.exists() {
        std::fs::rename(&best, &old)
            .map_err(|e| format!("cannot rotate previous best {}: {e}", best.display()))?;
    }
    if let Err(e) = std::fs::rename(&partial, &best) {
        if old.exists() {
            let _ = std::fs::rename(&old, &best);
        }
        return Err(format!("cannot publish best {}: {e}", best.display()));
    }
    if old.exists() {
        let _ = std::fs::remove_dir_all(old);
    }
    Ok(())
}

'''


def patch_model(text: str) -> str:
    if "fn load_best_validation" in text:
        raise SystemExit("model already B38 patched")
    # Insert best helpers before shuffle helper after impl block.
    marker = "/// Deterministic in-place Fisher–Yates shuffle of `order`"
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing shuffle helper marker")
    text = text[:pos] + BEST_HELPERS + text[pos:]

    text = must_replace(
        text,
        '''        let mut best_val = f32::INFINITY;\n        let mut best_step: Option<usize> = None;\n''',
        '''        let (mut best_step, mut best_val) = load_best_validation(&cfg.checkpoint_dir)\n            .map(|(step, val)| (Some(step), val))\n            .unwrap_or((None, f32::INFINITY));\n        if let Some(saved_best) = best_step\n        {\n            println!("restored best validation: {best_val:.4} @ step {saved_best}");\n        }\n        let mut last_eval: Option<(usize, f32)> = None;\n''',
        1,
    )
    text = must_replace(
        text,
        '''            if need_eval\n            {\n                let val = self.eval_loss_windows(tokens, s, &val_windows, cfg.eval_windows);\n                println!("            └─ held-out val loss {val:>9.4}");\n            }\n''',
        '''            if need_eval\n            {\n                let val = self.eval_loss_windows(tokens, s, &val_windows, cfg.eval_windows);\n                last_eval = Some((step, val));\n                println!("            └─ held-out val loss {val:>9.4}");\n            }\n''',
        1,
    )
    old_save_body = '''                    Ok(()) =>\n                    {\n                        println!("  checkpoint → {}", dir.display());\n                        if !val_windows.is_empty()\n                        {\n                            let v =\n                                self.eval_loss_windows(tokens, s, &val_windows, cfg.eval_windows);\n                            if v < best_val\n                            {\n                                best_val = v;\n                                best_step = Some(step);\n                                println!("    (best val {v:.4} @ step {step} → protected)");\n                            }\n                        }\n                        prune_checkpoints(&cfg.checkpoint_dir, cfg.keep_last, best_step);\n                    },\n'''
    new_save_body = '''                    Ok(()) =>\n                    {\n                        println!("  exact checkpoint → {}", dir.display());\n                        if !val_windows.is_empty()\n                        {\n                            let v = match last_eval\n                            {\n                                Some((eval_step, v)) if eval_step == step => v,\n                                _ =>\n                                {\n                                    let v = self.eval_loss_windows(\n                                        tokens, s, &val_windows, cfg.eval_windows,\n                                    );\n                                    last_eval = Some((step, v));\n                                    v\n                                },\n                            };\n                            if v < best_val\n                            {\n                                best_val = v;\n                                best_step = Some(step);\n                                match save_best_validation_model(model, &meta, v, &cfg.checkpoint_dir)\n                                {\n                                    Ok(()) => println!(\n                                        "    best model-only → {}/best (val {v:.4} @ step {step})",\n                                        cfg.checkpoint_dir\n                                    ),\n                                    Err(e) => eprintln!("    best-model save failed: {e}"),\n                                }\n                            }\n                        }\n                        prune_checkpoints(&cfg.checkpoint_dir, cfg.keep_last);\n                    },\n'''
    text = must_replace(text, old_save_body, new_save_body, 1)

    old_tail = '''        // Defensive flush for any non-standard early exit. Normal total-step exit\n        // already flushes above, so the returned vector remains exactly per-step.\n        for diag in pending.drain(..)\n        {\n            losses.push(self.finish_step_diagnostics(diag));\n        }\n        losses\n'''
    new_tail = '''        // Defensive flush for any non-standard early exit. Normal total-step exit\n        // already flushes above, so the returned vector remains exactly per-step.\n        for diag in pending.drain(..)\n        {\n            last_loss = self.finish_step_diagnostics(diag);\n            losses.push(last_loss);\n        }\n\n        // A run target is an exact recovery boundary even when it falls between the\n        // periodic save cadence. Historically the example only synced host weights\n        // here and falsely claimed a final checkpoint existed.\n        if cfg.save_interval > 0\n            && step > cfg.start_step\n            && !step.is_multiple_of(cfg.save_interval)\n        {\n            self.sync_to_model(model);\n            let lr = schedule.lr_at(step.saturating_sub(1));\n            let dir = Path::new(&cfg.checkpoint_dir).join(format!("step_{step}"));\n            let meta = CheckpointMeta {\n                step,\n                loss: last_loss,\n                lr,\n                config: config.clone(),\n            };\n            match self.save_training_checkpoint(model, &meta, cfg, &dir)\n            {\n                Ok(()) =>\n                {\n                    println!("  final exact checkpoint → {}", dir.display());\n                    if !val_windows.is_empty()\n                    {\n                        let v = match last_eval\n                        {\n                            Some((eval_step, v)) if eval_step == step => v,\n                            _ => self.eval_loss_windows(tokens, s, &val_windows, cfg.eval_windows),\n                        };\n                        if v < best_val\n                        {\n                            best_val = v;\n                            best_step = Some(step);\n                            if let Err(e) =\n                                save_best_validation_model(model, &meta, v, &cfg.checkpoint_dir)\n                            {\n                                eprintln!("    final best-model save failed: {e}");\n                            }\n                        }\n                    }\n                    prune_checkpoints(&cfg.checkpoint_dir, cfg.keep_last);\n                },\n                Err(e) => eprintln!("  final checkpoint at step {step} failed: {e}"),\n            }\n        }\n        let _ = best_step; // retained for diagnostics/documentation; best/ is model-only.\n        let _ = best_val;\n        losses\n'''
    text = must_replace(text, old_tail, new_tail, 1)

    old_prune = '''/// Delete old `step_N/` checkpoints under `dir`, keeping only the most recent\n/// `keep_last` (by numeric step) plus `protect` (the best-val step, if any).\n/// `keep_last == 0` disables pruning (keep everything). Best-effort — I/O errors on\n/// individual removals are ignored so a failed delete never aborts training.\nfn prune_checkpoints(dir: &str, keep_last: usize, protect: Option<usize>) {\n'''
    new_prune = '''/// Delete old exact-resume `step_N/` checkpoints, keeping only the most recent\n/// `keep_last`. Historical best validation lives separately in model-only `best/`,\n/// so it no longer pins another multi-GB AdamW sidecar.\nfn prune_checkpoints(dir: &str, keep_last: usize) {\n'''
    text = must_replace(text, old_prune, new_prune, 1)
    text = must_replace(
        text,
        '''        if protect == Some(*n as usize)\n        {\n            continue; // the best-val checkpoint\n        }\n''',
        '''        let _ = n;\n''',
        1,
    )
    text = must_replace(
        text,
        '''    /// Fraction of the token stream held out (from the tail) for validation\n    /// (`0.0` disables held-out eval). Default `0.02`.\n''',
        '''    /// Fraction of deterministic distributed windows held out for validation\n    /// (`0.0` disables held-out eval). Default `0.02`.\n''',
        1,
    )
    text = must_replace(
        text,
        '''    /// Checkpoint retention: keep only the most recent `keep_last` `step_N/` dirs\n    /// (plus the best-val one, which is never pruned). `0` keeps everything — the old\n    /// behavior, which fills the disk on a long run. Default `3`.\n''',
        '''    /// Exact-resume checkpoint retention. Historical best validation is stored\n    /// separately as model-only `best/`; `0` keeps every exact checkpoint. Default `2`.\n''',
        1,
    )
    text = must_replace(text, "            keep_last: 3,\n", "            keep_last: 2,\n", 1)
    return text


def patch_example(text: str) -> str:
    if "exact checkpoint estimate" in text:
        raise SystemExit("example already B38 patched")
    text = must_replace(
        text,
        '''    println!(\n        "resident VRAM estimate: fp32 master ~{weight_mb:.0} MB + bf16 view ~{bf16_mb:.0} MB + \\\n         AdamW state ~{opt_mb:.0} MB (activations extra)\\n"\n    );\n''',
        '''    println!(\n        "resident VRAM estimate: fp32 master ~{weight_mb:.0} MB + bf16 view ~{bf16_mb:.0} MB + \\\n         AdamW state ~{opt_mb:.0} MB (activations extra)"\n    );\n    println!(\n        "exact checkpoint estimate: model + AdamW m/v ~{:.1} GB; best/ is model-only ~{:.1} GB\\n",\n        (weight_mb + opt_mb) / 1000.0,\n        weight_mb / 1000.0\n    );\n''',
        1,
    )
    text = must_replace(text, "    let keep_last = env_usize(\"SCIAGENT_KEEP\", 3);\n", "    let keep_last = env_usize(\"SCIAGENT_KEEP\", 2);\n", 1)
    text = must_replace(
        text,
        '''    // Final sync + checkpoint so the last weights are always persisted.\n    trainer.sync_to_model(&mut model);\n    println!("trained fp32 masters synced back into the SciAgentModel; resume from {ckpt_dir}.");\n''',
        '''    // `pretrain` now guarantees a final exact `step_N/` recovery checkpoint when\n    // checkpointing is enabled, even if the target falls between periodic saves.\n    trainer.sync_to_model(&mut model);\n    println!("trained fp32 masters synced; exact resume checkpoints are under {ckpt_dir}/step_N.");\n''',
        1,
    )
    return text


MODEL.write_text(patch_model(MODEL.read_text()))
EXAMPLE.write_text(patch_example(EXAMPLE.read_text()))
print("patched B38 checkpoint lifecycle")
