#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"
EXAMPLE = ROOT / "scirust-sciagent/examples/cuda_pretrain.rs"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:120]!r}")
    return text.replace(old, new, count)


def replace_fn(text: str, marker: str, new_src: str) -> str:
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"missing {marker!r}")
    brace = text.find("{", start)
    depth = 0
    in_str = False
    esc = False
    for i in range(brace, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[:start] + new_src.rstrip() + text[i + 1 :]
    raise SystemExit(f"unterminated {marker}")


DIAG_STRUCT = r'''
/// Resident diagnostics for one optimizer step. Keeping these tiny buffers alive
/// lets pretraining enqueue several complete steps before any host synchronization.
struct CudaStepDiagnostics {
    loss_rows: CudaF32,
    grad_sumsq: CudaF32,
}
'''

TRAIN_BATCH = r'''    pub fn train_step_batch(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        batch_size: usize,
        seq_len: usize,
        lr: f32,
        betas: (f32, f32),
        adam_eps: f32,
        weight_decay: f32,
    ) -> f32 {
        let diag = self.train_step_batch_deferred(
            tokens,
            targets,
            batch_size,
            seq_len,
            lr,
            betas,
            adam_eps,
            weight_decay,
        );
        self.finish_step_diagnostics(diag)
    }

    /// Enqueue a complete optimizer step without reading diagnostics back to the host.
    /// Public one-step APIs stay synchronous; the production pretrainer batches these
    /// diagnostics so loss/gnorm telemetry no longer inserts a barrier every step.
    #[allow(clippy::too_many_arguments)]
    fn train_step_batch_deferred(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        batch_size: usize,
        seq_len: usize,
        lr: f32,
        betas: (f32, f32),
        adam_eps: f32,
        weight_decay: f32,
    ) -> CudaStepDiagnostics {
        assert!(batch_size > 0 && seq_len > 0, "train_step_batch: empty batch");
        assert_eq!(tokens.len(), batch_size * seq_len, "train_step_batch: token shape");
        assert_eq!(targets.len(), tokens.len(), "train_step_batch: target shape");
        self.step += 1;
        let (logits, cache) = self.model.forward_train(tokens, batch_size, seq_len);
        let (loss_rows, dlogits) = self
            .model
            .chain
            .cross_entropy_loss_grad_resident(&logits, targets);
        let grads = self.model.backward_cached(tokens, &dlogits, &cache);

        let mut grad_refs: Vec<&CudaMatrix> = vec![&grads.d_embedding, &grads.d_final_norm];
        for bg in &grads.blocks {
            grad_refs.extend([
                &bg.dnorm1, &bg.dwq, &bg.dwk, &bg.dwv, &bg.dwo,
                &bg.dnorm2, &bg.dwg, &bg.dwu, &bg.dwd,
            ]);
        }
        let ch = &self.model.chain;
        let grad_sumsq = ch.global_grad_sumsq(&grad_refs);
        drop(grad_refs);
        let step = self.step;
        let max_norm = self.max_grad_norm;

        ch.adamw_step_with_norm(
            &mut self.master_embedding, &mut self.m_embedding, &mut self.v_embedding,
            &grads.d_embedding, &mut self.model.embedding, lr, betas, adam_eps,
            weight_decay, step, &grad_sumsq, max_norm,
        );
        ch.adamw_step_with_norm(
            &mut self.master_final_norm, &mut self.m_final_norm, &mut self.v_final_norm,
            &grads.d_final_norm, &mut self.model.final_norm, lr, betas, adam_eps,
            weight_decay, step, &grad_sumsq, max_norm,
        );
        for i in 0..self.model.blocks.len() {
            let bg = &grads.blocks[i];
            let (mb, mm, mv) = (
                &mut self.master_blocks[i], &mut self.m_blocks[i], &mut self.v_blocks[i],
            );
            let b = &mut self.model.blocks[i];
            let one = |master: &mut CudaF32,
                       mo: &mut CudaF32,
                       vo: &mut CudaF32,
                       grad: &CudaMatrix,
                       view: &mut CudaMatrix| {
                ch.adamw_step_with_norm(
                    master, mo, vo, grad, view, lr, betas, adam_eps, weight_decay,
                    step, &grad_sumsq, max_norm,
                );
            };
            one(&mut mb.norm1, &mut mm.norm1, &mut mv.norm1, &bg.dnorm1, &mut b.norm1);
            one(&mut mb.wq, &mut mm.wq, &mut mv.wq, &bg.dwq, &mut b.wq);
            one(&mut mb.wk, &mut mm.wk, &mut mv.wk, &bg.dwk, &mut b.wk);
            one(&mut mb.wv, &mut mm.wv, &mut mv.wv, &bg.dwv, &mut b.wv);
            one(&mut mb.wo, &mut mm.wo, &mut mv.wo, &bg.dwo, &mut b.wo);
            one(&mut mb.norm2, &mut mm.norm2, &mut mv.norm2, &bg.dnorm2, &mut b.norm2);
            one(&mut mb.wg, &mut mm.wg, &mut mv.wg, &bg.dwg, &mut b.wg);
            one(&mut mb.wu, &mut mm.wu, &mut mv.wu, &bg.dwu, &mut b.wu);
            one(&mut mb.wd, &mut mm.wd, &mut mv.wd, &bg.dwd, &mut b.wd);
        }
        CudaStepDiagnostics { loss_rows, grad_sumsq }
    }

    fn finish_step_diagnostics(&mut self, diag: CudaStepDiagnostics) -> f32 {
        let loss = self.model.chain.mean_f32(&diag.loss_rows);
        self.last_grad_norm = self.model.chain.grad_norm_from_sumsq(&diag.grad_sumsq);
        loss
    }'''

PRETRAIN = r'''    pub fn pretrain(
        &mut self,
        tokens: &[u32],
        model: &mut SciAgentModel,
        config: &SciAgentConfig,
        cfg: &CudaPretrainConfig,
    ) -> Vec<f32> {
        let s = cfg.seq_len;
        let batch = cfg.batch_size.max(1);
        let telemetry = cfg.telemetry_interval.max(1);
        let mut losses = Vec::new();
        if tokens.len() <= s {
            eprintln!(
                "cuda pretrain: token stream ({}) shorter than a single window ({}); nothing to do",
                tokens.len(), s + 1
            );
            return losses;
        }
        self.max_grad_norm = cfg.max_grad_norm;

        let val_len = ((tokens.len() as f32 * cfg.val_frac.max(0.0)) as usize)
            .min(tokens.len().saturating_sub(s + 1));
        let (train_tokens, val_tokens): (&[u32], &[u32]) = if val_len > s + 1 {
            let cut = tokens.len() - val_len;
            (&tokens[..cut], &tokens[cut..])
        } else {
            (tokens, &[])
        };
        if !val_tokens.is_empty() {
            println!(
                "held-out validation: {} tokens ({:.0}% tail)\n",
                val_tokens.len(), cfg.val_frac * 100.0
            );
        }

        let schedule =
            WarmupCosineSchedule::new(cfg.base_lr, cfg.min_lr, cfg.warmup_steps, cfg.total_steps);
        let mut step = cfg.start_step;
        let n_windows = train_tokens.len().saturating_sub(1) / s;
        let mut order: Vec<usize> = (0..n_windows).collect();
        let mut epoch: u64 = 0;
        let reshuffle = |order: &mut [usize], epoch: u64| {
            shuffle_windows(
                order,
                (cfg.start_step as u64).wrapping_add(epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
        };
        if cfg.shuffle { reshuffle(&mut order, epoch); }
        let mut wi = 0usize;
        let mut best_val = f32::INFINITY;
        let mut best_step: Option<usize> = None;
        let t0 = std::time::Instant::now();
        let mut packed_inputs = Vec::with_capacity(batch * s);
        let mut packed_targets = Vec::with_capacity(batch * s);
        let mut pending: Vec<CudaStepDiagnostics> = Vec::with_capacity(telemetry);
        let mut last_loss = f32::NAN;

        while step < cfg.total_steps && n_windows > 0 {
            packed_inputs.clear();
            packed_targets.clear();
            for _ in 0..batch {
                if wi >= order.len() {
                    epoch += 1;
                    if cfg.shuffle { reshuffle(&mut order, epoch); }
                    wi = 0;
                }
                let start = order[wi] * s;
                wi += 1;
                packed_inputs.extend_from_slice(&train_tokens[start..start + s]);
                packed_targets.extend_from_slice(&train_tokens[start + 1..start + s + 1]);
            }
            let lr = schedule.lr_at(step);
            pending.push(self.train_step_batch_deferred(
                &packed_inputs,
                &packed_targets,
                batch,
                s,
                lr,
                cfg.betas,
                cfg.adam_eps,
                cfg.weight_decay,
            ));
            step += 1;

            let need_log = cfg.log_interval > 0 && step.is_multiple_of(cfg.log_interval);
            let need_eval = cfg.eval_interval > 0
                && !val_tokens.is_empty()
                && step.is_multiple_of(cfg.eval_interval);
            let need_save = cfg.save_interval > 0 && step.is_multiple_of(cfg.save_interval);
            let need_flush = pending.len() >= telemetry
                || need_log
                || need_eval
                || need_save
                || step == cfg.total_steps;

            if need_flush {
                for diag in pending.drain(..) {
                    last_loss = self.finish_step_diagnostics(diag);
                    losses.push(last_loss);
                }
            }

            if need_log {
                let done = (step - cfg.start_step) * s * batch;
                let secs = t0.elapsed().as_secs_f64().max(1e-9);
                let tps = done as f64 / secs;
                let gnorm = self.last_grad_norm();
                println!(
                    "[cuda step {step:>6}] B{batch}×T{s} loss {last_loss:>9.4} | lr {lr:.3e} | gnorm {gnorm:>7.2} | {tps:>8.0} tok/s"
                );
            }
            if need_eval {
                let val = self.eval_loss(val_tokens, s, cfg.eval_windows);
                println!("            └─ held-out val loss {val:>9.4}");
            }
            if need_save {
                self.sync_to_model(model);
                let dir = std::path::Path::new(&cfg.checkpoint_dir).join(format!("step_{step}"));
                let meta = CheckpointMeta { step, loss: last_loss, lr, config: config.clone() };
                match save_checkpoint(model, &meta, &dir) {
                    Ok(()) => {
                        println!("  checkpoint → {}", dir.display());
                        if !val_tokens.is_empty() {
                            let v = self.eval_loss(val_tokens, s, cfg.eval_windows);
                            if v < best_val {
                                best_val = v;
                                best_step = Some(step);
                                println!("    (best val {v:.4} @ step {step} → protected)");
                            }
                        }
                        prune_checkpoints(&cfg.checkpoint_dir, cfg.keep_last, best_step);
                    }
                    Err(e) => eprintln!("  checkpoint at step {step} failed: {e}"),
                }
            }
        }

        // Defensive flush for any non-standard early exit. Normal total-step exit
        // already flushes above, so the returned vector remains exactly per-step.
        for diag in pending.drain(..) {
            losses.push(self.finish_step_diagnostics(diag));
        }
        losses
    }'''


def patch_model(text: str) -> str:
    if "CudaStepDiagnostics" in text:
        raise SystemExit("already B30 patched")
    anchor = "/// A trainable [`CudaModel`]: the bf16 model plus **fp32 master** copies (or AdamW moments) —"
    # Actual source has BlockMasters before CudaTrainer; insert just before CudaTrainer docs.
    marker = "/// A trainable [`CudaModel`]: the bf16 model plus **fp32 master weights and AdamW\n"
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing CudaTrainer doc marker")
    text = text[:pos] + DIAG_STRUCT + "\n" + text[pos:]
    text = replace_fn(text, "    pub fn train_step_batch(\n", TRAIN_BATCH)
    text = replace_fn(text, "    pub fn pretrain(\n", PRETRAIN)
    text = must_replace(
        text,
        "    /// Print a loss/lr line every this many steps (0 = never).\n    pub log_interval: usize,\n",
        "    /// Maximum number of optimizer steps queued before exact loss/gnorm telemetry\n    /// is copied to the host. Lower values improve observability; higher values reduce\n    /// synchronization frequency. Logging/eval/checkpoint boundaries always flush.\n    pub telemetry_interval: usize,\n    /// Print a loss/lr line every this many steps (0 = never).\n    pub log_interval: usize,\n",
    )
    text = must_replace(
        text,
        "            batch_size: 1,\n            log_interval: 100,\n",
        "            batch_size: 1,\n            telemetry_interval: 25,\n            log_interval: 100,\n",
    )
    return text


def patch_example(text: str) -> str:
    text = must_replace(
        text,
        "//!   `SCIAGENT_SEQ` (128), `SCIAGENT_BATCH` (1), `SCIAGENT_LR` — run knobs.\n",
        "//!   `SCIAGENT_SEQ` (128), `SCIAGENT_BATCH` (1), `SCIAGENT_TELEMETRY` (25),\n//!   `SCIAGENT_LR` — run knobs.\n",
    )
    text = must_replace(
        text,
        "    let batch_size = env_usize(\"SCIAGENT_BATCH\", 1).max(1);\n",
        "    let batch_size = env_usize(\"SCIAGENT_BATCH\", 1).max(1);\n    let telemetry_interval = env_usize(\"SCIAGENT_TELEMETRY\", 25).max(1);\n",
    )
    text = must_replace(
        text,
        "        batch_size,\n        weight_decay: 0.0,\n",
        "        batch_size,\n        telemetry_interval,\n        weight_decay: 0.0,\n",
    )
    text = must_replace(
        text,
        '        "batch {batch_size} × seq_len {seq_len} | steps {start_step}..{total_steps} | base_lr {base_lr:.1e} | \\\n',
        '        "batch {batch_size} × seq_len {seq_len} | telemetry/{telemetry_interval} | steps {start_step}..{total_steps} | base_lr {base_lr:.1e} | \\\n',
    )
    return text


MODEL.write_text(patch_model(MODEL.read_text()))
EXAMPLE.write_text(patch_example(EXAMPLE.read_text()))
print("patched B30 deferred telemetry")
