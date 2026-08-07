#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "scirust-cuda/src/chain.rs"
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
        raise SystemExit(f"missing marker {marker!r}")
    brace = text.find("{", start)
    if brace < 0:
        raise SystemExit("missing opening brace")
    depth = 0
    in_str = False
    escaped = False
    i = brace
    while i < len(text):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
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
        i += 1
    raise SystemExit(f"unterminated function {marker}")


ROW_KERNELS = r'''
// Row slicing/placement for true B×T training. Projection/MLP/head GEMMs operate
// on packed B*T rows, while attention slices each sequence back to T rows so the
// causal mask can never leak information across samples.
extern "C" __global__ void slice_rows_kernel(
    unsigned short* out, const unsigned short* x,
    const size_t cols, const size_t row_start, const size_t nrows)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nrows * cols) {
        const size_t r = idx / cols, c = idx % cols;
        out[idx] = x[(row_start + r) * cols + c];
    }
}

extern "C" __global__ void place_rows_kernel(
    unsigned short* out, const unsigned short* x,
    const size_t cols, const size_t row_start, const size_t nrows)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nrows * cols) {
        const size_t r = idx / cols, c = idx % cols;
        out[(row_start + r) * cols + c] = x[idx];
    }
}
'''


def patch_chain(text: str) -> str:
    if "slice_rows_kernel" in text:
        raise SystemExit("chain already B28 patched")
    text = must_replace(
        text,
        "// Scatter a narrow block into a zero-padded wide matrix at col_start.\n",
        ROW_KERNELS + "\n// Scatter a narrow block into a zero-padded wide matrix at col_start.\n",
    )
    text = must_replace(
        text,
        "    slice_cols: CudaFunction,\n    place_cols: CudaFunction,\n",
        "    slice_cols: CudaFunction,\n    place_cols: CudaFunction,\n    slice_rows: CudaFunction,\n    place_rows: CudaFunction,\n",
    )
    text = must_replace(
        text,
        '            slice_cols: f("slice_cols_kernel"),\n            place_cols: f("place_cols_kernel"),\n',
        '            slice_cols: f("slice_cols_kernel"),\n            place_cols: f("place_cols_kernel"),\n            slice_rows: f("slice_rows_kernel"),\n            place_rows: f("place_rows_kernel"),\n',
    )
    marker = "    /// Scatter a `rows × ncols` block into a zero-padded `rows × dst_cols` matrix\n"
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing place_cols insertion point")
    methods = r'''    /// Copy a contiguous row range into a new resident matrix.
    pub fn slice_rows(&self, x: &CudaMatrix, row_start: usize, nrows: usize) -> CudaMatrix {
        assert!(row_start + nrows <= x.rows, "slice_rows: range out of bounds");
        let cols = x.cols;
        let total = nrows * cols;
        let mut out = self.stream.alloc_zeros::<bf16>(total).expect("cuda alloc rows");
        let (cols_a, start_a, nrows_a) = (cols, row_start, nrows);
        let mut builder = self.stream.launch_builder(&self.kernels().slice_rows);
        builder.arg(&mut out);
        builder.arg(&x.buf);
        builder.arg(&cols_a);
        builder.arg(&start_a);
        builder.arg(&nrows_a);
        // SAFETY: host range assertion guarantees all source rows are valid.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .expect("launch slice_rows_kernel");
        }
        CudaMatrix { buf: out, rows: nrows, cols }
    }

    /// Concatenate equal-width matrices along rows without a host round-trip.
    pub fn concat_rows(&self, parts: &[&CudaMatrix]) -> CudaMatrix {
        assert!(!parts.is_empty(), "concat_rows: empty parts");
        let cols = parts[0].cols;
        assert!(parts.iter().all(|p| p.cols == cols), "concat_rows: column mismatch");
        let rows: usize = parts.iter().map(|p| p.rows).sum();
        let mut out = self
            .stream
            .alloc_zeros::<bf16>(rows * cols)
            .expect("cuda alloc concat rows");
        let cols_a = cols;
        let mut row_start = 0usize;
        for p in parts {
            let (start_a, nrows_a) = (row_start, p.rows);
            let mut builder = self.stream.launch_builder(&self.kernels().place_rows);
            builder.arg(&mut out);
            builder.arg(&p.buf);
            builder.arg(&cols_a);
            builder.arg(&start_a);
            builder.arg(&nrows_a);
            // SAFETY: destination was allocated for the sum of all part rows.
            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems((p.rows * cols) as u32))
                    .expect("launch place_rows_kernel");
            }
            row_start += p.rows;
        }
        CudaMatrix { buf: out, rows, cols }
    }

'''
    return text[:pos] + methods + text[pos:]


CACHE_TYPES_OLD = '''/// Training-only cache for one attention invocation. Keeping RoPE outputs and\n/// per-head softmax weights avoids rebuilding the whole attention forward in the\n/// backward pass. Inference and parity APIs keep their existing recompute path.\nstruct CudaAttentionTrainCache {\n    qr: CudaMatrix,\n    kr: CudaMatrix,\n    weights: Vec<CudaMatrix>,\n}\n'''
CACHE_TYPES_NEW = '''/// Training cache for one sequence's attention.\nstruct CudaAttentionSequenceCache {\n    qr: CudaMatrix,\n    kr: CudaMatrix,\n    weights: Vec<CudaMatrix>,\n}\n\n/// Batch attention cache. Each sequence has an independent T×T causal attention\n/// graph even though all projection/MLP matrices are packed as B*T rows.\nstruct CudaAttentionTrainCache {\n    sequences: Vec<CudaAttentionSequenceCache>,\n    batch_size: usize,\n    seq_len: usize,\n}\n'''

ATTN_TRAIN = r'''    /// One sequence's cached training attention.
    fn attention_train_sequence(
        &self,
        q: &CudaMatrix,
        k: &CudaMatrix,
        v: &CudaMatrix,
    ) -> (CudaMatrix, CudaAttentionSequenceCache) {
        let ch = &self.chain;
        let dh = self.d_model / self.n_heads;
        let seq = q.rows();
        let qr = ch.rope(q, seq, 0, self.theta);
        let kr = ch.rope(k, seq, 0, self.theta);
        let repeat = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut out: Option<CudaMatrix> = None;
        let mut weights_all = Vec::with_capacity(self.n_heads);
        for head in 0..self.n_heads {
            let kv = head / repeat;
            let qs = ch.slice_cols(&qr, head * dh, dh);
            let ks = ch.slice_cols(&kr, kv * dh, dh);
            let vs = ch.slice_cols(v, kv * dh, dh);
            let scores = ch.matmul_bt(&qs, &ks);
            let scaled = ch.scale_causal_mask(&scores, scale, self.causal);
            let weights = ch.softmax(&scaled);
            let ctx = ch.matmul(&weights, &vs);
            let padded = ch.place_cols(&ctx, head * dh, self.d_model);
            out = Some(match out {
                None => padded,
                Some(acc) => ch.add(&acc, &padded),
            });
            weights_all.push(weights);
        }
        (
            out.expect("n_heads >= 1"),
            CudaAttentionSequenceCache { qr, kr, weights: weights_all },
        )
    }

    /// True batched attention: packed projection rows, isolated per-sequence causal
    /// attention. No token in one sample can attend to another sample.
    fn attention_train_batched(
        &self,
        q: &CudaMatrix,
        k: &CudaMatrix,
        v: &CudaMatrix,
        batch_size: usize,
        seq_len: usize,
    ) -> (CudaMatrix, CudaAttentionTrainCache) {
        assert_eq!(q.rows(), batch_size * seq_len, "batched attention q rows");
        assert_eq!(k.rows(), q.rows(), "batched attention k rows");
        assert_eq!(v.rows(), q.rows(), "batched attention v rows");
        let ch = &self.chain;
        let mut outputs = Vec::with_capacity(batch_size);
        let mut sequences = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let row = b * seq_len;
            let qs = ch.slice_rows(q, row, seq_len);
            let ks = ch.slice_rows(k, row, seq_len);
            let vs = ch.slice_rows(v, row, seq_len);
            let (out, cache) = self.attention_train_sequence(&qs, &ks, &vs);
            outputs.push(out);
            sequences.push(cache);
        }
        let refs: Vec<&CudaMatrix> = outputs.iter().collect();
        (
            ch.concat_rows(&refs),
            CudaAttentionTrainCache { sequences, batch_size, seq_len },
        )
    }'''

BLOCK_TRAIN = r'''    /// Training block forward: dense projections/MLP run over packed B*T rows,
    /// attention remains strictly separated by sample.
    fn block_train(
        &self,
        x: &CudaMatrix,
        b: &CudaBlock,
        batch_size: usize,
        seq_len: usize,
    ) -> (CudaMatrix, CudaBlockTrainCache) {
        let ch = &self.chain;
        let xn = ch.rms_norm(x, &b.norm1, self.eps);
        let q = ch.matmul(&xn, &b.wq);
        let k = ch.matmul(&xn, &b.wk);
        let v = ch.matmul(&xn, &b.wv);
        let (ctx, attention) = self.attention_train_batched(&q, &k, &v, batch_size, seq_len);
        let attn_out = ch.matmul(&ctx, &b.wo);
        let h = ch.add(x, &attn_out);
        let hn = ch.rms_norm(&h, &b.norm2, self.eps);
        let gate = ch.matmul(&hn, &b.wg);
        let up = ch.matmul(&hn, &b.wu);
        let act = ch.swiglu(&gate, &up);
        let mlp = ch.matmul(&act, &b.wd);
        let out = ch.add(&h, &mlp);
        (
            out,
            CudaBlockTrainCache { xn, q, k, v, attention, ctx, h, hn, gate, up, act },
        )
    }'''

FORWARD_TRAIN = r'''    /// Full packed B×T training forward with independent attention sequences.
    fn forward_train(
        &self,
        tokens: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> (CudaMatrix, CudaTrainCache) {
        assert!(batch_size > 0 && seq_len > 0, "forward_train: empty batch");
        assert_eq!(tokens.len(), batch_size * seq_len, "forward_train: B*T mismatch");
        let ch = &self.chain;
        let mut xs = Vec::with_capacity(self.blocks.len() + 1);
        let mut caches = Vec::with_capacity(self.blocks.len());
        xs.push(ch.embed(tokens, &self.embedding));
        for b in &self.blocks {
            let (out, cache) = self.block_train(
                xs.last().expect("block input"),
                b,
                batch_size,
                seq_len,
            );
            xs.push(out);
            caches.push(cache);
        }
        let normed = ch.rms_norm(xs.last().expect("trunk"), &self.final_norm, self.eps);
        let logits = ch.matmul_bt(&normed, &self.embedding);
        (logits, CudaTrainCache { xs, blocks: caches, normed })
    }'''

ATTN_BWD = r'''    fn attention_backward_sequence_cached(
        &self,
        v: &CudaMatrix,
        dout: &CudaMatrix,
        cache: &CudaAttentionSequenceCache,
    ) -> (CudaMatrix, CudaMatrix, CudaMatrix) {
        let ch = &self.chain;
        let dh = self.d_model / self.n_heads;
        let seq = cache.qr.rows();
        let kv_dim = self.n_kv_heads * dh;
        let repeat = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut dqr: Option<CudaMatrix> = None;
        let mut dkr: Option<CudaMatrix> = None;
        let mut dvv: Option<CudaMatrix> = None;
        for head in 0..self.n_heads {
            let kv = head / repeat;
            let qs = ch.slice_cols(&cache.qr, head * dh, dh);
            let ks = ch.slice_cols(&cache.kr, kv * dh, dh);
            let vs = ch.slice_cols(v, kv * dh, dh);
            let weights = &cache.weights[head];
            let d_ctx = ch.slice_cols(dout, head * dh, dh);
            let dweights = ch.matmul_bt(&d_ctx, &vs);
            let dvs = ch.matmul_at(weights, &d_ctx);
            let dscaled = ch.softmax_backward(weights, &dweights);
            let dscores = ch.scale_causal_mask_backward(&dscaled, scale, self.causal);
            let dqs = ch.matmul(&dscores, &ks);
            let dks = ch.matmul_at(&dscores, &qs);
            let dqs_full = ch.place_cols(&dqs, head * dh, self.d_model);
            let dks_full = ch.place_cols(&dks, kv * dh, kv_dim);
            let dvs_full = ch.place_cols(&dvs, kv * dh, kv_dim);
            dqr = Some(match dqr { None => dqs_full, Some(acc) => ch.add(&acc, &dqs_full) });
            dkr = Some(match dkr { None => dks_full, Some(acc) => ch.add(&acc, &dks_full) });
            dvv = Some(match dvv { None => dvs_full, Some(acc) => ch.add(&acc, &dvs_full) });
        }
        let dq = ch.rope_backward(&dqr.expect("heads"), seq, 0, self.theta);
        let dk = ch.rope_backward(&dkr.expect("heads"), seq, 0, self.theta);
        (dq, dk, dvv.expect("heads"))
    }

    fn attention_backward_cached(
        &self,
        v: &CudaMatrix,
        dout: &CudaMatrix,
        cache: &CudaAttentionTrainCache,
    ) -> (CudaMatrix, CudaMatrix, CudaMatrix) {
        let ch = &self.chain;
        assert_eq!(v.rows(), cache.batch_size * cache.seq_len, "attention backward v rows");
        assert_eq!(dout.rows(), v.rows(), "attention backward dout rows");
        let mut dqs = Vec::with_capacity(cache.batch_size);
        let mut dks = Vec::with_capacity(cache.batch_size);
        let mut dvs = Vec::with_capacity(cache.batch_size);
        for b in 0..cache.batch_size {
            let row = b * cache.seq_len;
            let vs = ch.slice_rows(v, row, cache.seq_len);
            let ds = ch.slice_rows(dout, row, cache.seq_len);
            let (dq, dk, dv) =
                self.attention_backward_sequence_cached(&vs, &ds, &cache.sequences[b]);
            dqs.push(dq);
            dks.push(dk);
            dvs.push(dv);
        }
        let qrefs: Vec<&CudaMatrix> = dqs.iter().collect();
        let krefs: Vec<&CudaMatrix> = dks.iter().collect();
        let vrefs: Vec<&CudaMatrix> = dvs.iter().collect();
        (ch.concat_rows(&qrefs), ch.concat_rows(&krefs), ch.concat_rows(&vrefs))
    }'''

TRAIN_STEP = r'''    pub fn train_step(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        lr: f32,
        betas: (f32, f32),
        adam_eps: f32,
        weight_decay: f32,
    ) -> f32 {
        self.train_step_batch(tokens, targets, 1, tokens.len(), lr, betas, adam_eps, weight_decay)
    }

    /// One true B×T optimizer step. Dense Transformer GEMMs consume all B*T rows in
    /// one call; attention is isolated per sequence so samples never cross-attend.
    #[allow(clippy::too_many_arguments)]
    pub fn train_step_batch(
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
        let loss = ch.mean_f32(&loss_rows);
        self.last_grad_norm = ch.grad_norm_from_sumsq(&grad_sumsq);
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
            let loss = self.train_step_batch(
                &packed_inputs,
                &packed_targets,
                batch,
                s,
                lr,
                cfg.betas,
                cfg.adam_eps,
                cfg.weight_decay,
            );
            losses.push(loss);
            step += 1;

            if cfg.log_interval > 0 && step.is_multiple_of(cfg.log_interval) {
                let done = (step - cfg.start_step) * s * batch;
                let secs = t0.elapsed().as_secs_f64().max(1e-9);
                let tps = done as f64 / secs;
                let gnorm = self.last_grad_norm();
                println!(
                    "[cuda step {step:>6}] B{batch}×T{s} loss {loss:>9.4} | lr {lr:.3e} | gnorm {gnorm:>7.2} | {tps:>8.0} tok/s"
                );
            }
            if cfg.eval_interval > 0
                && !val_tokens.is_empty()
                && step.is_multiple_of(cfg.eval_interval)
            {
                let val = self.eval_loss(val_tokens, s, cfg.eval_windows);
                println!("            └─ held-out val loss {val:>9.4}");
            }
            if cfg.save_interval > 0 && step.is_multiple_of(cfg.save_interval) {
                self.sync_to_model(model);
                let dir = std::path::Path::new(&cfg.checkpoint_dir).join(format!("step_{step}"));
                let meta = CheckpointMeta { step, loss, lr, config: config.clone() };
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
        losses
    }'''


def patch_model(text: str) -> str:
    if "train_step_batch" in text:
        raise SystemExit("model already B28 patched")
    text = must_replace(text, CACHE_TYPES_OLD, CACHE_TYPES_NEW)
    text = replace_fn(text, "    fn attention_train(\n", ATTN_TRAIN)
    text = replace_fn(text, "    fn block_train(", BLOCK_TRAIN)
    text = replace_fn(text, "    fn forward_train(", FORWARD_TRAIN)
    text = replace_fn(text, "    fn attention_backward_cached(\n", ATTN_BWD)
    text = replace_fn(text, "    pub fn train_step(\n", TRAIN_STEP)
    text = replace_fn(text, "    pub fn pretrain(\n", PRETRAIN)
    text = must_replace(
        text,
        "    /// Sequence length of each training window.\n    pub seq_len: usize,\n",
        "    /// Sequence length of each training sample.\n    pub seq_len: usize,\n    /// Number of independent sequences packed into each optimizer step.\n    pub batch_size: usize,\n",
    )
    text = must_replace(
        text,
        "            seq_len: 128,\n            log_interval: 100,\n",
        "            seq_len: 128,\n            batch_size: 1,\n            log_interval: 100,\n",
    )
    return text


def patch_example(text: str) -> str:
    text = must_replace(
        text,
        "//!   `SCIAGENT_SEQ` (128), `SCIAGENT_LR` — run knobs.\n",
        "//!   `SCIAGENT_SEQ` (128), `SCIAGENT_BATCH` (1), `SCIAGENT_LR` — run knobs.\n",
    )
    text = must_replace(
        text,
        "    let seq_len = env_usize(\"SCIAGENT_SEQ\", 128).min(config.max_seq_len);\n",
        "    let seq_len = env_usize(\"SCIAGENT_SEQ\", 128).min(config.max_seq_len);\n    let batch_size = env_usize(\"SCIAGENT_BATCH\", 1).max(1);\n",
    )
    text = must_replace(
        text,
        "        seq_len,\n        weight_decay: 0.0,\n",
        "        seq_len,\n        batch_size,\n        weight_decay: 0.0,\n",
    )
    text = must_replace(
        text,
        '        "seq_len {seq_len} | steps {start_step}..{total_steps} | base_lr {base_lr:.1e} | \\\n',
        '        "batch {batch_size} × seq_len {seq_len} | steps {start_step}..{total_steps} | base_lr {base_lr:.1e} | \\\n',
    )
    return text


CHAIN.write_text(patch_chain(CHAIN.read_text()))
MODEL.write_text(patch_model(MODEL.read_text()))
EXAMPLE.write_text(patch_example(EXAMPLE.read_text()))
print("patched B28 true batch")
