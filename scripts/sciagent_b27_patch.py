#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "scirust-cuda/src/chain.rs"
MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:120]!r}")
    return text.replace(old, new, count)


def replace_fn(text: str, marker: str, new_src: str) -> str:
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"missing function marker: {marker}")
    brace = text.find("{", start)
    if brace < 0:
        raise SystemExit(f"missing opening brace: {marker}")
    depth = 0
    i = brace
    in_str = False
    escaped = False
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
                    end = i + 1
                    return text[:start] + new_src.rstrip() + text[end:]
        i += 1
    raise SystemExit(f"unterminated function: {marker}")


ADAM_NORM_KERNEL = r'''
// AdamW variant for uninterrupted training: consume the resident global gradient
// sum-of-squares directly. This removes the mid-step GPU->CPU synchronization that
// previously existed only to compute the clipping factor on the host.
extern "C" __global__ void adamw_norm_kernel(
    float* param, float* m, float* v, const unsigned short* grad, unsigned short* param_bf,
    const size_t n, const float lr, const float b1, const float b2,
    const float eps, const float wd, const float bc1, const float bc2,
    const float* grad_sumsq, const float max_grad_norm)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float sq = grad_sumsq[0];
        const float norm = sqrtf(sq);
        float scale = 1.0f;
        // NaN fails `sq >= 0`; +inf exceeds the finite guard. Match the host path:
        // a non-finite norm skips the update instead of corrupting the parameters.
        if (!(sq >= 0.0f) || sq > 3.0e38f) scale = 0.0f;
        else if (max_grad_norm > 0.0f && norm > max_grad_norm)
            scale = max_grad_norm / norm;

        float g = b2f(grad[i]) * scale;
        float mi = b1 * m[i] + (1.0f - b1) * g;
        float vi = b2 * v[i] + (1.0f - b2) * g * g;
        m[i] = mi;
        v[i] = vi;
        float mhat = mi / bc1;
        float vhat = vi / bc2;
        float p = param[i];
        p -= lr * (mhat / (sqrtf(vhat) + eps) + wd * p);
        param[i] = p;
        param_bf[i] = f2b(p);
    }
}
'''


def patch_chain(text: str) -> str:
    if "adamw_norm_kernel" in text:
        raise SystemExit("chain already B27 patched")
    text = must_replace(
        text,
        "// Sum of squares of a bf16 buffer, accumulated (fp32) into accum[0] — the building\n",
        ADAM_NORM_KERNEL + "\n// Sum of squares of a bf16 buffer, accumulated (fp32) into accum[0] — the building\n",
    )
    text = must_replace(text, "    adamw: CudaFunction,\n    sumsq: CudaFunction,\n", "    adamw: CudaFunction,\n    adamw_norm: CudaFunction,\n    sumsq: CudaFunction,\n")
    text = must_replace(text, '            adamw: f("adamw_kernel"),\n            sumsq: f("sumsq_kernel"),\n', '            adamw: f("adamw_kernel"),\n            adamw_norm: f("adamw_norm_kernel"),\n            sumsq: f("sumsq_kernel"),\n')

    old_ce = '''    /// Mean cross-entropy plus its resident bf16 logit gradient in one pass.\n    /// The only D2H transfer is `rows` fp32 scalars used for deterministic logging.\n    pub fn cross_entropy_loss_grad(\n        &self,\n        logits: &CudaMatrix,\n        targets: &[u32],\n    ) -> (f32, CudaMatrix) {'''
    if old_ce not in text:
        raise SystemExit("missing cross_entropy_loss_grad")

    # Replace only the fused CE method; the eval-only cross_entropy_loss remains a
    # convenient synchronous wrapper.
    text = replace_fn(text, "    pub fn cross_entropy_loss_grad(\n", r'''    pub fn cross_entropy_loss_grad(
        &self,
        logits: &CudaMatrix,
        targets: &[u32],
    ) -> (f32, CudaMatrix) {
        let (loss_rows, grad) = self.cross_entropy_loss_grad_resident(logits, targets);
        (self.mean_f32(&loss_rows), grad)
    }''')

    insert_at = text.find("    pub fn cross_entropy_loss_grad(\n")
    if insert_at < 0:
        raise SystemExit("cannot find CE wrapper insertion")
    resident_ce = r'''    /// Cross-entropy loss rows plus resident bf16 logit gradient in one CUDA pass.
    /// No host synchronization occurs: callers may enqueue backward and AdamW first,
    /// then read the tiny loss vector after the optimizer has completed on the stream.
    pub fn cross_entropy_loss_grad_resident(
        &self,
        logits: &CudaMatrix,
        targets: &[u32],
    ) -> (CudaF32, CudaMatrix) {
        let (rows, cols) = (logits.rows, logits.cols);
        assert!(
            rows > 0 && cols > 0,
            "cross_entropy_loss_grad_resident: empty logits"
        );
        assert_eq!(
            targets.len(),
            rows,
            "cross_entropy_loss_grad_resident: target count"
        );
        let tgt = self.stream.clone_htod(targets).expect("cuda htod targets");
        let mut loss_rows = self
            .stream
            .alloc_zeros::<f32>(rows)
            .expect("cuda alloc CE loss");
        let mut d = self
            .stream
            .alloc_zeros::<bf16>(rows * cols)
            .expect("cuda alloc CE grad");
        let (rows_a, cols_a) = (rows, cols);
        let mut builder = self.stream.launch_builder(&self.kernels().ce_loss_grad);
        builder.arg(&mut loss_rows);
        builder.arg(&mut d);
        builder.arg(&logits.buf);
        builder.arg(&tgt);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: one block owns each row and writes disjoint loss/gradient data.
        unsafe { builder.launch(cfg).expect("launch ce_loss_grad_kernel") };
        (
            CudaF32 {
                buf: loss_rows,
                len: rows,
            },
            CudaMatrix { buf: d, rows, cols },
        )
    }

    /// Deterministic host mean of a small resident fp32 vector. This is intentionally
    /// a synchronization point and should be called only after all step kernels are
    /// enqueued when used for training diagnostics.
    pub fn mean_f32(&self, x: &CudaF32) -> f32 {
        if x.len == 0 {
            return f32::NAN;
        }
        let host = self.stream.clone_dtoh(&x.buf).expect("cuda dtoh f32 mean");
        (host.iter().map(|&v| v as f64).sum::<f64>() / host.len() as f64) as f32
    }

'''
    text = text[:insert_at] + resident_ce + text[insert_at:]

    text = must_replace(
        text,
        "    /// Cross-entropy gradient w.r.t. the logits: `(softmax(logits) − onehot(target))\n    /// / rows`, one row per target. The loss itself is computed host-side from the\n    /// downloaded logits (as Route A does), so only the grad is resident here.\n",
        "    /// Cross-entropy gradient w.r.t. the logits: `(softmax(logits) − onehot(target))\n    /// / rows`, one row per target. Delegates to the fused device CE path.\n",
    )

    # Add resident-clipping AdamW beside the existing compatibility API.
    marker = "    /// The global L2 norm `sqrt(Σᵢ ‖gᵢ‖²)` over a set of gradient matrices — for\n"
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing global grad docs")
    adam_method = r'''    /// AdamW update using a resident global gradient sum-of-squares. The clipping
    /// factor is computed inside the kernel, so training does not synchronize with
    /// the host between backward and optimizer update.
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step_with_norm(
        &self,
        param: &mut CudaF32,
        m: &mut CudaF32,
        v: &mut CudaF32,
        grad: &CudaMatrix,
        param_bf16: &mut CudaMatrix,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        step: u32,
        grad_sumsq: &CudaF32,
        max_grad_norm: f32,
    ) {
        let n = param.len;
        assert_eq!(m.len, n, "adamw_step_with_norm: m len");
        assert_eq!(v.len, n, "adamw_step_with_norm: v len");
        assert_eq!(grad.rows * grad.cols, n, "adamw_step_with_norm: grad len");
        assert_eq!(param_bf16.rows * param_bf16.cols, n, "adamw_step_with_norm: view len");
        assert_eq!(grad_sumsq.len, 1, "adamw_step_with_norm: sumsq must be scalar");
        let (b1, b2) = betas;
        let bc1 = 1.0 - b1.powi(step as i32);
        let bc2 = 1.0 - b2.powi(step as i32);
        let (n_a, lr_a, b1_a, b2_a, eps_a, wd_a, bc1_a, bc2_a, max_a) = (
            n,
            lr,
            b1,
            b2,
            eps,
            weight_decay,
            bc1,
            bc2,
            max_grad_norm,
        );
        let mut builder = self.stream.launch_builder(&self.kernels().adamw_norm);
        builder.arg(&mut param.buf);
        builder.arg(&mut m.buf);
        builder.arg(&mut v.buf);
        builder.arg(&grad.buf);
        builder.arg(&mut param_bf16.buf);
        builder.arg(&n_a);
        builder.arg(&lr_a);
        builder.arg(&b1_a);
        builder.arg(&b2_a);
        builder.arg(&eps_a);
        builder.arg(&wd_a);
        builder.arg(&bc1_a);
        builder.arg(&bc2_a);
        builder.arg(&grad_sumsq.buf);
        builder.arg(&max_a);
        // SAFETY: argument layout matches adamw_norm_kernel and grid covers n.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(n as u32))
                .expect("launch adamw_norm_kernel");
        }
    }

    /// Accumulate the global gradient sum-of-squares into a resident fp32 scalar.
    /// This is the asynchronous half of global gradient clipping.
    pub fn global_grad_sumsq(&self, grads: &[&CudaMatrix]) -> CudaF32 {
        let mut accum = self.stream.alloc_zeros::<f32>(1).expect("cuda alloc accum");
        for g in grads {
            let n = g.rows * g.cols;
            if n == 0 {
                continue;
            }
            let block = 256u32;
            let grid = (n as u32).div_ceil(block);
            let n_a = n;
            let mut builder = self.stream.launch_builder(&self.kernels().sumsq);
            builder.arg(&g.buf);
            builder.arg(&n_a);
            builder.arg(&mut accum);
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: argument layout matches sumsq_kernel; block size is 256.
            unsafe { builder.launch(cfg).expect("launch sumsq_kernel") };
        }
        CudaF32 { buf: accum, len: 1 }
    }

    /// Read a resident gradient sum-of-squares after the optimizer stream is done.
    pub fn grad_norm_from_sumsq(&self, sumsq: &CudaF32) -> f32 {
        assert_eq!(sumsq.len, 1, "grad_norm_from_sumsq: scalar required");
        let host = self.stream.clone_dtoh(&sumsq.buf).expect("cuda dtoh grad sumsq");
        host[0].sqrt()
    }

'''
    text = text[:pos] + adam_method + text[pos:]

    text = replace_fn(text, "    pub fn global_grad_norm(&self, grads: &[&CudaMatrix]) -> f32 ", r'''    pub fn global_grad_norm(&self, grads: &[&CudaMatrix]) -> f32 {
        let sumsq = self.global_grad_sumsq(grads);
        self.grad_norm_from_sumsq(&sumsq)
    }''')
    return text


CACHE_TYPES = r'''
/// Training-only cache for one attention invocation. Keeping RoPE outputs and
/// per-head softmax weights avoids rebuilding the whole attention forward in the
/// backward pass. Inference and parity APIs keep their existing recompute path.
struct CudaAttentionTrainCache {
    qr: CudaMatrix,
    kr: CudaMatrix,
    weights: Vec<CudaMatrix>,
}

/// Training-only activations for one Transformer block.
struct CudaBlockTrainCache {
    xn: CudaMatrix,
    q: CudaMatrix,
    k: CudaMatrix,
    v: CudaMatrix,
    attention: CudaAttentionTrainCache,
    ctx: CudaMatrix,
    h: CudaMatrix,
    hn: CudaMatrix,
    gate: CudaMatrix,
    up: CudaMatrix,
    act: CudaMatrix,
}

/// Full training forward cache. `xs[i]` is block i's input and `xs[n]` the trunk.
struct CudaTrainCache {
    xs: Vec<CudaMatrix>,
    blocks: Vec<CudaBlockTrainCache>,
    normed: CudaMatrix,
}
'''

TRAIN_METHODS = r'''
    /// Training attention forward retaining only the activations required by its VJP.
    fn attention_train(
        &self,
        q: &CudaMatrix,
        k: &CudaMatrix,
        v: &CudaMatrix,
    ) -> (CudaMatrix, CudaAttentionTrainCache) {
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
            CudaAttentionTrainCache {
                qr,
                kr,
                weights: weights_all,
            },
        )
    }

    /// Training block forward retaining activations instead of recomputing them later.
    fn block_train(&self, x: &CudaMatrix, b: &CudaBlock) -> (CudaMatrix, CudaBlockTrainCache) {
        let ch = &self.chain;
        let xn = ch.rms_norm(x, &b.norm1, self.eps);
        let q = ch.matmul(&xn, &b.wq);
        let k = ch.matmul(&xn, &b.wk);
        let v = ch.matmul(&xn, &b.wv);
        let (ctx, attention) = self.attention_train(&q, &k, &v);
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
            CudaBlockTrainCache {
                xn,
                q,
                k,
                v,
                attention,
                ctx,
                h,
                hn,
                gate,
                up,
                act,
            },
        )
    }

    /// Full training forward with an explicit activation cache.
    fn forward_train(&self, tokens: &[u32]) -> (CudaMatrix, CudaTrainCache) {
        let ch = &self.chain;
        let mut xs = Vec::with_capacity(self.blocks.len() + 1);
        let mut caches = Vec::with_capacity(self.blocks.len());
        xs.push(ch.embed(tokens, &self.embedding));
        for b in &self.blocks {
            let (out, cache) = self.block_train(xs.last().expect("block input"), b);
            xs.push(out);
            caches.push(cache);
        }
        let normed = ch.rms_norm(xs.last().expect("trunk"), &self.final_norm, self.eps);
        let logits = ch.matmul_bt(&normed, &self.embedding);
        (
            logits,
            CudaTrainCache {
                xs,
                blocks: caches,
                normed,
            },
        )
    }

    fn attention_backward_cached(
        &self,
        v: &CudaMatrix,
        dout: &CudaMatrix,
        cache: &CudaAttentionTrainCache,
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

    fn block_backward_cached(
        &self,
        x: &CudaMatrix,
        b: &CudaBlock,
        cache: &CudaBlockTrainCache,
        dout: &CudaMatrix,
    ) -> (CudaMatrix, CudaBlockGrads) {
        let ch = &self.chain;
        let dact = ch.matmul_bt(dout, &b.wd);
        let dwd = ch.matmul_at(&cache.act, dout);
        let (dgate, dup) = ch.swiglu_backward(&cache.gate, &cache.up, &dact);
        let dwg = ch.matmul_at(&cache.hn, &dgate);
        let dwu = ch.matmul_at(&cache.hn, &dup);
        let dhn = ch.add(&ch.matmul_bt(&dgate, &b.wg), &ch.matmul_bt(&dup, &b.wu));
        let dnorm2 = ch.rms_norm_gain_backward(&cache.h, &dhn, self.eps);
        let dh = ch.add(dout, &ch.rms_norm_backward(&cache.h, &b.norm2, &dhn, self.eps));

        let dwo = ch.matmul_at(&cache.ctx, &dh);
        let d_ctx = ch.matmul_bt(&dh, &b.wo);
        let (dq, dk, dv) = self.attention_backward_cached(&cache.v, &d_ctx, &cache.attention);
        let dwq = ch.matmul_at(&cache.xn, &dq);
        let dwk = ch.matmul_at(&cache.xn, &dk);
        let dwv = ch.matmul_at(&cache.xn, &dv);
        let dxn = ch.add(
            &ch.add(&ch.matmul_bt(&dq, &b.wq), &ch.matmul_bt(&dk, &b.wk)),
            &ch.matmul_bt(&dv, &b.wv),
        );
        let dnorm1 = ch.rms_norm_gain_backward(x, &dxn, self.eps);
        let dx = ch.add(&dh, &ch.rms_norm_backward(x, &b.norm1, &dxn, self.eps));
        (
            dx,
            CudaBlockGrads {
                dwq,
                dwk,
                dwv,
                dwo,
                dwg,
                dwu,
                dwd,
                dnorm1,
                dnorm2,
            },
        )
    }

    fn backward_cached(
        &self,
        tokens: &[u32],
        dlogits: &CudaMatrix,
        cache: &CudaTrainCache,
    ) -> CudaModelGrads {
        let ch = &self.chain;
        let trunk = cache.xs.last().expect("trunk");
        let d_normed = ch.matmul(dlogits, &self.embedding);
        let de_head = ch.matmul_at(dlogits, &cache.normed);
        let d_final_norm = ch.rms_norm_gain_backward(trunk, &d_normed, self.eps);
        let mut d_cur = ch.rms_norm_backward(trunk, &self.final_norm, &d_normed, self.eps);
        let mut block_grads = Vec::with_capacity(self.blocks.len());
        for i in (0..self.blocks.len()).rev() {
            let (dx, grads) = self.block_backward_cached(
                &cache.xs[i],
                &self.blocks[i],
                &cache.blocks[i],
                &d_cur,
            );
            d_cur = dx;
            block_grads.push(grads);
        }
        block_grads.reverse();
        let de_embed = ch.embed_backward(tokens, &d_cur, self.vocab);
        let d_embedding = ch.add(&de_head, &de_embed);
        CudaModelGrads {
            d_embedding,
            blocks: block_grads,
            d_final_norm,
        }
    }
'''


def patch_model(text: str) -> str:
    if "CudaTrainCache" in text:
        raise SystemExit("model already B27 patched")
    # Cache types after public gradient bundle.
    anchor = '''pub struct CudaModelGrads {\n    pub d_embedding: CudaMatrix,\n    pub blocks: Vec<CudaBlockGrads>,\n    pub d_final_norm: CudaMatrix,\n}\n'''
    text = must_replace(text, anchor, anchor + CACHE_TYPES + "\n")

    # Insert training-specific forward/backward methods just before the validation
    # embedding_grad API, leaving the existing recompute backward untouched for parity.
    marker = "    /// The tied-embedding gradient for `(tokens, targets)`, downloaded — the single\n"
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing embedding_grad insertion point")
    text = text[:pos] + TRAIN_METHODS + "\n" + text[pos:]

    # Remove the orphaned host-CE rustdoc from B23-B26 and correct training docs.
    text = must_replace(
        text,
        "/// Host mean cross-entropy `−(1/rows)·Σ log P[i, tgtᵢ]` over row-major logits —\n/// the pre-update loss (matches `train::cross_entropy_loss`). Kept here so the CUDA\n/// path needs no `scirust-gpu` dependency.\n\n",
        "",
    )
    text = must_replace(text, "    /// One mixed-precision AdamW training step on `(tokens, targets)`: forward →\n    /// host cross-entropy grad → backward → AdamW update of every trainable weight\n", "    /// One mixed-precision AdamW training step on `(tokens, targets)`: forward →\n    /// resident cross-entropy grad → cached backward → AdamW update of every trainable weight\n")

    # Replace the full training-step body/API. The public signature is unchanged.
    text = replace_fn(text, "    pub fn train_step(\n", r'''    pub fn train_step(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        lr: f32,
        betas: (f32, f32),
        adam_eps: f32,
        weight_decay: f32,
    ) -> f32 {
        self.step += 1;

        // One uninterrupted CUDA stream: cached forward -> fused resident CE ->
        // cached backward -> resident grad norm -> AdamW. Host diagnostics are read
        // only after every optimizer kernel has been queued.
        let (logits, cache) = self.model.forward_train(tokens);
        let (loss_rows, dlogits) = self
            .model
            .chain
            .cross_entropy_loss_grad_resident(&logits, targets);
        let grads = self.model.backward_cached(tokens, &dlogits, &cache);

        let mut grad_refs: Vec<&CudaMatrix> = vec![&grads.d_embedding, &grads.d_final_norm];
        for bg in &grads.blocks {
            grad_refs.push(&bg.dnorm1);
            grad_refs.push(&bg.dwq);
            grad_refs.push(&bg.dwk);
            grad_refs.push(&bg.dwv);
            grad_refs.push(&bg.dwo);
            grad_refs.push(&bg.dnorm2);
            grad_refs.push(&bg.dwg);
            grad_refs.push(&bg.dwu);
            grad_refs.push(&bg.dwd);
        }
        let ch = &self.model.chain;
        let grad_sumsq = ch.global_grad_sumsq(&grad_refs);
        drop(grad_refs);

        let step = self.step;
        let max_norm = self.max_grad_norm;
        ch.adamw_step_with_norm(
            &mut self.master_embedding,
            &mut self.m_embedding,
            &mut self.v_embedding,
            &grads.d_embedding,
            &mut self.model.embedding,
            lr,
            betas,
            adam_eps,
            weight_decay,
            step,
            &grad_sumsq,
            max_norm,
        );
        ch.adamw_step_with_norm(
            &mut self.master_final_norm,
            &mut self.m_final_norm,
            &mut self.v_final_norm,
            &grads.d_final_norm,
            &mut self.model.final_norm,
            lr,
            betas,
            adam_eps,
            weight_decay,
            step,
            &grad_sumsq,
            max_norm,
        );
        for i in 0..self.model.blocks.len() {
            let bg = &grads.blocks[i];
            let (mb, mm, mv) = (
                &mut self.master_blocks[i],
                &mut self.m_blocks[i],
                &mut self.v_blocks[i],
            );
            let b = &mut self.model.blocks[i];
            let one = |master: &mut CudaF32,
                       mo: &mut CudaF32,
                       vo: &mut CudaF32,
                       grad: &CudaMatrix,
                       view: &mut CudaMatrix| {
                ch.adamw_step_with_norm(
                    master,
                    mo,
                    vo,
                    grad,
                    view,
                    lr,
                    betas,
                    adam_eps,
                    weight_decay,
                    step,
                    &grad_sumsq,
                    max_norm,
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

        // First host reads of the step. Because all work shares one CUDA stream,
        // these synchronize only after the optimizer has completed.
        let loss = ch.mean_f32(&loss_rows);
        self.last_grad_norm = ch.grad_norm_from_sumsq(&grad_sumsq);
        loss
    }''')
    return text


chain = CHAIN.read_text()
model = MODEL.read_text()
CHAIN.write_text(patch_chain(chain))
MODEL.write_text(patch_model(model))
print("patched B27", CHAIN.relative_to(ROOT), MODEL.relative_to(ROOT))
