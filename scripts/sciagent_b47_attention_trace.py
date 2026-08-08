#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"
BENCH = ROOT / "scirust-sciagent/examples/cuda_production_bench.rs"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:180]!r}")
    return text.replace(old, new, count)


def patch_model(text: str) -> str:
    if "cache_attention_parity_teacher_forced" in text:
        return text
    marker = '''    /// Compare the first block's row-local inputs/projections under a 1×D cached\n'''
    if marker not in text:
        raise SystemExit("missing B46 insertion marker")
    method = r'''    /// Compare the exact internals of block-0 GQA under cached 1×T attention and
    /// full T×T causal attention for the same teacher-forced prefix. Components are
    /// concatenated across all query heads before comparison so every head is covered.
    pub fn cache_attention_parity_teacher_forced(
        &self,
        prompt: &[u32],
        forced_tokens: &[u32],
        target_step: usize,
    ) -> Vec<CudaCacheProjectionParity> {
        assert!(target_step > 0 && target_step <= forced_tokens.len());
        let ch = &self.chain;
        let mut prefix: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt.to_vec() };
        let mut kcache: Vec<Option<CudaMatrix>> = (0..self.blocks.len()).map(|_| None).collect();
        let mut vcache: Vec<Option<CudaMatrix>> = (0..self.blocks.len()).map(|_| None).collect();
        let _ = self.prefill_cached(&prefix, &mut kcache, &mut vcache);
        for (step, &token) in forced_tokens.iter().take(target_step).enumerate() {
            let pos = prefix.len();
            prefix.push(token);
            if step + 1 < target_step {
                let _ = self.decode_step_cached(token, pos, &mut kcache, &mut vcache);
                continue;
            }

            let b = self.blocks.first().expect("at least one block");
            let cached_x = ch.embed(&[token], &self.embedding);
            let cached_xn = ch.rms_norm(&cached_x, &b.norm1, self.eps);
            let cached_q = ch.matmul(&cached_xn, &b.wq);
            let cached_k = ch.matmul(&cached_xn, &b.wk);
            let cached_v = ch.matmul(&cached_xn, &b.wv);
            let cached_qr = self.rope_heads(&cached_q, self.n_heads, 1, pos);
            let cached_kr = self.rope_heads(&cached_k, self.n_kv_heads, 1, pos);
            let cached_k_all = match kcache[0].take() {
                Some(prev) => ch.concat_rows(&[&prev, &cached_kr]),
                None => cached_kr,
            };
            let cached_v_all = match vcache[0].take() {
                Some(prev) => ch.concat_rows(&[&prev, &cached_v]),
                None => cached_v,
            };

            let full_x = ch.embed(&prefix, &self.embedding);
            let full_xn = ch.rms_norm(&full_x, &b.norm1, self.eps);
            let full_q = ch.matmul(&full_xn, &b.wq);
            let full_k = ch.matmul(&full_xn, &b.wk);
            let full_v = ch.matmul(&full_xn, &b.wv);
            let full_qr = self.rope_heads(&full_q, self.n_heads, prefix.len(), 0);
            let full_kr = self.rope_heads(&full_k, self.n_kv_heads, prefix.len(), 0);

            let dh = self.d_model / self.n_heads;
            let repeat = self.n_heads / self.n_kv_heads;
            let scale = 1.0 / (dh as f32).sqrt();
            let mut cached_scores = Vec::with_capacity(self.n_heads);
            let mut full_scores = Vec::with_capacity(self.n_heads);
            let mut cached_scaled = Vec::with_capacity(self.n_heads);
            let mut full_scaled = Vec::with_capacity(self.n_heads);
            let mut cached_weights = Vec::with_capacity(self.n_heads);
            let mut full_weights = Vec::with_capacity(self.n_heads);
            let mut cached_ctx = Vec::with_capacity(self.n_heads);
            let mut full_ctx = Vec::with_capacity(self.n_heads);

            for head in 0..self.n_heads {
                let kv = head / repeat;
                let cqs = ch.slice_cols(&cached_qr, head * dh, dh);
                let cks = ch.slice_cols(&cached_k_all, kv * dh, dh);
                let cvs = ch.slice_cols(&cached_v_all, kv * dh, dh);
                let fsq = ch.slice_cols(&full_qr, head * dh, dh);
                let fsk = ch.slice_cols(&full_kr, kv * dh, dh);
                let fsv = ch.slice_cols(&full_v, kv * dh, dh);

                let cs = ch.matmul_bt(&cqs, &cks);
                let fs_all = ch.matmul_bt(&fsq, &fsk);
                let fs = ch.slice_rows(&fs_all, pos, 1);
                let cscaled = ch.scale_causal_mask(&cs, scale, false);
                let fscaled_all = ch.scale_causal_mask(&fs_all, scale, self.causal);
                let fscaled = ch.slice_rows(&fscaled_all, pos, 1);
                let cw = ch.softmax(&cscaled);
                let fw_all = ch.softmax(&fscaled_all);
                let fw = ch.slice_rows(&fw_all, pos, 1);
                let cc = ch.matmul(&cw, &cvs);
                let fc_all = ch.matmul(&fw_all, &fsv);
                let fc = ch.slice_rows(&fc_all, pos, 1);

                cached_scores.push(cs);
                full_scores.push(fs);
                cached_scaled.push(cscaled);
                full_scaled.push(fscaled);
                cached_weights.push(cw);
                full_weights.push(fw);
                cached_ctx.push(cc);
                full_ctx.push(fc);
            }

            let concat = |xs: &[CudaMatrix]| {
                let refs: Vec<&CudaMatrix> = xs.iter().collect();
                ch.concat_cols(&refs)
            };
            let compare = |component: &'static str, a: &CudaMatrix, b: &CudaMatrix| {
                let ah = ch.download(a);
                let bh = ch.download(b);
                assert_eq!(ah.len(), bh.len());
                let mut num = 0.0f64;
                let mut den = 0.0f64;
                let mut max_abs = 0.0f32;
                for (&x, &y) in ah.iter().zip(&bh) {
                    let d = x - y;
                    num += (d as f64) * (d as f64);
                    den += (y as f64) * (y as f64);
                    max_abs = max_abs.max(d.abs());
                }
                CudaCacheProjectionParity {
                    component,
                    rel_l2: (num.sqrt() / den.sqrt().max(1e-30)) as f32,
                    max_abs,
                }
            };

            let cs = concat(&cached_scores);
            let fs = concat(&full_scores);
            let cscaled = concat(&cached_scaled);
            let fscaled = concat(&full_scaled);
            let cw = concat(&cached_weights);
            let fw = concat(&full_weights);
            let cc = concat(&cached_ctx);
            let fc = concat(&full_ctx);

            let cached_attn_out = ch.matmul(&cc, &b.wo);
            let full_attn_out = ch.matmul(&fc, &b.wo);
            let cached_h = ch.add(&cached_x, &cached_attn_out);
            let full_last_x = ch.slice_rows(&full_x, pos, 1);
            let full_h = ch.add(&full_last_x, &full_attn_out);
            let cached_hn = ch.rms_norm(&cached_h, &b.norm2, self.eps);
            let full_hn = ch.rms_norm(&full_h, &b.norm2, self.eps);
            let cached_gate = ch.matmul(&cached_hn, &b.wg);
            let full_gate = ch.matmul(&full_hn, &b.wg);
            let cached_up = ch.matmul(&cached_hn, &b.wu);
            let full_up = ch.matmul(&full_hn, &b.wu);
            let cached_act = ch.swiglu(&cached_gate, &cached_up);
            let full_act = ch.swiglu(&full_gate, &full_up);
            let cached_mlp = ch.matmul(&cached_act, &b.wd);
            let full_mlp = ch.matmul(&full_act, &b.wd);
            let cached_out = ch.add(&cached_h, &cached_mlp);
            let full_out = ch.add(&full_h, &full_mlp);

            return vec![
                compare("kcache", &cached_k_all, &full_kr),
                compare("vcache", &cached_v_all, &full_v),
                compare("scores", &cs, &fs),
                compare("scaled", &cscaled, &fscaled),
                compare("weights", &cw, &fw),
                compare("context", &cc, &fc),
                compare("wo", &cached_attn_out, &full_attn_out),
                compare("h", &cached_h, &full_h),
                compare("norm2", &cached_hn, &full_hn),
                compare("gate", &cached_gate, &full_gate),
                compare("up", &cached_up, &full_up),
                compare("act", &cached_act, &full_act),
                compare("mlp", &cached_mlp, &full_mlp),
                compare("block_out", &cached_out, &full_out),
            ];
        }
        unreachable!("target_step guarantees one traced token")
    }

'''
    return must_replace(text, marker, method + marker)


def patch_bench(text: str) -> str:
    if "SCIAGENT_THOR_KV_ATTENTION" in text:
        return text
    old = '''                for component in\n                    cuda.cache_projection_parity_teacher_forced(&prompt, forced, first.step)\n                {\n'''
    new = '''                for component in\n                    cuda.cache_projection_parity_teacher_forced(&prompt, forced, first.step)\n                {\n'''
    # Keep existing projection trace, then inject attention internals before layer trace.
    anchor = '''                for layer in cuda.cache_layer_parity_teacher_forced(&prompt, forced, first.step)\n                {\n'''
    insert = '''                for component in\n                    cuda.cache_attention_parity_teacher_forced(&prompt, forced, first.step)\n                {\n                    println!(\n                        "SCIAGENT_THOR_KV_ATTENTION step={} component={} rel_l2={:.8e} max_abs={:.8e}",\n                        first.step, component.component, component.rel_l2, component.max_abs\n                    );\n                }\n'''
    if old not in text:
        raise SystemExit("missing B46 projection trace")
    return must_replace(text, anchor, insert + anchor)


MODEL.write_text(patch_model(MODEL.read_text()))
BENCH.write_text(patch_bench(BENCH.read_text()))
print("B47 patched: attention-internal teacher-forced parity trace")
