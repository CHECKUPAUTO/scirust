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
    if "CudaCacheLayerParity" in text:
        return text
    struct_anchor = '''#[derive(Clone, Debug)]\npub struct CudaCacheParityStep {\n'''
    idx = text.find(struct_anchor)
    if idx < 0:
        raise SystemExit("missing CudaCacheParityStep")
    # Insert the layer record before the existing step record.
    layer_struct = '''#[derive(Clone, Debug)]\npub struct CudaCacheLayerParity {\n    pub layer: usize,\n    pub rel_l2: f32,\n    pub max_abs: f32,\n}\n\n'''
    text = text[:idx] + layer_struct + text[idx:]

    marker = '''    /// Backward of [`Self::attention`] (the GQA analogue of Route A's\n'''
    if marker not in text:
        raise SystemExit("missing insertion marker")
    method = r'''    /// Localize cached-vs-full numerical drift after each complete transformer
    /// block for one teacher-forced prediction step. `target_step=1` means the state
    /// after processing `forced_tokens[0]`; step 0 is prefill and is already checked
    /// directly by `cache_parity_teacher_forced`.
    pub fn cache_layer_parity_teacher_forced(
        &self,
        prompt: &[u32],
        forced_tokens: &[u32],
        target_step: usize,
    ) -> Vec<CudaCacheLayerParity> {
        assert!(target_step > 0, "layer trace target must be after prefill");
        assert!(target_step <= forced_tokens.len(), "layer trace target out of range");
        let ch = &self.chain;
        let mut prefix: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt.to_vec() };
        let mut kcache: Vec<Option<CudaMatrix>> = (0..self.blocks.len()).map(|_| None).collect();
        let mut vcache: Vec<Option<CudaMatrix>> = (0..self.blocks.len()).map(|_| None).collect();
        let _ = self.prefill_cached(&prefix, &mut kcache, &mut vcache);

        let mut cached_trace: Vec<Vec<f32>> = Vec::new();
        for step in 0..target_step {
            let token = forced_tokens[step];
            let pos = prefix.len();
            prefix.push(token);
            if step + 1 < target_step {
                let _ = self.decode_step_cached(token, pos, &mut kcache, &mut vcache);
                continue;
            }

            // Final teacher-forced token: same decode as production, but retain one
            // host row after every block solely for this failure diagnostic.
            let mut x = ch.embed(&[token], &self.embedding);
            for (layer, b) in self.blocks.iter().enumerate() {
                let xn = ch.rms_norm(&x, &b.norm1, self.eps);
                let q = ch.matmul(&xn, &b.wq);
                let k = ch.matmul(&xn, &b.wk);
                let v = ch.matmul(&xn, &b.wv);
                let qr = self.rope_heads(&q, self.n_heads, 1, pos);
                let kr = self.rope_heads(&k, self.n_kv_heads, 1, pos);
                kcache[layer] = Some(match kcache[layer].take() {
                    None => kr,
                    Some(prev) => ch.concat_rows(&[&prev, &kr]),
                });
                vcache[layer] = Some(match vcache[layer].take() {
                    None => v,
                    Some(prev) => ch.concat_rows(&[&prev, &v]),
                });
                let ctx = self.incremental_attention(
                    &qr,
                    kcache[layer].as_ref().expect("K cache"),
                    vcache[layer].as_ref().expect("V cache"),
                );
                let attn_out = ch.matmul(&ctx, &b.wo);
                let h = ch.add(&x, &attn_out);
                let hn = ch.rms_norm(&h, &b.norm2, self.eps);
                let gate = ch.matmul(&hn, &b.wg);
                let up = ch.matmul(&hn, &b.wu);
                let act = ch.swiglu(&gate, &up);
                let mlp = ch.matmul(&act, &b.wd);
                x = ch.add(&h, &mlp);
                cached_trace.push(ch.download(&x));
            }
        }

        // Independent full-forward trace under the exact same prefix. Causality means
        // the last row is the mathematical reference for the cached new-token row.
        let mut full_x = ch.embed(&prefix, &self.embedding);
        let mut full_trace: Vec<Vec<f32>> = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            full_x = self.block(&full_x, b);
            let last = ch.slice_rows(&full_x, prefix.len() - 1, 1);
            full_trace.push(ch.download(&last));
        }
        assert_eq!(cached_trace.len(), full_trace.len());

        cached_trace
            .iter()
            .zip(&full_trace)
            .enumerate()
            .map(|(layer, (cached, full))| {
                let mut num = 0.0f64;
                let mut den = 0.0f64;
                let mut max_abs = 0.0f32;
                for (&a, &b) in cached.iter().zip(full) {
                    let d = a - b;
                    num += (d as f64) * (d as f64);
                    den += (b as f64) * (b as f64);
                    max_abs = max_abs.max(d.abs());
                }
                CudaCacheLayerParity {
                    layer,
                    rel_l2: (num.sqrt() / den.sqrt().max(1e-30)) as f32,
                    max_abs,
                }
            })
            .collect()
    }

'''
    return must_replace(text, marker, method + marker)


def patch_bench(text: str) -> str:
    if "SCIAGENT_THOR_KV_LAYER" in text:
        return text
    old = '''        if let Some(first) = report.iter().find(|row| row.cached_top1 != row.full_top1)\n        {\n            eprintln!(\n                "ERROR: cached CUDA decoding first teacher-forced top-1 divergence at step {} (cached={} full={}, rel_l2={:.3e}, max_abs={:.3e})",\n                first.step, first.cached_top1, first.full_top1, first.rel_l2, first.max_abs\n            );\n        }\n'''
    new = '''        if let Some(first) = report.iter().find(|row| row.cached_top1 != row.full_top1)\n        {\n            eprintln!(\n                "ERROR: cached CUDA decoding first teacher-forced top-1 divergence at step {} (cached={} full={}, rel_l2={:.3e}, max_abs={:.3e})",\n                first.step, first.cached_top1, first.full_top1, first.rel_l2, first.max_abs\n            );\n            if first.step > 0\n            {\n                for layer in cuda.cache_layer_parity_teacher_forced(&prompt, forced, first.step)\n                {\n                    println!(\n                        "SCIAGENT_THOR_KV_LAYER step={} layer={} rel_l2={:.8e} max_abs={:.8e}",\n                        first.step, layer.layer, layer.rel_l2, layer.max_abs\n                    );\n                }\n            }\n        }\n'''
    return must_replace(text, old, new)


MODEL.write_text(patch_model(MODEL.read_text()))
BENCH.write_text(patch_bench(BENCH.read_text()))
print("B45 patched: layer-by-layer KV cache divergence localization")
