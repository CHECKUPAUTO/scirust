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
    if "CudaCacheProjectionParity" in text:
        return text
    anchor = '''#[derive(Clone, Debug)]\npub struct CudaCacheLayerParity {\n'''
    idx = text.find(anchor)
    if idx < 0:
        raise SystemExit("missing layer parity struct")
    record = '''#[derive(Clone, Debug)]\npub struct CudaCacheProjectionParity {\n    pub component: &'static str,\n    pub rel_l2: f32,\n    pub max_abs: f32,\n}\n\n'''
    text = text[:idx] + record + text[idx:]

    marker = '''    /// Localize cached-vs-full numerical drift after each complete transformer\n'''
    method = r'''    /// Compare the first block's row-local inputs/projections under a 1×D cached
    /// decode shape against the last row of the equivalent T×D full-forward shape.
    /// If norm1 is identical but Q/K/V diverge, the error is introduced by the
    /// shape-dependent matrix multiply rather than by cache contents or RoPE.
    pub fn cache_projection_parity_teacher_forced(
        &self,
        prompt: &[u32],
        forced_tokens: &[u32],
        target_step: usize,
    ) -> Vec<CudaCacheProjectionParity> {
        assert!(target_step > 0 && target_step <= forced_tokens.len());
        let ch = &self.chain;
        let mut prefix: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt.to_vec() };
        prefix.extend_from_slice(&forced_tokens[..target_step]);
        let token = *prefix.last().expect("non-empty teacher-forced prefix");
        let pos = prefix.len() - 1;
        let b = self.blocks.first().expect("at least one block");

        let cached_x = ch.embed(&[token], &self.embedding);
        let full_x = ch.embed(&prefix, &self.embedding);
        let cached_xn = ch.rms_norm(&cached_x, &b.norm1, self.eps);
        let full_xn = ch.rms_norm(&full_x, &b.norm1, self.eps);
        let full_xn_last = ch.slice_rows(&full_xn, pos, 1);

        let cached_q = ch.matmul(&cached_xn, &b.wq);
        let cached_k = ch.matmul(&cached_xn, &b.wk);
        let cached_v = ch.matmul(&cached_xn, &b.wv);
        let full_q_all = ch.matmul(&full_xn, &b.wq);
        let full_k_all = ch.matmul(&full_xn, &b.wk);
        let full_v_all = ch.matmul(&full_xn, &b.wv);
        let full_q = ch.slice_rows(&full_q_all, pos, 1);
        let full_k = ch.slice_rows(&full_k_all, pos, 1);
        let full_v = ch.slice_rows(&full_v_all, pos, 1);

        let cached_qr = self.rope_heads(&cached_q, self.n_heads, 1, pos);
        let cached_kr = self.rope_heads(&cached_k, self.n_kv_heads, 1, pos);
        let full_qr_all = self.rope_heads(&full_q_all, self.n_heads, prefix.len(), 0);
        let full_kr_all = self.rope_heads(&full_k_all, self.n_kv_heads, prefix.len(), 0);
        let full_qr = ch.slice_rows(&full_qr_all, pos, 1);
        let full_kr = ch.slice_rows(&full_kr_all, pos, 1);

        let compare = |component: &'static str,
                       cached: &CudaMatrix,
                       full: &CudaMatrix|
         -> CudaCacheProjectionParity {
            let a = ch.download(cached);
            let b = ch.download(full);
            assert_eq!(a.len(), b.len());
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            let mut max_abs = 0.0f32;
            for (&x, &y) in a.iter().zip(&b) {
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

        vec![
            compare("norm1", &cached_xn, &full_xn_last),
            compare("q", &cached_q, &full_q),
            compare("k", &cached_k, &full_k),
            compare("v", &cached_v, &full_v),
            compare("q_rope", &cached_qr, &full_qr),
            compare("k_rope", &cached_kr, &full_kr),
        ]
    }

'''
    return must_replace(text, marker, method + marker)


def patch_bench(text: str) -> str:
    if "SCIAGENT_THOR_KV_PROJECTION" in text:
        return text
    old = '''            if first.step > 0\n            {\n                for layer in cuda.cache_layer_parity_teacher_forced(&prompt, forced, first.step)\n                {\n'''
    new = '''            if first.step > 0\n            {\n                for component in\n                    cuda.cache_projection_parity_teacher_forced(&prompt, forced, first.step)\n                {\n                    println!(\n                        "SCIAGENT_THOR_KV_PROJECTION step={} component={} rel_l2={:.8e} max_abs={:.8e}",\n                        first.step, component.component, component.rel_l2, component.max_abs\n                    );\n                }\n                for layer in cuda.cache_layer_parity_teacher_forced(&prompt, forced, first.step)\n                {\n'''
    return must_replace(text, old, new)


MODEL.write_text(patch_model(MODEL.read_text()))
BENCH.write_text(patch_bench(BENCH.read_text()))
print("B46 patched: first-block projection shape parity trace")
