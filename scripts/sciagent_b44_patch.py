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
    # B40's first transformation inserted the fused backward helper while leaving the
    # historical per-head helper behind. Remove exactly that obsolete duplicate.
    duplicate = '''    fn rope_heads_backward(\n        &self,\n        dy: &CudaMatrix,\n        n_heads: usize,\n        seq_len: usize,\n        offset: usize,\n    ) -> CudaMatrix {\n        assert!(n_heads > 0 && dy.cols().is_multiple_of(n_heads));\n        let dh = dy.cols() / n_heads;\n        let mut heads = Vec::with_capacity(n_heads);\n        for head in 0..n_heads\n        {\n            let raw = self.chain.slice_cols(dy, head * dh, dh);\n            heads.push(self.chain.rope_backward(&raw, seq_len, offset, self.theta));\n        }\n        let refs: Vec<&CudaMatrix> = heads.iter().collect();\n        self.chain.concat_cols(&refs)\n    }\n\n'''
    if duplicate in text:
        text = must_replace(text, duplicate, "")
    elif text.count("    fn rope_heads_backward(\n") != 1:
        raise SystemExit("unexpected rope_heads_backward state")

    if "pub struct CudaCacheParityStep" not in text:
        anchor = '''/// A [`SciAgentModel`] mirrored into VRAM as bf16 matrices, running the whole\n/// decoder forward on the Tensor-core [`CudaChain`]. Tied-embedding models only.\npub struct CudaModel {\n'''
        diag = '''/// Teacher-forced diagnostic comparing the cached decoder's prediction logits with\n/// a full-sequence forward under the exact same prefix. This is intentionally a\n/// numerical diagnostic rather than a relaxed correctness criterion: production\n/// greedy parity remains strict until the measured cause of any mismatch is known.\n#[derive(Clone, Debug)]\npub struct CudaCacheParityStep {\n    pub step: usize,\n    pub expected_token: u32,\n    pub cached_top1: u32,\n    pub full_top1: u32,\n    pub cached_margin: f32,\n    pub full_margin: f32,\n    pub rel_l2: f32,\n    pub max_abs: f32,\n}\n\n'''
        text = must_replace(text, anchor, diag + anchor)

    if "pub fn cache_parity_teacher_forced" not in text:
        marker = '''    /// Backward of [`Self::attention`] (the GQA analogue of Route A's\n'''
        method = r'''    /// Compare KV-cached logits against full-forward logits while forcing both paths
    /// through exactly the same continuation tokens. One record is emitted for every
    /// forced token *before* that token is appended, so record 0 compares prompt
    /// prefill and record N compares the two paths after N identical decode tokens.
    /// This isolates cache arithmetic from autoregressive error amplification.
    pub fn cache_parity_teacher_forced(
        &self,
        prompt: &[u32],
        forced_tokens: &[u32],
    ) -> Vec<CudaCacheParityStep> {
        let mut prefix: Vec<u32> = if prompt.is_empty() {
            vec![0]
        } else {
            prompt.to_vec()
        };
        if forced_tokens.is_empty() {
            return Vec::new();
        }

        let mut kcache: Vec<Option<CudaMatrix>> =
            (0..self.blocks.len()).map(|_| None).collect();
        let mut vcache: Vec<Option<CudaMatrix>> =
            (0..self.blocks.len()).map(|_| None).collect();
        let mut cached_logits = self.prefill_cached(&prefix, &mut kcache, &mut vcache);
        let mut out = Vec::with_capacity(forced_tokens.len());

        let top2 = |row: &[f32]| -> (u32, f32) {
            assert!(!row.is_empty(), "top2 requires a non-empty logit row");
            let mut best_i = 0usize;
            let mut best = row[0];
            let mut second = f32::NEG_INFINITY;
            for (i, &value) in row.iter().enumerate().skip(1) {
                if value > best {
                    second = best;
                    best = value;
                    best_i = i;
                } else if value > second {
                    second = value;
                }
            }
            (best_i as u32, best - second)
        };

        for (step, &expected_token) in forced_tokens.iter().enumerate() {
            let full = self.forward(&prefix);
            let full_last = &full[full.len() - self.vocab..];
            assert_eq!(cached_logits.len(), self.vocab, "cached logit width");

            let mut num = 0.0f64;
            let mut den = 0.0f64;
            let mut max_abs = 0.0f32;
            for (&cached, &reference) in cached_logits.iter().zip(full_last) {
                let delta = cached - reference;
                num += (delta as f64) * (delta as f64);
                den += (reference as f64) * (reference as f64);
                max_abs = max_abs.max(delta.abs());
            }
            let rel_l2 = (num.sqrt() / den.sqrt().max(1e-30)) as f32;
            let (cached_top1, cached_margin) = top2(&cached_logits);
            let (full_top1, full_margin) = top2(full_last);
            out.push(CudaCacheParityStep {
                step,
                expected_token,
                cached_top1,
                full_top1,
                cached_margin,
                full_margin,
                rel_l2,
                max_abs,
            });

            if step + 1 < forced_tokens.len() {
                let pos = prefix.len();
                prefix.push(expected_token);
                cached_logits =
                    self.decode_step_cached(expected_token, pos, &mut kcache, &mut vcache);
            }
        }
        out
    }

'''
        text = must_replace(text, marker, method + marker)
    return text


def patch_bench(text: str) -> str:
    if "SCIAGENT_THOR_KV_PARITY" in text:
        return text
    old = '''    if !parity\n    {\n        eprintln!("ERROR: cached CUDA decoding changed greedy tokens");\n        std::process::exit(3);\n    }\n'''
    new = '''    if !parity\n    {\n        // Diagnose under the naïve greedy continuation, so cached and full-forward\n        // paths see identical prefixes. This distinguishes a cache-logic error from\n        // a tiny bf16/cuBLASLt rank flip that autoregressive generation amplifies.\n        let forced = &naive[prompt.len()..];\n        let report = cuda.cache_parity_teacher_forced(&prompt, forced);\n        for row in &report\n        {\n            println!(\n                "SCIAGENT_THOR_KV_PARITY step={} expected={} cached_top1={} full_top1={} cached_margin={:.8e} full_margin={:.8e} rel_l2={:.8e} max_abs={:.8e}",\n                row.step,\n                row.expected_token,\n                row.cached_top1,\n                row.full_top1,\n                row.cached_margin,\n                row.full_margin,\n                row.rel_l2,\n                row.max_abs\n            );\n        }\n        if let Some(first) = report.iter().find(|row| row.cached_top1 != row.full_top1)\n        {\n            eprintln!(\n                "ERROR: cached CUDA decoding first teacher-forced top-1 divergence at step {} (cached={} full={}, rel_l2={:.3e}, max_abs={:.3e})",\n                first.step, first.cached_top1, first.full_top1, first.rel_l2, first.max_abs\n            );\n        }\n        else\n        {\n            eprintln!(\n                "ERROR: autoregressive sequence diverged although teacher-forced raw top-1 stayed equal; inspect sampling/repetition state"\n            );\n        }\n        std::process::exit(3);\n    }\n'''
    return must_replace(text, old, new)


MODEL.write_text(patch_model(MODEL.read_text()))
BENCH.write_text(patch_bench(BENCH.read_text()))
print("B44 patched: remove duplicate RoPE backward + teacher-forced KV diagnostics")
