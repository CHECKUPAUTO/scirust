#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"
TEST = ROOT / "scirust-sciagent/tests/cuda_parity.rs"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:120]!r}")
    return text.replace(old, new, count)


CACHE_METHODS = r'''
    /// Incremental single-query GQA over already-RoPE'd resident keys and raw values.
    /// `qr` is one row at the new token's absolute position; cached keys/values contain
    /// every visible position, so no causal mask is required here.
    fn incremental_attention(
        &self,
        qr: &CudaMatrix,
        kcache: &CudaMatrix,
        vcache: &CudaMatrix,
    ) -> CudaMatrix {
        let ch = &self.chain;
        let dh = self.d_model / self.n_heads;
        let repeat = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut heads = Vec::with_capacity(self.n_heads);
        for head in 0..self.n_heads {
            let kv = head / repeat;
            let qs = ch.slice_cols(qr, head * dh, dh);
            let ks = ch.slice_cols(kcache, kv * dh, dh);
            let vs = ch.slice_cols(vcache, kv * dh, dh);
            let scores = ch.matmul_bt(&qs, &ks);
            let scaled = ch.scale_causal_mask(&scores, scale, false);
            let weights = ch.softmax(&scaled);
            heads.push(ch.matmul(&weights, &vs));
        }
        let refs: Vec<&CudaMatrix> = heads.iter().collect();
        ch.concat_cols(&refs)
    }

    /// Wide prompt prefill for the CUDA KV-cache path. The prompt traverses the
    /// model once with normal causal attention; each layer keeps its RoPE'd K and
    /// raw V matrices resident for subsequent one-token decode steps. Only the last
    /// prompt position's vocab row is downloaded.
    fn prefill_cached(
        &self,
        prompt: &[u32],
        kcache: &mut [Option<CudaMatrix>],
        vcache: &mut [Option<CudaMatrix>],
    ) -> Vec<f32> {
        debug_assert!(!prompt.is_empty());
        debug_assert_eq!(kcache.len(), self.blocks.len());
        debug_assert_eq!(vcache.len(), self.blocks.len());
        let ch = &self.chain;
        let p = prompt.len();
        let mut x = ch.embed(prompt, &self.embedding);
        for (layer, b) in self.blocks.iter().enumerate() {
            let xn = ch.rms_norm(&x, &b.norm1, self.eps);
            let q = ch.matmul(&xn, &b.wq);
            let k = ch.matmul(&xn, &b.wk);
            let v = ch.matmul(&xn, &b.wv);
            let ctx = self.attention(&q, &k, &v);
            // `attention` computes the same K RoPE internally; retain an identical
            // resident copy here so future queries can attend without re-prefill.
            kcache[layer] = Some(ch.rope(&k, p, 0, self.theta));
            vcache[layer] = Some(v);
            let attn_out = ch.matmul(&ctx, &b.wo);
            let h = ch.add(&x, &attn_out);
            let hn = ch.rms_norm(&h, &b.norm2, self.eps);
            let gate = ch.matmul(&hn, &b.wg);
            let up = ch.matmul(&hn, &b.wu);
            let act = ch.swiglu(&gate, &up);
            let mlp = ch.matmul(&act, &b.wd);
            x = ch.add(&h, &mlp);
        }
        let normed = ch.rms_norm(&x, &self.final_norm, self.eps);
        let logits = ch.matmul_bt(&normed, &self.embedding);
        let last = ch.slice_rows(&logits, p - 1, 1);
        ch.download(&last)
    }

    /// Process exactly one newly generated token at absolute position `pos`, append
    /// its per-layer RoPE'd K/raw V rows to the resident caches, and return the
    /// vocab logits predicting the following token. The trunk is O(1) rows; only
    /// attention grows with context length, replacing the old full-sequence O(n^2)
    /// recomputation at every decode step.
    fn decode_step_cached(
        &self,
        token: u32,
        pos: usize,
        kcache: &mut [Option<CudaMatrix>],
        vcache: &mut [Option<CudaMatrix>],
    ) -> Vec<f32> {
        let ch = &self.chain;
        let mut x = ch.embed(&[token], &self.embedding);
        for (layer, b) in self.blocks.iter().enumerate() {
            let xn = ch.rms_norm(&x, &b.norm1, self.eps);
            let q = ch.matmul(&xn, &b.wq);
            let k = ch.matmul(&xn, &b.wk);
            let v = ch.matmul(&xn, &b.wv);
            let qr = ch.rope(&q, 1, pos, self.theta);
            let kr = ch.rope(&k, 1, pos, self.theta);

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
        }
        let normed = ch.rms_norm(&x, &self.final_norm, self.eps);
        let logits = ch.matmul_bt(&normed, &self.embedding);
        ch.download(&logits)
    }

    /// KV-cached autoregressive generation on CUDA Tensor cores. Sampling semantics
    /// are identical to [`Self::generate`]: same shared sampler/RNG, repetition
    /// context and token-0 early stop. The original non-cached method remains as an
    /// independent parity reference.
    pub fn generate_cached(
        &self,
        prompt: &[u32],
        max_new: usize,
        params: &SamplingParams,
        seed: u64,
    ) -> Vec<u32> {
        let mut tokens: Vec<u32> = if prompt.is_empty() {
            vec![0]
        } else {
            prompt.to_vec()
        };
        if max_new == 0 {
            return tokens;
        }
        let mut kcache: Vec<Option<CudaMatrix>> =
            (0..self.blocks.len()).map(|_| None).collect();
        let mut vcache: Vec<Option<CudaMatrix>> =
            (0..self.blocks.len()).map(|_| None).collect();
        let mut logits = self.prefill_cached(&tokens, &mut kcache, &mut vcache);
        let mut rng = seed_to_state(seed);

        for i in 0..max_new {
            let recent: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
            let next = sample_row(&logits, params, &recent, &mut rng) as u32;
            let pos = tokens.len();
            tokens.push(next);
            if next == 0 || i + 1 == max_new {
                break;
            }
            logits = self.decode_step_cached(next, pos, &mut kcache, &mut vcache);
        }
        tokens
    }

'''


def patch_model(text: str) -> str:
    if "pub fn generate_cached(" in text:
        raise SystemExit("CUDA cached generation already present")
    marker = "    /// Backward of [`Self::attention`] (the GQA analogue of Route A's\n"
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("missing insertion point before backward")
    return text[:pos] + CACHE_METHODS + text[pos:]


TEST_SRC = r'''

/// B31: CUDA KV-cached greedy decoding must reproduce the original non-cached
/// CUDA decoder token-for-token. This pins prompt prefill, absolute-position RoPE,
/// per-layer K/V cache growth and incremental GQA as one end-to-end contract.
#[test]
fn cuda_cached_generation_matches_naive_greedy() {
    let config = tiny_tied();
    let model = SciAgentModel::new(&config);
    let Some(cm) = CudaModel::from_model(&model) else {
        eprintln!("cuda: no device, skipping cached-generation parity");
        return;
    };
    let prompt = vec![3u32, 5, 7, 11];
    let params = scirust_sciagent::generate::SamplingParams::default();
    let naive = cm.generate(&prompt, 6, &params, 0xB31);
    let cached = cm.generate_cached(&prompt, 6, &params, 0xB31);
    assert_eq!(cached, naive, "CUDA KV cache changed greedy decode tokens");
}

/// Empty-prompt behavior is part of the existing CUDA generation API: generation
/// starts from token 0. The cached path must preserve that behavior exactly.
#[test]
fn cuda_cached_generation_preserves_empty_prompt_semantics() {
    let config = tiny_tied();
    let model = SciAgentModel::new(&config);
    let Some(cm) = CudaModel::from_model(&model) else {
        eprintln!("cuda: no device, skipping cached empty-prompt parity");
        return;
    };
    let params = scirust_sciagent::generate::SamplingParams::default();
    assert_eq!(
        cm.generate_cached(&[], 3, &params, 7),
        cm.generate(&[], 3, &params, 7)
    );
}
'''


def patch_test(text: str) -> str:
    if "cuda_cached_generation_matches_naive_greedy" in text:
        raise SystemExit("tests already B31 patched")
    return text.rstrip() + TEST_SRC + "\n"


MODEL.write_text(patch_model(MODEL.read_text()))
TEST.write_text(patch_test(TEST.read_text()))
print("patched B31 CUDA KV cache")
