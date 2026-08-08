//! I250 CUDA decode parity gates.
//!
//! The latency-oriented decoder is intentionally separate from Route B.  These
//! tests keep the historical cached decoder as the oracle and require greedy token
//! streams to remain identical before the fast path can be promoted.
#![cfg(feature = "cuda")]

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::cuda_decode::CudaDecodeModel;
use scirust_sciagent::cuda_model::CudaModel;
use scirust_sciagent::generate::SamplingParams;
use scirust_sciagent::model::SciAgentModel;

fn tiny_tied() -> SciAgentConfig {
    SciAgentConfig {
        vocab_size: 48,
        d_model: 32,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        d_ff: 64,
        max_seq_len: 32,
        rope_theta: 10_000.0,
        tie_embeddings: true,
        use_bias: false,
        eps: 1e-5,
    }
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        repetition_penalty: 1.0,
        repetition_window: 64,
    }
}

#[test]
fn fused_cuda_decode_matches_b49_cached_greedy() {
    let config = tiny_tied();
    let model = SciAgentModel::new(&config);
    let Some(oracle) = CudaModel::from_model(&model)
    else
    {
        eprintln!("cuda: no device, skipping I250 decode parity");
        return;
    };
    let Some(fast) = CudaDecodeModel::from_model(&model)
    else
    {
        eprintln!("cuda: fast decode runtime unavailable, skipping I250 decode parity");
        return;
    };

    let params = greedy();
    for (prompt, max_new, seed) in [
        (vec![3u32, 5, 7, 11], 6usize, 0x1250u64),
        (vec![1u32], 8usize, 0x2250u64),
        (vec![9u32, 2, 17, 4, 6, 8], 5usize, 0x3250u64),
    ]
    {
        let expected = oracle.generate_cached(&prompt, max_new, &params, seed);
        let got = fast.generate(&prompt, max_new, &params, seed);
        assert_eq!(
            got, expected,
            "fused CUDA decode diverged from B49 cached oracle for prompt {prompt:?}"
        );
    }
}

#[test]
fn fused_cuda_decode_preserves_empty_prompt_semantics() {
    let config = tiny_tied();
    let model = SciAgentModel::new(&config);
    let Some(oracle) = CudaModel::from_model(&model)
    else
    {
        eprintln!("cuda: no device, skipping I250 empty-prompt parity");
        return;
    };
    let Some(fast) = CudaDecodeModel::from_model(&model)
    else
    {
        eprintln!("cuda: fast decode runtime unavailable, skipping I250 empty-prompt parity");
        return;
    };
    let params = greedy();
    assert_eq!(
        fast.generate(&[], 4, &params, 0x4250),
        oracle.generate_cached(&[], 4, &params, 0x4250)
    );
}
