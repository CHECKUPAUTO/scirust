//! I250 CUDA decode parity gates.
#![cfg(feature = "cuda")]

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::cuda_decode::{
    CudaDecodeDownMode, CudaDecodeFfnMode, CudaDecodeLmHeadMode, CudaDecodeModel, CudaDecodeModes,
};
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
fn canonical_cuda_decode_and_device_feedback_match_b49_greedy() {
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
        let host = fast.generate(&prompt, max_new, &params, seed);
        let device = fast.generate_greedy_device_feedback(&prompt, max_new);
        let ffn_baseline = fast.generate_greedy_device_feedback_with_modes(
            &prompt,
            max_new,
            CudaDecodeModes {
                ffn: CudaDecodeFfnMode::CublasLt,
                down: CudaDecodeDownMode::CublasLt,
                lm_head: CudaDecodeLmHeadMode::FusedArgmax,
            },
        );
        let lm_baseline = fast.generate_greedy_device_feedback_with_modes(
            &prompt,
            max_new,
            CudaDecodeModes {
                ffn: CudaDecodeFfnMode::CublasLt,
                down: CudaDecodeDownMode::CublasLt,
                lm_head: CudaDecodeLmHeadMode::FullLogits,
            },
        );
        assert_eq!(host, expected, "host-sampler I250 diverged for {prompt:?}");
        assert_eq!(
            device, expected,
            "device-feedback I250 diverged for {prompt:?}"
        );
        assert_eq!(
            ffn_baseline, expected,
            "cuBLASLt FFN baseline diverged for {prompt:?}"
        );
        assert_eq!(
            lm_baseline, expected,
            "full-logits LM baseline diverged for {prompt:?}"
        );
        let down_candidate = fast.generate_greedy_device_feedback_with_modes(
            &prompt,
            max_new,
            CudaDecodeModes {
                ffn: CudaDecodeFfnMode::CublasLt,
                down: CudaDecodeDownMode::TiledGemv,
                lm_head: CudaDecodeLmHeadMode::FusedArgmax,
            },
        );
        assert_eq!(
            down_candidate, expected,
            "tiled down candidate diverged for {prompt:?}"
        );
    }
}

#[test]
fn device_feedback_preserves_empty_prompt_semantics() {
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
    let expected = oracle.generate_cached(&[], 4, &params, 0x4250);
    assert_eq!(fast.generate(&[], 4, &params, 0x4250), expected);
    assert_eq!(fast.generate_greedy_device_feedback(&[], 4), expected);
}

#[test]
fn device_feedback_zero_generation_matches_prompt_normalization() {
    let config = tiny_tied();
    let model = SciAgentModel::new(&config);
    let Some(fast) = CudaDecodeModel::from_model(&model)
    else
    {
        eprintln!("cuda: fast decode runtime unavailable, skipping zero-generation gate");
        return;
    };
    assert_eq!(fast.generate_greedy_device_feedback(&[3, 4], 0), vec![3, 4]);
    assert_eq!(fast.generate_greedy_device_feedback(&[], 0), vec![0]);
}
