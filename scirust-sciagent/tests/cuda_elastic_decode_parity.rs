//! Full-rank identity gate for I250-B reconstruction-free ElasticKV CUDA decode.
#![cfg(feature = "cuda")]

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::cuda_elastic_decode::{CudaElasticDecodeError, CudaElasticDecodeModel};
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

fn elastic_or_skip(model: &SciAgentModel) -> Option<CudaElasticDecodeModel> {
    match CudaElasticDecodeModel::from_model_identity(model)
    {
        Ok(runtime) => Some(runtime),
        Err(CudaElasticDecodeError::RuntimeUnavailable) =>
        {
            eprintln!("cuda: Elastic decode runtime unavailable, skipping");
            None
        },
        Err(error) => panic!("Elastic identity decode construction failed: {error}"),
    }
}

#[test]
fn full_rank_elastic_decode_matches_b49_greedy_stream() {
    let config = tiny_tied();
    let model = SciAgentModel::new(&config);
    let Some(oracle) = CudaModel::from_model(&model)
    else
    {
        eprintln!("cuda: no Route-B device, skipping Elastic parity");
        return;
    };
    let Some(elastic) = elastic_or_skip(&model)
    else
    {
        return;
    };
    assert!(elastic.is_dense_equivalent_identity());

    let params = greedy();
    for (prompt, max_new, seed) in [
        (vec![3u32, 5, 7, 11], 6usize, 0xE250u64),
        (vec![1u32], 8usize, 0xE251u64),
        (vec![9u32, 2, 17, 4, 6, 8], 5usize, 0xE252u64),
    ]
    {
        let expected = oracle.generate_cached(&prompt, max_new, &params, seed);
        let actual = elastic.generate(&prompt, max_new, &params, seed);
        assert_eq!(
            actual, expected,
            "full-rank Elastic CUDA decode diverged for prompt {prompt:?}"
        );
    }
}

#[test]
fn full_rank_elastic_decode_preserves_empty_prompt_semantics() {
    let config = tiny_tied();
    let model = SciAgentModel::new(&config);
    let Some(oracle) = CudaModel::from_model(&model)
    else
    {
        eprintln!("cuda: no Route-B device, skipping Elastic empty-prompt parity");
        return;
    };
    let Some(elastic) = elastic_or_skip(&model)
    else
    {
        return;
    };
    let params = greedy();
    assert_eq!(
        elastic.generate(&[], 4, &params, 0xE253),
        oracle.generate_cached(&[], 4, &params, 0xE253)
    );
}

#[test]
fn reduced_native_pair_elastic_decode_replays_deterministically() {
    let config = tiny_tied();
    let model = SciAgentModel::new(&config);
    let reduced = match CudaElasticDecodeModel::from_model_native_pair_prefix(&model, 4, 4)
    {
        Ok(runtime) => runtime,
        Err(CudaElasticDecodeError::RuntimeUnavailable) =>
        {
            eprintln!("cuda: Elastic decode runtime unavailable, skipping reduced replay");
            return;
        },
        Err(error) => panic!("reduced Elastic decode construction failed: {error}"),
    };
    assert_eq!(reduced.key_rank(), 4);
    assert_eq!(reduced.value_rank(), 4);
    assert!(!reduced.is_dense_equivalent_identity());

    let params = greedy();
    let prompt = [4u32, 3, 9, 1];
    let first = reduced.generate(&prompt, 6, &params, 0xE254);
    let second = reduced.generate(&prompt, 6, &params, 0xE254);
    assert_eq!(first, second);
}
