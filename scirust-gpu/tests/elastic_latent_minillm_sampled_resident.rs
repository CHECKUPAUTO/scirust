#![cfg(feature = "wgpu")]

use scirust_core::nn::sampling::SamplingConfig;
use scirust_core::nn::transformer::mini_llm::{CharTokenizer, MiniLLM, MiniLLMConfig};
use scirust_gpu::{
    WgpuLatentHeadBasis, WgpuLatentLayerBasis, WgpuResidentSampledMiniLlm,
    WgpuResidentSampledMiniLlmError,
};

fn identity_basis(dimension: usize) -> Vec<f32> {
    let mut basis = vec![0.0; dimension * dimension];
    for index in 0..dimension {
        basis[index * dimension + index] = 1.0;
    }
    basis
}

fn sampled_with_resident_runtime(
    runtime: &mut WgpuResidentSampledMiniLlm,
    prompt: &[usize],
    max_tokens: usize,
    max_seq_len: usize,
) -> Vec<usize> {
    let mut ids = prompt.to_vec();
    if ids.is_empty() {
        return ids;
    }

    for (pos, &token_id) in prompt.iter().enumerate() {
        runtime.ingest_at(token_id, pos).unwrap();
    }

    for _ in 0..max_tokens {
        let pos = ids.len();
        if pos >= max_seq_len {
            break;
        }
        let next = runtime.sample_next().unwrap();
        ids.push(next);
        if next == 0 {
            break;
        }
        runtime.ingest_at(next, pos).unwrap();
    }
    ids
}

fn parity_case(
    config: SamplingConfig,
    seed: u64,
    max_tokens: usize,
    expect_parallel: bool,
) {
    let tokenizer = CharTokenizer::new(&["hello world abcdefghijklmnopqrstuvwxyz"]);
    let model_config = MiniLLMConfig {
        vocab_size: tokenizer.vocab_size,
        d_model: 8,
        n_heads: 2,
        n_layers: 2,
        d_ff: 16,
        max_seq_len: 32,
    };
    let mut cpu_model = MiniLLM::new(model_config.clone(), tokenizer.clone());
    let gpu_model = MiniLLM::new(model_config.clone(), tokenizer.clone());
    let prompt = tokenizer.encode("hello");
    let expected = cpu_model.generate_ids_cached_sampled(&prompt, max_tokens, &config, seed);

    let d_head = model_config.d_model / model_config.n_heads;
    let basis = identity_basis(d_head);
    let heads = vec![
        WgpuLatentHeadBasis {
            key: &basis,
            value: &basis,
        };
        model_config.n_heads
    ];
    let layers = vec![WgpuLatentLayerBasis { heads: &heads }; model_config.n_layers];
    let mut resident = WgpuResidentSampledMiniLlm::new(
        gpu_model.inference_snapshot(),
        model_config.max_seq_len,
        d_head,
        &layers,
        config,
        seed,
    )
    .expect("Phase 26 validation requires an available WGPU adapter");

    assert_eq!(
        resident.uses_parallel_sampler(),
        expect_parallel,
        "Phase 26 selected the wrong resident sampling backend"
    );

    let actual =
        sampled_with_resident_runtime(&mut resident, &prompt, max_tokens, model_config.max_seq_len);
    assert_eq!(actual, expected, "resident sampled generation diverged");

    let telemetry = resident.telemetry();
    assert_eq!(telemetry.upload_bytes_per_ingest, 4);
    assert_eq!(telemetry.download_bytes_per_sample, 4);
    assert_eq!(telemetry.vocab_size, model_config.vocab_size);
    assert_eq!(telemetry.rank, d_head);
}

#[test]
fn resident_sampled_minillm_keeps_sequential_fallback_for_unbounded_sampling() {
    parity_case(
        SamplingConfig {
            temperature: 0.85,
            top_k: 0,
            top_p: 1.0,
        },
        42,
        8,
        false,
    );
}

#[test]
fn resident_sampled_minillm_promotes_exact_parallel_top_k() {
    parity_case(
        SamplingConfig {
            temperature: 1.15,
            top_k: 7,
            top_p: 0.82,
        },
        7,
        8,
        true,
    );
}

#[test]
fn prompt_priming_consumes_no_rng_and_reset_restores_seeded_stream() {
    let tokenizer = CharTokenizer::new(&["abcdefghijk"]);
    let model_config = MiniLLMConfig {
        vocab_size: tokenizer.vocab_size,
        d_model: 8,
        n_heads: 2,
        n_layers: 1,
        d_ff: 16,
        max_seq_len: 16,
    };
    let model = MiniLLM::new(model_config.clone(), tokenizer.clone());
    let prompt = tokenizer.encode("abc");
    let sampling = SamplingConfig {
        temperature: 0.9,
        top_k: 5,
        top_p: 0.9,
    };
    let seed = 123;
    let d_head = model_config.d_model / model_config.n_heads;
    let basis = identity_basis(d_head);
    let heads = vec![
        WgpuLatentHeadBasis {
            key: &basis,
            value: &basis,
        };
        model_config.n_heads
    ];
    let layers = vec![WgpuLatentLayerBasis { heads: &heads }; model_config.n_layers];
    let mut resident = WgpuResidentSampledMiniLlm::new(
        model.inference_snapshot(),
        model_config.max_seq_len,
        d_head,
        &layers,
        sampling,
        seed,
    )
    .expect("Phase 26 validation requires an available WGPU adapter");
    assert!(resident.uses_parallel_sampler());
    let resident_bytes = resident.telemetry().resident_bytes;

    for (pos, &token) in prompt.iter().enumerate() {
        resident.ingest_at(token, pos).unwrap();
    }
    let primed = resident.telemetry();
    assert_eq!(primed.sampling_draws, 0);
    assert_eq!(primed.ingested_tokens, prompt.len());
    assert!(primed.sample_ready);

    let first = resident.sample_next().unwrap();
    assert_eq!(resident.telemetry().sampling_draws, 1);
    assert!(!resident.telemetry().sample_ready);
    assert!(matches!(
        resident.sample_next().unwrap_err(),
        WgpuResidentSampledMiniLlmError::NoPendingLogits
    ));

    resident.reset().unwrap();
    let reset = resident.telemetry();
    assert_eq!(reset.sampling_draws, 0);
    assert_eq!(reset.ingested_tokens, 0);
    assert_eq!(reset.resident_tokens, 0);
    assert_eq!(reset.resident_bytes, resident_bytes);

    for (pos, &token) in prompt.iter().enumerate() {
        resident.ingest_at(token, pos).unwrap();
    }
    let replayed_first = resident.sample_next().unwrap();
    assert_eq!(replayed_first, first, "reset did not restore seeded stream");
}
