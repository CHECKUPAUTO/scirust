#![cfg(feature = "wgpu")]

use scirust_core::nn::sampling::SamplingConfig;
use scirust_core::nn::transformer::mini_llm::{CharTokenizer, MiniLLM, MiniLLMConfig};
use scirust_gpu::{WgpuLatentHeadBasis, WgpuLatentLayerBasis, WgpuResidentDeviceFeedbackMiniLlm};

fn identity_basis(dimension: usize) -> Vec<f32> {
    let mut basis = vec![0.0; dimension * dimension];
    for index in 0..dimension
    {
        basis[index * dimension + index] = 1.0;
    }
    basis
}

fn build_runtime(
    model: &MiniLLM,
    config: &MiniLLMConfig,
    layers: &[WgpuLatentLayerBasis<'_>],
    sampling: SamplingConfig,
    seed: u64,
) -> WgpuResidentDeviceFeedbackMiniLlm {
    WgpuResidentDeviceFeedbackMiniLlm::new(
        model.inference_snapshot(),
        config.max_seq_len,
        config.d_model / config.n_heads,
        layers,
        sampling,
        seed,
    )
    .expect("Phase 21 validation requires an available WGPU adapter")
}

fn parity_case(sampling: SamplingConfig, seed: u64, max_tokens: usize) {
    let tokenizer = CharTokenizer::new(&["hello world abcdefghijklmnopqrstuvwxyz"]);
    let config = MiniLLMConfig {
        vocab_size: tokenizer.vocab_size,
        d_model: 8,
        n_heads: 2,
        n_layers: 2,
        d_ff: 16,
        max_seq_len: 32,
    };
    let mut cpu = MiniLLM::new(config.clone(), tokenizer.clone());
    let gpu = MiniLLM::new(config.clone(), tokenizer.clone());
    let prompt = tokenizer.encode("hello");
    let expected = cpu.generate_ids_cached_sampled(&prompt, max_tokens, &sampling, seed);

    let d_head = config.d_model / config.n_heads;
    let basis = identity_basis(d_head);
    let heads = vec![
        WgpuLatentHeadBasis {
            key: &basis,
            value: &basis,
        };
        config.n_heads
    ];
    let layers = vec![WgpuLatentLayerBasis { heads: &heads }; config.n_layers];
    let mut resident = build_runtime(&gpu, &config, &layers, sampling, seed);

    let actual = resident
        .generate_ids_resident(&prompt, max_tokens)
        .expect("device-feedback generation must succeed");
    assert_eq!(actual, expected, "Phase 21 generated sequence diverged");

    let generated = actual.len() - prompt.len();
    let telemetry = resident.telemetry();
    assert_eq!(telemetry.generated_upload_bytes_per_token, 0);
    assert_eq!(telemetry.generated_download_bytes_per_token, 0);
    assert_eq!(telemetry.prompt_upload_bytes_per_token, 4);
    assert_eq!(telemetry.sampling_draws, generated);
    assert_eq!(
        telemetry.ingested_tokens,
        prompt.len() + generated.saturating_sub(1)
    );
    let effective_limit = max_tokens.min(config.max_seq_len - prompt.len());
    assert_eq!(
        telemetry.last_burst_readback_bytes,
        (4 + effective_limit) * core::mem::size_of::<u32>()
    );
}

#[test]
fn device_feedback_matches_seeded_temperature_generation() {
    parity_case(
        SamplingConfig {
            temperature: 0.85,
            top_k: 0,
            top_p: 1.0,
        },
        42,
        8,
    );
}

#[test]
fn device_feedback_matches_seeded_top_k_top_p_generation() {
    parity_case(
        SamplingConfig {
            temperature: 1.15,
            top_k: 7,
            top_p: 0.82,
        },
        7,
        8,
    );
}

#[test]
fn prompt_only_burst_consumes_no_rng_and_reset_replays() {
    let tokenizer = CharTokenizer::new(&["abcdefghijk"]);
    let config = MiniLLMConfig {
        vocab_size: tokenizer.vocab_size,
        d_model: 8,
        n_heads: 2,
        n_layers: 1,
        d_ff: 16,
        max_seq_len: 16,
    };
    let gpu = MiniLLM::new(config.clone(), tokenizer.clone());
    let prompt = tokenizer.encode("abc");
    let sampling = SamplingConfig {
        temperature: 0.9,
        top_k: 5,
        top_p: 0.9,
    };
    let seed = 123;
    let d_head = config.d_model / config.n_heads;
    let basis = identity_basis(d_head);
    let heads = vec![
        WgpuLatentHeadBasis {
            key: &basis,
            value: &basis,
        };
        config.n_heads
    ];
    let layers = vec![WgpuLatentLayerBasis { heads: &heads }; config.n_layers];
    let mut resident = build_runtime(&gpu, &config, &layers, sampling, seed);
    let resident_bytes = resident.telemetry().resident_bytes;

    let primed = resident.generate_ids_resident(&prompt, 0).unwrap();
    assert_eq!(primed, prompt);
    assert_eq!(resident.telemetry().sampling_draws, 0);
    assert_eq!(resident.telemetry().last_burst_readback_bytes, 0);

    let first = resident.generate_ids_resident(&prompt, 5).unwrap();
    resident.reset().unwrap();
    assert_eq!(resident.telemetry().resident_bytes, resident_bytes);
    let replay = resident.generate_ids_resident(&prompt, 5).unwrap();
    assert_eq!(replay, first, "reset did not replay the seeded burst");
}

#[test]
fn eos_stops_device_rng_and_kv_state_without_per_token_readback() {
    let tokenizer = CharTokenizer::new(&["abcdef"]);
    let config = MiniLLMConfig {
        vocab_size: tokenizer.vocab_size,
        d_model: 8,
        n_heads: 2,
        n_layers: 1,
        d_ff: 16,
        max_seq_len: 16,
    };
    let sampling = SamplingConfig {
        // A very high temperature makes the tiny test vocabulary close to
        // uniform, so a short deterministic seed search reliably finds EOS.
        temperature: 1000.0,
        top_k: 0,
        top_p: 1.0,
    };
    let prompt = tokenizer.encode("a");
    let mut cpu = MiniLLM::new(config.clone(), tokenizer.clone());
    let seed = (0u64..2048)
        .find(|&candidate| {
            let out = cpu.generate_ids_cached_sampled(&prompt, 1, &sampling, candidate);
            out.len() == prompt.len() + 1 && out.last() == Some(&0)
        })
        .expect("deterministic seed search must find an EOS draw");
    let expected = cpu.generate_ids_cached_sampled(&prompt, 8, &sampling, seed);
    assert_eq!(expected.len(), prompt.len() + 1);
    assert_eq!(expected.last(), Some(&0));

    let gpu = MiniLLM::new(config.clone(), tokenizer);
    let d_head = config.d_model / config.n_heads;
    let basis = identity_basis(d_head);
    let heads = vec![
        WgpuLatentHeadBasis {
            key: &basis,
            value: &basis,
        };
        config.n_heads
    ];
    let layers = vec![WgpuLatentLayerBasis { heads: &heads }; config.n_layers];
    let mut resident = build_runtime(&gpu, &config, &layers, sampling, seed);

    let actual = resident.generate_ids_resident(&prompt, 8).unwrap();
    assert_eq!(actual, expected);
    let telemetry = resident.telemetry();
    assert_eq!(telemetry.sampling_draws, 1, "PCG advanced after EOS");
    assert_eq!(
        telemetry.ingested_tokens,
        prompt.len(),
        "latent KV state advanced after EOS"
    );
    assert_eq!(telemetry.generated_upload_bytes_per_token, 0);
    assert_eq!(telemetry.generated_download_bytes_per_token, 0);
}
