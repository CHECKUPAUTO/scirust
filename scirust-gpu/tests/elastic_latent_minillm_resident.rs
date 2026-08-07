#![cfg(feature = "wgpu")]

use scirust_core::nn::transformer::mini_llm::{CharTokenizer, MiniLLM, MiniLLMConfig};
use scirust_gpu::{
    WgpuLatentHeadBasis, WgpuLatentLayerBasis, WgpuResidentMiniLlm,
    WgpuResidentMiniLlmError,
};

fn identity_basis(dimension: usize) -> Vec<f32> {
    let mut basis = vec![0.0; dimension * dimension];
    for index in 0..dimension
    {
        basis[index * dimension + index] = 1.0;
    }
    basis
}

fn greedy_with_resident_runtime(
    runtime: &mut WgpuResidentMiniLlm,
    prompt: &[usize],
    max_tokens: usize,
    max_seq_len: usize,
) -> Vec<usize> {
    let mut ids = prompt.to_vec();
    if ids.is_empty()
    {
        return ids;
    }

    let mut next = 0usize;
    for (pos, &token_id) in prompt.iter().enumerate()
    {
        next = runtime.step_argmax_at(token_id, pos).unwrap();
    }

    for _ in 0..max_tokens
    {
        let pos = ids.len();
        if pos >= max_seq_len
        {
            break;
        }
        ids.push(next);
        if next == 0
        {
            break;
        }
        next = runtime.step_argmax_at(next, pos).unwrap();
    }
    ids
}

#[test]
fn resident_minillm_greedy_sequence_matches_cached_cpu_path() {
    let tokenizer = CharTokenizer::new(&["hello world abcdefghijklmnopqrstuvwxyz"]);
    let config = MiniLLMConfig {
        vocab_size: tokenizer.vocab_size,
        d_model: 8,
        n_heads: 2,
        n_layers: 2,
        d_ff: 16,
        max_seq_len: 32,
    };
    let mut cpu_model = MiniLLM::new(config.clone(), tokenizer.clone());
    let gpu_model = MiniLLM::new(config.clone(), tokenizer.clone());
    let prompt = tokenizer.encode("hello");
    let expected = cpu_model.generate_ids_cached(&prompt, 6);

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
    let snapshot = gpu_model.inference_snapshot();
    let mut resident = WgpuResidentMiniLlm::new(snapshot, config.max_seq_len, d_head, &layers)
        .expect("Phase 18 validation requires an available WGPU adapter");

    let actual = greedy_with_resident_runtime(&mut resident, &prompt, 6, config.max_seq_len);
    assert_eq!(actual, expected, "resident greedy generation diverged");

    let telemetry = resident.telemetry();
    assert_eq!(telemetry.upload_bytes_per_step, 4);
    assert_eq!(telemetry.download_bytes_per_step, 4);
    assert_eq!(telemetry.d_model, config.d_model);
    assert_eq!(telemetry.vocab_size, config.vocab_size);
    assert_eq!(telemetry.rank, d_head);
}

#[test]
fn resident_minillm_rejects_invalid_input_and_reset_reuses_storage() {
    let tokenizer = CharTokenizer::new(&["abcd"]);
    let config = MiniLLMConfig {
        vocab_size: tokenizer.vocab_size,
        d_model: 8,
        n_heads: 2,
        n_layers: 1,
        d_ff: 16,
        max_seq_len: 4,
    };
    let model = MiniLLM::new(config.clone(), tokenizer);
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
    let mut resident = WgpuResidentMiniLlm::new(
        model.inference_snapshot(),
        config.max_seq_len,
        d_head,
        &layers,
    )
    .expect("Phase 18 validation requires an available WGPU adapter");
    let resident_bytes = resident.telemetry().resident_bytes;

    let wrong_position = resident.step_argmax_at(2, 1).unwrap_err();
    assert!(matches!(
        wrong_position,
        WgpuResidentMiniLlmError::PositionMismatch {
            expected: 0,
            actual: 1
        }
    ));
    assert_eq!(resident.telemetry().steps, 0);

    let invalid_token = resident
        .step_argmax_at(config.vocab_size, 0)
        .unwrap_err();
    assert!(matches!(
        invalid_token,
        WgpuResidentMiniLlmError::TokenOutOfRange { .. }
    ));
    assert_eq!(resident.telemetry().steps, 0);

    resident.step_argmax_at(2, 0).unwrap();
    assert_eq!(resident.telemetry().steps, 1);
    assert_eq!(resident.telemetry().resident_bytes, resident_bytes);

    resident.reset().unwrap();
    let telemetry = resident.telemetry();
    assert_eq!(telemetry.steps, 0);
    assert_eq!(telemetry.resident_tokens, 0);
    assert_eq!(telemetry.resident_bytes, resident_bytes);
    assert_eq!(telemetry.upload_bytes_per_step, 4);
    assert_eq!(telemetry.download_bytes_per_step, 4);
}
