#![cfg(feature = "wgpu")]

use scirust_core::nn::rng::PcgEngine;
use scirust_core::nn::sampling::{SamplingConfig, sample_token};
use scirust_gpu::{WgpuDeterministicSampler, WgpuDeterministicSamplerError};

fn assert_stream_matches(config: SamplingConfig, seed: u64, sequences: &[Vec<f32>]) {
    let vocab_size = sequences[0].len();
    let mut cpu_rng = PcgEngine::new(seed);
    let mut gpu = WgpuDeterministicSampler::new(vocab_size, config, seed)
        .expect("deterministic sampler validation requires an available WGPU adapter");

    for (draw, logits) in sequences.iter().enumerate()
    {
        let expected = sample_token(logits, &config, &mut cpu_rng);
        let actual = gpu.sample(logits).unwrap();
        assert_eq!(actual, expected, "sampling diverged at draw {draw}");
    }
}

#[test]
fn seeded_temperature_stream_matches_cpu_sampler() {
    let sequences: Vec<Vec<f32>> = (0..32)
        .map(|draw| {
            (0..9)
                .map(|token| {
                    let phase = (draw * 9 + token) as f32 * 0.173;
                    phase.sin() * 1.7 + phase.cos() * 0.3
                })
                .collect()
        })
        .collect();
    let config = SamplingConfig {
        temperature: 0.85,
        top_k: 0,
        top_p: 1.0,
    };
    assert_stream_matches(config, 42, &sequences);
}

#[test]
fn seeded_top_k_top_p_stream_matches_cpu_sampler() {
    let sequences: Vec<Vec<f32>> = (0..32)
        .map(|draw| {
            (0..11)
                .map(|token| {
                    let phase = (draw * 7 + token * 3) as f32 * 0.119;
                    phase.cos() * 2.1 - phase.sin() * 0.2
                })
                .collect()
        })
        .collect();
    let config = SamplingConfig {
        temperature: 1.15,
        top_k: 5,
        top_p: 0.82,
    };
    assert_stream_matches(config, 7, &sequences);
}

#[test]
fn bounded_top_k_reports_only_k_ranking_passes() {
    let config = SamplingConfig {
        temperature: 0.9,
        top_k: 7,
        top_p: 0.91,
    };
    let gpu = WgpuDeterministicSampler::new(257, config, 17)
        .expect("Phase 22 validation requires an available WGPU adapter");
    assert!(gpu.uses_bounded_top_k_fast_path());
    assert_eq!(gpu.ranking_passes_per_sample(), 7);
}

#[test]
fn bounded_top_k_ties_and_nucleus_match_cpu_order_exactly() {
    let vocab_size = 32;
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 4,
        top_p: 0.74,
    };
    let logits = vec![0.0; vocab_size];
    let sequences = vec![logits; 24];

    // Equal probabilities exercise the stable CPU tie-break: lower token ids
    // must occupy the selected top-k prefix. top_p then narrows that prefix.
    assert_stream_matches(config, 0x5eed, &sequences);

    let mut gpu = WgpuDeterministicSampler::new(vocab_size, config, 0x5eed)
        .expect("Phase 22 validation requires an available WGPU adapter");
    assert!(gpu.uses_bounded_top_k_fast_path());
    assert_eq!(gpu.ranking_passes_per_sample(), 4);
    for _ in 0..16
    {
        let token = gpu.sample(&vec![0.0; vocab_size]).unwrap();
        assert!(token < 3, "top-k/top-p tie ordering admitted token {token}");
    }
}

#[test]
fn unbounded_and_non_restrictive_top_k_keep_full_ranking_fallback() {
    let unbounded = SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 0.9,
    };
    let gpu = WgpuDeterministicSampler::new(29, unbounded, 3)
        .expect("Phase 22 validation requires an available WGPU adapter");
    assert!(!gpu.uses_bounded_top_k_fast_path());
    assert_eq!(gpu.ranking_passes_per_sample(), 29);

    let non_restrictive = SamplingConfig {
        temperature: 1.0,
        top_k: 29,
        top_p: 0.9,
    };
    let gpu = WgpuDeterministicSampler::new(29, non_restrictive, 3)
        .expect("Phase 22 validation requires an available WGPU adapter");
    assert!(!gpu.uses_bounded_top_k_fast_path());
    assert_eq!(gpu.ranking_passes_per_sample(), 29);

    let greedy = WgpuDeterministicSampler::new(29, SamplingConfig::greedy(), 3)
        .expect("Phase 22 validation requires an available WGPU adapter");
    assert!(!greedy.uses_bounded_top_k_fast_path());
    assert_eq!(greedy.ranking_passes_per_sample(), 0);
}

#[test]
fn greedy_tie_break_matches_sampling_argmax_without_consuming_rng() {
    let config = SamplingConfig::greedy();
    let logits = [2.0, 5.0, 5.0, 1.0];
    let mut gpu = WgpuDeterministicSampler::new(logits.len(), config, 99)
        .expect("deterministic sampler validation requires an available WGPU adapter");

    for _ in 0..4
    {
        assert_eq!(gpu.sample(&logits).unwrap(), 1);
    }
    let telemetry = gpu.telemetry();
    assert_eq!(telemetry.draws, 0);
    assert_eq!(telemetry.upload_bytes_per_sample, logits.len() * 4);
    assert_eq!(telemetry.download_bytes_per_sample, 4);
}

#[test]
fn sampler_rejects_bad_shapes_and_non_finite_logits_without_advancing() {
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
    };
    let mut gpu = WgpuDeterministicSampler::new(4, config, 5)
        .expect("deterministic sampler validation requires an available WGPU adapter");

    let shape = gpu.sample(&[1.0, 2.0]).unwrap_err();
    assert!(matches!(
        shape,
        WgpuDeterministicSamplerError::LogitLength {
            expected: 4,
            actual: 2
        }
    ));
    let non_finite = gpu.sample(&[1.0, f32::NAN, 2.0, 3.0]).unwrap_err();
    assert!(matches!(
        non_finite,
        WgpuDeterministicSamplerError::NonFiniteLogit { index: 1 }
    ));
    assert_eq!(gpu.telemetry().draws, 0);
}
