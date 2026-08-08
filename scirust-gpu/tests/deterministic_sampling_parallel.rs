#![cfg(feature = "wgpu")]

use scirust_core::nn::rng::PcgEngine;
use scirust_core::nn::sampling::{SamplingConfig, sample_token};
use scirust_gpu::{PARALLEL_TOP_K_LANES, WgpuDeterministicSampler, WgpuParallelTopKSampler};

fn logits(draw: usize, vocab: usize) -> Vec<f32> {
    (0..vocab)
        .map(|token| {
            let x = (draw * 17 + token * 5) as f32 * 0.071;
            x.sin() * 1.9 + x.cos() * 0.37 + (token % 11) as f32 * 0.002
        })
        .collect()
}

#[test]
fn parallel_matches_cpu_and_sequential_wgpu() {
    let vocab = 67;
    let config = SamplingConfig {
        temperature: 1.1,
        top_k: 7,
        top_p: 0.83,
    };
    let seed = 42;
    let mut cpu = PcgEngine::new(seed);
    let mut sequential = WgpuDeterministicSampler::new(vocab, config, seed).unwrap();
    let mut parallel = WgpuParallelTopKSampler::new(vocab, config, seed).unwrap();

    for draw in 0..24
    {
        let values = logits(draw, vocab);
        let expected = sample_token(&values, &config, &mut cpu);
        assert_eq!(parallel.sample(&values).unwrap(), expected);
        assert_eq!(sequential.sample(&values).unwrap(), expected);
    }

    assert_eq!(parallel.ranking_lanes_per_sample(), PARALLEL_TOP_K_LANES);
    assert_eq!(parallel.ranking_passes_per_sample(), 7);
    assert_eq!(parallel.telemetry().draws, 24);
}

#[test]
fn ties_and_reset_preserve_exact_stream() {
    let vocab = 64;
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 4,
        top_p: 0.74,
    };
    let seed = 0x5eed;
    let values = vec![0.0; vocab];
    let mut parallel = WgpuParallelTopKSampler::new(vocab, config, seed).unwrap();
    let resident = parallel.telemetry().resident_bytes;
    let first: Vec<_> = (0..16).map(|_| parallel.sample(&values).unwrap()).collect();
    assert!(first.iter().all(|&token| token < 3));

    parallel.reset().unwrap();
    let second: Vec<_> = (0..16).map(|_| parallel.sample(&values).unwrap()).collect();
    assert_eq!(first, second);
    assert_eq!(parallel.telemetry().resident_bytes, resident);
}
