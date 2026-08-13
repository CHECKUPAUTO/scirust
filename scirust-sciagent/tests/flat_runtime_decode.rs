#![cfg(feature = "flat-attention")]

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::gpu::ResidentModel;
use scirust_sciagent::model::SciAgentModel;

fn flat_decode_config() -> SciAgentConfig {
    SciAgentConfig {
        vocab_size: 32,
        d_model: 128,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        d_ff: 256,
        max_seq_len: 24,
        rope_theta: 10_000.0,
        tie_embeddings: true,
        use_bias: false,
        eps: 1e-5,
    }
}

#[test]
fn flat_m32_prefill_and_m15_decode_match_full_greedy() {
    let config = flat_decode_config();
    let model = SciAgentModel::new(&config);
    let Some(resident) = ResidentModel::from_model(&model)
    else
    {
        eprintln!("wgpu: no adapter, skipping FLAT M15 runtime parity");
        return;
    };

    eprintln!(
        "FLAT M32 prefill + M15 decode parity on: {}",
        resident.adapter_name()
    );
    for (prompt, steps) in [
        (vec![3u32, 7, 1, 4], 6usize),
        (vec![5u32, 2, 9, 1, 7, 3, 8, 0], 5usize),
    ]
    {
        let full = resident.generate(&prompt, steps);
        let cached = resident.generate_cached(&prompt, steps);
        assert_eq!(
            cached,
            full,
            "FLAT M32 prefill + M15 cached decode must match whole-sequence greedy for prompt len {}",
            prompt.len()
        );
    }
}
