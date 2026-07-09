//! **Speculative-decode benchmark** — measure the resident speculative decoder on
//! real hardware: acceptance rate, actual tok/s vs plain greedy, and the
//! cost-model speedup / break-even acceptance.
//!
//! Speculative decoding is **exact** — its output equals plain greedy regardless
//! of the draft (asserted here). The *speedup* depends entirely on how often the
//! draft's guess matches the target's greedy pick, which needs a draft **trained
//! to agree** with the target. These weights are random, so measured acceptance is
//! ~chance and the draft/verify overhead dominates — the honest result is the
//! **cost model**: how cheap the draft is, and the acceptance a real draft would
//! need to pay off. All costs are weight-independent (identical FLOPs).
//!
//! Size via env: `SCIAGENT_D_MODEL` (512), `SCIAGENT_LAYERS` (8),
//! `SCIAGENT_DRAFT_LAYERS` (2), `SCIAGENT_FF` (1408), `SCIAGENT_HEADS` (8),
//! `SCIAGENT_KV_HEADS` (2), `SCIAGENT_MAX_SEQ` (1024), `SCIAGENT_DECODE_N` (48).
//!
//! ```text
//! SCIAGENT_D_MODEL=1024 SCIAGENT_LAYERS=24 SCIAGENT_DRAFT_LAYERS=4 SCIAGENT_FF=4096 \
//!   cargo run -p scirust-sciagent --features gpu --release --example resident_spec_bench
//! ```
//!
//! Exit code 2 means no GPU adapter was found — run on the Thor or install a
//! Vulkan ICD.

use std::time::Instant;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::gpu::ResidentModel;
use scirust_sciagent::model::SciAgentModel;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn prompt_of(len: usize, vocab: usize) -> Vec<u32> {
    (0..len)
        .map(|i| (i as u32 * 7 + 1) % vocab as u32)
        .collect()
}

fn ms_of(f: impl FnOnce()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let vocab = 256usize;
    let target_cfg = SciAgentConfig {
        vocab_size: vocab,
        d_model: env_usize("SCIAGENT_D_MODEL", 512),
        n_layers: env_usize("SCIAGENT_LAYERS", 8),
        n_heads: env_usize("SCIAGENT_HEADS", 8),
        n_kv_heads: env_usize("SCIAGENT_KV_HEADS", 2),
        d_ff: env_usize("SCIAGENT_FF", 1408),
        max_seq_len: env_usize("SCIAGENT_MAX_SEQ", 1024),
        rope_theta: 10_000.0,
        tie_embeddings: true,
        use_bias: false,
        eps: 1e-5,
    };
    let draft_cfg = SciAgentConfig {
        n_layers: env_usize("SCIAGENT_DRAFT_LAYERS", 2),
        ..target_cfg.clone()
    };

    let target_model = SciAgentModel::new(&target_cfg);
    let draft_model = SciAgentModel::new(&draft_cfg);
    let (Some(target), Some(draft)) = (
        ResidentModel::from_model(&target_model),
        ResidentModel::from_model(&draft_model),
    )
    else
    {
        eprintln!("no GPU adapter available. Install a Vulkan ICD or run on the Jetson Thor.");
        std::process::exit(2);
    };
    println!(
        "resident speculative-decode bench on: {}",
        target.adapter_name()
    );
    println!(
        "target: d_model {} · {} layers · ff {}   |   draft: {} layers · vocab {}\n",
        target_cfg.d_model, target_cfg.n_layers, target_cfg.d_ff, draft_cfg.n_layers, vocab
    );

    let n = env_usize("SCIAGENT_DECODE_N", 48).max(4);
    let base = prompt_of(8, vocab);

    // Warm up both (pipeline compile / first alloc out of the timed region).
    let _ = target.generate_cached(&base, 2);
    let _ = draft.generate_cached(&base, 2);

    // Baselines: greedy target tok/s, and how much cheaper the draft is per token.
    let greedy_ms = ms_of(|| {
        let _ = target.generate_cached(&base, n);
    });
    let greedy = target.generate_cached(&base, n); // reference tokens for the exactness check
    let draft_ms = ms_of(|| {
        let _ = draft.generate_cached(&base, n);
    });
    let t_g = greedy_ms / n as f64; // ms / target token
    let t_d = draft_ms / n as f64; // ms / draft token
    let greedy_tps = 1e3 / t_g;
    let cheap = t_g / t_d; // draft is `cheap`x cheaper per token
    println!(
        "greedy target: {greedy_tps:.1} tok/s ({t_g:.2} ms/tok)   draft: {cheap:.1}x cheaper ({t_d:.2} ms/tok)\n"
    );

    // Measured speculative runs (exact by construction; here for tok/s + acceptance).
    println!("  k   accept%   spec_tok/s   speedup   exact");
    for k in [2usize, 4, 8]
    {
        let mut got = Vec::new();
        let spec_ms = ms_of(|| {
            got = target.speculative_generate(&draft, &base, n, k).0;
        });
        let (_, stats) = target.speculative_generate(&draft, &base, n, k);
        let spec_tps = (got.len() - base.len()) as f64 / (spec_ms / 1e3);
        println!(
            "{k:>3}   {:>6.1}   {spec_tps:>10.1}   {:>6.2}x   {}",
            stats.acceptance_rate() * 100.0,
            spec_tps / greedy_tps,
            if got == greedy { "yes" } else { "NO!" },
        );
    }

    // Cost model — a round accepting `a` of k tokens costs (k draft steps + one
    // verify forward ≈ t_g + one target step t_g) and emits a+1 tokens:
    //   speedup(a) = (a+1)·t_g / (k·t_d + 2·t_g)
    // Break-even (speedup = 1): a = k·t_d/t_g + 1 = k/cheap + 1.
    println!("\ncost-model speedup vs acceptance (what a TRAINED draft would buy):");
    println!("   k   accept 50%   accept 75%   accept 100%   break-even accept");
    for k in [2usize, 4, 8]
    {
        let sp = |a: f64| ((a + 1.0) * t_g) / (k as f64 * t_d + 2.0 * t_g);
        let breakeven = (k as f64 / cheap + 1.0).min(k as f64);
        println!(
            "{k:>4}   {:>9.2}x   {:>9.2}x   {:>10.2}x   {:>6.1} / {k}",
            sp(0.5 * k as f64),
            sp(0.75 * k as f64),
            sp(k as f64),
            breakeven,
        );
    }
    println!(
        "\nExact regardless of draft; the win is all in acceptance. On random weights\n\
         acceptance ≈ chance, so measured speedup is <1× — a draft trained to track the\n\
         target (acceptance ~0.6-0.8) is what turns the cost model above into real gains."
    );
}
