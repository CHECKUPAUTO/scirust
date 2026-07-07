//! **Resident pretraining demo** — trains a small SCIAGENT on a repeating token
//! pattern using the fully-resident GPU path (`ResidentModel::train_tokens`), the
//! one that beats the per-op tape path ~4× on the Jetson Thor. Prints the loss
//! curve, then syncs the trained weights back into the model.
//!
//!   cargo run -p scirust-sciagent --features gpu --release --example resident_train
//!
//! Exit code 2 means no GPU adapter was found (run on the Thor, or install a
//! Vulkan ICD). For a real run, replace the synthetic pattern with your token
//! shards (load them into a `Vec<u32>` and pass to `train_tokens`).

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::gpu::{ResidentModel, ResidentTrainConfig};
use scirust_sciagent::model::SciAgentModel;

fn main() {
    let config = SciAgentConfig {
        vocab_size: 256,
        d_model: 128,
        n_layers: 4,
        n_heads: 4,
        n_kv_heads: 2,
        d_ff: 256,
        max_seq_len: 128,
        rope_theta: 10_000.0,
        tie_embeddings: true,
        use_bias: false,
        eps: 1e-5,
    };
    let mut model = SciAgentModel::new(&config);
    let Some(mut rm) = ResidentModel::from_model(&model)
    else
    {
        eprintln!("no GPU adapter available. Install a Vulkan ICD or run on the Jetson Thor.");
        std::process::exit(2);
    };
    println!("resident training on: {}\n", rm.adapter_name());
    println!(
        "model: d {}, {} layers, {}h/{}kv, d_ff {}, vocab {}\n",
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.d_ff,
        config.vocab_size
    );

    // A learnable repeating pattern of valid token ids.
    let pattern: Vec<u32> = (0..32u32)
        .map(|i| (i * 7 + 3) % config.vocab_size as u32)
        .collect();
    let seq = 64usize;
    let tokens: Vec<u32> = (0..seq * 200).map(|i| pattern[i % pattern.len()]).collect();
    let cfg = ResidentTrainConfig {
        lr: 0.01,
        seq_len: seq,
        weight_decay: 0.0,
        ..Default::default()
    };

    let losses = rm.train_tokens(&tokens, &cfg);
    println!("{:>6}  {:>10}", "step", "loss");
    for (i, l) in losses.iter().enumerate()
    {
        if i % 20 == 0 || i + 1 == losses.len()
        {
            println!("{i:>6}  {l:>10.4}");
        }
    }

    let n = losses.len().clamp(1, 5);
    let first: f32 = losses[..n].iter().sum::<f32>() / n as f32;
    let last: f32 = losses[losses.len() - n..].iter().sum::<f32>() / n as f32;
    println!(
        "\n{} resident steps: loss {first:.4} -> {last:.4}  ({:.1}% reduction)",
        losses.len(),
        (1.0 - last / first) * 100.0
    );

    // Write the trained weights back into the model (for checkpointing / inference).
    rm.sync_to_model(&mut model);
    println!("trained weights synced back into the SciAgentModel.");
}
