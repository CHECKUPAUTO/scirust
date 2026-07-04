//! **How much does keeping the whole model resident buy over the per-op tape
//! path?** Times the VRAM-resident `GpuChain` full-model forward and
//! forward+backward against the deterministic CPU oracle forward, on the *same*
//! synthetic weights, at the **same config as `scirust-sciagent`'s
//! `gpu_forward_parity` example** (the C1 tape path). So the
//! `resident GPU forward + backward` number printed here is directly comparable
//! to that example's `GPU` column — resident vs C1 — on the Jetson Thor.
//!
//!   cargo run -p scirust-gpu --features wgpu --release --example gpu_model_bench
//!
//! `--release` matters a lot — the CPU oracle is a scalar triple loop. Exit code
//! 2 means no GPU adapter was found (run on the Thor, or install a Vulkan ICD).

use std::time::Instant;

use scirust_gpu::ops::{cpu_gqa_transformer_block, cpu_rms_norm};
use scirust_gpu::{GpuChain, GpuMatrix, GqaBlockWeights, GqaModelWeights};

/// Raw (host) weights of one GQA block — kept so the CPU oracle can run the same
/// forward the resident path does.
struct Bw {
    n1: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    n2: Vec<f32>,
    wg: Vec<f32>,
    wu: Vec<f32>,
    wd: Vec<f32>,
}

fn main() {
    let Some(chain) = GpuChain::new()
    else
    {
        eprintln!("no GPU adapter available. Install a Vulkan ICD or run on the Jetson Thor.");
        std::process::exit(2);
    };
    println!("GPU adapter: {}\n", chain.adapter_name());

    // Matches examples/gpu_forward_parity.rs (the C1 tape path) so the resident
    // forward+backward time here is directly comparable to that example's GPU time.
    let (vocab, d, n_layers, n_heads, n_kv_heads, d_ff, seq) = (
        2048usize, 256usize, 4usize, 8usize, 2usize, 512usize, 128usize,
    );
    let dh = d / n_heads;
    let kv_dim = n_kv_heads * dh;
    let (theta, eps) = (10_000.0f32, 1e-5f32);
    let iters = 20usize;
    let cpu_iters = 5usize; // the CPU oracle is a scalar triple loop — fewer iters

    let gen = |n: usize, phase: f32, amp: f32| -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.0007 + phase).sin() * amp)
            .collect()
    };
    let embedding = gen(vocab * d, 0.2, 0.05);
    let final_norm: Vec<f32> = (0..d).map(|i| 0.9 + 0.0001 * i as f32).collect();
    let blocks_raw: Vec<Bw> = (0..n_layers)
        .map(|l| {
            let s = l as f32 * 3.0;
            Bw {
                n1: vec![1.0f32; d],
                wq: gen(d * d, s + 0.5, 0.03),
                wk: gen(d * kv_dim, s + 1.1, 0.03),
                wv: gen(d * kv_dim, s + 1.7, 0.03),
                wo: gen(d * d, s + 2.3, 0.03),
                n2: vec![1.0f32; d],
                wg: gen(d * d_ff, s + 2.9, 0.03),
                wu: gen(d * d_ff, s + 3.5, 0.03),
                wd: gen(d_ff * d, s + 4.1, 0.03),
            }
        })
        .collect();

    let up = |data: &[f32], r: usize, c: usize| chain.upload(data, r, c);
    let gemb = up(&embedding, vocab, d);
    let gfn = up(&final_norm, 1, d);
    let gblocks: Vec<[GpuMatrix; 9]> = blocks_raw
        .iter()
        .map(|b| {
            [
                up(&b.n1, 1, d),
                up(&b.wq, d, d),
                up(&b.wk, d, kv_dim),
                up(&b.wv, d, kv_dim),
                up(&b.wo, d, d),
                up(&b.n2, 1, d),
                up(&b.wg, d, d_ff),
                up(&b.wu, d, d_ff),
                up(&b.wd, d_ff, d),
            ]
        })
        .collect();
    let weights_blocks: Vec<GqaBlockWeights> = gblocks
        .iter()
        .map(|g| GqaBlockWeights {
            norm1: &g[0],
            wq: &g[1],
            wk: &g[2],
            wv: &g[3],
            wo: &g[4],
            norm2: &g[5],
            wg: &g[6],
            wu: &g[7],
            wd: &g[8],
            n_heads,
            n_kv_heads,
            theta,
        })
        .collect();
    let mw = GqaModelWeights {
        embedding: &gemb,
        blocks: &weights_blocks,
        final_norm: &gfn,
    };

    let tokens: Vec<u32> = (0..seq).map(|i| ((i * 13 + 7) % vocab) as u32).collect();
    let targets: Vec<u32> = (0..seq).map(|i| ((i * 29 + 3) % vocab) as u32).collect();

    println!(
        "SCIAGENT resident model: d {d}, {n_layers} layers, {n_heads}h/{n_kv_heads}kv, \
         d_ff {d_ff}, vocab {vocab}, seq {seq}\n"
    );

    // Warm up (device init, pipeline caches, allocations).
    for _ in 0..3
    {
        let logits = chain.gqa_model_forward(&tokens, &mw, eps, true).unwrap();
        let dl = chain.cross_entropy_grad(&logits, &targets).unwrap();
        let grads = chain
            .gqa_model_backward(&tokens, &mw, &dl, eps, true)
            .unwrap();
        let _ = chain.download(&grads.d_embedding).unwrap();
    }

    // Resident GPU forward only (one download flushes the queue).
    let t = Instant::now();
    for _ in 0..iters
    {
        let logits = chain.gqa_model_forward(&tokens, &mw, eps, true).unwrap();
        std::hint::black_box(&chain.download(&logits).unwrap());
    }
    let gpu_fwd = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // Resident GPU forward + backward — a training step's compute (cross-entropy
    // grad seeds the backward). One download of the embedding grad flushes.
    let t = Instant::now();
    for _ in 0..iters
    {
        let logits = chain.gqa_model_forward(&tokens, &mw, eps, true).unwrap();
        let dl = chain.cross_entropy_grad(&logits, &targets).unwrap();
        let grads = chain
            .gqa_model_backward(&tokens, &mw, &dl, eps, true)
            .unwrap();
        std::hint::black_box(&chain.download(&grads.d_embedding).unwrap());
    }
    let gpu_fwdbwd = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // CPU forward oracle on the same weights (deterministic scalar triple loop).
    let t = Instant::now();
    for _ in 0..cpu_iters
    {
        let mut x: Vec<f32> = tokens
            .iter()
            .flat_map(|&tk| embedding[(tk as usize) * d..(tk as usize) * d + d].to_vec())
            .collect();
        for b in &blocks_raw
        {
            x = cpu_gqa_transformer_block(
                &x, &b.n1, &b.wq, &b.wk, &b.wv, &b.wo, &b.n2, &b.wg, &b.wu, &b.wd, seq, d, kv_dim,
                d_ff, n_heads, n_kv_heads, dh, theta, eps, true,
            );
        }
        let normed = cpu_rms_norm(&x, &final_norm, eps, seq, d);
        let mut logits = vec![0.0f32; seq * vocab];
        for i in 0..seq
        {
            for vv in 0..vocab
            {
                let mut acc = 0.0f32;
                for dd in 0..d
                {
                    acc += normed[i * d + dd] * embedding[vv * d + dd];
                }
                logits[i * vocab + vv] = acc;
            }
        }
        std::hint::black_box(&logits);
    }
    let cpu_fwd = t.elapsed().as_secs_f64() * 1e3 / cpu_iters as f64;

    println!("{:<38} {:>12}", "path", "ms / iter");
    println!("{:<38} {:>12.1}", "CPU forward (oracle)", cpu_fwd);
    println!("{:<38} {:>12.1}", "resident GPU forward", gpu_fwd);
    println!(
        "{:<38} {:>12.1}",
        "resident GPU forward + backward", gpu_fwdbwd
    );
    println!(
        "\nforward speedup (CPU / resident GPU): {:.1}x",
        cpu_fwd / gpu_fwd.max(f64::MIN_POSITIVE)
    );
    println!(
        "\nCompare the `resident GPU forward + backward` number above to the `GPU`\n\
         column of `cargo run -p scirust-sciagent --features gpu --release --example\n\
         gpu_forward_parity` (the C1 per-op tape path) at this same config — that is\n\
         the resident-vs-C1 comparison the residency work was for."
    );
}
