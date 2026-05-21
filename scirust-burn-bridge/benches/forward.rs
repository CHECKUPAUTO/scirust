//! Benchmark : mesurer le throughput de forward pass via le bridge.
//!
//! Cible Phase 0 : **≥ 1 000 000 forwards/s en single-thread sur petit MLP**
//! sur un CPU moderne (NdArray, f32, 4→8→2).
//!
//! Si on n'atteint pas cette barre, on a un problème fondamental qu'il faut
//! résoudre avant d'avancer.
//!
//! Pas de `criterion` en phase 0 (moins de deps). Une mesure simple,
//! reproductible, lisible.
//!
//! Lancer :
//! ```bash
//! cargo run --release -p scirust-burn-bridge --bench forward
//! ```

use burn::{
    backend::{ndarray::NdArrayDevice, NdArray},
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{Tensor, TensorData},
};
use scirust_burn_bridge::{InferenceOnly, Policy};
use std::time::Instant;

type B = NdArray<f32>;

#[derive(Module, Debug, Clone)]
struct BenchMlp<BB: burn::tensor::backend::Backend> {
    l1: Linear<BB>,
    l2: Linear<BB>,
    act: Relu,
}

impl<BB: burn::tensor::backend::Backend> BenchMlp<BB> {
    fn new(device: &BB::Device) -> Self {
        Self {
            l1: LinearConfig::new(4, 8).init(device),
            l2: LinearConfig::new(8, 2).init(device),
            act: Relu::new(),
        }
    }
}

impl<BB: burn::tensor::backend::Backend> Policy<BB> for BenchMlp<BB> {
    type Input = Tensor<BB, 2>;
    type Output = Tensor<BB, 2>;

    fn forward(&self, input: Tensor<BB, 2>) -> Tensor<BB, 2> {
        let x = self.l1.forward(input);
        let x = self.act.forward(x);
        self.l2.forward(x)
    }
}

fn warmup(bridge: &InferenceOnly<B, BenchMlp<B>>, n: usize) {
    let input: Tensor<B, 2> =
        Tensor::from_data(TensorData::from([[0.1f32, 0.2, 0.3, 0.4]]), bridge.device());
    for _ in 0..n {
        let _ = bridge.eval(input.clone());
    }
}

fn measure_single(bridge: &InferenceOnly<B, BenchMlp<B>>, n_iters: usize) -> f64 {
    let input: Tensor<B, 2> =
        Tensor::from_data(TensorData::from([[0.1f32, 0.2, 0.3, 0.4]]), bridge.device());

    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..n_iters {
        let out = bridge.eval(input.clone());
        // Force la matérialisation pour empêcher l'optimiseur d'éliminer le calcul.
        let v: Vec<f32> = out.into_data().to_vec().expect("to_vec");
        sink += v[0];
    }
    let dur = start.elapsed();
    std::hint::black_box(sink);

    n_iters as f64 / dur.as_secs_f64()
}

fn measure_batch(bridge: &InferenceOnly<B, BenchMlp<B>>, batch_size: usize, n_iters: usize) -> f64 {
    let data: Vec<f32> = (0..batch_size * 4).map(|i| (i as f32) * 0.001).collect();
    let input: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(data, [batch_size, 4]), bridge.device());

    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..n_iters {
        let out = bridge.eval(input.clone());
        let v: Vec<f32> = out.into_data().to_vec().expect("to_vec");
        sink += v[0];
    }
    let dur = start.elapsed();
    std::hint::black_box(sink);

    let total_forwards = (batch_size * n_iters) as f64;
    total_forwards / dur.as_secs_f64()
}

fn main() {
    let device = NdArrayDevice::Cpu;
    let mlp = BenchMlp::<B>::new(&device);
    let bridge = InferenceOnly::new(mlp, device);

    println!("=== scirust-burn-bridge — bench forward ===");
    println!("Backend: NdArray<f32> · CPU");
    println!("Network: 4 → 8 (relu) → 2");
    println!();

    print!("Warmup (10 000 forwards single)... ");
    warmup(&bridge, 10_000);
    println!("done");
    println!();

    // ── Single-input ────────────────────────────────────────────────
    println!("--- Single-input (batch=1) ---");
    for &n in &[10_000usize, 100_000, 1_000_000] {
        let throughput = measure_single(&bridge, n);
        println!(
            "  n={:>9} → {:>12.0} forwards/s ({:>6.2} µs each)",
            n,
            throughput,
            1_000_000.0 / throughput
        );
    }
    println!();

    // ── Batched ─────────────────────────────────────────────────────
    println!("--- Batched (effective forward count = batch × iters) ---");
    for &(batch, iters) in &[(32usize, 10_000usize), (256, 5_000), (1024, 1_000)] {
        let throughput = measure_batch(&bridge, batch, iters);
        println!(
            "  batch={:>5} iters={:>6} → {:>12.0} forwards/s",
            batch, iters, throughput
        );
    }
    println!();

    println!("--- Cible Phase 0 ---");
    println!("  ≥ 1 000 000 forwards/s (single-input ou batched)");
    println!();
    println!("Si non atteint en single-thread, considérer :");
    println!("  - vérifier le profil release (lto = thin, opt-level = 3)");
    println!("  - profiler avec `perf` pour identifier le bottleneck");
    println!("  - mesurer aussi l'overhead allocation (Tensor::from_data + into_data)");
    println!("  - migrer vers backend Wgpu pour le batched (latence réseau-côté-GPU négligeable au-delà de batch=256)");
}
