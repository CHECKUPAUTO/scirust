// Bench latence du runtime d'inference (#2 latence bornee). Mesure, pas optim.
// Correctness deja garantie bit-exact par #1 ; ici on mesure le temps.
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::data::{DataLoader, MnistDataset};
use scirust_core::nn::{KaimingNormal, Linear, Module, PcgEngine, ReLU, Sequential, Zeros};
use scirust_runtime::load_weights;
use std::time::Instant;

fn percentile(sorted: &[u64], q: f64) -> u64 {
    let idx = (((sorted.len() - 1) as f64) * q).round() as usize;
    sorted[idx]
}

fn run_once(model: &mut Sequential, x: &Tensor) -> f32 {
    let tape = Tape::new();
    let v = tape.input(x.clone());
    let logits = model.forward(&tape, v);
    tape.value(logits.idx()).data[0]
}

fn main() {
    let data_dir =
        std::env::var("MNIST_DIR").unwrap_or_else(|_| "/root/scirust/data/mnist".to_string());
    let test = MnistDataset::load_idx(
        format!("{}/t10k-images-idx3-ubyte", data_dir),
        format!("{}/t10k-labels-idx1-ubyte", data_dir),
    )
    .expect("chargement test MNIST");

    let mut rng = PcgEngine::new(7);
    let mut model = Sequential::new()
        .add(Linear::new(784, 256, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(256, 10, &KaimingNormal, &Zeros, &mut rng));
    let sd = load_weights("mnist_mlp.srt").expect("load_weights (lance train_artifact d'abord)");
    model.load_state_dict(&sd).expect("load_state_dict");

    let mut l1 = DataLoader::new(test.subsample(256), 1, false, 42);
    let sample1 = l1.iter().next().map(|(x, _)| x).expect(">=1 echantillon");

    let mut sink = 0.0f32;
    for _ in 0..300 {
        sink += run_once(&mut model, &sample1);
    }

    let n = 5000usize;
    let mut lat: Vec<u64> = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        sink += run_once(&mut model, &sample1);
        lat.push(t.elapsed().as_nanos() as u64);
    }
    lat.sort_unstable();
    let mean = lat.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let us = |v: u64| v as f64 / 1000.0;
    let p50 = percentile(&lat, 0.50);
    let p99 = percentile(&lat, 0.99);

    println!("=== Latence inference batch=1 (n={}) ===", n);
    println!("  min    : {:>8.2} us", us(lat[0]));
    println!("  p50    : {:>8.2} us", us(p50));
    println!("  mean   : {:>8.2} us", mean / 1000.0);
    println!("  p90    : {:>8.2} us", us(percentile(&lat, 0.90)));
    println!("  p99    : {:>8.2} us", us(p99));
    println!("  p99.9  : {:>8.2} us", us(percentile(&lat, 0.999)));
    println!("  max    : {:>8.2} us", us(*lat.last().unwrap()));
    println!("  p99/p50: {:>8.2}x   (indicateur de queue)", p99 as f64 / p50 as f64);
    println!("  debit  : {:>8.0} inf/s (1/p50)", 1.0e9 / p50 as f64);

    let mut l64 = DataLoader::new(test.subsample(256), 64, false, 42);
    let sample64 = l64.iter().next().map(|(x, _)| x).expect("batch64");
    for _ in 0..50 {
        sink += run_once(&mut model, &sample64);
    }
    let m = 1000usize;
    let t = Instant::now();
    for _ in 0..m {
        sink += run_once(&mut model, &sample64);
    }
    let el = t.elapsed().as_secs_f64();
    println!("=== Debit batch=64 ===");
    println!("  {:.0} echantillons/s ({} batches / {:.3}s)", (m * 64) as f64 / el, m, el);
    println!("(sink={:.3}, anti-DCE)", sink);
}
