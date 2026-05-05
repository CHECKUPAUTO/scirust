// SciRust CLI — Industrial-grade deep learning framework
//
// Usage:
//   scirust                 Run capability overview
//   scirust simd            SIMD benchmark
//   scirust autodiff        Autodiff demo (XOR MLP)
//   scirust symbolic        Symbolic math demo
//   scirust bench           Full benchmark suite

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("overview");

    match mode {
        "simd" => simd_bench(),
        "autodiff" => autodiff_demo(),
        "symbolic" => symbolic_demo(),
        "bench" => {
            simd_bench();
            autodiff_demo();
            symbolic_demo();
        }
        _ => overview(),
    }
}

fn overview() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         SciRust v0.13 — Industrial ML Framework         ║");
    println!("║            Pure Rust · Autodiff · SIMD · GPU             ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("Capabilities detected:");

    #[cfg(target_arch = "x86_64")]
    {
        println!("  CPU arch      : x86_64");
        println!("  SSE2          : {}", std::is_x86_feature_detected!("sse2"));
        println!("  AVX           : {}", std::is_x86_feature_detected!("avx"));
        println!("  AVX2          : {}", std::is_x86_feature_detected!("avx2"));
        println!("  FMA           : {}", std::is_x86_feature_detected!("fma"));
        println!("  AVX-512F      : {}", std::is_x86_feature_detected!("avx512f"));
    }
    #[cfg(target_arch = "aarch64")]
    {
        println!("  CPU arch      : aarch64");
        println!("  NEON          : {}", std::arch::is_aarch64_feature_detected!("neon"));
    }

    let backend = scirust_simd::dispatch::detect_backend();
    println!("  SIMD backend  : {}", backend.label());
    println!();

    println!("Quick demos:");
    println!("  scirust simd           SIMD vs scalar performance");
    println!("  scirust autodiff       Train XOR classifier via autodiff");
    println!("  scirust symbolic       Symbolic math (derive, simplify)");
    println!("  scirust bench          Run all benchmarks");
    println!();

    println!("Packages (workspace):");
    println!("  scirust-core           Core autodiff, NN layers, data loaders");
    println!("  scirust-autodiff       Autodiff engine (tape-based reverse-mode)");
    println!("  scirust-simd           SIMD dispatch (SSE2/AVX2/AVX512/NEON)");
    println!("  scirust-gpu            GPU backend (CUDA/wgpu/Metal)");
    println!("  scirust-symbolic       Symbolic math engine");
    println!("  scirust-learning       ML utilities (regression, patterns)");
    println!("  turboquant             KV-cache compression proxy");
    println!();

    println!("Examples:");
    println!("  examples/quickstart_v2     XOR classifier (100% accuracy)");
    println!("  examples/mnist_classifier  MNIST digit recognition");
    println!("  examples/cifar10_classifier CIFAR-10 image classification");
    println!("  examples/transformer_demo  Transformer encoder/decoder demo");
}

fn simd_bench() {
    println!("=== SIMD Benchmark: SAXPY (y += alpha * x) ===\n");

    let sizes = [1024, 16384, 262144, 1_048_576];
    let alpha = 2.0f32;
    let iterations = 50;

    for &n in &sizes {
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut y_scalar: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let mut y_simd = y_scalar.clone();

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            for j in 0..n {
                y_scalar[j] += alpha * x[j];
            }
        }
        let scalar_time = start.elapsed().as_secs_f64() / iterations as f64;

        let backend = scirust_simd::dispatch::runtime_backend();
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            backend.saxpy_f32(alpha, &x, &mut y_simd);
        }
        let simd_time = start.elapsed().as_secs_f64() / iterations as f64;

        println!(
            "  n={:>8}  scalar={:>8.3}µs  simd={:>8.3}µs  speedup={:.2}x",
            n,
            scalar_time * 1e6,
            simd_time * 1e6,
            scalar_time / simd_time
        );
    }
    println!();
}

fn autodiff_demo() {
    println!("=== Autodiff Demo: 2-Layer MLP on XOR ===\n");

    use scirust_core::autodiff::optim::{Adam, Optimizer};
    use scirust_core::autodiff::reverse::{Tape, Tensor};
    use scirust_core::nn::{
        init::{KaimingNormal, Zeros},
        Linear, Module, PcgEngine, ReLU, Sequential,
    };

    let inputs: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]];
    let targets: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

    let mut rng = PcgEngine::new(42);
    let weight_init = KaimingNormal;
    let bias_init = Zeros;

    let mut model = Sequential::new()
        .add(Linear::new(2, 8, &weight_init, &bias_init, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(8, 1, &weight_init, &bias_init, &mut rng));

    let mut opt = Adam::new(0.05);

    for epoch in 0..300 {
        let mut total_loss = 0.0f32;
        for (x_arr, &t) in inputs.iter().zip(targets.iter()) {
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(x_arr.to_vec(), 1, 2));
            let target_t = tape.input(Tensor::from_vec(vec![t], 1, 1));
            let pred = model.forward(&tape, x);
            let diff = pred.sub(target_t);
            #[allow(clippy::clone_on_copy)]
            let diff2 = diff.clone();
            let sqr = diff.hadamard(diff2);
            let loss_var = sqr.sum();
            let loss_val = tape.value(loss_var.idx()).data[0];
            total_loss += loss_val;
            tape.backward(loss_var.idx());
            opt.step(&model.parameter_indices(), &tape);
            model.sync(&tape);
        }
        total_loss /= inputs.len() as f32;
        if epoch % 60 == 0 || epoch == 299 {
            println!("  epoch {:>3}  loss={:.6}", epoch, total_loss);
        }
    }

    let mut correct = 0;
    for (i, x_arr) in inputs.iter().enumerate() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(x_arr.to_vec(), 1, 2));
        let pred = model.forward(&tape, x);
        let val = tape.value(pred.idx()).data[0];
        let class = if val > 0.5 { 1.0 } else { 0.0 };
        if (class - targets[i]).abs() < 0.01 {
            correct += 1;
        }
    }
    println!(
        "  Accuracy: {}/{} ({:.0}%)\n",
        correct,
        inputs.len(),
        100.0 * correct as f32 / inputs.len() as f32
    );
}

fn symbolic_demo() {
    println!("=== Symbolic Math Demo ===\n");

    use scirust_symbolic::{diff, eval, parse, simplify, solve_quadratic};
    use std::collections::HashMap;

    let expr = parse("2*x^2 + 3*x + 1").expect("parse");
    println!("  Expression  : 2x² + 3x + 1");
    println!("  Parsed      : {}", expr);
    println!("  Simplified  : {}", simplify(&expr));

    let deriv = diff(&expr, "x");
    println!("  d/dx        : {}", deriv);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);
    println!("  eval(x=2)   : {}", eval(&expr, &vars).unwrap());

    let sol = solve_quadratic(&expr, "x");
    println!("  Solve       : roots = {:?}", sol);
    println!();
}
