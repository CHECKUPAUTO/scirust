use scirust_core::autodiff;
use std::time::Instant;

#[autodiff]
fn square(x: f64) -> f64 {
    x * x
}

#[autodiff]
fn rosenbrock(x: f64, y: f64) -> f64 {
    (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
}

#[autodiff]
fn neural_activation(x: f64) -> f64 {
    x.sin().exp() / (1.0 + x.powi(2))
}

fn matmul_scalar(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let p = b.len();
    let mut c = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            for k in 0..p {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn main() {
    // 1. AutoDiff exact (forward-mode Dual)
    println!("=== SciRust Exact AutoDiff (Forward-Mode Dual) ===");
    println!("square(3.0)                  = {}", square(3.0));
    println!("grad of square at 3.0        = {}", square_grad(3.0));
    println!("expected: 6.0");
    println!("grad of square at 0.0        = {}", square_grad(0.0));
    println!("expected: 0.0");
    println!("grad of square at -2.0       = {}", square_grad(-2.0));
    println!("expected: -4.0");

    println!("rosenbrock(1.0, 1.0)         = {}", rosenbrock(1.0, 1.0));
    let (dx, dy) = rosenbrock_grad(1.0, 1.0);
    println!("grad of rosenbrock at (1,1)  = ({}, {})", dx, dy);
    println!("expected: (0, 0)");
    let (dx2, dy2) = rosenbrock_grad(0.0, 0.0);
    println!("grad of rosenbrock at (0,0)  = ({}, {})", dx2, dy2);
    println!("expected: (-2, 0)");

    println!("\nneural_activation(1.0)       = {}", neural_activation(1.0));
    let da = neural_activation_grad(1.0);
    println!("grad of neural_activation at 1.0 = {}", da);
    println!("(analytical derivative of exp(sin(x))/(1+x^2))");

    // 2. SIMD demo
    println!("\n=== SciRust SIMD Auto-Vectorization ===");
    let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    println!("before simd_add_one: {:?}", v);
    scirust_core::simd_add_one(&mut v);
    println!("after  simd_add_one: {:?}", v);

    // 3. GPU/parallel dispatch demo
    println!("\n=== SciRust GPU/Parallel Dispatch ===");
    let mut g = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    println!("before scale*2: {:?}", g);
    scirust_core::dispatch::gpu_or_cpu(&mut g, |chunk| {
        for x in chunk {
            *x *= 2.0;
        }
    });
    println!("after  scale*2: {:?}", g);

    // 4. Show the Dual type directly
    println!("\n=== SciRust Dual Number Direct Use ===");
    use scirust_core::Dual;
    let x = Dual::var(2.0);
    let y = x.powi(3) + x.sin();
    println!("f(x) = x^3 + sin(x) at x=2");
    println!("value     = {}", y.val());
    println!("derivative= {} (expected: 12 + cos(2) = {})", y.grad(), 12.0 + 2.0f64.cos());

    // 5. Simple matmul benchmark
    println!("\n=== SciRust Matrix Multiplication Benchmark ===");
    let size = 256;
    let a: Vec<Vec<f64>> = (0..size).map(|i| (0..size).map(|j| (i + j) as f64).collect()).collect();
    let b: Vec<Vec<f64>> = (0..size).map(|i| (0..size).map(|j| (i * j + 1) as f64).collect()).collect();

    let start = Instant::now();
    let c = matmul_scalar(&a, &b);
    let elapsed = start.elapsed();
    println!("Scalar matmul {}x{}: {:.2?}", size, size, elapsed);
    println!("Result[0][0] = {}, Result[{}][{}] = {}", c[0][0], size - 1, size - 1, c[size - 1][size - 1]);
}
