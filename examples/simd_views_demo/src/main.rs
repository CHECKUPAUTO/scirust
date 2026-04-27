// examples/simd_views_demo/src/main.rs
//
// Démo complète : MatrixView + PortableSimdBackend + AutoDiff
// cargo run --package simd_views_demo --features scirust-core/portable-simd

use scirust_core::{
    matrix::{
        view::{MatrixView, MatrixViewMut, MatrixShape},
        backend::{best_backend, ScalarBackend, SimdBackend},
    },
};

// ------------------------------------------------------------------ //
//  1. Démo MatrixView — zéro allocation                               //
// ------------------------------------------------------------------ //

fn demo_views() {
    println!("\n=== MatrixView — Sous-matrices sans allocation ===");

    // Matrice 4×4 (row-major)
    let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let mat = MatrixView::from_slice(&data, 4, 4);

    println!("Matrice 4×4 :");
    for r in 0..4 {
        for c in 0..4 { print!("{:5.0}", mat[(r, c)]); }
        println!();
    }

    // Sous-vue 2×2 centrée — zéro copie
    let sub = mat.subview(1, 2, 1, 2);
    println!("\nSous-vue [1..3, 1..3] ({} × {}) :", sub.rows(), sub.cols());
    for r in 0..sub.rows() {
        for c in 0..sub.cols() { print!("{:5.0}", sub[(r, c)]); }
        println!();
    }

    // Ligne 2 comme slice contiguë
    if let Some(row) = mat.row_slice(2) {
        println!("\nLigne 2 (slice) : {:?}", row);
    }
}

// ------------------------------------------------------------------ //
//  2. Démo GEMM via backend                                           //
// ------------------------------------------------------------------ //

fn demo_gemm() {
    println!("\n=== GEMM SIMD — C = A × B ===");

    // A = 3×4, B = 4×2  →  C = 3×2
    #[rustfmt::skip]
    let a_data = vec![
        1.0f32, 2.0, 0.0, 1.0,
        0.0,    1.0, 3.0, 2.0,
        2.0,    0.0, 1.0, 0.0,
    ];
    #[rustfmt::skip]
    let b_data = vec![
        1.0f32, 2.0,
        0.0,    1.0,
        3.0,    1.0,
        1.0,    0.0,
    ];
    let mut c_data = vec![0.0f32; 6];

    let a = MatrixView::from_slice(&a_data, 3, 4);
    let b = MatrixView::from_slice(&b_data, 4, 2);
    let c = MatrixViewMut::from_slice(&mut c_data, 3, 2);

    let backend = best_backend();
    println!("Backend actif : {}", backend.name());

    backend.sgemm_f32(1.0, a, b, 0.0, c);

    let c_view = MatrixView::from_slice(&c_data, 3, 2);
    println!("C (3×2) :");
    for r in 0..3 {
        for c in 0..2 { print!("{:6.1}", c_view[(r, c)]); }
        println!();
    }
}

// ------------------------------------------------------------------ //
//  3. Démo AXPY + ReLU                                                //
// ------------------------------------------------------------------ //

fn demo_axpy_relu() {
    println!("\n=== AXPY + ReLU SIMD ===");

    let x: Vec<f32> = (0..12).map(|i| i as f32 - 5.0).collect();
    let mut y = vec![0.0f32; 12];

    let b = best_backend();

    // y = 2 * x
    b.saxpy_f32(2.0, &x, &mut y);
    println!("Avant ReLU : {:?}", y);

    // ReLU in-place
    b.relu_f32(&mut y);
    println!("Après ReLU : {:?}", y);
}

// ------------------------------------------------------------------ //
//  4. Mini-benchmark maison (pas criterion, juste timing)             //
// ------------------------------------------------------------------ //

fn bench_dot(n: usize) {
    use std::time::Instant;

    let a: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32).cos()).collect();

    let scalar_b = ScalarBackend;
    let t0 = Instant::now();
    let r_scalar = (0..100).map(|_| scalar_b.sdot_f32(&a, &b)).last().unwrap();
    let t_scalar = t0.elapsed();

    let simd_b = best_backend();
    let t1 = Instant::now();
    let r_simd = (0..100).map(|_| simd_b.sdot_f32(&a, &b)).last().unwrap();
    let t_simd = t1.elapsed();

    println!("\n=== Benchmark dot_f32 (n={n}, 100 itérations) ===");
    println!("  Scalar   : {:?}  (résultat = {:.4})", t_scalar, r_scalar);
    println!("  {} : {:?}  (résultat = {:.4})", simd_b.name(), t_simd, r_simd);
    println!(
        "  Speedup  : {:.2}×",
        t_scalar.as_secs_f64() / t_simd.as_secs_f64()
    );
}

fn main() {
    demo_views();
    demo_gemm();
    demo_axpy_relu();
    bench_dot(100_000);
}
