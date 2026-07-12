//! Benchmark comparatif des noyaux `scirust-simd` (axe 4/4).
//!
//! Exécuter en release :
//! ```text
//! cargo run -p scirust-simd --release --example bench
//! ```
//!
//! Mesure et compare, sur la machine courante (backend détecté au runtime) :
//! * SGEMM naïf (triple boucle scalaire) vs tuilé/packé 1 thread vs multi-thread ;
//! * la couche dense fusionnée (`sgemm_bias_act`, `Y = ReLU(X·W + b)`) vs la
//!   même opération en scalaire naïf.
//!
//! Affiche temps, GFLOP/s et accélération. Aucun `assert` de perf (les chiffres
//! dépendent du CPU) — juste un contrôle de cohérence numérique.

use std::time::Instant;

use scirust_simd::dispatch::detect_backend;
use scirust_simd::gemm::{Activation, sgemm_bias_act, sgemm_parallel, sgemm_tiled};
use scirust_simd::matrix::backend::{ScalarBackend, SimdBackend};
use scirust_simd::matrix::view::{MatrixView, MatrixViewMut};

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i % 251) as f32 + seed) * 0.004 - 0.5)
        .collect()
}

fn main() {
    println!("Backend SIMD détecté : {}", detect_backend().label());
    let threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(4);
    println!("Threads disponibles  : {threads}\n");

    bench_sgemm(1024, threads);
    // NB : le chemin AVX-512 fusionné de `sgemm_bias_act` exige k ≤ KC (256) ;
    // au-delà il retombe en scalaire. On benche donc din = 256 (fast path).
    bench_dense(4096, 256, 1024);
}

fn bench_sgemm(dim: usize, threads: usize) {
    let (m, k, n) = (dim, dim, dim);
    let a = fill(m * k, 1.0);
    let b = fill(k * n, 2.0);
    let c0 = vec![0.0f32; m * n];
    let flops = 2.0 * m as f64 * k as f64 * n as f64;

    println!("== SGEMM {m}×{k}×{n} ==");

    // Naïf scalaire.
    let t = Instant::now();
    let mut c_naive = c0.clone();
    ScalarBackend.sgemm_f32(
        1.0,
        MatrixView::new(&a, m, k),
        MatrixView::new(&b, k, n),
        0.0,
        MatrixViewMut::new(&mut c_naive, m, n),
    );
    let dt_naive = t.elapsed().as_secs_f64();
    report("scalaire naïf   ", dt_naive, flops, dt_naive);

    // Tuilé 1 thread.
    let t = Instant::now();
    let mut c1 = c0.clone();
    sgemm_tiled(
        1.0,
        MatrixView::new(&a, m, k),
        MatrixView::new(&b, k, n),
        0.0,
        MatrixViewMut::new(&mut c1, m, n),
    );
    let dt1 = t.elapsed().as_secs_f64();
    report("tuilé 1 thread  ", dt1, flops, dt_naive);

    // Tuilé multi-thread.
    let t = Instant::now();
    let mut cp = c0.clone();
    sgemm_parallel(1.0, &a, m, k, &b, n, 0.0, &mut cp, threads);
    let dtp = t.elapsed().as_secs_f64();
    report(&format!("tuilé {threads} threads "), dtp, flops, dt_naive);

    // Cohérence numérique (échantillon).
    let mut ok = true;
    for idx in (0..m * n).step_by(7919)
    {
        let r = c1[idx];
        if (r - c_naive[idx]).abs() > 1e-2 * (1.0 + c_naive[idx].abs())
            || (cp[idx] - c_naive[idx]).abs() > 1e-2 * (1.0 + c_naive[idx].abs())
        {
            ok = false;
            break;
        }
    }
    println!(
        "  cohérence numérique : {}\n",
        if ok { "OK" } else { "ÉCART" }
    );
}

fn bench_dense(batch: usize, din: usize, dout: usize) {
    let x = fill(batch * din, 1.0);
    let w = fill(din * dout, 2.0);
    let bias = fill(dout, 3.0);
    // Une couche dense = 1 GEMM (2·b·din·dout flops) + biais + activation.
    let flops = 2.0 * batch as f64 * din as f64 * dout as f64;

    println!("== Couche dense ReLU(X·W+b)  X:{batch}×{din}  W:{din}×{dout} ==");

    // Fusionné (sgemm_bias_act).
    let t = Instant::now();
    let mut y = vec![0.0f32; batch * dout];
    sgemm_bias_act(
        1.0,
        MatrixView::new(&x, batch, din),
        MatrixView::new(&w, din, dout),
        &bias,
        Activation::Relu,
        MatrixViewMut::new(&mut y, batch, dout),
    );
    let dt_fused = t.elapsed().as_secs_f64();

    // Naïf scalaire (matmul + biais + relu en 3 passes).
    let t = Instant::now();
    let mut yn = vec![0.0f32; batch * dout];
    for i in 0..batch
    {
        for j in 0..dout
        {
            let mut acc = bias[j];
            for p in 0..din
            {
                acc += x[i * din + p] * w[p * dout + j];
            }
            yn[i * dout + j] = acc.max(0.0);
        }
    }
    let dt_naive = t.elapsed().as_secs_f64();

    report("scalaire naïf   ", dt_naive, flops, dt_naive);
    report("fusionné SIMD   ", dt_fused, flops, dt_naive);

    let mut ok = true;
    for idx in (0..batch * dout).step_by(4093)
    {
        if (y[idx] - yn[idx]).abs() > 1e-2 * (1.0 + yn[idx].abs())
        {
            ok = false;
            break;
        }
    }
    println!(
        "  cohérence numérique : {}\n",
        if ok { "OK" } else { "ÉCART" }
    );
}

fn report(label: &str, dt: f64, flops: f64, baseline: f64) {
    println!(
        "  {label} : {:8.1} ms   {:7.2} GFLOP/s   ×{:.1}",
        dt * 1e3,
        flops / dt / 1e9,
        baseline / dt
    );
}
