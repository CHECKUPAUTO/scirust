// scirust-simd/benches/linalg_bench.rs
//
// Benchmarks criterion du GEMM virgule fixe (`fixed::linalg`), comparé à une
// baseline flottante `f32` naïve.
//
// Mesure le **débit** en multiplications-accumulations (MAC/s) de `matmul` et
// `matvec` pour `Q16_16` (virgule fixe, produit scalaire SIMD `dot`) et `f32`
// (triple boucle naïve de référence). L'objectif est de situer le coût relatif :
// la virgule fixe apporte le **déterminisme bit-à-bit** (indépendant de l'ordre
// de sommation, des lanes et de l'architecture), à un coût qui doit rester
// raisonnable.
//
// Lancement (cible AVX2 pour éviter la sur-détection AVX-512 en VM) :
//   RUSTFLAGS="-C target-cpu=x86-64-v3" \
//     cargo bench -p scirust-simd --features portable-simd --bench linalg_bench

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_simd::fixed::Q16_16;
use scirust_simd::fixed::linalg as flin;

/// Côté des matrices carrées ; `D³` MAC par produit, `256 Kio` par matrice f32.
const D: usize = 128;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

fn fixed_data(seed: u64, len: usize) -> Vec<Q16_16> {
    let mut rng = Lcg(seed);
    // Valeurs dans [-1, 1) : les produits partiels restent dans la plage.
    (0..len)
        .map(|_| Q16_16::try_from(rng.unit()).unwrap())
        .collect()
}
fn f32_data(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..len).map(|_| rng.unit() as f32).collect()
}

/// GEMM flottant naïf (référence non déterministe) : `C = A·B`, row-major.
fn naive_matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m
    {
        for j in 0..n
        {
            let mut acc = 0.0f32;
            for l in 0..k
            {
                acc += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn bench_matmul(c: &mut Criterion) {
    let a = fixed_data(0x1, D * D);
    let b = fixed_data(0x2, D * D);
    let fa = f32_data(0x1, D * D);
    let fb = f32_data(0x2, D * D);

    let mut g = c.benchmark_group("matmul_128");
    g.throughput(Throughput::Elements((D * D * D) as u64)); // MAC
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::matmul(black_box(&a), black_box(&b), D, D, D))
    });
    g.bench_function(BenchmarkId::new("f32", "naive"), |bch| {
        bch.iter(|| naive_matmul_f32(black_box(&fa), black_box(&fb), D, D, D))
    });
    g.finish();
}

fn bench_matvec(c: &mut Criterion) {
    let a = fixed_data(0x3, D * D);
    let x = fixed_data(0x4, D);

    let mut g = c.benchmark_group("matvec_128");
    g.throughput(Throughput::Elements((D * D) as u64)); // MAC
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::matvec(black_box(&a), black_box(&x), D, D))
    });
    g.finish();
}

/// Côté des matrices pour les décompositions (coût cubique dominé par les
/// divisions/`sqrt`, plus onéreuses qu'un MAC : taille réduite vs `D`).
const N: usize = 48;

/// Matrice `n×n` symétrique définie positive : `A = BᵀB + n·I`.
fn spd_data(seed: u64, n: usize) -> Vec<Q16_16> {
    let b = fixed_data(seed, n * n);
    let bt = flin::transpose(&b, n, n);
    let mut a = flin::matmul(&bt, &b, n, n, n);
    for i in 0..n
    {
        a[i * n + i] += Q16_16::from(n as i32);
    }
    a
}

/// Matrice `n×n` à diagonale strictement dominante (inversible, bien
/// conditionnée).
fn diag_dominant_data(seed: u64, n: usize) -> Vec<Q16_16> {
    let mut a = fixed_data(seed, n * n);
    for i in 0..n
    {
        a[i * n + i] = Q16_16::from(4 * n as i32);
    }
    a
}

fn bench_cholesky(c: &mut Criterion) {
    let a = spd_data(0x5, N);

    let mut g = c.benchmark_group("cholesky_48");
    g.throughput(Throughput::Elements((N * N * N) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::cholesky(black_box(&a), N))
    });
    g.finish();

    let b = fixed_data(0x6, N);
    let mut g = c.benchmark_group("cholesky_solve_48");
    g.throughput(Throughput::Elements((N * N * N) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::cholesky_solve(black_box(&a), black_box(&b), N))
    });
    g.finish();
}

fn bench_lu(c: &mut Criterion) {
    let a = diag_dominant_data(0x7, N);

    let mut g = c.benchmark_group("lu_decompose_48");
    g.throughput(Throughput::Elements((N * N * N) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::lu_decompose(black_box(&a), N))
    });
    g.finish();

    let b = fixed_data(0x8, N);
    let mut g = c.benchmark_group("lu_solve_48");
    g.throughput(Throughput::Elements((N * N * N) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::lu_solve(black_box(&a), black_box(&b), N))
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_matvec,
    bench_cholesky,
    bench_lu
);
criterion_main!(benches);
