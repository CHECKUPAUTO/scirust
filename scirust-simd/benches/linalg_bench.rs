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
use scirust_simd::fixed::layer::Linear;
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

/// Matrice `m×n` (`m > n`) de rang plein : bloc `n×n` à diagonale dominante
/// surmonté de `m − n` lignes de coefficients modestes (surdétermination
/// typique d'un système de moindres carrés).
fn full_rank_data(seed: u64, m: usize, n: usize) -> Vec<Q16_16> {
    let mut a = vec![Q16_16::from(0); m * n];
    let top = diag_dominant_data(seed, n);
    a[..n * n].copy_from_slice(&top);
    let extra = fixed_data(seed ^ 0xA5A5, (m - n) * n);
    a[n * n..].copy_from_slice(&extra);
    a
}

/// `M > N` : système surdéterminé typique des moindres carrés.
const M_QR: usize = 64;

fn bench_qr(c: &mut Criterion) {
    let a = full_rank_data(0x9, M_QR, N);

    let mut g = c.benchmark_group("qr_decompose_64x48");
    g.throughput(Throughput::Elements((M_QR * N * N) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::qr_decompose(black_box(&a), M_QR, N))
    });
    g.finish();

    let b = fixed_data(0xA, M_QR);
    let mut g = c.benchmark_group("qr_solve_64x48");
    g.throughput(Throughput::Elements((M_QR * N * N) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::qr_solve(black_box(&a), black_box(&b), M_QR, N))
    });
    g.finish();
}

/// `matmul_bt` (Bᵀ déjà disponible) vs `matmul` (transposition interne) sur
/// la même paire de matrices carrées — mesure le coût économisé.
fn bench_matmul_bt(c: &mut Criterion) {
    let a = fixed_data(0xB, D * D);
    let bt = fixed_data(0xC, D * D);
    let b = flin::transpose(&bt, D, D);

    let mut g = c.benchmark_group("matmul_bt_128");
    g.throughput(Throughput::Elements((D * D * D) as u64));
    g.bench_function(BenchmarkId::new("fixed", "matmul_bt"), |bch| {
        bch.iter(|| flin::matmul_bt(black_box(&a), black_box(&bt), D, D, D))
    });
    g.bench_function(BenchmarkId::new("fixed", "matmul_avec_transpose"), |bch| {
        bch.iter(|| flin::matmul(black_box(&a), black_box(&b), D, D, D))
    });
    g.finish();
}

/// `Linear::forward_batch` (un seul GEMM `matmul_bt`) vs `batch` appels de
/// `Linear::forward` (un `matvec` chacun) — même résultat bit-à-bit
/// (cf. tests), débit différent.
fn bench_linear_batch(c: &mut Criterion) {
    let (out_f, in_f, batch) = (64, 128, 32);
    let w = fixed_data(0xD, out_f * in_f);
    let bias = fixed_data(0xE, out_f);
    let x = fixed_data(0xF, batch * in_f);
    let layer = Linear::new(w, bias, out_f, in_f);

    let mut g = c.benchmark_group("linear_forward_batch32");
    g.throughput(Throughput::Elements((batch * out_f * in_f) as u64));
    g.bench_function(BenchmarkId::new("fixed", "batched"), |bch| {
        bch.iter(|| layer.forward_batch(black_box(&x), batch))
    });
    g.bench_function(BenchmarkId::new("fixed", "looped"), |bch| {
        bch.iter(|| {
            let mut out = Vec::with_capacity(batch * out_f);
            for row in black_box(&x).chunks_exact(in_f)
            {
                out.extend(layer.forward(row));
            }
            out
        })
    });
    g.finish();
}

/// Matrice `n×n` symétrique (pas nécessairement définie positive) :
/// `A = B + Bᵀ`, `B` à coefficients modestes.
fn symmetric_data(seed: u64, n: usize) -> Vec<Q16_16> {
    let b = fixed_data(seed, n * n);
    let bt = flin::transpose(&b, n, n);
    (0..n * n).map(|i| b[i] + bt[i]).collect()
}

/// Décomposition spectrale de Jacobi (rotations cycliques) : itérative,
/// contrairement à Cholesky/LU/QR — le débit dépend du nombre de passes
/// jusqu'à convergence (borné par `max_sweeps`), pas d'une formule fermée.
fn bench_jacobi_eigen(c: &mut Criterion) {
    let a = symmetric_data(0x10, N);

    let mut g = c.benchmark_group("jacobi_eigen_48");
    g.throughput(Throughput::Elements((N * N * N) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::jacobi_eigen(black_box(&a), N, Q16_16::try_from(1e-4).unwrap(), 100))
    });
    g.finish();
}

/// Décomposition en valeurs singulières (`jacobi_eigen` sur `AᵀA`, `m > n`).
fn bench_svd(c: &mut Criterion) {
    let a = full_rank_data(0x11, M_QR, N);

    let mut g = c.benchmark_group("svd_64x48");
    g.throughput(Throughput::Elements((M_QR * N * N) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::svd(black_box(&a), M_QR, N, Q16_16::try_from(1e-4).unwrap(), 100))
    });
    g.finish();
}

/// Réduction de Hessenberg (Householder, appliquée des deux côtés) : une
/// seule fois par matrice, contrairement à `eigenvalues_general` qui itère
/// dessus ensuite — isole le coût de la seule réduction.
fn bench_hessenberg(c: &mut Criterion) {
    let a = fixed_data(0x12, N * N);

    let mut g = c.benchmark_group("hessenberg_48");
    g.throughput(Throughput::Elements((N * N * N) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::hessenberg(black_box(&a), N))
    });
    g.finish();
}

/// Valeurs propres d'une matrice **quelconque** (non symétrique) :
/// Hessenberg puis QR décalé avec déflation — à comparer à `jacobi_eigen`
/// (réservé aux matrices symétriques, sans valeurs propres complexes).
fn bench_eigenvalues_general(c: &mut Criterion) {
    let a = fixed_data(0x13, N * N);

    let mut g = c.benchmark_group("eigenvalues_general_48");
    g.throughput(Throughput::Elements((N * N * N) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| {
            flin::eigenvalues_general(black_box(&a), N, Q16_16::try_from(1e-4).unwrap(), 100 * N)
        })
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_matvec,
    bench_cholesky,
    bench_lu,
    bench_qr,
    bench_matmul_bt,
    bench_linear_batch,
    bench_jacobi_eigen,
    bench_svd,
    bench_hessenberg,
    bench_eigenvalues_general
);
criterion_main!(benches);
