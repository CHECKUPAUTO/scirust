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

// Migration note (scirust-bench-schema): inputs come from `fixed_data`/
// `f32_data(seed, len)`, backed by the in-file `Lcg`; bench_matmul pins a/fa
// to seed 0x1 and b/fb to seed 0x2. D=128 (D^3 MACs/product). Example
// conversion for the "matmul" group's "fixed"/"Q16_16" case (after `cargo
// bench --bench linalg_bench`, reading
// target/criterion/matmul/fixed/Q16_16/new/estimates.json):
//
//   scirust_bench_schema::criterion_estimate_to_record(
//       &estimates_json,
//       "scirust-simd/linalg_matmul", // kernel
//       "D=128",                       // dataset
//       "fixed:Q16_16",                // method
//       0x1,                           // seed: a's fixed_data() seed
//   )
// See scirust-bench-schema's crate docs ("Migrating criterion targets") for the full pattern.

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

/// Vecteur propre réel par itération inverse (`eigenvector_real`) : réutilise
/// `lu_solve` à chaque itération — coût dominé par le nombre d'itérations
/// nécessaires à la convergence (borné par `max_iter`), pas une formule
/// fermée. La valeur propre ciblée est calculée une fois, hors mesure, par
/// `eigenvalues_general` (déjà benchée séparément ci-dessus).
fn bench_eigenvector_real(c: &mut Criterion) {
    let a = fixed_data(0x18, N * N);
    let eigenvalues = flin::eigenvalues_general(&a, N, Q16_16::try_from(1e-4).unwrap(), 100 * N)
        .expect("pas de débordement / convergence");
    let lambda = eigenvalues
        .iter()
        .find_map(|&e| match e
        {
            flin::Eigenvalue::Real(x) => Some(x),
            flin::Eigenvalue::Complex(_, _) => None,
        })
        .expect("au moins une valeur propre réelle pour cette échelle");

    let mut g = c.benchmark_group("eigenvector_real_48");
    g.throughput(Throughput::Elements((N * N) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| {
            flin::eigenvector_real(
                black_box(&a),
                N,
                lambda,
                Q16_16::try_from(1e-5).unwrap(),
                50,
            )
        })
    });
    g.finish();
}

/// Racines d'un polynôme de degré 48 (coefficients bornés, aucune racine
/// connue à l'avance) : matrice compagnon + `eigenvalues_general`.
fn bench_poly_roots(c: &mut Criterion) {
    let mut coeffs = fixed_data(0x14, N + 1);
    coeffs[0] = Q16_16::one(); // coefficient dominant non nul (forme monique directe).

    let mut g = c.benchmark_group("poly_roots_deg48");
    g.throughput(Throughput::Elements((N * N * N) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::poly_roots(black_box(&coeffs), Q16_16::try_from(1e-4).unwrap(), 100 * N))
    });
    g.finish();
}

/// Exponentielle de matrice (mise à l'échelle et carrés répétés, Padé `[3/3]`).
fn bench_matrix_exp(c: &mut Criterion) {
    let a = fixed_data(0x15, N * N);

    let mut g = c.benchmark_group("matrix_exp_48");
    g.throughput(Throughput::Elements((N * N * N) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::matrix_exp(black_box(&a), N))
    });
    g.finish();
}

/// Problème aux valeurs propres généralisé `A·x = λ·B·x` : réduction de
/// Cholesky (`B`) puis `jacobi_eigen` — coût dominé par ce dernier, la
/// réduction n'ajoutant que `n` substitutions triangulaires et deux GEMM.
fn bench_generalized_eig_symmetric(c: &mut Criterion) {
    let a = symmetric_data(0x16, N);
    let b = spd_data(0x17, N);

    let mut g = c.benchmark_group("generalized_eig_symmetric_48");
    g.throughput(Throughput::Elements((N * N * N) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| {
            flin::generalized_eig_symmetric(
                black_box(&a),
                black_box(&b),
                N,
                Q16_16::try_from(1e-4).unwrap(),
                100,
            )
        })
    });
    g.finish();
}

/// Racine carrée/logarithme de matrice, à une taille bien plus petite que
/// `N` (48) : contrairement aux autres décompositions de ce fichier
/// (passe unique), Denman-Beavers **inverse une matrice complète** deux
/// fois par itération (colonne par colonne via [`flin::lu_solve`]), et
/// `matrix_log` enchaîne plusieurs `matrix_sqrt` — coût qui grimpe bien
/// plus vite avec `n` ; `N_SQRT` garde le bench sous une seconde.
const N_SQRT: usize = 8;

/// Racine carrée de matrice (Denman-Beavers), sur `A` symétrique définie
/// positive.
fn bench_matrix_sqrt(c: &mut Criterion) {
    let a = spd_data(0x18, N_SQRT);
    let tol = Q16_16::try_from(1e-4).unwrap();

    let mut g = c.benchmark_group("matrix_sqrt_8");
    g.throughput(Throughput::Elements((N_SQRT * N_SQRT * N_SQRT) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::matrix_sqrt(black_box(&a), N_SQRT, tol, 50))
    });
    g.finish();
}

/// Logarithme de matrice (mise à l'échelle inverse + racines carrées
/// itérées), sur `A` symétrique définie positive.
fn bench_matrix_log(c: &mut Criterion) {
    let a = spd_data(0x19, N_SQRT);
    let tol = Q16_16::try_from(1e-4).unwrap();

    let mut g = c.benchmark_group("matrix_log_8");
    g.throughput(Throughput::Elements((N_SQRT * N_SQRT * N_SQRT) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::matrix_log(black_box(&a), N_SQRT, tol, 50))
    });
    g.finish();
}

/// Recalage rigide (Kabsch) d'un nuage de `M_POINTS` points en dimension 3 —
/// échelle typique d'un alignement de nuage de points/étalonnage de capteur.
const M_POINTS: usize = 100;
const DIM: usize = 3;

fn bench_kabsch(c: &mut Criterion) {
    let p = fixed_data(0x1A, M_POINTS * DIM);
    let r = diag_dominant_data(0x1B, DIM);
    let q = flin::matmul(&p, &r, M_POINTS, DIM, DIM);
    let tol = Q16_16::try_from(1e-4).unwrap();

    let mut g = c.benchmark_group("kabsch_100x3");
    g.throughput(Throughput::Elements((M_POINTS * DIM * DIM) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| flin::kabsch(black_box(&p), black_box(&q), M_POINTS, DIM, tol, 60))
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
    bench_eigenvalues_general,
    bench_eigenvector_real,
    bench_poly_roots,
    bench_matrix_exp,
    bench_generalized_eig_symmetric,
    bench_matrix_sqrt,
    bench_matrix_log,
    bench_kabsch
);
criterion_main!(benches);
