// scirust-simd/benches/norm_bench.rs
//
// Benchmarks criterion des normalisations Transformer virgule fixe
// (`fixed::norm`), comparées à une baseline flottante `f32` naïve (référence
// non déterministe, même structure que `crate::norm` mais sans dispatch
// AVX-512 — celui-ci est gardé par la feature optionnelle
// `transformer-inference`, non requise ici).
//
// Mesure le **débit** (éléments/s) de `rmsnorm`/`layer_norm`/`rope_apply` pour
// `Q16_16` (virgule fixe, déterministe — `d` divisions réelles vérifiées par
// ligne pour les normalisations, cf. doc de tête de module) face à `f32`.
// L'objectif est de situer le coût relatif, pas de « battre » le flottant.
//
// Lancement (cible AVX2 pour éviter la sur-détection AVX-512 en VM) :
//   RUSTFLAGS="-C target-cpu=x86-64-v3" \
//     cargo bench -p scirust-simd --features portable-simd --bench norm_bench

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_simd::fixed::Q16_16;
use scirust_simd::fixed::norm::{batch_norm, layer_norm, rmsnorm, rope_apply};

/// 32 lignes de 256 canaux : taille type d'une normalisation Transformer.
const ROWS: usize = 32;
const D: usize = 256;

/// 32 canaux, 8×8 spatial : taille type d'une carte de caractéristiques CNN
/// après quelques convolutions/poolings.
const CHANNELS: usize = 32;
const SPATIAL: usize = 64;

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
    (0..len)
        .map(|_| Q16_16::try_from(rng.unit()).unwrap())
        .collect()
}
fn f32_data(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..len).map(|_| rng.unit() as f32).collect()
}

/// RMSNorm flottante naïve (référence non déterministe), en place.
fn naive_rmsnorm_f32(x: &mut [f32], d: usize, gamma: &[f32], eps: f32) {
    for row in x.chunks_exact_mut(d)
    {
        let ss: f32 = row.iter().map(|&v| v * v).sum();
        let rms = (ss / d as f32 + eps).sqrt();
        for (v, &g) in row.iter_mut().zip(gamma)
        {
            *v = *v / rms * g;
        }
    }
}

/// LayerNorm flottante naïve (référence non déterministe), en place.
fn naive_layer_norm_f32(x: &mut [f32], d: usize, gamma: &[f32], beta: &[f32], eps: f32) {
    for row in x.chunks_exact_mut(d)
    {
        let mean: f32 = row.iter().sum::<f32>() / d as f32;
        let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let denom = (var + eps).sqrt();
        for ((v, &g), &b) in row.iter_mut().zip(gamma).zip(beta)
        {
            *v = (*v - mean) / denom * g + b;
        }
    }
}

fn bench_rmsnorm(c: &mut Criterion) {
    let x = fixed_data(0x1, ROWS * D);
    let gamma = fixed_data(0x2, D);
    let eps = Q16_16::try_from(1e-3).unwrap();
    let fx = f32_data(0x1, ROWS * D);
    let fgamma = f32_data(0x2, D);

    let mut g = c.benchmark_group("rmsnorm_32x256");
    g.throughput(Throughput::Elements((ROWS * D) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| rmsnorm(black_box(&x), ROWS, D, black_box(&gamma), eps))
    });
    g.bench_function(BenchmarkId::new("f32", "naive"), |bch| {
        bch.iter_batched(
            || fx.clone(),
            |mut buf| {
                naive_rmsnorm_f32(black_box(&mut buf), D, black_box(&fgamma), 1e-3);
                buf
            },
            criterion::BatchSize::SmallInput,
        )
    });
    g.finish();
}

fn bench_layer_norm(c: &mut Criterion) {
    let x = fixed_data(0x3, ROWS * D);
    let gamma = fixed_data(0x4, D);
    let beta = fixed_data(0x5, D);
    let eps = Q16_16::try_from(1e-3).unwrap();
    let fx = f32_data(0x3, ROWS * D);
    let fgamma = f32_data(0x4, D);
    let fbeta = f32_data(0x5, D);

    let mut g = c.benchmark_group("layer_norm_32x256");
    g.throughput(Throughput::Elements((ROWS * D) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| {
            layer_norm(
                black_box(&x),
                ROWS,
                D,
                black_box(&gamma),
                black_box(&beta),
                eps,
            )
        })
    });
    g.bench_function(BenchmarkId::new("f32", "naive"), |bch| {
        bch.iter_batched(
            || fx.clone(),
            |mut buf| {
                naive_layer_norm_f32(
                    black_box(&mut buf),
                    D,
                    black_box(&fgamma),
                    black_box(&fbeta),
                    1e-3,
                );
                buf
            },
            criterion::BatchSize::SmallInput,
        )
    });
    g.finish();
}

/// BatchNorm flottante naïve (référence non déterministe, inférence :
/// statistiques déjà fournies, comme `batch_norm`).
#[allow(clippy::too_many_arguments)]
fn naive_batch_norm_f32(
    x: &[f32],
    channels: usize,
    spatial: usize,
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut y = vec![0.0f32; channels * spatial];
    for c in 0..channels
    {
        let denom = (var[c] + eps).sqrt();
        for s in 0..spatial
        {
            y[c * spatial + s] = (x[c * spatial + s] - mean[c]) / denom * gamma[c] + beta[c];
        }
    }
    y
}

fn bench_batch_norm(c: &mut Criterion) {
    let x = fixed_data(0x7, CHANNELS * SPATIAL);
    let mean = fixed_data(0x8, CHANNELS);
    let var = fixed_data(0x9, CHANNELS);
    let gamma = fixed_data(0xA, CHANNELS);
    let beta = fixed_data(0xB, CHANNELS);
    let eps = Q16_16::try_from(1e-3).unwrap();
    let fx = f32_data(0x7, CHANNELS * SPATIAL);
    let fmean = f32_data(0x8, CHANNELS);
    let fvar = f32_data(0x9, CHANNELS);
    let fgamma = f32_data(0xA, CHANNELS);
    let fbeta = f32_data(0xB, CHANNELS);

    let mut g = c.benchmark_group("batch_norm_32x64");
    g.throughput(Throughput::Elements((CHANNELS * SPATIAL) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| {
            batch_norm(
                black_box(&x),
                CHANNELS,
                SPATIAL,
                black_box(&mean),
                black_box(&var),
                black_box(&gamma),
                black_box(&beta),
                eps,
            )
        })
    });
    g.bench_function(BenchmarkId::new("f32", "naive"), |bch| {
        bch.iter(|| {
            naive_batch_norm_f32(
                black_box(&fx),
                CHANNELS,
                SPATIAL,
                black_box(&fmean),
                black_box(&fvar),
                black_box(&fgamma),
                black_box(&fbeta),
                1e-3,
            )
        })
    });
    g.finish();
}

/// RoPE flottante naïve (référence non déterministe), en place.
fn naive_rope_f32(x: &mut [f32], d: usize, base: f32, pos_offset: usize) {
    let half = d / 2;
    for (r, row) in x.chunks_exact_mut(d).enumerate()
    {
        let pos = (pos_offset + r) as f32;
        for i in 0..half
        {
            let theta = base.powf(-2.0 * i as f32 / d as f32);
            let angle = pos * theta;
            let (s, c) = angle.sin_cos();
            let a = row[2 * i];
            let b = row[2 * i + 1];
            row[2 * i] = a * c - b * s;
            row[2 * i + 1] = a * s + b * c;
        }
    }
}

fn bench_rope(c: &mut Criterion) {
    let x = fixed_data(0x6, ROWS * D);
    let base = Q16_16::try_from(10000.0).unwrap();
    let fx = f32_data(0x6, ROWS * D);

    let mut g = c.benchmark_group("rope_apply_32x256");
    g.throughput(Throughput::Elements((ROWS * D) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter_batched(
            || x.clone(),
            |mut buf| {
                rope_apply(black_box(&mut buf), ROWS, D, base, 0);
                buf
            },
            criterion::BatchSize::SmallInput,
        )
    });
    g.bench_function(BenchmarkId::new("f32", "naive"), |bch| {
        bch.iter_batched(
            || fx.clone(),
            |mut buf| {
                naive_rope_f32(black_box(&mut buf), D, 10000.0, 0);
                buf
            },
            criterion::BatchSize::SmallInput,
        )
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_rmsnorm,
    bench_layer_norm,
    bench_batch_norm,
    bench_rope
);
criterion_main!(benches);
