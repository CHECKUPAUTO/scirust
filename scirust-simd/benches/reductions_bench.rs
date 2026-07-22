// scirust-simd/benches/reductions_bench.rs
//
// Benchmarks criterion du socle de réductions.
//
// Compare, pour f32 et f64, les variantes SIMD (fast / déterministe / Kahan)
// à la baseline scalaire naïve (`Iterator::sum`), plus `dot` et `l2_norm`.
// Débit en éléments/s.
//
//   RUSTFLAGS="-C target-cpu=native" \
//     cargo bench -p scirust-simd --features portable-simd --bench reductions_bench
//
// Note : dans certains environnements virtualisés, `-C target-cpu=native`
// sur-détecte des extensions AVX-512 (ex. avx512vnni) que le CPU exposé ne sait
// pas exécuter, ce qui peut provoquer un SIGILL. Le cas échéant, préférer une
// cible explicite : `-C target-cpu=x86-64-v3` (AVX2) ou `x86-64-v4` (AVX-512).

// Migration note (scirust-bench-schema): inputs come from `data_f32`/
// `data_f64(seed)`, backed by the in-file `Lcg`; bench_sum pins a32 to
// 0x501 and a64 to 0x502. N=65536. Example conversion for the "sum_f32"
// group's fast-variant case (after `cargo bench --bench reductions_bench`,
// reading target/criterion/sum_f32/.../new/estimates.json):
//
//   scirust_bench_schema::criterion_estimate_to_record(
//       &estimates_json,
//       "scirust-simd/reductions_sum_f32", // kernel
//       "N=65536",                           // dataset
//       "sum_fast",                          // method
//       0x501,                               // seed: a32's data_f32() seed
//   )
// See scirust-bench-schema's crate docs ("Migrating criterion targets") for the full pattern.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_simd::reductions::{
    ReductionMode, argmin, dot, l2_norm, linf_norm, mean, sum_deterministic, sum_fast, sum_kahan,
    variance_population,
};

/// 65 536 éléments : le tableau f32 (256 Kio) déborde le L1/L2 → débit soutenu.
const N: usize = 1 << 16;

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

fn data_f32(seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..N).map(|_| rng.unit() as f32).collect()
}
fn data_f64(seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..N).map(|_| rng.unit()).collect()
}

fn bench_sum(c: &mut Criterion) {
    let a32 = data_f32(0x501);
    let a64 = data_f64(0x502);

    let mut g = c.benchmark_group("sum_f32");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("scalar", "naive"), |b| {
        b.iter(|| black_box(&a32).iter().copied().sum::<f32>())
    });
    g.bench_function(BenchmarkId::new("simd", "fast"), |b| {
        b.iter(|| sum_fast(black_box(&a32)))
    });
    g.bench_function(BenchmarkId::new("simd", "deterministic"), |b| {
        b.iter(|| sum_deterministic(black_box(&a32)))
    });
    g.bench_function(BenchmarkId::new("simd", "kahan"), |b| {
        b.iter(|| sum_kahan(black_box(&a32)))
    });
    g.finish();

    let mut g = c.benchmark_group("sum_f64");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("scalar", "naive"), |b| {
        b.iter(|| black_box(&a64).iter().copied().sum::<f64>())
    });
    g.bench_function(BenchmarkId::new("simd", "fast"), |b| {
        b.iter(|| sum_fast(black_box(&a64)))
    });
    g.bench_function(BenchmarkId::new("simd", "kahan"), |b| {
        b.iter(|| sum_kahan(black_box(&a64)))
    });
    g.finish();
}

fn bench_dot_norm(c: &mut Criterion) {
    let a = data_f32(0x601);
    let b_ = data_f32(0x602);

    let mut g = c.benchmark_group("dot_f32");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("scalar", "naive"), |bch| {
        bch.iter(|| {
            black_box(&a)
                .iter()
                .zip(black_box(&b_))
                .map(|(x, y)| x * y)
                .sum::<f32>()
        })
    });
    g.bench_function(BenchmarkId::new("simd", "fast"), |bch| {
        bch.iter(|| dot(black_box(&a), black_box(&b_), ReductionMode::Fast))
    });
    g.finish();

    let mut g = c.benchmark_group("l2_norm_f32");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("simd", "fast"), |bch| {
        bch.iter(|| l2_norm(black_box(&a), ReductionMode::Fast))
    });
    g.finish();
}

/// `linf_norm` (max absolu, SIMD) et `argmin` (min SIMD + balayage linéaire)
/// vs équivalents scalaires naïfs.
fn bench_linf_argmin(c: &mut Criterion) {
    let a = data_f32(0x701);

    let mut g = c.benchmark_group("linf_norm_f32");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("scalar", "naive"), |bch| {
        bch.iter(|| black_box(&a).iter().fold(0.0f32, |m, &x| m.max(x.abs())))
    });
    g.bench_function(BenchmarkId::new("simd", "fast"), |bch| {
        bch.iter(|| linf_norm(black_box(&a)))
    });
    g.finish();

    let mut g = c.benchmark_group("argmin_f32");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("scalar", "naive"), |bch| {
        bch.iter(|| {
            black_box(&a)
                .iter()
                .enumerate()
                .min_by(|(_, x), (_, y)| x.total_cmp(y))
                .map(|(i, _)| i)
        })
    });
    g.bench_function(BenchmarkId::new("simd", "fast"), |bch| {
        bch.iter(|| argmin(black_box(&a)))
    });
    g.finish();
}

/// `mean`/`variance_population` (Welford lane-parallèle + fusion de Chan)
/// vs référence scalaire naïve **deux passes** (`Σx/n` puis `Σ(x−mean)²/n`) —
/// mesure le surcoût de l'algorithme en ligne face à la formule directe,
/// pour la même exactitude sur données bien conditionnées (cf. en-tête de
/// module de `reductions::mod`).
fn bench_moments(c: &mut Criterion) {
    let a = data_f32(0x801);

    let mut g = c.benchmark_group("mean_f32");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("scalar", "naive"), |bch| {
        bch.iter(|| black_box(&a).iter().copied().sum::<f32>() / N as f32)
    });
    g.bench_function(BenchmarkId::new("simd", "fast"), |bch| {
        bch.iter(|| mean(black_box(&a), ReductionMode::Fast))
    });
    g.finish();

    let mut g = c.benchmark_group("variance_population_f32");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("scalar", "naive_two_pass"), |bch| {
        bch.iter(|| {
            let data = black_box(&a);
            let m = data.iter().copied().sum::<f32>() / N as f32;
            data.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / N as f32
        })
    });
    g.bench_function(BenchmarkId::new("simd", "welford"), |bch| {
        bch.iter(|| variance_population(black_box(&a)))
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_sum,
    bench_dot_norm,
    bench_linf_argmin,
    bench_moments
);
criterion_main!(benches);
