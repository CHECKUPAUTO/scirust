// scirust-simd/benches/cnn_bench.rs
//
// Benchmarks criterion des couches CNN virgule fixe (`fixed::conv`,
// `fixed::pool`), comparées à une baseline flottante `f32` naïve.
//
// Mesure le **débit** (multiplications-accumulations/s pour la convolution,
// éléments/s pour le pooling) de `conv1d` (im2col + GEMM) et de
// `max_pool1d`/`avg_pool1d`, pour `Q16_16` (virgule fixe, déterministe) face à
// une implémentation `f32` directe. L'objectif est de situer le coût relatif,
// pas de « battre » le flottant : la virgule fixe apporte le **déterminisme
// bit-à-bit**, à un coût qui doit rester raisonnable.
//
// Lancement (cible AVX2 pour éviter la sur-détection AVX-512 en VM) :
//   RUSTFLAGS="-C target-cpu=x86-64-v3" \
//     cargo bench -p scirust-simd --features portable-simd --bench cnn_bench

// Migrating this file's results onto scirust-bench-schema::BenchRecord:
// inputs are seeded by the file-local `Lcg(u64)` (fixed-seed LCG, no OS/clock
// entropy) via `fixed_data(seed, len)` / `f32_data(seed, len)`; each call site
// passes its own literal seed -- e.g. `bench_conv1d` uses `fixed_data(0x1, ..)`
// for `x`, `0x2` for `w`, `0x3` for `b` (mirrored by `f32_data` with the same
// seeds for the f32 baseline); `bench_pool1d` uses `fixed_data(0x4, ..)`;
// `bench_conv1d_batch` uses `0x5`/`0x6`/`0x7`. Example, after
// `cargo bench -p scirust-simd --features portable-simd --bench cnn_bench`,
// converting the "conv1d" group's "fixed/Q16_16" result (its `x` input,
// seeded 0x1):
//
//   let json = std::fs::read_to_string(
//       "target/criterion/conv1d/fixed/Q16_16/new/estimates.json",
//   ).unwrap();
//   let record = scirust_bench_schema::criterion_estimate_to_record(
//       &json,
//       "scirust-simd/conv1d",
//       "in=8/len=1024/out=16/k=9",
//       "Q16_16",
//       0x1,
//   ).unwrap();
//
// See scirust-bench-schema's crate docs ("Migrating criterion targets") for the full pattern.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_simd::fixed::Q16_16;
use scirust_simd::fixed::conv::{Conv1dShape, conv1d, conv1d_batch};
use scirust_simd::fixed::pool::{Pool1dShape, avg_pool1d, max_pool1d};

/// Taille de lot pour `bench_conv1d_batch`.
const BATCH: usize = 8;

/// 8 canaux, longueur 1024 : entrée type d'une couche convolutive audio.
const IN_CHANNELS: usize = 8;
const LENGTH: usize = 1024;
const OUT_CHANNELS: usize = 16;
const KERNEL_SIZE: usize = 9;

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

/// Convolution 1D flottante naïve (référence non déterministe) : mêmes
/// conventions que `conv1d` (poids `out×in×kernel`, biais `out`).
fn naive_conv1d_f32(
    x: &[f32],
    weights: &[f32],
    bias: &[f32],
    in_channels: usize,
    length: usize,
    out_channels: usize,
    kernel_size: usize,
) -> Vec<f32> {
    let length_out = length - kernel_size + 1;
    let mut y = vec![0.0f32; out_channels * length_out];
    for co in 0..out_channels
    {
        for j in 0..length_out
        {
            let mut acc = bias[co];
            for ci in 0..in_channels
            {
                for k in 0..kernel_size
                {
                    acc += weights[co * (in_channels * kernel_size) + ci * kernel_size + k]
                        * x[ci * length + j + k];
                }
            }
            y[co * length_out + j] = acc;
        }
    }
    y
}

fn bench_conv1d(c: &mut Criterion) {
    let shape = Conv1dShape {
        in_channels: IN_CHANNELS,
        length: LENGTH,
        out_channels: OUT_CHANNELS,
        kernel_size: KERNEL_SIZE,
        stride: 1,
    };
    let x = fixed_data(0x1, IN_CHANNELS * LENGTH);
    let w = fixed_data(0x2, OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE);
    let b = fixed_data(0x3, OUT_CHANNELS);
    let fx = f32_data(0x1, IN_CHANNELS * LENGTH);
    let fw = f32_data(0x2, OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE);
    let fb = f32_data(0x3, OUT_CHANNELS);

    let mac_count = (OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * shape.length_out()) as u64;
    let mut g = c.benchmark_group("conv1d");
    g.throughput(Throughput::Elements(mac_count));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| conv1d(black_box(&x), black_box(&w), black_box(&b), shape))
    });
    g.bench_function(BenchmarkId::new("f32", "naive"), |bch| {
        bch.iter(|| {
            naive_conv1d_f32(
                black_box(&fx),
                black_box(&fw),
                black_box(&fb),
                IN_CHANNELS,
                LENGTH,
                OUT_CHANNELS,
                KERNEL_SIZE,
            )
        })
    });
    g.finish();
}

fn bench_pool1d(c: &mut Criterion) {
    let shape = Pool1dShape {
        channels: IN_CHANNELS,
        length: LENGTH,
        window: 4,
        stride: 4,
    };
    let x = fixed_data(0x4, IN_CHANNELS * LENGTH);

    let mut g = c.benchmark_group("max_pool1d");
    g.throughput(Throughput::Elements((IN_CHANNELS * LENGTH) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| max_pool1d(black_box(&x), shape))
    });
    g.finish();

    let mut g = c.benchmark_group("avg_pool1d");
    g.throughput(Throughput::Elements((IN_CHANNELS * LENGTH) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| avg_pool1d(black_box(&x), shape))
    });
    g.finish();
}

/// `conv1d_batch` (un seul GEMM sur tout le lot) vs `BATCH` appels de
/// `conv1d` (un GEMM chacun) — même résultat bit-à-bit (cf. tests), débit
/// différent.
fn bench_conv1d_batch(c: &mut Criterion) {
    let shape = Conv1dShape {
        in_channels: IN_CHANNELS,
        length: LENGTH,
        out_channels: OUT_CHANNELS,
        kernel_size: KERNEL_SIZE,
        stride: 1,
    };
    let x = fixed_data(0x5, BATCH * IN_CHANNELS * LENGTH);
    let w = fixed_data(0x6, OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE);
    let b = fixed_data(0x7, OUT_CHANNELS);

    let mac_count = (BATCH * OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * shape.length_out()) as u64;
    let mut g = c.benchmark_group("conv1d_batch8");
    g.throughput(Throughput::Elements(mac_count));
    g.bench_function(BenchmarkId::new("fixed", "batched"), |bch| {
        bch.iter(|| conv1d_batch(black_box(&x), BATCH, black_box(&w), black_box(&b), shape))
    });
    let sample_len = IN_CHANNELS * LENGTH;
    g.bench_function(BenchmarkId::new("fixed", "looped"), |bch| {
        bch.iter(|| {
            let mut out = Vec::with_capacity(BATCH * OUT_CHANNELS * shape.length_out());
            for sample in black_box(&x).chunks_exact(sample_len)
            {
                out.extend(conv1d(sample, &w, &b, shape));
            }
            out
        })
    });
    g.finish();
}

criterion_group!(benches, bench_conv1d, bench_pool1d, bench_conv1d_batch);
criterion_main!(benches);
