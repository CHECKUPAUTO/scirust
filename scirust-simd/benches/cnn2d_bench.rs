// scirust-simd/benches/cnn2d_bench.rs
//
// Benchmarks criterion des couches CNN 2D virgule fixe (`fixed::conv2d`,
// `fixed::pool2d`), comparées à une baseline flottante `f32` naïve. Même
// structure que `cnn_bench` (variantes 1D), pour les données à deux
// dimensions spatiales (images, spectrogrammes).
//
// Mesure le **débit** (multiplications-accumulations/s pour la convolution,
// éléments/s pour le pooling) de `conv2d` (im2col2d + GEMM) et de
// `max_pool2d`/`avg_pool2d`, pour `Q16_16` (virgule fixe, déterministe) face à
// une implémentation `f32` directe. L'objectif est de situer le coût relatif,
// pas de « battre » le flottant : la virgule fixe apporte le **déterminisme
// bit-à-bit**, à un coût qui doit rester raisonnable.
//
// Lancement (cible AVX2 pour éviter la sur-détection AVX-512 en VM) :
//   RUSTFLAGS="-C target-cpu=x86-64-v3" \
//     cargo bench -p scirust-simd --features portable-simd --bench cnn2d_bench

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_simd::fixed::Q16_16;
use scirust_simd::fixed::conv2d::{Conv2dShape, conv2d};
use scirust_simd::fixed::pool2d::{Pool2dShape, avg_pool2d, max_pool2d};

/// 4 canaux, 32×32 : entrée type d'une couche convolutive image/spectrogramme.
const IN_CHANNELS: usize = 4;
const HEIGHT: usize = 32;
const WIDTH: usize = 32;
const OUT_CHANNELS: usize = 8;
const KERNEL: usize = 3;

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

/// Convolution 2D flottante naïve (référence non déterministe) : mêmes
/// conventions que `conv2d` (poids `out×in×kh×kw`, biais `out`).
#[allow(clippy::too_many_arguments)]
fn naive_conv2d_f32(
    x: &[f32],
    weights: &[f32],
    bias: &[f32],
    in_channels: usize,
    height: usize,
    width: usize,
    out_channels: usize,
    kernel: usize,
) -> Vec<f32> {
    let height_out = height - kernel + 1;
    let width_out = width - kernel + 1;
    let mut y = vec![0.0f32; out_channels * height_out * width_out];
    for co in 0..out_channels
    {
        for oh in 0..height_out
        {
            for ow in 0..width_out
            {
                let mut acc = bias[co];
                for ci in 0..in_channels
                {
                    for kh in 0..kernel
                    {
                        for kw in 0..kernel
                        {
                            acc += weights[co * (in_channels * kernel * kernel)
                                + ci * (kernel * kernel)
                                + kh * kernel
                                + kw]
                                * x[ci * (height * width) + (oh + kh) * width + (ow + kw)];
                        }
                    }
                }
                y[co * (height_out * width_out) + oh * width_out + ow] = acc;
            }
        }
    }
    y
}

fn bench_conv2d(c: &mut Criterion) {
    let shape = Conv2dShape {
        in_channels: IN_CHANNELS,
        height: HEIGHT,
        width: WIDTH,
        out_channels: OUT_CHANNELS,
        kernel_h: KERNEL,
        kernel_w: KERNEL,
        stride_h: 1,
        stride_w: 1,
    };
    let x = fixed_data(0x1, IN_CHANNELS * HEIGHT * WIDTH);
    let w = fixed_data(0x2, OUT_CHANNELS * IN_CHANNELS * KERNEL * KERNEL);
    let b = fixed_data(0x3, OUT_CHANNELS);
    let fx = f32_data(0x1, IN_CHANNELS * HEIGHT * WIDTH);
    let fw = f32_data(0x2, OUT_CHANNELS * IN_CHANNELS * KERNEL * KERNEL);
    let fb = f32_data(0x3, OUT_CHANNELS);

    let mac_count =
        (OUT_CHANNELS * IN_CHANNELS * KERNEL * KERNEL * shape.height_out() * shape.width_out())
            as u64;
    let mut g = c.benchmark_group("conv2d");
    g.throughput(Throughput::Elements(mac_count));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| conv2d(black_box(&x), black_box(&w), black_box(&b), shape))
    });
    g.bench_function(BenchmarkId::new("f32", "naive"), |bch| {
        bch.iter(|| {
            naive_conv2d_f32(
                black_box(&fx),
                black_box(&fw),
                black_box(&fb),
                IN_CHANNELS,
                HEIGHT,
                WIDTH,
                OUT_CHANNELS,
                KERNEL,
            )
        })
    });
    g.finish();
}

fn bench_pool2d(c: &mut Criterion) {
    let shape = Pool2dShape {
        channels: IN_CHANNELS,
        height: HEIGHT,
        width: WIDTH,
        window_h: 2,
        window_w: 2,
        stride_h: 2,
        stride_w: 2,
    };
    let x = fixed_data(0x4, IN_CHANNELS * HEIGHT * WIDTH);

    let mut g = c.benchmark_group("max_pool2d");
    g.throughput(Throughput::Elements((IN_CHANNELS * HEIGHT * WIDTH) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| max_pool2d(black_box(&x), shape))
    });
    g.finish();

    let mut g = c.benchmark_group("avg_pool2d");
    g.throughput(Throughput::Elements((IN_CHANNELS * HEIGHT * WIDTH) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| avg_pool2d(black_box(&x), shape))
    });
    g.finish();
}

criterion_group!(benches, bench_conv2d, bench_pool2d);
criterion_main!(benches);
