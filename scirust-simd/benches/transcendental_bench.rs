// scirust-simd/benches/transcendental_bench.rs
//
// Benchmarks criterion des transcendantes en virgule fixe (`Q16_16`), comparées
// aux fonctions `f32` de la bibliothèque standard.
//
// Mesure le **débit** (éléments/s) de `exp`, `ln`, `sin`, `tanh`, `sigmoid`,
// `bessel_i0`, `erf`/`erfc` et de `softmax_into`. Le but n'est pas de battre le
// flottant matériel (qui dispose d'unités dédiées) mais de situer le coût du
// chemin **entièrement entier, déterministe bit-à-bit** — chaque appel est une
// poignée de multiplications `i128` et un polynôme de Horner, sans FPU ni
// table.
//
// Lancement (cible AVX2 pour éviter la sur-détection AVX-512 en VM) :
//   RUSTFLAGS="-C target-cpu=x86-64-v3" \
//     cargo bench -p scirust-simd --features portable-simd --bench transcendental_bench

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_simd::fixed::Q16_16;
use scirust_simd::fixed::RealScalar;
use scirust_simd::fixed::transcendental as tr;

const N: usize = 1 << 14;

struct Lcg(u64);
impl Lcg {
    /// Flottant déterministe dans `[-1, 1)`.
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Données fixes dans `[-scale, scale)`, et leur miroir `f32`.
fn data(seed: u64, scale: f64) -> (Vec<Q16_16>, Vec<f32>) {
    let mut rng = Lcg(seed);
    let mut fx = Vec::with_capacity(N);
    let mut ff = Vec::with_capacity(N);
    for _ in 0..N
    {
        let v = rng.unit() * scale;
        fx.push(Q16_16::try_from(v).unwrap());
        ff.push(v as f32);
    }
    (fx, ff)
}

/// Compare une transcendante virgule fixe et son équivalent `f32`.
fn bench_unary(
    c: &mut Criterion,
    name: &str,
    scale: f64,
    fixed_fn: fn(Q16_16) -> Q16_16,
    float_fn: fn(f32) -> f32,
) {
    let (fx, ff) = data(0xC0FFEE ^ name.len() as u64, scale);
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut acc = Q16_16::zero();
            for &x in black_box(&fx)
            {
                acc += fixed_fn(x);
            }
            acc
        })
    });
    g.bench_function(BenchmarkId::new("f32", "std"), |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for &x in black_box(&ff)
            {
                acc += float_fn(x);
            }
            acc
        })
    });
    g.finish();
}

fn bench_exp(c: &mut Criterion) {
    bench_unary(c, "exp", 5.0, tr::exp::<16>, f32::exp);
}
fn bench_ln(c: &mut Criterion) {
    // ln exige x > 0 : décale dans (0, 6].
    let (mut fx, mut ff) = data(0x111, 3.0);
    for x in &mut fx
    {
        *x += Q16_16::try_from(3.0).unwrap();
    }
    for x in &mut ff
    {
        *x += 3.0;
    }
    let mut g = c.benchmark_group("ln");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut acc = Q16_16::zero();
            for &x in black_box(&fx)
            {
                acc += tr::ln(x);
            }
            acc
        })
    });
    g.bench_function(BenchmarkId::new("f32", "std"), |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for &x in black_box(&ff)
            {
                acc += x.ln();
            }
            acc
        })
    });
    g.finish();
}
fn bench_sin(c: &mut Criterion) {
    bench_unary(c, "sin", 3.0, tr::sin::<16>, f32::sin);
}
fn bench_tanh(c: &mut Criterion) {
    bench_unary(c, "tanh", 4.0, tr::tanh::<16>, f32::tanh);
}
fn bench_sigmoid(c: &mut Criterion) {
    bench_unary(c, "sigmoid", 6.0, tr::sigmoid::<16>, |x| {
        1.0 / (1.0 + (-x).exp())
    });
}
fn bench_atan(c: &mut Criterion) {
    bench_unary(c, "atan", 8.0, tr::atan::<16>, f32::atan);
}
fn bench_acos(c: &mut Criterion) {
    // Domaine [-1, 1] (argument valide de acos).
    bench_unary(c, "acos", 1.0, tr::acos::<16>, f32::acos);
}

/// `f32::bessel_i0` n'existe pas dans `std` : passe par `RealScalar` (pas de
/// fonction libre à référencer directement comme `f32::exp`/`f32::sin`).
fn bessel_i0_f32(x: f32) -> f32 {
    RealScalar::bessel_i0(x)
}
fn bench_bessel_i0(c: &mut Criterion) {
    // Domaine [-8, 8) : couvre le domaine garanti [0, 12] du chemin fixe
    // (I₀ étant paire, les valeurs négatives exercent la même normalisation).
    bench_unary(c, "bessel_i0", 8.0, tr::bessel_i0::<16>, bessel_i0_f32);
}

/// `f32::erf`/`erfc` n'existent pas dans `std` : passe par `RealScalar`.
fn erf_f32(x: f32) -> f32 {
    RealScalar::erf(x)
}
fn erfc_f32(x: f32) -> f32 {
    RealScalar::erfc(x)
}
fn bench_erf(c: &mut Criterion) {
    bench_unary(c, "erf", 4.0, tr::erf::<16>, erf_f32);
}
fn bench_erfc(c: &mut Criterion) {
    bench_unary(c, "erfc", 4.0, tr::erfc::<16>, erfc_f32);
}

/// Softmax sur un vecteur (activation déterministe, deux passes).
fn bench_softmax(c: &mut Criterion) {
    let (fx, ff) = data(0x50F7, 4.0);
    let mut out = vec![Q16_16::zero(); N];
    let mut g = c.benchmark_group("softmax");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            tr::softmax_into(black_box(&fx), black_box(&mut out));
            out[0]
        })
    });
    g.bench_function(BenchmarkId::new("f32", "std"), |b| {
        b.iter(|| {
            let mut mx = f32::NEG_INFINITY;
            for &x in black_box(&ff)
            {
                mx = mx.max(x);
            }
            let mut sum = 0.0f32;
            for &x in &ff
            {
                sum += (x - mx).exp();
            }
            (ff[0] - mx).exp() / sum
        })
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_atan,
    bench_acos,
    bench_exp,
    bench_ln,
    bench_sin,
    bench_tanh,
    bench_sigmoid,
    bench_softmax,
    bench_bessel_i0,
    bench_erf,
    bench_erfc
);
criterion_main!(benches);
