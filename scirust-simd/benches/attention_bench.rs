// scirust-simd/benches/attention_bench.rs
//
// Benchmarks criterion de l'attention produit-scalaire virgule fixe
// (`fixed::attention`), comparée à une baseline flottante `f32` naïve
// (référence non déterministe, même structure que `crate::attention` mais
// sans dispatch AVX-512/NEON — celui-ci est gardé par la feature optionnelle
// `transformer-inference`, non requise ici).
//
// Mesure le **débit** de `attention` (une tête), `causal_attention` (masquage
// causal, décodeur/LLM) et `multi_head_attention`, pour `Q16_16` (virgule
// fixe, déterministe) face à `f32`. L'objectif est de situer le coût relatif,
// pas de « battre » le flottant : la virgule fixe apporte le **déterminisme
// bit-à-bit**, à un coût qui doit rester raisonnable.
//
// Lancement (cible AVX2 pour éviter la sur-détection AVX-512 en VM) :
//   RUSTFLAGS="-C target-cpu=x86-64-v3" \
//     cargo bench -p scirust-simd --features portable-simd --bench attention_bench

// Migration des résultats de ce fichier vers scirust-bench-schema::BenchRecord :
// chaque fonction de benchmark seede son propre générateur en inline via
// fixed_data(seed, len) / f32_data(seed, len) (LCG à seed fixe, voir la
// struct Lcg ci-dessous) -- bench_attention utilise 0x1/0x2/0x3 (q/k/v),
// bench_causal_attention 0x4/0x5/0x6, bench_multi_head_attention 0x7/0x8/0x9.
// Exemple, après `cargo bench -p scirust-simd --features portable-simd
// --bench attention_bench`, conversion du résultat "fixed/Q16_16" du groupe
// "attention_64x64x32" (entrée q seedée par fixed_data(0x1, ...)) :
//
//   let json = std::fs::read_to_string(
//       "target/criterion/attention_64x64x32/fixed/Q16_16/new/estimates.json",
//   ).unwrap();
//   let record = scirust_bench_schema::criterion_estimate_to_record(
//       &json,
//       "scirust-simd/attention",
//       "64x64x32",
//       "Q16_16",
//       0x1,
//   ).unwrap();
//
// See scirust-bench-schema's crate docs ("Migrating criterion targets") for the full pattern.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_simd::fixed::Q16_16;
use scirust_simd::fixed::attention::{attention, causal_attention, multi_head_attention};

/// Séquence de 64 tokens, dimension de modèle 32 : taille type d'un petit
/// bloc d'attention embarqué/edge.
const S: usize = 64;
const T: usize = 64;
const D: usize = 32;
const H: usize = 4;
const DH: usize = D / H;

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

/// Attention flottante naïve (référence non déterministe) : mêmes
/// conventions que `fixed::attention::attention`.
fn naive_attention_f32(
    q: &[f32],
    s: usize,
    d: usize,
    k: &[f32],
    t: usize,
    v: &[f32],
    scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; s * t];
    for i in 0..s
    {
        for j in 0..t
        {
            let mut acc = 0.0f32;
            for e in 0..d
            {
                acc += q[i * d + e] * k[j * d + e];
            }
            scores[i * t + j] = scale * acc;
        }
    }
    for row in scores.chunks_exact_mut(t)
    {
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in row.iter_mut()
        {
            *x = (*x - m).exp();
            sum += *x;
        }
        for x in row.iter_mut()
        {
            *x /= sum;
        }
    }
    let mut out = vec![0.0f32; s * d];
    for i in 0..s
    {
        for e in 0..d
        {
            let mut acc = 0.0f32;
            for j in 0..t
            {
                acc += scores[i * t + j] * v[j * d + e];
            }
            out[i * d + e] = acc;
        }
    }
    out
}

/// `Q·Kᵀ` puis `P·V` : deux GEMM de `S·T·D` MAC chacun.
fn bench_attention(c: &mut Criterion) {
    let q = fixed_data(0x1, S * D);
    let k = fixed_data(0x2, T * D);
    let v = fixed_data(0x3, T * D);
    let scale = Q16_16::try_from(1.0 / (D as f64).sqrt()).unwrap();
    let fq = f32_data(0x1, S * D);
    let fk = f32_data(0x2, T * D);
    let fv = f32_data(0x3, T * D);
    let fscale = 1.0 / (D as f32).sqrt();

    let mac_count = (2 * S * T * D) as u64;
    let mut g = c.benchmark_group("attention_64x64x32");
    g.throughput(Throughput::Elements(mac_count));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| attention(black_box(&q), S, D, black_box(&k), T, black_box(&v), scale))
    });
    g.bench_function(BenchmarkId::new("f32", "naive"), |bch| {
        bch.iter(|| {
            naive_attention_f32(
                black_box(&fq),
                S,
                D,
                black_box(&fk),
                T,
                black_box(&fv),
                fscale,
            )
        })
    });
    g.finish();
}

/// Masquage causal : travail borné au triangle inférieur (~moitié de
/// `attention`).
fn bench_causal_attention(c: &mut Criterion) {
    let q = fixed_data(0x4, S * D);
    let k = fixed_data(0x5, S * D);
    let v = fixed_data(0x6, S * D);
    let scale = Q16_16::try_from(1.0 / (D as f64).sqrt()).unwrap();

    let mac_count = (S * S * D) as u64; // ordre de grandeur (triangle, pas plein)
    let mut g = c.benchmark_group("causal_attention_64x32");
    g.throughput(Throughput::Elements(mac_count));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| causal_attention(black_box(&q), S, D, black_box(&k), black_box(&v), scale))
    });
    g.finish();
}

/// `H` têtes de dimension `D/H` chacune.
fn bench_multi_head_attention(c: &mut Criterion) {
    let q = fixed_data(0x7, S * D);
    let k = fixed_data(0x8, T * D);
    let v = fixed_data(0x9, T * D);
    let scale = Q16_16::try_from(1.0 / (DH as f64).sqrt()).unwrap();

    let mac_count = (2 * S * T * D) as u64;
    let mut g = c.benchmark_group("multi_head_attention_64x64x32_h4");
    g.throughput(Throughput::Elements(mac_count));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| {
            multi_head_attention(
                black_box(&q),
                S,
                T,
                H,
                DH,
                black_box(&k),
                black_box(&v),
                scale,
                false,
            )
        })
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_attention,
    bench_causal_attention,
    bench_multi_head_attention
);
criterion_main!(benches);
