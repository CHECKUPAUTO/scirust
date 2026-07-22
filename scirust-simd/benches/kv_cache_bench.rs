// scirust-simd/benches/kv_cache_bench.rs
//
// Benchmarks criterion du cache KV virgule fixe (`fixed::kv_cache`),
// déterministe (le module flottant `crate::kv_cache` est gardé par la
// feature optionnelle `transformer-inference`, non requise ici).
//
// * `kv_cache_decode_64_steps` : débit du décodage incrémental complet
//   (`append` + `decode_step` à chaque pas) sur une séquence de 64 tokens.
// * `kv_cache_vs_batched_causal_64` : coût d'un pas incrémental cumulé sur
//   toute la séquence face à un **unique** appel `multi_head_attention`
//   causale sur la séquence entière. Comparaison purement informative — en
//   génération autoregressive réelle, le mode « batché » n'est **pas** une
//   alternative valable (les tokens futurs sont inconnus au moment de la
//   génération) ; elle ne fait que quantifier le surcoût du traitement pas à
//   pas face à un unique GEMM sur toute la séquence.
//
// Lancement (cible AVX2 pour éviter la sur-détection AVX-512 en VM) :
//   RUSTFLAGS="-C target-cpu=x86-64-v3" \
//     cargo bench -p scirust-simd --features portable-simd --bench kv_cache_bench

// Migration note (scirust-bench-schema): inputs come from `fixed_data(seed,
// len)`, backed by the in-file `Lcg`; bench_kv_cache_incremental_decode
// pins q/k/v to seeds 0x1/0x2/0x3. S=64 tokens, DM=64. Example conversion
// for the "kv_cache_decode_64_steps" group's "fixed"/"Q16_16" case (after
// `cargo bench --bench kv_cache_bench`, reading
// target/criterion/kv_cache_decode_64_steps/fixed/Q16_16/new/estimates.json):
//
//   scirust_bench_schema::criterion_estimate_to_record(
//       &estimates_json,
//       "scirust-simd/kv_cache_incremental_decode", // kernel
//       "S=64,DM=64",                                 // dataset
//       "fixed:Q16_16",                               // method
//       0x1,                                           // seed: q's fixed_data() seed
//   )
// See scirust-bench-schema's crate docs ("Migrating criterion targets") for the full pattern.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_simd::fixed::Q16_16;
use scirust_simd::fixed::attention::multi_head_attention;
use scirust_simd::fixed::kv_cache::KvCache;

/// Séquence de 64 tokens, 4 têtes de dimension 16 (`dm = 64`).
const S: usize = 64;
const H: usize = 4;
const DH: usize = 16;
const DM: usize = H * DH;

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

fn bench_kv_cache_incremental_decode(c: &mut Criterion) {
    let q = fixed_data(0x1, S * DM);
    let k = fixed_data(0x2, S * DM);
    let v = fixed_data(0x3, S * DM);
    let scale = Q16_16::try_from(1.0 / (DH as f64).sqrt()).unwrap();

    let mut g = c.benchmark_group("kv_cache_decode_64_steps");
    g.throughput(Throughput::Elements(S as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| {
            let mut cache: KvCache<16> = KvCache::new(S, DM);
            let mut acc = Q16_16::zero();
            for i in 0..S
            {
                cache.append(
                    black_box(&k[i * DM..i * DM + DM]),
                    black_box(&v[i * DM..i * DM + DM]),
                );
                let o = cache.decode_step(black_box(&q[i * DM..i * DM + DM]), H, DH, scale);
                acc += o[0];
            }
            acc
        })
    });
    g.finish();
}

fn bench_kv_cache_vs_batched(c: &mut Criterion) {
    let q = fixed_data(0x4, S * DM);
    let k = fixed_data(0x5, S * DM);
    let v = fixed_data(0x6, S * DM);
    let scale = Q16_16::try_from(1.0 / (DH as f64).sqrt()).unwrap();

    let mut g = c.benchmark_group("kv_cache_vs_batched_causal_64");
    g.bench_function(BenchmarkId::new("fixed", "incremental_kv_cache"), |bch| {
        bch.iter(|| {
            let mut cache: KvCache<16> = KvCache::new(S, DM);
            let mut acc = Q16_16::zero();
            for i in 0..S
            {
                cache.append(
                    black_box(&k[i * DM..i * DM + DM]),
                    black_box(&v[i * DM..i * DM + DM]),
                );
                let o = cache.decode_step(black_box(&q[i * DM..i * DM + DM]), H, DH, scale);
                acc += o[0];
            }
            acc
        })
    });
    g.bench_function(
        BenchmarkId::new("fixed", "batched_causal_one_shot"),
        |bch| {
            bch.iter(|| {
                multi_head_attention(
                    black_box(&q),
                    S,
                    S,
                    H,
                    DH,
                    black_box(&k),
                    black_box(&v),
                    scale,
                    true,
                )
            })
        },
    );
    g.finish();
}

criterion_group!(
    benches,
    bench_kv_cache_incremental_decode,
    bench_kv_cache_vs_batched
);
criterion_main!(benches);
