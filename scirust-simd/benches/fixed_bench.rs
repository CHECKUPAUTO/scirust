#![feature(portable_simd)]
// le bench utilise `std::simd::f32x8` comme référence
// `chunks_exact(8)` est délibéré (chargement d'un vecteur de 8 lanes) ; le lint
// nightly `chunks_exact_to_as_chunks` ne s'applique pas au benchmark.
#![allow(clippy::chunks_exact_to_as_chunks)]
//
// scirust-simd/benches/fixed_bench.rs
//
// Benchmarks criterion du sous-système virgule fixe, comparé au flottant `f32`.
//
// Mesure le **débit** (éléments/s) de : addition, multiplication, produit
// scalaire, somme, norme L2 et similarité cosinus, pour `Q16_16` (virgule fixe)
// et `f32` (référence). L'objectif est de situer le coût relatif, pas de
// « battre » le flottant : la virgule fixe apporte le **déterminisme**, à un
// coût qui doit rester raisonnable.
//
// Lancement (cible AVX2 pour éviter la sur-détection AVX-512 en VM) :
//   RUSTFLAGS="-C target-cpu=x86-64-v3" \
//     cargo bench -p scirust-simd --features portable-simd --bench fixed_bench

// Migration note (scirust-bench-schema): inputs come from `fixed_data`/
// `f32_data(seed)`, backed by the in-file `Lcg`; each bench function pins a
// literal seed (bench_add=0x1, bench_mul=0x2, ...). N=65536. Example
// conversion for the "add" group's "fixed"/"Q16_16x8" case (after `cargo
// bench --bench fixed_bench`, reading
// target/criterion/add/fixed/Q16_16x8/new/estimates.json):
//
//   scirust_bench_schema::criterion_estimate_to_record(
//       &estimates_json,
//       "scirust-simd/fixed_add", // kernel
//       "N=65536",                 // dataset
//       "fixed:Q16_16x8",          // method
//       0x1,                       // seed: bench_add's fixed_data/f32_data seed
//   )
// See scirust-bench-schema's crate docs ("Migrating criterion targets") for the full pattern.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_simd::fixed::reductions as fred;
use scirust_simd::fixed::{FixedI32x8, Q16_16};
use scirust_simd::reductions as f32red;
use std::simd::f32x8;

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

fn fixed_data(seed: u64) -> Vec<Q16_16> {
    let mut rng = Lcg(seed);
    // Valeurs dans [-1, 1) : mul reste dans la plage.
    (0..N)
        .map(|_| Q16_16::try_from(rng.unit()).unwrap())
        .collect()
}
fn f32_data(seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..N).map(|_| rng.unit() as f32).collect()
}

/// Addition SIMD sur tout le tableau (accumulation, dépendance de données).
fn bench_add(c: &mut Criterion) {
    let fx = fixed_data(0x1);
    let ff = f32_data(0x1);
    let mut g = c.benchmark_group("add");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16x8"), |b| {
        b.iter(|| {
            let mut acc = FixedI32x8::<16>::zero();
            for chunk in black_box(&fx).chunks_exact(8)
            {
                let v = FixedI32x8::<16>::from_array(chunk.try_into().unwrap());
                acc = acc + v;
            }
            acc
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32x8"), |b| {
        b.iter(|| {
            let mut acc = f32x8::splat(0.0);
            for chunk in black_box(&ff).chunks_exact(8)
            {
                acc += f32x8::from_slice(chunk);
            }
            acc
        })
    });
    g.finish();
}

/// Multiplication SIMD (produit accumulé).
fn bench_mul(c: &mut Criterion) {
    let fx = fixed_data(0x2);
    let ff = f32_data(0x2);
    let mut g = c.benchmark_group("mul");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16x8"), |b| {
        b.iter(|| {
            let mut acc = FixedI32x8::<16>::splat(Q16_16::one());
            for chunk in black_box(&fx).chunks_exact(8)
            {
                let v = FixedI32x8::<16>::from_array(chunk.try_into().unwrap());
                acc = acc * v;
            }
            acc
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32x8"), |b| {
        b.iter(|| {
            let mut acc = f32x8::splat(1.0);
            for chunk in black_box(&ff).chunks_exact(8)
            {
                acc *= f32x8::from_slice(chunk);
            }
            acc
        })
    });
    g.finish();
}

/// Produit scalaire sur tout le tableau.
fn bench_dot(c: &mut Criterion) {
    let a = fixed_data(0x3);
    let b_ = fixed_data(0x4);
    let fa = f32_data(0x3);
    let fb = f32_data(0x4);
    let mut g = c.benchmark_group("dot");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| fred::dot(black_box(&a), black_box(&b_)))
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |bch| {
        bch.iter(|| f32red::dot(black_box(&fa), black_box(&fb), f32red::ReductionMode::Fast))
    });
    g.finish();
}

/// Somme sur tout le tableau.
fn bench_sum(c: &mut Criterion) {
    let a = fixed_data(0x5);
    let fa = f32_data(0x5);
    let mut g = c.benchmark_group("sum");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| fred::sum(black_box(&a)))
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |bch| {
        bch.iter(|| f32red::sum_fast(black_box(&fa)))
    });
    g.finish();
}

/// Norme L2 et similarité cosinus.
fn bench_norm_cosine(c: &mut Criterion) {
    let a = fixed_data(0x6);
    let b_ = fixed_data(0x7);
    let mut g = c.benchmark_group("l2_norm");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| fred::l2_norm(black_box(&a)))
    });
    g.finish();

    let mut g = c.benchmark_group("cosine");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| fred::cosine_similarity(black_box(&a), black_box(&b_)))
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_add,
    bench_mul,
    bench_dot,
    bench_sum,
    bench_norm_cosine
);
criterion_main!(benches);
