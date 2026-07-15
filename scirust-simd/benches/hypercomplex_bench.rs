// scirust-simd/benches/hypercomplex_bench.rs
//
// Benchmarks criterion des algèbres hypercomplexes SIMD.
//
// Deux groupes de mesures :
//
// 1. **Temps mur + débit** (`Throughput::Elements`) : multiplication
//    SIMD (shuffle/FMA en registres) vs deux baselines scalaires —
//    la récursion de Cayley-Dickson sur `[f32; N]` et la double boucle
//    « boucle par boucle » sur table de constantes de structure.
//
// 2. **Cycles par opération** (x86_64 uniquement) : mesure directe via le
//    Time-Stamp Counter (`rdtsc`, invariant sur tout x86_64 moderne),
//    exposée comme mesure criterion personnalisée. Sur les autres
//    architectures ce groupe est absent (le temps mur du groupe 1 reste
//    la référence).
//
// Lancement recommandé (sur l'hôte physique, jeu d'instructions natif) :
//
//   RUSTFLAGS="-C target-cpu=native" \
//     cargo bench -p scirust-simd --features portable-simd
//
// Les opérandes sont générés hors zone de mesure (LCG déterministe) et
// consommés par accumulation + `black_box` pour interdire au compilateur
// d'éliminer ou de sortir le produit de la boucle.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group};
use scirust_simd::hypercomplex::{OctonionSimd, SedenionSimd, scalar};

/// Nombre de paires d'opérandes par itération mesurée.
/// 4096 octonions = 128 Kio × 2 opérandes : le jeu de données déborde
/// volontairement le L1D pour mesurer un débit soutenu réaliste.
const N_PAIRS: usize = 4096;

/// LCG déterministe (mêmes opérandes à chaque exécution du bench).
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Valeurs dans [-1, 1) : régime FP normal, pas de dénormaux.
        ((self.0 >> 40) as f32) / (1u64 << 23) as f32 - 1.0
    }

    fn array<const N: usize>(&mut self) -> [f32; N] {
        let mut out = [0.0f32; N];
        for x in &mut out
        {
            *x = self.next_f32();
        }
        out
    }
}

fn operand_pairs<const N: usize>(seed: u64) -> Vec<([f32; N], [f32; N])> {
    let mut rng = Lcg(seed);
    (0..N_PAIRS).map(|_| (rng.array(), rng.array())).collect()
}

// ------------------------------------------------------------------ //
//  Corps de mesure partagés (utilisés par les deux groupes)           //
// ------------------------------------------------------------------ //

#[inline(always)]
fn run_oct_simd(pairs: &[(OctonionSimd, OctonionSimd)]) -> OctonionSimd {
    let mut acc = OctonionSimd::ZERO;
    for &(x, y) in pairs
    {
        // L'accumulation crée une dépendance de données entre produits :
        // on mesure la latence/débit réels du kernel, pas un pipeline vide.
        acc = acc + black_box(x) * black_box(y);
    }
    acc
}

#[inline(always)]
fn run_oct_scalar(pairs: &[([f32; 8], [f32; 8])]) -> [f32; 8] {
    let mut acc = [0.0f32; 8];
    for &(x, y) in pairs
    {
        let p = scalar::oct_mul(black_box(x), black_box(y));
        for (a, b) in acc.iter_mut().zip(p)
        {
            *a += b;
        }
    }
    acc
}

#[inline(always)]
fn run_oct_table(table: &scalar::MulTable<8>, pairs: &[([f32; 8], [f32; 8])]) -> [f32; 8] {
    let mut acc = [0.0f32; 8];
    for (x, y) in pairs
    {
        let p = table.mul(black_box(x), black_box(y));
        for (a, b) in acc.iter_mut().zip(p)
        {
            *a += b;
        }
    }
    acc
}

#[inline(always)]
fn run_sed_simd(pairs: &[(SedenionSimd, SedenionSimd)]) -> SedenionSimd {
    let mut acc = SedenionSimd::ZERO;
    for &(x, y) in pairs
    {
        acc = acc + black_box(x) * black_box(y);
    }
    acc
}

#[inline(always)]
fn run_sed_scalar(pairs: &[([f32; 16], [f32; 16])]) -> [f32; 16] {
    let mut acc = [0.0f32; 16];
    for &(x, y) in pairs
    {
        let p = scalar::sed_mul(black_box(x), black_box(y));
        for (a, b) in acc.iter_mut().zip(p)
        {
            *a += b;
        }
    }
    acc
}

#[inline(always)]
fn run_sed_table(table: &scalar::MulTable<16>, pairs: &[([f32; 16], [f32; 16])]) -> [f32; 16] {
    let mut acc = [0.0f32; 16];
    for (x, y) in pairs
    {
        let p = table.mul(black_box(x), black_box(y));
        for (a, b) in acc.iter_mut().zip(p)
        {
            *a += b;
        }
    }
    acc
}

// ------------------------------------------------------------------ //
//  Groupe 1 : temps mur + débit (éléments/s)                          //
// ------------------------------------------------------------------ //

fn bench_octonion_mul(c: &mut Criterion) {
    let raw = operand_pairs::<8>(0x0C70_BE0C);
    let simd_pairs: Vec<(OctonionSimd, OctonionSimd)> = raw
        .iter()
        .map(|&(x, y)| (OctonionSimd::from_array(x), OctonionSimd::from_array(y)))
        .collect();
    let table = scalar::oct_table();

    let mut group = c.benchmark_group("octonion_mul");
    group.throughput(Throughput::Elements(N_PAIRS as u64));

    group.bench_function(BenchmarkId::new("simd", "f32x8"), |b| {
        b.iter(|| run_oct_simd(&simd_pairs))
    });
    group.bench_function(BenchmarkId::new("scalar", "cayley_dickson"), |b| {
        b.iter(|| run_oct_scalar(&raw))
    });
    group.bench_function(BenchmarkId::new("scalar", "table_loop"), |b| {
        b.iter(|| run_oct_table(&table, &raw))
    });
    group.finish();
}

fn bench_sedenion_mul(c: &mut Criterion) {
    let raw = operand_pairs::<16>(0x5ED_BE0C);
    let simd_pairs: Vec<(SedenionSimd, SedenionSimd)> = raw
        .iter()
        .map(|&(x, y)| (SedenionSimd::from_array(x), SedenionSimd::from_array(y)))
        .collect();
    let table = scalar::sed_table();

    let mut group = c.benchmark_group("sedenion_mul");
    group.throughput(Throughput::Elements(N_PAIRS as u64));

    group.bench_function(BenchmarkId::new("simd", "f32x16"), |b| {
        b.iter(|| run_sed_simd(&simd_pairs))
    });
    group.bench_function(BenchmarkId::new("scalar", "cayley_dickson"), |b| {
        b.iter(|| run_sed_scalar(&raw))
    });
    group.bench_function(BenchmarkId::new("scalar", "table_loop"), |b| {
        b.iter(|| run_sed_table(&table, &raw))
    });
    group.finish();
}

criterion_group!(wall_time, bench_octonion_mul, bench_sedenion_mul);

// ------------------------------------------------------------------ //
//  Groupe 2 : cycles/opération via TSC (x86_64)                       //
// ------------------------------------------------------------------ //

#[cfg(target_arch = "x86_64")]
mod tsc {
    //! Mesure criterion personnalisée basée sur le Time-Stamp Counter.
    //!
    //! `rdtsc` compte les cycles de référence (fréquence de base fixe,
    //! « invariant TSC » sur tout x86_64 depuis Nehalem) : c'est la
    //! définition standard de « cycles/op » indépendante du turbo.
    //! `lfence` sérialise le flux d'instructions autour de la lecture
    //! pour empêcher l'exécution out-of-order de déborder la fenêtre.

    use core::arch::x86_64::{__rdtscp, _mm_lfence, _rdtsc};
    use criterion::Throughput;
    use criterion::measurement::{Measurement, ValueFormatter};

    pub struct Tsc;

    impl Measurement for Tsc {
        type Intermediate = u64;
        type Value = u64;

        fn start(&self) -> u64 {
            // lfence + rdtsc : toutes les instructions précédentes ont
            // été retirées avant l'échantillonnage du compteur.
            unsafe {
                _mm_lfence();
                let t = _rdtsc();
                _mm_lfence();
                t
            }
        }

        fn end(&self, start: u64) -> u64 {
            // rdtscp attend la fin des instructions précédentes ; le
            // lfence final bloque le réordonnancement des suivantes.
            let mut aux = 0u32;
            let stop = unsafe {
                let t = __rdtscp(&mut aux);
                _mm_lfence();
                t
            };
            stop.saturating_sub(start)
        }

        fn add(&self, v1: &u64, v2: &u64) -> u64 {
            v1 + v2
        }

        fn zero(&self) -> u64 {
            0
        }

        fn to_f64(&self, value: &u64) -> f64 {
            *value as f64
        }

        fn formatter(&self) -> &dyn ValueFormatter {
            &TscFormatter
        }
    }

    struct TscFormatter;

    impl ValueFormatter for TscFormatter {
        fn scale_values(&self, _typical: f64, _values: &mut [f64]) -> &'static str {
            "cycles"
        }

        fn scale_throughputs(
            &self,
            _typical: f64,
            throughput: &Throughput,
            values: &mut [f64],
        ) -> &'static str {
            match throughput
            {
                // criterion fournit des cycles totaux par itération : on
                // les divise par le nombre de multiplications pour
                // obtenir la métrique demandée, cycles/op.
                Throughput::Elements(elems) =>
                {
                    for v in values.iter_mut()
                    {
                        *v /= *elems as f64;
                    }
                    "cycles/op"
                },
                _ => "cycles",
            }
        }

        fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
            "cycles"
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn bench_cycles(c: &mut Criterion<tsc::Tsc>) {
    let oct_raw = operand_pairs::<8>(0x0C70_BE0C);
    let oct_simd: Vec<(OctonionSimd, OctonionSimd)> = oct_raw
        .iter()
        .map(|&(x, y)| (OctonionSimd::from_array(x), OctonionSimd::from_array(y)))
        .collect();
    let oct_table = scalar::oct_table();

    let sed_raw = operand_pairs::<16>(0x5ED_BE0C);
    let sed_simd: Vec<(SedenionSimd, SedenionSimd)> = sed_raw
        .iter()
        .map(|&(x, y)| (SedenionSimd::from_array(x), SedenionSimd::from_array(y)))
        .collect();
    let sed_table = scalar::sed_table();

    let mut group = c.benchmark_group("cycles_per_op");
    group.throughput(Throughput::Elements(N_PAIRS as u64));

    group.bench_function("octonion/simd", |b| b.iter(|| run_oct_simd(&oct_simd)));
    group.bench_function("octonion/scalar_cd", |b| {
        b.iter(|| run_oct_scalar(&oct_raw))
    });
    group.bench_function("octonion/scalar_table", |b| {
        b.iter(|| run_oct_table(&oct_table, &oct_raw))
    });
    group.bench_function("sedenion/simd", |b| b.iter(|| run_sed_simd(&sed_simd)));
    group.bench_function("sedenion/scalar_cd", |b| {
        b.iter(|| run_sed_scalar(&sed_raw))
    });
    group.bench_function("sedenion/scalar_table", |b| {
        b.iter(|| run_sed_table(&sed_table, &sed_raw))
    });
    group.finish();
}

#[cfg(target_arch = "x86_64")]
criterion_group! {
    name = cycles;
    config = Criterion::default().with_measurement(tsc::Tsc);
    targets = bench_cycles
}

#[cfg(target_arch = "x86_64")]
criterion::criterion_main!(wall_time, cycles);

#[cfg(not(target_arch = "x86_64"))]
criterion::criterion_main!(wall_time);
