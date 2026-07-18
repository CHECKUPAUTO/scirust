// scirust-signal/benches/denoise_bench.rs
//
// Throughput benchmarks for `denoise` — the first performance-tracked
// benchmark this module has had (see its own doc comments and
// `examples/denoise_benchmark.rs`/`examples/denoise_kernel_timing.rs` for the
// pre-existing *quality*-focused measurement scripts, which are `cargo run`
// examples, not Criterion, and are not re-run here).
//
// Three groups:
// - `denoise_family_representative_cost`: one representative call per family
//   (linear/rank/transform/STFT/variational/adaptive) plus the three
//   heavier patch/collaborative-filtering methods, all on the same
//   `n=4096` signal (roughly the scale of the real ECG fixture used
//   elsewhere in this crate's tests) — a single-glance cost comparison
//   across very different algorithmic strategies.
// - `denoise_auto_pipelines`: the three "auto" entry points
//   (`denoise_auto`/`denoise_best`/`denoise_cascade_auto`), which each try
//   more than one candidate internally and so are the ones most likely to
//   cost more than their single-method building blocks suggest.
// - `denoise_scaling_with_length`: how four representative methods
//   (one cheap linear baseline, one FFT/DWT-based, one patch-based, one
//   "exact, provably linear" variational method) actually scale as the
//   signal grows, rather than assumed from their big-O descriptions.
//
// Run: cargo bench -p scirust-signal --bench denoise_bench

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_signal::denoise::{
    Wavelet, collab1d_auto, denoise_auto, denoise_best, denoise_cascade_auto, kalman_smooth_auto,
    median_filter, moving_average, nlm1d_auto, stft_wiener_auto, total_variation,
    total_variation_exact, wavelet_denoise_sure,
};

const FS: f64 = 360.0;

/// Deterministic 64-bit LCG (Knuth's MMIX multiplier), matching the generator
/// already used in `benches/vi_cfar_bench.rs` and `examples/denoise_benchmark.rs`
/// — no OS/clock entropy, exactly reproducible benchmark inputs.
struct Lcg(u64);
impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.uniform().max(1.0e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// A three-tone clean signal plus white Gaussian noise at a fixed seed —
/// representative test input, not a claim about any particular real noise
/// environment (see `examples/denoise_real_ecg.rs`/`denoise_real_speech.rs`
/// for this crate's real-data quality measurements).
fn noisy_signal(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|i| {
            let t = i as f64 / FS;
            let clean = (2.0 * std::f64::consts::PI * 5.0 * t).sin()
                + 0.7 * (2.0 * std::f64::consts::PI * 11.3 * t + 0.6).sin();
            clean + 0.3 * rng.gauss()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// One representative call per family, fixed n=4096
// ---------------------------------------------------------------------------

fn bench_family_representative_cost(c: &mut Criterion) {
    let signal = noisy_signal(4096, 0x4445_4e4f_4953_4531);
    let mut group = c.benchmark_group("denoise_family_representative_cost");
    group.throughput(Throughput::Elements(signal.len() as u64));

    group.bench_function("linear/moving_average", |b| {
        b.iter(|| black_box(moving_average(black_box(&signal), 5)))
    });
    group.bench_function("rank/median_filter", |b| {
        b.iter(|| black_box(median_filter(black_box(&signal), 4)))
    });
    group.bench_function("transform/wavelet_denoise_sure", |b| {
        b.iter(|| black_box(wavelet_denoise_sure(black_box(&signal), 4, Wavelet::Db4)))
    });
    group.bench_function("stft/stft_wiener_auto", |b| {
        b.iter(|| black_box(stft_wiener_auto(black_box(&signal))))
    });
    group.bench_function("variational/total_variation_irls", |b| {
        b.iter(|| black_box(total_variation(black_box(&signal), 2.0, 10)))
    });
    group.bench_function("variational/total_variation_exact", |b| {
        b.iter(|| black_box(total_variation_exact(black_box(&signal), 2.0)))
    });
    group.bench_function("adaptive/kalman_smooth_auto", |b| {
        b.iter(|| black_box(kalman_smooth_auto(black_box(&signal))))
    });
    group.bench_function("adaptive/nlm1d_auto", |b| {
        b.iter(|| black_box(nlm1d_auto(black_box(&signal))))
    });
    group.bench_function("adaptive/collab1d_auto", |b| {
        b.iter(|| black_box(collab1d_auto(black_box(&signal))))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// The three "auto" pipelines, which try more than one candidate internally
// ---------------------------------------------------------------------------

fn bench_auto_pipelines(c: &mut Criterion) {
    let signal = noisy_signal(4096, 0x4155_544f_5049_5045);
    let mut group = c.benchmark_group("denoise_auto_pipelines");
    group.throughput(Throughput::Elements(signal.len() as u64));

    group.bench_function("denoise_auto", |b| {
        b.iter(|| black_box(denoise_auto(black_box(&signal), FS)))
    });
    group.bench_function("denoise_best", |b| {
        b.iter(|| black_box(denoise_best(black_box(&signal), FS)))
    });
    group.bench_function("denoise_cascade_auto", |b| {
        b.iter(|| black_box(denoise_cascade_auto(black_box(&signal), FS)))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Scaling with signal length: cheap linear baseline vs. FFT/DWT-based vs.
// patch-based vs. "exact, asymptotically linear" variational.
// ---------------------------------------------------------------------------

const SCALING_SIZES: [usize; 3] = [1024, 4096, 16384];

fn bench_scaling_with_length(c: &mut Criterion) {
    let mut group = c.benchmark_group("denoise_scaling_with_length");

    for &n in &SCALING_SIZES
    {
        let signal = noisy_signal(n, 0x5343_414c_494e_4721);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("moving_average", n), &signal, |b, s| {
            b.iter(|| black_box(moving_average(black_box(s), 5)))
        });
        group.bench_with_input(
            BenchmarkId::new("wavelet_denoise_sure", n),
            &signal,
            |b, s| b.iter(|| black_box(wavelet_denoise_sure(black_box(s), 4, Wavelet::Db4))),
        );
        group.bench_with_input(BenchmarkId::new("nlm1d_auto", n), &signal, |b, s| {
            b.iter(|| black_box(nlm1d_auto(black_box(s))))
        });
        group.bench_with_input(
            BenchmarkId::new("total_variation_exact", n),
            &signal,
            |b, s| b.iter(|| black_box(total_variation_exact(black_box(s), 2.0))),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_family_representative_cost,
    bench_auto_pipelines,
    bench_scaling_with_length
);
criterion_main!(benches);
