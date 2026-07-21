// scirust-signal/benches/vi_cfar_bench.rs
//
// Benchmarks for `sliding_stats::SlidingMoments` and `radar::vi_cfar`:
// - direct O(N) two-pass recomputation vs. O(1) sliding moments, per window size;
// - classical (ClassicalViCfar) vs. robust (AlwaysRobust) VI-CFAR throughput,
//   per window size, on both a homogeneous and a contaminated signal;
// - one-time calibration cost (what CfarDetector::new pays once and amortizes,
//   and what evaluate_slice pays on every call — see its own docs).
//
// `CfarDetector` (not `evaluate_slice`) is used for all per-CUT throughput
// numbers so calibration (GO/SO's quadrature-bisection calibration, see
// `radar::vi_cfar`'s module docs, "Threshold calibration") is paid once
// outside the timed region, not once per Criterion iteration — see
// `radar::vi_cfar::CfarDetector`'s docs.
//
// Run: cargo bench -p scirust-signal --bench vi_cfar_bench

// Migrating this file's results onto scirust-bench-schema::BenchRecord:
// inputs are seeded, not random. `homogeneous_signal` drives a fixed-seed
// `Lcg(0x5647_4152)`; `contaminated_signal` builds on that same stream and
// then layers a second `Lcg(0x434f_4e54_414d)` on top to place interferers.
// (`vi_cfar_calibration_cost`'s `CfarDetector::new` calls use no randomness
// at all -- they're driven purely by `CfarConfig`.) Example, after
// `cargo bench -p scirust-signal --bench vi_cfar_bench`, converting the
// "vi_cfar_classical_vs_robust" group's "classical_homogeneous/n=32" result:
//
//   let json = std::fs::read_to_string(
//       "target/criterion/vi_cfar_classical_vs_robust/classical_homogeneous/n=32/new/estimates.json",
//   ).unwrap();
//   let record = scirust_bench_schema::criterion_estimate_to_record(
//       &json,
//       "scirust-signal/vi_cfar_evaluate",
//       "homogeneous/n=32",
//       "ClassicalViCfar",
//       0x5647_4152,
//   ).unwrap();
//
// See scirust-bench-schema's crate docs ("Migrating criterion targets") for the full pattern.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_signal::radar::vi_cfar::{
    CfarConfig, CfarDetector, DetectorPolicy, EdgePolicy, InputValidationPolicy,
    RobustNoiseEstimator, SwitchingThresholds,
};
use scirust_signal::sliding_stats::SlidingMoments;

const WINDOW_SIZES: [usize; 4] = [8, 16, 32, 64];

/// Deterministic LCG (no OS/clock entropy), matching the generator already
/// used in `radar::cfar`'s own tests, for reproducible benchmark inputs.
struct Lcg(u64);
impl Lcg {
    fn uniform01(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }
    fn exponential(&mut self) -> f64 {
        -self.uniform01().ln()
    }
}

fn homogeneous_signal(len: usize) -> Vec<f64> {
    let mut rng = Lcg(0x5647_4152);
    (0..len).map(|_| rng.exponential()).collect()
}

/// A homogeneous floor with several interfering targets scattered through it
/// (contaminating both reference half-windows for most CUTs) — representative
/// of the double-contamination case the robust path targets.
fn contaminated_signal(len: usize) -> Vec<f64> {
    let mut signal = homogeneous_signal(len);
    let mut rng = Lcg(0x434f_4e54_414d);
    let n_interferers = len / 20;
    for _ in 0..n_interferers
    {
        let idx = (rng.uniform01() * len as f64) as usize % len;
        signal[idx] = 50.0 + rng.exponential() * 20.0;
    }
    signal
}

// ---------------------------------------------------------------------------
// Direct O(N) two-pass recomputation vs. O(1) sliding moments
// ---------------------------------------------------------------------------

/// The naive baseline every streaming-variance discussion compares against:
/// a plain ring buffer, recomputed from scratch (two full passes) on every
/// push. O(window) per push, by construction.
struct NaiveSlidingWindow {
    buf: Vec<f64>,
    head: usize,
    len: usize,
}

impl NaiveSlidingWindow {
    fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0; capacity],
            head: 0,
            len: 0,
        }
    }

    fn push(&mut self, x: f64) -> (f64, f64) {
        let cap = self.buf.len();
        self.buf[self.head] = x;
        self.head = (self.head + 1) % cap;
        self.len = (self.len + 1).min(cap);
        let data = if self.len < cap
        {
            &self.buf[..self.len]
        }
        else
        {
            &self.buf[..]
        };
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let m2 = data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>();
        (mean, m2 / n)
    }
}

fn bench_sliding_moments_vs_naive(c: &mut Criterion) {
    let stream = homogeneous_signal(20_000);
    let mut group = c.benchmark_group("sliding_moments_vs_naive_recompute");
    group.throughput(Throughput::Elements(stream.len() as u64));

    macro_rules! bench_window {
        ($n:literal) => {
            group.bench_function(BenchmarkId::new("sliding_moments_o1", $n), |b| {
                b.iter(|| {
                    let mut sm = SlidingMoments::<$n>::new().unwrap();
                    let mut acc = 0.0;
                    for &x in &stream
                    {
                        let u = sm.push(black_box(x)).unwrap();
                        acc += u.mean;
                    }
                    black_box(acc)
                })
            });
            group.bench_function(BenchmarkId::new("naive_on_recompute", $n), |b| {
                b.iter(|| {
                    let mut w = NaiveSlidingWindow::new($n);
                    let mut acc = 0.0;
                    for &x in &stream
                    {
                        let (mean, _var) = w.push(black_box(x));
                        acc += mean;
                    }
                    black_box(acc)
                })
            });
        };
    }
    bench_window!(8);
    bench_window!(16);
    bench_window!(32);
    bench_window!(64);
    group.finish();
}

// ---------------------------------------------------------------------------
// Classical vs. robust VI-CFAR, per window size, homogeneous vs. contaminated
// ---------------------------------------------------------------------------

fn config_for(reference_cells: usize, detector: DetectorPolicy) -> CfarConfig {
    CfarConfig {
        reference_cells,
        guard_cells: 2,
        pfa: 0.01,
        edge_policy: EdgePolicy::Exclude,
        input_validation: InputValidationPolicy::RejectNegative,
        detector,
        robust_estimator: RobustNoiseEstimator::TrimmedMean {
            trim_low: reference_cells / 4,
            trim_high: reference_cells / 4,
        },
    }
}

fn bench_classical_vs_robust(c: &mut Criterion) {
    let homogeneous = homogeneous_signal(5_000);
    let contaminated = contaminated_signal(5_000);

    let mut group = c.benchmark_group("vi_cfar_classical_vs_robust");
    group.throughput(Throughput::Elements(homogeneous.len() as u64));

    for &n in &WINDOW_SIZES
    {
        let classical_thresholds = SwitchingThresholds {
            k_vi: 6.0,
            k_mr: 2.0,
        };

        let mut classical_detector = CfarDetector::new(config_for(
            n,
            DetectorPolicy::ClassicalViCfar(classical_thresholds),
        ))
        .unwrap();
        let mut robust_detector =
            CfarDetector::new(config_for(n, DetectorPolicy::AlwaysRobust)).unwrap();

        group.bench_with_input(
            BenchmarkId::new(format!("classical_homogeneous/n={n}"), n),
            &homogeneous,
            |b, signal| b.iter(|| black_box(classical_detector.evaluate(signal).unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new(format!("classical_contaminated/n={n}"), n),
            &contaminated,
            |b, signal| b.iter(|| black_box(classical_detector.evaluate(signal).unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new(format!("robust_homogeneous/n={n}"), n),
            &homogeneous,
            |b, signal| b.iter(|| black_box(robust_detector.evaluate(signal).unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new(format!("robust_contaminated/n={n}"), n),
            &contaminated,
            |b, signal| b.iter(|| black_box(robust_detector.evaluate(signal).unwrap())),
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// One-time calibration cost (what CfarDetector::new pays once)
// ---------------------------------------------------------------------------

fn bench_calibration_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("vi_cfar_calibration_cost");
    for &n in &WINDOW_SIZES
    {
        group.bench_with_input(BenchmarkId::new("ca_only", n), &n, |b, &n| {
            b.iter(|| black_box(CfarDetector::new(config_for(n, DetectorPolicy::Ca)).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("classical_vi_cfar", n), &n, |b, &n| {
            b.iter(|| {
                let thresholds = SwitchingThresholds {
                    k_vi: 6.0,
                    k_mr: 2.0,
                };
                black_box(
                    CfarDetector::new(config_for(n, DetectorPolicy::ClassicalViCfar(thresholds)))
                        .unwrap(),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sliding_moments_vs_naive,
    bench_classical_vs_robust,
    bench_calibration_cost
);
criterion_main!(benches);
