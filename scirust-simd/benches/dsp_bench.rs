// scirust-simd/benches/dsp_bench.rs
//
// Benchmarks criterion des filtres DSP génériques : biquad (IIR) et FIR, en
// `Q16_16` (virgule fixe déterministe) vs `f32` (référence). Mesure le **débit**
// (échantillons/s). Le biquad n'utilise que des opérations d'anneau ; le chemin
// virgule fixe y est proche du flottant, pour un filtrage reproductible bit-à-bit.
//
// Lancement (cible AVX2 pour éviter la sur-détection AVX-512 en VM) :
//   RUSTFLAGS="-C target-cpu=x86-64-v3" \
//     cargo bench -p scirust-simd --features portable-simd --bench dsp_bench

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_simd::dsp::{Biquad, Complex, Fir, Plan, fft, rfft};
use scirust_simd::fixed::Q16_16;

const N: usize = 1 << 14;

fn signal_f32() -> Vec<f32> {
    let mut lcg = 0x2468u64;
    (0..N)
        .map(|_| {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((lcg >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}
fn signal_fixed(src: &[f32]) -> Vec<Q16_16> {
    src.iter()
        .map(|&x| Q16_16::try_from(x as f64).unwrap())
        .collect()
}

fn bench_biquad(c: &mut Criterion) {
    let sf = signal_f32();
    let sx = signal_fixed(&sf);
    let mut of = vec![0.0f32; N];
    let mut ox = vec![Q16_16::zero(); N];

    let mut g = c.benchmark_group("biquad_lowpass");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        let mut f = Biquad::<Q16_16>::lowpass(
            Q16_16::try_from(8.0).unwrap(),
            Q16_16::try_from(1.0).unwrap(),
            Q16_16::try_from(0.707).unwrap(),
        );
        b.iter(|| {
            f.reset();
            f.process_block(black_box(&sx), black_box(&mut ox));
            ox[0]
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        let mut f = Biquad::<f32>::lowpass(8.0, 1.0, 0.707);
        b.iter(|| {
            f.reset();
            f.process_block(black_box(&sf), black_box(&mut of));
            of[0]
        })
    });
    g.finish();
}

fn bench_fir(c: &mut Criterion) {
    let sf = signal_f32();
    let sx = signal_fixed(&sf);
    let mut of = vec![0.0f32; N];
    let mut ox = vec![Q16_16::zero(); N];
    // FIR passe-bas symétrique à 15 coefficients (moyenne fenêtrée simple).
    let taps_f: [f32; 15] = [1.0 / 15.0; 15];
    let mut taps_x = [Q16_16::zero(); 15];
    for (t, &f) in taps_x.iter_mut().zip(taps_f.iter())
    {
        *t = Q16_16::try_from(f as f64).unwrap();
    }

    let mut g = c.benchmark_group("fir15");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        let mut f = Fir::<Q16_16, 15>::new(taps_x);
        b.iter(|| {
            f.reset();
            f.process_block(black_box(&sx), black_box(&mut ox));
            ox[0]
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        let mut f = Fir::<f32, 15>::new(taps_f);
        b.iter(|| {
            f.reset();
            f.process_block(black_box(&sf), black_box(&mut of));
            of[0]
        })
    });
    g.finish();
}

/// FFT de longueur 1024 (fixe vs f32). Débit en points transformés/s.
fn bench_fft(c: &mut Criterion) {
    const M: usize = 1 << 10;
    let sf = signal_f32();
    let base_f: Vec<Complex<f32>> = (0..M).map(|i| Complex::from_real(sf[i])).collect();
    let base_x: Vec<Complex<Q16_16>> = (0..M)
        .map(|i| Complex::from_real(Q16_16::try_from(sf[i] as f64).unwrap()))
        .collect();

    let mut g = c.benchmark_group("fft1024");
    g.throughput(Throughput::Elements(M as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        let mut buf = base_x.clone();
        b.iter(|| {
            buf.copy_from_slice(&base_x);
            fft(black_box(&mut buf));
            buf[0]
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        let mut buf = base_f.clone();
        b.iter(|| {
            buf.copy_from_slice(&base_f);
            fft(black_box(&mut buf));
            buf[0]
        })
    });
    g.finish();
}

/// FFT-1024 virgule fixe : fonction libre (twiddles recalculés) vs plan
/// (twiddles précalculés). Quantifie le gain du plan sur le chemin fixe.
fn bench_fft_plan(c: &mut Criterion) {
    const M: usize = 1 << 10;
    let sf = signal_f32();
    let base_x: Vec<Complex<Q16_16>> = (0..M)
        .map(|i| Complex::from_real(Q16_16::try_from(sf[i] as f64).unwrap()))
        .collect();
    let plan = Plan::<Q16_16>::new(M);

    let mut g = c.benchmark_group("fft1024_plan_vs_free");
    g.throughput(Throughput::Elements(M as u64));
    g.bench_function(BenchmarkId::new("free", "Q16_16"), |b| {
        let mut buf = base_x.clone();
        b.iter(|| {
            buf.copy_from_slice(&base_x);
            fft(black_box(&mut buf));
            buf[0]
        })
    });
    g.bench_function(BenchmarkId::new("plan", "Q16_16"), |b| {
        let mut buf = base_x.clone();
        b.iter(|| {
            buf.copy_from_slice(&base_x);
            plan.fft(black_box(&mut buf));
            buf[0]
        })
    });
    g.finish();
}

/// FFT réelle vs FFT complexe (longueur 1024, Q16.16) : la rfft empaquette le
/// signal réel dans une FFT complexe de moitié → ~2× moins de travail.
fn bench_rfft(c: &mut Criterion) {
    const M: usize = 1 << 10;
    let sf = signal_f32();
    let real_x: Vec<Q16_16> = (0..M)
        .map(|i| Q16_16::try_from(sf[i] as f64).unwrap())
        .collect();
    let cplx_x: Vec<Complex<Q16_16>> = real_x.iter().map(|&r| Complex::from_real(r)).collect();

    let mut g = c.benchmark_group("rfft1024_vs_complex");
    g.throughput(Throughput::Elements(M as u64));
    g.bench_function(BenchmarkId::new("rfft", "Q16_16"), |b| {
        b.iter(|| rfft(black_box(&real_x)))
    });
    g.bench_function(BenchmarkId::new("complex_fft", "Q16_16"), |b| {
        let mut buf = cplx_x.clone();
        b.iter(|| {
            buf.copy_from_slice(&cplx_x);
            fft(black_box(&mut buf));
            buf[0]
        })
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_biquad,
    bench_fir,
    bench_fft,
    bench_fft_plan,
    bench_rfft
);
criterion_main!(benches);
