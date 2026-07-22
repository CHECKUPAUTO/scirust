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

// Migration note (scirust-bench-schema): unlike most criterion targets in
// this workspace, every benchmark function here shares ONE fixed global
// signal from `signal_f32()`'s inline LCG (seed 0x2468, N=16384 samples) --
// there is no per-call seed parameter to vary. Example conversion for the
// "biquad_lowpass" group's "fixed"/"Q16_16" case (after `cargo bench
// --bench dsp_bench`, reading
// target/criterion/biquad_lowpass/fixed/Q16_16/new/estimates.json):
//
//   scirust_bench_schema::criterion_estimate_to_record(
//       &estimates_json,
//       "scirust-simd/dsp_biquad_lowpass", // kernel
//       "N=16384",                          // dataset
//       "fixed:Q16_16",                     // method
//       0x2468,                             // seed: signal_f32()'s LCG seed
//   )
// See scirust-bench-schema's crate docs ("Migrating criterion targets") for the full pattern.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_simd::dsp::goertzel::goertzel_power;
use scirust_simd::dsp::hilbert::analytic_signal;
use scirust_simd::dsp::kalman::{KalmanFilter, UnscentedKalmanFilter};
use scirust_simd::dsp::mel::MelFilterbank;
use scirust_simd::dsp::mfcc::Mfcc;
use scirust_simd::dsp::pll::Pll;
use scirust_simd::dsp::savgol::savgol_filter;
use scirust_simd::dsp::stft::{power_spectrogram, stft};
use scirust_simd::dsp::timing::SymbolTimingLoop;
use scirust_simd::dsp::wavelet::{Wavelet, dwt_decompose};
use scirust_simd::dsp::welch::welch;
use scirust_simd::dsp::window;
use scirust_simd::dsp::{
    Biquad, BiquadCascade, Complex, Fir, Lms, Nlms, Plan, Rls, fft, fft_convolve, group_delay,
    phase, rfft, unwrap_phase,
};
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

/// Cascade de Butterworth d'ordre 8 (4 sections) : coût `4×` un biquad seul,
/// pour un rejet de bande bien plus net (48 dB/octave contre 12).
fn bench_butterworth_cascade(c: &mut Criterion) {
    let sf = signal_f32();
    let sx = signal_fixed(&sf);
    let mut of = vec![0.0f32; N];
    let mut ox = vec![Q16_16::zero(); N];

    let mut g = c.benchmark_group("butterworth_lowpass_order8");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        let mut f = BiquadCascade::<Q16_16>::butterworth_lowpass(
            Q16_16::try_from(8.0).unwrap(),
            Q16_16::try_from(1.0).unwrap(),
            8,
        );
        b.iter(|| {
            f.reset();
            f.process_block(black_box(&sx), black_box(&mut ox));
            ox[0]
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        let mut f = BiquadCascade::<f32>::butterworth_lowpass(8.0, 1.0, 8);
        b.iter(|| {
            f.reset();
            f.process_block(black_box(&sf), black_box(&mut of));
            of[0]
        })
    });
    g.finish();
}

/// Chebyshev de type I vs Butterworth, même ordre : même coût de traitement
/// (mêmes sections biquad, seule la conception des coefficients diffère).
fn bench_chebyshev1_cascade(c: &mut Criterion) {
    let sf = signal_f32();
    let sx = signal_fixed(&sf);
    let mut of = vec![0.0f32; N];
    let mut ox = vec![Q16_16::zero(); N];

    let mut g = c.benchmark_group("chebyshev1_lowpass_order8");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        let mut f = BiquadCascade::<Q16_16>::chebyshev1_lowpass(
            Q16_16::try_from(8.0).unwrap(),
            Q16_16::try_from(1.0).unwrap(),
            8,
            Q16_16::try_from(1.0).unwrap(),
        );
        b.iter(|| {
            f.reset();
            f.process_block(black_box(&sx), black_box(&mut ox));
            ox[0]
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        let mut f = BiquadCascade::<f32>::chebyshev1_lowpass(8.0, 1.0, 8, 1.0);
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

/// Noyau **long** (256 prises) : `fft_convolve` (recouvrement-addition) face
/// à la convolution en temps direct (`Fir`) — l'avantage classique de la
/// convolution rapide n'apparaît que pour de longs noyaux (cf. en-tête de
/// module de `fftconv`).
fn bench_fft_convolve_long_kernel(c: &mut Criterion) {
    const KERNEL_LEN: usize = 256;
    const FFT_SIZE: usize = 1024;
    let sf = signal_f32();
    let sx = signal_fixed(&sf);
    let taps_f: [f32; KERNEL_LEN] = core::array::from_fn(|i| ((i as f32 * 0.037).sin()) * 0.01);
    let mut taps_x = [Q16_16::zero(); KERNEL_LEN];
    for (t, &f) in taps_x.iter_mut().zip(taps_f.iter())
    {
        *t = Q16_16::try_from(f as f64).unwrap();
    }

    let mut g = c.benchmark_group("convolve_kernel256");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "fft_convolve"), |b| {
        b.iter(|| fft_convolve(black_box(&sx), black_box(&taps_x), FFT_SIZE))
    });
    g.bench_function(BenchmarkId::new("f32", "fft_convolve"), |b| {
        b.iter(|| fft_convolve(black_box(&sf), black_box(&taps_f), FFT_SIZE))
    });
    g.bench_function(BenchmarkId::new("fixed", "fir_direct"), |b| {
        let mut f = Fir::<Q16_16, KERNEL_LEN>::new(taps_x);
        let mut out = vec![Q16_16::zero(); N];
        b.iter(|| {
            f.reset();
            f.process_block(black_box(&sx), black_box(&mut out));
            out[0]
        })
    });
    g.bench_function(BenchmarkId::new("f32", "fir_direct"), |b| {
        let mut f = Fir::<f32, KERNEL_LEN>::new(taps_f);
        let mut out = vec![0.0f32; N];
        b.iter(|| {
            f.reset();
            f.process_block(black_box(&sf), black_box(&mut out));
            out[0]
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

/// Fenêtre de Hann périodique, longueur 1024 (fixe vs f32). Débit en
/// coefficients générés/s.
fn bench_window(c: &mut Criterion) {
    const M: usize = 1 << 10;
    let mut out_f = vec![0.0f32; M];
    let mut out_x = vec![Q16_16::zero(); M];

    let mut g = c.benchmark_group("window_hann1024");
    g.throughput(Throughput::Elements(M as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            window::hann_into(black_box(&mut out_x));
            out_x[0]
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            window::hann_into(black_box(&mut out_f));
            out_f[0]
        })
    });
    g.finish();
}

/// Fenêtre de Kaiser, longueur 1024, beta = 8.0 (fixe vs f32). Plus coûteuse
/// que `hann`/`hamming`/`blackman` : deux évaluations de `bessel_i0` par
/// coefficient (dont une, `bessel_i0(beta)`, redondante ici puisque
/// recalculée à chaque `n` — un futur appelant sensible au débit
/// précalculerait `I₀(beta)` une seule fois).
fn bench_kaiser(c: &mut Criterion) {
    const M: usize = 1 << 10;
    let mut out_f = vec![0.0f32; M];
    let mut out_x = vec![Q16_16::zero(); M];

    let mut g = c.benchmark_group("window_kaiser1024");
    g.throughput(Throughput::Elements(M as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            window::kaiser_into(black_box(&mut out_x), Q16_16::try_from(8.0).unwrap());
            out_x[0]
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            window::kaiser_into(black_box(&mut out_f), 8.0f32);
            out_f[0]
        })
    });
    g.finish();
}

/// STFT (fenêtrage de Hann + rfft) sur un signal de N échantillons, trame
/// 1024, saut 512 (fixe vs f32). Débit en échantillons d'entrée/s.
fn bench_stft(c: &mut Criterion) {
    const FRAME: usize = 1 << 10;
    const HOP: usize = FRAME / 2;
    let sf = signal_f32();
    let sx = signal_fixed(&sf);
    let win_f: Vec<f32> = window::hann(FRAME);
    let win_x: Vec<Q16_16> = window::hann(FRAME);

    let mut g = c.benchmark_group("stft_hop512");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| stft(black_box(&sx), black_box(&win_x), HOP))
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| stft(black_box(&sf), black_box(&win_f), HOP))
    });
    g.finish();
}

/// Banque de filtres mel (40 bandes) appliquée à un spectrogramme de
/// puissance précalculé (fixe vs f32). Débit en bandes mel produites/s.
fn bench_mel(c: &mut Criterion) {
    const FRAME: usize = 1 << 10;
    const HOP: usize = FRAME / 2;
    const N_MELS: usize = 40;
    let sf = signal_f32();
    let sx = signal_fixed(&sf);
    let win_f: Vec<f32> = window::hann(FRAME);
    let win_x: Vec<Q16_16> = window::hann(FRAME);
    let bins = FRAME / 2 + 1;

    let power_f = power_spectrogram(&stft(&sf, &win_f, HOP));
    let power_x = power_spectrogram(&stft(&sx, &win_x, HOP));
    let frames = power_f.len() / bins;

    let fb_f = MelFilterbank::<f32>::new(N_MELS, bins, 16000.0, 0.0, 8000.0);
    let fb_x = MelFilterbank::<Q16_16>::new(
        N_MELS,
        bins,
        Q16_16::try_from(16000.0).unwrap(),
        Q16_16::zero(),
        Q16_16::try_from(8000.0).unwrap(),
    );

    let mut g = c.benchmark_group("mel40_filterbank");
    g.throughput(Throughput::Elements((frames * N_MELS) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| fb_x.apply(black_box(&power_x)))
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| fb_f.apply(black_box(&power_f)))
    });
    g.finish();
}

/// MFCC (13 coefficients sur 40 bandes mel) : banque mel + `ln` + DCT-II
/// tronquée, fixe vs `f32`.
fn bench_mfcc(c: &mut Criterion) {
    const FRAME: usize = 1 << 10;
    const HOP: usize = FRAME / 2;
    const N_MELS: usize = 40;
    const N_COEFFS: usize = 13;
    let sf = signal_f32();
    let sx = signal_fixed(&sf);
    let win_f: Vec<f32> = window::hann(FRAME);
    let win_x: Vec<Q16_16> = window::hann(FRAME);
    let bins = FRAME / 2 + 1;

    let power_f = power_spectrogram(&stft(&sf, &win_f, HOP));
    let power_x = power_spectrogram(&stft(&sx, &win_x, HOP));
    let frames = power_f.len() / bins;

    let mfcc_f = Mfcc::<f32>::new(N_MELS, N_COEFFS, bins, 16000.0, 0.0, 8000.0);
    let mfcc_x = Mfcc::<Q16_16>::new(
        N_MELS,
        N_COEFFS,
        bins,
        Q16_16::try_from(16000.0).unwrap(),
        Q16_16::zero(),
        Q16_16::try_from(8000.0).unwrap(),
    );

    let mut g = c.benchmark_group("mfcc13_over_mel40");
    g.throughput(Throughput::Elements((frames * N_COEFFS) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| mfcc_x.apply(black_box(&power_x)))
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| mfcc_f.apply(black_box(&power_f)))
    });
    g.finish();
}

/// Ré-échantillonnage rationnel `3/2` (polyphase), fixe vs `f32`.
fn bench_resample(c: &mut Criterion) {
    let sf = signal_f32();
    let sx = signal_fixed(&sf);
    let (l, m, half_taps) = (3usize, 2usize, 8usize);
    let out_len = sf.len() * l / m;

    let mut g = c.benchmark_group("resample_3_2");
    g.throughput(Throughput::Elements(out_len as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| scirust_simd::dsp::resample(black_box(&sx), l, m, half_taps))
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| scirust_simd::dsp::resample(black_box(&sf), l, m, half_taps))
    });
    g.finish();
}

/// Filtres adaptatifs (LMS / NLMS / RLS), 8 poids, fixe vs `f32` : coût par
/// échantillon — `O(N)` pour LMS/NLMS (`update` ne fait qu'un produit
/// scalaire et une mise à jour de poids), `O(N²)` pour RLS (mise à jour de la
/// covariance inverse `N×N`). `desired = x` (arbitraire, seul le débit
/// compte ici — la convergence est validée dans les tests).
fn bench_adaptive(c: &mut Criterion) {
    const TAPS: usize = 8;
    let sf = signal_f32();
    let sx = signal_fixed(&sf);

    let mut g = c.benchmark_group("adaptive_lms");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut f = Lms::<Q16_16, TAPS>::new(Q16_16::try_from(0.01).unwrap());
            for &x in &sx
            {
                black_box(f.update(x, x));
            }
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            let mut f = Lms::<f32, TAPS>::new(0.01);
            for &x in &sf
            {
                black_box(f.update(x, x));
            }
        })
    });
    g.finish();

    let mut g = c.benchmark_group("adaptive_nlms");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut f = Nlms::<Q16_16, TAPS>::new(
                Q16_16::try_from(0.5).unwrap(),
                Q16_16::try_from(1e-3).unwrap(),
            );
            for &x in &sx
            {
                black_box(f.update(x, x));
            }
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            let mut f = Nlms::<f32, TAPS>::new(0.5, 1e-3);
            for &x in &sf
            {
                black_box(f.update(x, x));
            }
        })
    });
    g.finish();

    let mut g = c.benchmark_group("adaptive_rls");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut f = Rls::<Q16_16, TAPS>::new(
                Q16_16::try_from(0.995).unwrap(),
                Q16_16::try_from(0.01).unwrap(),
            );
            for &x in &sx
            {
                black_box(f.update(x, x));
            }
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            let mut f = Rls::<f32, TAPS>::new(0.995, 0.01);
            for &x in &sf
            {
                black_box(f.update(x, x));
            }
        })
    });
    g.finish();
}

/// Réponse en fréquence d'une cascade de Butterworth d'ordre 8 sur une
/// grille de `M` pulsations, plus phase déballée et délai de groupe
/// (chaîne complète du module `freqz`).
fn bench_freqz(c: &mut Criterion) {
    const M: usize = 512;
    let omegas_f: Vec<f32> = (0..M)
        .map(|i| core::f32::consts::PI * (i as f32) / (M as f32 - 1.0))
        .collect();
    let omegas_x: Vec<Q16_16> = omegas_f
        .iter()
        .map(|&w| Q16_16::try_from(w as f64).unwrap())
        .collect();
    let d_omega_f = omegas_f[1] - omegas_f[0];
    let d_omega_x = omegas_x[1] - omegas_x[0];

    let mut g = c.benchmark_group("freqz_butterworth_order8");
    g.throughput(Throughput::Elements(M as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        let f = BiquadCascade::<Q16_16>::butterworth_lowpass(
            Q16_16::try_from(8.0).unwrap(),
            Q16_16::try_from(1.0).unwrap(),
            8,
        );
        b.iter(|| {
            let responses = f.frequency_response_grid(black_box(&omegas_x));
            let mut phases: Vec<Q16_16> = responses.iter().map(|&h| phase(h)).collect();
            unwrap_phase(&mut phases);
            group_delay(&phases, d_omega_x)
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        let f = BiquadCascade::<f32>::butterworth_lowpass(8.0, 1.0, 8);
        b.iter(|| {
            let responses = f.frequency_response_grid(black_box(&omegas_f));
            let mut phases: Vec<f32> = responses.iter().map(|&h| phase(h)).collect();
            unwrap_phase(&mut phases);
            group_delay(&phases, d_omega_f)
        })
    });
    g.finish();
}

/// Boucle à verrouillage de phase : traitement d'un bloc via [`Pll::process`]
/// (entrée réelle) et [`Pll::process_quadrature`] (entrée I/Q).
fn bench_pll(c: &mut Criterion) {
    let sf = signal_f32();
    let sx = signal_fixed(&sf);

    let mut g = c.benchmark_group("pll_process");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut p = Pll::<Q16_16>::new(
                Q16_16::try_from(100.0).unwrap(),
                Q16_16::try_from(10.0).unwrap(),
                Q16_16::try_from(2.0).unwrap(),
                Q16_16::try_from(0.707).unwrap(),
            );
            for &x in &sx
            {
                black_box(p.process(x));
            }
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            let mut p = Pll::<f32>::new(100.0, 10.0, 2.0, 0.707);
            for &x in &sf
            {
                black_box(p.process(x));
            }
        })
    });
    g.finish();

    let mut g = c.benchmark_group("pll_process_quadrature");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut p = Pll::<Q16_16>::new(
                Q16_16::try_from(100.0).unwrap(),
                Q16_16::try_from(10.0).unwrap(),
                Q16_16::try_from(2.0).unwrap(),
                Q16_16::try_from(0.707).unwrap(),
            );
            for &x in &sx
            {
                black_box(p.process_quadrature(x, x));
            }
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            let mut p = Pll::<f32>::new(100.0, 10.0, 2.0, 0.707);
            for &x in &sf
            {
                black_box(p.process_quadrature(x, x));
            }
        })
    });
    g.finish();
}

/// Récupération d'horloge symbole (détecteur de Gardner) : traitement d'un
/// bloc échantillonné à `SPS = 8` échantillons/symbole.
fn bench_symbol_timing(c: &mut Criterion) {
    const SPS: usize = 8;
    let sf = signal_f32();
    let sx = signal_fixed(&sf);

    let mut g = c.benchmark_group("symbol_timing_loop");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut t = SymbolTimingLoop::<Q16_16, 16>::new(
                SPS,
                Q16_16::try_from(0.01).unwrap(),
                Q16_16::try_from(0.707).unwrap(),
            );
            for &x in &sx
            {
                black_box(t.step(x));
            }
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            let mut t = SymbolTimingLoop::<f32, 16>::new(SPS, 0.01, 0.707);
            for &x in &sf
            {
                black_box(t.step(x));
            }
        })
    });
    g.finish();
}

/// Cycle prédiction + mise à jour d'un filtre de Kalman (état
/// `[position, vitesse]`, mesure = position seule) — modèle vitesse
/// constante générique le plus courant. Mesure le coût du chemin **linéaire**
/// ([`KalmanFilter::predict`]/[`KalmanFilter::update`]) : produits/inversion
/// `2×2`/`1×1` par échantillon, comparable en coût à [`Rls`] (covariance
/// `N×N` mise à jour à chaque pas) mais avec en plus une étape de prédiction
/// distincte et une inversion `M×M` (ici triviale, `M=1`).
fn bench_kalman(c: &mut Criterion) {
    let sf = signal_f32();
    let sx = signal_fixed(&sf);

    let f_f = [[1.0f32, 1.0], [0.0, 1.0]];
    let q_f = [[0.001f32, 0.0], [0.0, 0.001]];
    let h_f = [[1.0f32, 0.0]];
    let r_f = [[0.01f32]];

    let f_x = [
        [
            Q16_16::try_from(1.0).unwrap(),
            Q16_16::try_from(1.0).unwrap(),
        ],
        [
            Q16_16::try_from(0.0).unwrap(),
            Q16_16::try_from(1.0).unwrap(),
        ],
    ];
    let q_x = [
        [Q16_16::try_from(0.001).unwrap(), Q16_16::zero()],
        [Q16_16::zero(), Q16_16::try_from(0.001).unwrap()],
    ];
    let h_x = [[Q16_16::try_from(1.0).unwrap(), Q16_16::zero()]];
    let r_x = [[Q16_16::try_from(0.01).unwrap()]];

    let mut g = c.benchmark_group("kalman_predict_update");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut kf = KalmanFilter::<Q16_16, 2, 1>::new(
                [Q16_16::zero(), Q16_16::zero()],
                [
                    [Q16_16::try_from(1.0).unwrap(), Q16_16::zero()],
                    [Q16_16::zero(), Q16_16::try_from(1.0).unwrap()],
                ],
            );
            for &x in &sx
            {
                kf.predict(&f_x, &q_x);
                black_box(kf.update(&[x], &h_x, &r_x));
            }
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            let mut kf = KalmanFilter::<f32, 2, 1>::new([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]);
            for &x in &sf
            {
                kf.predict(&f_f, &q_f);
                black_box(kf.update(&[x], &h_f, &r_f));
            }
        })
    });
    g.finish();
}

/// UKF sur le même système linéaire que [`bench_kalman`] — comparaison
/// directe. La transition/mesure sont ici des fermetures (aucune jacobienne),
/// appliquées à `2N+1 = 5` points sigma par pas au lieu d'un unique produit
/// matriciel : le surcoût mesurable face à [`bench_kalman`] est le prix de
/// l'absence de linéarisation (cf. en-tête de module).
fn bench_ukf(c: &mut Criterion) {
    let sf = signal_f32();
    let sx = signal_fixed(&sf);

    let f_f = [[1.0f32, 1.0], [0.0, 1.0]];
    let q_f = [[0.001f32, 0.0], [0.0, 0.001]];
    let h_f = [[1.0f32, 0.0]];
    let r_f = [[0.01f32]];

    let f_x = [
        [
            Q16_16::try_from(1.0).unwrap(),
            Q16_16::try_from(1.0).unwrap(),
        ],
        [
            Q16_16::try_from(0.0).unwrap(),
            Q16_16::try_from(1.0).unwrap(),
        ],
    ];
    let q_x = [
        [Q16_16::try_from(0.001).unwrap(), Q16_16::zero()],
        [Q16_16::zero(), Q16_16::try_from(0.001).unwrap()],
    ];
    let h_x = [[Q16_16::try_from(1.0).unwrap(), Q16_16::zero()]];
    let r_x = [[Q16_16::try_from(0.01).unwrap()]];

    let mut g = c.benchmark_group("ukf_predict_update");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut kf = UnscentedKalmanFilter::<Q16_16, 2, 1>::new(
                [Q16_16::zero(), Q16_16::zero()],
                [
                    [Q16_16::try_from(1.0).unwrap(), Q16_16::zero()],
                    [Q16_16::zero(), Q16_16::try_from(1.0).unwrap()],
                ],
                Q16_16::try_from(1.0).unwrap(),
                Q16_16::try_from(2.0).unwrap(),
                Q16_16::try_from(1.0).unwrap(),
            );
            for &x in &sx
            {
                black_box(kf.predict(
                    |s| {
                        [
                            f_x[0][0] * s[0] + f_x[0][1] * s[1],
                            f_x[1][0] * s[0] + f_x[1][1] * s[1],
                        ]
                    },
                    &q_x,
                ));
                black_box(kf.update(&[x], |s| [h_x[0][0] * s[0] + h_x[0][1] * s[1]], &r_x));
            }
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            let mut kf = UnscentedKalmanFilter::<f32, 2, 1>::new(
                [0.0, 0.0],
                [[1.0, 0.0], [0.0, 1.0]],
                1.0,
                2.0,
                1.0,
            );
            for &x in &sf
            {
                black_box(kf.predict(
                    |s| {
                        [
                            f_f[0][0] * s[0] + f_f[0][1] * s[1],
                            f_f[1][0] * s[0] + f_f[1][1] * s[1],
                        ]
                    },
                    &q_f,
                ));
                black_box(kf.update(&[x], |s| [h_f[0][0] * s[0] + h_f[0][1] * s[1]], &r_f));
            }
        })
    });
    g.finish();
}

/// Décomposition en pyramide (4 niveaux) via lifting, Haar vs CDF 5/3 —
/// CDF 5/3 prédit/met à jour sur deux voisins (contre un pour Haar), donc un
/// surcoût par échantillon plus élevé mais une bien meilleure compaction
/// d'énergie (cf. `cdf53_better_compaction_than_haar_all_scalars` dans les
/// tests, et l'en-tête de module).
fn bench_wavelet(c: &mut Criterion) {
    let sf = signal_f32();
    let sx = signal_fixed(&sf);
    const LEVELS: usize = 4;

    let mut g = c.benchmark_group("dwt_decompose_haar");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut x = sx.clone();
            dwt_decompose(&mut x, LEVELS, Wavelet::Haar);
            x[0]
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            let mut x = sf.clone();
            dwt_decompose(&mut x, LEVELS, Wavelet::Haar);
            x[0]
        })
    });
    g.finish();

    let mut g = c.benchmark_group("dwt_decompose_cdf53");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut x = sx.clone();
            dwt_decompose(&mut x, LEVELS, Wavelet::Cdf53);
            x[0]
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            let mut x = sf.clone();
            dwt_decompose(&mut x, LEVELS, Wavelet::Cdf53);
            x[0]
        })
    });
    g.finish();
}

/// Signal analytique par FFT (transformée de Hilbert), longueur 1024, fixe vs
/// f32 : deux FFT en cascade (directe + inverse) plus le masque fréquentiel.
fn bench_hilbert(c: &mut Criterion) {
    const M: usize = 1 << 10;
    let sf = signal_f32();
    let real_f: Vec<f32> = sf[..M].to_vec();
    let real_x: Vec<Q16_16> = real_f
        .iter()
        .map(|&v| Q16_16::try_from(v as f64).unwrap())
        .collect();

    let mut g = c.benchmark_group("hilbert_analytic1024");
    g.throughput(Throughput::Elements(M as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| analytic_signal(black_box(&real_x)))
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| analytic_signal(black_box(&real_f)))
    });
    g.finish();
}

/// Goertzel (puissance à une seule fréquence, `O(N)`) sur le signal complet,
/// fixe vs f32 : une récurrence à un multiplieur par échantillon, sans table.
fn bench_goertzel(c: &mut Criterion) {
    let sf = signal_f32();
    let sx = signal_fixed(&sf);
    // ω = 2π·1000/8000 = π/4.
    let omega_f = core::f32::consts::PI / 4.0;
    let omega_x = Q16_16::try_from(core::f64::consts::PI / 4.0).unwrap();

    let mut g = c.benchmark_group("goertzel_power");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| goertzel_power(black_box(&sx), black_box(omega_x)))
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| goertzel_power(black_box(&sf), black_box(omega_f)))
    });
    g.finish();
}

/// PSD par la méthode de Welch (trames de 1024, saut 512, fenêtre de Hann)
/// sur le signal complet, fixe vs f32 : STFT + accumulation + normalisation.
fn bench_welch(c: &mut Criterion) {
    const FRAME: usize = 1 << 10;
    const HOP: usize = FRAME / 2;
    let sf = signal_f32();
    let sx = signal_fixed(&sf);
    let win_f: Vec<f32> = window::hann(FRAME);
    let win_x: Vec<Q16_16> = window::hann(FRAME);

    let mut g = c.benchmark_group("welch_psd");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            welch(
                black_box(&sx),
                black_box(&win_x),
                HOP,
                Q16_16::try_from(8000.0).unwrap(),
            )
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| welch(black_box(&sf), black_box(&win_f), HOP, 8000.0f32))
    });
    g.finish();
}

/// Filtre de Savitzky–Golay (lissage, fenêtre 11, ordre 3) sur le signal
/// complet, fixe vs f32 : les coefficients sont résolus une fois (petit
/// système normal), puis une convolution FIR par échantillon.
fn bench_savgol(c: &mut Criterion) {
    let sf = signal_f32();
    let sx = signal_fixed(&sf);

    let mut g = c.benchmark_group("savgol_smooth_w11_p3");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| savgol_filter(black_box(&sx), 11, 3, 0))
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| savgol_filter(black_box(&sf), 11, 3, 0))
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_biquad,
    bench_butterworth_cascade,
    bench_chebyshev1_cascade,
    bench_fir,
    bench_fft_convolve_long_kernel,
    bench_fft,
    bench_fft_plan,
    bench_rfft,
    bench_window,
    bench_kaiser,
    bench_stft,
    bench_mel,
    bench_mfcc,
    bench_resample,
    bench_adaptive,
    bench_freqz,
    bench_pll,
    bench_symbol_timing,
    bench_kalman,
    bench_ukf,
    bench_wavelet,
    bench_hilbert,
    bench_goertzel,
    bench_welch,
    bench_savgol
);
criterion_main!(benches);
