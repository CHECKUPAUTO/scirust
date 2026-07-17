// scirust-simd/src/dsp/tests.rs
//
// Validation des filtres DSP. Le cœur est **générique** : les mêmes assertions
// s'exécutent sur `f32`, `f64` et deux formats virgule fixe (`Q16_16`, `Q8_24`),
// prouvant que l'implémentation unique est correcte pour tous les scalaires. On
// vérifie les propriétés spectrales (gains au continu / à Nyquist), la réponse
// impulsionnelle, l'accord virgule fixe ↔ flottant, et le déterminisme bit-à-bit.

use super::fft::{Complex, Plan, fft, ifft, irfft, rfft};
use super::mel::MelFilterbank;
use super::resample::design_prototype;
use super::stft::{istft, magnitude_spectrogram, num_frames, power_spectrogram, stft};
use super::window;
use super::{Biquad, Fir, resample};
use crate::fixed::conv2d::{Conv2dShape, conv2d};
use crate::fixed::{Q8_8, Q8_24, Q16_16, RealScalar};

// Petit pont scalaire ↔ f64 pour des tests génériques (comme en géométrie).
trait Scalar: RealScalar {
    fn to_f64(self) -> f64;
    fn of(v: f64) -> Self;
    const TOL: f64;
}
impl Scalar for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn of(v: f64) -> Self {
        v as f32
    }
    const TOL: f64 = 1e-5;
}
impl Scalar for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn of(v: f64) -> Self {
        v
    }
    const TOL: f64 = 1e-9;
}
impl Scalar for Q16_16 {
    fn to_f64(self) -> f64 {
        Q16_16::to_f64(self)
    }
    fn of(v: f64) -> Self {
        Q16_16::try_from(v).unwrap()
    }
    const TOL: f64 = 5e-3;
}
impl Scalar for Q8_24 {
    fn to_f64(self) -> f64 {
        Q8_24::to_f64(self)
    }
    fn of(v: f64) -> Self {
        Q8_24::try_from(v).unwrap()
    }
    const TOL: f64 = 3e-4;
}

/// `|H(z)|` en un point réel `z = ±1` depuis les coefficients (continu/Nyquist).
fn gain_at<T: Scalar>(f: &Biquad<T>, z: f64) -> f64 {
    let (b0, b1, b2, a1, a2) = f.coefficients();
    let num = b0.to_f64() + b1.to_f64() * z + b2.to_f64() * z * z;
    let den = 1.0 + a1.to_f64() * z + a2.to_f64() * z * z;
    (num / den).abs()
}

// ------------------------------------------------------------------ //
//  Biquad : identité, réponse impulsionnelle, gains spectraux         //
// ------------------------------------------------------------------ //

fn check_identity<T: Scalar + core::fmt::Debug>() {
    let mut f = Biquad::<T>::identity();
    for &v in &[0.5, -0.25, 0.75, -0.9, 0.1]
    {
        assert_eq!(f.process(T::of(v)), T::of(v)); // passe-tout exact
    }
}

#[test]
fn biquad_identity_passthrough_all_scalars() {
    check_identity::<f32>();
    check_identity::<f64>();
    check_identity::<Q16_16>();
    check_identity::<Q8_24>();
}

fn check_impulse_response<T: Scalar>() {
    // Premier échantillon de la réponse impulsionnelle = b0.
    let mut f = Biquad::<T>::lowpass(T::of(8.0), T::of(1.0), T::of(0.707));
    let (b0, ..) = f.coefficients();
    let y0 = f.process(T::one());
    assert!((y0.to_f64() - b0.to_f64()).abs() <= T::TOL);
}

#[test]
fn biquad_impulse_response_all_scalars() {
    check_impulse_response::<f32>();
    check_impulse_response::<f64>();
    check_impulse_response::<Q16_16>();
    check_impulse_response::<Q8_24>();
}

fn check_lowpass_highpass_gains<T: Scalar>() {
    // f0/fs = 1/8 (bien conditionné pour la virgule fixe).
    let (fs, f0, q) = (T::of(8.0), T::of(1.0), T::of(0.707));
    let lp = Biquad::<T>::lowpass(fs, f0, q);
    // Passe-bas : gain 1 au continu (z=1), ~0 à Nyquist (z=−1).
    assert!((gain_at(&lp, 1.0) - 1.0).abs() <= T::TOL * 4.0, "LP DC");
    assert!(gain_at(&lp, -1.0) < 0.05, "LP Nyquist");

    let hp = Biquad::<T>::highpass(fs, f0, q);
    // Passe-haut : ~0 au continu, 1 à Nyquist.
    assert!(gain_at(&hp, 1.0) < 0.05, "HP DC");
    assert!(
        (gain_at(&hp, -1.0) - 1.0).abs() <= T::TOL * 4.0,
        "HP Nyquist"
    );

    // Passe-bande : nul au continu ET à Nyquist.
    let bp = Biquad::<T>::bandpass(fs, f0, q);
    assert!(
        gain_at(&bp, 1.0) < 0.05 && gain_at(&bp, -1.0) < 0.05,
        "BP bords"
    );
}

#[test]
fn biquad_frequency_response_all_scalars() {
    check_lowpass_highpass_gains::<f32>();
    check_lowpass_highpass_gains::<f64>();
    check_lowpass_highpass_gains::<Q16_16>();
    check_lowpass_highpass_gains::<Q8_24>();
}

fn check_stable_bounded<T: Scalar>() {
    // Un filtre stable ne diverge pas sur une entrée bornée.
    let mut f = Biquad::<T>::lowpass(T::of(8.0), T::of(1.0), T::of(0.707));
    let mut lcg = 0x1234_5678u64;
    let mut peak = 0.0f64;
    for _ in 0..4000
    {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((lcg >> 40) as f64 / (1u64 << 24) as f64) - 0.5; // [−0.5,0.5)
        let y = f.process(T::of(x)).to_f64();
        peak = peak.max(y.abs());
        assert!(y.is_finite());
    }
    assert!(peak < 4.0, "sortie bornée, pic = {peak}");
}

#[test]
fn biquad_stable_bounded_all_scalars() {
    check_stable_bounded::<f32>();
    check_stable_bounded::<f64>();
    check_stable_bounded::<Q16_16>();
    check_stable_bounded::<Q8_24>();
}

// ------------------------------------------------------------------ //
//  Biquad : accord virgule fixe ↔ flottant + déterminisme             //
// ------------------------------------------------------------------ //

#[test]
fn biquad_fixed_matches_float() {
    // Même filtre passe-bas conçu et exécuté en f64 et en Q8.24 : sorties
    // proches (l'écart = quantification des coefficients + arithmétique fixe).
    let mut ff = Biquad::<f64>::lowpass(8.0, 1.0, 0.707);
    let mut fx = Biquad::<Q8_24>::lowpass(
        Q8_24::try_from(8.0).unwrap(),
        Q8_24::try_from(1.0).unwrap(),
        Q8_24::try_from(0.707).unwrap(),
    );
    let mut lcg = 0xBEEFu64;
    for _ in 0..1000
    {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((lcg >> 40) as f64 / (1u64 << 24) as f64) - 0.5;
        let yf = ff.process(x);
        let yx = fx.process(Q8_24::try_from(x).unwrap()).to_f64();
        assert!(
            (yf - yx).abs() < 2e-3,
            "écart fixe/flottant {}",
            (yf - yx).abs()
        );
    }
}

#[test]
fn biquad_fixed_is_bit_deterministic() {
    let design = || {
        Biquad::<Q16_16>::lowpass(
            Q16_16::try_from(8.0).unwrap(),
            Q16_16::try_from(1.0).unwrap(),
            Q16_16::try_from(0.707).unwrap(),
        )
    };
    let run = |mut f: Biquad<Q16_16>| {
        let mut out = [Q16_16::zero(); 64];
        for (i, o) in out.iter_mut().enumerate()
        {
            *o = f.process(Q16_16::try_from(((i % 7) as f64) * 0.1 - 0.3).unwrap());
        }
        out
    };
    let a = run(design());
    let b = run(design());
    for i in 0..64
    {
        assert_eq!(
            a[i].to_raw(),
            b[i].to_raw(),
            "échantillon {i} non déterministe"
        );
    }
}

// ------------------------------------------------------------------ //
//  FIR                                                                 //
// ------------------------------------------------------------------ //

fn check_fir_moving_average<T: Scalar>() {
    // Moyenne mobile : réponse impulsionnelle = coefficients (dans l'ordre).
    let taps = [T::of(0.25), T::of(0.25), T::of(0.25), T::of(0.25)];
    let mut f = Fir::<T, 4>::new(taps);
    // Impulsion → sort successivement h0, h1, h2, h3.
    assert!((f.process(T::one()).to_f64() - 0.25).abs() <= T::TOL);
    for _ in 0..3
    {
        assert!((f.process(T::zero()).to_f64() - 0.25).abs() <= T::TOL);
    }
    assert!(f.process(T::zero()).to_f64().abs() <= T::TOL); // au-delà : 0
    // Gain au continu = Σ taps = 1.
    f.reset();
    let mut dc = 0.0;
    for _ in 0..8
    {
        dc = f.process(T::one()).to_f64();
    }
    assert!((dc - 1.0).abs() <= T::TOL * 4.0, "FIR DC = {dc}");
}

#[test]
fn fir_moving_average_all_scalars() {
    check_fir_moving_average::<f32>();
    check_fir_moving_average::<f64>();
    check_fir_moving_average::<Q16_16>();
    check_fir_moving_average::<Q8_24>();
}

#[test]
fn fir_passthrough_and_determinism() {
    // Un seul tap = 1 : passe-tout.
    let mut f = Fir::<Q16_16, 1>::new([Q16_16::one()]);
    for i in 0..5
    {
        let x = Q16_16::try_from((i as f64) * 0.1).unwrap();
        assert_eq!(f.process(x), x);
    }
    // Déterminisme bit-à-bit d'un FIR symétrique (phase linéaire).
    let taps = [
        Q8_24::try_from(0.1).unwrap(),
        Q8_24::try_from(0.2).unwrap(),
        Q8_24::try_from(0.4).unwrap(),
        Q8_24::try_from(0.2).unwrap(),
        Q8_24::try_from(0.1).unwrap(),
    ];
    let run = || {
        let mut f = Fir::<Q8_24, 5>::new(taps);
        let mut acc = [Q8_24::zero(); 32];
        for (i, o) in acc.iter_mut().enumerate()
        {
            *o = f.process(Q8_24::try_from(((i * 3 % 11) as f64) * 0.05).unwrap());
        }
        acc
    };
    let a = run();
    let b = run();
    for i in 0..32
    {
        assert_eq!(a[i].to_raw(), b[i].to_raw());
    }
}

// ------------------------------------------------------------------ //
//  FFT radix-2                                                         //
// ------------------------------------------------------------------ //

/// Vecteur complexe depuis des échantillons réels.
fn cvec<T: Scalar>(reals: &[f64]) -> Vec<Complex<T>> {
    reals
        .iter()
        .map(|&r| Complex::from_real(T::of(r)))
        .collect()
}

/// DFT naïve de référence (f64) : `X[k] = Σₙ x[n]·e^{−2iπkn/N}`.
fn naive_dft(x: &[f64]) -> Vec<(f64, f64)> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (0.0, 0.0);
            for (nn, &xn) in x.iter().enumerate()
            {
                let ang = -2.0 * std::f64::consts::PI * (k * nn) as f64 / n as f64;
                re += xn * ang.cos();
                im += xn * ang.sin();
            }
            (re, im)
        })
        .collect()
}

fn check_fft_vs_dft<T: Scalar>(signal: &[f64]) {
    let n = signal.len();
    let mut data = cvec::<T>(signal);
    fft(&mut data);
    let reference = naive_dft(signal);
    // Tolérance : l'accumulation sur log₂ n étages amplifie l'arrondi fixe.
    let tol = T::TOL * (n as f64) * 6.0 + 1e-6;
    for (got, (re, im)) in data.iter().zip(reference.iter())
    {
        assert!(
            (got.re.to_f64() - re).abs() <= tol && (got.im.to_f64() - im).abs() <= tol,
            "FFT bin: ({}, {}) vs ({re}, {im}), tol {tol}",
            got.re.to_f64(),
            got.im.to_f64()
        );
    }
}

#[test]
fn fft_matches_naive_dft_all_scalars() {
    // Signal déterministe de longueur 16.
    let sig: Vec<f64> = (0..16).map(|i| ((i * 5 % 7) as f64) * 0.1 - 0.3).collect();
    check_fft_vs_dft::<f32>(&sig);
    check_fft_vs_dft::<f64>(&sig);
    check_fft_vs_dft::<Q16_16>(&sig);
}

fn check_fft_dc_and_impulse<T: Scalar>() {
    // DC : x = [1;8] → X[0] = 8, reste ≈ 0.
    let mut dc = cvec::<T>(&[1.0; 8]);
    fft(&mut dc);
    let tol = T::TOL * 32.0 + 1e-6;
    assert!((dc[0].re.to_f64() - 8.0).abs() <= tol && dc[0].im.to_f64().abs() <= tol);
    for c in &dc[1..]
    {
        assert!(
            c.re.to_f64().abs() <= tol && c.im.to_f64().abs() <= tol,
            "DC bin non nul"
        );
    }
    // Impulsion : x = [1,0,…] → tous les bins ≈ 1.
    let mut imp = vec![Complex::<T>::zero(); 8];
    imp[0] = Complex::from_real(T::one());
    fft(&mut imp);
    for c in &imp
    {
        assert!(
            (c.re.to_f64() - 1.0).abs() <= tol && c.im.to_f64().abs() <= tol,
            "impulsion plate"
        );
    }
}

#[test]
fn fft_dc_and_impulse_all_scalars() {
    check_fft_dc_and_impulse::<f32>();
    check_fft_dc_and_impulse::<f64>();
    check_fft_dc_and_impulse::<Q16_16>();
}

fn check_fft_roundtrip<T: Scalar>() {
    let sig: Vec<f64> = (0..16).map(|i| ((i * 3 % 5) as f64) * 0.15 - 0.3).collect();
    let orig = cvec::<T>(&sig);
    let mut data = orig.clone();
    fft(&mut data);
    ifft(&mut data);
    let tol = T::TOL * 16.0 + 1e-6;
    for (a, b) in data.iter().zip(orig.iter())
    {
        assert!(
            (a.re.to_f64() - b.re.to_f64()).abs() <= tol
                && (a.im.to_f64() - b.im.to_f64()).abs() <= tol,
            "round-trip ifft(fft) ≠ id"
        );
    }
}

#[test]
fn fft_roundtrip_all_scalars() {
    check_fft_roundtrip::<f32>();
    check_fft_roundtrip::<f64>();
    check_fft_roundtrip::<Q16_16>();
}

#[test]
fn fft_parseval_and_sinusoid() {
    // Parseval : Σ|x|² = (1/N) Σ|X|².
    let sig: Vec<f64> = (0..32).map(|i| (i as f64 * 0.37).sin() * 0.5).collect();
    let energy_time: f64 = sig.iter().map(|v| v * v).sum();
    let mut data = cvec::<f64>(&sig);
    fft(&mut data);
    let energy_freq: f64 = data.iter().map(|c| c.norm_sqr()).sum::<f64>() / sig.len() as f64;
    assert!((energy_time - energy_freq).abs() < 1e-9, "Parseval");

    // Sinusoïde pure à la fréquence bin m : énergie concentrée en m et N−m.
    let n = 32;
    let m = 4;
    let s: Vec<f64> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * (m * i) as f64 / n as f64).cos())
        .collect();
    let mut d = cvec::<f64>(&s);
    fft(&mut d);
    // Bins m et N−m ≈ N/2 ; les autres ≈ 0.
    for (k, c) in d.iter().enumerate()
    {
        let mag = c.norm_sqr().sqrt();
        if k == m || k == n - m
        {
            assert!((mag - (n as f64 / 2.0)).abs() < 1e-6, "bin {k} = {mag}");
        }
        else
        {
            assert!(mag < 1e-6, "bin {k} devrait être nul: {mag}");
        }
    }
}

#[test]
fn fft_fixed_matches_float_and_deterministic() {
    let sig: Vec<f64> = (0..16).map(|i| ((i * 7 % 9) as f64) * 0.08 - 0.3).collect();
    // Accord Q16.16 ↔ f64.
    let mut df = cvec::<f64>(&sig);
    let mut dx = cvec::<Q16_16>(&sig);
    fft(&mut df);
    fft(&mut dx);
    for (a, b) in df.iter().zip(dx.iter())
    {
        assert!((a.re - b.re.to_f64()).abs() < 5e-3 && (a.im - b.im.to_f64()).abs() < 5e-3);
    }
    // Déterminisme bit-à-bit du chemin virgule fixe.
    let mut dx2 = cvec::<Q16_16>(&sig);
    fft(&mut dx2);
    for (a, b) in dx.iter().zip(dx2.iter())
    {
        assert_eq!(a.re.to_raw(), b.re.to_raw());
        assert_eq!(a.im.to_raw(), b.im.to_raw());
    }
}

// ------------------------------------------------------------------ //
//  FFT : plan à twiddles précalculés                                  //
// ------------------------------------------------------------------ //

#[test]
fn fft_plan_matches_free_bit_for_bit_fixed() {
    // Le plan calcule les twiddles avec la MÊME expression que la fonction
    // libre ⇒ résultat identique au bit près en virgule fixe.
    let sig: Vec<f64> = (0..32)
        .map(|i| ((i * 5 % 13) as f64) * 0.05 - 0.3)
        .collect();
    let plan = Plan::<Q16_16>::new(32);
    let mut planned = cvec::<Q16_16>(&sig);
    let mut freed = cvec::<Q16_16>(&sig);
    plan.fft(&mut planned);
    fft(&mut freed);
    for (a, b) in planned.iter().zip(freed.iter())
    {
        assert_eq!(a.re.to_raw(), b.re.to_raw(), "twiddle plan ≠ libre (re)");
        assert_eq!(a.im.to_raw(), b.im.to_raw(), "twiddle plan ≠ libre (im)");
    }
}

#[test]
fn fft_plan_reuse_and_roundtrip() {
    let plan = Plan::<f64>::new(16);
    assert_eq!(plan.len(), 16);
    assert!(!plan.is_empty());
    let sig: Vec<f64> = (0..16).map(|i| (i as f64 * 0.4).sin() * 0.5).collect();

    // Réutilisation : deux transformations successives donnent le même résultat.
    let mut a = cvec::<f64>(&sig);
    let mut b = cvec::<f64>(&sig);
    plan.fft(&mut a);
    plan.fft(&mut b);
    for (x, y) in a.iter().zip(b.iter())
    {
        assert_eq!(x, y);
    }
    // Round-trip plan.ifft(plan.fft(x)) ≈ x.
    let orig = cvec::<f64>(&sig);
    let mut data = orig.clone();
    plan.fft(&mut data);
    plan.ifft(&mut data);
    for (r, o) in data.iter().zip(orig.iter())
    {
        assert!((r.re - o.re).abs() < 1e-9 && (r.im - o.im).abs() < 1e-9);
    }
}

#[test]
fn fft_plan_matches_dft() {
    let sig: Vec<f64> = (0..16)
        .map(|i| ((i * 7 % 11) as f64) * 0.06 - 0.3)
        .collect();
    let plan = Plan::<f64>::new(16);
    let mut data = cvec::<f64>(&sig);
    plan.fft(&mut data);
    let reference = naive_dft(&sig);
    for (got, (re, im)) in data.iter().zip(reference.iter())
    {
        assert!((got.re - re).abs() < 1e-9 && (got.im - im).abs() < 1e-9);
    }
}

// ------------------------------------------------------------------ //
//  FFT à entrée réelle (rfft / irfft)                                 //
// ------------------------------------------------------------------ //

fn check_rfft<T: Scalar>() {
    let n = 16;
    let sig: Vec<f64> = (0..n).map(|i| ((i * 5 % 9) as f64) * 0.07 - 0.3).collect();
    let real: Vec<T> = sig.iter().map(|&r| T::of(r)).collect();
    let spec = rfft(&real);
    assert_eq!(spec.len(), n / 2 + 1);
    // rfft = premiers n/2+1 bins de la DFT du signal réel.
    let full = naive_dft(&sig);
    let tol = T::TOL * (n as f64) * 6.0 + 1e-6;
    for (k, c) in spec.iter().enumerate()
    {
        assert!(
            (c.re.to_f64() - full[k].0).abs() <= tol && (c.im.to_f64() - full[k].1).abs() <= tol,
            "rfft bin {k}"
        );
    }
    // Bins 0 et n/2 réels (partie imaginaire nulle).
    assert!(spec[0].im.to_f64().abs() <= tol && spec[n / 2].im.to_f64().abs() <= tol);
}

#[test]
fn rfft_matches_dft_all_scalars() {
    check_rfft::<f32>();
    check_rfft::<f64>();
    check_rfft::<Q16_16>();
}

fn check_rfft_roundtrip<T: Scalar>() {
    let n = 32;
    let sig: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin() * 0.4 - 0.1).collect();
    let real: Vec<T> = sig.iter().map(|&r| T::of(r)).collect();
    let spec = rfft(&real);
    let recon = irfft(&spec, n);
    assert_eq!(recon.len(), n);
    let tol = T::TOL * (n as f64) + 1e-6;
    for (r, o) in recon.iter().zip(real.iter())
    {
        assert!((r.to_f64() - o.to_f64()).abs() <= tol, "irfft(rfft) ≠ id");
    }
}

#[test]
fn rfft_roundtrip_all_scalars() {
    check_rfft_roundtrip::<f32>();
    check_rfft_roundtrip::<f64>();
    check_rfft_roundtrip::<Q16_16>();
}

#[test]
fn rfft_dc_and_determinism() {
    // DC réel : x = [1;8] → bin 0 = 8, reste ≈ 0.
    let spec = rfft(&[1.0f64; 8]);
    assert!((spec[0].re - 8.0).abs() < 1e-9 && spec[0].im.abs() < 1e-9);
    for c in &spec[1..]
    {
        assert!(c.re.abs() < 1e-9 && c.im.abs() < 1e-9);
    }
    // Déterminisme bit-à-bit du chemin virgule fixe.
    let real: Vec<Q16_16> = (0..16)
        .map(|i| Q16_16::try_from(((i * 3 % 7) as f64) * 0.1 - 0.3).unwrap())
        .collect();
    let a = rfft(&real);
    let b = rfft(&real);
    for (x, y) in a.iter().zip(b.iter())
    {
        assert_eq!(x.re.to_raw(), y.re.to_raw());
        assert_eq!(x.im.to_raw(), y.im.to_raw());
    }
}

// ------------------------------------------------------------------ //
//  DSP sur stockage 16 bits (FixedI16) — audio embarqué               //
// ------------------------------------------------------------------ //

#[test]
fn dsp_filters_run_on_fixed_i16() {
    // Les MÊMES filtres (déjà validés sur f32/f64/Q16_16/Q8_24) tournent sur le
    // stockage i16 sans réécriture — 16 bits pour l'audio embarqué déterministe.
    let q = |v: f64| Q8_8::try_from(v).unwrap();
    // Biquad stable (pôles complexes de module √0.1 ≈ 0.316), coefficients Q8.8.
    let mut bx = Biquad::<Q8_8>::new(q(0.2), q(0.4), q(0.2), q(-0.3), q(0.1));
    let mut peak = 0.0f64;
    for i in 0..500
    {
        let x = q(((i % 17) as f64) * 0.03 - 0.25);
        let y = bx.process(x).to_f64();
        assert!(y.is_finite());
        peak = peak.max(y.abs());
    }
    assert!(peak < 4.0, "biquad i16 borné, pic = {peak}");

    // FIR moyenne mobile 16 bits : réponse impulsionnelle = coefficients.
    let mut fir = Fir::<Q8_8, 4>::new([q(0.25); 4]);
    assert!((fir.process(Q8_8::one()).to_f64() - 0.25).abs() < 1e-2);
    for _ in 0..3
    {
        assert!((fir.process(Q8_8::zero()).to_f64() - 0.25).abs() < 1e-2);
    }

    // Déterminisme bit-à-bit du chemin i16.
    let run = || {
        let mut b = Biquad::<Q8_8>::new(q(0.2), q(0.4), q(0.2), q(-0.3), q(0.1));
        let mut out = [0i16; 32];
        for (i, o) in out.iter_mut().enumerate()
        {
            *o = b.process(q(((i % 7) as f64) * 0.1 - 0.3)).to_raw();
        }
        out
    };
    assert_eq!(run(), run());
}

// ------------------------------------------------------------------ //
//  Fenêtres d'apodisation : Hann / Hamming / Blackman                 //
// ------------------------------------------------------------------ //

/// Valeurs connues aux extrémités (`n=0`) et au centre (`n=len/2`, `len`
/// pair) : `cos(0)=1`, `cos(π)=−1` donnent des constantes exactes.
fn check_window_known_values<T: Scalar + core::fmt::Debug>() {
    let len = 8;
    assert!((window::hann_coeff::<T>(0, len).to_f64() - 0.0).abs() < T::TOL);
    assert!((window::hamming_coeff::<T>(0, len).to_f64() - 0.08).abs() < T::TOL);
    assert!((window::blackman_coeff::<T>(0, len).to_f64() - 0.0).abs() < T::TOL);
    assert!((window::blackman_harris_coeff::<T>(0, len).to_f64() - 0.0).abs() < T::TOL);

    let mid = len / 2;
    assert!((window::hann_coeff::<T>(mid, len).to_f64() - 1.0).abs() < T::TOL);
    assert!((window::hamming_coeff::<T>(mid, len).to_f64() - 1.0).abs() < T::TOL);
    assert!((window::blackman_coeff::<T>(mid, len).to_f64() - 1.0).abs() < T::TOL);
    assert!((window::blackman_harris_coeff::<T>(mid, len).to_f64() - 1.0).abs() < T::TOL);
}

#[test]
fn window_known_values_all_scalars() {
    check_window_known_values::<f32>();
    check_window_known_values::<f64>();
    check_window_known_values::<Q16_16>();
}

/// Symétrie de la convention périodique : `w[n] == w[len-n]` (car
/// `cos(2π−θ) = cos(θ)`).
fn check_window_symmetry<T: Scalar + core::fmt::Debug>() {
    let len = 11; // impair : aucun n n'est son propre miroir sauf 0
    for n in 1..len
    {
        let a = window::hann_coeff::<T>(n, len).to_f64();
        let b = window::hann_coeff::<T>(len - n, len).to_f64();
        assert!((a - b).abs() < T::TOL, "hann symétrie n={n}: {a} vs {b}");
        let a = window::hamming_coeff::<T>(n, len).to_f64();
        let b = window::hamming_coeff::<T>(len - n, len).to_f64();
        assert!((a - b).abs() < T::TOL, "hamming symétrie n={n}: {a} vs {b}");
        let a = window::blackman_coeff::<T>(n, len).to_f64();
        let b = window::blackman_coeff::<T>(len - n, len).to_f64();
        assert!(
            (a - b).abs() < T::TOL,
            "blackman symétrie n={n}: {a} vs {b}"
        );
        let a = window::blackman_harris_coeff::<T>(n, len).to_f64();
        let b = window::blackman_harris_coeff::<T>(len - n, len).to_f64();
        assert!(
            (a - b).abs() < T::TOL,
            "blackman_harris symétrie n={n}: {a} vs {b}"
        );
    }
}

#[test]
fn window_symmetry_all_scalars() {
    check_window_symmetry::<f64>();
    check_window_symmetry::<Q16_16>();
}

/// Comparaison à la définition mathématique directe en `f64`.
fn check_window_matches_reference<T: Scalar + core::fmt::Debug>() {
    let len = 13;
    for n in 0..len
    {
        let theta = 2.0 * core::f64::consts::PI * (n as f64) / (len as f64);
        let want_hann = 0.5 - 0.5 * theta.cos();
        let want_hamming = 0.54 - 0.46 * theta.cos();
        let want_blackman = 0.42 - 0.5 * theta.cos() + 0.08 * (2.0 * theta).cos();
        let want_blackman_harris =
            0.36 - 0.49 * theta.cos() + 0.14 * (2.0 * theta).cos() - 0.01 * (3.0 * theta).cos();

        let got_hann = window::hann_coeff::<T>(n, len).to_f64();
        let got_hamming = window::hamming_coeff::<T>(n, len).to_f64();
        let got_blackman = window::blackman_coeff::<T>(n, len).to_f64();
        let got_blackman_harris = window::blackman_harris_coeff::<T>(n, len).to_f64();

        assert!(
            (got_hann - want_hann).abs() < T::TOL,
            "hann n={n}: {got_hann} vs {want_hann}"
        );
        assert!(
            (got_hamming - want_hamming).abs() < T::TOL,
            "hamming n={n}: {got_hamming} vs {want_hamming}"
        );
        assert!(
            (got_blackman - want_blackman).abs() < T::TOL,
            "blackman n={n}: {got_blackman} vs {want_blackman}"
        );
        assert!(
            (got_blackman_harris - want_blackman_harris).abs() < T::TOL,
            "blackman_harris n={n}: {got_blackman_harris} vs {want_blackman_harris}"
        );
    }
}

#[test]
fn window_matches_reference_all_scalars() {
    check_window_matches_reference::<f32>();
    check_window_matches_reference::<f64>();
    check_window_matches_reference::<Q16_16>();
    check_window_matches_reference::<Q8_24>();
}

#[test]
fn window_vec_builders_match_coeff() {
    let len = 16;
    let hann_v: Vec<f64> = window::hann(len);
    let hamming_v: Vec<f64> = window::hamming(len);
    let blackman_v: Vec<f64> = window::blackman(len);
    let blackman_harris_v: Vec<f64> = window::blackman_harris(len);
    assert_eq!(hann_v.len(), len);
    assert_eq!(hamming_v.len(), len);
    assert_eq!(blackman_v.len(), len);
    assert_eq!(blackman_harris_v.len(), len);
    for n in 0..len
    {
        assert_eq!(hann_v[n], window::hann_coeff::<f64>(n, len));
        assert_eq!(hamming_v[n], window::hamming_coeff::<f64>(n, len));
        assert_eq!(blackman_v[n], window::blackman_coeff::<f64>(n, len));
        assert_eq!(
            blackman_harris_v[n],
            window::blackman_harris_coeff::<f64>(n, len)
        );
    }
}

// ------------------------------------------------------------------ //
//  Kaiser (paramètre beta, bessel_i0)                                 //
// ------------------------------------------------------------------ //

/// Référence indépendante de `I₀`, série directe (comme
/// `fixed::tests::i0_series_ref`, dupliquée ici pour ne pas dépendre du
/// module de tests interne de `fixed`).
fn i0_series_ref(x: f64) -> f64 {
    let mut term = 1.0f64;
    let mut total = 1.0f64;
    for k in 1..100
    {
        term *= (x / (2.0 * k as f64)).powi(2);
        total += term;
        if term < 1e-18 * total
        {
            break;
        }
    }
    total
}

fn kaiser_ref(n: usize, len: usize, beta: f64) -> f64 {
    let r = 2.0 * (n as f64) / (len as f64) - 1.0;
    i0_series_ref(beta * (1.0 - r * r).sqrt()) / i0_series_ref(beta)
}

fn check_kaiser_center_and_endpoints<T: Scalar + core::ops::Div<Output = T> + core::fmt::Debug>() {
    let len = 16; // pair : n = len/2 tombe exactement au centre (r = 0)
    for &beta in &[0.0f64, 4.0, 8.5]
    {
        let b = T::of(beta);
        // Centre : r = 0 ⇒ I₀(beta)/I₀(beta) = 1, quel que soit beta.
        let center = window::kaiser_coeff::<T>(len / 2, len, b).to_f64();
        assert!(
            (center - 1.0).abs() <= T::TOL * 4.0,
            "kaiser centre beta={beta}: {center}"
        );
        // n = 0 : r = -1 ⇒ I₀(0)/I₀(beta) = 1/I₀(beta).
        let edge = window::kaiser_coeff::<T>(0, len, b).to_f64();
        let want_edge = 1.0 / i0_series_ref(beta);
        assert!(
            (edge - want_edge).abs() <= T::TOL * 20.0,
            "kaiser bord beta={beta}: {edge} vs {want_edge}"
        );
    }
}

#[test]
fn kaiser_center_and_endpoints_all_scalars() {
    check_kaiser_center_and_endpoints::<f32>();
    check_kaiser_center_and_endpoints::<f64>();
    check_kaiser_center_and_endpoints::<Q16_16>();
}

fn check_kaiser_matches_reference<T: Scalar + core::ops::Div<Output = T> + core::fmt::Debug>() {
    let len = 13;
    for &beta in &[0.0f64, 3.0, 6.0, 8.5]
    {
        let b = T::of(beta);
        for n in 0..len
        {
            let got = window::kaiser_coeff::<T>(n, len, b).to_f64();
            let want = kaiser_ref(n, len, beta);
            assert!(
                (got - want).abs() < T::TOL * 20.0,
                "kaiser beta={beta} n={n}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn kaiser_matches_reference_all_scalars() {
    check_kaiser_matches_reference::<f32>();
    check_kaiser_matches_reference::<f64>();
    check_kaiser_matches_reference::<Q16_16>();
}

#[test]
fn kaiser_beta_zero_is_rectangular() {
    // I₀(0) = 1 partout ⇒ arg = 0 pour tout n ⇒ coefficient = 1 partout.
    let len = 10;
    for n in 0..len
    {
        assert_eq!(window::kaiser_coeff::<f64>(n, len, 0.0), 1.0);
    }
}

#[test]
fn kaiser_vec_builder_matches_coeff() {
    let len = 12;
    let beta = 6.0;
    let v: Vec<f64> = window::kaiser(len, beta);
    assert_eq!(v.len(), len);
    for (n, &vn) in v.iter().enumerate()
    {
        assert_eq!(vn, window::kaiser_coeff::<f64>(n, len, beta));
    }
}

#[test]
fn kaiser_higher_beta_reduces_leakage_further_than_hann() {
    // Un beta élevé (lobes secondaires très atténués) doit capter moins
    // d'énergie loin du pic qu'un beta faible, sur le même ton non entier
    // que `window_reduces_spectral_leakage_vs_rectangular`.
    let n = 64;
    let bin = 5.5;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * core::f64::consts::PI * bin * (i as f64) / (n as f64)).sin())
        .collect();

    let far_energy = |spec: &[Complex<f64>]| -> f64 {
        spec.iter()
            .enumerate()
            .filter(|&(k, _)| !(2..=9).contains(&k))
            .map(|(_, c)| c.re * c.re + c.im * c.im)
            .sum()
    };

    let mut low = signal.clone();
    window::apply(&mut low, &window::kaiser::<f64>(n, 2.0));
    let energy_low_beta = far_energy(&rfft(&low));

    let mut high = signal.clone();
    window::apply(&mut high, &window::kaiser::<f64>(n, 10.0));
    let energy_high_beta = far_energy(&rfft(&high));

    assert!(
        energy_high_beta < energy_low_beta,
        "beta élevé devrait réduire la fuite: {energy_high_beta} vs {energy_low_beta}"
    );
}

#[test]
fn window_apply_multiplies_elementwise() {
    let mut signal = vec![2.0f64, 4.0, 6.0, 8.0];
    let win = vec![0.5f64, 1.0, 0.25, 0.0];
    window::apply(&mut signal, &win);
    assert_eq!(signal, vec![1.0, 4.0, 1.5, 0.0]);
}

#[test]
#[should_panic(expected = "apply")]
fn window_apply_dim_mismatch_panics() {
    let mut signal = vec![1.0f64, 2.0, 3.0];
    let win = vec![1.0f64, 1.0];
    window::apply(&mut signal, &win);
}

#[test]
fn window_reduces_spectral_leakage_vs_rectangular() {
    // Un ton pur non multiple de la fréquence d'échantillonnage/N fuit dans
    // les bins voisins. La fenêtre de Hann doit réduire l'énergie captée par
    // les bins éloignés du pic par rapport à la fenêtre rectangulaire
    // (aucune fenêtre = boxcar implicite).
    let n = 64;
    let bin = 5.5; // non entier : fuite garantie sous fenêtre rectangulaire
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * core::f64::consts::PI * bin * (i as f64) / (n as f64)).sin())
        .collect();

    let spec_rect = rfft(&signal);
    let mut windowed = signal.clone();
    window::apply(&mut windowed, &window::hann::<f64>(n));
    let spec_hann = rfft(&windowed);

    // Énergie loin du pic (bins 0..2 et 9..fin), hors lobe principal.
    let far_energy = |spec: &[Complex<f64>]| -> f64 {
        spec.iter()
            .enumerate()
            .filter(|&(k, _)| k <= 2 || k >= 9)
            .map(|(_, c)| c.re * c.re + c.im * c.im)
            .sum()
    };
    let leak_rect = far_energy(&spec_rect);
    let leak_hann = far_energy(&spec_hann);
    assert!(
        leak_hann < leak_rect,
        "fuite Hann {leak_hann} devrait être < fuite rectangulaire {leak_rect}"
    );
}

// ------------------------------------------------------------------ //
//  STFT / ISTFT (spectrogramme, recouvrement-addition)                //
// ------------------------------------------------------------------ //

fn check_stft_single_frame_matches_windowed_rfft<T: Scalar + core::fmt::Debug>() {
    // Un seul cadre (signal.len() == frame_size) : stft == rfft(signal .* fenêtre).
    let frame_size = 8;
    let win: Vec<T> = window::hann(frame_size);
    let signal: Vec<T> = (0..frame_size)
        .map(|i| T::of(((i % 5) as f64) * 0.1 - 0.2))
        .collect();
    let spec = stft(&signal, &win, 4);
    let mut windowed = signal.clone();
    window::apply(&mut windowed, &win);
    let want = rfft(&windowed);
    assert_eq!(spec.len(), want.len());
    for (a, b) in spec.iter().zip(&want)
    {
        assert!(
            (a.re.to_f64() - b.re.to_f64()).abs() < T::TOL,
            "re: {:?} vs {:?}",
            a.re,
            b.re
        );
        assert!(
            (a.im.to_f64() - b.im.to_f64()).abs() < T::TOL,
            "im: {:?} vs {:?}",
            a.im,
            b.im
        );
    }
}

#[test]
fn stft_single_frame_matches_windowed_rfft_all_scalars() {
    check_stft_single_frame_matches_windowed_rfft::<f32>();
    check_stft_single_frame_matches_windowed_rfft::<f64>();
    check_stft_single_frame_matches_windowed_rfft::<Q16_16>();
}

#[test]
fn stft_shape_matches_num_frames() {
    for &(signal_len, frame_size, hop) in &[(48usize, 8usize, 4usize), (32, 16, 8), (20, 8, 8)]
    {
        let win: Vec<f64> = window::hann(frame_size);
        let signal: Vec<f64> = (0..signal_len).map(|i| (i as f64) * 0.01).collect();
        let spec = stft(&signal, &win, hop);
        let bins = frame_size / 2 + 1;
        let frames = num_frames(signal_len, frame_size, hop);
        assert_eq!(
            spec.len(),
            frames * bins,
            "signal_len={signal_len} frame_size={frame_size} hop={hop}"
        );
    }
}

fn check_stft_istft_roundtrip_cola<T: Scalar + core::fmt::Debug>() {
    // Hann périodique + recouvrement 50 % : propriété COLA exacte (w(i) +
    // w(i+N/2) = 1), donc reconstruction exacte (aux bords près, où les
    // trames ne se chevauchent pas complètement).
    let frame_size = 16;
    let hop = frame_size / 2;
    let win: Vec<T> = window::hann(frame_size);
    let n = frame_size * 6;
    let signal: Vec<T> = (0..n)
        .map(|i| T::of(((i % 7) as f64) * 0.1 - 0.3))
        .collect();

    let spec = stft(&signal, &win, hop);
    let recon = istft(&spec, frame_size, hop);
    assert_eq!(
        recon.len(),
        (num_frames(n, frame_size, hop) - 1) * hop + frame_size
    );

    // Zone intérieure : au moins une trame de marge à chaque bord pour éviter
    // les effets de bord (chevauchement incomplet).
    for i in frame_size..(n - frame_size)
    {
        let a = signal[i].to_f64();
        let b = recon[i].to_f64();
        assert!(
            (a - b).abs() < T::TOL * 20.0,
            "istft roundtrip i={i}: {a} vs {b}"
        );
    }
}

#[test]
fn stft_istft_roundtrip_cola_all_scalars() {
    check_stft_istft_roundtrip_cola::<f32>();
    check_stft_istft_roundtrip_cola::<f64>();
    check_stft_istft_roundtrip_cola::<Q16_16>();
}

#[test]
fn istft_empty_spectrogram_is_empty() {
    let out: Vec<f64> = istft(&[], 8, 4);
    assert!(out.is_empty());
}

#[test]
#[should_panic(expected = "stft")]
fn stft_non_power_of_two_frame_panics() {
    let win = vec![1.0f64; 6]; // pas une puissance de 2
    let signal = vec![0.0f64; 10];
    let _ = stft(&signal, &win, 2);
}

#[test]
#[should_panic(expected = "istft")]
fn istft_spectrogram_not_multiple_of_bins_panics() {
    // frame_size=8 → bins=5 ; longueur 7 n'est pas un multiple de 5.
    let spec = vec![Complex::new(0.0f64, 0.0); 7];
    let _ = istft(&spec, 8, 4);
}

// ------------------------------------------------------------------ //
//  Spectrogramme réel : puissance / magnitude                         //
// ------------------------------------------------------------------ //

#[test]
fn power_and_magnitude_spectrogram_known_values() {
    // Triangle 3-4-5 : |X|² = 25, |X| = 5, exact dans tous les scalaires.
    let spec = vec![
        Complex::new(3.0f64, 4.0),
        Complex::new(0.0, 0.0),
        Complex::new(-1.0, 0.0),
    ];
    assert_eq!(power_spectrogram(&spec), vec![25.0, 0.0, 1.0]);
    assert_eq!(magnitude_spectrogram(&spec), vec![5.0, 0.0, 1.0]);
}

fn check_power_matches_magnitude_squared<T: Scalar + core::fmt::Debug>() {
    let spec: Vec<Complex<T>> = (0..10)
        .map(|i| Complex::new(T::of((i as f64) * 0.3 - 1.0), T::of((i as f64) * 0.2 - 0.5)))
        .collect();
    let power = power_spectrogram(&spec);
    let mag = magnitude_spectrogram(&spec);
    assert_eq!(power.len(), spec.len());
    assert_eq!(mag.len(), spec.len());
    for i in 0..power.len()
    {
        let want = mag[i].to_f64() * mag[i].to_f64();
        assert!(
            (power[i].to_f64() - want).abs() < T::TOL * 4.0,
            "i={i}: power={} mag²={want}",
            power[i].to_f64()
        );
    }
}

#[test]
fn power_matches_magnitude_squared_all_scalars() {
    check_power_matches_magnitude_squared::<f32>();
    check_power_matches_magnitude_squared::<f64>();
    check_power_matches_magnitude_squared::<Q16_16>();
}

#[test]
fn stft_magnitude_spectrogram_feeds_conv2d() {
    // Pipeline concret : signal → stft → spectrogramme de magnitude (« image »
    // temps × fréquence à 1 canal) → fixed::conv2d, comme en reconnaissance
    // audio quantifiée. Vérifie la forme et le déterminisme bit-à-bit.
    let frame_size = 8;
    let hop = 4;
    let win: Vec<Q16_16> = window::hann(frame_size);
    let n = 40;
    let signal: Vec<Q16_16> = (0..n)
        .map(|i| Q16_16::try_from(((i % 6) as f64) * 0.1 - 0.25).unwrap())
        .collect();
    let spec = stft(&signal, &win, hop);
    let bins = frame_size / 2 + 1;
    let frames = num_frames(n, frame_size, hop);
    assert_eq!(spec.len(), frames * bins);

    let mag = magnitude_spectrogram(&spec); // frames × bins réel, 1 canal
    let w = [Q16_16::try_from(0.25).unwrap(); 4]; // noyau 2×2, moyenne locale
    let b = [Q16_16::zero()];
    let shape = Conv2dShape {
        in_channels: 1,
        height: frames,
        width: bins,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
    };
    let out1 = conv2d(&mag, &w, &b, shape);
    let out2 = conv2d(&mag, &w, &b, shape); // déterminisme bit-à-bit
    assert_eq!(out1, out2);
    assert_eq!(out1.len(), shape.height_out() * shape.width_out());
    for v in &out1
    {
        assert!(v.to_f64().is_finite());
    }
}

// ------------------------------------------------------------------ //
//  Banque de filtres mel                                              //
// ------------------------------------------------------------------ //

fn hz_to_mel_ref(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}
fn mel_to_hz_ref(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

/// Référence indépendante (en `f64` pur) de la construction de la banque de
/// filtres, pour comparaison bit-indépendante.
fn naive_mel_filterbank_ref(
    n_mels: usize,
    bins: usize,
    sample_rate: f64,
    f_min: f64,
    f_max: f64,
) -> Vec<f64> {
    let mel_min = hz_to_mel_ref(f_min);
    let mel_max = hz_to_mel_ref(f_max);
    let n_points = n_mels + 2;
    let step = (mel_max - mel_min) / ((n_points - 1) as f64);
    let hz_points: Vec<f64> = (0..n_points)
        .map(|i| mel_to_hz_ref(mel_min + (i as f64) * step))
        .collect();
    let n_fft = 2 * (bins - 1);
    let bin_hz: Vec<f64> = (0..bins)
        .map(|k| (k as f64) * sample_rate / (n_fft as f64))
        .collect();
    let mut weights = vec![0.0; n_mels * bins];
    for m in 0..n_mels
    {
        let (left, center, right) = (hz_points[m], hz_points[m + 1], hz_points[m + 2]);
        for (k, &f) in bin_hz.iter().enumerate()
        {
            let w = if f <= left || f >= right
            {
                0.0
            }
            else if f <= center
            {
                (f - left) / (center - left)
            }
            else
            {
                (right - f) / (right - center)
            };
            weights[m * bins + k] = w;
        }
    }
    weights
}

fn check_mel_filterbank_matches_reference<
    T: Scalar + core::ops::Div<Output = T> + core::fmt::Debug,
>() {
    let (n_mels, bins, sample_rate, f_min, f_max) = (10usize, 65usize, 16000.0, 0.0, 8000.0);
    let fb = MelFilterbank::<T>::new(n_mels, bins, T::of(sample_rate), T::of(f_min), T::of(f_max));
    assert_eq!(fb.n_mels(), n_mels);
    assert_eq!(fb.bins(), bins);
    let want = naive_mel_filterbank_ref(n_mels, bins, sample_rate, f_min, f_max);

    // Sonde par impulsion : appliquer un spectrogramme à un seul bin non nul
    // extrait la colonne k de la matrice de filtres — vérifie la banque
    // entière via l'API publique (apply), sans exposer les poids internes.
    for k in 0..bins
    {
        let mut onehot = vec![T::zero(); bins];
        onehot[k] = T::one();
        let got = fb.apply(&onehot);
        for m in 0..n_mels
        {
            let g = got[m].to_f64();
            let w = want[m * bins + k];
            assert!(
                (g - w).abs() < T::TOL * 4.0,
                "mel m={m} bin={k}: {g} vs {w}"
            );
        }
    }
}

#[test]
fn mel_filterbank_matches_reference_all_scalars() {
    check_mel_filterbank_matches_reference::<f32>();
    check_mel_filterbank_matches_reference::<f64>();
    check_mel_filterbank_matches_reference::<Q16_16>();
}

#[test]
#[should_panic(expected = "MelFilterbank::new")]
fn mel_filterbank_invalid_range_panics() {
    let _ = MelFilterbank::<f64>::new(5, 9, 8000.0, 4000.0, 1000.0); // f_min ≥ f_max
}

#[test]
#[should_panic(expected = "MelFilterbank::apply")]
fn mel_apply_dim_mismatch_panics() {
    let fb = MelFilterbank::<f64>::new(5, 9, 8000.0, 0.0, 4000.0);
    let bad = vec![0.0f64; 8]; // pas multiple de bins()=9
    let _ = fb.apply(&bad);
}

#[test]
fn stft_power_spectrogram_feeds_mel_filterbank() {
    // Pipeline concret : signal → stft → power_spectrogram → banque de
    // filtres mel, comme en reconnaissance vocale quantifiée. Vérifie la
    // forme et le déterminisme bit-à-bit.
    let frame_size = 32;
    let hop = 16;
    let sample_rate = 8000.0;
    let win: Vec<Q16_16> = window::hann(frame_size);
    let n = 128;
    let signal: Vec<Q16_16> = (0..n)
        .map(|i| Q16_16::try_from(((i % 6) as f64) * 0.1 - 0.25).unwrap())
        .collect();
    let spec = stft(&signal, &win, hop);
    let bins = frame_size / 2 + 1;
    let power = power_spectrogram(&spec);

    let fb = MelFilterbank::<Q16_16>::new(
        8,
        bins,
        Q16_16::try_from(sample_rate).unwrap(),
        Q16_16::zero(),
        Q16_16::try_from(sample_rate / 2.0).unwrap(),
    );
    let mel1 = fb.apply(&power);
    let mel2 = fb.apply(&power);
    assert_eq!(mel1, mel2); // déterminisme bit-à-bit

    let frames = num_frames(n, frame_size, hop);
    assert_eq!(mel1.len(), frames * fb.n_mels());
}

// ------------------------------------------------------------------ //
//  Ré-échantillonnage rationnel L/M, polyphase                        //
// ------------------------------------------------------------------ //

/// Référence naïve indépendante : insertion explicite de `l−1` zéros,
/// convolution **complète** avec le prototype (aucun raccourci polyphase),
/// puis décimation par `m`. Sert à vérifier que [`resample::resample`]
/// (qui saute les produits nuls) calcule exactement la même somme.
fn naive_resample_zero_stuff<T: RealScalar + core::ops::Div<Output = T>>(
    x: &[T],
    l: usize,
    m: usize,
    half_taps: usize,
) -> Vec<T> {
    let h = design_prototype::<T>(l, m, half_taps);
    let center = half_taps * l;
    let up_len = x.len() * l;
    let mut up = vec![T::zero(); up_len];
    for (i, &xi) in x.iter().enumerate()
    {
        up[i * l] = xi;
    }

    let out_len = x.len() * l / m;
    let mut y = Vec::with_capacity(out_len);
    for n in 0..out_len
    {
        let pos = n * m;
        let mut acc = T::zero();
        for (k, &hk) in h.iter().enumerate()
        {
            let idx = pos as isize + k as isize - center as isize;
            if idx >= 0 && (idx as usize) < up.len()
            {
                acc = acc + hk * up[idx as usize];
            }
        }
        y.push(acc);
    }
    y
}

#[test]
fn resample_polyphase_matches_naive_zero_stuff_reference() {
    let mut rng_seed = 0x5E5D_0001u64;
    let mut next = move || {
        rng_seed = rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_seed >> 40) as i64
    };
    let x: Vec<Q16_16> = (0..17)
        .map(|_| Q16_16::from_raw((next() as i32) >> 10))
        .collect();

    for &(l, m, half_taps) in &[
        (1usize, 1usize, 3usize),
        (2, 1, 4),
        (1, 2, 4),
        (3, 2, 3),
        (2, 3, 5),
        (5, 3, 2),
    ]
    {
        let got = resample::resample(&x, l, m, half_taps);
        let want = naive_resample_zero_stuff(&x, l, m, half_taps);
        assert_eq!(
            got, want,
            "l={l} m={m} half_taps={half_taps} : le raccourci polyphase doit être bit-exact"
        );
    }
}

#[test]
fn resample_output_length_matches_formula() {
    let x = vec![Q16_16::zero(); 23];
    for &(l, m) in &[(1usize, 1usize), (2, 1), (1, 2), (3, 2), (5, 4)]
    {
        let y = resample::resample(&x, l, m, 4);
        assert_eq!(y.len(), x.len() * l / m, "l={l} m={m}");
    }
}

fn check_resample_preserves_low_frequency_sine_amplitude<T: Scalar + core::ops::Div<Output = T>>() {
    // Sinusoïde largement sous la fréquence de coupure (~quelques cycles sur
    // toute la fenêtre) : un ré-échantillonnage correct doit en préserver
    // l'amplitude à la résolution du filtre près, quel que soit L/M.
    let n = 200;
    let amplitude = 0.6;
    let cycles = 3.0;
    let x: Vec<T> = (0..n)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * cycles * (i as f64) / (n as f64);
            T::of(amplitude * phase.sin())
        })
        .collect();

    for &(l, m) in &[(2usize, 1usize), (1, 2), (3, 2), (4, 3)]
    {
        let y = resample::resample(&x, l, m, 8);
        // Amplitude crête mesurée sur le régime établi (exclut les
        // transitoires de bord, où le filtre est zéro-complété).
        let out_n = y.len();
        let margin = out_n / 4;
        let peak = y[margin..out_n - margin]
            .iter()
            .map(|v| v.to_f64().abs())
            .fold(0.0, f64::max);
        assert!(
            (peak - amplitude).abs() <= 0.1,
            "l={l} m={m}: amplitude crête {peak} vs attendue ~{amplitude}"
        );
    }
}

#[test]
fn resample_preserves_low_frequency_sine_amplitude_all_scalars() {
    check_resample_preserves_low_frequency_sine_amplitude::<f32>();
    check_resample_preserves_low_frequency_sine_amplitude::<f64>();
    check_resample_preserves_low_frequency_sine_amplitude::<Q16_16>();
}

#[test]
#[should_panic(expected = "l doit être")]
fn resample_rejects_zero_l() {
    let x = vec![Q16_16::zero(); 4];
    let _ = resample::resample(&x, 0, 1, 4);
}

#[test]
#[should_panic(expected = "m doit être")]
fn resample_rejects_zero_m() {
    let x = vec![Q16_16::zero(); 4];
    let _ = resample::resample(&x, 1, 0, 4);
}

#[test]
#[should_panic(expected = "half_taps doit être")]
fn resample_rejects_zero_half_taps() {
    let x = vec![Q16_16::zero(); 4];
    let _ = resample::resample(&x, 1, 1, 0);
}
