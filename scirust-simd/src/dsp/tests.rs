// scirust-simd/src/dsp/tests.rs
//
// Validation des filtres DSP. Le cœur est **générique** : les mêmes assertions
// s'exécutent sur `f32`, `f64` et deux formats virgule fixe (`Q16_16`, `Q8_24`),
// prouvant que l'implémentation unique est correcte pour tous les scalaires. On
// vérifie les propriétés spectrales (gains au continu / à Nyquist), la réponse
// impulsionnelle, l'accord virgule fixe ↔ flottant, et le déterminisme bit-à-bit.

use super::fft::{Complex, Plan, fft, ifft, irfft, rfft};
use super::{Biquad, Fir};
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
