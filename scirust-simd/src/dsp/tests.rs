// scirust-simd/src/dsp/tests.rs
//
// Validation des filtres DSP. Le cœur est **générique** : les mêmes assertions
// s'exécutent sur `f32`, `f64` et deux formats virgule fixe (`Q16_16`, `Q8_24`),
// prouvant que l'implémentation unique est correcte pour tous les scalaires. On
// vérifie les propriétés spectrales (gains au continu / à Nyquist), la réponse
// impulsionnelle, l'accord virgule fixe ↔ flottant, et le déterminisme bit-à-bit.

use super::{Biquad, Fir};
use crate::fixed::{Q8_24, Q16_16, RealScalar};

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
