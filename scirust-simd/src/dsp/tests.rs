// scirust-simd/src/dsp/tests.rs
//
// Validation des filtres DSP. Le cœur est **générique** : les mêmes assertions
// s'exécutent sur `f32`, `f64` et deux formats virgule fixe (`Q16_16`, `Q8_24`),
// prouvant que l'implémentation unique est correcte pour tous les scalaires. On
// vérifie les propriétés spectrales (gains au continu / à Nyquist), la réponse
// impulsionnelle, l'accord virgule fixe ↔ flottant, et le déterminisme bit-à-bit.

use super::adaptive::{Lms, Nlms, Rls};
use super::biquad::{butterworth_qs, chebyshev1_pole_params};
use super::fft::{Complex, Plan, fft, ifft, irfft, rfft};
use super::fftconv::fft_convolve;
use super::mel::MelFilterbank;
use super::mfcc::{Mfcc, dct2, dct2_basis};
use super::pll::{Nco, PiLoopFilter, Pll};
use super::resample::design_prototype;
use super::stft::{istft, magnitude_spectrogram, num_frames, power_spectrogram, stft};
use super::window;
use super::{
    Biquad, BiquadCascade, Fir, group_delay, magnitude, magnitude_db, phase, resample, unwrap_phase,
};
use crate::fixed::conv2d::{Conv2dShape, conv2d};
use crate::fixed::{Q8_8, Q8_24, Q16_16, Q32_32, RealScalar};

// Petit pont scalaire ↔ f64 pour des tests génériques (comme en géométrie).
// `Div<Output = Self>` (implémenté sans condition par `Fixed<I, FRAC>` et les
// flottants) : requis par `magnitude_db`/`group_delay`/`frequency_response_hz`
// (division réelle non-puissance-de-deux, cf. module `freqz`).
trait Scalar: RealScalar + core::ops::Div<Output = Self> {
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
//  BiquadCascade : filtres de Butterworth d'ordre supérieur            //
// ------------------------------------------------------------------ //

/// Gain `|H(e^{jω})|` d'une cascade à la pulsation numérique `omega`
/// (radians ; `0` = continu, `π` = Nyquist), évalué en `f64` indépendamment
/// du scalaire du filtre — chaque section contribue par produit (cascade en
/// série), `zⁿ` calculé par élévation directe (pas de FFT).
fn cascade_gain_at<T: Scalar>(cascade: &BiquadCascade<T>, omega: f64) -> f64 {
    let (cos_w, sin_w) = (omega.cos(), omega.sin());
    // z⁻¹ = e^{−jω} = cos ω − j·sin ω.
    let (zr, zi) = (cos_w, -sin_w);
    let (z2r, z2i) = (zr * zr - zi * zi, 2.0 * zr * zi);
    let mut gain = 1.0;
    for stage in cascade.stages()
    {
        let (b0, b1, b2, a1, a2) = stage.coefficients();
        let (b0, b1, b2, a1, a2) = (
            b0.to_f64(),
            b1.to_f64(),
            b2.to_f64(),
            a1.to_f64(),
            a2.to_f64(),
        );
        let (numr, numi) = (b0 + b1 * zr + b2 * z2r, b1 * zi + b2 * z2i);
        let (denr, deni) = (1.0 + a1 * zr + a2 * z2r, a1 * zi + a2 * z2i);
        gain *= (numr * numr + numi * numi).sqrt() / (denr * denr + deni * deni).sqrt();
    }
    gain
}

#[test]
fn butterworth_q_matches_known_values() {
    // Ordre 2 : Q = 1/√2 (cas RBJ usuel).
    let q2: Vec<f64> = butterworth_qs::<f64>(2);
    assert_eq!(q2.len(), 1);
    assert!((q2[0] - core::f64::consts::FRAC_1_SQRT_2).abs() <= 1e-9);

    // Ordre 4 : Q₁ ≈ 0.5411961, Q₂ ≈ 1.3065630 (constantes classiques de la
    // cascade de Butterworth du 4ᵉ ordre).
    let q4: Vec<f64> = butterworth_qs::<f64>(4);
    assert_eq!(q4.len(), 2);
    assert!((q4[0] - 0.5411961).abs() <= 1e-6);
    assert!((q4[1] - 1.3065630).abs() <= 1e-6);
}

#[test]
#[should_panic(expected = "ordre")]
fn butterworth_qs_rejects_odd_order() {
    let _: Vec<f64> = butterworth_qs(3);
}

#[test]
#[should_panic(expected = "ordre")]
fn butterworth_qs_rejects_too_small_order() {
    let _: Vec<f64> = butterworth_qs(0);
}

fn check_butterworth_lowpass_dc_unity_and_section_count<T: Scalar + core::ops::Div<Output = T>>() {
    for &order in &[2usize, 4, 6]
    {
        let lp = BiquadCascade::<T>::butterworth_lowpass(T::of(8.0), T::of(1.0), order);
        assert_eq!(lp.order(), order);
        assert_eq!(lp.stages().len(), order / 2);
        let dc = cascade_gain_at(&lp, 0.0);
        assert!(
            (dc - 1.0).abs() <= T::TOL * 4.0,
            "order={order}: gain continu {dc}"
        );
    }
}

#[test]
fn butterworth_lowpass_dc_unity_and_section_count_all_scalars() {
    check_butterworth_lowpass_dc_unity_and_section_count::<f32>();
    check_butterworth_lowpass_dc_unity_and_section_count::<f64>();
    check_butterworth_lowpass_dc_unity_and_section_count::<Q16_16>();
}

fn check_butterworth_highpass_blocks_dc<T: Scalar + core::ops::Div<Output = T>>() {
    for &order in &[2usize, 4, 6]
    {
        let hp = BiquadCascade::<T>::butterworth_highpass(T::of(8.0), T::of(1.0), order);
        assert_eq!(hp.order(), order);
        let dc = cascade_gain_at(&hp, 0.0);
        assert!(
            dc < 0.05,
            "order={order}: gain continu {dc} devrait être ≈0"
        );
        let nyquist = cascade_gain_at(&hp, core::f64::consts::PI);
        assert!(
            (nyquist - 1.0).abs() <= T::TOL * 4.0,
            "order={order}: gain Nyquist {nyquist} devrait être ≈1"
        );
    }
}

#[test]
fn butterworth_highpass_blocks_dc_all_scalars() {
    check_butterworth_highpass_blocks_dc::<f32>();
    check_butterworth_highpass_blocks_dc::<f64>();
    check_butterworth_highpass_blocks_dc::<Q16_16>();
}

#[test]
fn butterworth_higher_order_attenuates_more_beyond_cutoff() {
    // À fréquence de coupure et échantillonnage égaux, un ordre plus élevé
    // doit atténuer davantage au-delà de la coupure — la raison d'être même
    // d'une cascade (12 dB/octave par section, pas seulement une section).
    let (fs, f0) = (
        Q16_16::try_from(8.0).unwrap(),
        Q16_16::try_from(1.0).unwrap(),
    );
    let lp2 = BiquadCascade::butterworth_lowpass(fs, f0, 2);
    let lp4 = BiquadCascade::butterworth_lowpass(fs, f0, 4);
    let lp6 = BiquadCascade::butterworth_lowpass(fs, f0, 6);

    // Pulsation numérique à 2× la fréquence de coupure (f0/fs = 1/8 ⇒ ω =
    // 2π·(2/8) = π/2).
    let omega = core::f64::consts::FRAC_PI_2;
    let g2 = cascade_gain_at(&lp2, omega);
    let g4 = cascade_gain_at(&lp4, omega);
    let g6 = cascade_gain_at(&lp6, omega);
    assert!(
        g4 < g2,
        "ordre 4 ({g4}) devrait atténuer plus que l'ordre 2 ({g2})"
    );
    assert!(
        g6 < g4,
        "ordre 6 ({g6}) devrait atténuer plus que l'ordre 4 ({g4})"
    );
}

#[test]
#[should_panic(expected = "ordre")]
fn butterworth_lowpass_rejects_odd_order() {
    let _ = BiquadCascade::<Q16_16>::butterworth_lowpass(
        Q16_16::try_from(8.0).unwrap(),
        Q16_16::try_from(1.0).unwrap(),
        3,
    );
}

#[test]
#[should_panic(expected = "au moins une section")]
fn biquad_cascade_new_rejects_empty() {
    let _: BiquadCascade<Q16_16> = BiquadCascade::new(Vec::new());
}

// ------------------------------------------------------------------ //
//  BiquadCascade : filtres de Chebyshev de type I                     //
// ------------------------------------------------------------------ //

#[test]
fn chebyshev1_pole_params_section_count_and_positivity() {
    for &order in &[2usize, 4, 6]
    {
        let params: Vec<(f64, f64)> = chebyshev1_pole_params(order, 1.0);
        assert_eq!(params.len(), order / 2);
        for &(wn, q) in &params
        {
            assert!(wn > 0.0, "order={order}: ωₙ={wn} devrait être positive");
            assert!(q > 0.0, "order={order}: Q={q} devrait être positif");
        }
    }
}

#[test]
#[should_panic(expected = "ordre")]
fn chebyshev1_pole_params_rejects_odd_order() {
    let _: Vec<(f64, f64)> = chebyshev1_pole_params(3, 1.0);
}

#[test]
#[should_panic(expected = "ordre")]
fn chebyshev1_pole_params_rejects_too_small_order() {
    let _: Vec<(f64, f64)> = chebyshev1_pole_params(0, 1.0);
}

fn check_chebyshev1_lowpass_dc_at_ripple_floor<T: Scalar + core::ops::Div<Output = T>>() {
    // Ordre pair : le continu touche exactement le plancher d'ondulation
    // −ripple_db (extremum du polynôme de Chebyshev sous-jacent en `ω=0`,
    // cf. en-tête de module) — propriété exacte, pas une simple tendance.
    for &order in &[2usize, 4, 6]
    {
        let ripple_db = 1.0;
        let lp =
            BiquadCascade::<T>::chebyshev1_lowpass(T::of(8.0), T::of(1.0), order, T::of(ripple_db));
        assert_eq!(lp.order(), order);
        assert_eq!(lp.stages().len(), order / 2);
        let want_floor = 10f64.powf(-ripple_db / 20.0);
        let dc = cascade_gain_at(&lp, 0.0);
        assert!(
            (dc - want_floor).abs() <= T::TOL * 16.0,
            "order={order}: gain continu {dc} vs plancher {want_floor}"
        );
    }
}

#[test]
fn chebyshev1_lowpass_dc_at_ripple_floor_all_scalars() {
    check_chebyshev1_lowpass_dc_at_ripple_floor::<f32>();
    check_chebyshev1_lowpass_dc_at_ripple_floor::<f64>();
    check_chebyshev1_lowpass_dc_at_ripple_floor::<Q16_16>();
}

/// `Tₙ(x)` (polynôme de Chebyshev de première espèce) par récurrence
/// `T₀=1, T₁=x, Tₖ₊₁=2x·Tₖ−Tₖ₋₁` — référence indépendante du filtre
/// numérique, pour comparer la réponse de [`BiquadCascade::chebyshev1_lowpass`]
/// à la formule analytique `|H(jΩ)|² = 1/(1+ε²·Tₙ²(Ω))`.
fn chebyshev_tn(n: usize, x: f64) -> f64 {
    let mut t0 = 1.0;
    let mut t1 = x;
    if n == 0
    {
        return t0;
    }
    for _ in 1..n
    {
        let t2 = 2.0 * x * t1 - t0;
        t0 = t1;
        t1 = t2;
    }
    t1
}

#[test]
fn chebyshev1_lowpass_matches_analytic_ripple_shape() {
    // Contrairement à Butterworth (descente monotone), l'ondulation de
    // Chebyshev oscille dans la bande passante — comparée ici point par
    // point à la formule analytique |H(jΩ)|² = 1/(1+ε²·Tₙ²(Ω)) (référence
    // indépendante, cf. chebyshev_tn). Coupure basse par rapport à
    // l'échantillonnage (`cutoff/fs = 1/32`) pour limiter la distorsion de
    // fréquence propre au « cookbook » RBJ (chaque section est bilinéaire-
    // transformée indépendamment à sa propre fréquence, sans prédistorsion
    // — approximation d'autant plus fidèle que `cutoff/fs` est petit ;
    // Butterworth partage cette même limitation mais elle n'apparaît que
    // pour une comparaison point par point comme celle-ci, jamais testée
    // jusqu'ici puisqu'aucune section n'a de fréquence propre différente de
    // la coupure nominale).
    let ripple_db = 1.0;
    let epsilon = (10f64.powf(ripple_db / 10.0) - 1.0).sqrt();
    let (fs, f0, order) = (32.0, 1.0, 6usize);
    let lp = BiquadCascade::<f64>::chebyshev1_lowpass(fs, f0, order, ripple_db);
    let cutoff_omega = 2.0 * core::f64::consts::PI * f0 / fs;
    let mut saw_near_unity = false;
    for i in 0..=20
    {
        let big_omega = (i as f64) / 20.0; // Ω normalisé ∈ [0, 1].
        let tn = chebyshev_tn(order, big_omega);
        let want = 1.0 / (1.0 + epsilon * epsilon * tn * tn).sqrt();
        let got = cascade_gain_at(&lp, cutoff_omega * big_omega);
        assert!(
            (got - want).abs() <= 0.02,
            "Ω={big_omega}: gain {got} vs analytique {want}"
        );
        if got >= 0.99
        {
            saw_near_unity = true;
        }
    }
    assert!(
        saw_near_unity,
        "la bande passante devrait toucher ~0 dB à au moins un point (ondulation, pas descente monotone)"
    );
}

#[test]
fn chebyshev1_lowpass_steeper_than_butterworth_same_order() {
    // À ordre égal, Chebyshev doit atténuer davantage que Butterworth
    // au-delà de la coupure — la raison d'être du compromis ondulation
    // contre raideur (cf. en-tête de module).
    let (fs, f0, order) = (8.0, 1.0, 4);
    let cheb = BiquadCascade::<f64>::chebyshev1_lowpass(fs, f0, order, 1.0);
    let butter = BiquadCascade::<f64>::butterworth_lowpass(fs, f0, order);
    let omega = core::f64::consts::FRAC_PI_2; // 2× la coupure.
    let g_cheb = cascade_gain_at(&cheb, omega);
    let g_butter = cascade_gain_at(&butter, omega);
    assert!(
        g_cheb < g_butter,
        "Chebyshev ({g_cheb}) devrait atténuer plus que Butterworth ({g_butter}) au même ordre"
    );
}

fn check_chebyshev1_highpass_dc_blocked_and_nyquist_at_ripple_floor<
    T: Scalar + core::ops::Div<Output = T>,
>() {
    // Passe-haut : le continu est fortement atténué (côté « coupure lointaine »
    // du prototype), et Nyquist touche le plancher d'ondulation (image de
    // `ω=0` du passe-bas par la substitution `s → 1/s`, cf. en-tête de module).
    for &order in &[2usize, 4, 6]
    {
        let ripple_db = 1.0;
        let hp = BiquadCascade::<T>::chebyshev1_highpass(
            T::of(8.0),
            T::of(1.0),
            order,
            T::of(ripple_db),
        );
        assert_eq!(hp.order(), order);
        let dc = cascade_gain_at(&hp, 0.0);
        assert!(
            dc < 0.05,
            "order={order}: gain continu {dc} devrait être ≈0"
        );
        let want_floor = 10f64.powf(-ripple_db / 20.0);
        let nyquist = cascade_gain_at(&hp, core::f64::consts::PI);
        assert!(
            (nyquist - want_floor).abs() <= T::TOL * 16.0,
            "order={order}: gain Nyquist {nyquist} vs plancher {want_floor}"
        );
    }
}

#[test]
fn chebyshev1_highpass_dc_blocked_and_nyquist_at_ripple_floor_all_scalars() {
    check_chebyshev1_highpass_dc_blocked_and_nyquist_at_ripple_floor::<f32>();
    check_chebyshev1_highpass_dc_blocked_and_nyquist_at_ripple_floor::<f64>();
    check_chebyshev1_highpass_dc_blocked_and_nyquist_at_ripple_floor::<Q16_16>();
}

#[test]
#[should_panic(expected = "ordre")]
fn chebyshev1_lowpass_rejects_odd_order() {
    let _ = BiquadCascade::<Q16_16>::chebyshev1_lowpass(
        Q16_16::try_from(8.0).unwrap(),
        Q16_16::try_from(1.0).unwrap(),
        3,
        Q16_16::try_from(1.0).unwrap(),
    );
}

#[test]
#[should_panic(expected = "ordre")]
fn chebyshev1_highpass_rejects_too_small_order() {
    let _ = BiquadCascade::<Q16_16>::chebyshev1_highpass(
        Q16_16::try_from(8.0).unwrap(),
        Q16_16::try_from(1.0).unwrap(),
        0,
        Q16_16::try_from(1.0).unwrap(),
    );
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
//  Réponse en fréquence (freqz) : magnitude, phase, délai de groupe    //
// ------------------------------------------------------------------ //

fn check_frequency_response_matches_gain_at<T: Scalar>() {
    // Preuve croisée directe : `Biquad::frequency_response` (nouveau, module
    // `freqz`) doit coïncider avec `gain_at` (référence indépendante déjà
    // établie ci-dessus, utilisée par `biquad_frequency_response_all_scalars`)
    // aux deux points réels du cercle unité (`ω=0` ↔ `z=1`, `ω=π` ↔ `z=−1`).
    let (fs, f0, q) = (T::of(8.0), T::of(1.0), T::of(0.707));
    for f in [
        Biquad::<T>::lowpass(fs, f0, q),
        Biquad::<T>::highpass(fs, f0, q),
        Biquad::<T>::bandpass(fs, f0, q),
    ]
    {
        let mag_dc = magnitude(f.frequency_response(T::zero())).to_f64();
        let mag_ny = magnitude(f.frequency_response(T::pi())).to_f64();
        assert!(
            (mag_dc - gain_at(&f, 1.0)).abs() <= T::TOL * 4.0,
            "DC : {mag_dc} vs {}",
            gain_at(&f, 1.0)
        );
        assert!(
            (mag_ny - gain_at(&f, -1.0)).abs() <= T::TOL * 4.0,
            "Nyquist : {mag_ny} vs {}",
            gain_at(&f, -1.0)
        );
    }
}

#[test]
fn frequency_response_matches_gain_at_all_scalars() {
    check_frequency_response_matches_gain_at::<f32>();
    check_frequency_response_matches_gain_at::<f64>();
    check_frequency_response_matches_gain_at::<Q16_16>();
    check_frequency_response_matches_gain_at::<Q8_24>();
}

fn check_cascade_frequency_response_matches_cascade_gain_at<T: Scalar>() {
    // Même preuve croisée que ci-dessus, contre `cascade_gain_at` (référence
    // indépendante déjà établie pour les tests Butterworth/Chebyshev), sur
    // toute une grille de pulsations plutôt que les deux seuls extrema.
    let cascade = BiquadCascade::<T>::butterworth_lowpass(T::of(8.0), T::of(1.0), 4);
    for &omega in &[0.0, 0.3, 0.7, 1.2, 2.0, core::f64::consts::PI]
    {
        let got = magnitude(cascade.frequency_response(T::of(omega))).to_f64();
        let want = cascade_gain_at(&cascade, omega);
        assert!(
            (got - want).abs() <= T::TOL * 8.0,
            "omega={omega}: {got} vs {want}"
        );
    }
}

#[test]
fn cascade_frequency_response_matches_cascade_gain_at_all_scalars() {
    check_cascade_frequency_response_matches_cascade_gain_at::<f32>();
    check_cascade_frequency_response_matches_cascade_gain_at::<f64>();
    check_cascade_frequency_response_matches_cascade_gain_at::<Q16_16>();
    check_cascade_frequency_response_matches_cascade_gain_at::<Q8_24>();
}

fn check_frequency_response_hz_matches_normalized<T: Scalar>() {
    let (fs, f0, q) = (T::of(8.0), T::of(1.0), T::of(0.707));
    let f = Biquad::<T>::lowpass(fs, f0, q);
    // ω = 2π·freq_hz/fs, ici freq_hz = 1 Hz, fs = 8 Hz ⟹ ω = π/4.
    let omega = core::f64::consts::PI / 4.0;
    let via_hz = f.frequency_response_hz(fs, T::of(1.0));
    let via_omega = f.frequency_response(T::of(omega));
    assert!((via_hz.re.to_f64() - via_omega.re.to_f64()).abs() <= T::TOL * 4.0);
    assert!((via_hz.im.to_f64() - via_omega.im.to_f64()).abs() <= T::TOL * 4.0);
}

#[test]
fn frequency_response_hz_matches_normalized_all_scalars() {
    check_frequency_response_hz_matches_normalized::<f32>();
    check_frequency_response_hz_matches_normalized::<f64>();
    check_frequency_response_hz_matches_normalized::<Q16_16>();
    check_frequency_response_hz_matches_normalized::<Q8_24>();
}

fn check_fir_frequency_response_matches_direct_evaluation<T: Scalar>() {
    // Moyenne glissante 3 prises [1/3,1/3,1/3] : H(z) = (1+z⁻¹+z⁻²)/3.
    // Continu (z=1) : gain 1 (préserve une entrée constante, somme des
    // prises = 1). Nyquist (z=−1, z⁻¹=−1, z⁻²=1) : gain 1/3 (calcul à la
    // main, cf. en-tête de fonction).
    let third = T::of(1.0 / 3.0);
    let f = Fir::<T, 3>::new([third, third, third]);
    let mag_dc = magnitude(f.frequency_response(T::zero())).to_f64();
    let mag_ny = magnitude(f.frequency_response(T::of(core::f64::consts::PI))).to_f64();
    assert!((mag_dc - 1.0).abs() <= T::TOL * 4.0, "DC : {mag_dc}");
    assert!(
        (mag_ny - 1.0 / 3.0).abs() <= T::TOL * 4.0,
        "Nyquist : {mag_ny}"
    );
}

#[test]
fn fir_frequency_response_matches_direct_evaluation_all_scalars() {
    check_fir_frequency_response_matches_direct_evaluation::<f32>();
    check_fir_frequency_response_matches_direct_evaluation::<f64>();
    check_fir_frequency_response_matches_direct_evaluation::<Q16_16>();
    check_fir_frequency_response_matches_direct_evaluation::<Q8_24>();
}

fn check_magnitude_db_known_values<T: Scalar>() {
    assert!((magnitude_db(Complex::from_real(T::of(1.0))).to_f64()).abs() <= T::TOL * 4.0);
    assert!((magnitude_db(Complex::from_real(T::of(10.0))).to_f64() - 20.0).abs() <= T::TOL * 20.0);
    assert!(
        (magnitude_db(Complex::from_real(T::of(0.1))).to_f64() - (-20.0)).abs() <= T::TOL * 20.0
    );
}

#[test]
fn magnitude_db_known_values_all_scalars() {
    check_magnitude_db_known_values::<f32>();
    check_magnitude_db_known_values::<f64>();
    check_magnitude_db_known_values::<Q16_16>();
    check_magnitude_db_known_values::<Q8_24>();
}

fn check_phase_known_values<T: Scalar>() {
    let half_pi = core::f64::consts::FRAC_PI_2;
    let pi = core::f64::consts::PI;
    let cases = [
        ((1.0, 0.0), 0.0),
        ((0.0, 1.0), half_pi),
        ((-1.0, 0.0), pi),
        ((0.0, -1.0), -half_pi),
    ];
    for ((re, im), want) in cases
    {
        let got = phase(Complex::new(T::of(re), T::of(im))).to_f64();
        assert!(
            (got - want).abs() <= T::TOL * 4.0,
            "re={re} im={im}: {got} vs {want}"
        );
    }
}

#[test]
fn phase_known_values_all_scalars() {
    check_phase_known_values::<f32>();
    check_phase_known_values::<f64>();
    check_phase_known_values::<Q16_16>();
    check_phase_known_values::<Q8_24>();
}

fn check_unwrap_phase<T: Scalar>() {
    // Phase vraie continûment croissante (pente 0,6 rad/pas, < π : chaque
    // saut consécutif reste dans le domaine non ambigu de `unwrap_phase`),
    // repliée manuellement dans `(−π, π]` (ce que produirait `phase` en
    // pratique) — `unwrap_phase` doit exactement la reconstruire.
    let n = 13;
    let slope = 0.6;
    let two_pi = 2.0 * core::f64::consts::PI;
    let wrap = |x: f64| {
        let mut w = (x + core::f64::consts::PI).rem_euclid(two_pi);
        w -= core::f64::consts::PI;
        w
    };
    let true_phase: Vec<f64> = (0..n).map(|i| slope * i as f64).collect();
    let mut wrapped: Vec<T> = true_phase.iter().map(|&p| T::of(wrap(p))).collect();
    unwrap_phase(&mut wrapped);
    for i in 0..n
    {
        let got = wrapped[i].to_f64();
        assert!(
            (got - true_phase[i]).abs() <= T::TOL * 10.0,
            "i={i}: {got} vs {}",
            true_phase[i]
        );
    }
}

#[test]
fn unwrap_phase_reconstructs_continuous_phase_all_scalars() {
    check_unwrap_phase::<f32>();
    check_unwrap_phase::<f64>();
    check_unwrap_phase::<Q16_16>();
    check_unwrap_phase::<Q8_24>();
}

fn check_fir_linear_phase_has_constant_group_delay<T: Scalar>() {
    // Prises symétriques [1,2,3,2,1] : réponse d'amplitude réelle
    // A(ω) = (2·cos ω + 1)² (calcul à la main, cf. en-tête de fonction) —
    // un carré, donc **jamais négative** : la phase reste exactement
    // linéaire (`φ(ω) = −2ω`, N=5 prises, délai = (N−1)/2 = 2) sur tout
    // l'intervalle testé, sans le saut de π qu'un passage par un A(ω)
    // négatif provoquerait ailleurs. Grille choisie pour éviter le zéro
    // exact de A (ω = 2π/3).
    let taps = [T::of(1.0), T::of(2.0), T::of(3.0), T::of(2.0), T::of(1.0)];
    let f = Fir::<T, 5>::new(taps);
    let n_points = 21;
    let lo = 0.1;
    let hi = core::f64::consts::PI - 0.1;
    let d_omega = (hi - lo) / (n_points as f64 - 1.0);
    let omegas: Vec<T> = (0..n_points)
        .map(|i| T::of(lo + i as f64 * d_omega))
        .collect();

    let mut phases: Vec<T> = omegas
        .iter()
        .map(|&w| phase(f.frequency_response(w)))
        .collect();
    unwrap_phase(&mut phases);
    let delays = group_delay(&phases, T::of(d_omega));

    for (i, &delay) in delays.iter().enumerate()
    {
        let got = delay.to_f64();
        assert!(
            (got - 2.0).abs() <= T::TOL * 8.0,
            "i={i}: délai de groupe {got} vs 2.0 attendu"
        );
    }
}

#[test]
fn fir_linear_phase_has_constant_group_delay_all_scalars() {
    check_fir_linear_phase_has_constant_group_delay::<f32>();
    check_fir_linear_phase_has_constant_group_delay::<f64>();
    check_fir_linear_phase_has_constant_group_delay::<Q16_16>();
    check_fir_linear_phase_has_constant_group_delay::<Q8_24>();
}

#[test]
#[should_panic(expected = "group_delay")]
fn group_delay_requires_at_least_two_points() {
    let _ = group_delay::<f64>(&[1.0], 0.1);
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
//  Convolution rapide (recouvrement-addition), via rfft/irfft         //
// ------------------------------------------------------------------ //

/// Référence naïve indépendante : convolution linéaire complète en temps
/// direct (double boucle), longueur `signal.len() + kernel.len() - 1`.
fn naive_convolve(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let out_len = signal.len() + kernel.len() - 1;
    let mut out = vec![0.0f64; out_len];
    for (i, &s) in signal.iter().enumerate()
    {
        for (k, &h) in kernel.iter().enumerate()
        {
            out[i + k] += s * h;
        }
    }
    out
}

fn check_fft_convolve_matches_naive<T: Scalar>() {
    let cases = [
        (20usize, 5usize, 32usize),
        (50, 13, 64),
        (17, 17, 64), // signal et noyau de même longueur
        (100, 3, 16), // noyau très court face à un bloc large
    ];
    for &(sig_len, ker_len, fft_size) in &cases
    {
        let sig_f: Vec<f64> = (0..sig_len)
            .map(|i| ((i * 7 % 11) as f64) * 0.05 - 0.25)
            .collect();
        let ker_f: Vec<f64> = (0..ker_len)
            .map(|i| ((i * 3 % 5) as f64) * 0.1 - 0.2)
            .collect();
        let sig: Vec<T> = sig_f.iter().map(|&v| T::of(v)).collect();
        let ker: Vec<T> = ker_f.iter().map(|&v| T::of(v)).collect();

        let got = fft_convolve(&sig, &ker, fft_size);
        let want = naive_convolve(&sig_f, &ker_f);
        assert_eq!(got.len(), sig_len + ker_len - 1);
        assert_eq!(want.len(), got.len());

        let tol = T::TOL * (fft_size as f64) * 8.0 + 1e-6;
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate()
        {
            assert!(
                (g.to_f64() - w).abs() <= tol,
                "sig_len={sig_len} ker_len={ker_len} fft_size={fft_size} i={i}: {} vs {} (tol {tol})",
                g.to_f64(),
                w
            );
        }
    }
}

#[test]
fn fft_convolve_matches_naive_all_scalars() {
    check_fft_convolve_matches_naive::<f32>();
    check_fft_convolve_matches_naive::<f64>();
    check_fft_convolve_matches_naive::<Q16_16>();
}

#[test]
fn fft_convolve_impulse_kernel_is_identity() {
    let sig: Vec<Q16_16> = (0..10)
        .map(|i| Q16_16::try_from(((i % 5) as f64) * 0.1 - 0.2).unwrap())
        .collect();
    let ker = [Q16_16::one()];
    let got = fft_convolve(&sig, &ker, 16);
    assert_eq!(got.len(), sig.len());
    for (g, s) in got.iter().zip(&sig)
    {
        assert!(
            (g.to_f64() - s.to_f64()).abs() <= 1e-2,
            "{} vs {}",
            g.to_f64(),
            s.to_f64()
        );
    }
}

#[test]
#[should_panic(expected = "puissance de deux")]
fn fft_convolve_rejects_non_power_of_two_fft_size() {
    let sig = [Q16_16::one(); 10];
    let ker = [Q16_16::one(); 3];
    let _ = fft_convolve(&sig, &ker, 20);
}

#[test]
#[should_panic(expected = "noyau vide")]
fn fft_convolve_rejects_empty_kernel() {
    let sig = [Q16_16::one(); 10];
    let ker: [Q16_16; 0] = [];
    let _ = fft_convolve(&sig, &ker, 16);
}

#[test]
#[should_panic(expected = "fft_size")]
fn fft_convolve_rejects_fft_size_too_small() {
    let sig = [Q16_16::one(); 10];
    let ker = [Q16_16::one(); 16];
    let _ = fft_convolve(&sig, &ker, 16); // devrait être > kernel.len()
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
//  MFCC : DCT-II orthonormée + coefficients cepstraux sur l'échelle mel //
// ------------------------------------------------------------------ //

#[test]
fn dct2_basis_is_orthonormal() {
    // Base ortho DCT-II : C·Cᵀ = I (transformée orthogonale). Vérifie la
    // matrice `n × n` complète (`dct2_basis(n, n)`) plutôt que de passer par
    // `dct2` sur un vecteur, pour isoler la propriété structurelle.
    let n = 8;
    let basis: Vec<f64> = dct2_basis(n, n);
    for i in 0..n
    {
        for j in 0..n
        {
            let dot: f64 = (0..n).map(|k| basis[i * n + k] * basis[j * n + k]).sum();
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - want).abs() <= 1e-9,
                "i={i} j={j}: produit scalaire des lignes {dot} vs {want}"
            );
        }
    }
}

fn check_dct2_known_2point_values<T: Scalar + core::ops::Div<Output = T>>() {
    // N=2 : X₀ = √(1/2)·(a+b), X₁ = √(2/2)·(a−b) — dérivation fermée simple,
    // vérifiée indépendamment (cf. en-tête de module pour la formule générale).
    let x = [T::of(1.0), T::of(3.0)];
    let got = dct2(&x);
    assert_eq!(got.len(), 2);
    let want0 = (1.0f64 / 2.0).sqrt() * 4.0;
    let want1 = (1.0f64 / 2.0).sqrt() * (-2.0);
    assert!((got[0].to_f64() - want0).abs() <= T::TOL * 4.0);
    assert!((got[1].to_f64() - want1).abs() <= T::TOL * 4.0);
}

#[test]
fn dct2_known_2point_values_all_scalars() {
    check_dct2_known_2point_values::<f32>();
    check_dct2_known_2point_values::<f64>();
    check_dct2_known_2point_values::<Q16_16>();
}

#[test]
#[should_panic(expected = "dct2")]
fn dct2_rejects_empty_input() {
    let x: [f64; 0] = [];
    let _ = dct2(&x);
}

fn check_mfcc_truncated_matches_full_dct2<T: Scalar + core::ops::Div<Output = T>>() {
    // Mfcc précalcule une base DCT-II TRONQUÉE (n_coeffs premières lignes) —
    // preuve croisée directe : avec n_coeffs = n_mels, le résultat doit
    // coïncider avec le calcul indépendant banque-mel → ln → dct2 complet
    // (même principe que les preuves « raccourci vs calcul explicite »
    // utilisées ailleurs dans le crate, ex. depthwise_conv2d vs conv2d).
    let (n_mels, bins, sample_rate, f_min, f_max) = (10usize, 65usize, 16000.0, 0.0, 8000.0);
    let mfcc = Mfcc::<T>::new(
        n_mels,
        n_mels,
        bins,
        T::of(sample_rate),
        T::of(f_min),
        T::of(f_max),
    );
    let fb = MelFilterbank::<T>::new(n_mels, bins, T::of(sample_rate), T::of(f_min), T::of(f_max));

    let mut rng_seed = 0xABC1_2345u64;
    let mut next = move || {
        rng_seed = rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_seed >> 40) as f64 / (1u64 << 24) as f64) * 0.5 // valeurs positives modestes
    };
    let spectrogram: Vec<T> = (0..bins * 3).map(|_| T::of(next())).collect();

    let got = mfcc.apply(&spectrogram);
    let mel = fb.apply(&spectrogram);
    let frames = mel.len() / n_mels;
    let mut want = Vec::with_capacity(frames * n_mels);
    for f in 0..frames
    {
        let log_mel: Vec<T> = mel[f * n_mels..(f + 1) * n_mels]
            .iter()
            .map(|&e| e.ln())
            .collect();
        want.extend(dct2(&log_mel));
    }

    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate()
    {
        let diff = (g.to_f64() - w.to_f64()).abs();
        assert!(
            diff <= T::TOL * 8.0,
            "i={i}: {} vs {} (écart {diff})",
            g.to_f64(),
            w.to_f64()
        );
    }
}

#[test]
fn mfcc_truncated_matches_full_dct2_all_scalars() {
    check_mfcc_truncated_matches_full_dct2::<f32>();
    check_mfcc_truncated_matches_full_dct2::<f64>();
    check_mfcc_truncated_matches_full_dct2::<Q16_16>();
}

#[test]
fn mfcc_shape_and_truncation() {
    let (n_mels, n_coeffs, bins) = (12usize, 5usize, 33usize);
    let mfcc = Mfcc::<f64>::new(n_mels, n_coeffs, bins, 16000.0, 0.0, 8000.0);
    assert_eq!(mfcc.n_mels(), n_mels);
    assert_eq!(mfcc.n_coeffs(), n_coeffs);
    assert_eq!(mfcc.bins(), bins);
    let spectrogram = vec![0.3f64; bins * 4];
    let out = mfcc.apply(&spectrogram);
    assert_eq!(out.len(), 4 * n_coeffs);
}

#[test]
#[should_panic(expected = "n_coeffs")]
fn mfcc_new_rejects_zero_n_coeffs() {
    let _ = Mfcc::<f64>::new(10, 0, 33, 16000.0, 0.0, 8000.0);
}

#[test]
#[should_panic(expected = "n_coeffs")]
fn mfcc_new_rejects_n_coeffs_exceeding_n_mels() {
    let _ = Mfcc::<f64>::new(10, 11, 33, 16000.0, 0.0, 8000.0);
}

#[test]
#[should_panic(expected = "MelFilterbank::apply")]
fn mfcc_apply_dim_mismatch_panics() {
    let mfcc = Mfcc::<f64>::new(10, 5, 33, 16000.0, 0.0, 8000.0);
    let bad = vec![0.0f64; 32]; // pas multiple de bins()=33
    let _ = mfcc.apply(&bad);
}

#[test]
fn stft_mel_feeds_mfcc() {
    // Pipeline complet : signal → stft → power_spectrogram → MFCC, comme en
    // reconnaissance vocale quantifiée. Vérifie la forme et le déterminisme
    // bit-à-bit.
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

    let mfcc = Mfcc::<Q16_16>::new(
        8,
        5,
        bins,
        Q16_16::try_from(sample_rate).unwrap(),
        Q16_16::zero(),
        Q16_16::try_from(sample_rate / 2.0).unwrap(),
    );
    let out1 = mfcc.apply(&power);
    let out2 = mfcc.apply(&power);
    assert_eq!(out1, out2); // déterminisme bit-à-bit

    let frames = num_frames(n, frame_size, hop);
    assert_eq!(out1.len(), frames * mfcc.n_coeffs());
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

// -------------------------------------------------------------------- //
//  Filtres adaptatifs : Lms / Nlms / Rls                                //
// -------------------------------------------------------------------- //
//
// Système à identifier, connu et fixe (référence indépendante en `f64`, pas
// de code de la bibliothèque) : `d[n] = Σ TRUE_TAPS[k]·x[n−k]`, **sans
// bruit** — le signal désiré est une fonction linéaire exacte de l'historique
// d'entrée. Chaque filtre adaptatif, alimenté par un bruit d'excitation
// persistant, doit donc converger vers `TRUE_TAPS` (pas seulement s'en
// approcher à un plancher de bruit près).

const TRUE_TAPS: [f64; 4] = [0.5, -0.3, 0.2, -0.1];

fn lcg_noise(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (*seed >> 40) as u32;
    (bits as f64 / (1u32 << 24) as f64) * 2.0 - 1.0
}

/// `d[n] = Σ TRUE_TAPS[k]·hist[k]`, `hist[0]` = échantillon le plus récent.
fn true_fir_output(hist: &[f64; 4]) -> f64 {
    TRUE_TAPS.iter().zip(hist).map(|(&h, &x)| h * x).sum()
}

fn push_hist(hist: &mut [f64; 4], x: f64) {
    hist[3] = hist[2];
    hist[2] = hist[1];
    hist[1] = hist[0];
    hist[0] = x;
}

fn check_lms_recovers_known_fir<T: Scalar>() {
    let mut seed = 0xADAF_0001u64;
    let mut hist = [0.0f64; 4];
    let mut filt = Lms::<T, 4>::new(T::of(0.05));

    for _ in 0..4000
    {
        let xf = lcg_noise(&mut seed);
        push_hist(&mut hist, xf);
        let df = true_fir_output(&hist);
        filt.update(T::of(xf), T::of(df));
    }

    for (k, &w) in filt.weights().iter().enumerate()
    {
        assert!(
            (w.to_f64() - TRUE_TAPS[k]).abs() < 0.05,
            "Lms : poids {k} = {} attendu proche de {}",
            w.to_f64(),
            TRUE_TAPS[k]
        );
    }
}

#[test]
fn lms_recovers_known_fir_all_scalars() {
    check_lms_recovers_known_fir::<f32>();
    check_lms_recovers_known_fir::<f64>();
    check_lms_recovers_known_fir::<Q16_16>();
}

fn check_nlms_recovers_known_fir<T: Scalar + core::ops::Div<Output = T>>() {
    let mut seed = 0xADAF_0002u64;
    let mut hist = [0.0f64; 4];
    let mut filt = Nlms::<T, 4>::new(T::of(0.5), T::of(1e-3));

    for _ in 0..1500
    {
        let xf = lcg_noise(&mut seed);
        push_hist(&mut hist, xf);
        let df = true_fir_output(&hist);
        filt.update(T::of(xf), T::of(df));
    }

    for (k, &w) in filt.weights().iter().enumerate()
    {
        assert!(
            (w.to_f64() - TRUE_TAPS[k]).abs() < 0.03,
            "Nlms : poids {k} = {} attendu proche de {}",
            w.to_f64(),
            TRUE_TAPS[k]
        );
    }
}

#[test]
fn nlms_recovers_known_fir_all_scalars() {
    check_nlms_recovers_known_fir::<f32>();
    check_nlms_recovers_known_fir::<f64>();
    check_nlms_recovers_known_fir::<Q16_16>();
}

fn check_rls_recovers_known_fir<T: Scalar + core::ops::Div<Output = T>>() {
    let mut seed = 0xADAF_0003u64;
    let mut hist = [0.0f64; 4];
    let mut filt = Rls::<T, 4>::new(T::of(0.995), T::of(0.01));

    for _ in 0..300
    {
        let xf = lcg_noise(&mut seed);
        push_hist(&mut hist, xf);
        let df = true_fir_output(&hist);
        filt.update(T::of(xf), T::of(df));
    }

    for (k, &w) in filt.weights().iter().enumerate()
    {
        assert!(
            (w.to_f64() - TRUE_TAPS[k]).abs() < 0.01,
            "Rls : poids {k} = {} attendu proche de {}",
            w.to_f64(),
            TRUE_TAPS[k]
        );
    }
}

#[test]
fn rls_recovers_known_fir_all_scalars() {
    check_rls_recovers_known_fir::<f32>();
    check_rls_recovers_known_fir::<f64>();
    check_rls_recovers_known_fir::<Q16_16>();
}

#[test]
fn rls_i64_storage() {
    // NumericScalar (donc Lms/Nlms/Rls) est implémenté pour tout FixedStorage,
    // pas seulement i32 : Q32_32 (stockage i64) doit fonctionner à l'identique.
    let mut seed = 0xADAF_0004u64;
    let mut hist = [0.0f64; 4];
    let mut filt = Rls::<Q32_32, 4>::new(
        Q32_32::try_from(0.995).unwrap(),
        Q32_32::try_from(0.01).unwrap(),
    );

    for _ in 0..300
    {
        let xf = lcg_noise(&mut seed);
        push_hist(&mut hist, xf);
        let df = true_fir_output(&hist);
        filt.update(Q32_32::try_from(xf).unwrap(), Q32_32::try_from(df).unwrap());
    }

    for (k, &w) in filt.weights().iter().enumerate()
    {
        assert!(
            (Q32_32::to_f64(w) - TRUE_TAPS[k]).abs() < 0.01,
            "Rls (i64) : poids {k} = {} attendu proche de {}",
            Q32_32::to_f64(w),
            TRUE_TAPS[k]
        );
    }
}

/// RLS converge en très peu d'échantillons (proche de la solution exacte des
/// moindres carrés dès que l'excitation couvre l'espace des poids) alors que
/// LMS/NLMS progressent par petits pas de gradient : sur la même séquence
/// bruitée, sans bruit de mesure, l'erreur RLS doit être nettement plus
/// faible après un nombre d'échantillons trop court pour LMS/NLMS.
#[test]
fn rls_converges_faster_than_lms_and_nlms_after_few_samples() {
    let mut seed = 0xF00D_0001u64;
    let mut hist = [0.0f64; 4];
    let n_samples = 30;
    let mut xs = Vec::with_capacity(n_samples);
    let mut ds = Vec::with_capacity(n_samples);
    for _ in 0..n_samples
    {
        let xf = lcg_noise(&mut seed);
        push_hist(&mut hist, xf);
        xs.push(xf);
        ds.push(true_fir_output(&hist));
    }

    let mut lms = Lms::<f64, 4>::new(0.05);
    let mut nlms = Nlms::<f64, 4>::new(0.5, 1e-3);
    let mut rls = Rls::<f64, 4>::new(0.995, 0.01);
    for i in 0..n_samples
    {
        lms.update(xs[i], ds[i]);
        nlms.update(xs[i], ds[i]);
        rls.update(xs[i], ds[i]);
    }

    let sq_err = |w: &[f64; 4]| -> f64 {
        w.iter()
            .zip(&TRUE_TAPS)
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    };
    let e_lms = sq_err(lms.weights());
    let e_nlms = sq_err(nlms.weights());
    let e_rls = sq_err(rls.weights());

    assert!(
        e_rls < e_lms,
        "RLS ({e_rls}) devrait converger plus vite que LMS ({e_lms}) après {n_samples} échantillons"
    );
    assert!(
        e_rls < e_nlms,
        "RLS ({e_rls}) devrait converger plus vite que NLMS ({e_nlms}) après {n_samples} échantillons"
    );
}

#[test]
fn lms_zero_input_leaves_weights_at_zero() {
    let mut filt = Lms::<Q16_16, 4>::new(Q16_16::try_from(0.1).unwrap());
    for _ in 0..50
    {
        let (y, error) = filt.update(Q16_16::zero(), Q16_16::zero());
        assert_eq!(y, Q16_16::zero());
        assert_eq!(error, Q16_16::zero());
    }
    assert_eq!(filt.weights(), &[Q16_16::zero(); 4]);
}

#[test]
fn nlms_zero_input_does_not_panic_and_leaves_weights_at_zero() {
    // `energy + eps` reste strictement positif même à énergie nulle : pas de
    // division par zéro, pas de NaN/saturation, poids inchangés.
    let mut filt = Nlms::<Q16_16, 4>::new(
        Q16_16::try_from(0.5).unwrap(),
        Q16_16::try_from(1e-3).unwrap(),
    );
    for _ in 0..50
    {
        filt.update(Q16_16::zero(), Q16_16::zero());
    }
    assert_eq!(filt.weights(), &[Q16_16::zero(); 4]);
}

#[test]
fn rls_zero_input_leaves_weights_and_covariance_unchanged() {
    // `lambda = 1` (pas d'oubli) : sans excitation, le gain est nul et la
    // covariance n'évolue pas (pas de « covariance windup »).
    let lambda = Q16_16::one();
    let delta = Q16_16::try_from(0.01).unwrap();
    let mut filt = Rls::<Q16_16, 3>::new(lambda, delta);
    let p0 = *filt.covariance();

    for _ in 0..50
    {
        filt.update(Q16_16::zero(), Q16_16::zero());
    }

    assert_eq!(filt.weights(), &[Q16_16::zero(); 3]);
    assert_eq!(*filt.covariance(), p0);
}

#[test]
fn lms_reset_zeroes_weights_and_delay_line() {
    let mut seed = 0x1234u64;
    let mut hist = [0.0f64; 4];
    let mut filt = Lms::<Q16_16, 4>::new(Q16_16::try_from(0.05).unwrap());
    for _ in 0..200
    {
        let xf = lcg_noise(&mut seed);
        push_hist(&mut hist, xf);
        let df = true_fir_output(&hist);
        filt.update(Q16_16::try_from(xf).unwrap(), Q16_16::try_from(df).unwrap());
    }
    assert_ne!(filt.weights(), &[Q16_16::zero(); 4]);

    filt.reset();
    assert_eq!(filt.weights(), &[Q16_16::zero(); 4]);
    // Après reset, une entrée nulle doit reproduire l'état initial exact.
    let (y, error) = filt.update(Q16_16::zero(), Q16_16::zero());
    assert_eq!(y, Q16_16::zero());
    assert_eq!(error, Q16_16::zero());
}

#[test]
fn rls_reset_restores_initial_covariance() {
    let delta = Q16_16::try_from(0.01).unwrap();
    let mut seed = 0x5678u64;
    let mut hist = [0.0f64; 4];
    let mut filt = Rls::<Q16_16, 3>::new(Q16_16::try_from(0.99).unwrap(), delta);
    let p0 = *filt.covariance();

    for _ in 0..100
    {
        let xf = lcg_noise(&mut seed);
        push_hist(&mut hist, xf);
        let df = true_fir_output(&hist);
        filt.update(Q16_16::try_from(xf).unwrap(), Q16_16::try_from(df).unwrap());
    }
    assert_ne!(*filt.covariance(), p0);

    filt.reset();
    assert_eq!(filt.weights(), &[Q16_16::zero(); 3]);
    assert_eq!(*filt.covariance(), p0);
}

#[test]
fn lms_single_tap_tracks_gain() {
    // N=1 : un simple gain adaptatif doit converger vers TRUE_TAPS[0] quand
    // le désiré est un multiple pur de l'entrée (pas d'historique impliqué).
    let mut seed = 0x9ABCu64;
    let mut filt = Lms::<f64, 1>::new(0.05);
    for _ in 0..2000
    {
        let xf = lcg_noise(&mut seed);
        filt.update(xf, 0.5 * xf);
    }
    assert!((filt.weights()[0] - 0.5).abs() < 0.02);
}

// ------------------------------------------------------------------ //
//  PLL : Nco, PiLoopFilter, Pll                                       //
// ------------------------------------------------------------------ //

fn check_nco_known_unit_circle_points<T: Scalar>() {
    // Incrément π/2 : la référence tourne d'un quart de tour par échantillon,
    // (sin, cos) doit visiter exactement les quatre points cardinaux.
    let mut nco = Nco::<T>::new(T::of(core::f64::consts::FRAC_PI_2));
    let expected = [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)];
    for &(es, ec) in &expected
    {
        let (s, c) = nco.tick();
        assert!(
            (s.to_f64() - es).abs() < 1e-3,
            "sin = {} attendu {es}",
            s.to_f64()
        );
        assert!(
            (c.to_f64() - ec).abs() < 1e-3,
            "cos = {} attendu {ec}",
            c.to_f64()
        );
    }
}

#[test]
fn nco_known_unit_circle_points_all_scalars() {
    check_nco_known_unit_circle_points::<f32>();
    check_nco_known_unit_circle_points::<f64>();
    check_nco_known_unit_circle_points::<Q16_16>();
}

fn check_nco_phase_stays_bounded<T: Scalar>() {
    // 0.7 rad/échantillon ne divise pas 2π : rembobinages fréquents sur de
    // nombreux tours, la phase doit rester dans (−π, π] à chaque pas.
    let mut nco = Nco::<T>::new(T::of(0.7));
    let pi = core::f64::consts::PI;
    for _ in 0..2000
    {
        let _ = nco.tick();
        let p = nco.phase().to_f64();
        assert!(p > -pi - 1e-6 && p <= pi + 1e-6, "phase hors bornes : {p}");
    }
}

#[test]
fn nco_phase_stays_bounded_all_scalars() {
    check_nco_phase_stays_bounded::<f32>();
    check_nco_phase_stays_bounded::<f64>();
    check_nco_phase_stays_bounded::<Q16_16>();
}

#[test]
fn nco_set_frequency_changes_next_increment() {
    let mut nco = Nco::<f64>::new(0.1);
    let _ = nco.tick(); // phase = 0.1
    nco.set_frequency(0.5);
    let _ = nco.tick(); // phase = 0.1 + 0.5 = 0.6
    assert!((nco.phase() - 0.6).abs() < 1e-9, "phase = {}", nco.phase());
}

fn check_pi_loop_filter_pure_integral<T: Scalar>() {
    // Kp = 0 : la sortie est exactement l'intégrateur, qui accumule Ki·erreur.
    let mut f = PiLoopFilter::<T>::new(T::of(0.0), T::of(1.0));
    let mut expected = 0.0;
    for _ in 0..10
    {
        let out = f.step(T::of(0.1));
        expected += 0.1;
        assert!(
            (out.to_f64() - expected).abs() < 1e-2,
            "intégrateur = {} attendu {expected}",
            out.to_f64()
        );
    }
}

#[test]
fn pi_loop_filter_pure_integral_all_scalars() {
    check_pi_loop_filter_pure_integral::<f32>();
    check_pi_loop_filter_pure_integral::<f64>();
    check_pi_loop_filter_pure_integral::<Q16_16>();
}

fn check_pi_loop_filter_design_scales_with_bandwidth<T: Scalar + core::ops::Div<Output = T>>() {
    // Une bande passante de boucle plus large doit donner des gains Kp/Ki
    // plus grands (loi de conception, cf. en-tête de module).
    let narrow = PiLoopFilter::<T>::design(T::of(0.5), T::of(0.707), T::of(100.0));
    let wide = PiLoopFilter::<T>::design(T::of(5.0), T::of(0.707), T::of(100.0));
    assert!(
        wide.kp().to_f64() > narrow.kp().to_f64(),
        "Kp large = {} attendu > Kp étroit = {}",
        wide.kp().to_f64(),
        narrow.kp().to_f64()
    );
    assert!(
        wide.ki().to_f64() > narrow.ki().to_f64(),
        "Ki large = {} attendu > Ki étroit = {}",
        wide.ki().to_f64(),
        narrow.ki().to_f64()
    );
}

#[test]
fn pi_loop_filter_design_scales_with_bandwidth_all_scalars() {
    check_pi_loop_filter_design_scales_with_bandwidth::<f32>();
    check_pi_loop_filter_design_scales_with_bandwidth::<f64>();
    check_pi_loop_filter_design_scales_with_bandwidth::<Q16_16>();
}

#[test]
fn pi_loop_filter_reset_clears_integrator() {
    let mut f = PiLoopFilter::<f64>::new(0.0, 1.0);
    f.step(1.0);
    assert_ne!(f.integrator(), 0.0);
    f.reset();
    assert_eq!(f.integrator(), 0.0);
}

/// `sample_rate`/`center_freq`/`loop_bandwidth` en unités normalisées, comme
/// les tests de [`Biquad`] (`sample_rate = 8.0` etc.) — l'implémentation est
/// générique, ces valeurs ne représentent aucune fréquence audio réelle.
const PLL_SAMPLE_RATE: f64 = 100.0;
const PLL_CENTER_FREQ: f64 = 10.0;
const PLL_LOOP_BW: f64 = 2.0;
const PLL_DAMPING: f64 = 0.707;

fn check_pll_process_locks_onto_frequency_offset<T: Scalar + core::ops::Div<Output = T>>() {
    let freq_offset = 1.0; // Hz
    let actual_freq_per_sample =
        2.0 * core::f64::consts::PI * (PLL_CENTER_FREQ + freq_offset) / PLL_SAMPLE_RATE;

    let mut pll = Pll::<T>::new(
        T::of(PLL_SAMPLE_RATE),
        T::of(PLL_CENTER_FREQ),
        T::of(PLL_LOOP_BW),
        T::of(PLL_DAMPING),
    );

    let mut n = 0usize;
    for _ in 0..20_000
    {
        let true_phase = actual_freq_per_sample * n as f64;
        let _ = pll.process(T::of(true_phase.cos()));
        n += 1;
    }

    // Moyenne glissante sur une fenêtre finale : le détecteur de phase par
    // produit (entrée réelle) laisse passer une ondulation résiduelle à
    // `2ω` non filtrée séparément (cf. en-tête de module — valide seulement
    // pour `loop_bandwidth ≪ center_freq`, ratio modeste ici pour un test
    // rapide) ; une lecture instantanée de `frequency_estimate()` peut donc
    // tomber sur n'importe quelle phase de cette ondulation. La moyenner sur
    // plusieurs dizaines de périodes porteuses donne la valeur de
    // convergence réelle.
    let window = 500;
    let mut sum_freq = 0.0;
    for _ in 0..window
    {
        let true_phase = actual_freq_per_sample * n as f64;
        let _ = pll.process(T::of(true_phase.cos()));
        sum_freq += pll.frequency_estimate().to_f64();
        n += 1;
    }
    let avg_freq = sum_freq / window as f64;
    assert!(
        (avg_freq - actual_freq_per_sample).abs() < 5e-3,
        "fréquence estimée (moyenne) = {avg_freq} attendue proche de {actual_freq_per_sample}"
    );
}

#[test]
fn pll_process_locks_onto_frequency_offset_all_scalars() {
    check_pll_process_locks_onto_frequency_offset::<f32>();
    check_pll_process_locks_onto_frequency_offset::<f64>();
    check_pll_process_locks_onto_frequency_offset::<Q16_16>();
}

fn check_pll_process_quadrature_locks_onto_phase_and_frequency<
    T: Scalar + core::ops::Div<Output = T>,
>() {
    let freq_offset = 0.5; // Hz
    let phase_offset = 0.3; // rad
    let actual_freq_per_sample =
        2.0 * core::f64::consts::PI * (PLL_CENTER_FREQ + freq_offset) / PLL_SAMPLE_RATE;

    let mut pll = Pll::<T>::new(
        T::of(PLL_SAMPLE_RATE),
        T::of(PLL_CENTER_FREQ),
        T::of(PLL_LOOP_BW),
        T::of(PLL_DAMPING),
    );

    let mut last_error = 1.0f64;
    for n in 0..20_000usize
    {
        let true_phase = actual_freq_per_sample * n as f64 + phase_offset;
        let (i, q) = (true_phase.cos(), true_phase.sin());
        last_error = pll.process_quadrature(T::of(i), T::of(q)).to_f64();
    }

    assert!(
        last_error.abs() < 0.05,
        "erreur de phase en régime établi = {last_error} attendue proche de 0"
    );
    let est = pll.frequency_estimate().to_f64();
    assert!(
        (est - actual_freq_per_sample).abs() < 5e-3,
        "fréquence estimée = {est} attendue proche de {actual_freq_per_sample}"
    );
    // Le détecteur de phase par atan2 est exact (pas de terme battant à 2ω,
    // contrairement à process() sur entrée réelle) : une lecture
    // instantanée de locked() est donc pertinente ici.
    assert!(
        pll.locked(T::of(0.05)),
        "PLL non verrouillée après convergence (dernière erreur = {last_error})"
    );
}

#[test]
fn pll_process_quadrature_locks_onto_phase_and_frequency_all_scalars() {
    check_pll_process_quadrature_locks_onto_phase_and_frequency::<f32>();
    check_pll_process_quadrature_locks_onto_phase_and_frequency::<f64>();
    check_pll_process_quadrature_locks_onto_phase_and_frequency::<Q16_16>();
}

#[test]
fn pll_not_locked_before_convergence() {
    // Grand écart de fréquence (+5 Hz), seuil serré : après seulement 5
    // échantillons (bien avant convergence), la PLL ne doit pas se
    // rapporter verrouillée.
    let freq_offset = 5.0;
    let actual_freq_per_sample =
        2.0 * core::f64::consts::PI * (PLL_CENTER_FREQ + freq_offset) / PLL_SAMPLE_RATE;
    let mut pll = Pll::<f64>::new(PLL_SAMPLE_RATE, PLL_CENTER_FREQ, PLL_LOOP_BW, PLL_DAMPING);
    for n in 0..5usize
    {
        let true_phase = actual_freq_per_sample * n as f64;
        let _ = pll.process(true_phase.cos());
    }
    assert!(
        !pll.locked(1e-4),
        "verrouillée de façon inattendue trop tôt"
    );
}

#[test]
fn pll_reset_restores_initial_state() {
    let mut pll = Pll::<f64>::new(PLL_SAMPLE_RATE, PLL_CENTER_FREQ, PLL_LOOP_BW, PLL_DAMPING);
    let initial_freq = pll.frequency_estimate();
    for n in 0..500usize
    {
        let true_phase = 2.0 * core::f64::consts::PI * 12.0 * n as f64 / PLL_SAMPLE_RATE;
        let _ = pll.process(true_phase.cos());
    }
    assert_ne!(pll.frequency_estimate(), initial_freq);
    pll.reset();
    assert_eq!(pll.frequency_estimate(), initial_freq);
    assert_eq!(pll.phase_estimate(), 0.0);
}
