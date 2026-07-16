// scirust-simd/src/reductions/tests.rs
//
// Validation du socle de réductions.
//
// Stratégie :
// * **Exactitude bit-à-bit** sur des données à valeurs entières petites
//   (représentables sans arrondi en f32/f64) → assertions `==`.
// * **Tolérance** sur des données aléatoires réelles.
// * **Propriétés métamorphiques** (mise à l'échelle des normes, bornes du
//   cosinus) et **cas limites** (vide, longueur non multiple de la largeur).

use super::*;

/// LCG déterministe → f64. Reproductible bit à bit.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    fn small_int(&mut self) -> f64 {
        ((self.next_u64() >> 40) % 17) as f64 - 8.0
    }
}

fn int_vec(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..n).map(|_| rng.small_int()).collect()
}

fn real_vec(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..n).map(|_| rng.unit()).collect()
}

// ------------------------------------------------------------------ //
//  SimdScalar : opérations de base                                    //
// ------------------------------------------------------------------ //

#[test]
fn simd_scalar_basic_ops() {
    use std::simd::{f32x8, f64x4};
    assert_eq!(<f32x8 as SimdScalar>::LANES, 8);
    assert_eq!(<f64x4 as SimdScalar>::LANES, 4);

    let v = f32x8::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(v.reduce_sum(), 36.0);
    assert_eq!(v.reduce_max(), 8.0);
    assert_eq!(v.reduce_min(), 1.0);
    assert_eq!(v.lane(0), 1.0);
    assert_eq!(v.lane(7), 8.0);

    let z = f32x8::zero();
    assert_eq!(z.reduce_sum(), 0.0);
    let s = f32x8::splat(2.0);
    // FMA : v*s + s = 2v + 2
    let fma = v.mul_add(s, s);
    assert_eq!(fma.lane(0), 1.0 * 2.0 + 2.0);
    assert_eq!(fma.lane(7), 8.0 * 2.0 + 2.0);

    let neg = f32x8::from_slice(&[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]);
    assert_eq!(SimdScalar::abs(neg).reduce_sum(), 36.0);
}

// ------------------------------------------------------------------ //
//  Sommes : exactitude sur entiers                                    //
// ------------------------------------------------------------------ //

#[test]
fn sums_exact_on_integer_data() {
    // Longueurs variées, dont non-multiples de la largeur (8 pour f32, 4 pour f64).
    for &n in &[0usize, 1, 3, 7, 8, 9, 15, 16, 100, 257]
    {
        let data64 = int_vec(0xA11 + n as u64, n);
        let exact: f64 = data64.iter().sum();
        assert_eq!(sum_fast(&data64), exact, "f64 fast n={n}");
        assert_eq!(sum_deterministic(&data64), exact, "f64 det n={n}");
        assert_eq!(sum_kahan(&data64), exact, "f64 kahan n={n}");
        assert_eq!(sum(&data64, ReductionMode::Fast), exact, "f64 sum() n={n}");

        let data32: Vec<f32> = data64.iter().map(|&x| x as f32).collect();
        let exact32: f32 = data32.iter().sum();
        assert_eq!(sum_fast(&data32), exact32, "f32 fast n={n}");
        assert_eq!(sum_deterministic(&data32), exact32, "f32 det n={n}");
        assert_eq!(sum_kahan(&data32), exact32, "f32 kahan n={n}");
    }
}

#[test]
fn sums_close_to_reference_on_real_data() {
    let data = real_vec(0x5011, 1000);
    let reference: f64 = data.iter().sum();
    // Toutes les variantes doivent être proches de la somme naïve.
    for (name, got) in [
        ("fast", sum_fast(&data)),
        ("det", sum_deterministic(&data)),
        ("kahan", sum_kahan(&data)),
    ]
    {
        assert!(
            (got - reference).abs() < 1e-9,
            "{name}: {got} vs {reference}"
        );
    }
}

#[test]
fn deterministic_sum_is_reproducible() {
    // La somme déterministe ne dépend PAS de l'ordre matériel de réduction :
    // on la compare à une réimplémentation scalaire de son ordre exact
    // (lanes k, k+W, … puis réduction des lanes en ordre d'indice).
    let data = real_vec(0xDE7, 333);
    let w = <f64 as SimdReducible>::WIDTH;
    let full = data.len() / w * w;
    let mut lanes = vec![0.0f64; w];
    let mut i = 0;
    while i < full
    {
        for (k, lane) in lanes.iter_mut().enumerate()
        {
            *lane += data[i + k];
        }
        i += w;
    }
    let mut expected = 0.0;
    for &lane in &lanes
    {
        expected += lane;
    }
    for &v in &data[full..]
    {
        expected += v;
    }
    assert_eq!(sum_deterministic(&data), expected);
}

#[test]
fn kahan_beats_naive_on_ill_conditioned_sum() {
    // 1.0 suivi de N petits incréments : la somme naïve f32 perd les petits
    // termes (absorption), Kahan les récupère. Vérité calculée en f64.
    let n = 100_000;
    let small = 1e-3f32;
    let mut data = vec![small; n];
    data[0] = 1.0;
    let truth = 1.0f64 + (n as f64 - 1.0) * small as f64;

    let naive: f32 = data.iter().copied().sum();
    let kahan = sum_kahan(&data);

    let err_naive = (naive as f64 - truth).abs();
    let err_kahan = (kahan as f64 - truth).abs();
    assert!(
        err_kahan < err_naive,
        "Kahan ({err_kahan:e}) doit battre naïf ({err_naive:e})"
    );
    // Kahan doit être très proche de la vérité.
    assert!(
        err_kahan / truth < 1e-5,
        "erreur relative Kahan trop grande"
    );
}

// ------------------------------------------------------------------ //
//  Produit scalaire, normes, cosinus                                  //
// ------------------------------------------------------------------ //

#[test]
fn dot_exact_on_integers() {
    for &n in &[0usize, 1, 5, 8, 13, 64, 129]
    {
        let a = int_vec(0xD0 + n as u64, n);
        let b = int_vec(0xD1 + n as u64, n);
        let exact: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert_eq!(dot(&a, &b, ReductionMode::Fast), exact, "fast n={n}");
        assert_eq!(
            dot(&a, &b, ReductionMode::Deterministic),
            exact,
            "det n={n}"
        );
    }
}

#[test]
#[should_panic(expected = "longueurs")]
fn dot_length_mismatch_panics() {
    let _ = dot(&[1.0f32, 2.0], &[1.0f32], ReductionMode::Fast);
}

#[test]
fn norms_known_values() {
    // [3, 4] → L2 = 5, L1 = 7, L2² = 25.
    let v = [3.0f64, 4.0];
    assert_eq!(l2_norm_sqr(&v, ReductionMode::Fast), 25.0);
    assert_eq!(l2_norm(&v, ReductionMode::Fast), 5.0);
    assert_eq!(l1_norm(&v, ReductionMode::Fast), 7.0);

    // [-3, 4] : L1 = 7, L2 = 5 (abs).
    let v2 = [-3.0f64, 4.0];
    assert_eq!(l1_norm(&v2, ReductionMode::Fast), 7.0);
    assert_eq!(l2_norm(&v2, ReductionMode::Fast), 5.0);
}

#[test]
fn l2_norm_is_absolutely_homogeneous() {
    // Propriété métamorphique : ‖a·x‖ = |a|·‖x‖.
    let x = real_vec(0x2, 200);
    let base = l2_norm(&x, ReductionMode::Fast);
    for &a in &[2.0f64, -3.5, 0.25, -1.0]
    {
        let scaled: Vec<f64> = x.iter().map(|&v| a * v).collect();
        let got = l2_norm(&scaled, ReductionMode::Fast);
        assert!((got - a.abs() * base).abs() < 1e-9, "a={a}");
    }
}

#[test]
fn cosine_similarity_bounds_and_identities() {
    let x = real_vec(0x11, 128);
    // cos(x, x) = 1.
    assert!((cosine_similarity(&x, &x, ReductionMode::Fast) - 1.0).abs() < 1e-9);
    // cos(x, -x) = -1.
    let neg: Vec<f64> = x.iter().map(|&v| -v).collect();
    assert!((cosine_similarity(&x, &neg, ReductionMode::Fast) + 1.0).abs() < 1e-9);
    // cos ∈ [-1, 1] sur des paires aléatoires.
    let y = real_vec(0x22, 128);
    let c = cosine_similarity(&x, &y, ReductionMode::Fast);
    assert!((-1.0..=1.0).contains(&c), "cos hors bornes: {c}");
    // Vecteur nul → 0 par convention.
    let zero = vec![0.0f64; 128];
    assert_eq!(cosine_similarity(&x, &zero, ReductionMode::Fast), 0.0);
}

#[test]
fn cosine_orthogonal_is_zero() {
    // Deux vecteurs orthogonaux → cosinus nul.
    let a = [1.0f64, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let b = [0.0f64, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    assert_eq!(dot(&a, &b, ReductionMode::Fast), 0.0);
    assert_eq!(cosine_similarity(&a, &b, ReductionMode::Fast), 0.0);
}

// ------------------------------------------------------------------ //
//  Extrema & argmax                                                   //
// ------------------------------------------------------------------ //

#[test]
fn extrema_and_argmax() {
    // Max/min dans la queue (indice > multiple de largeur) et au milieu.
    let data = [
        1.0f32, -2.0, 3.0, 0.5, -7.0, 4.0, 2.0, 1.5, // premier chunk f32x8
        9.0, -1.0, 3.0, // queue : max global 9.0 à l'indice 8
    ];
    assert_eq!(reduce_max(&data), Some(9.0));
    assert_eq!(reduce_min(&data), Some(-7.0));
    assert_eq!(argmax(&data), Some(8));

    // Max dans le corps vectoriel.
    let body_max = [1.0f64, 2.0, 8.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    assert_eq!(reduce_max(&body_max), Some(8.0));
    assert_eq!(argmax(&body_max), Some(2));

    // Premier indice en cas d'égalité.
    let ties = [5.0f32, 1.0, 5.0, 2.0, 5.0];
    assert_eq!(argmax(&ties), Some(0));
}

#[test]
fn empty_edge_cases() {
    let empty: [f32; 0] = [];
    assert_eq!(sum_fast(&empty), 0.0);
    assert_eq!(sum_deterministic(&empty), 0.0);
    assert_eq!(sum_kahan(&empty), 0.0);
    assert_eq!(reduce_max(&empty), None);
    assert_eq!(reduce_min(&empty), None);
    assert_eq!(argmax(&empty), None);
    assert_eq!(l1_norm(&empty, ReductionMode::Fast), 0.0);
    assert_eq!(l2_norm(&empty, ReductionMode::Fast), 0.0);
}

#[test]
fn single_element() {
    let one = [42.0f64];
    assert_eq!(sum_fast(&one), 42.0);
    assert_eq!(sum_kahan(&one), 42.0);
    assert_eq!(reduce_max(&one), Some(42.0));
    assert_eq!(argmax(&one), Some(0));
    assert_eq!(l2_norm(&one, ReductionMode::Deterministic), 42.0);
}
