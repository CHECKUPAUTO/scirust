//! Voie f32 **portable** : transcendantales et noyaux bit-exacts
//! inter-plates-formes **par construction**.
//!
//! ## Pourquoi c'est bit-exact partout
//!
//! Tout ce module n'utilise que des opérations IEEE-754 **de base**
//! (`+ − × ÷`, comparaisons, casts f32↔f64, manipulations de bits entières) :
//! le standard impose leur arrondi correct, donc chacune donne le même bit
//! pattern sur toute plate-forme conforme. L'ordre des opérations est figé par
//! le code, Rust ne fusionne jamais `a*b + c` en FMA implicite et aucune
//! fonction de la **libm** n'est appelée (contrairement à `f32::exp`, dont le
//! résultat dépend de la libm de la plate-forme). Les constantes sont des
//! littéraux arrondis correctement à la compilation. Conséquence : deux
//! binaires Rust sur deux architectures différentes (x86-64, aarch64, …)
//! produisent des sorties **bit-identiques** — c'est l'axe « cross-platform
//! f32 » identifié dans `AUDIT_REPDL_2026-07-10.md` et acté comme travail
//! futur dans `paper/RELATED_WORK.md`, réalisé ici en Rust pur, zéro FFI.
//!
//! ## Classe de garantie (à énoncer précisément)
//!
//! - **Portabilité** : bit-identique sur toute cible Rust dont f32/f64 suivent
//!   IEEE-754 (toutes les cibles tier-1 ; exclut les cibles x87 sans SSE2 type
//!   `i586`, dont l'arithmétique 80 bits n'est pas conforme).
//! - **Précision** : l'évaluation interne en f64 (erreur relative ≲ 2⁻⁴⁷)
//!   rend le résultat f32 **fidèlement arrondi** (≤ 1 ulp) partout, et
//!   correctement arrondi sauf si la valeur exacte tombe à ≈ 2⁻⁴⁷ (relatif)
//!   d'une frontière d'arrondi f32. L'arrondi correct **prouvé** (dilemme du
//!   fabricant de tables) reste un travail futur — on ne le revendique pas.
//! - **Performance** : voie de référence/d'audit, pas optimisée (GEMM naïf
//!   mono-thread, softmax allouant). Pour la vitesse intra-architecture,
//!   utiliser les chemins SIMD ; pour le bit-exact cross-platform rapide,
//!   utiliser la voie int8 (`quantization`).
//!
//! ## Algorithmes (méthodes mathématiques publiques, implémentation originale)
//!
//! `exp` : réduction d'argument classique e^y = 2^k·e^r avec k = ⌊y/ln 2⌉ et
//! ln 2 scindé hi/lo pour une réduction quasi exacte, série de Taylor de degré
//! 13 sur r ∈ [−ln 2/2, ln 2/2] (troncature < 2⁻⁵⁷), remise à l'échelle 2^k
//! par construction directe de l'exposant. `ln` : normalisation de la
//! mantisse dans [√2/2, √2], puis ln m = 2 atanh(s) avec s = (m−1)/(m+1) et
//! la série impaire de atanh (|s| ≤ 0,172, troncature < 2⁻⁶⁰). Les
//! coefficients sont des faits mathématiques (1/n!, 1/(2k+1)) — aucune table
//! ni code d'une implémentation existante (fdlibm, musl, RepDL, …) n'a été
//! consulté ni copié : réimplémentation clean-room depuis les mathématiques.
//!
//! Les réductions restent couvertes par [`crate::reproducible`] (sommes
//! correctement arrondies, indépendantes de l'ordre — elles aussi portables
//! par construction) ; [`softmax_f32`] compose les deux.

use crate::reproducible::reproducible_sum;

/// NaN canonique (bits figés) : contrairement à `f32::NAN`, dont le bit
/// pattern n'est pas garanti stable, celui-ci est identique partout.
const CANONICAL_NAN: f32 = f32::from_bits(0x7fc0_0000);

/// ln 2 tronqué à 28 bits de mantisse : k·LN2_HI est exact pour |k| ≤ 2⁸.
const LN2_HI: f64 = f64::from_bits(core::f64::consts::LN_2.to_bits() & 0xFFFF_FFFF_FF00_0000);
/// Reste ln 2 − LN2_HI (différence exacte en f64).
const LN2_LO: f64 = core::f64::consts::LN_2 - LN2_HI;

/// `exp(x)` portable : bit-exact inter-plates-formes par construction,
/// fidèlement arrondi (cf. doc du module). NaN → NaN canonique ;
/// saturations : `x ≥ 89` → `+∞` (e^88,73 > `f32::MAX`), `x ≤ −105` → `0`
/// (sous le plus petit sous-normal) ; les sous-normaux en sortie sont
/// produits par l'arrondi du cast f64→f32.
pub fn exp_f32(x: f32) -> f32 {
    if x.is_nan()
    {
        return CANONICAL_NAN;
    }
    if x >= 89.0
    {
        return f32::INFINITY;
    }
    if x <= -105.0
    {
        return 0.0;
    }
    let y = x as f64; // promotion exacte
    // Réduction : y = k·ln2 + r, |r| ≤ ln2/2 (+ arrondi de k)
    let k = (y * core::f64::consts::LOG2_E).round();
    let r = (y - k * LN2_HI) - k * LN2_LO;
    // e^r par Taylor degré 13 (Horner) : troncature < 2⁻⁵⁷ sur |r| ≤ 0,3533
    let c13 = 1.0 / 6_227_020_800.0;
    let c12 = 1.0 / 479_001_600.0;
    let c11 = 1.0 / 39_916_800.0;
    let c10 = 1.0 / 3_628_800.0;
    let c9 = 1.0 / 362_880.0;
    let c8 = 1.0 / 40_320.0;
    let c7 = 1.0 / 5_040.0;
    let c6 = 1.0 / 720.0;
    let c5 = 1.0 / 120.0;
    let c4 = 1.0 / 24.0;
    let c3 = 1.0 / 6.0;
    let mut p = c13;
    p = p * r + c12;
    p = p * r + c11;
    p = p * r + c10;
    p = p * r + c9;
    p = p * r + c8;
    p = p * r + c7;
    p = p * r + c6;
    p = p * r + c5;
    p = p * r + c4;
    p = p * r + c3;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    // 2^k exact par construction de l'exposant (k ∈ [−152, 129] ⇒ f64 normal)
    let scale = f64::from_bits(((1023 + k as i64) as u64) << 52);
    (p * scale) as f32
}

/// `ln(x)` portable : bit-exact inter-plates-formes par construction,
/// fidèlement arrondi (cf. doc du module). `ln(NaN)` et `ln(x<0)` → NaN
/// canonique, `ln(±0)` → `−∞`, `ln(+∞)` → `+∞` ; les entrées sous-normales
/// sont exactes après promotion en f64.
pub fn ln_f32(x: f32) -> f32 {
    if x.is_nan() || x < 0.0
    {
        return CANONICAL_NAN;
    }
    if x == 0.0
    {
        return f32::NEG_INFINITY;
    }
    if x == f32::INFINITY
    {
        return f32::INFINITY;
    }
    let y = x as f64; // promotion exacte (les sous-normaux f32 deviennent normaux)
    let bits = y.to_bits();
    let mut e = (((bits >> 52) & 0x7ff) as i64) - 1023;
    let mut m = f64::from_bits((bits & 0x000F_FFFF_FFFF_FFFF) | (1023u64 << 52));
    if m > core::f64::consts::SQRT_2
    {
        m *= 0.5;
        e += 1;
    }
    // ln m = 2·atanh(s), s = (m−1)/(m+1) ∈ [−0,1716, 0,1716]
    let s = (m - 1.0) / (m + 1.0);
    let z = s * s;
    // q = 1/3 + z/5 + … + z¹¹/25 (troncature < 2⁻⁶⁰)
    let mut q = 1.0 / 25.0;
    q = q * z + 1.0 / 23.0;
    q = q * z + 1.0 / 21.0;
    q = q * z + 1.0 / 19.0;
    q = q * z + 1.0 / 17.0;
    q = q * z + 1.0 / 15.0;
    q = q * z + 1.0 / 13.0;
    q = q * z + 1.0 / 11.0;
    q = q * z + 1.0 / 9.0;
    q = q * z + 1.0 / 7.0;
    q = q * z + 1.0 / 5.0;
    q = q * z + 1.0 / 3.0;
    let two_s = s + s; // exact
    let ln_m = two_s * (z * q) + two_s;
    let ef = e as f64; // exact (|e| ≤ 1075)
    (ef * LN2_HI + (ln_m + ef * LN2_LO)) as f32
}

/// Softmax portable d'une ligne : soustraction du max (stabilité), [`exp_f32`]
/// élément par élément, normalisation par une somme **indépendante de
/// l'ordre** ([`reproducible_sum`]), division IEEE. Toute la chaîne est
/// bit-exacte inter-plates-formes ; permuter l'entrée permute la sortie
/// bit à bit. Une entrée NaN ne contamine que sa propre composante
/// (le max IEEE ignore les NaN). `±0` dans le max est sans effet :
/// `exp_f32(x − 0) == exp_f32(x − (−0))` bit à bit.
pub fn softmax_f32(xs: &[f32]) -> Vec<f32> {
    let m = xs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let e: Vec<f32> = xs.iter().map(|&x| exp_f32(x - m)).collect();
    let s = reproducible_sum(&e);
    e.iter().map(|&v| v / s).collect()
}

/// Produit scalaire portable : produits **exacts** en f64 (24 × 24 ≤ 53 bits),
/// accumulation séquentielle f64 en ordre fixe, cast final. Bit-exact
/// inter-plates-formes par construction ; plus précis que l'accumulation f32
/// (l'erreur ne vient que des additions f64, ≈ n·2⁻⁵³ relatif).
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_f32: length mismatch");
    let mut acc = 0.0f64;
    for i in 0..a.len()
    {
        acc += a[i] as f64 * b[i] as f64;
    }
    acc as f32
}

/// GEMM portable `C = A·B` (row-major, A : m×k, B : k×n) : chaque coefficient
/// est un [`dot_f32`] en ordre fixe. Voie de référence bit-exacte
/// inter-plates-formes (naïve, mono-thread — cf. doc du module pour les
/// alternatives rapides).
pub fn gemm_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "gemm_f32: A doit être m×k");
    assert_eq!(b.len(), k * n, "gemm_f32: B doit être k×n");
    let mut c = vec![0.0f32; m * n];
    for i in 0..m
    {
        for j in 0..n
        {
            let mut acc = 0.0f64;
            for l in 0..k
            {
                acc += a[i * k + l] as f64 * b[l * n + j] as f64;
            }
            c[i * n + j] = acc as f32;
        }
    }
    c
}

// ================================================================== //
//  Contrat de preuve cross-platform                                   //
// ================================================================== //
// Constantes et générateurs partagés par les tests unitaires et par le
// binaire de preuve `src/bin/proof_portable_f32.rs`, à exécuter sur chaque
// plate-forme cible (x86-64 Debian, Jetson/aarch64, …) : toute plate-forme
// conforme doit reproduire exactement ces bits. Les empreintes ont été
// calculées une fois sur x86-64 et sont commises comme contrat.

/// Pas du balayage-contrat de l'espace des bits f32 (≈ 65 536 entrées).
pub const PROOF_STEP_CONTRACT: u64 = 65_537;
/// Pas du balayage dense (≈ 16,7 M d'entrées, ≈ 1 s en release).
pub const PROOF_STEP_DENSE: u64 = 257;

/// Empreinte attendue de `exp_f32` sur le balayage-contrat.
pub const PROOF_EXP_FP_CONTRACT: u64 = 0x71e6_3f5e_1688_a7f1;
/// Empreinte attendue de `ln_f32` sur le balayage-contrat.
pub const PROOF_LN_FP_CONTRACT: u64 = 0x8892_b8ba_72ff_b8b6;
/// Empreinte attendue de `exp_f32` sur le balayage dense.
pub const PROOF_EXP_FP_DENSE: u64 = 0x6495_da04_866c_1c4b;
/// Empreinte attendue de `ln_f32` sur le balayage dense.
pub const PROOF_LN_FP_DENSE: u64 = 0x19e7_fd49_7cff_d94b;
/// Empreinte attendue de `exp_f32` sur le balayage **exhaustif** (pas 1 :
/// les 2³² entrées possibles — `--full` du binaire de preuve).
pub const PROOF_EXP_FP_EXHAUSTIVE: u64 = 0xda65_ffaf_8fe9_f4f4;
/// Empreinte attendue de `ln_f32` sur le balayage **exhaustif**.
pub const PROOF_LN_FP_EXHAUSTIVE: u64 = 0xb9ad_67e0_8ae8_f0fa;

/// Entrées des goldens ponctuels de `exp_f32`.
pub const PROOF_EXP_GOLDEN_INPUTS: [f32; 10] =
    [0.5, 1.0, -1.0, 2.0, 10.0, -10.0, 88.0, -87.0, 1e-8, -103.9];
/// Bits attendus de `exp_f32` sur [`PROOF_EXP_GOLDEN_INPUTS`].
pub const PROOF_EXP_GOLDEN_BITS: [u32; 10] = [
    1070795084, 1076754516, 1052531378, 1089237798, 1185682670, 943614926, 2130215607, 11744903,
    1065353216, 1,
];
/// Entrées des goldens ponctuels de `ln_f32`.
pub const PROOF_LN_GOLDEN_INPUTS: [f32; 10] = [
    0.5,
    1.5,
    2.0,
    10.0,
    0.1,
    1e30,
    1e-30,
    f32::MIN_POSITIVE,
    3.4e38,
    1.0000001,
];
/// Bits attendus de `ln_f32` sur [`PROOF_LN_GOLDEN_INPUTS`].
pub const PROOF_LN_GOLDEN_BITS: [u32; 10] = [
    3207688728, 1053792543, 1060205080, 1075010958, 3222494606, 1116350389, 3263834037, 3266227280,
    1118925227, 872415231,
];

/// Empreinte attendue du softmax-contrat (PCG(7), n = 64, plage [−10, 10)).
pub const PROOF_SOFTMAX_FP: u64 = 0x2b0c_3ead_12aa_19d5;
/// Empreinte attendue du GEMM-contrat (PCG(1113), 17×13 · 13×11, [−2, 2)).
pub const PROOF_GEMM_FP: u64 = 0x53df_bea9_109b_bd20;

/// État initial FNV-1a 64 bits (même discipline que scirust-runtime).
pub const fn fnv1a_init() -> u64 {
    0xcbf2_9ce4_8422_2325
}

/// Replie les bits d'un f32 dans une empreinte FNV-1a 64 bits.
pub const fn fnv1a_fold_bits(mut fp: u64, bits: u32) -> u64 {
    fp ^= bits as u64;
    fp.wrapping_mul(0x0000_0100_0000_01b3)
}

/// Empreinte FNV-1a de `f` sur le balayage de l'espace des bits f32 par pas
/// `step` (NaN et infinis inclus — les sorties sont canonicalisées).
pub fn sweep_fingerprint(f: fn(f32) -> f32, step: u64) -> u64 {
    assert!(step > 0, "sweep_fingerprint: pas nul");
    let mut fp = fnv1a_init();
    let mut i = 0u64;
    while i <= u32::MAX as u64
    {
        fp = fnv1a_fold_bits(fp, f(f32::from_bits(i as u32)).to_bits());
        i += step;
    }
    fp
}

/// Recalcule l'empreinte du softmax-contrat ([`PROOF_SOFTMAX_FP`]).
pub fn proof_softmax_fingerprint() -> u64 {
    let mut rng = crate::nn::PcgEngine::new(7);
    let xs: Vec<f32> = (0..64).map(|_| rng.float() * 20.0 - 10.0).collect();
    softmax_f32(&xs)
        .iter()
        .fold(fnv1a_init(), |fp, v| fnv1a_fold_bits(fp, v.to_bits()))
}

/// Recalcule l'empreinte du GEMM-contrat ([`PROOF_GEMM_FP`]).
pub fn proof_gemm_fingerprint() -> u64 {
    let mut rng = crate::nn::PcgEngine::new(1113);
    let a: Vec<f32> = (0..17 * 13).map(|_| rng.float() * 4.0 - 2.0).collect();
    let b: Vec<f32> = (0..13 * 11).map(|_| rng.float() * 4.0 - 2.0).collect();
    gemm_f32(&a, &b, 17, 13, 11)
        .iter()
        .fold(fnv1a_init(), |fp, v| fnv1a_fold_bits(fp, v.to_bits()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::PcgEngine;

    /// Clé monotone sur les f32 non-NaN (ordre total) pour compter les ulps.
    fn ord_key(x: f32) -> i64 {
        let b = x.to_bits();
        if b >> 31 == 1
        {
            -((b & 0x7fff_ffff) as i64)
        }
        else
        {
            b as i64
        }
    }

    fn ulp_diff(a: f32, b: f32) -> i64 {
        (ord_key(a) - ord_key(b)).abs()
    }

    #[test]
    fn exp_specials() {
        assert_eq!(exp_f32(f32::NAN).to_bits(), 0x7fc0_0000);
        assert_eq!(exp_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp_f32(f32::NEG_INFINITY), 0.0);
        assert_eq!(exp_f32(0.0), 1.0);
        assert_eq!(exp_f32(-0.0), 1.0);
        assert_eq!(exp_f32(100.0), f32::INFINITY); // e^100 > f32::MAX
        assert_eq!(exp_f32(-120.0), 0.0); // sous le plus petit sous-normal
        // Sortie sous-normale produite par le cast
        let sub = exp_f32(-100.0); // e^-100 ≈ 3,72e-44
        assert!(sub > 0.0 && sub < f32::MIN_POSITIVE, "e^-100 = {sub}");
    }

    #[test]
    fn ln_specials() {
        assert_eq!(ln_f32(f32::NAN).to_bits(), 0x7fc0_0000);
        assert_eq!(ln_f32(-1.0).to_bits(), 0x7fc0_0000);
        assert_eq!(ln_f32(0.0), f32::NEG_INFINITY);
        assert_eq!(ln_f32(-0.0), f32::NEG_INFINITY);
        assert_eq!(ln_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(ln_f32(1.0), 0.0);
        // Entrée sous-normale : ln(1,4e-45) ≈ -103,28
        let l = ln_f32(f32::from_bits(1)); // plus petit sous-normal
        assert!((l + 103.28).abs() < 0.01, "ln(min sous-normal) = {l}");
    }

    /// Contrat de portabilité : bits exacts attendus sur QUELQUES entrées
    /// remarquables. Toute plate-forme qui calcule autrement échoue ici.
    #[test]
    fn exp_golden_bits() {
        let got: Vec<u32> = PROOF_EXP_GOLDEN_INPUTS
            .iter()
            .map(|&x| exp_f32(x).to_bits())
            .collect();
        assert_eq!(got, PROOF_EXP_GOLDEN_BITS.to_vec());
    }

    #[test]
    fn ln_golden_bits() {
        let got: Vec<u32> = PROOF_LN_GOLDEN_INPUTS
            .iter()
            .map(|&x| ln_f32(x).to_bits())
            .collect();
        assert_eq!(got, PROOF_LN_GOLDEN_BITS.to_vec());
    }

    /// Empreinte FNV sur un balayage systématique de TOUT l'espace des bits
    /// f32 (pas 65 537) — NaN et infinis inclus (sorties canonicalisées).
    /// C'est l'empreinte à comparer entre x86-64 et ARM (le binaire
    /// `proof_portable_f32` rejoue ce contrat sur chaque machine).
    #[test]
    fn exp_fingerprint_bit_sweep() {
        let fp = sweep_fingerprint(exp_f32, PROOF_STEP_CONTRACT);
        assert_eq!(fp, PROOF_EXP_FP_CONTRACT, "empreinte exp : 0x{fp:016x}");
    }

    #[test]
    fn ln_fingerprint_bit_sweep() {
        let fp = sweep_fingerprint(ln_f32, PROOF_STEP_CONTRACT);
        assert_eq!(fp, PROOF_LN_FP_CONTRACT, "empreinte ln : 0x{fp:016x}");
    }

    /// Oracle de précision (plate-forme de dev) : ≤ 1 ulp de la référence
    /// libm f64 sur un échantillon dense — fidèlement arrondi.
    #[test]
    fn exp_faithful_vs_f64_oracle() {
        let mut rng = PcgEngine::new(2026);
        let mut max_ulp = 0i64;
        for _ in 0..100_000
        {
            let bits = (rng.float() as f64 * 4_294_967_296.0) as u32;
            let x = f32::from_bits(bits);
            if !x.is_finite()
            {
                continue;
            }
            let reference = ((x as f64).exp()) as f32;
            let got = exp_f32(x);
            if reference.is_nan() || got.is_nan()
            {
                assert_eq!(reference.is_nan(), got.is_nan());
                continue;
            }
            let d = ulp_diff(got, reference);
            max_ulp = max_ulp.max(d);
            assert!(d <= 1, "exp_f32({x}) = {got}, libm = {reference}, {d} ulp");
        }
        // Le résultat coïncide presque partout exactement avec l'oracle.
        assert!(max_ulp <= 1);
    }

    #[test]
    fn ln_faithful_vs_f64_oracle() {
        let mut rng = PcgEngine::new(4052);
        for _ in 0..100_000
        {
            let bits = (rng.float() as f64 * 2_139_095_040.0) as u32; // (0, +inf) exclus
            let x = f32::from_bits(bits);
            if x <= 0.0 || !x.is_finite()
            {
                continue;
            }
            let reference = ((x as f64).ln()) as f32;
            let got = ln_f32(x);
            let d = ulp_diff(got, reference);
            assert!(d <= 1, "ln_f32({x}) = {got}, libm = {reference}, {d} ulp");
        }
    }

    /// ln(exp(x)) ≈ x sur la plage sûre (cohérence interne des deux voies).
    #[test]
    fn exp_ln_roundtrip() {
        for i in 0..1000
        {
            let x = -80.0 + i as f32 * 0.16; // [-80, 80)
            let y = ln_f32(exp_f32(x));
            assert!(
                (y - x).abs() <= 1e-5 * x.abs().max(1.0),
                "roundtrip({x}) = {y}"
            );
        }
    }

    #[test]
    fn softmax_portable_properties() {
        let mut rng = PcgEngine::new(7);
        let xs: Vec<f32> = (0..64).map(|_| rng.float() * 20.0 - 10.0).collect();
        let q = softmax_f32(&xs);
        // Normalisation
        let s: f64 = q.iter().map(|&v| v as f64).sum();
        assert!((s - 1.0).abs() < 1e-5, "somme = {s}");
        // Équivariance bit à bit sous permutation (renversement)
        let rev: Vec<f32> = xs.iter().rev().copied().collect();
        let q_rev = softmax_f32(&rev);
        for i in 0..xs.len()
        {
            assert_eq!(
                q[i].to_bits(),
                q_rev[xs.len() - 1 - i].to_bits(),
                "softmax non équivariant bit à bit en {i}"
            );
        }
        // Empreinte (contrat de portabilité)
        let fp = proof_softmax_fingerprint();
        assert_eq!(fp, PROOF_SOFTMAX_FP, "empreinte softmax : 0x{fp:016x}");
    }

    #[test]
    fn dot_and_gemm_exact_on_small_integers() {
        // Produits et sommes de petits entiers : exacts en f64 ⇒ résultat exact.
        let a: Vec<f32> = (1..=64).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=64).map(|i| (65 - i) as f32).collect();
        let expected: i64 = (1..=64i64).map(|i| i * (65 - i)).sum();
        assert_eq!(dot_f32(&a, &b), expected as f32);

        let c = gemm_f32(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn dot_close_to_correctly_rounded_reference() {
        let mut rng = PcgEngine::new(31);
        let a: Vec<f32> = (0..256).map(|_| rng.float() * 2.0 - 1.0).collect();
        let b: Vec<f32> = (0..256).map(|_| rng.float() * 2.0 - 1.0).collect();
        let reference = crate::reproducible::reproducible_dot(&a, &b);
        let got = dot_f32(&a, &b);
        assert!(
            ulp_diff(got, reference) <= 1,
            "dot_f32 = {got}, référence correctement arrondie = {reference}"
        );
    }

    #[test]
    fn gemm_fingerprint() {
        let fp = proof_gemm_fingerprint();
        assert_eq!(fp, PROOF_GEMM_FP, "empreinte gemm : 0x{fp:016x}");
    }
}
