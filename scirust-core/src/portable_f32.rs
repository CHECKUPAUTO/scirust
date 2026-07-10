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
    exp_f64_core(x as f64) as f32
}

/// Cœur f64 de l'exponentielle portable (réduction k·ln 2 + Taylor degré 13 +
/// remise à l'échelle 2^k). Précondition : `y` fini, |y| ≤ 300 (l'exposant k
/// reste dans la plage normale f64) — garanti par les gardes des appelants
/// ([`exp_f32`], [`sigmoid_f32`], [`tanh_f32`]).
fn exp_f64_core(y: f64) -> f64 {
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
    // 2^k exact par construction de l'exposant (|k| ≤ 434 ⇒ f64 normal)
    let scale = f64::from_bits(((1023 + k as i64) as u64) << 52);
    p * scale
}

/// `sigmoid(x) = 1/(1+e⁻ˣ)` portable : bit-exact inter-plates-formes par
/// construction, fidèlement arrondi (cf. doc du module). Forme stable des
/// deux côtés (pas de cancellation) ; NaN → NaN canonique ;
/// saturations : `x ≥ 30` → `1`, `x ≤ −120` → `0` (les sorties sous-normales
/// intermédiaires sont produites par le cast).
pub fn sigmoid_f32(x: f32) -> f32 {
    if x.is_nan()
    {
        return CANONICAL_NAN;
    }
    if x >= 30.0
    {
        return 1.0; // 1 − e⁻³⁰ ≈ 1 − 9,4e-14 arrondit à 1 en f32
    }
    if x <= -120.0
    {
        return 0.0; // e⁻¹²⁰ est sous la moitié du plus petit sous-normal
    }
    let y = x as f64;
    if y >= 0.0
    {
        (1.0 / (1.0 + exp_f64_core(-y))) as f32
    }
    else
    {
        let t = exp_f64_core(y);
        (t / (1.0 + t)) as f32
    }
}

/// `tanh(x)` portable : bit-exact inter-plates-formes par construction,
/// fidèlement arrondi (cf. doc du module). `tanh(±0) = ±0` ; NaN → NaN
/// canonique ; saturation `|x| ≥ 10` → `±1` (1 − tanh(10) ≈ 4e-9 arrondit
/// à 1 en f32) ; pour `|x| < 1e-4`, `tanh(x)` arrondit à `x` (le terme x³/3
/// est sous le demi-ulp).
pub fn tanh_f32(x: f32) -> f32 {
    if x.is_nan()
    {
        return CANONICAL_NAN;
    }
    let ax = x.abs();
    if ax >= 10.0
    {
        return 1.0f32.copysign(x);
    }
    if ax < 1e-4
    {
        return x; // préserve aussi le signe de ±0
    }
    let t = exp_f64_core(-2.0 * (ax as f64));
    let r = ((1.0 - t) / (1.0 + t)) as f32;
    if x > 0.0 { r } else { -r }
}

/// Bits fractionnaires de 2/π (448 bits, mots MSB-first) pour la réduction
/// d'argument de Payne & Hanek. Constante mathématique **générée par nos
/// soins** (π par Chudnovsky en précision arbitraire, cf. volet 113) —
/// aucune table copiée d'une implémentation existante.
const TWO_OVER_PI_BITS: [u64; 7] = [
    0xa2f9_836e_4e44_1529,
    0xfc27_57d1_f534_ddc0,
    0xdb62_9599_3c43_9041,
    0xfe51_63ab_debb_c561,
    0xb724_6e3a_424d_d2e0,
    0x0649_2eea_09d1_921c,
    0xfe1d_eb1c_b129_a73e,
];

/// 2⁻¹²⁸ (échelle exacte de la fraction signée de la réduction).
const SCALE_2_M128: f64 = f64::from_bits(((1023 - 128) as u64) << 52);

/// Réduction d'argument de **Payne & Hanek** en arithmétique entière pure
/// (portable par construction) : pour `|x| > π/4` fini, renvoie `(k mod 4, r)`
/// tels que `|x| = k·(π/2) + r`, `|r| ≤ π/4 (+ε)`. Le produit
/// mantisse × (448 bits de 2/π) est calculé exactement en u128 ; le quadrant
/// (2 bits) et 128 bits de fraction signée sont extraits, puis
/// `r = fraction·(π/2)` avec une erreur **relative** ~2⁻⁵² (la conversion
/// i128 → f64 est correctement arrondie), ce qui reste fidèle même aux pires
/// cas de réduction du format f32 (|r| ≳ 2⁻³²).
fn payne_hanek_reduce(x: f32) -> (u32, f64) {
    let bits = x.to_bits();
    // |x| > π/4 ⇒ nombre normal : mantisse avec bit implicite, |x| = m·2^e
    let m = ((bits & 0x007f_ffff) | 0x0080_0000) as u128;
    let e = (((bits >> 23) & 0xff) as i32) - 150;

    // P = m × B où B = ⌊2/π·2⁴⁴⁸⌋ : 512 bits en 8 mots LSB-first.
    // Le bit j de P (LSB = 0) pèse 2^(j+e−448) dans |x|·(2/π).
    let mut p = [0u64; 8];
    let mut carry: u128 = 0;
    for i in 0..7
    {
        let t = m * (TWO_OVER_PI_BITS[6 - i] as u128) + carry;
        p[i] = t as u64;
        carry = t >> 64;
    }
    p[7] = carry as u64;

    // Extrait le u64 dont le bit 63 correspond au bit `top_j` de P
    // (bits hors de P = 0).
    let extract64 = |top_j: i32| -> u64 {
        let word = top_j.div_euclid(64);
        let off = top_j.rem_euclid(64);
        let hi = if (0..8).contains(&word)
        {
            p[word as usize]
        }
        else
        {
            0
        };
        let lo = if (1..9).contains(&word)
        {
            p[(word - 1) as usize]
        }
        else
        {
            0
        };
        if off == 63
        {
            hi
        }
        else
        {
            (hi << (63 - off)) | (lo >> (off + 1))
        }
    };
    let bit = |j: i32| -> u32 {
        if (0..512).contains(&j)
        {
            ((p[(j / 64) as usize] >> (j % 64)) & 1) as u32
        }
        else
        {
            0
        }
    };

    // Quadrant : bits de poids 2¹ et 2⁰ ; fraction : 128 bits en dessous.
    let jq = 449 - e;
    let mut q = (bit(jq) << 1) | bit(jq - 1);
    let frac_hi = extract64(jq - 2);
    let frac_lo = extract64(jq - 66);
    let fs = (((frac_hi as u128) << 64) | frac_lo as u128) as i128;
    if fs < 0
    {
        // fraction ≥ 1/2 : on arrondit au quadrant supérieur ; la fraction
        // signée (complément à deux) devient exactement l'écart négatif.
        q = (q + 1) & 3;
    }
    let ff = (fs as f64) * SCALE_2_M128;
    (q & 3, ff * core::f64::consts::FRAC_PI_2)
}

/// sin(r) sur |r| ≤ π/4 : Taylor impair jusqu'à r¹⁵ (troncature < 2⁻⁵³ rel.),
/// écrit `r + r·(z·s)` pour préserver la précision relative de r.
fn sin_poly(r: f64) -> f64 {
    let z = r * r;
    let mut s = -1.0 / 1_307_674_368_000.0; // −1/15!
    s = s * z + 1.0 / 6_227_020_800.0; // +1/13!
    s = s * z - 1.0 / 39_916_800.0; // −1/11!
    s = s * z + 1.0 / 362_880.0; // +1/9!
    s = s * z - 1.0 / 5_040.0; // −1/7!
    s = s * z + 1.0 / 120.0; // +1/5!
    s = s * z - 1.0 / 6.0; // −1/3!
    r + r * (z * s)
}

/// cos(r) sur |r| ≤ π/4 : Taylor pair jusqu'à r¹⁶ (troncature < 2⁻⁵⁷ rel.).
fn cos_poly(r: f64) -> f64 {
    let z = r * r;
    let mut c = 1.0 / 20_922_789_888_000.0; // +1/16!
    c = c * z - 1.0 / 87_178_291_200.0; // −1/14!
    c = c * z + 1.0 / 479_001_600.0; // +1/12!
    c = c * z - 1.0 / 3_628_800.0; // −1/10!
    c = c * z + 1.0 / 40_320.0; // +1/8!
    c = c * z - 1.0 / 720.0; // −1/6!
    c = c * z + 1.0 / 24.0; // +1/4!
    c = c * z - 0.5; // −1/2!
    1.0 + z * c
}

/// `sin(x)` portable : bit-exact inter-plates-formes par construction,
/// fidèlement arrondi (cf. doc du module). Réduction de Payne–Hanek en
/// arithmétique entière (exacte pour TOUT f32 fini, y compris ~10³⁸) ;
/// `sin(±0) = ±0` ; NaN et ±∞ → NaN canonique ; pour `|x| < 1e-4`,
/// `sin(x)` arrondit à `x`.
pub fn sin_f32(x: f32) -> f32 {
    if !x.is_finite()
    {
        return CANONICAL_NAN;
    }
    let ax = x.abs();
    if ax < 1e-4
    {
        return x; // préserve aussi ±0
    }
    if (ax as f64) <= core::f64::consts::FRAC_PI_4
    {
        return sin_poly(x as f64) as f32;
    }
    let (q, r) = payne_hanek_reduce(x);
    let s = match q
    {
        0 => sin_poly(r),
        1 => cos_poly(r),
        2 => -sin_poly(r),
        _ => -cos_poly(r),
    };
    (if x < 0.0 { -s } else { s }) as f32
}

/// `cos(x)` portable : bit-exact inter-plates-formes par construction,
/// fidèlement arrondi (cf. doc du module). Même réduction de Payne–Hanek ;
/// NaN et ±∞ → NaN canonique ; pour `|x| < 1e-4`, `cos(x)` arrondit à `1`.
pub fn cos_f32(x: f32) -> f32 {
    if !x.is_finite()
    {
        return CANONICAL_NAN;
    }
    let ax = x.abs();
    if ax < 1e-4
    {
        return 1.0; // 1 − x²/2 arrondit à 1 (x²/2 < 5e-9 ≪ demi-ulp)
    }
    if (ax as f64) <= core::f64::consts::FRAC_PI_4
    {
        return cos_poly(ax as f64) as f32;
    }
    let (q, r) = payne_hanek_reduce(x);
    (match q
    {
        0 => cos_poly(r),
        1 => -sin_poly(r),
        2 => -cos_poly(r),
        _ => sin_poly(r),
    }) as f32
}

/// π/2 tronqué à ~32 bits de mantisse : k·PIO2_HI est exact pour |k| ≤ 2²⁰.
const PIO2_HI: f64 = f64::from_bits(core::f64::consts::FRAC_PI_2.to_bits() & 0xFFFF_FFFF_F000_0000);
/// Reste π/2 − PIO2_HI (différence exacte en f64).
const PIO2_LO: f64 = core::f64::consts::FRAC_PI_2 - PIO2_HI;

/// (sin y, cos y) **en f64**, portable, pour |y| ≤ 100 : réduction de
/// Cody & Waite (π/2 scindé hi/lo, k·PIO2_HI exact) + les polynômes de
/// Taylor de la voie portable. Erreur **absolue** ≤ ~2⁻⁵² — exactement ce
/// qu'il faut pour des twiddle factors de FFT (|T| = 1) ou des angles
/// bornés (RoPE) ; pour l'argument f32 arbitraire, utiliser
/// [`sin_f32`]/[`cos_f32`] (réduction de Payne–Hanek).
pub fn sincos_small_f64(y: f64) -> (f64, f64) {
    assert!(
        y.is_finite() && y.abs() <= 100.0,
        "sincos_small_f64: |y| ≤ 100 requis (reçu {y})"
    );
    let k = (y * core::f64::consts::FRAC_2_PI).round();
    let r = (y - k * PIO2_HI) - k * PIO2_LO;
    let (s, c) = (sin_poly(r), cos_poly(r));
    match (k as i64).rem_euclid(4)
    {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// `erf(x)` portable : bit-exact inter-plates-formes par construction,
/// fidèlement arrondi. Série de Maclaurin en f64
/// (erf(x) = 2/√π · Σ (−1)ⁿ x²ⁿ⁺¹/(n!(2n+1))) avec arrêt **relatif**
/// déterministe (comparaisons IEEE identiques partout) et plafond
/// d'itérations ; `erf(±0) = ±0`, fonction impaire bit-exacte ; NaN et
/// ±∞ → NaN canonique / ±1 ; saturation `|x| ≥ 4` → `±1`
/// (erfc(4) ≈ 1,5e-8 arrondit à 1 en f32). La cancellation alternée
/// culmine à ~2¹⁷ vers x = 4 : l'erreur f64 résiduelle (~2⁻³⁶) reste très
/// au-dessous du demi-ulp f32.
pub fn erf_f32(x: f32) -> f32 {
    if x.is_nan()
    {
        return CANONICAL_NAN;
    }
    let ax = x.abs();
    if ax >= 4.0
    {
        return 1.0f32.copysign(x); // couvre aussi ±∞
    }
    if ax < 1e-4
    {
        // erf(x) ≈ (2/√π)·x (terme suivant < 2⁻²⁸ relatif) ; préserve ±0
        // (la série perdrait le signe du zéro : (−0) + (+0) = +0 en IEEE).
        return ((x as f64) * core::f64::consts::FRAC_2_SQRT_PI) as f32;
    }
    (erf_f64_core(x as f64) * core::f64::consts::FRAC_2_SQRT_PI) as f32
}

/// Cœur f64 de la série de Maclaurin d'erf (sans le facteur 2/√π).
/// Précondition : |y| < ~4,1 (garanti par les gardes des appelants).
fn erf_f64_core(y: f64) -> f64 {
    let z = -y * y;
    let mut term = y; // x^(2n+1)·(−1)ⁿ/n! par récurrence
    let mut sum = y;
    let mut n = 1.0f64;
    while n < 80.0
    {
        term = term * z / n;
        let contrib = term / (2.0 * n + 1.0);
        sum += contrib;
        if contrib.abs() < sum.abs() * 1e-18
        {
            break;
        }
        n += 1.0;
    }
    sum
}

/// GELU **exact** portable : `x/2 · (1 + erf(x/√2))` — l'activation standard
/// des transformers, composée d'opérations IEEE de base et du cœur d'erf en
/// f64 (aucun cast intermédiaire), donc bit-exacte inter-plates-formes et
/// fidèlement arrondie (ni RepDL ni la voie libm de scirust-special ne
/// l'offrent sous garantie de portabilité).
pub fn gelu_f32(x: f32) -> f32 {
    if x.is_nan()
    {
        return CANONICAL_NAN;
    }
    if x == f32::NEG_INFINITY
    {
        return -0.0; // x·Φ(x) → 0⁻ (évite −∞·0 = NaN)
    }
    if x == 0.0
    {
        return x; // ±0 (et évite le tour complet de la série à zéro)
    }
    let y = x as f64;
    let u = y * core::f64::consts::FRAC_1_SQRT_2;
    let e = if u.abs() >= 4.0
    {
        1.0f64.copysign(u) // saturation d'erf (couvre aussi ±∞)
    }
    else
    {
        erf_f64_core(u) * core::f64::consts::FRAC_2_SQRT_PI
    };
    (0.5 * y * (1.0 + e)) as f32
}

/// Certification d'**arrondi correct** (premier pas concret vers la
/// résolution du dilemme du fabricant de tables, cartographie volet 111).
///
/// Pour une entrée `x` du chemin analytique d'une fonction, on connaît la
/// valeur interne f64 `y` et une borne d'erreur relative `b` (analyse de
/// l'implémentation). Si l'intervalle `[y·(1−b), y·(1+b)]` — qui contient la
/// valeur exacte — tombe strictement entre les deux frontières d'arrondi f32
/// qui encadrent `y as f32`, alors le résultat publié est **prouvé
/// correctement arrondi** pour cette entrée. Les entrées des chemins de garde
/// (saturations, raccourcis petit-argument) sont correctes par analyse
/// directe (« analytic »). Les entrées restantes (« uncertified ») ne sont
/// pas fausses — leur statut se tranche hors ligne en précision arbitraire.
///
/// L'évaluateur interne est REVALIDÉ contre la fonction publiée sur chaque
/// entrée (`assert`) : le certificat porte bien sur le code expédié.
pub mod certify {
    use super::*;

    /// Résultat d'un balayage de certification.
    #[derive(Debug, Clone, Default)]
    pub struct Report {
        /// Entrées correctes par analyse du chemin de garde.
        pub analytic: u64,
        /// Entrées prouvées correctement arrondies par le certificat.
        pub certified: u64,
        /// Entrées non certifiées (statut à trancher hors ligne).
        pub uncertified: u64,
        /// Bits de TOUTES les entrées non certifiées (pour la vérification
        /// hors ligne en précision arbitraire).
        pub samples: Vec<u32>,
    }

    /// (valeur interne f64, borne d'erreur relative) ou None = chemin de
    /// garde correct par analyse.
    type Eval = fn(f32) -> Option<(f64, f64)>;

    fn eval_exp(x: f32) -> Option<(f64, f64)> {
        if x.is_nan() || x >= 89.0 || x <= -105.0
        {
            return None;
        }
        Some((exp_f64_core(x as f64), 2f64.powi(-46)))
    }

    fn eval_ln(x: f32) -> Option<(f64, f64)> {
        if x.is_nan() || x <= 0.0 || x == f32::INFINITY
        {
            return None;
        }
        Some((ln_f64_core(x as f64), 2f64.powi(-46)))
    }

    fn eval_tanh(x: f32) -> Option<(f64, f64)> {
        let ax = x.abs();
        if x.is_nan() || ax >= 10.0 || ax < 1e-4
        {
            return None;
        }
        let t = exp_f64_core(-2.0 * (ax as f64));
        let r = (1.0 - t) / (1.0 + t);
        Some((if x > 0.0 { r } else { -r }, 2f64.powi(-45)))
    }

    fn eval_sigmoid(x: f32) -> Option<(f64, f64)> {
        if x.is_nan() || x >= 30.0 || x <= -120.0
        {
            return None;
        }
        let y = x as f64;
        let v = if y >= 0.0
        {
            1.0 / (1.0 + exp_f64_core(-y))
        }
        else
        {
            let t = exp_f64_core(y);
            t / (1.0 + t)
        };
        Some((v, 2f64.powi(-46)))
    }

    fn eval_sin(x: f32) -> Option<(f64, f64)> {
        if !x.is_finite() || x.abs() < 1e-4
        {
            return None;
        }
        let ax = x.abs();
        let v = if (ax as f64) <= core::f64::consts::FRAC_PI_4
        {
            sin_poly(x as f64)
        }
        else
        {
            let (q, r) = payne_hanek_reduce(x);
            let s = match q
            {
                0 => sin_poly(r),
                1 => cos_poly(r),
                2 => -sin_poly(r),
                _ => -cos_poly(r),
            };
            if x < 0.0 { -s } else { s }
        };
        Some((v, 2f64.powi(-44)))
    }

    fn eval_cos(x: f32) -> Option<(f64, f64)> {
        if !x.is_finite() || x.abs() < 1e-4
        {
            return None;
        }
        let ax = x.abs();
        let v = if (ax as f64) <= core::f64::consts::FRAC_PI_4
        {
            cos_poly(ax as f64)
        }
        else
        {
            let (q, r) = payne_hanek_reduce(x);
            match q
            {
                0 => cos_poly(r),
                1 => -sin_poly(r),
                2 => -cos_poly(r),
                _ => sin_poly(r),
            }
        };
        Some((v, 2f64.powi(-44)))
    }

    fn eval_erf(x: f32) -> Option<(f64, f64)> {
        let ax = x.abs();
        if x.is_nan() || ax >= 4.0 || ax < 1e-4
        {
            return None;
        }
        // Rejoue la série en suivant le pic de cancellation pour une borne
        // PAR ENTRÉE : erreur ≈ (max |terme| / |somme|) · itérations · 2⁻⁵³.
        let y = x as f64;
        let z = -y * y;
        let mut term = y;
        let mut sum = y;
        let mut max_c = y.abs();
        let mut n = 1.0f64;
        let mut iters = 1.0f64;
        while n < 80.0
        {
            term = term * z / n;
            let contrib = term / (2.0 * n + 1.0);
            sum += contrib;
            max_c = max_c.max(contrib.abs());
            if contrib.abs() < sum.abs() * 1e-18
            {
                break;
            }
            n += 1.0;
            iters += 1.0;
        }
        let v = sum * core::f64::consts::FRAC_2_SQRT_PI;
        let bound = (max_c / sum.abs()) * iters * 2f64.powi(-53) + 2f64.powi(-52);
        Some((v, bound))
    }

    /// Une entrée certifiable : (nom, fonction publiée, évaluateur).
    pub type Entry = (&'static str, fn(f32) -> f32, Eval);

    /// Les fonctions certifiables.
    pub const FUNCTIONS: [Entry; 7] = [
        ("exp", exp_f32, eval_exp),
        ("ln", ln_f32, eval_ln),
        ("tanh", tanh_f32, eval_tanh),
        ("sigmoid", sigmoid_f32, eval_sigmoid),
        ("sin", sin_f32, eval_sin),
        ("cos", cos_f32, eval_cos),
        ("erf", erf_f32, eval_erf),
    ];

    /// f32 suivant/précédent en direction de ±∞ (hors NaN), via les bits.
    fn next_up(x: f32) -> f32 {
        let b = x.to_bits();
        if x == f32::INFINITY
        {
            return x;
        }
        f32::from_bits(
            if x >= 0.0
            {
                if b == 0x8000_0000 { 1 } else { b + 1 }
            }
            else
            {
                b - 1
            },
        )
    }

    fn next_down(x: f32) -> f32 {
        -next_up(-x)
    }

    /// Balayage de certification de `f` par pas `step` sur tout l'espace des
    /// bits f32.
    pub fn sweep(public: fn(f32) -> f32, eval: Eval, step: u64) -> Report {
        let mut rep = Report::default();
        let mut i = 0u64;
        while i <= u32::MAX as u64
        {
            let x = f32::from_bits(i as u32);
            match eval(x)
            {
                None => rep.analytic += 1,
                Some((y, b)) =>
                {
                    let r = y as f32;
                    assert_eq!(
                        r.to_bits(),
                        public(x).to_bits(),
                        "certify: évaluateur ≠ fonction publiée en {x}"
                    );
                    // frontières d'arrondi encadrant r (milieux exacts en f64)
                    let lo_mid = (next_down(r) as f64 + r as f64) * 0.5;
                    let hi_mid = (r as f64 + next_up(r) as f64) * 0.5;
                    let eps = y.abs() * b;
                    if y - eps > lo_mid && y + eps < hi_mid
                    {
                        rep.certified += 1;
                    }
                    else
                    {
                        rep.uncertified += 1;
                        rep.samples.push(i as u32);
                    }
                },
            }
            i += step;
        }
        rep
    }
}

/// Cœur f64 du logarithme portable (cf. [`ln_f32`]). Précondition : `y` fini,
/// strictement positif.
fn ln_f64_core(y: f64) -> f64 {
    let bits = y.to_bits();
    let mut e = (((bits >> 52) & 0x7ff) as i64) - 1023;
    let mut m = f64::from_bits((bits & 0x000F_FFFF_FFFF_FFFF) | (1023u64 << 52));
    if m > core::f64::consts::SQRT_2
    {
        m *= 0.5;
        e += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let z = s * s;
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
    let two_s = s + s;
    let ln_m = two_s * (z * q) + two_s;
    let ef = e as f64;
    ef * LN2_HI + (ln_m + ef * LN2_LO)
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
    // promotion exacte (les sous-normaux f32 deviennent des f64 normaux)
    ln_f64_core(x as f64) as f32
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

/// Empreinte attendue de `tanh_f32` sur le balayage-contrat.
pub const PROOF_TANH_FP_CONTRACT: u64 = 0x418f_903e_1025_7c1e;
/// Empreinte attendue de `sigmoid_f32` sur le balayage-contrat.
pub const PROOF_SIGMOID_FP_CONTRACT: u64 = 0xea08_4f06_22bd_fec4;
/// Empreinte attendue de `tanh_f32` sur le balayage dense.
pub const PROOF_TANH_FP_DENSE: u64 = 0xa25d_e634_2fae_d6e8;
/// Empreinte attendue de `sigmoid_f32` sur le balayage dense.
pub const PROOF_SIGMOID_FP_DENSE: u64 = 0xb826_7671_7c58_1433;
/// Empreinte attendue de `tanh_f32` sur le balayage exhaustif.
pub const PROOF_TANH_FP_EXHAUSTIVE: u64 = 0xd6f9_e850_8d19_f785;
/// Empreinte attendue de `sigmoid_f32` sur le balayage exhaustif.
pub const PROOF_SIGMOID_FP_EXHAUSTIVE: u64 = 0x6796_eabe_dfe7_cb02;

/// Empreinte attendue de `sin_f32` sur le balayage-contrat.
pub const PROOF_SIN_FP_CONTRACT: u64 = 0x39c9_9b71_fdbc_e247;
/// Empreinte attendue de `cos_f32` sur le balayage-contrat.
pub const PROOF_COS_FP_CONTRACT: u64 = 0xcdc0_7dac_0d40_1d29;
/// Empreinte attendue de `sin_f32` sur le balayage dense.
pub const PROOF_SIN_FP_DENSE: u64 = 0x084d_235e_4d8d_dac7;
/// Empreinte attendue de `cos_f32` sur le balayage dense.
pub const PROOF_COS_FP_DENSE: u64 = 0xcde8_a193_db4b_2f5c;
/// Empreinte attendue de `sin_f32` sur le balayage exhaustif.
pub const PROOF_SIN_FP_EXHAUSTIVE: u64 = 0xc071_9c2d_610d_8685;
/// Empreinte attendue de `cos_f32` sur le balayage exhaustif.
pub const PROOF_COS_FP_EXHAUSTIVE: u64 = 0xb9b0_750e_e67e_5475;

/// Empreinte attendue de `erf_f32` sur le balayage-contrat.
pub const PROOF_ERF_FP_CONTRACT: u64 = 0xfe81_7b5a_5db4_0dc8;
/// Empreinte attendue de `erf_f32` sur le balayage dense.
pub const PROOF_ERF_FP_DENSE: u64 = 0xb7d5_4a90_6051_32c5;
/// Empreinte attendue de `erf_f32` sur le balayage exhaustif.
pub const PROOF_ERF_FP_EXHAUSTIVE: u64 = 0x3765_5614_b70c_f42d;
/// Empreinte attendue de `gelu_f32` sur le balayage-contrat.
pub const PROOF_GELU_FP_CONTRACT: u64 = 0x8f06_fb9e_b406_d63f;
/// Empreinte attendue de `gelu_f32` sur le balayage dense.
pub const PROOF_GELU_FP_DENSE: u64 = 0xf1a6_e6ae_9f03_349b;

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

    #[test]
    fn tanh_sigmoid_specials() {
        assert_eq!(tanh_f32(f32::NAN).to_bits(), 0x7fc0_0000);
        assert_eq!(sigmoid_f32(f32::NAN).to_bits(), 0x7fc0_0000);
        assert_eq!(tanh_f32(0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(tanh_f32(-0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(tanh_f32(f32::INFINITY), 1.0);
        assert_eq!(tanh_f32(f32::NEG_INFINITY), -1.0);
        assert_eq!(tanh_f32(15.0), 1.0);
        assert_eq!(tanh_f32(-15.0), -1.0);
        assert_eq!(sigmoid_f32(0.0), 0.5);
        assert_eq!(sigmoid_f32(f32::INFINITY), 1.0);
        assert_eq!(sigmoid_f32(f32::NEG_INFINITY), 0.0);
        assert_eq!(sigmoid_f32(40.0), 1.0);
        assert_eq!(sigmoid_f32(-130.0), 0.0);
        // symétries : tanh impaire, sigmoid(−x) = 1 − sigmoid(x) (à l'ulp près)
        for i in 1..200
        {
            let x = i as f32 * 0.05;
            assert_eq!(tanh_f32(-x).to_bits(), (-tanh_f32(x)).to_bits());
        }
    }

    /// tanh/sigmoid : ≤ 1 ulp de la référence libm f64 sur un échantillon
    /// dense — fidèlement arrondis.
    #[test]
    fn tanh_sigmoid_faithful_vs_f64_oracle() {
        let mut rng = PcgEngine::new(777);
        for _ in 0..100_000
        {
            let bits = (rng.float() as f64 * 4_294_967_296.0) as u32;
            let x = f32::from_bits(bits);
            if !x.is_finite()
            {
                continue;
            }
            let t_ref = ((x as f64).tanh()) as f32;
            let t_got = tanh_f32(x);
            assert!(
                ulp_diff(t_got, t_ref) <= 1,
                "tanh_f32({x}) = {t_got}, libm = {t_ref}"
            );
            let s_ref = (1.0 / (1.0 + (-(x as f64)).exp())) as f32;
            let s_got = sigmoid_f32(x);
            assert!(
                ulp_diff(s_got, s_ref) <= 1,
                "sigmoid_f32({x}) = {s_got}, libm = {s_ref}"
            );
        }
    }

    #[test]
    fn sin_cos_specials_and_symmetry() {
        assert_eq!(sin_f32(f32::NAN).to_bits(), 0x7fc0_0000);
        assert_eq!(cos_f32(f32::NAN).to_bits(), 0x7fc0_0000);
        assert_eq!(sin_f32(f32::INFINITY).to_bits(), 0x7fc0_0000);
        assert_eq!(cos_f32(f32::NEG_INFINITY).to_bits(), 0x7fc0_0000);
        assert_eq!(sin_f32(0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(sin_f32(-0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(cos_f32(0.0), 1.0);
        assert_eq!(cos_f32(-0.0), 1.0);
        // valeurs cardinales (à ≤ 1 ulp : π f32 n'est pas π)
        assert!((sin_f32(core::f32::consts::FRAC_PI_2) - 1.0).abs() < 1e-6);
        assert!((cos_f32(core::f32::consts::PI) + 1.0).abs() < 1e-6);
        // parité : sin impaire, cos paire — bit à bit
        for i in 1..400
        {
            let x = i as f32 * 0.7;
            assert_eq!(sin_f32(-x).to_bits(), (-sin_f32(x)).to_bits());
            assert_eq!(cos_f32(-x).to_bits(), cos_f32(x).to_bits());
        }
        // grands arguments : Payne–Hanek doit rester borné et fini
        for &x in &[1e10f32, 1e20, 1e30, 3.4e38, 12345678.0]
        {
            let s = sin_f32(x);
            let c = cos_f32(x);
            assert!(s.abs() <= 1.0 && c.abs() <= 1.0, "sin/cos({x}) hors [-1,1]");
            // identité pythagoricienne à la précision f32
            let id = (s as f64) * (s as f64) + (c as f64) * (c as f64);
            assert!((id - 1.0).abs() < 1e-6, "sin²+cos²({x}) = {id}");
        }
    }

    /// sin/cos : ≤ 1 ulp de la référence libm f64 sur un échantillon dense
    /// couvrant TOUTES les magnitudes f32 (y compris les très grands
    /// arguments, qui exercent la réduction de Payne–Hanek).
    #[test]
    fn sin_cos_faithful_vs_f64_oracle() {
        let mut rng = PcgEngine::new(31337);
        for _ in 0..100_000
        {
            let bits = (rng.float() as f64 * 4_294_967_296.0) as u32;
            let x = f32::from_bits(bits);
            if !x.is_finite()
            {
                continue;
            }
            let s_ref = ((x as f64).sin()) as f32;
            let s_got = sin_f32(x);
            assert!(
                ulp_diff(s_got, s_ref) <= 1,
                "sin_f32({x}) = {s_got}, libm = {s_ref}"
            );
            let c_ref = ((x as f64).cos()) as f32;
            let c_got = cos_f32(x);
            assert!(
                ulp_diff(c_got, c_ref) <= 1,
                "cos_f32({x}) = {c_got}, libm = {c_ref}"
            );
        }
    }

    /// erf : ≤ 1 ulp d'une table de référence calculée indépendamment en
    /// précision arbitraire (série de Maclaurin en Decimal, 60 chiffres,
    /// volet 113 — pas la libm). Bits attendus commis.
    #[test]
    fn erf_matches_independent_reference() {
        // (bits de l'entrée f32, bits attendus de erf) — table générée en
        // Decimal 60 chiffres (les entrées en bits évitent tout littéral
        // à précision excessive).
        let table: [(u32, u32); 16] = [
            (0x358637bd, 0x359772d0), // 1e-6
            (0x38d1b717, 0x38eca365), // 1e-4
            (0x3c23d70a, 0x3c38de13), // 0,01
            (0x3dcccccd, 0x3de652f5), // 0,1
            (0x3e800000, 0x3e8d7aa7), // 0,25
            (0x3f000000, 0x3f053f7b), // 0,5
            (0x3f400000, 0x3f360e4c), // 0,75
            (0x3f800000, 0x3f57bb3d), // 1
            (0x3fa00000, 0x3f6c432f), // 1,25
            (0x3fc00000, 0x3f7752ab), // 1,5
            (0x40000000, 0x3f7ecd71), // 2
            (0x40200000, 0x3f7fe554), // 2,5
            (0x40400000, 0x3f7ffe8d), // 3
            (0x40600000, 0x3f7ffff4), // 3,5
            (0x4079999a, 0x3f7fffff), // 3,9
            (0x407f5c29, 0x3f800000), // 3,99
        ];
        for &(x_bits, ref_bits) in &table
        {
            let x = f32::from_bits(x_bits);
            let got = erf_f32(x);
            let reference = f32::from_bits(ref_bits);
            assert!(
                ulp_diff(got, reference) <= 1,
                "erf_f32({x}) = {got} ({:#010x}), référence = {reference}",
                got.to_bits()
            );
        }
    }

    #[test]
    fn erf_gelu_specials_and_symmetry() {
        assert_eq!(erf_f32(f32::NAN).to_bits(), 0x7fc0_0000);
        assert_eq!(gelu_f32(f32::NAN).to_bits(), 0x7fc0_0000);
        assert_eq!(erf_f32(0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(erf_f32(-0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(erf_f32(f32::INFINITY), 1.0);
        assert_eq!(erf_f32(f32::NEG_INFINITY), -1.0);
        assert_eq!(erf_f32(5.0), 1.0);
        assert_eq!(erf_f32(-5.0), -1.0);
        // erf impaire, bit à bit
        for i in 1..400
        {
            let x = i as f32 * 0.011;
            assert_eq!(erf_f32(-x).to_bits(), (-erf_f32(x)).to_bits());
        }
        // GELU : 0 en 0, ≈ x pour x grand, ≈ 0⁻ pour x très négatif
        assert_eq!(gelu_f32(0.0), 0.0);
        assert_eq!(gelu_f32(10.0), 10.0);
        assert_eq!(gelu_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(gelu_f32(f32::NEG_INFINITY).to_bits(), (-0.0f32).to_bits());
        assert_eq!(gelu_f32(-40.0), -0.0);
        // valeur de référence : gelu(1) = 0,5·(1+erf(1/√2)) = 0,8413447…
        assert!((gelu_f32(1.0) - 0.841_344_7).abs() < 1e-6);
        // monotonie locale autour de 0 (propriété de GELU exact)
        assert!(gelu_f32(-0.5) < gelu_f32(0.0));
        assert!(gelu_f32(0.0) < gelu_f32(0.5));
    }

    #[test]
    fn erf_fingerprint_bit_sweep() {
        let fp = sweep_fingerprint(erf_f32, PROOF_STEP_CONTRACT);
        assert_eq!(fp, PROOF_ERF_FP_CONTRACT, "empreinte erf : 0x{fp:016x}");
    }

    #[test]
    fn gelu_fingerprint_bit_sweep() {
        let fp = sweep_fingerprint(gelu_f32, PROOF_STEP_CONTRACT);
        assert_eq!(fp, PROOF_GELU_FP_CONTRACT, "empreinte gelu : 0x{fp:016x}");
    }

    #[test]
    fn sin_fingerprint_bit_sweep() {
        let fp = sweep_fingerprint(sin_f32, PROOF_STEP_CONTRACT);
        assert_eq!(fp, PROOF_SIN_FP_CONTRACT, "empreinte sin : 0x{fp:016x}");
    }

    #[test]
    fn cos_fingerprint_bit_sweep() {
        let fp = sweep_fingerprint(cos_f32, PROOF_STEP_CONTRACT);
        assert_eq!(fp, PROOF_COS_FP_CONTRACT, "empreinte cos : 0x{fp:016x}");
    }

    #[test]
    fn tanh_fingerprint_bit_sweep() {
        let fp = sweep_fingerprint(tanh_f32, PROOF_STEP_CONTRACT);
        assert_eq!(fp, PROOF_TANH_FP_CONTRACT, "empreinte tanh : 0x{fp:016x}");
    }

    #[test]
    fn sigmoid_fingerprint_bit_sweep() {
        let fp = sweep_fingerprint(sigmoid_f32, PROOF_STEP_CONTRACT);
        assert_eq!(
            fp, PROOF_SIGMOID_FP_CONTRACT,
            "empreinte sigmoid : 0x{fp:016x}"
        );
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
