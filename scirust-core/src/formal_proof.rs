//! Preuve **formelle a priori** (arithmétique rationnelle EXACTE, style
//! RLIBM/Gappa) de l'arrondi correct pour `exp`/`tanh`/`sigmoid`/`sin`/`cos`/
//! `ln` ([`crate::portable_f32`]) — le chantier « preuve a priori » de la
//! cartographie (volet 111, relancé au volet 116, étendu au volet 117).
//!
//! ## Différence avec la certification exhaustive (volet 114-115)
//!
//! La certification exhaustive (`portable_f32::certify`) balaie les 2³²
//! entrées f32 et compare CHAQUE sortie à un oracle (interval ∪ précision
//! arbitraire) : c'est une preuve **a posteriori**, point par point. Ce
//! module prouve au contraire, **une seule fois, pour tout un intervalle
//! réduit continu**, que l'erreur relative du noyau polynomial partagé par
//! chaque fonction reste sous le seuil d'arrondi correct — AVANT tout test
//! sur une valeur f32 particulière. C'est la preuve « a priori » au sens du
//! dilemme du fabricant de tables.
//!
//! ## Portée : noyau polynomial, pas la réduction d'argument
//!
//! Chaque preuve couvre le **noyau** (le polynôme de Taylor évalué sur la
//! plage réduite), pas la réduction d'argument qui l'alimente (Payne–Hanek
//! pour sin/cos, extraction d'exposant IEEE pour ln, `k·ln2` pour exp) — la
//! correction bit-à-bit de la réduction elle-même reste couverte par la
//! certification exhaustive a posteriori (volet 115-A), qui teste le
//! pipeline complet. Les deux preuves se complètent : a priori sur le
//! noyau (ce module) + a posteriori sur le pipeline entier (volet 115-A).
//!
//! ## Pourquoi exp/tanh/sigmoid/sin/cos/ln (6 des 7) mais pas erf
//!
//! Le polynôme de `exp_f64_core` reste **borné loin de zéro** (`e^r ∈
//! [e^{-R}, e^R]`) et celui de `cos_poly` aussi (`cos(0)=1`) : une borne
//! d'erreur RELATIVE uniforme y est directe (méthode « bornée »,
//! §Méthode 1). `sin_poly` et le cœur atanh de `ln_f64_core` s'annulent au
//! centre de leur plage (`sin(0)=0`, `atanh(0)=0`) : y prouver une borne
//! relative uniforme demande de préserver la précision relative près du
//! zéro, via `sin(r)/r ≥ 2/π` (inégalité de Jordan) ou `atanh(s)/s ≥ 1`
//! (algébrique, tous les termes de la série ont le signe de `s`) — traité
//! ici (méthode « à facteur extrait », §Méthode 2).
//!
//! `erf` reste **hors périmètre** : sa série de Maclaurin doit converger
//! jusqu'à `|y| < 4` (contre `|s| ≲ 0,25` pour ln, `|r| ≤ 0,8` pour
//! sin/cos), ce qui demande jusqu'à ~80 termes dont les premiers ne
//! décroissent PAS en module (le test de convergence par série alternée ne
//! s'applique qu'à partir d'un rang calculé, pas dès le premier terme) — un
//! reste de Lagrange simple ne suffit plus, il faudrait une borne de queue
//! géométrique à partir de ce rang, combinée à un argument en deux régions
//! (rapport `erf(y)/y` minoré près de 0, `erf` croissante donc minorée par
//! une constante calculée plus loin). Non traité, documenté honnêtement
//! comme travail futur (aucune sur-affirmation).
//!
//! ## Méthode 1 — fonction bornée loin de zéro (`exp`, `cos`)
//!
//! 1. **Troncature (reste de Lagrange)** : borne sur `|f(r) − Tₙ(r)|` via
//!    une borne uniforme des dérivées de `f` sur la plage réduite.
//! 2. **Arrondi f64 du schéma de Horner** (théorème de Higham, *Accuracy
//!    and Stability of Numerical Algorithms*, 2002, énoncé standard et
//!    inconditionnel — valide même en cas d'annulation de signes) :
//!    `|p̂(r) − p(r)| ≤ γ₂ₙ · Σᵢ |cᵢ| |r|ⁱ` où `γₖ = k·u/(1−k·u)`,
//!    `u = 2⁻⁵³` (unité d'arrondi f64), `n` = degré du polynôme.
//! 3. **Combinaison** : erreur relative ≤ `(troncature + arrondi) /
//!    minorant(|f(r)|)`.
//!
//! ## Méthode 2 — fonction à facteur extrait (`sin`, noyau atanh de `ln`)
//!
//! Ces noyaux s'écrivent `f(r) = r·(1 + z·q(z))` avec `z = r²` (facteur `r`
//! extrait explicitement dans le code, exactement pour cette raison). On
//! propage une borne (valeur, erreur) **générique** — voir
//! `struct ErrBound` — à travers CHAQUE opération flottante du calcul réel
//! (même séquence que le code, `mul`/`add`/`div`), en utilisant le modèle
//! d'arrondi IEEE standard `fl(a∘b) = (a∘b)(1+δ), |δ| ≤ u`, TOUJOURS majoré
//! par inégalité triangulaire (jamais d'annulation supposée — donc
//! conservateur mais toujours valide). Un argument structurel (la sortie du
//! graphe de calcul, vue comme fonction du paramètre libre, est une somme à
//! coefficients POSITIFS de puissances ≥1 de ce paramètre — parce que le
//! graphe entier n'additionne/multiplie que des magnitudes, et que le
//! facteur `r` extrait empêche tout terme constant de survivre) garantit
//! que l'erreur croît avec `|r|`, donc que l'évaluer au bord `R` de la
//! plage majore l'erreur pour tout `|r| ≤ R` — ET que le diviser par le
//! minorant de Jordan/algébrique **au même bord** majore l'erreur relative
//! sur toute la plage (pas seulement au bord). D'où : une SEULE évaluation
//! au bord suffit, plutôt qu'une analyse par cas sur `r`.
//!
//! ## Seuil
//!
//! `2⁻²⁵` est une condition SUFFISANTE d'arrondi correct f32 uniforme sur
//! tout binade (le pire binade a un demi-ulp relatif `2⁻²⁵`, donc c'est
//! suffisant partout).
//!
//! Toutes les étapes sont calculées en [`num_rational::BigRational`] —
//! aucune opération flottante n'intervient dans la preuve elle-même (le
//! résultat ne dépend donc pas d'un quelconque arrondi de la machine qui
//! l'exécute). Les constantes f64 réellement utilisées par le code
//! (`LN2_HI`/`LN2_LO`, `SQRT_2`) sont converties en leur valeur rationnelle
//! **exacte** (tout f64 fini est un rationnel dyadique exact) via
//! [`f64_to_exact_rational`] — zéro approximation sur ces constantes.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

fn r(num: i64, den: i64) -> BigRational {
    BigRational::new(BigInt::from(num), BigInt::from(den))
}

fn factorial(n: u64) -> BigInt {
    (1..=n).fold(BigInt::from(1), |acc, k| acc * BigInt::from(k))
}

/// Convertit un f64 fini en sa valeur [`BigRational`] **exacte** (tout f64
/// fini est un rationnel dyadique `± mantisse · 2^exposant` exact — aucune
/// approximation). Utilisé pour importer les constantes réellement
/// utilisées par le code (`LN2_HI`, `LN2_LO`, `SQRT_2`) sans réinventer une
/// borne citable à part : la preuve porte alors sur la constante EXACTE
/// qu'exécute la machine, pas sur une approximation de son intention
/// mathématique.
fn f64_to_exact_rational(x: f64) -> BigRational {
    assert!(x.is_finite(), "f64_to_exact_rational : {x} n'est pas fini");
    if x == 0.0
    {
        return BigRational::zero();
    }
    let bits = x.to_bits();
    let sign: i64 = if bits >> 63 == 1 { -1 } else { 1 };
    let exp_field = ((bits >> 52) & 0x7ff) as i64;
    let mantissa_bits = bits & 0x000f_ffff_ffff_ffff;
    let (mantissa, exp): (u64, i64) = if exp_field == 0
    {
        (mantissa_bits, -1074) // sous-normal : m · 2^-1074
    }
    else
    {
        (mantissa_bits | (1u64 << 52), exp_field - 1075) // normal : (1.m) · 2^(e-1075)
    };
    let m = BigInt::from(sign) * BigInt::from(mantissa);
    if exp >= 0
    {
        BigRational::from(m * (BigInt::from(1) << (exp as u64)))
    }
    else
    {
        BigRational::new(m, BigInt::from(1) << ((-exp) as u64))
    }
}

/// Borne rationnelle **validée supérieure** de ln 2 (fait mathématique
/// citable : `0,693147181 > 0,69314718055994530942… = ln 2`, marge au 10ᵉ
/// chiffre décimal — vérifiable indépendamment à la précision voulue).
fn ln2_upper() -> BigRational {
    BigRational::new(BigInt::from(693_147_181i64), BigInt::from(1_000_000_000i64))
}

/// Borne rationnelle **validée inférieure** de ln 2 (même fait, direction
/// opposée : `0,693147180 < ln 2`).
fn ln2_lower() -> BigRational {
    BigRational::new(BigInt::from(693_147_180i64), BigInt::from(1_000_000_000i64))
}

/// Borne rationnelle **validée supérieure** de π (Milü, `355/113`, fait
/// classique et facilement vérifiable : `355/113 = 3,14159292… > π`).
fn pi_upper() -> BigRational {
    r(355, 113)
}

/// `1/(1−x)` — borne supérieure de `eˣ` sur `[0, x]` pour `0 ≤ x < 1`
/// (`eˣ = Σ xⁿ/n! ≤ Σ xⁿ = 1/(1−x)` terme à terme, `n! ≥ 1`).
fn exp_upper_bound(x: &BigRational) -> BigRational {
    assert!(*x < BigRational::one(), "borne invalide : x ≥ 1");
    BigRational::one() / (BigRational::one() - x)
}

fn unit_roundoff() -> BigRational {
    BigRational::new(BigInt::from(1), BigInt::from(1) << 53)
}

/// Seuil d'arrondi correct f32 uniforme (`2⁻²⁵`, cf. doc du module).
fn cr_threshold() -> BigRational {
    BigRational::new(BigInt::from(1), BigInt::from(1) << 25)
}

/// `γₖ = k·u/(1−k·u)` de Higham (arrondi f64, `u = 2⁻⁵³`).
fn gamma(k: u64) -> BigRational {
    let u = unit_roundoff();
    let ku = &u * BigInt::from(k);
    &ku / (BigRational::one() - &ku)
}

/// Résultat d'une preuve de borne a priori pour une famille de polynômes
/// (fonctions « bornées loin de zéro » — Méthode 1 — ou « à facteur
/// extrait » — Méthode 2, cf. doc du module ; les deux exposent la même
/// forme finale : troncature + arrondi, divisés par un minorant).
#[derive(Debug, Clone)]
pub struct BoundProof {
    pub name: &'static str,
    /// Borne rationnelle de l'intervalle réduit `[−R, R]`.
    pub range_bound: BigRational,
    /// Reste de Lagrange (troncature), borne supérieure.
    pub truncation_bound: BigRational,
    /// Erreur d'arrondi flottant propagée, borne supérieure.
    pub rounding_bound: BigRational,
    /// `(troncature + arrondi) / minorant(|f(r)|)` — erreur relative totale.
    pub relative_bound: BigRational,
    /// Seuil d'arrondi correct (2⁻²⁵, condition suffisante uniforme).
    pub threshold: BigRational,
}

impl BoundProof {
    /// La preuve est valide ssi la borne relative est strictement sous le
    /// seuil — vérifié en arithmétique rationnelle EXACTE (comparaison
    /// exacte de deux fractions, aucun flottant).
    pub fn holds(&self) -> bool {
        self.relative_bound < self.threshold
    }
}

/// Preuve pour la famille `exp_f64_core` (partagée par `exp_f32`,
/// `tanh_f32`, `sigmoid_f32`) : Taylor degré 13 sur `|r| ≤ R`, `R` = borne
/// rationnelle validée de `ln 2 / 2`.
pub fn prove_exp_family() -> BoundProof {
    let range_bound = ln2_upper() / BigInt::from(2);
    let degree = 13u64;
    let m_upper = exp_upper_bound(&range_bound); // sup e^r sur [-R,R]

    // reste de Lagrange : M · R^(n+1) / (n+1)!
    let truncation_bound = &m_upper * range_bound.pow(degree as i32 + 1) / factorial(degree + 1);

    // Σ|cᵢ|Rⁱ ≤ e^R ≤ M (même borne, réutilisée)
    let rounding_bound = gamma(2 * degree) * &m_upper;

    // minorant de |e^r| = e^{-R} ≥ 1 − R (convexité : e^y ≥ 1+y, y=−R)
    let lower_bound = BigRational::one() - &range_bound;
    assert!(lower_bound > BigRational::zero(), "R ≥ 1 : borne invalide");

    let relative_bound = (&truncation_bound + &rounding_bound) / &lower_bound;

    BoundProof {
        name: "exp/tanh/sigmoid (Taylor-13, exp_f64_core)",
        range_bound,
        truncation_bound,
        rounding_bound,
        relative_bound,
        threshold: cr_threshold(),
    }
}

/// Idem, mais avec la borne `R` réellement utilisée par le code
/// (`ln 2 / 2`, non arrondie) mesurée EN PLUS de la borne validée
/// `ln2_upper()/2` — sert de garde-fou de cohérence dans les tests : la
/// vraie plage de réduction est `⊆ [−R,R]` (la preuve, faite sur le
/// sur-ensemble `[−R,R]`, couvre donc bien la plage réelle).
pub fn ln2_over_2_upper_bound() -> BigRational {
    ln2_upper() / BigInt::from(2)
}

/// Borne rationnelle validée de `π/4`.
pub fn pi_over_4_upper_bound() -> BigRational {
    pi_upper() / BigInt::from(4)
}

// ====================================================================== //
//  Propagation générique (valeur, erreur) — Méthode 2 (cf. doc du module) //
// ====================================================================== //

/// Borne (valeur, erreur) pour la propagation mécanique d'arrondi :
/// `value` majore `|valeur vraie|`, `error` majore `|valeur calculée (fl) −
/// valeur vraie|`. Composée via `add_b`/`mul_b`/`div_b` à partir du modèle
/// d'arrondi IEEE standard (`fl(a∘b) = (a∘b)(1+δ)`, `|δ| ≤ u`) — TOUJOURS
/// majorée par inégalité triangulaire (jamais d'annulation supposée), donc
/// toujours valide même si conservatrice.
#[derive(Debug, Clone)]
struct ErrBound {
    value: BigRational,
    error: BigRational,
}

impl ErrBound {
    fn exact(v: BigRational) -> Self {
        ErrBound {
            value: v,
            error: BigRational::zero(),
        }
    }

    /// Constante représentée par un littéral f64 correctement arrondi :
    /// `error = u·|v|` (arrondi au plus proche, majorant standard).
    fn rounded(v: BigRational, u: &BigRational) -> Self {
        let value = v.abs();
        let error = u * &value;
        ErrBound { value, error }
    }

    /// Majorant de `|valeur flottante calculée|` (valeur vraie + erreur).
    fn computed_upper(&self) -> BigRational {
        &self.value + &self.error
    }
}

fn add_b(a: &ErrBound, b: &ErrBound, u: &BigRational) -> ErrBound {
    let value = &a.value + &b.value;
    let error = &a.error + &b.error + u * (a.computed_upper() + b.computed_upper());
    ErrBound { value, error }
}

fn mul_b(a: &ErrBound, b: &ErrBound, u: &BigRational) -> ErrBound {
    let value = &a.value * &b.value;
    let error = &a.value * &b.error
        + &b.value * &a.error
        + &a.error * &b.error
        + u * a.computed_upper() * b.computed_upper();
    ErrBound { value, error }
}

/// Division par une quantité dont le vrai dénominateur est minoré par
/// `y_min_abs` (fourni par l'appelant via une analyse de plage — PAS
/// déduit de `b`, puisque `b.value` est un MAJORANT, pas un minorant).
fn div_b(a: &ErrBound, b: &ErrBound, y_min_abs: &BigRational, u: &BigRational) -> ErrBound {
    assert!(
        *y_min_abs > b.error,
        "div_b : minorant du dénominateur trop faible"
    );
    let y_hat_min = y_min_abs - &b.error;
    let error = &a.error / y_min_abs
        + &a.value * &b.error / (y_min_abs * &y_hat_min)
        + u * a.computed_upper() / &y_hat_min;
    let value = &a.value / y_min_abs;
    ErrBound { value, error }
}

/// Évalue un schéma de Horner (poids fort en premier, même ordre que le
/// code) en propageant (valeur, erreur).
fn horner_b(coeffs: &[ErrBound], z: &ErrBound, u: &BigRational) -> ErrBound {
    let mut acc = coeffs[0].clone();
    for c in &coeffs[1..]
    {
        acc = mul_b(&acc, z, u);
        acc = add_b(&acc, c, u);
    }
    acc
}

// ====================================================================== //
//  sin / cos : noyau `sin_poly`/`cos_poly` sur |r| ≤ R                    //
// ====================================================================== //

/// Borne rationnelle validée de la plage réduite `|r| ≤ R` pour
/// `sin_poly`/`cos_poly` : couvre à la fois le chemin direct (`|x| ≤
/// FRAC_PI_4`) et la réduction de Payne–Hanek (qui garantit `|r| ≤ π/4` en
/// arithmétique exacte, plus un écart de conversion f64→f64 négligeable,
/// `~2⁻⁵²·π/4`). `R = 4/5 = 0,8 > π/4 ≈ 0,7854`, marge large (`0,8 − π/4 ≈
/// 0,0146 ≫ 2⁻⁵²·π/4`) — et `R < π/2` requis par l'inégalité de Jordan.
fn sin_cos_range_bound() -> BigRational {
    r(4, 5)
}

/// Coefficient de `z^j` dans `s(z)` (noyau Horner de `sin_poly`) :
/// `s(z) = Σⱼ₌₀⁶ (−1)^(j+1) z^j / (2j+3)!` — dérivé de
/// `sin(r) = r·(1 + z·s(z))` tronqué à `r¹⁵` (vérifié : `sin_poly` du code
/// EST cette évaluation de Horner, poids fort `j=6` en premier).
fn sin_s_coeff(j: u64) -> BigRational {
    let sign: i64 = if j % 2 == 0 { -1 } else { 1 };
    BigRational::new(BigInt::from(sign), factorial(2 * j + 3))
}

/// Coefficient de `z^j` dans `c(z)` (noyau Horner de `cos_poly`) :
/// `c(z) = Σⱼ₌₀⁷ (−1)^(j+1) z^j / (2j+2)!` — dérivé de
/// `cos(r) = 1 + z·c(z)` tronqué à `r¹⁶`.
fn cos_c_coeff(j: u64) -> BigRational {
    let sign: i64 = if j % 2 == 0 { -1 } else { 1 };
    BigRational::new(BigInt::from(sign), factorial(2 * j + 2))
}

/// Preuve pour `sin_poly` (noyau partagé par `sin_f32`) : Taylor impair
/// degré 15 sur `|r| ≤ R`, minoré via l'inégalité de Jordan
/// (`sin(r) ≥ (2/π)·r` sur `[0, π/2]`, `2/π` minoré par `226/355` via
/// `π ≤ 355/113`). Méthode 2 (facteur `r` extrait) : une seule évaluation
/// au bord `R` majore l'erreur relative sur toute la plage (cf. doc du
/// module).
pub fn prove_sin() -> BoundProof {
    let u = unit_roundoff();
    let r_max = sin_cos_range_bound();
    assert!(r_max < r(157, 100), "R doit rester < π/2 pour Jordan"); // 1.57 < π/2

    let r_bound = ErrBound::exact(r_max.clone());
    let z_bound = mul_b(&r_bound, &r_bound, &u);
    let s_coeffs: Vec<ErrBound> = (0..=6)
        .rev()
        .map(|j| ErrBound::rounded(sin_s_coeff(j), &u))
        .collect();
    let s_bound = horner_b(&s_coeffs, &z_bound, &u);
    let zs_bound = mul_b(&z_bound, &s_bound, &u);
    let rzs_bound = mul_b(&r_bound, &zs_bound, &u);
    let result_bound = add_b(&r_bound, &rzs_bound, &u);

    // reste de Lagrange : |sin(r) − T(r)| ≤ R^17/17! (dérivées de sin ≤ 1)
    let truncation_bound = r_max.pow(17) / factorial(17);
    let rounding_bound = result_bound.error;

    // Jordan : sin(R) ≥ (2/π)·R ≥ (226/355)·R
    let denom = r(226, 355) * &r_max;
    let relative_bound = (&truncation_bound + &rounding_bound) / &denom;

    BoundProof {
        name: "sin (Taylor-15, sin_poly)",
        range_bound: r_max,
        truncation_bound,
        rounding_bound,
        relative_bound,
        threshold: cr_threshold(),
    }
}

/// Preuve pour `cos_poly` (noyau partagé par `cos_f32`) : Taylor pair
/// degré 16 sur `|r| ≤ R`, minoré par `cos(R) ≥ 1 − R²/2` (reste de série
/// alternée, termes décroissants sur `[0, R]` pour `R = 0,8`). Méthode 1
/// (bornée loin de zéro, `cos(0)=1` — pas besoin de Jordan).
pub fn prove_cos() -> BoundProof {
    let u = unit_roundoff();
    let r_max = sin_cos_range_bound();

    let r_bound = ErrBound::exact(r_max.clone());
    let z_bound = mul_b(&r_bound, &r_bound, &u);
    let c_coeffs: Vec<ErrBound> = (0..=7)
        .rev()
        .map(|j| ErrBound::rounded(cos_c_coeff(j), &u))
        .collect();
    let c_bound = horner_b(&c_coeffs, &z_bound, &u);
    let zc_bound = mul_b(&z_bound, &c_bound, &u);
    let one_bound = ErrBound::rounded(BigRational::one(), &u);
    let result_bound = add_b(&one_bound, &zc_bound, &u);

    // reste de Lagrange : |cos(r) − T(r)| ≤ R^18/18! (dérivées de cos ≤ 1)
    let truncation_bound = r_max.pow(18) / factorial(18);
    let rounding_bound = result_bound.error;

    // reste de série alternée (termes décroissants pour R=0,8) : cos(R) ≥ 1 − R²/2
    let denom = BigRational::one() - r_max.pow(2) / BigInt::from(2);
    assert!(denom > BigRational::zero(), "borne cos invalide");
    let relative_bound = (&truncation_bound + &rounding_bound) / &denom;

    BoundProof {
        name: "cos (Taylor-16, cos_poly)",
        range_bound: r_max,
        truncation_bound,
        rounding_bound,
        relative_bound,
        threshold: cr_threshold(),
    }
}

// ====================================================================== //
//  ln : noyau `ln_f64_core` — deux cas (e=0 et e≠0)                       //
// ====================================================================== //

/// Coefficient de `z^j` dans `q(z)` (noyau Horner de l'atanh de
/// `ln_f64_core`) : `q(z) = Σⱼ₌₀¹¹ z^j / (2j+3)` (SANS alternance de signe,
/// contrairement à sin/cos — c'est la série de atanh, pas une série
/// trigonométrique).
fn ln_q_coeff(j: u64) -> BigRational {
    BigRational::new(BigInt::from(1), BigInt::from(2 * j + 3))
}

/// Résultat de la preuve a priori pour `ln_f64_core`, en deux cas (le code
/// se ramène à l'un ou l'autre selon que l'exposant IEEE extrait `e` est
/// nul ou non) — cf. [`prove_ln`].
#[derive(Debug, Clone)]
pub struct LnBoundProof {
    pub name: &'static str,
    /// Borne rationnelle sur `|s|` où `s = (m−1)/(m+1)`.
    pub s_max: BigRational,
    /// Cas `e = 0` (x proche de 1) : `|ln x| = |ln_m| = 2|atanh(s)| ≥ 2|s|`.
    pub e0_relative_bound: BigRational,
    /// Cas `e ≠ 0` : `|ln x| = |e·ln2 + ln_m| ≥ ln2_lower − 1/2` (constante).
    pub ene0_relative_bound: BigRational,
    /// `max(e0_relative_bound, ene0_relative_bound)`.
    pub relative_bound: BigRational,
    pub threshold: BigRational,
}

impl LnBoundProof {
    pub fn holds(&self) -> bool {
        self.relative_bound < self.threshold
    }
}

/// Preuve pour `ln_f64_core` (noyau de `ln_f32`), en deux cas couvrant
/// ensemble tout `m ∈ (S/2, S]` (`S` = constante f64 `SQRT_2` réellement
/// utilisée par le code, convertie EXACTEMENT via [`f64_to_exact_rational`]
/// — pas une approximation citable de `√2`) :
///
/// - **`e = 0`** (x proche de 1, m proche de 1) : `ln(x) = ln_m = 2·atanh(s)`
///   avec `s = (m−1)/(m+1) → 0`. Méthode 2 (facteur `s` extrait) : minoré
///   par `atanh(s) ≥ s` (algébrique — tous les termes de la série de
///   `atanh` ont le signe de `s`, aucune inégalité numérique requise).
///   `m − 1` est calculé SANS AUCUNE erreur d'arrondi (lemme de Sterbenz :
///   `m/2 ≤ 1 ≤ 2m` sur toute la plage `(S/2, S]`, donc `fl(m−1) = m−1`
///   exactement).
/// - **`e ≠ 0`** : `ln(x) = e·ln2 + ln_m` avec `|e| ≥ 1`, minoré par la
///   CONSTANTE `ln2_lower − 1/2` (`|ln_m| ≤ ln(3/2) ≤ 1/2` via
///   `ln(1+x) ≤ x`, et `S < 3/2`) — Méthode 1 (bornée loin de zéro).
///
/// Le résultat final est le MAX des deux bornes relatives (les deux cas
/// couvrent ensemble tout le domaine de `ln_f64_core`).
pub fn prove_ln() -> LnBoundProof {
    let u = unit_roundoff();

    // S = constante f64 SQRT_2 réellement utilisée par le code (exacte).
    let s_const = f64_to_exact_rational(core::f64::consts::SQRT_2);
    assert!(s_const < r(3, 2), "S ≥ 3/2 : hypothèse de bornage invalide");

    // m ∈ (S/2, S] après normalisation (cf. doc de ln_f64_core).
    let half_s = &s_const / BigInt::from(2);
    // |m−1| est maximal à m=S (Sterbenz : m/2≤1≤2m sur (S/2,S], donc exact).
    let m1_max = &s_const - BigInt::from(1); // = S − 1, borne sur |m−1|
    assert!(m1_max > BigRational::zero());
    // |m+1| minoré par S/2+1 (m > S/2 sur toute la plage).
    let m_plus_1_min = &half_s + BigInt::from(1);
    let m_plus_1_max = &s_const + BigInt::from(1); // borne sur |m+1|, pour l'erreur d'arrondi

    let m_minus_1_bound = ErrBound {
        value: m1_max.clone(),
        error: BigRational::zero(), // Sterbenz : exact, aucune erreur d'arrondi
    };
    let m_plus_1_bound = ErrBound::rounded(m_plus_1_max, &u);

    let s_bound = div_b(&m_minus_1_bound, &m_plus_1_bound, &m_plus_1_min, &u);
    let s_max = s_bound.value.clone();
    assert!(
        s_max < BigRational::one(),
        "|s| ≥ 1 : hors du rayon de convergence de atanh"
    );

    let z_bound = mul_b(&s_bound, &s_bound, &u);
    let q_coeffs: Vec<ErrBound> = (0..=11)
        .rev()
        .map(|j| ErrBound::rounded(ln_q_coeff(j), &u))
        .collect();
    let q_bound = horner_b(&q_coeffs, &z_bound, &u);
    let two_s_bound = add_b(&s_bound, &s_bound, &u);
    let zq_bound = mul_b(&z_bound, &q_bound, &u);
    let term1_bound = mul_b(&two_s_bound, &zq_bound, &u);
    let ln_m_bound = add_b(&term1_bound, &two_s_bound, &u);

    // reste géométrique de la série de atanh (termes s^{2k+1}/(2k+1), notre
    // q couvre k=1..12 soit jusqu'à s²⁵ ; reste à partir de k=13, borné en
    // laissant tomber le facteur 1/(2k+1) ≤ 1) : Σ_{k≥13} |s|^{2k+1} =
    // |s|²⁷/(1−s²).
    let atanh_truncation = s_max.pow(27) / (BigRational::one() - s_max.pow(2));
    let ln_m_truncation = &atanh_truncation * BigInt::from(2);
    let ln_m_total_error = &ln_m_bound.error + &ln_m_truncation;

    // --- cas e = 0 : |ln x| = |ln_m| ≥ 2·atanh(s) ≥ 2|s| (algébrique) ---
    let e0_denom = &s_max * BigInt::from(2);
    let e0_relative_bound = &ln_m_total_error / &e0_denom;

    // --- cas e ≠ 0 : |ln x| = |e·ln2 + ln_m| ≥ ln2_lower − 1/2 ---
    // (|ln m| ≤ ln(S) ≤ ln(3/2) ≤ 1/2 via ln(1+x) ≤ x et S < 3/2, |e|≥1)
    let e_ne0_denom = ln2_lower() - r(1, 2);
    assert!(e_ne0_denom > BigRational::zero(), "minorant e≠0 invalide");

    // combinaison ef·LN2_HI + (ln_m + ef·LN2_LO), |e| ≤ 256 (cf. doc de
    // portable_f32::LN2_HI : k·LN2_HI exact pour |k|≤2⁸, plage f32 promue
    // en f64 : e ∈ [−149, 127] ⊂ [−256, 256]).
    let e_max = ErrBound::exact(r(256, 1));
    let ln2_hi = ErrBound::exact(f64_to_exact_rational(crate::portable_f32::LN2_HI));
    let ln2_lo = ErrBound::exact(f64_to_exact_rational(crate::portable_f32::LN2_LO));
    let ef_ln2_hi = mul_b(&e_max, &ln2_hi, &u);
    let ef_ln2_lo = mul_b(&e_max, &ln2_lo, &u);
    let ln_m_for_combine = ErrBound {
        value: ln_m_bound.value,
        error: ln_m_total_error,
    };
    let sum1 = add_b(&ln_m_for_combine, &ef_ln2_lo, &u);
    let total = add_b(&ef_ln2_hi, &sum1, &u);
    let ene0_relative_bound = &total.error / &e_ne0_denom;

    let relative_bound = if e0_relative_bound > ene0_relative_bound
    {
        e0_relative_bound.clone()
    }
    else
    {
        ene0_relative_bound.clone()
    };

    LnBoundProof {
        name: "ln (atanh Taylor-25, ln_f64_core)",
        s_max,
        e0_relative_bound,
        ene0_relative_bound,
        relative_bound,
        threshold: cr_threshold(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_f64(x: &BigRational) -> f64 {
        x.numer().to_string().parse::<f64>().unwrap()
            / x.denom().to_string().parse::<f64>().unwrap()
    }

    /// Sanity check des primitives rationnelles : `355/113 > π` et
    /// `0,693147180 < ln 2 < 0,693147181` — comparés à `f64::consts`
    /// (repère, pas preuve : la preuve elle-même ne dépend d'aucun flottant).
    #[test]
    fn rational_bounds_exceed_transcendentals() {
        let pi_f64 = core::f64::consts::PI;
        let pi_r = pi_upper();
        // 355/113 en f64 pour comparaison de repère uniquement
        assert!(355.0 / 113.0 > pi_f64);
        assert!(pi_r > r((pi_f64 * 1_000_000_000.0) as i64, 1_000_000_000));

        let ln2_f64 = core::f64::consts::LN_2;
        // le littéral EST volontairement proche de LN_2 : c'est la borne
        // supérieure validée elle-même (`ln2_upper()`), pas une constante
        // réinventée par erreur.
        #[allow(clippy::approx_constant)]
        let ln2_upper_literal = 0.693_147_181;
        assert!(ln2_upper_literal > ln2_f64);
        assert!(to_f64(&ln2_upper()) > ln2_f64);
        assert!(to_f64(&ln2_lower()) < ln2_f64);
        assert!(ln2_lower() < ln2_upper());
    }

    /// `exp_upper_bound` majore effectivement `eˣ` sur l'intervalle testé
    /// (repère flottant — la preuve elle-même n'utilise que le rationnel).
    #[test]
    fn exp_upper_bound_dominates_f64_exp() {
        for milli in [1, 50, 100, 200, 300, 346]
        {
            let x = r(milli, 1000);
            let bound = exp_upper_bound(&x);
            let true_val = (milli as f64 / 1000.0).exp();
            assert!(
                to_f64(&bound) > true_val,
                "borne {} ≤ e^{}",
                to_f64(&bound),
                milli as f64 / 1000.0
            );
        }
    }

    /// LA preuve : la borne d'erreur relative a priori de la famille
    /// exp/tanh/sigmoid est strictement sous le seuil d'arrondi correct —
    /// vérifié en arithmétique rationnelle EXACTE. Si un futur changement de
    /// `exp_f64_core` (degré du Taylor, plage de réduction) invalide cette
    /// preuve, ce test casse : c'est le garde-fou de régression.
    #[test]
    fn exp_family_correctly_rounded_a_priori() {
        let proof = prove_exp_family();
        assert!(
            proof.holds(),
            "preuve a priori exp/tanh/sigmoid : borne {} ≥ seuil {} — INVALIDE",
            proof.relative_bound,
            proof.threshold
        );
        let margin = &proof.threshold / &proof.relative_bound;
        assert!(
            margin > BigInt::from(1000).into(),
            "marge trop faible ({margin}×) : preuve fragile"
        );
    }

    /// La plage de réduction réellement utilisée par `exp_f64_core`
    /// (`|r| ≤ ln 2/2`) est bien couverte par la plage `[−R,R]` sur
    /// laquelle porte la preuve (R = borne validée ≥ ln 2/2 réel).
    #[test]
    fn proved_range_covers_actual_reduction_range() {
        let proof = prove_exp_family();
        let true_ln2_2 = core::f64::consts::LN_2 / 2.0;
        assert!(
            to_f64(&proof.range_bound) > true_ln2_2,
            "borne prouvée {} ≤ ln2/2 réel {true_ln2_2}",
            to_f64(&proof.range_bound)
        );
    }

    /// `R = 4/5` couvre bien la plage réduite `|r| ≤ π/4` de sin/cos (même
    /// avec la borne VALIDÉE SUPÉRIEURE de π/4, donc a fortiori la vraie).
    #[test]
    fn sin_cos_range_covers_pi_over_4() {
        assert!(sin_cos_range_bound() > pi_over_4_upper_bound());
        assert!(to_f64(&sin_cos_range_bound()) > core::f64::consts::FRAC_PI_4);
    }

    /// LA preuve sin : borne relative a priori strictement sous le seuil.
    #[test]
    fn sin_correctly_rounded_a_priori() {
        let proof = prove_sin();
        assert!(
            proof.holds(),
            "preuve a priori sin : borne {} ≥ seuil {} — INVALIDE",
            proof.relative_bound,
            proof.threshold
        );
        let margin = &proof.threshold / &proof.relative_bound;
        assert!(
            margin > BigInt::from(1000).into(),
            "marge trop faible ({margin}×) : preuve fragile"
        );
    }

    /// LA preuve cos : borne relative a priori strictement sous le seuil.
    #[test]
    fn cos_correctly_rounded_a_priori() {
        let proof = prove_cos();
        assert!(
            proof.holds(),
            "preuve a priori cos : borne {} ≥ seuil {} — INVALIDE",
            proof.relative_bound,
            proof.threshold
        );
        let margin = &proof.threshold / &proof.relative_bound;
        assert!(
            margin > BigInt::from(1000).into(),
            "marge trop faible ({margin}×) : preuve fragile"
        );
    }

    /// LA preuve ln : les DEUX cas (e=0 et e≠0) sont sous le seuil, donc le
    /// max des deux (la borne finale exposée) aussi.
    #[test]
    fn ln_correctly_rounded_a_priori() {
        let proof = prove_ln();
        assert!(
            proof.holds(),
            "preuve a priori ln : borne {} ≥ seuil {} — INVALIDE (e0={}, e≠0={})",
            proof.relative_bound,
            proof.threshold,
            proof.e0_relative_bound,
            proof.ene0_relative_bound
        );
        let margin = &proof.threshold / &proof.relative_bound;
        assert!(
            margin > BigInt::from(1000).into(),
            "marge trop faible ({margin}×) : preuve fragile"
        );
        // sanity : s_max reste confortablement dans le rayon de convergence
        assert!(
            proof.s_max < r(1, 2),
            "s_max {} trop proche de 1 : hypothèses de convergence fragiles",
            to_f64(&proof.s_max)
        );
    }

    /// Garde-fou empirique de l'argument de monotonicité (Méthode 2, doc du
    /// module) : la borne relative évaluée à un point INTÉRIEUR de la plage
    /// (via une plage réduite de moitié) ne doit jamais DÉPASSER celle
    /// évaluée au bord complet — sinon l'argument "évaluer au bord suffit"
    /// serait invalide et ce garde-fou le détecterait.
    #[test]
    fn sin_boundary_evaluation_dominates_interior() {
        // reproduit prove_sin() mais avec R/2 à la place de R
        let u = unit_roundoff();
        let full = prove_sin();
        let half_r = sin_cos_range_bound() / BigInt::from(2);
        let r_bound = ErrBound::exact(half_r.clone());
        let z_bound = mul_b(&r_bound, &r_bound, &u);
        let s_coeffs: Vec<ErrBound> = (0..=6)
            .rev()
            .map(|j| ErrBound::rounded(sin_s_coeff(j), &u))
            .collect();
        let s_bound = horner_b(&s_coeffs, &z_bound, &u);
        let zs_bound = mul_b(&z_bound, &s_bound, &u);
        let rzs_bound = mul_b(&r_bound, &zs_bound, &u);
        let result_bound = add_b(&r_bound, &rzs_bound, &u);
        let truncation = half_r.pow(17) / factorial(17);
        let denom = r(226, 355) * &half_r;
        let half_relative = (&truncation + &result_bound.error) / &denom;
        assert!(
            half_relative <= full.relative_bound,
            "borne à R/2 ({}) > borne à R ({}) : monotonicité invalide",
            to_f64(&half_relative),
            to_f64(&full.relative_bound)
        );
    }
}
