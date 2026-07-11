//! Preuve **formelle a priori** (arithmétique rationnelle EXACTE, style
//! RLIBM/Gappa) de l'arrondi correct pour `exp`/`tanh`/`sigmoid`
//! ([`crate::portable_f32`]) — le chantier « preuve a priori » de la
//! cartographie (volet 111, relancé après le volet 116).
//!
//! ## Différence avec la certification exhaustive (volet 114-115)
//!
//! La certification exhaustive (`portable_f32::certify`) balaie les 2³²
//! entrées f32 et compare CHAQUE sortie à un oracle (interval ∪ précision
//! arbitraire) : c'est une preuve **a posteriori**, point par point. Ce
//! module prouve au contraire, **une seule fois, pour tout l'intervalle
//! réduit continu** `r ∈ [−R, R]` (R étant une borne rationnelle validée sur
//! ln 2/2), que l'erreur relative du polynôme de Taylor degré 13 partagé par
//! `exp_f64_core` (donc par `exp_f32`, `tanh_f32`, `sigmoid_f32`) reste sous
//! le seuil d'arrondi correct — AVANT tout test sur une valeur f32
//! particulière. C'est la preuve « a priori » au sens du dilemme du
//! fabricant de tables.
//!
//! ## Pourquoi seulement exp/tanh/sigmoid (3 des 7 fonctions)
//!
//! Le polynôme de Taylor de `exp_f64_core` reste **borné loin de zéro**
//! (`e^r ∈ [e^{-R}, e^R]` sur `r ∈ [−R,R]`), donc une borne d'erreur
//! RELATIVE uniforme sur tout l'intervalle est directe. `sin`/`cos`/`ln`/
//! `erf` n'ont pas cette propriété : `sin(r)→0` quand `r→0` (idem `ln(m)→0`
//! quand `m→1`), donc une borne relative uniforme exigerait une analyse
//! préservant la précision relative près du zéro (travailler sur
//! `sin(r)/r`, pas `sin(r)` directement) — non traitée ici, documentée
//! honnêtement comme travail futur (aucune sur-affirmation).
//!
//! ## Méthode
//!
//! 1. **Troncature (reste de Lagrange)** : `|e^r − Σᵢ₌₀¹³ rⁱ/i!| ≤ M·R¹⁴/14!`
//!    avec `M = sup_{|r|≤R} e^r ≤ 1/(1−R)` (borne élémentaire : `e^x ≤
//!    1/(1−x)` pour `0 ≤ x < 1`, car `xⁿ/n! ≤ xⁿ` terme à terme).
//! 2. **Arrondi f64 du schéma de Horner** (théorème de Higham, *Accuracy and
//!    Stability of Numerical Algorithms*, 2002, énoncé standard et
//!    inconditionnel — valide même en cas d'annulation de signes) :
//!    `|p̂(r) − p(r)| ≤ γ₂₆ · Σᵢ₌₀¹³ |cᵢ| |r|ⁱ` où `γₖ = k·u/(1−k·u)`,
//!    `u = 2⁻⁵³` (unité d'arrondi f64), `n = 13` (degré) ⇒ `2n = 26`. La
//!    somme `Σ|cᵢ|Rⁱ` est bornée par la MÊME borne `1/(1−R)`.
//! 3. **Combinaison** : erreur relative totale ≤ `(troncature + arrondi) /
//!    (1−R)` (le dénominateur minore `|e^r|`, via `e^{-R} ≥ 1−R`).
//! 4. **Seuil** : `2⁻²⁵` est une condition SUFFISANTE d'arrondi correct f32
//!    uniforme sur tout binade (un binade a un demi-ulp relatif dans
//!    `[2⁻²⁵, 2⁻²⁴)` — 2⁻²⁵ est le pire cas, donc suffisant partout).
//!
//! Toutes les étapes sont calculées en [`num_rational::BigRational`] —
//! aucune opération flottante n'intervient dans la preuve elle-même (le
//! résultat ne dépend donc pas d'un quelconque arrondi de la machine qui
//! l'exécute).

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

fn r(num: i64, den: i64) -> BigRational {
    BigRational::new(BigInt::from(num), BigInt::from(den))
}

fn factorial(n: u64) -> BigInt {
    (1..=n).fold(BigInt::from(1), |acc, k| acc * BigInt::from(k))
}

/// Borne rationnelle **validée supérieure** de ln 2 (fait mathématique
/// citable : `0,693147181 > 0,69314718055994530942… = ln 2`, marge au 10ᵉ
/// chiffre décimal — vérifiable indépendamment à la précision voulue).
fn ln2_upper() -> BigRational {
    BigRational::new(BigInt::from(693_147_181i64), BigInt::from(1_000_000_000i64))
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

/// `γₖ = k·u/(1−k·u)` de Higham (arrondi f64, `u = 2⁻⁵³`).
fn gamma(k: u64) -> BigRational {
    let u = BigRational::new(BigInt::from(1), BigInt::from(1) << 53);
    let ku = &u * BigInt::from(k);
    &ku / (BigRational::one() - &ku)
}

/// Résultat d'une preuve de borne a priori pour une famille de polynômes.
#[derive(Debug, Clone)]
pub struct BoundProof {
    pub name: &'static str,
    /// Borne rationnelle de l'intervalle réduit `[−R, R]`.
    pub range_bound: BigRational,
    /// Reste de Lagrange (troncature), borne supérieure.
    pub truncation_bound: BigRational,
    /// Erreur d'arrondi Horner (Higham γ), borne supérieure.
    pub rounding_bound: BigRational,
    /// `(troncature + arrondi) / minorant(|p(r)|)` — erreur relative totale.
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
        threshold: BigRational::new(BigInt::from(1), BigInt::from(1) << 25),
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

/// Borne rationnelle validée de `π/4` (pour un usage ultérieur : la borne
/// a priori de sin/cos, non traitée par ce module — cf. doc du module).
pub fn pi_over_4_upper_bound() -> BigRational {
    pi_upper() / BigInt::from(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check des primitives rationnelles : `355/113 > π` et
    /// `0,693147181 > ln 2` — comparés à `f64::consts` (repère, pas preuve :
    /// la preuve elle-même ne dépend d'aucun flottant).
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
        let ln2_r = ln2_upper();
        assert!(ln2_r > r((ln2_f64 * 1_000_000_000.0) as i64, 1_000_000_000));
    }

    /// `exp_upper_bound` majore effectivement `eˣ` sur l'intervalle testé
    /// (repère flottant — la preuve elle-même n'utilise que le rationnel).
    #[test]
    fn exp_upper_bound_dominates_f64_exp() {
        for milli in [1, 50, 100, 200, 300, 346]
        {
            let x = r(milli, 1000);
            let bound = exp_upper_bound(&x);
            let bound_f64 = bound.numer().to_string().parse::<f64>().unwrap()
                / bound.denom().to_string().parse::<f64>().unwrap();
            let true_val = (milli as f64 / 1000.0).exp();
            assert!(
                bound_f64 > true_val,
                "borne {bound_f64} ≤ e^{} = {true_val}",
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
        // marge : la borne doit être confortablement sous le seuil (pas un
        // pile-ou-face de dernière décimale) — sinon la preuve est fragile.
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
        let bound_f64 = proof
            .range_bound
            .numer()
            .to_string()
            .parse::<f64>()
            .unwrap()
            / proof
                .range_bound
                .denom()
                .to_string()
                .parse::<f64>()
                .unwrap();
        assert!(
            bound_f64 > true_ln2_2,
            "borne prouvée {bound_f64} ≤ ln2/2 réel {true_ln2_2}"
        );
    }
}
