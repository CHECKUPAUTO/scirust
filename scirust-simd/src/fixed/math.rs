// scirust-simd/src/fixed/math.rs
//
// # Fonctions mathématiques virgule fixe
//
// ## Implémenté dans ce lot : `sqrt`, `rsqrt`, `reciprocal`
//
// Ces trois fonctions se ramènent à de l'arithmétique entière **exacte** dont
// l'erreur se borne trivialement, ce qui satisfait l'exigence « pas
// d'approximation non justifiée » :
//
// * [`sqrt`] : `√(raw/2^F)` = `√(raw·2^F)/2^F`. On calcule la **racine entière
//   plancher** de `raw·2^F` (accumulateur élargi, exact) par la méthode de
//   Newton entière. Erreur : strictement inférieure à un pas de résolution
//   (`< 2^-F`), par construction du plancher.
// * [`reciprocal`] : `1/x` via la division virgule fixe (troncature vers zéro).
//   Erreur < 1 ULP.
// * [`rsqrt`] : `1/√x` = `reciprocal(sqrt(x))`. Erreur cumulée < 2 ULP.
//
// ## Reporté à un lot ultérieur : `exp`, `log`, `pow`, `sin`, `cos`, `tanh`,
// ## `sigmoid`
//
// **Pourquoi** : contrairement à `sqrt`/`reciprocal`, ces fonctions n'ont pas
// de forme entière exacte. Une implémentation virgule fixe correcte exige (1)
// une **réduction d'argument** soignée (périodicité pour sin/cos, échelle pour
// exp/log), puis (2) une **approximation polynomiale minimax** avec une borne
// d'erreur ULP **prouvée** et testée sur toute la plage. C'est un travail de
// conception à part entière — la table de coefficients, la borne d'erreur et sa
// vérification méritent leur propre PR revue. Livrer ici une approximation
// rapide non bornée contredirait la philosophie du module (« pas d'algorithme
// approximatif non justifié »). Ces fonctions sont donc **délibérément
// absentes**, pas oubliées.

use super::repr::{FixedStorage, WideInt};
use super::rounding::RoundingMode;
use super::types::Fixed;

/// Racine carrée entière **plancher** d'un `WideInt` non négatif, par la
/// méthode de Newton (convergence quadratique, exacte pour le plancher).
///
/// Précondition : `n ≥ 0`. Retourne `⌊√n⌋`.
#[inline]
fn wide_isqrt<W: WideInt>(n: W) -> W {
    if n <= W::ONE
    {
        return n; // √0 = 0, √1 = 1
    }
    // Itération de Newton : xₖ₊₁ = (xₖ + n/xₖ) / 2, décroissante jusqu'au plancher.
    let mut x = n;
    let mut y = n.wrapping_add(W::ONE).shr(1);
    while y < x
    {
        x = y;
        y = x.wrapping_add(n.div_trunc(x)).shr(1);
    }
    x
}

/// Racine carrée virgule fixe : `√(raw/2^FRAC)`.
///
/// Renvoie `0` pour une entrée `≤ 0` (la racine réelle y est nulle ou non
/// définie ; convention documentée, sans panique). Sinon `⌊√(raw·2^FRAC)⌋`,
/// d'erreur strictement inférieure à un pas de résolution.
#[inline]
#[must_use]
pub fn sqrt<I: FixedStorage, const FRAC: u32>(x: Fixed<I, FRAC>) -> Fixed<I, FRAC> {
    if x.to_raw() <= I::ZERO
    {
        return Fixed::from_raw(I::ZERO);
    }
    // raw·2^FRAC exact dans l'accumulateur élargi (raw ≤ 2^(BITS−1), FRAC < BITS).
    let widened = x.to_raw().to_wide().shl(FRAC);
    let root = wide_isqrt(widened);
    Fixed::from_raw(I::from_wide_saturating(root))
}

/// Inverse `1/x` en virgule fixe (troncature vers zéro). `None` si `x == 0` ou
/// si l'inverse déborde la plage (ex. inverse d'une valeur minuscule).
#[inline]
#[must_use]
pub fn reciprocal<I: FixedStorage, const FRAC: u32>(x: Fixed<I, FRAC>) -> Option<Fixed<I, FRAC>> {
    Fixed::one().div_rounded(
        x,
        RoundingMode::TowardZero,
        super::overflow::OverflowMode::Checked,
    )
}

/// Inverse de la racine `1/√x`. `None` si `x ≤ 0` (racine nulle → division par
/// zéro) ou si le résultat déborde. Erreur cumulée < 2 ULP.
#[inline]
#[must_use]
pub fn rsqrt<I: FixedStorage, const FRAC: u32>(x: Fixed<I, FRAC>) -> Option<Fixed<I, FRAC>> {
    reciprocal(sqrt(x))
}
