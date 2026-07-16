// scirust-simd/src/transformed/metrics.rs
//
// # Métriques déterministes du cadre transformé
//
// Outils de mesure réutilisables, tous **déterministes** (arithmétique `f64`
// pure, aucun ordre dépendant du parallélisme), agissant sur des éléments
// [`Hypercomplex<f64, N>`] (l'espace encodé où vivent les deux modèles).
//
// Le premier objet de recherche est le **défaut de transformation**
// `Δ = φ(A⋆B) − φ(A)⋆φ(B)` (Modèle A − Modèle B) : quand il est non nul, la
// transformation « déforme » l'algèbre. On fournit aussi des mesures de
// distorsion de norme et des invariants algébriques (commutateur, associateur)
// utiles aux expériences.

use super::hypercomplex::Hypercomplex;

/// Norme euclidienne `‖x‖ = √Σ cᵢ²`.
#[must_use]
pub fn norm<const N: usize>(x: &Hypercomplex<f64, N>) -> f64 {
    x.norm_sqr().sqrt()
}

/// Erreur absolue `‖a − b‖`.
#[must_use]
pub fn abs_error<const N: usize>(a: &Hypercomplex<f64, N>, b: &Hypercomplex<f64, N>) -> f64 {
    norm(&(*a - *b))
}

/// Erreur relative `‖a − b‖ / ‖a‖` (`0` si `‖a‖ = 0`).
#[must_use]
pub fn rel_error<const N: usize>(a: &Hypercomplex<f64, N>, b: &Hypercomplex<f64, N>) -> f64 {
    let na = norm(a);
    if na == 0.0 { 0.0 } else { abs_error(a, b) / na }
}

/// Erreur `L∞` (plus grand écart composante par composante).
#[must_use]
pub fn linf_error<const N: usize>(a: &Hypercomplex<f64, N>, b: &Hypercomplex<f64, N>) -> f64 {
    let mut m = 0.0f64;
    for i in 0..N
    {
        m = m.max((a.components()[i] - b.components()[i]).abs());
    }
    m
}

/// Distorsion de norme `| ‖a‖ − ‖b‖ |` (la transformation préserve-t-elle la
/// magnitude entre les deux modèles ?).
#[must_use]
pub fn norm_distortion<const N: usize>(a: &Hypercomplex<f64, N>, b: &Hypercomplex<f64, N>) -> f64 {
    (norm(a) - norm(b)).abs()
}

/// Rapport de bilan complet du défaut de transformation `Modèle A vs Modèle B`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DefectReport {
    /// Défaut absolu `‖Δ‖`.
    pub abs_l2: f64,
    /// Défaut relatif `‖Δ‖ / ‖ModèleA‖`.
    pub rel_l2: f64,
    /// Défaut `L∞`.
    pub linf: f64,
    /// Distorsion de norme entre les deux modèles.
    pub norm_distortion: f64,
}

/// Calcule le [`DefectReport`] entre `model_a` (`φ(A⋆B)`) et `model_b`
/// (`φ(A)⋆φ(B)`).
#[must_use]
pub fn defect_report<const N: usize>(
    model_a: &Hypercomplex<f64, N>,
    model_b: &Hypercomplex<f64, N>,
) -> DefectReport {
    DefectReport {
        abs_l2: abs_error(model_a, model_b),
        rel_l2: rel_error(model_a, model_b),
        linf: linf_error(model_a, model_b),
        norm_distortion: norm_distortion(model_a, model_b),
    }
}

/// Norme du commutateur `‖x·y − y·x‖` (défaut de commutativité).
#[must_use]
pub fn commutator_norm<const N: usize>(x: &Hypercomplex<f64, N>, y: &Hypercomplex<f64, N>) -> f64 {
    norm(&x.commutator(*y))
}

/// Norme de l'associateur `‖(x·y)·z − x·(y·z)‖` (défaut d'associativité).
#[must_use]
pub fn associator_norm<const N: usize>(
    x: &Hypercomplex<f64, N>,
    y: &Hypercomplex<f64, N>,
    z: &Hypercomplex<f64, N>,
) -> f64 {
    norm(&x.associator(*y, *z))
}
