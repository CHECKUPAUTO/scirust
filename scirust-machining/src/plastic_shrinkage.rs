//! Injection plastique — **retrait au moulage** : compensation des dimensions
//! d'empreinte à partir du taux de retrait matière.
//!
//! ```text
//! dimension d'empreinte        Lc = L·(1 + s)
//! dimension réelle de pièce    L  = Lc/(1 + s)
//! surdimensionnement empreinte Δ  = L·s
//! taux de retrait déduit       s  = (Lc − L)/L
//! ```
//!
//! `L` dimension nominale visée sur la pièce (m), `Lc` dimension à usiner dans
//! l'empreinte (m), `Δ` surépaisseur d'empreinte à prévoir (m), `s` taux de
//! retrait de la matière (sans dimension : `0.02` = 2 %). Toutes les longueurs
//! sont dans la **même** unité (SI : mètres), `s` étant un rapport pur, les
//! relations sont valables quelle que soit l'unité de longueur pourvu qu'elle
//! soit commune à `L` et `Lc`.
//!
//! **Convention** : SI cohérent, longueurs en mètres. **Limite honnête** :
//! retrait supposé **isotrope et uniforme** (un même `s` dans toutes les
//! directions) — dans la réalité le retrait est **anisotrope** (différent
//! parallèlement et perpendiculairement à l'écoulement, dépendant de la
//! pression de maintien, de l'épaisseur et des fibres). Le **taux de retrait
//! matière** `s` est **fourni par l'appelant** d'après la fiche matière ou un
//! essai : aucune valeur « par défaut » n'est supposée ici.

/// Valide un taux de retrait `s` : fini et strictement supérieur à `-1`
/// (sinon `1 + s <= 0` rendrait l'empreinte nulle ou négative).
///
/// Panique si `shrinkage_rate` n'est pas fini ou si `shrinkage_rate <= -1`.
fn check_shrinkage_rate(shrinkage_rate: f64) {
    assert!(
        shrinkage_rate.is_finite(),
        "le taux de retrait doit être un nombre fini"
    );
    assert!(
        shrinkage_rate > -1.0,
        "le taux de retrait doit vérifier s > -1 pour que (1 + s) reste positif"
    );
}

/// Dimension à usiner dans l'empreinte `Lc = L·(1 + s)` (m).
///
/// La surface du moule est agrandie pour compenser le retrait de la matière
/// au refroidissement. Relation réciproque de [`plastic_actual_part_dimension`].
///
/// Panique si `part_dimension <= 0`, ou si `shrinkage_rate` n'est pas fini
/// ou vérifie `shrinkage_rate <= -1`.
pub fn cavity_dimension(part_dimension: f64, shrinkage_rate: f64) -> f64 {
    assert!(
        part_dimension > 0.0,
        "la dimension de pièce doit être strictement positive"
    );
    check_shrinkage_rate(shrinkage_rate);
    part_dimension * (1.0 + shrinkage_rate)
}

/// Dimension réelle de la pièce après retrait `L = Lc/(1 + s)` (m).
///
/// Inverse de [`cavity_dimension`] : à partir de l'empreinte usinée, prédit la
/// cote obtenue une fois la pièce refroidie.
///
/// Panique si `cavity_dimension <= 0`, ou si `shrinkage_rate` n'est pas fini
/// ou vérifie `shrinkage_rate <= -1`.
pub fn plastic_actual_part_dimension(cavity_dimension: f64, shrinkage_rate: f64) -> f64 {
    assert!(
        cavity_dimension > 0.0,
        "la dimension d'empreinte doit être strictement positive"
    );
    check_shrinkage_rate(shrinkage_rate);
    cavity_dimension / (1.0 + shrinkage_rate)
}

/// Surdimensionnement de l'empreinte `Δ = L·s` (m) : écart absolu à ajouter à
/// la cote nominale de pièce pour tailler l'empreinte, `Δ = Lc − L`.
///
/// Positif pour un retrait `s > 0` (empreinte plus grande que la pièce).
///
/// Panique si `part_dimension <= 0`, ou si `shrinkage_rate` n'est pas fini
/// ou vérifie `shrinkage_rate <= -1`.
pub fn cavity_shrinkage_compensation(part_dimension: f64, shrinkage_rate: f64) -> f64 {
    assert!(
        part_dimension > 0.0,
        "la dimension de pièce doit être strictement positive"
    );
    check_shrinkage_rate(shrinkage_rate);
    part_dimension * shrinkage_rate
}

/// Taux de retrait déduit de la mesure `s = (Lc − L)/L` (sans dimension).
///
/// Réciproque de [`cavity_dimension`] vue comme fonction de `s` : à empreinte et
/// pièce mesurées, retrouve le retrait effectif de la matière.
///
/// Panique si `part_dimension <= 0` ou `cavity_dimension <= 0`.
pub fn plastic_shrinkage_rate(cavity_dimension: f64, part_dimension: f64) -> f64 {
    assert!(
        cavity_dimension > 0.0,
        "la dimension d'empreinte doit être strictement positive"
    );
    assert!(
        part_dimension > 0.0,
        "la dimension de pièce doit être strictement positive"
    );
    (cavity_dimension - part_dimension) / part_dimension
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cavity_and_actual_are_reciprocal() {
        // Réciprocité : usiner l'empreinte puis mouler redonne la cote visée.
        let (l, s) = (0.120_f64, 0.018);
        let lc = cavity_dimension(l, s);
        assert_relative_eq!(plastic_actual_part_dimension(lc, s), l, epsilon = 1e-12);
    }

    #[test]
    fn compensation_is_cavity_minus_part() {
        // Identité : Δ = Lc − L exactement.
        let (l, s) = (0.085_f64, 0.025);
        let delta = cavity_shrinkage_compensation(l, s);
        assert_relative_eq!(delta, cavity_dimension(l, s) - l, epsilon = 1e-12);
    }

    #[test]
    fn rate_inverts_cavity_dimension() {
        // Réciprocité : le taux déduit d'une empreinte redonne s.
        let (l, s) = (0.050_f64, 0.012);
        let lc = cavity_dimension(l, s);
        assert_relative_eq!(plastic_shrinkage_rate(lc, l), s, epsilon = 1e-12);
    }

    #[test]
    fn zero_shrinkage_keeps_dimensions() {
        // Cas limite s = 0 : empreinte = pièce, compensation nulle.
        let l = 0.030_f64;
        assert_relative_eq!(cavity_dimension(l, 0.0), l, epsilon = 1e-12);
        assert_relative_eq!(cavity_shrinkage_compensation(l, 0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn compensation_scales_linearly_with_dimension() {
        // Proportionnalité : Δ ∝ L à taux fixé.
        let s = 0.02_f64;
        let d1 = cavity_shrinkage_compensation(0.040, s);
        let d2 = cavity_shrinkage_compensation(0.120, s);
        assert_relative_eq!(d2 / d1, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_case_two_percent() {
        // Cas chiffré : pièce visée 100 mm, retrait 2 % → empreinte 102 mm.
        let l = 0.100_f64;
        let s = 0.02_f64;
        assert_relative_eq!(cavity_dimension(l, s), 0.102, epsilon = 1e-12);
        assert_relative_eq!(cavity_shrinkage_compensation(l, s), 0.002, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "1 + s")]
    fn shrinkage_rate_at_minus_one_panics() {
        // s = -1 annulerait l'empreinte : (1 + s) = 0.
        cavity_dimension(0.100, -1.0);
    }
}
