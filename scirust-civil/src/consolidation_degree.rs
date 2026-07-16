//! **Géotechnique — degré de consolidation dans le temps (Terzaghi)** :
//! facteur temps `Tv` et degré de consolidation **moyen** `U` d'une couche
//! d'argile en consolidation **unidimensionnelle**, avec les deux approximations
//! usuelles de la série de Terzaghi selon que `U < 0,6` ou `U > 0,6`, plus la
//! réciproque donnant le temps nécessaire pour atteindre un degré `U < 0,6`.
//!
//! ```text
//! facteur temps                 Tv = cv · t / Hdr²
//! degré moyen (U < 0,6)         U  = 2·√(Tv / π)              [Tv = (π/4)·U²]
//! degré moyen (U > 0,6)         U  = 1 − (8/π²)·exp(−π²·Tv/4)
//! temps pour un degré (U < 0,6) t  = (π/4)·U²·Hdr² / cv       (réciproque)
//! ```
//!
//! `Tv` facteur temps (sans dimension), `U` degré de consolidation moyen sur
//! l'épaisseur (sans dimension, entre `0` et `1`), `cv` =
//! `consolidation_coefficient` coefficient de consolidation (m²/s), `t` = `time`
//! temps écoulé (s), `Hdr` = `drainage_path` longueur du chemin de drainage (m).
//!
//! **Convention** : SI strict — **m, s** (avec `cv` en **m²/s**). Les temps
//! ressortent en **secondes**, les longueurs en **mètres** ; le facteur temps `Tv`
//! et le degré `U` sont **sans dimension**.
//!
//! **Limite honnête** : consolidation **unidimensionnelle** de Terzaghi, sol
//! **saturé** et **homogène**, drainage vertical. Le coefficient de consolidation
//! `cv` et la longueur de drainage `Hdr` (`Hdr = H` pour drainage **simple**,
//! `Hdr = H/2` pour drainage **double**) sont **fournis par l'appelant** d'après
//! les essais (œdomètre) ; aucune valeur « par défaut » n'est inventée. Les deux
//! approximations couvrent respectivement `U < 0,6` ([`consol_degree_low`]) et
//! `U > 0,6` ([`consol_degree_high`]) : c'est à l'appelant de choisir la branche
//! adaptée au degré attendu. Le degré `U` est **moyen sur l'épaisseur** de la
//! couche (et non local). Les éventuelles résistances caractéristiques du sol
//! **et** les coefficients partiels de sécurité (`γM`…) relèvent de l'appelant
//! selon l'Eurocode 7 et son Annexe Nationale. Ce module est **distinct** du
//! calcul de **tassement** (amplitude) traité par le module `settlement`.

use core::f64::consts::PI;

/// Facteur temps de la théorie de Terzaghi `Tv = cv · t / Hdr²`
/// (sans dimension), avec `cv` en m²/s, `t` en s et `Hdr` en m.
///
/// Panique si `consolidation_coefficient < 0`, si `time < 0` ou si
/// `drainage_path <= 0` (division par zéro).
pub fn consol_time_factor(consolidation_coefficient: f64, time: f64, drainage_path: f64) -> f64 {
    assert!(
        consolidation_coefficient >= 0.0,
        "le coefficient de consolidation cv doit être ≥ 0"
    );
    assert!(time >= 0.0, "le temps t doit être ≥ 0");
    assert!(
        drainage_path > 0.0,
        "le chemin de drainage Hdr doit être strictement positif"
    );
    consolidation_coefficient * time / (drainage_path * drainage_path)
}

/// Degré de consolidation **moyen** `U = 2·√(Tv / π)` (sans dimension), valable
/// pour `U < 0,6` (approximation `Tv = (π/4)·U²` de la série de Terzaghi).
///
/// Panique si `time_factor < 0` (racine d'un nombre négatif).
pub fn consol_degree_low(time_factor: f64) -> f64 {
    assert!(
        time_factor >= 0.0,
        "le facteur temps Tv doit être ≥ 0 pour l'approximation U < 0,6"
    );
    2.0 * (time_factor / PI).sqrt()
}

/// Degré de consolidation **moyen** `U = 1 − (8/π²)·exp(−π²·Tv/4)`
/// (sans dimension), valable pour `U > 0,6` (premier terme de la série de
/// Terzaghi).
///
/// Panique si `time_factor < 0`.
pub fn consol_degree_high(time_factor: f64) -> f64 {
    assert!(
        time_factor >= 0.0,
        "le facteur temps Tv doit être ≥ 0 pour l'approximation U > 0,6"
    );
    1.0 - (8.0 / (PI * PI)) * (-PI * PI * time_factor / 4.0).exp()
}

/// Temps nécessaire pour atteindre un degré de consolidation moyen `degree`
/// (`U < 0,6`) `t = (π/4)·U²·Hdr² / cv` (s), réciproque de
/// [`consol_degree_low`] via [`consol_time_factor`], avec `Hdr` en m et `cv`
/// en m²/s.
///
/// Panique si `degree` n'est pas dans `[0, 1]`, si `drainage_path < 0` ou si
/// `consolidation_coefficient <= 0` (division par zéro).
pub fn consol_time_from_degree_low(
    degree: f64,
    consolidation_coefficient: f64,
    drainage_path: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&degree),
        "le degré de consolidation U doit être compris entre 0 et 1"
    );
    assert!(
        drainage_path >= 0.0,
        "le chemin de drainage Hdr doit être ≥ 0"
    );
    assert!(
        consolidation_coefficient > 0.0,
        "le coefficient de consolidation cv doit être strictement positif"
    );
    (PI / 4.0) * degree * degree * drainage_path * drainage_path / consolidation_coefficient
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn time_factor_scales_with_time_and_inverse_square_of_drainage() {
        // Tv est linéaire en t et varie en 1/Hdr² : cv = 3e-8 m²/s, t = 1 an
        // (31 536 000 s), Hdr = 2 m (double drainage d'une couche de 4 m).
        //   Tv = 3e-8 · 31 536 000 / 2² = 0,94608 / 4 = 0,23652
        let cv = 3.0e-8_f64;
        let one_year = 31_536_000.0_f64;
        let tv = consol_time_factor(cv, one_year, 2.0);
        assert_relative_eq!(tv, 0.236_52, max_relative = 1e-9);
        // Deux fois le temps ⇒ deux fois Tv.
        let tv2 = consol_time_factor(cv, 2.0 * one_year, 2.0);
        assert_relative_eq!(tv2, 2.0 * tv, max_relative = 1e-12);
        // Chemin de drainage divisé par 2 ⇒ Tv multiplié par 4.
        let tv_half = consol_time_factor(cv, one_year, 1.0);
        assert_relative_eq!(tv_half, 4.0 * tv, max_relative = 1e-12);
    }

    #[test]
    fn degree_low_matches_hand_computation() {
        // Cas chiffré : Tv = 0,2.
        //   U = 2·√(0,2 / π) = 2·√(0,063661977) = 2·0,252313252
        //     ≈ 0,504626504
        let u = consol_degree_low(0.2);
        assert_relative_eq!(u, 0.504_626_504, max_relative = 1e-6);
    }

    #[test]
    fn degree_high_matches_hand_computation() {
        // Cas chiffré : Tv = 0,5.
        //   U = 1 − (8/π²)·exp(−π²·0,5/4)
        //     = 1 − 0,810569469·exp(−1,233700550)
        //     = 1 − 0,810569469·0,291218
        //     ≈ 0,763951687
        let u = consol_degree_high(0.5);
        assert_relative_eq!(u, 0.763_951_687, max_relative = 1e-6);
    }

    #[test]
    fn degree_low_is_reciprocal_of_time_from_degree() {
        // Réciprocité : partir de U, calculer t, en déduire Tv puis U à nouveau.
        // Avec U = 0,5, cv = 3e-8 m²/s, Hdr = 2 m.
        let (u0, cv, hdr) = (0.5_f64, 3.0e-8_f64, 2.0_f64);
        let t = consol_time_from_degree_low(u0, cv, hdr);
        let tv = consol_time_factor(cv, t, hdr);
        let u1 = consol_degree_low(tv);
        assert_relative_eq!(u1, u0, max_relative = 1e-12);
    }

    #[test]
    fn time_from_degree_scales_with_square_of_degree() {
        // t est quadratique en U : doubler U quadruple le temps requis.
        let base = consol_time_from_degree_low(0.25, 4.0e-8, 3.0);
        let double = consol_time_from_degree_low(0.50, 4.0e-8, 3.0);
        assert_relative_eq!(double, 4.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn both_approximations_vanish_and_start_near_agreement() {
        // Cas limites : Tv = 0 ⇒ U = 0 pour l'approximation basse ;
        // les deux branches encadrent la transition autour de U ≈ 0,6.
        let u_start = consol_degree_low(0.0);
        assert_relative_eq!(u_start, 0.0, epsilon = 1e-12);
        // Au voisinage de la transition (Tv ≈ 0,283), les deux formules donnent
        // des degrés proches (≈ 0,6), écart inférieur à quelques centièmes.
        let tv_mid = 0.283_f64;
        let low = consol_degree_low(tv_mid);
        let high = consol_degree_high(tv_mid);
        assert_relative_eq!(low, high, max_relative = 1e-1);
    }

    #[test]
    #[should_panic(expected = "le coefficient de consolidation cv doit être strictement positif")]
    fn time_from_degree_rejects_zero_coefficient() {
        // cv = 0 interdit : division par zéro.
        consol_time_from_degree_low(0.5, 0.0, 2.0);
    }
}
