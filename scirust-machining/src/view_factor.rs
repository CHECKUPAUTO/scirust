//! Facteurs de forme en rayonnement thermique — réciprocité, relation de
//! fermeture d'une enceinte et échange net entre corps noirs.
//!
//! ```text
//! réciprocité       A1·F12 = A2·F21        ⇒  F21 = F12·A1/A2
//! sommation         Σj F1j = 1             (enceinte fermée, N surfaces)
//! plans ∥ infinis   F12 = 1               (tout ce qui quitte 1 atteint 2)
//! échange net       Q12 = A1·F12·(Eb1 − Eb2)   (corps noirs)
//! ```
//!
//! `Fij` facteur de forme de la surface `i` vers `j` (sans dimension, `0`–`1`),
//! `A` aire (m²), `Eb` émittance de corps noir (W/m², typiquement `σ·T⁴`),
//! `Q12` puissance nette échangée de `1` vers `2` (W).
//!
//! **Convention** : unités SI. **Limite honnête** : surfaces grises/noires
//! diffuses ; les facteurs de forme géométriques `Fij` sont **fournis** par
//! l'appelant (sauf les cas triviaux comme deux plans parallèles infinis), car
//! ils dépendent de la géométrie exacte de l'enceinte. Les émittances `Eb` (et
//! donc les températures et émissivités qui les produisent) sont **fournies**.
//! Ce module complète [`crate::radiation`] (loi de Stefan-Boltzmann).

/// Facteur de forme réciproque `F21 = F12·A1/A2` déduit de la relation de
/// réciprocité `A1·F12 = A2·F21`.
///
/// Panique si `view_factor_1_to_2` est hors `[0, 1]` ou si une aire n'est pas
/// strictement positive.
pub fn view_factor_reciprocity(view_factor_1_to_2: f64, area_1: f64, area_2: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&view_factor_1_to_2),
        "le facteur de forme doit être dans [0, 1]"
    );
    assert!(area_1 > 0.0, "l'aire de la surface 1 doit être positive");
    assert!(area_2 > 0.0, "l'aire de la surface 2 doit être positive");
    view_factor_1_to_2 * area_1 / area_2
}

/// Facteur de forme manquant d'une enceinte fermée par la relation de sommation
/// `F1,last = 1 − Σ others`.
///
/// Panique si un facteur de `others` est hors `[0, 1]` ou si leur somme excède
/// `1` (enceinte physiquement impossible).
pub fn view_factor_summation_last(others: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for &f in others
    {
        assert!(
            (0.0..=1.0).contains(&f),
            "chaque facteur de forme doit être dans [0, 1]"
        );
        sum += f;
    }
    assert!(
        sum <= 1.0 + 1e-12,
        "la somme des facteurs de forme ne peut excéder 1"
    );
    1.0 - sum
}

/// Facteur de forme entre deux plans parallèles infinis en vis-à-vis : `F12 = 1`
/// (tout le rayonnement quittant une surface atteint l'autre).
///
/// Ne panique jamais.
pub fn view_factor_infinite_parallel_plates() -> f64 {
    1.0
}

/// Puissance nette échangée de la surface 1 vers la surface 2, corps noirs :
/// `Q12 = A1·F12·(Eb1 − Eb2)` (W). Positive si `Eb1 > Eb2`.
///
/// Panique si `view_factor_1_to_2` est hors `[0, 1]` ou si `area_1` n'est pas
/// strictement positive.
pub fn view_factor_net_exchange(
    view_factor_1_to_2: f64,
    area_1: f64,
    emissive_power_1: f64,
    emissive_power_2: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&view_factor_1_to_2),
        "le facteur de forme doit être dans [0, 1]"
    );
    assert!(area_1 > 0.0, "l'aire de la surface 1 doit être positive");
    area_1 * view_factor_1_to_2 * (emissive_power_1 - emissive_power_2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reciprocity_preserves_a_times_f() {
        // Identité de réciprocité : A1·F12 doit égaler A2·F21.
        let (f12, a1, a2) = (0.6_f64, 2.0_f64, 4.0_f64);
        let f21 = view_factor_reciprocity(f12, a1, a2);
        assert_relative_eq!(a1 * f12, a2 * f21, epsilon = 1e-12);
        // Valeur attendue : 0,6·2/4 = 0,3.
        assert_relative_eq!(f21, 0.3, epsilon = 1e-12);
    }

    #[test]
    fn reciprocity_is_involutive() {
        // Appliquer deux fois la réciprocité redonne le facteur initial.
        let (f12, a1, a2) = (0.35_f64, 1.5_f64, 5.0_f64);
        let f21 = view_factor_reciprocity(f12, a1, a2);
        let f12_back = view_factor_reciprocity(f21, a2, a1);
        assert_relative_eq!(f12_back, f12, epsilon = 1e-12);
    }

    #[test]
    fn summation_closes_the_enclosure() {
        // Enceinte à 4 surfaces : le facteur manquant complète la somme à 1.
        let others = [0.3_f64, 0.2_f64, 0.1_f64];
        let last = view_factor_summation_last(&others);
        assert_relative_eq!(last, 0.4, epsilon = 1e-12);
        let total: f64 = others.iter().sum::<f64>() + last;
        assert_relative_eq!(total, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn infinite_plates_capture_everything() {
        assert_relative_eq!(view_factor_infinite_parallel_plates(), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn net_exchange_is_antisymmetric() {
        // Avec la réciprocité, Q21 = A2·F21·(Eb2−Eb1) = −Q12.
        let (f12, a1, a2) = (0.5_f64, 3.0_f64, 6.0_f64);
        let (eb1, eb2) = (56_704.0_f64, 3_544.0_f64);
        let f21 = view_factor_reciprocity(f12, a1, a2);
        let q12 = view_factor_net_exchange(f12, a1, eb1, eb2);
        let q21 = view_factor_net_exchange(f21, a2, eb2, eb1);
        assert_relative_eq!(q12, -q21, epsilon = 1e-9);
        // Cas chiffré : 3·0,5·(56704−3544) = 1,5·53160 = 79 740 W.
        assert_relative_eq!(q12, 79_740.0, epsilon = 1e-9);
    }

    #[test]
    fn net_exchange_vanishes_at_equal_emissive_power() {
        // Émittances égales → aucun échange net.
        assert_relative_eq!(
            view_factor_net_exchange(0.7, 2.0, 12_000.0, 12_000.0),
            0.0,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "l'aire de la surface 2 doit être positive")]
    fn reciprocity_rejects_zero_area() {
        let _ = view_factor_reciprocity(0.5, 1.0, 0.0);
    }
}
