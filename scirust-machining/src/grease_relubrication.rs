//! **Regraissage des roulements** — règles empiriques usuelles (type SKF) pour
//! estimer la quantité de graisse, le facteur de vitesse `n·dm` et la réduction
//! de durée de vie de la graisse avec la température.
//!
//! ```text
//! quantité initiale        G   = 0.005 · D · B                (grammes)
//! diamètre moyen           dm  = (d + D) / 2                  (mm)
//! facteur de vitesse       ndm = dm · n                       (mm·tr/min)
//! réduction durée de vie   Lred = 2^(−(T − Tref)/15)          (—)
//! ```
//!
//! `D` diamètre extérieur du roulement (mm), `B` largeur du roulement (mm), `d`
//! diamètre d'alésage (mm), `dm` diamètre moyen (mm), `n` vitesse de rotation
//! (tr/min), `ndm` facteur de vitesse (mm·tr/min), `G` masse de graisse
//! (grammes), `T` température de service (°C), `Tref` température de référence
//! (°C), `Lred` facteur multiplicatif de durée de vie (—).
//!
//! **Convention** : dimensions en mm, vitesse en tr/min, températures en °C,
//! masse en grammes ; ces règles sont **dimensionnelles** (mm) et non SI par
//! commodité d'atelier. **Limite honnête** : ce sont des **règles empiriques
//! usuelles** (constructeurs de roulements) ; les constantes (0.005 g/mm², pas de
//! 15 °C de la loi d'Arrhenius simplifiée) et toutes les dimensions/températures
//! sont **fournies par le catalogue/l'appelant** — aucune valeur « par défaut »
//! n'est inventée. L'intervalle de regraissage réel dépend fortement de
//! l'environnement (contamination, charge, humidité, orientation) : ces valeurs
//! sont **indicatives**. Voir [`crate::bearings`] et [`crate::bearing_preload`].

/// Quantité de graisse pour un regraissage : `G = 0.005 · D · B` (grammes).
///
/// Règle empirique usuelle où `D` est le diamètre extérieur et `B` la largeur du
/// roulement, tous deux en mm ; le résultat est en grammes.
///
/// Panique si `outer_diameter_mm <= 0` ou `width_mm <= 0`.
pub fn grease_quantity_grams(outer_diameter_mm: f64, width_mm: f64) -> f64 {
    assert!(
        outer_diameter_mm > 0.0,
        "le diamètre extérieur D doit être strictement positif (mm)"
    );
    assert!(
        width_mm > 0.0,
        "la largeur B doit être strictement positive (mm)"
    );
    0.005_f64 * outer_diameter_mm * width_mm
}

/// Diamètre moyen d'un roulement : `dm = (d + D) / 2` (mm).
///
/// Moyenne du diamètre d'alésage `d` et du diamètre extérieur `D` (mm),
/// utilisée comme diamètre caractéristique pour le facteur de vitesse.
///
/// Panique si `bore_diameter_mm <= 0`, `outer_diameter_mm <= 0`, ou si
/// `outer_diameter_mm <= bore_diameter_mm`.
pub fn grease_mean_diameter_mm(bore_diameter_mm: f64, outer_diameter_mm: f64) -> f64 {
    assert!(
        bore_diameter_mm > 0.0,
        "le diamètre d'alésage d doit être strictement positif (mm)"
    );
    assert!(
        outer_diameter_mm > bore_diameter_mm,
        "le diamètre extérieur D doit être strictement supérieur à l'alésage d (mm)"
    );
    0.5_f64 * (bore_diameter_mm + outer_diameter_mm)
}

/// Facteur de vitesse : `ndm = dm · n` (mm·tr/min).
///
/// Produit du diamètre moyen `dm` (mm) par la vitesse de rotation `n`
/// (tr/min) ; grandeur de référence pour dimensionner l'intervalle de
/// regraissage et vérifier l'aptitude d'une graisse à une vitesse donnée.
///
/// Panique si `mean_diameter_mm <= 0` ou `speed_rpm < 0`.
pub fn grease_speed_factor_ndm(mean_diameter_mm: f64, speed_rpm: f64) -> f64 {
    assert!(
        mean_diameter_mm > 0.0,
        "le diamètre moyen dm doit être strictement positif (mm)"
    );
    assert!(
        speed_rpm >= 0.0,
        "la vitesse de rotation n doit être positive (tr/min)"
    );
    mean_diameter_mm * speed_rpm
}

/// Facteur de réduction de durée de vie de la graisse :
/// `Lred = 2^(−(T − Tref)/15)`.
///
/// La durée de vie utile de la graisse est approximativement **divisée par deux
/// tous les 15 °C** au-dessus de la température de référence (loi d'Arrhenius
/// simplifiée). Vaut `1` à `T = Tref`, `> 1` en dessous, `< 1` au-dessus. `T` et
/// `Tref` sont en °C ; le résultat est un facteur multiplicatif sans dimension.
///
/// Panique si `temperature_celsius` ou `reference_temperature` est non fini
/// (NaN ou infini).
pub fn grease_life_reduction_factor(temperature_celsius: f64, reference_temperature: f64) -> f64 {
    assert!(
        temperature_celsius.is_finite(),
        "la température de service T doit être finie (°C)"
    );
    assert!(
        reference_temperature.is_finite(),
        "la température de référence Tref doit être finie (°C)"
    );
    2.0_f64.powf(-(temperature_celsius - reference_temperature) / 15.0_f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Cas chiffré réaliste : roulement 6206 (D = 62 mm, B = 16 mm).
    /// G = 0.005 · 62 · 16 = 4.96 g.
    #[test]
    fn quantity_6206_realistic() {
        assert_relative_eq!(grease_quantity_grams(62.0, 16.0), 4.96, epsilon = 1e-12);
    }

    /// Proportionnalité : doubler la largeur double la quantité de graisse.
    #[test]
    fn quantity_proportional_to_width() {
        let g1 = grease_quantity_grams(62.0, 16.0);
        let g2 = grease_quantity_grams(62.0, 32.0);
        assert_relative_eq!(g2, 2.0 * g1, epsilon = 1e-12);
    }

    /// Cohérence dm/ndm : d = 30, D = 62 → dm = 46 mm ; à 3000 tr/min,
    /// ndm = 46 · 3000 = 138 000 mm·tr/min.
    #[test]
    fn mean_diameter_and_speed_factor() {
        let dm = grease_mean_diameter_mm(30.0, 62.0);
        assert_relative_eq!(dm, 46.0, epsilon = 1e-12);
        assert_relative_eq!(
            grease_speed_factor_ndm(dm, 3000.0),
            138_000.0,
            epsilon = 1e-9
        );
    }

    /// Cas limite : à T = Tref, la réduction vaut exactement 1.
    #[test]
    fn life_reduction_unity_at_reference() {
        assert_relative_eq!(
            grease_life_reduction_factor(70.0, 70.0),
            1.0,
            epsilon = 1e-12
        );
    }

    /// Loi de division par deux : +15 °C → 0.5, +30 °C → 0.25.
    #[test]
    fn life_reduction_halves_every_15c() {
        assert_relative_eq!(
            grease_life_reduction_factor(85.0, 70.0),
            0.5,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            grease_life_reduction_factor(100.0, 70.0),
            0.25,
            epsilon = 1e-12
        );
    }

    /// Identité de symétrie : Lred(Tref+ΔT) · Lred(Tref−ΔT) = 1.
    #[test]
    fn life_reduction_symmetric_identity() {
        let up = grease_life_reduction_factor(85.0, 70.0);
        let down = grease_life_reduction_factor(55.0, 70.0);
        assert_relative_eq!(up * down, 1.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "la largeur B doit être strictement positive")]
    fn quantity_rejects_nonpositive_width() {
        let _ = grease_quantity_grams(62.0, 0.0);
    }
}
