//! Titre de vapeur — propriétés d'un mélange liquide-vapeur saturé à
//! l'équilibre, obtenues par interpolation linéaire entre les valeurs de
//! saturation `f` (liquide) et `g` (vapeur) au moyen du titre `x`.
//!
//! ```text
//! propriété du mélange   y = y_f + x·(y_g − y_f)          (v, h, s, u ; même unité que y_f, y_g)
//! titre depuis enthalpie  x = (h − h_f) / (h_g − h_f)      (sans dimension)
//! fraction d'humidité     w = 1 − x                        (sans dimension)
//! enthalpie du mélange    h = h_f + x·h_fg                 (J/kg, avec h_fg = h_g − h_f)
//! ```
//!
//! `y_f`, `y_g` valeurs de saturation d'une propriété massique (volume massique
//! v en m³/kg, enthalpie h en J/kg, entropie s en J/(kg·K), énergie interne u
//! en J/kg) ; `x` titre de vapeur (fraction massique de vapeur, sans dimension,
//! ∈ [0, 1]) ; `w` fraction d'humidité (sans dimension) ; `h_fg = h_g − h_f`
//! chaleur latente de vaporisation (J/kg).
//!
//! **Limite honnête** : mélange saturé à l'**équilibre** (liquide et vapeur à
//! la même pression et température de saturation). Les **valeurs de saturation
//! `f` et `g`** sont **lues dans les tables de vapeur et fournies par
//! l'appelant** ; ce module ne calcule **pas** les propriétés de saturation
//! elles-mêmes ni la relation pression-température de saturation. Le titre `x`
//! est supposé dans `[0, 1]` ; en dehors, l'état n'est plus un mélange saturé.

/// Propriété massique du mélange `y = y_f + x·(y_g − y_f)` (même unité que les
/// valeurs de saturation).
///
/// `saturated_liquid_value` = y_f valeur à saturation du liquide,
/// `saturated_vapor_value` = y_g valeur à saturation de la vapeur,
/// `quality` = x titre de vapeur (sans dimension). S'applique à v, h, s ou u.
///
/// Panique si `quality` n'est pas dans `[0, 1]`.
pub fn steam_property_from_quality(
    saturated_liquid_value: f64,
    saturated_vapor_value: f64,
    quality: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&quality),
        "le titre x doit être dans [0, 1]"
    );
    saturated_liquid_value + quality * (saturated_vapor_value - saturated_liquid_value)
}

/// Titre de vapeur depuis l'enthalpie `x = (h − h_f) / (h_g − h_f)`
/// (sans dimension).
///
/// `mixture_enthalpy` = h enthalpie massique du mélange (J/kg),
/// `liquid_enthalpy` = h_f enthalpie du liquide saturé (J/kg),
/// `vapor_enthalpy` = h_g enthalpie de la vapeur saturée (J/kg).
///
/// Panique si `vapor_enthalpy <= liquid_enthalpy` (dénominateur h_fg non
/// strictement positif).
pub fn steam_quality_from_enthalpy(
    mixture_enthalpy: f64,
    liquid_enthalpy: f64,
    vapor_enthalpy: f64,
) -> f64 {
    assert!(
        vapor_enthalpy > liquid_enthalpy,
        "l'enthalpie de vapeur h_g doit dépasser l'enthalpie de liquide h_f"
    );
    (mixture_enthalpy - liquid_enthalpy) / (vapor_enthalpy - liquid_enthalpy)
}

/// Fraction d'humidité `w = 1 − x` (sans dimension).
///
/// `quality` = x titre de vapeur (sans dimension) ; le résultat est la fraction
/// massique de liquide dans le mélange saturé.
///
/// Panique si `quality` n'est pas dans `[0, 1]`.
pub fn steam_wetness_fraction(quality: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&quality),
        "le titre x doit être dans [0, 1]"
    );
    1.0 - quality
}

/// Enthalpie du mélange `h = h_f + x·h_fg` (J/kg).
///
/// `liquid_enthalpy` = h_f enthalpie du liquide saturé (J/kg),
/// `latent_heat` = h_fg chaleur latente de vaporisation (J/kg),
/// `quality` = x titre de vapeur (sans dimension).
///
/// Panique si `latent_heat < 0` ou si `quality` n'est pas dans `[0, 1]`.
pub fn steam_mixture_enthalpy(liquid_enthalpy: f64, latent_heat: f64, quality: f64) -> f64 {
    assert!(
        latent_heat >= 0.0,
        "la chaleur latente h_fg doit être positive ou nulle"
    );
    assert!(
        (0.0..=1.0).contains(&quality),
        "le titre x doit être dans [0, 1]"
    );
    liquid_enthalpy + quality * latent_heat
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Eau saturée à 100 °C (tables) : h_f = 419.04 kJ/kg, h_fg = 2257.0 kJ/kg,
    // donc h_g = h_f + h_fg = 2676.04 kJ/kg. En unités SI (J/kg) ci-dessous.
    const HF: f64 = 419_040.0;
    const HFG: f64 = 2_257_000.0;
    const HG: f64 = HF + HFG; // 2 676 040 J/kg

    // Cas chiffré : x = 0,9 → h = 419040 + 0,9·2257000 = 2 450 340 J/kg.
    #[test]
    fn mixture_enthalpy_reference_case() {
        let x = 0.9_f64;
        assert_relative_eq!(
            steam_mixture_enthalpy(HF, HFG, x),
            2_450_340.0,
            epsilon = 1e-6
        );
    }

    // Réciprocité : le titre reconstruit depuis l'enthalpie du mélange redonne x.
    #[test]
    fn quality_from_enthalpy_inverts_mixture_enthalpy() {
        let x = 0.9_f64;
        let h = steam_mixture_enthalpy(HF, HFG, x);
        assert_relative_eq!(steam_quality_from_enthalpy(h, HF, HG), x, epsilon = 1e-12);
    }

    // Cohérence : l'interpolation générale avec f = h_f, g = h_g doit coïncider
    // avec l'enthalpie du mélange h_f + x·h_fg.
    #[test]
    fn property_from_quality_matches_mixture_enthalpy() {
        let x = 0.9_f64;
        assert_relative_eq!(
            steam_property_from_quality(HF, HG, x),
            steam_mixture_enthalpy(HF, HFG, x),
            epsilon = 1e-6
        );
    }

    // Cas limites : x = 0 donne le liquide saturé, x = 1 la vapeur saturée.
    #[test]
    fn saturation_endpoints() {
        let (vf, vg) = (0.001_f64, 1.694_f64); // volume massique m³/kg à 100 °C
        assert_relative_eq!(
            steam_property_from_quality(vf, vg, 0.0),
            vf,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            steam_property_from_quality(vf, vg, 1.0),
            vg,
            epsilon = 1e-12
        );
    }

    // Titre et fraction d'humidité sont complémentaires : x + w = 1.
    #[test]
    fn quality_and_wetness_sum_to_one() {
        let x = 0.72_f64;
        assert_relative_eq!(x + steam_wetness_fraction(x), 1.0, epsilon = 1e-12);
    }

    // Volume massique du mélange à mi-titre : v = vf + 0,5·(vg − vf).
    // vf = 0,001, vg = 1,694 → v = 0,001 + 0,5·1,693 = 0,8475 m³/kg.
    #[test]
    fn specific_volume_at_half_quality() {
        let (vf, vg) = (0.001_f64, 1.694_f64);
        assert_relative_eq!(
            steam_property_from_quality(vf, vg, 0.5),
            0.8475,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "le titre x doit être dans [0, 1]")]
    fn wetness_panics_on_quality_above_one() {
        steam_wetness_fraction(1.2);
    }
}
