//! **Pouvoir calorifique d'un combustible** — estimation du pouvoir calorifique
//! supérieur (PCS) par la formule de Dulong à partir de l'analyse élémentaire,
//! passage au pouvoir calorifique inférieur (PCI) et énergie libérée par une masse.
//!
//! ```text
//! PCS (Dulong)   HHV = 33.8·C + 144.2·(H − O/8) + 9.4·S      (MJ/kg)
//! PCI            LHV = HHV − L·(9·H + w)                      (MJ/kg)
//! énergie        Q   = m·LHV                                 (MJ)
//! ```
//!
//! `C`, `H`, `O`, `S` fractions massiques (sans unité) de carbone, hydrogène,
//! oxygène et soufre du combustible ; `HHV` pouvoir calorifique supérieur (MJ/kg) ;
//! `L` chaleur latente de vaporisation de l'eau (MJ/kg) ; `w` fraction massique
//! d'humidité (eau libre) du combustible ; `LHV` pouvoir calorifique inférieur
//! (MJ/kg) ; `m` masse de combustible (kg) ; `Q` énergie dégagée (MJ). Le terme
//! `9·H` traduit l'eau formée par combustion de l'hydrogène (9 kg d'eau par kg d'H₂).
//!
//! **Convention** : SI (énergies massiques en MJ/kg, masses en kg, énergie en MJ).
//! **Limite honnête** : la formule de Dulong est une **approximation** par analyse
//! élémentaire (elle ignore les liaisons chimiques réelles et surestime souvent les
//! combustibles riches en oxygène) ; elle **ne remplace pas** une mesure au
//! calorimètre. Les fractions massiques (`C`, `H`, `O`, `S`, `w`) et la chaleur
//! latente de vaporisation de l'eau (`L ≈ 2.442` MJ/kg à 25 °C) sont **fournies par
//! l'appelant** ; aucune valeur matériau ni condition de procédé « par défaut »
//! n'est inventée.

/// Coefficient carbone de la formule de Dulong (MJ/kg par unité de fraction).
const DULONG_CARBON: f64 = 33.8;
/// Coefficient hydrogène (net d'oxygène) de la formule de Dulong (MJ/kg).
const DULONG_HYDROGEN: f64 = 144.2;
/// Coefficient soufre de la formule de Dulong (MJ/kg).
const DULONG_SULFUR: f64 = 9.4;
/// Masse d'eau formée par unité de masse d'hydrogène brûlé (H₂ + ½O₂ → H₂O).
const WATER_PER_HYDROGEN: f64 = 9.0;

/// Pouvoir calorifique supérieur (PCS) par la formule de Dulong
/// `HHV = 33.8·C + 144.2·(H − O/8) + 9.4·S` (MJ/kg, fractions massiques).
///
/// Panique si l'une des fractions n'est pas dans `[0, 1]`.
pub fn calorific_dulong_hhv(
    carbon_fraction: f64,
    hydrogen_fraction: f64,
    oxygen_fraction: f64,
    sulfur_fraction: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&carbon_fraction),
        "la fraction massique de carbone doit être dans [0, 1]"
    );
    assert!(
        (0.0..=1.0).contains(&hydrogen_fraction),
        "la fraction massique d'hydrogène doit être dans [0, 1]"
    );
    assert!(
        (0.0..=1.0).contains(&oxygen_fraction),
        "la fraction massique d'oxygène doit être dans [0, 1]"
    );
    assert!(
        (0.0..=1.0).contains(&sulfur_fraction),
        "la fraction massique de soufre doit être dans [0, 1]"
    );
    DULONG_CARBON * carbon_fraction
        + DULONG_HYDROGEN * (hydrogen_fraction - oxygen_fraction / 8.0)
        + DULONG_SULFUR * sulfur_fraction
}

/// Pouvoir calorifique inférieur (PCI) déduit du PCS
/// `LHV = HHV − L·(9·H + w)` (MJ/kg), où `L` est la chaleur latente de
/// vaporisation de l'eau et `w` la fraction d'humidité.
///
/// Panique si `hhv < 0`, si `hydrogen_fraction` ou `moisture_fraction` n'est pas
/// dans `[0, 1]`, ou si `latent_heat < 0`.
pub fn calorific_lhv_from_hhv(
    hhv: f64,
    hydrogen_fraction: f64,
    moisture_fraction: f64,
    latent_heat: f64,
) -> f64 {
    assert!(hhv >= 0.0, "le PCS (HHV) doit être positif");
    assert!(
        (0.0..=1.0).contains(&hydrogen_fraction),
        "la fraction massique d'hydrogène doit être dans [0, 1]"
    );
    assert!(
        (0.0..=1.0).contains(&moisture_fraction),
        "la fraction massique d'humidité doit être dans [0, 1]"
    );
    assert!(
        latent_heat >= 0.0,
        "la chaleur latente de vaporisation doit être positive"
    );
    hhv - latent_heat * (WATER_PER_HYDROGEN * hydrogen_fraction + moisture_fraction)
}

/// Énergie dégagée par la combustion d'une masse de combustible `Q = m·LHV` (MJ).
///
/// Panique si `mass < 0` ou `lower_heating_value < 0`.
pub fn calorific_fuel_energy(mass: f64, lower_heating_value: f64) -> f64 {
    assert!(mass >= 0.0, "la masse de combustible doit être positive");
    assert!(lower_heating_value >= 0.0, "le PCI (LHV) doit être positif");
    mass * lower_heating_value
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dulong_hydrogen_term_vanishes_when_h_equals_o_over_8() {
        // Si H = O/8 et S = 0, le terme hydrogène s'annule : HHV = 33.8·C.
        let hhv = calorific_dulong_hhv(0.60, 0.01, 0.08, 0.0);
        assert_relative_eq!(hhv, 33.8 * 0.60, epsilon = 1e-12);
    }

    #[test]
    fn dulong_is_linear_in_each_element() {
        // La formule est affine : HHV(2C,2H,2O,2S) − HHV(0) = 2·[HHV(C,H,O,S) − HHV(0)].
        let base = calorific_dulong_hhv(0.0, 0.0, 0.0, 0.0);
        assert_relative_eq!(base, 0.0, epsilon = 1e-12);
        let single = calorific_dulong_hhv(0.30, 0.02, 0.04, 0.005);
        let double = calorific_dulong_hhv(0.60, 0.04, 0.08, 0.010);
        assert_relative_eq!(double, 2.0 * single, epsilon = 1e-12);
    }

    #[test]
    fn dulong_realistic_bituminous_coal() {
        // Charbon bitumineux : C=0.75, H=0.05, O=0.08, S=0.01.
        // 33.8·0.75 + 144.2·(0.05 − 0.01) + 9.4·0.01
        // = 25.35 + 5.768 + 0.094 = 31.212 MJ/kg.
        let hhv = calorific_dulong_hhv(0.75, 0.05, 0.08, 0.01);
        assert_relative_eq!(hhv, 31.212, epsilon = 1e-9);
    }

    #[test]
    fn lhv_equals_hhv_without_hydrogen_or_moisture() {
        // Sans hydrogène ni humidité, aucune eau à vaporiser : LHV = HHV.
        let lhv = calorific_lhv_from_hhv(30.0, 0.0, 0.0, 2.442);
        assert_relative_eq!(lhv, 30.0, epsilon = 1e-12);
    }

    #[test]
    fn lhv_realistic_from_dulong() {
        // PCS = 31.212 MJ/kg, H=0.05, humidité w=0.02, L=2.442 MJ/kg.
        // LHV = 31.212 − 2.442·(9·0.05 + 0.02) = 31.212 − 2.442·0.47 = 30.06426 MJ/kg.
        let lhv = calorific_lhv_from_hhv(31.212, 0.05, 0.02, 2.442);
        assert_relative_eq!(lhv, 30.064_26, epsilon = 1e-9);
    }

    #[test]
    fn fuel_energy_is_proportional_to_mass() {
        // Q = m·LHV : doubler la masse double l'énergie ; cas chiffré 10 kg × 30.06426.
        let q1 = calorific_fuel_energy(10.0, 30.064_26);
        let q2 = calorific_fuel_energy(20.0, 30.064_26);
        assert_relative_eq!(q2, 2.0 * q1, epsilon = 1e-9);
        assert_relative_eq!(q1, 300.642_6, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "fraction massique de carbone")]
    fn dulong_rejects_out_of_range_fraction() {
        calorific_dulong_hhv(1.5, 0.05, 0.08, 0.01);
    }
}
