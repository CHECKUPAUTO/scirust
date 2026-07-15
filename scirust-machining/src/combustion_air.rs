//! Air de combustion stœchiométrique — **bilan massique élémentaire** d'un
//! combustible, air réel avec excès, et estimation de l'excès d'air à partir de
//! l'oxygène résiduel mesuré dans les fumées.
//!
//! ```text
//! rapport air/combustible stœchiométrique (fractions massiques)
//!   AFR_st = 11,53·C + 34,34·H + 4,29·S − 4,32·O      [kg air / kg combustible]
//!
//! air réel (avec excès d'air)
//!   AFR = AFR_st · (1 + e)
//!
//! excès d'air depuis l'O2 des fumées (% volumique, base sèche)
//!   e = O2 / (20,9 − O2)
//! ```
//!
//! `AFR_st`, `AFR` masse d'air par masse de combustible (`kg/kg`, adimensionnel) ;
//! `C`, `H`, `S`, `O` fractions **massiques** des éléments dans le combustible
//! (adimensionnelles, `0…1`) ; `e` excès d'air relatif (adimensionnel, p. ex.
//! `0,20` pour 20 % d'air en excès) ; `O2` teneur en oxygène des fumées en
//! **pour-cent volumique** sur base sèche (`%`) ; `20,9` teneur volumique en O2
//! de l'air ambiant (`%`).
//!
//! **Convention** : unités SI cohérentes, fractions massiques adimensionnelles,
//! rapports air/combustible en `kg/kg`. La formule massique standard suppose que
//! l'air apporte **23,2 % d'O2 en masse** et que le carbone, l'hydrogène et le
//! soufre s'oxydent complètement (combustion complète) ; le terme `−4,32·O`
//! crédite l'oxygène déjà présent dans le combustible. **Limite honnête** : la
//! **composition élémentaire** `C`, `H`, `S`, `O` provient de l'**analyse
//! ultime du combustible fournie par l'appelant** — aucune valeur « par défaut »
//! n'est inventée. Le résultat suppose une combustion complète et n'intègre ni
//! les imbrûlés, ni l'humidité de l'air, ni la dissociation à haute température.

/// Teneur volumique en oxygène de l'air ambiant sec, en **pour-cent** (`%`).
///
/// Référence de la relation excès d'air ↔ O2 des fumées : lorsque l'O2 mesuré
/// tend vers cette valeur, l'excès d'air tend vers l'infini (pas de combustion).
pub const COMBUSTION_AMBIENT_O2_VOLUME_PERCENT: f64 = 20.9;

/// Rapport air/combustible stœchiométrique par bilan massique élémentaire :
/// `AFR_st = 11,53·C + 34,34·H + 4,29·S − 4,32·O`.
///
/// Arguments : fractions **massiques** (adimensionnelles, `0…1`) de carbone
/// `carbon_mass_fraction`, d'hydrogène `hydrogen_mass_fraction`, de soufre
/// `sulfur_mass_fraction` et d'oxygène `oxygen_mass_fraction` du combustible.
/// Résultat en `kg` d'air sec par `kg` de combustible (adimensionnel).
///
/// Panique si une fraction massique est non finie ou hors de l'intervalle
/// `[0, 1]`.
pub fn combustion_stoichiometric_air_fuel_ratio(
    carbon_mass_fraction: f64,
    hydrogen_mass_fraction: f64,
    sulfur_mass_fraction: f64,
    oxygen_mass_fraction: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&carbon_mass_fraction),
        "la fraction massique de carbone doit être dans [0, 1]"
    );
    assert!(
        (0.0..=1.0).contains(&hydrogen_mass_fraction),
        "la fraction massique d'hydrogène doit être dans [0, 1]"
    );
    assert!(
        (0.0..=1.0).contains(&sulfur_mass_fraction),
        "la fraction massique de soufre doit être dans [0, 1]"
    );
    assert!(
        (0.0..=1.0).contains(&oxygen_mass_fraction),
        "la fraction massique d'oxygène doit être dans [0, 1]"
    );
    11.53 * carbon_mass_fraction + 34.34 * hydrogen_mass_fraction + 4.29 * sulfur_mass_fraction
        - 4.32 * oxygen_mass_fraction
}

/// Air réel avec excès d'air : `AFR = AFR_st · (1 + e)`.
///
/// `stoichiometric_air` rapport air/combustible stœchiométrique (`kg/kg`),
/// `excess_air_ratio` excès d'air relatif (adimensionnel, `≥ 0`). Résultat en
/// `kg` d'air par `kg` de combustible.
///
/// Panique si `stoichiometric_air` est négatif ou non fini, ou si
/// `excess_air_ratio` est négatif ou non fini.
pub fn combustion_actual_air(stoichiometric_air: f64, excess_air_ratio: f64) -> f64 {
    assert!(
        stoichiometric_air.is_finite() && stoichiometric_air >= 0.0,
        "l'air stœchiométrique doit être positif et fini"
    );
    assert!(
        excess_air_ratio.is_finite() && excess_air_ratio >= 0.0,
        "l'excès d'air doit être positif et fini"
    );
    stoichiometric_air * (1.0 + excess_air_ratio)
}

/// Excès d'air relatif déduit de l'oxygène des fumées : `e = O2 / (20,9 − O2)`.
///
/// `measured_oxygen_percent` teneur en O2 des fumées en **pour-cent volumique**
/// sur base sèche (`%`), avec `0 ≤ O2 < 20,9`. Résultat adimensionnel (p. ex.
/// `0,20` pour 20 % d'air en excès).
///
/// Panique si `measured_oxygen_percent` est non fini, négatif, ou supérieur ou
/// égal à la teneur en O2 de l'air ambiant (division impossible : pas de
/// combustion).
pub fn combustion_excess_air_from_o2(measured_oxygen_percent: f64) -> f64 {
    assert!(
        measured_oxygen_percent.is_finite(),
        "la teneur en O2 des fumées doit être un nombre fini"
    );
    assert!(
        measured_oxygen_percent >= 0.0,
        "la teneur en O2 des fumées doit être positive"
    );
    assert!(
        measured_oxygen_percent < COMBUSTION_AMBIENT_O2_VOLUME_PERCENT,
        "la teneur en O2 des fumées doit être inférieure à celle de l'air ambiant"
    );
    measured_oxygen_percent / (COMBUSTION_AMBIENT_O2_VOLUME_PERCENT - measured_oxygen_percent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pure_carbon_matches_its_coefficient() {
        // Un combustible de carbone pur (C = 1) rend exactement le coefficient
        // massique du carbone : 11,53 kg air / kg combustible.
        assert_relative_eq!(
            combustion_stoichiometric_air_fuel_ratio(1.0, 0.0, 0.0, 0.0),
            11.53,
            epsilon = 1e-12
        );
    }

    #[test]
    fn oxygen_in_fuel_reduces_air_demand() {
        // L'oxygène du combustible est crédité : le terme −4,32·O retranche
        // exactement 4,32 kg air par unité de fraction d'oxygène.
        let sans_o = combustion_stoichiometric_air_fuel_ratio(0.80, 0.05, 0.0, 0.0);
        let avec_o = combustion_stoichiometric_air_fuel_ratio(0.80, 0.05, 0.0, 0.10);
        assert_relative_eq!(sans_o - avec_o, 4.32 * 0.10, epsilon = 1e-12);
    }

    #[test]
    fn zero_excess_returns_stoichiometric() {
        // Cas limite : sans excès d'air, l'air réel se réduit à l'air
        // stœchiométrique (identité).
        let afr_st = 14.2644;
        assert_relative_eq!(combustion_actual_air(afr_st, 0.0), afr_st, epsilon = 1e-12);
    }

    #[test]
    fn realistic_fuel_oil() {
        // Analyse ultime d'un fioul lourd : C 0,85 ; H 0,13 ; S 0,01 ; O 0,01.
        // AFR_st = 11,53·0,85 + 34,34·0,13 + 4,29·0,01 − 4,32·0,01
        //        = 9,8005 + 4,4642 + 0,0429 − 0,0432 = 14,2644 kg/kg.
        let afr_st = combustion_stoichiometric_air_fuel_ratio(0.85, 0.13, 0.01, 0.01);
        assert_relative_eq!(afr_st, 14.2644, epsilon = 1e-9);
        // Avec 20 % d'excès d'air : 14,2644 · 1,20 = 17,11728 kg/kg.
        assert_relative_eq!(
            combustion_actual_air(afr_st, 0.20),
            17.117_28,
            epsilon = 1e-9
        );
    }

    #[test]
    fn excess_air_from_o2_zero_and_case() {
        // Sans O2 résiduel, l'excès d'air est nul (combustion exactement dosée).
        assert_relative_eq!(combustion_excess_air_from_o2(0.0), 0.0, epsilon = 1e-12);
        // 3 % d'O2 dans les fumées : e = 3 / (20,9 − 3) = 3 / 17,9.
        assert_relative_eq!(
            combustion_excess_air_from_o2(3.0),
            3.0 / 17.9,
            epsilon = 1e-12
        );
    }

    #[test]
    fn excess_air_consistency_with_actual_air() {
        // Chaîne cohérente : un excès mesuré appliqué à l'air stœchiométrique
        // reproduit AFR_st·(1 + e).
        let afr_st = combustion_stoichiometric_air_fuel_ratio(0.85, 0.13, 0.01, 0.01);
        let e = combustion_excess_air_from_o2(3.0);
        assert_relative_eq!(
            combustion_actual_air(afr_st, e),
            afr_st * (1.0 + 3.0 / 17.9),
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "inférieure à celle de l'air ambiant")]
    fn o2_at_ambient_panics() {
        combustion_excess_air_from_o2(COMBUSTION_AMBIENT_O2_VOLUME_PERCENT);
    }
}
