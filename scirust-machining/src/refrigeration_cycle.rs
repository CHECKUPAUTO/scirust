//! Cycle frigorifique à compression de vapeur — bilan d'enthalpies aux quatre
//! points caractéristiques (évaporateur, compresseur, condenseur, détendeur).
//!
//! ```text
//! effet frigo     qL = h1 − h4                (J/kg)
//! travail compr.  w  = h2 − h1                (J/kg)
//! COP froid       COP = qL / w               (sans dimension)
//! débit masse     m  = Q / qL                (kg/s)
//! ```
//!
//! `h1` enthalpie sortie évaporateur = entrée compresseur (J/kg) ; `h2`
//! enthalpie sortie compresseur (J/kg) ; `h4` enthalpie entrée évaporateur =
//! sortie détendeur (J/kg) ; `qL` effet frigorifique massique (J/kg) ; `w`
//! travail massique de compression (J/kg) ; `Q` puissance frigorifique
//! (capacité de refroidissement, W) ; `m` débit massique de fluide (kg/s).
//!
//! **Limite honnête** : cycle **idéal** à compression de vapeur (compression
//! isentropique, détente isenthalpique, pas de pertes de charge) ; un
//! sous-refroidissement ou une surchauffe ne sont pris en compte que via les
//! enthalpies fournies. Les **enthalpies** proviennent des tables/diagramme du
//! fluide frigorigène choisi (R134a, R717, R744…), fournies par l'appelant ;
//! aucune valeur de fluide n'est supposée par défaut. Ce COP est celui du cycle
//! réel-idéal (basé enthalpies), **distinct** du COP de Carnot de
//! [`crate::thermo_cycles`] (basé températures).

/// Effet frigorifique massique `qL = h1 − h4` (J/kg).
///
/// `enthalpy_evaporator_out` = h1 (sortie évaporateur), `enthalpy_evaporator_in`
/// = h4 (entrée évaporateur, sortie détendeur), en J/kg.
///
/// Panique si `enthalpy_evaporator_out <= enthalpy_evaporator_in` (pas de
/// chaleur absorbée à l'évaporateur).
pub fn refrig_refrigerating_effect(
    enthalpy_evaporator_out: f64,
    enthalpy_evaporator_in: f64,
) -> f64 {
    assert!(
        enthalpy_evaporator_out > enthalpy_evaporator_in,
        "h1 (sortie évaporateur) doit dépasser h4 (entrée évaporateur)"
    );
    enthalpy_evaporator_out - enthalpy_evaporator_in
}

/// Travail massique de compression `w = h2 − h1` (J/kg).
///
/// `enthalpy_compressor_out` = h2 (refoulement), `enthalpy_compressor_in` = h1
/// (aspiration), en J/kg.
///
/// Panique si `enthalpy_compressor_out <= enthalpy_compressor_in` (le
/// compresseur doit élever l'enthalpie).
pub fn refrig_compressor_work(enthalpy_compressor_out: f64, enthalpy_compressor_in: f64) -> f64 {
    assert!(
        enthalpy_compressor_out > enthalpy_compressor_in,
        "h2 (refoulement) doit dépasser h1 (aspiration)"
    );
    enthalpy_compressor_out - enthalpy_compressor_in
}

/// Coefficient de performance frigorifique `COP = qL / w` (sans dimension).
///
/// `refrigerating_effect` = qL (J/kg), `compressor_work` = w (J/kg).
///
/// Panique si `refrigerating_effect < 0` ou `compressor_work <= 0`.
pub fn refrig_cop_refrigeration(refrigerating_effect: f64, compressor_work: f64) -> f64 {
    assert!(
        refrigerating_effect >= 0.0,
        "l'effet frigorifique qL doit être positif ou nul"
    );
    assert!(
        compressor_work > 0.0,
        "le travail de compression w doit être strictement positif"
    );
    refrigerating_effect / compressor_work
}

/// Débit massique de fluide frigorigène `m = Q / qL` (kg/s).
///
/// `cooling_capacity` = Q puissance frigorifique (W), `refrigerating_effect` =
/// qL effet frigorifique massique (J/kg).
///
/// Panique si `cooling_capacity < 0` ou `refrigerating_effect <= 0`.
pub fn refrig_refrigerant_mass_flow(cooling_capacity: f64, refrigerating_effect: f64) -> f64 {
    assert!(
        cooling_capacity >= 0.0,
        "la puissance frigorifique Q doit être positive ou nulle"
    );
    assert!(
        refrigerating_effect > 0.0,
        "l'effet frigorifique qL doit être strictement positif"
    );
    cooling_capacity / refrigerating_effect
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Point de fonctionnement R134a réaliste (J/kg) :
    //   h1 = 395 000 (vapeur saturée, sortie évaporateur / aspiration)
    //   h2 = 425 000 (refoulement, compression isentropique)
    //   h4 = 255 000 (sortie détendeur = liquide détendu)
    const H1: f64 = 395_000.0;
    const H2: f64 = 425_000.0;
    const H4: f64 = 255_000.0;

    #[test]
    fn refrigerating_effect_realistic() {
        // qL = 395 000 − 255 000 = 140 000 J/kg.
        assert_relative_eq!(
            refrig_refrigerating_effect(H1, H4),
            140_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn compressor_work_realistic() {
        // w = 425 000 − 395 000 = 30 000 J/kg.
        assert_relative_eq!(refrig_compressor_work(H2, H1), 30_000.0, epsilon = 1e-6);
    }

    #[test]
    fn cop_from_enthalpy_definition() {
        // COP = qL/w = 140 000/30 000 = 14/3 ≈ 4,6667.
        let ql = refrig_refrigerating_effect(H1, H4);
        let w = refrig_compressor_work(H2, H1);
        let cop = refrig_cop_refrigeration(ql, w);
        assert_relative_eq!(cop, 14.0 / 3.0, epsilon = 1e-9);
        // Identité : COP·w = qL (réciprocité effet/travail).
        assert_relative_eq!(cop * w, ql, epsilon = 1e-6);
    }

    #[test]
    fn mass_flow_matches_capacity() {
        // Q = 3500 W, qL = 140 000 J/kg → m = 0,025 kg/s.
        let ql = refrig_refrigerating_effect(H1, H4);
        let m = refrig_refrigerant_mass_flow(3500.0, ql);
        assert_relative_eq!(m, 0.025, epsilon = 1e-12);
        // Réciprocité : m·qL = Q (bilan de puissance à l'évaporateur).
        assert_relative_eq!(m * ql, 3500.0, epsilon = 1e-6);
    }

    #[test]
    fn compressor_power_consistency() {
        // Puissance mécanique = m·w ; puissance rejetée = m·(qL+w) = m·(h2−h4).
        let ql = refrig_refrigerating_effect(H1, H4);
        let w = refrig_compressor_work(H2, H1);
        let m = refrig_refrigerant_mass_flow(3500.0, ql);
        let heat_rejected = m * (H2 - H4);
        assert_relative_eq!(heat_rejected, m * (ql + w), epsilon = 1e-6);
    }

    #[test]
    fn zero_capacity_gives_zero_flow() {
        // Cas limite : puissance nulle → débit nul.
        assert_relative_eq!(
            refrig_refrigerant_mass_flow(0.0, 140_000.0),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "h1 (sortie évaporateur) doit dépasser h4")]
    fn effect_panics_when_no_absorption() {
        refrig_refrigerating_effect(255_000.0, 395_000.0);
    }
}
