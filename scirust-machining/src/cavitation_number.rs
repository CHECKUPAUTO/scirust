//! Cavitation et **NPSH** (Net Positive Suction Head) : indice de cavitation
//! `σ`, charge nette absolue disponible à l'aspiration et marge de sécurité
//! vis-à-vis de la cavitation d'une pompe.
//!
//! ```text
//! indice de cavitation   σ    = (p − pv) / (½·ρ·v²)
//! NPSH disponible (m)     NPSHa = (p_asp − pv)/(ρ·g) + h_v
//! marge de cavitation     ΔNPSH = NPSHa − NPSHr
//! ```
//!
//! `p` pression locale absolue (Pa), `pv` pression de vapeur saturante (Pa),
//! `ρ` masse volumique (kg/m³), `v` vitesse de référence (m/s), `p_asp`
//! pression absolue à l'aspiration (Pa), `g` pesanteur (m/s²), `h_v` charge
//! cinétique à l'aspiration `v²/(2g)` (m). `σ` et `ΔNPSH` : `σ` est sans
//! dimension, `NPSHa`/`NPSHr`/`ΔNPSH` sont en mètres de colonne de fluide.
//!
//! **Convention** : SI cohérent, pressions **absolues**. **Limite honnête** :
//! la cavitation apparaît si `σ < σ_critique` ou si `NPSHa < NPSHr`. L'indice
//! critique `σ_critique`, la charge requise `NPSHr` (donnée par le constructeur
//! de la pompe) et la pression de vapeur `pv` (fonction de la température) sont
//! **fournis par l'appelant** : ce module ne présume d'aucune valeur « par
//! défaut » de fluide, de procédé ou de machine.

/// Indice (nombre) de cavitation `σ = (p − pv) / (½·ρ·v²)` (sans dimension).
///
/// La cavitation est jugée probable lorsque `σ` tombe sous une valeur critique
/// `σ_critique` fournie par l'appelant.
///
/// Panique si `½·ρ·v² <= 0` (masse volumique non strictement positive ou
/// vitesse de référence nulle : la pression dynamique de référence s'annule).
pub fn cavitation_number(
    local_pressure: f64,
    vapor_pressure: f64,
    density: f64,
    reference_velocity: f64,
) -> f64 {
    let dynamic = 0.5 * density * reference_velocity * reference_velocity;
    assert!(
        dynamic > 0.0,
        "la pression dynamique de référence ½·ρ·v² doit être strictement positive"
    );
    (local_pressure - vapor_pressure) / dynamic
}

/// Charge nette absolue disponible à l'aspiration
/// `NPSHa = (p_asp − pv)/(ρ·g) + h_v` (m de colonne de fluide).
///
/// `velocity_head` est la charge cinétique `v²/(2g)` déjà exprimée en mètres.
/// La pompe ne cavite pas tant que `NPSHa >= NPSHr` (charge requise fournie par
/// le constructeur).
///
/// Panique si `ρ·g <= 0`.
pub fn npsh_available(
    suction_pressure: f64,
    vapor_pressure: f64,
    density: f64,
    gravity: f64,
    velocity_head: f64,
) -> f64 {
    assert!(density * gravity > 0.0, "ρ·g doit être strictement positif");
    (suction_pressure - vapor_pressure) / (density * gravity) + velocity_head
}

/// Marge de cavitation `ΔNPSH = NPSHa − NPSHr` (m de colonne de fluide).
///
/// Une marge positive indique un fonctionnement sûr ; une marge négative
/// signale une cavitation de la pompe.
///
/// Panique si l'une des charges n'est pas finie.
pub fn cavitation_margin(npsh_available: f64, npsh_required: f64) -> f64 {
    assert!(
        npsh_available.is_finite() && npsh_required.is_finite(),
        "les charges NPSHa et NPSHr doivent être finies"
    );
    npsh_available - npsh_required
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cavitation_number_water_at_atmospheric_pressure() {
        // Eau à 20 °C : pv ≈ 2339 Pa, ρ=1000, p=101325 Pa (absolue), v=10 m/s.
        // ½·ρ·v² = 0,5·1000·100 = 50000 Pa.
        // σ = (101325 − 2339)/50000 = 98986/50000 = 1,97972.
        assert_relative_eq!(
            cavitation_number(101_325.0, 2339.0, 1000.0, 10.0),
            1.979_72,
            epsilon = 1e-9
        );
    }

    #[test]
    fn cavitation_number_scales_as_inverse_velocity_squared() {
        // σ ∝ 1/v² : doubler la vitesse de référence divise σ par 4.
        let base = cavitation_number(101_325.0, 2339.0, 1000.0, 10.0);
        let faster = cavitation_number(101_325.0, 2339.0, 1000.0, 20.0);
        assert_relative_eq!(faster, base / 4.0, epsilon = 1e-12);
    }

    #[test]
    fn cavitation_number_is_zero_at_vapor_pressure() {
        // Quand la pression locale atteint pv, l'indice s'annule (seuil physique).
        assert_relative_eq!(
            cavitation_number(2339.0, 2339.0, 1000.0, 7.5),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn npsh_available_water_suction() {
        // p_asp=101325 Pa, pv=2339 Pa, ρ=1000, g=9,81, h_v=0,5 m.
        // (101325 − 2339)/(1000·9,81) = 98986/9810 = 10,090316… m ; +0,5 = 10,590316…
        let expected = 98_986.0 / 9810.0 + 0.5;
        assert_relative_eq!(
            npsh_available(101_325.0, 2339.0, 1000.0, 9.81, 0.5),
            expected,
            epsilon = 1e-9
        );
    }

    #[test]
    fn npsh_available_is_additive_in_velocity_head() {
        // Ajouter une charge cinétique Δh_v se répercute mètre pour mètre sur NPSHa.
        let a = npsh_available(101_325.0, 2339.0, 1000.0, 9.81, 0.5);
        let b = npsh_available(101_325.0, 2339.0, 1000.0, 9.81, 1.5);
        assert_relative_eq!(b - a, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn cavitation_margin_is_the_npsh_difference() {
        // NPSHa=10,59 m, NPSHr=3,0 m → marge = 7,59 m (fonctionnement sûr).
        assert_relative_eq!(cavitation_margin(10.59, 3.0), 7.59, epsilon = 1e-12);
        // Marge négative → cavitation.
        assert!(cavitation_margin(2.0, 3.5) < 0.0);
    }

    #[test]
    #[should_panic(expected = "pression dynamique de référence")]
    fn zero_reference_velocity_panics() {
        cavitation_number(101_325.0, 2339.0, 1000.0, 0.0);
    }
}
