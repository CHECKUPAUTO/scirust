//! Résistance thermique d'un **mur composite** en **régime permanent 1D** :
//! conduction, convection, associations série/parallèle, coefficient global et
//! flux de chaleur.
//!
//! ```text
//! conduction     R_cond = L / (k·A)
//! convection     R_conv = 1 / (h·A)
//! série          R_tot  = Σ Rᵢ
//! parallèle      R_tot  = 1 / Σ (1/Rᵢ)
//! coeff. global  U      = 1 / (R_tot·A)
//! flux           q      = ΔT / R_tot
//! ```
//!
//! `L` épaisseur (m), `k` conductivité (W/(m·K)), `A` surface d'échange (m²),
//! `h` coefficient de convection (W/(m²·K)), `R` résistance thermique (K/W),
//! `U` coefficient global de transfert (W/(m²·K)), `ΔT` différence de
//! température (K), `q` flux de chaleur (W).
//!
//! **Convention** : SI cohérent. **Limite honnête** : régime **permanent** et
//! unidimensionnel ; les conductivités `k`, les coefficients de convection `h`
//! et les géométries sont **fournis par l'appelant** (aucune valeur matériau
//! « par défaut » inventée) ; contacts supposés **parfaits** (aucune résistance
//! de contact, sauf à l'ajouter explicitement comme un terme de la série). Pour
//! le régime instationnaire (Biot/Fourier, capacité localisée), voir
//! [`crate::transient_conduction`].

/// Résistance de conduction `R_cond = L / (k·A)` (K/W).
///
/// Panique si `thickness < 0`, `conductivity <= 0` ou `area <= 0`.
pub fn thermal_resistance_conduction(thickness: f64, conductivity: f64, area: f64) -> f64 {
    assert!(thickness >= 0.0, "l'épaisseur doit être positive ou nulle");
    assert!(
        conductivity > 0.0,
        "la conductivité doit être strictement positive"
    );
    assert!(area > 0.0, "la surface doit être strictement positive");
    thickness / (conductivity * area)
}

/// Résistance de convection `R_conv = 1 / (h·A)` (K/W).
///
/// Panique si `heat_transfer_coefficient <= 0` ou `area <= 0`.
pub fn thermal_resistance_convection(heat_transfer_coefficient: f64, area: f64) -> f64 {
    assert!(
        heat_transfer_coefficient > 0.0,
        "le coefficient de convection doit être strictement positif"
    );
    assert!(area > 0.0, "la surface doit être strictement positive");
    1.0 / (heat_transfer_coefficient * area)
}

/// Association **série** `R_tot = Σ Rᵢ` (K/W) — résistances traversées par le
/// même flux.
///
/// Panique si `resistances` est vide ou contient une valeur négative.
pub fn thermal_resistance_series(resistances: &[f64]) -> f64 {
    assert!(
        !resistances.is_empty(),
        "la liste des résistances ne doit pas être vide"
    );
    assert!(
        resistances.iter().all(|&r| r >= 0.0),
        "chaque résistance doit être positive ou nulle"
    );
    resistances.iter().sum()
}

/// Association **parallèle** `R_tot = 1 / Σ (1/Rᵢ)` (K/W) — résistances soumises
/// à la même différence de température.
///
/// Panique si `resistances` est vide ou contient une valeur négative ou nulle.
pub fn thermal_resistance_parallel(resistances: &[f64]) -> f64 {
    assert!(
        !resistances.is_empty(),
        "la liste des résistances ne doit pas être vide"
    );
    assert!(
        resistances.iter().all(|&r| r > 0.0),
        "chaque résistance doit être strictement positive"
    );
    1.0 / resistances.iter().map(|&r| 1.0 / r).sum::<f64>()
}

/// Coefficient global de transfert `U = 1 / (R_tot·A)` (W/(m²·K)).
///
/// Panique si `total_resistance <= 0` ou `area <= 0`.
pub fn thermal_overall_coefficient(total_resistance: f64, area: f64) -> f64 {
    assert!(
        total_resistance > 0.0,
        "la résistance totale doit être strictement positive"
    );
    assert!(area > 0.0, "la surface doit être strictement positive");
    1.0 / (total_resistance * area)
}

/// Flux de chaleur `q = ΔT / R_tot` (W).
///
/// Panique si `total_resistance <= 0`.
pub fn thermal_heat_rate(temperature_difference: f64, total_resistance: f64) -> f64 {
    assert!(
        total_resistance > 0.0,
        "la résistance totale doit être strictement positive"
    );
    temperature_difference / total_resistance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn conduction_cas_chiffre() {
        // L = 0,2 m ; k = 0,7 W/(m·K) ; A = 10 m² -> R = 0,2/(0,7·10) = 0,028571…
        let r = thermal_resistance_conduction(0.2, 0.7, 10.0);
        assert_relative_eq!(r, 0.2 / 7.0, max_relative = 1e-12);
    }

    #[test]
    fn convection_est_conduction_de_conductance_unite() {
        // 1/(h·A) coïncide avec L/(k·A) lorsque L/k = 1/h.
        let h = 25.0;
        let area = 4.0;
        let conv = thermal_resistance_convection(h, area);
        let cond = thermal_resistance_conduction(1.0, h, area);
        assert_relative_eq!(conv, cond, max_relative = 1e-12);
    }

    #[test]
    fn serie_somme_des_termes() {
        let r1 = thermal_resistance_convection(10.0, 10.0); // 0,01
        let r2 = thermal_resistance_conduction(0.2, 0.7, 10.0); // 0,028571…
        let r3 = thermal_resistance_convection(25.0, 10.0); // 0,004
        let total = thermal_resistance_series(&[r1, r2, r3]);
        assert_relative_eq!(total, r1 + r2 + r3, max_relative = 1e-12);
    }

    #[test]
    fn parallele_deux_egales_donne_la_moitie() {
        // Deux résistances identiques R en parallèle -> R/2.
        let r = 0.02;
        let eq = thermal_resistance_parallel(&[r, r]);
        assert_relative_eq!(eq, r / 2.0, max_relative = 1e-12);
    }

    #[test]
    fn flux_coherent_avec_coefficient_global() {
        // q = ΔT/R doit égaler U·A·ΔT avec U = 1/(R·A).
        let total = 0.0425714285714286; // série du test dédié
        let area = 10.0;
        let dt = 20.0;
        let u = thermal_overall_coefficient(total, area);
        let q_par_resistance = thermal_heat_rate(dt, total);
        let q_par_coefficient = u * area * dt;
        assert_relative_eq!(q_par_resistance, q_par_coefficient, max_relative = 1e-12);
    }

    #[test]
    fn flux_cas_chiffre() {
        // ΔT = 50 K ; R = 0,025 K/W -> q = 2000 W.
        let q = thermal_heat_rate(50.0, 0.025);
        assert_relative_eq!(q, 2000.0, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "la conductivité doit être strictement positive")]
    fn conduction_conductivite_nulle_panique() {
        let _ = thermal_resistance_conduction(0.1, 0.0, 1.0);
    }
}
