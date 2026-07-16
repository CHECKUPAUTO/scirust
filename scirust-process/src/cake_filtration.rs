//! Filtration sur gâteau à **pression constante** — intégration de la loi de
//! filtration (forme linéaire de `dt/dV`), temps de filtration, allure moyenne,
//! débit de lavage et détermination de la résistance du média filtrant à partir
//! de l'ordonnée à l'origine du tracé `t/V` en fonction de `V`.
//!
//! ```text
//! loi de filtration (débit instantané, forme linéaire en dV/dt)
//!   dt/dV = μ·α·c/(A²·ΔP)·V + μ·R_m/(A·ΔP)
//! temps de filtration à pression constante (intégration de V = 0 à V)
//!   t     = μ·α·c/(2·A²·ΔP)·V² + μ·R_m/(A·ΔP)·V                 [s]
//! allure (débit) moyenne de filtration
//!   Q_moy = V / t                                              [m³·s⁻¹]
//! débit de lavage (à pression égale)
//!   Q_lav = Q_fin                                              [m³·s⁻¹]
//! résistance du média depuis l'ordonnée à l'origine de t/V vs V
//!   R_m   = b · A · ΔP / μ                                     [m⁻¹]
//! ```
//!
//! `μ` viscosité dynamique du filtrat [Pa·s], `α` résistance spécifique du gâteau
//! [m·kg⁻¹], `c` masse de gâteau sec déposé par unité de volume de filtrat
//! [kg·m⁻³], `A` aire de la surface filtrante [m²], `ΔP` perte de charge (pression
//! motrice) à travers gâteau + média [Pa], `V` volume de filtrat recueilli [m³],
//! `R_m` résistance du média filtrant [m⁻¹], `t` temps de filtration [s], `Q_moy`
//! débit volumique moyen [m³·s⁻¹], `Q_fin` débit instantané en fin de filtration
//! [m³·s⁻¹], `Q_lav` débit de lavage [m³·s⁻¹], `b` ordonnée à l'origine du tracé
//! linéarisé `t/V = a·V + b` [s·m⁻³].
//!
//! **Limite honnête** : modèle à l'échelle des **opérations unitaires**,
//! filtration **à pression constante** et gâteau **INCOMPRESSIBLE** (la résistance
//! spécifique `α` ne dépend pas de `ΔP`). La résistance spécifique du gâteau `α`,
//! la masse sèche par volume de filtrat `c`, la viscosité `μ` du filtrat et la
//! résistance du média `R_m` sont **FOURNIES** par l'appelant d'après des essais
//! de filtration, des tables ou des corrélations — jamais inventées ni corrélées
//! ici (en particulier `α = α₀·ΔPⁿ` pour un gâteau compressible relève de
//! l'appelant). Aucune propriété physique (masse volumique, porosité du gâteau,
//! compressibilité) n'est supposée par défaut. Complète les opérations
//! solide-liquide de la crate (sédimentation, épaississement).

/// Temps de filtration **à pression constante** obtenu par intégration de la loi
/// de filtration `t = μ·α·c/(2·A²·ΔP)·V² + μ·R_m/(A·ΔP)·V` (s).
///
/// Le premier terme (quadratique en `V`) est la contribution du **gâteau**, le
/// second (linéaire en `V`) celle du **média filtrant**.
///
/// `specific_resistance` (α) résistance spécifique du gâteau [m·kg⁻¹],
/// `viscosity` (μ) viscosité du filtrat [Pa·s], `dry_cake_per_volume` (c) masse de
/// gâteau sec par volume de filtrat [kg·m⁻³], `area` (A) aire filtrante [m²],
/// `pressure_drop` (ΔP) perte de charge [Pa], `volume` (V) volume de filtrat [m³],
/// `medium_resistance` (R_m) résistance du média [m⁻¹].
///
/// Panique si `α < 0`, `μ ≤ 0`, `c < 0`, `A ≤ 0`, `ΔP ≤ 0`, `V < 0` ou `R_m < 0`.
pub fn filt_constant_pressure_time(
    specific_resistance: f64,
    viscosity: f64,
    dry_cake_per_volume: f64,
    area: f64,
    pressure_drop: f64,
    volume: f64,
    medium_resistance: f64,
) -> f64 {
    assert!(
        specific_resistance >= 0.0,
        "α ≥ 0 requis (résistance spécifique du gâteau)"
    );
    assert!(viscosity > 0.0, "μ > 0 requis (viscosité du filtrat)");
    assert!(
        dry_cake_per_volume >= 0.0,
        "c ≥ 0 requis (masse sèche par volume de filtrat)"
    );
    assert!(area > 0.0, "A > 0 requis (aire filtrante)");
    assert!(pressure_drop > 0.0, "ΔP > 0 requis (perte de charge)");
    assert!(volume >= 0.0, "V ≥ 0 requis (volume de filtrat)");
    assert!(
        medium_resistance >= 0.0,
        "R_m ≥ 0 requis (résistance du média)"
    );
    let cake_term = viscosity * specific_resistance * dry_cake_per_volume
        / (2.0 * area * area * pressure_drop)
        * volume
        * volume;
    let medium_term = viscosity * medium_resistance / (area * pressure_drop) * volume;
    cake_term + medium_term
}

/// Allure (débit volumique) **moyenne** de filtration `Q_moy = V / t` (m³·s⁻¹).
///
/// `volume` (V) volume de filtrat recueilli [m³], `time` (t) durée de filtration
/// correspondante [s].
///
/// Panique si `V < 0` ou si `t ≤ 0` (durée strictement positive requise).
pub fn filt_average_rate(volume: f64, time: f64) -> f64 {
    assert!(volume >= 0.0, "V ≥ 0 requis (volume de filtrat)");
    assert!(time > 0.0, "t > 0 requis (durée de filtration)");
    volume / time
}

/// Débit de **lavage** du gâteau `Q_lav = Q_fin` (m³·s⁻¹) : à pression égale, le
/// débit de lavage vaut le débit instantané atteint **en fin de filtration**.
///
/// `final_filtration_rate` (Q_fin) débit instantané en fin de filtration
/// [m³·s⁻¹].
///
/// Panique si `Q_fin < 0`.
pub fn filt_washing_rate(final_filtration_rate: f64) -> f64 {
    assert!(
        final_filtration_rate >= 0.0,
        "Q_fin ≥ 0 requis (débit final de filtration)"
    );
    final_filtration_rate
}

/// Résistance du **média filtrant** déduite de l'**ordonnée à l'origine** `b` du
/// tracé linéarisé `t/V = a·V + b`, soit `R_m = b·A·ΔP/μ` (m⁻¹).
///
/// `intercept` (b) ordonnée à l'origine du tracé `t/V` en fonction de `V`
/// [s·m⁻³], `viscosity` (μ) viscosité du filtrat [Pa·s], `area` (A) aire filtrante
/// [m²], `pressure_drop` (ΔP) perte de charge [Pa].
///
/// Panique si `b < 0`, `μ ≤ 0`, `A ≤ 0` ou `ΔP ≤ 0`.
pub fn filt_medium_resistance_from_intercept(
    intercept: f64,
    viscosity: f64,
    area: f64,
    pressure_drop: f64,
) -> f64 {
    assert!(intercept >= 0.0, "b ≥ 0 requis (ordonnée à l'origine)");
    assert!(viscosity > 0.0, "μ > 0 requis (viscosité du filtrat)");
    assert!(area > 0.0, "A > 0 requis (aire filtrante)");
    assert!(pressure_drop > 0.0, "ΔP > 0 requis (perte de charge)");
    intercept * area * pressure_drop / viscosity
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn constant_pressure_time_worked_case() {
        // Cas chiffré : α = 1e11 m/kg, μ = 1e-3 Pa·s, c = 20 kg/m³, A = 1 m²,
        // ΔP = 1e5 Pa, V = 0.1 m³, R_m = 1e10 m⁻¹.
        //   terme gâteau = μ·α·c/(2·A²·ΔP)·V²
        //               = 1e-3·1e11·20/(2·1·1e5)·0.1²
        //               = 2e9/2e5·0.01 = 1e4·0.01 = 100 s
        //   terme média  = μ·R_m/(A·ΔP)·V
        //               = 1e-3·1e10/(1·1e5)·0.1 = 100·0.1 = 10 s
        //   t = 100 + 10 = 110 s.
        let t = filt_constant_pressure_time(
            1.0e11_f64, 1.0e-3_f64, 20.0_f64, 1.0_f64, 1.0e5_f64, 0.1_f64, 1.0e10_f64,
        );
        assert_relative_eq!(t, 110.0, max_relative = 1e-9);
    }

    #[test]
    fn constant_pressure_time_medium_only_is_linear() {
        // α = 0 ⇒ pas de gâteau ⇒ t = μ·R_m/(A·ΔP)·V, strictement linéaire en V :
        // doubler V double le temps.
        let t1 = filt_constant_pressure_time(
            0.0_f64, 1.0e-3_f64, 20.0_f64, 2.0_f64, 5.0e4_f64, 0.05_f64, 1.0e10_f64,
        );
        let t2 = filt_constant_pressure_time(
            0.0_f64, 1.0e-3_f64, 20.0_f64, 2.0_f64, 5.0e4_f64, 0.10_f64, 1.0e10_f64,
        );
        assert_relative_eq!(t2, 2.0 * t1, max_relative = 1e-12);
    }

    #[test]
    fn constant_pressure_time_cake_scales_with_alpha() {
        // R_m = 0 ⇒ terme gâteau seul ∝ α : doubler α double le temps.
        let t1 = filt_constant_pressure_time(
            1.0e11_f64, 1.0e-3_f64, 15.0_f64, 1.0_f64, 2.0e5_f64, 0.2_f64, 0.0_f64,
        );
        let t2 = filt_constant_pressure_time(
            2.0e11_f64, 1.0e-3_f64, 15.0_f64, 1.0_f64, 2.0e5_f64, 0.2_f64, 0.0_f64,
        );
        assert_relative_eq!(t2, 2.0 * t1, max_relative = 1e-12);
    }

    #[test]
    fn average_rate_is_inverse_of_time_per_volume() {
        // Q_moy = V/t. Avec V = 0.1 m³ et t = 110 s (cas chiffré) ⇒
        //   Q_moy = 0.1/110 = 9.0909...e-4 m³/s.
        let t = filt_constant_pressure_time(
            1.0e11_f64, 1.0e-3_f64, 20.0_f64, 1.0_f64, 1.0e5_f64, 0.1_f64, 1.0e10_f64,
        );
        let q = filt_average_rate(0.1_f64, t);
        assert_relative_eq!(q, 0.1 / 110.0, max_relative = 1e-9);
        // Réciprocité : Q_moy · t = V.
        assert_relative_eq!(q * t, 0.1, max_relative = 1e-9);
    }

    #[test]
    fn washing_rate_equals_final_filtration_rate() {
        // À pression égale, le débit de lavage vaut le dernier débit de filtration.
        let q_final = 3.5e-4_f64;
        assert_relative_eq!(filt_washing_rate(q_final), q_final, max_relative = 1e-12);
    }

    #[test]
    fn medium_resistance_inverts_intercept() {
        // Réciprocité média ↔ ordonnée à l'origine. L'intercept vaut
        //   b = μ·R_m/(A·ΔP) = 1e-3·1e10/(1·1e5) = 100 s/m³.
        // On doit retrouver R_m = b·A·ΔP/μ = 100·1·1e5/1e-3 = 1e10 m⁻¹.
        let intercept = 1.0e-3_f64 * 1.0e10_f64 / (1.0_f64 * 1.0e5_f64);
        assert_relative_eq!(intercept, 100.0, max_relative = 1e-12);
        let r_m = filt_medium_resistance_from_intercept(intercept, 1.0e-3_f64, 1.0_f64, 1.0e5_f64);
        assert_relative_eq!(r_m, 1.0e10, max_relative = 1e-9);
    }

    #[test]
    #[should_panic(expected = "ΔP > 0 requis")]
    fn constant_pressure_time_panics_on_zero_pressure() {
        // Perte de charge nulle ⇒ division par zéro non physique ⇒ entrée rejetée.
        let _ = filt_constant_pressure_time(
            1.0e11_f64, 1.0e-3_f64, 20.0_f64, 1.0_f64, 0.0_f64, 0.1_f64, 1.0e10_f64,
        );
    }
}
