//! Coefficient de transfert thermique par **convection** obtenu à partir de
//! **corrélations adimensionnelles** — nombres de **Reynolds** et de
//! **Prandtl**, corrélation de **Dittus-Boelter** (Nusselt en conduite
//! turbulente établie), passage du **Nusselt** au coefficient de film `h`, puis
//! **coefficient global** par mise en série des résistances (film interne, paroi,
//! film externe).
//!
//! ```text
//! Reynolds        Re = ρ · u · L / μ                                     [-]
//! Prandtl         Pr = cp · μ / k                                        [-]
//! Dittus-Boelter  Nu = 0.023 · Re^0.8 · Pr^n   (n = 0.4 chauffage,
//!                                               n = 0.3 refroidissement) [-]
//! coeff. de film  h  = Nu · k / L                                 [W·m⁻²·K⁻¹]
//! coeff. global   1/U = 1/h_i + R_w + 1/h_o  ⇒  U                 [W·m⁻²·K⁻¹]
//! ```
//!
//! `Re` nombre de Reynolds [sans dimension], `ρ` masse volumique [kg·m⁻³], `u`
//! vitesse débitante [m·s⁻¹], `L` longueur caractéristique (diamètre
//! hydraulique en conduite) [m], `μ` viscosité dynamique [Pa·s = kg·m⁻¹·s⁻¹] ;
//! `Pr` nombre de Prandtl [sans dimension], `cp` capacité thermique massique
//! [J·kg⁻¹·K⁻¹], `k` conductivité thermique [W·m⁻¹·K⁻¹] ; `Nu` nombre de
//! Nusselt [sans dimension], `n` exposant de Prandtl [sans dimension] ; `h`
//! coefficient de transfert convectif de film [W·m⁻²·K⁻¹] ; `U` coefficient
//! global de transfert [W·m⁻²·K⁻¹], `h_i`/`h_o` coefficients de film interne et
//! externe [W·m⁻²·K⁻¹], `R_w` résistance thermique surfacique de la paroi
//! [m²·K·W⁻¹]. Températures en K.
//!
//! **Limite honnête** : les **propriétés physiques** (`ρ`, `μ`, `k`, `cp`) sont
//! **FOURNIES par l'appelant**, évaluées à la **température de film**, et ne
//! sont **jamais inventées** ici (elles proviennent de tables, d'états ou de
//! corrélations de propriétés). La **longueur caractéristique** `L`, la
//! **résistance de paroi** `R_w` et les **coefficients de film** externes sont
//! également **FOURNIS**. La corrélation de **Dittus-Boelter** n'est valable
//! qu'en **régime turbulent établi** dans une conduite lisse, pour
//! `Re > 10 000`, `0.6 < Pr < 160` et un rapport longueur/diamètre suffisant ;
//! hors de ce domaine elle n'est **pas** applicable (ces bornes sont
//! documentées, non imposées par les `assert!`). Ce module se limite aux
//! **corrélations de coefficient de transfert** ; il **complète** l'analogie de
//! **Chilton-Colburn** du transfert de matière (`mass_transfer`) sans la
//! dupliquer, et ne recouvre ni les **propriétés d'état** (`scirust-thermo`) ni
//! la **mécanique des fluides fondamentale** (`scirust-fluids`).

/// Nombre de **Reynolds** `Re = ρ · u · L / μ` (sans dimension), rapport des
/// forces d'inertie aux forces visqueuses.
///
/// `density` (ρ) [kg·m⁻³] ; `velocity` (u) vitesse débitante [m·s⁻¹] ;
/// `characteristic_length` (L) [m] ; `viscosity` (μ) viscosité dynamique
/// [Pa·s].
///
/// Panique si `density < 0`, `characteristic_length < 0` ou `viscosity <= 0`.
pub fn htc_reynolds(
    density: f64,
    velocity: f64,
    characteristic_length: f64,
    viscosity: f64,
) -> f64 {
    assert!(density >= 0.0, "ρ ≥ 0 requis (masse volumique, kg·m⁻³)");
    assert!(
        characteristic_length >= 0.0,
        "L ≥ 0 requis (longueur caractéristique, m)"
    );
    assert!(viscosity > 0.0, "μ > 0 requis (viscosité dynamique, Pa·s)");
    density * velocity * characteristic_length / viscosity
}

/// Nombre de **Prandtl** `Pr = cp · μ / k` (sans dimension), rapport de la
/// diffusivité de quantité de mouvement à la diffusivité thermique.
///
/// `heat_capacity` (cp) [J·kg⁻¹·K⁻¹] ; `viscosity` (μ) [Pa·s] ;
/// `thermal_conductivity` (k) [W·m⁻¹·K⁻¹].
///
/// Panique si `heat_capacity < 0`, `viscosity < 0` ou
/// `thermal_conductivity <= 0`.
pub fn htc_prandtl(heat_capacity: f64, viscosity: f64, thermal_conductivity: f64) -> f64 {
    assert!(
        heat_capacity >= 0.0,
        "cp ≥ 0 requis (capacité thermique massique, J·kg⁻¹·K⁻¹)"
    );
    assert!(viscosity >= 0.0, "μ ≥ 0 requis (viscosité dynamique, Pa·s)");
    assert!(
        thermal_conductivity > 0.0,
        "k > 0 requis (conductivité thermique, W·m⁻¹·K⁻¹)"
    );
    heat_capacity * viscosity / thermal_conductivity
}

/// Nombre de **Nusselt** par la corrélation de **Dittus-Boelter**
/// `Nu = 0.023 · Re^0.8 · Pr^n` (sans dimension), convection forcée turbulente
/// établie en conduite.
///
/// `exponent` (n) vaut par convention `0.4` en **chauffage** du fluide et `0.3`
/// en **refroidissement**. Domaine de validité (documenté, non imposé) :
/// `Re > 10 000`, `0.6 < Pr < 160`.
///
/// `reynolds` (Re) et `prandtl` (Pr) sont des nombres adimensionnels ;
/// `exponent` (n) sans dimension.
///
/// Panique si `reynolds <= 0` ou `prandtl <= 0`.
pub fn htc_dittus_boelter(reynolds: f64, prandtl: f64, exponent: f64) -> f64 {
    assert!(reynolds > 0.0, "Re > 0 requis (nombre de Reynolds)");
    assert!(prandtl > 0.0, "Pr > 0 requis (nombre de Prandtl)");
    0.023 * reynolds.powf(0.8) * prandtl.powf(exponent)
}

/// Coefficient de **film** convectif à partir du Nusselt `h = Nu · k / L`
/// (W·m⁻²·K⁻¹).
///
/// `nusselt` (Nu) nombre adimensionnel ; `thermal_conductivity` (k)
/// [W·m⁻¹·K⁻¹] ; `characteristic_length` (L) [m].
///
/// Panique si `nusselt < 0`, `thermal_conductivity < 0` ou
/// `characteristic_length <= 0`.
pub fn htc_coefficient_from_nusselt(
    nusselt: f64,
    thermal_conductivity: f64,
    characteristic_length: f64,
) -> f64 {
    assert!(nusselt >= 0.0, "Nu ≥ 0 requis (nombre de Nusselt)");
    assert!(
        thermal_conductivity >= 0.0,
        "k ≥ 0 requis (conductivité thermique, W·m⁻¹·K⁻¹)"
    );
    assert!(
        characteristic_length > 0.0,
        "L > 0 requis (longueur caractéristique, m)"
    );
    nusselt * thermal_conductivity / characteristic_length
}

/// Coefficient **global** de transfert `U` par mise en **série des
/// résistances** : `1/U = 1/h_i + R_w + 1/h_o`, d'où
/// `U = 1/(1/h_i + R_w + 1/h_o)` (W·m⁻²·K⁻¹).
///
/// `inside_coefficient` (h_i) et `outside_coefficient` (h_o) coefficients de
/// film interne et externe [W·m⁻²·K⁻¹] ; `wall_resistance` (R_w) résistance
/// thermique surfacique de la paroi [m²·K·W⁻¹].
///
/// Panique si `inside_coefficient <= 0`, `outside_coefficient <= 0` ou
/// `wall_resistance < 0`.
pub fn htc_wall_from_series(
    inside_coefficient: f64,
    wall_resistance: f64,
    outside_coefficient: f64,
) -> f64 {
    assert!(
        inside_coefficient > 0.0 && outside_coefficient > 0.0,
        "h_i > 0 et h_o > 0 requis (coefficients de film, W·m⁻²·K⁻¹)"
    );
    assert!(
        wall_resistance >= 0.0,
        "R_w ≥ 0 requis (résistance de paroi, m²·K·W⁻¹)"
    );
    1.0 / (1.0 / inside_coefficient + wall_resistance + 1.0 / outside_coefficient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reynolds_value_and_velocity_proportionality() {
        // ρ = 1000, u = 2, L = 0.05, μ = 0.001 :
        // Re = 1000·2·0.05/0.001 = 100/0.001 = 100 000.
        let re = htc_reynolds(1000.0_f64, 2.0_f64, 0.05_f64, 0.001_f64);
        assert_relative_eq!(re, 100_000.0, max_relative = 1e-12);
        // Proportionnalité à la vitesse : doubler u double Re.
        let re2 = htc_reynolds(1000.0_f64, 4.0_f64, 0.05_f64, 0.001_f64);
        assert_relative_eq!(re2, 2.0 * re, max_relative = 1e-12);
    }

    #[test]
    fn prandtl_value_and_reciprocity() {
        // cp = 2000, μ = 0.001, k = 0.5 : Pr = 2000·0.001/0.5 = 2/0.5 = 4.
        let pr = htc_prandtl(2000.0_f64, 0.001_f64, 0.5_f64);
        assert_relative_eq!(pr, 4.0, max_relative = 1e-12);
        // Réciprocité : Pr · k / μ redonne cp.
        assert_relative_eq!(pr * 0.5_f64 / 0.001_f64, 2000.0, max_relative = 1e-12);
    }

    #[test]
    fn dittus_boelter_numeric_case() {
        // Re = 10 000, Pr = 4, chauffage (n = 0.4) :
        // Nu = 0.023 · 10000^0.8 · 4^0.4.
        // 10000^0.8 = 10^3.2 = 1584.893192…
        // 4^0.4 = 2^0.8 = 1.741101127…
        // Nu = 0.023 · 1584.893192 · 1.741101127 = 63.46756…
        let nu = htc_dittus_boelter(10_000.0_f64, 4.0_f64, 0.4_f64);
        assert_relative_eq!(nu, 63.46756, max_relative = 1e-3);
        // Vérification indépendante par recalcul de l'expression.
        let check = 0.023 * 10_000.0_f64.powf(0.8) * 4.0_f64.powf(0.4);
        assert_relative_eq!(nu, check, max_relative = 1e-12);
    }

    #[test]
    fn dittus_boelter_prandtl_exponent_ratio() {
        // À Re fixé, Nu(Pr₂)/Nu(Pr₁) = (Pr₂/Pr₁)^n : ici Pr₂ = 4, Pr₁ = 1,
        // n = 0.5 ⇒ rapport = 4^0.5 = 2.
        let nu1 = htc_dittus_boelter(20_000.0_f64, 1.0_f64, 0.5_f64);
        let nu2 = htc_dittus_boelter(20_000.0_f64, 4.0_f64, 0.5_f64);
        assert_relative_eq!(nu2 / nu1, 2.0, max_relative = 1e-9);
    }

    #[test]
    fn coefficient_from_nusselt_value_and_reciprocity() {
        // Nu = 100, k = 0.6, L = 0.05 : h = 100·0.6/0.05 = 60/0.05 = 1200.
        let h = htc_coefficient_from_nusselt(100.0_f64, 0.6_f64, 0.05_f64);
        assert_relative_eq!(h, 1200.0, max_relative = 1e-12);
        // Réciprocité : h · L / k redonne Nu.
        assert_relative_eq!(h * 0.05_f64 / 0.6_f64, 100.0, max_relative = 1e-12);
    }

    #[test]
    fn wall_series_value_and_dominant_resistance_limit() {
        // h_i = 1000, R_w = 0.001, h_o = 500 :
        // 1/U = 1/1000 + 0.001 + 1/500 = 0.001 + 0.001 + 0.002 = 0.004
        // ⇒ U = 250 W·m⁻²·K⁻¹.
        let u = htc_wall_from_series(1000.0_f64, 0.001_f64, 500.0_f64);
        assert_relative_eq!(u, 250.0, max_relative = 1e-12);
        // Limite : quand la résistance de paroi domine (R_w très grand devant
        // 1/h_i et 1/h_o), U → 1/R_w.
        let u_wall = htc_wall_from_series(1.0e9_f64, 10.0_f64, 1.0e9_f64);
        assert_relative_eq!(u_wall, 1.0 / 10.0_f64, max_relative = 1e-6);
    }

    #[test]
    #[should_panic(expected = "μ > 0 requis")]
    fn reynolds_zero_viscosity_panics() {
        // Viscosité nulle : division non définie, rejetée par assert!.
        let _ = htc_reynolds(1000.0_f64, 2.0_f64, 0.05_f64, 0.0_f64);
    }
}
