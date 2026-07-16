//! Perte de charge en conduite (procédé) — nombre de Reynolds, facteur de
//! frottement de Darcy (laminaire et corrélation turbulente de Swamee-Jain),
//! perte de charge régulière de Darcy-Weisbach, pertes singulières et longueur
//! équivalente d'une singularité.
//!
//! ```text
//! nombre de Reynolds
//!   Re   = ρ·v·D / μ                                            [-]
//! facteur de frottement de Darcy, régime laminaire (Re ≲ 2300)
//!   f    = 64 / Re                                              [-]
//! facteur de frottement de Darcy, régime turbulent (Swamee-Jain)
//!   f    = 0.25 / [ log₁₀( ε/D/3.7 + 5.74/Re^0.9 ) ]²           [-]
//! perte de charge régulière (Darcy-Weisbach)
//!   Δp   = f · (L/D) · ρ·v²/2                                   [Pa]
//! pertes singulières (accessoires)
//!   Δp_s = K · ρ·v²/2                                           [Pa]
//! longueur équivalente d'une singularité
//!   L_eq = K · D / f                                            [m]
//! ```
//!
//! `ρ` masse volumique du fluide [kg·m⁻³], `v` vitesse débitante moyenne
//! [m·s⁻¹], `D` diamètre intérieur de la conduite [m], `μ` viscosité dynamique
//! [Pa·s], `Re` nombre de Reynolds [sans dimension], `f` facteur de frottement
//! de Darcy [sans dimension], `ε/D` rugosité relative de la paroi [sans
//! dimension], `L` longueur de conduite [m], `Δp` perte de charge régulière
//! [Pa], `K` coefficient de perte singulière de l'accessoire [sans dimension],
//! `Δp_s` perte de charge singulière [Pa], `L_eq` longueur de conduite droite
//! produisant la même perte que la singularité [m].
//!
//! **Limite honnête** : la masse volumique `ρ`, la viscosité dynamique `μ`, la
//! rugosité relative `ε/D` et les coefficients de perte singulière `K` sont
//! **FOURNIS** par l'appelant d'après des tables, des essais ou la
//! documentation du fabricant — aucune propriété physique ni coefficient n'est
//! calculé ni inventé ici. La corrélation de **Swamee-Jain** est une
//! approximation **explicite** de l'équation implicite de **Colebrook-White**,
//! valable en régime **turbulent** (`Re > 4000` environ) ; elle n'est pas
//! utilisée en régime laminaire, pour lequel on emploie `f = 64/Re`. La zone
//! critique/transitoire (`2300 < Re < 4000`) n'est pas modélisée. Fluide
//! **newtonien incompressible**, conduite **circulaire pleine** (section
//! entièrement remplie), écoulement établi et isotherme. Ce module fournit
//! séparément `Re` et `f` : le choix de la corrélation selon le régime revient
//! à l'appelant.

/// Nombre de Reynolds d'un écoulement en conduite
/// `Re = ρ·v·D / μ` (sans dimension), rapport des forces d'inertie aux forces
/// visqueuses.
///
/// `density` (ρ) [kg·m⁻³], `velocity` (v) vitesse débitante moyenne [m·s⁻¹],
/// `diameter` (D) diamètre intérieur [m], `viscosity` (μ) viscosité dynamique
/// [Pa·s].
///
/// Panique si `ρ ≤ 0`, si `v ≤ 0`, si `D ≤ 0`, ou si `μ ≤ 0`.
pub fn dp_reynolds(density: f64, velocity: f64, diameter: f64, viscosity: f64) -> f64 {
    assert!(density > 0.0, "ρ > 0 requis (masse volumique du fluide)");
    assert!(velocity > 0.0, "v > 0 requis (vitesse débitante)");
    assert!(diameter > 0.0, "D > 0 requis (diamètre intérieur)");
    assert!(viscosity > 0.0, "μ > 0 requis (viscosité dynamique)");
    density * velocity * diameter / viscosity
}

/// Facteur de frottement de Darcy en régime laminaire
/// `f = 64 / Re` (sans dimension), résultat exact de l'écoulement de
/// Hagen-Poiseuille en conduite circulaire (`Re ≲ 2300`).
///
/// `reynolds` (Re) nombre de Reynolds [sans dimension].
///
/// Panique si `Re ≤ 0`.
pub fn dp_friction_factor_laminar(reynolds: f64) -> f64 {
    assert!(reynolds > 0.0, "Re > 0 requis (nombre de Reynolds)");
    64.0 / reynolds
}

/// Facteur de frottement de Darcy en régime turbulent (corrélation de
/// Swamee-Jain)
/// `f = 0.25 / [ log₁₀( (ε/D)/3.7 + 5.74/Re^0.9 ) ]²` (sans dimension),
/// approximation explicite de l'équation implicite de Colebrook-White
/// (turbulent, `Re > 4000` environ).
///
/// `reynolds` (Re) nombre de Reynolds [sans dimension], `relative_roughness`
/// (ε/D) rugosité relative de la paroi [sans dimension].
///
/// Panique si `Re ≤ 0`, ou si `ε/D < 0`.
pub fn dp_friction_factor_swamee_jain(reynolds: f64, relative_roughness: f64) -> f64 {
    assert!(reynolds > 0.0, "Re > 0 requis (nombre de Reynolds)");
    assert!(
        relative_roughness >= 0.0,
        "ε/D ≥ 0 requis (rugosité relative)"
    );
    let argument = relative_roughness / 3.7 + 5.74 / reynolds.powf(0.9);
    0.25 / argument.log10().powi(2)
}

/// Perte de charge régulière (équation de Darcy-Weisbach)
/// `Δp = f · (L/D) · ρ·v²/2` (Pa), perte de pression par frottement le long
/// d'une conduite droite.
///
/// `friction_factor` (f) facteur de frottement de Darcy [sans dimension],
/// `length` (L) longueur de conduite [m], `diameter` (D) diamètre intérieur
/// [m], `density` (ρ) [kg·m⁻³], `velocity` (v) vitesse débitante [m·s⁻¹].
///
/// Panique si `f ≤ 0`, si `L ≤ 0`, si `D ≤ 0`, si `ρ ≤ 0`, ou si `v ≤ 0`.
pub fn dp_darcy_weisbach(
    friction_factor: f64,
    length: f64,
    diameter: f64,
    density: f64,
    velocity: f64,
) -> f64 {
    assert!(
        friction_factor > 0.0,
        "f > 0 requis (facteur de frottement)"
    );
    assert!(length > 0.0, "L > 0 requis (longueur de conduite)");
    assert!(diameter > 0.0, "D > 0 requis (diamètre intérieur)");
    assert!(density > 0.0, "ρ > 0 requis (masse volumique du fluide)");
    assert!(velocity > 0.0, "v > 0 requis (vitesse débitante)");
    friction_factor * (length / diameter) * density * velocity * velocity / 2.0
}

/// Perte de charge singulière d'un accessoire
/// `Δp_s = K · ρ·v²/2` (Pa), perte de pression localisée d'un coude, d'une
/// vanne, d'un raccord, etc.
///
/// `loss_coefficient` (K) coefficient de perte singulière [sans dimension],
/// `density` (ρ) [kg·m⁻³], `velocity` (v) vitesse débitante [m·s⁻¹].
///
/// Panique si `K < 0`, si `ρ ≤ 0`, ou si `v ≤ 0`.
pub fn dp_minor_loss(loss_coefficient: f64, density: f64, velocity: f64) -> f64 {
    assert!(
        loss_coefficient >= 0.0,
        "K ≥ 0 requis (coefficient de perte singulière)"
    );
    assert!(density > 0.0, "ρ > 0 requis (masse volumique du fluide)");
    assert!(velocity > 0.0, "v > 0 requis (vitesse débitante)");
    loss_coefficient * density * velocity * velocity / 2.0
}

/// Longueur équivalente d'une singularité
/// `L_eq = K · D / f` (m), longueur de conduite droite produisant la même perte
/// de charge que la singularité de coefficient `K`.
///
/// `loss_coefficient` (K) coefficient de perte singulière [sans dimension],
/// `friction_factor` (f) facteur de frottement de Darcy [sans dimension],
/// `diameter` (D) diamètre intérieur [m].
///
/// Panique si `K < 0`, si `f ≤ 0`, ou si `D ≤ 0`.
pub fn dp_equivalent_length(loss_coefficient: f64, friction_factor: f64, diameter: f64) -> f64 {
    assert!(
        loss_coefficient >= 0.0,
        "K ≥ 0 requis (coefficient de perte singulière)"
    );
    assert!(
        friction_factor > 0.0,
        "f > 0 requis (facteur de frottement)"
    );
    assert!(diameter > 0.0, "D > 0 requis (diamètre intérieur)");
    loss_coefficient * diameter / friction_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reynolds_realistic_case_and_linear_in_velocity() {
        // ρ = 1000, v = 2, D = 0.1, μ = 0.001 :
        //   Re = 1000·2·0.1 / 0.001 = 200 / 0.001 = 200000.
        let re = dp_reynolds(1000.0_f64, 2.0_f64, 0.1_f64, 0.001_f64);
        assert_relative_eq!(re, 200000.0, max_relative = 1e-12);
        // Re ∝ v : doubler la vitesse double le nombre de Reynolds.
        let re2 = dp_reynolds(1000.0_f64, 4.0_f64, 0.1_f64, 0.001_f64);
        assert_relative_eq!(re2, 2.0 * re, max_relative = 1e-12);
    }

    #[test]
    fn laminar_friction_factor_and_reciprocity() {
        // À Re = 2000 : f = 64/2000 = 0.032.
        let f = dp_friction_factor_laminar(2000.0_f64);
        assert_relative_eq!(f, 0.032, max_relative = 1e-12);
        // Réciprocité : Re = 64/f redonne le nombre de Reynolds.
        assert_relative_eq!(64.0_f64 / f, 2000.0, max_relative = 1e-12);
    }

    #[test]
    fn swamee_jain_turbulent_realistic_case() {
        // Re = 100000, ε/D = 0.0001 :
        //   Re^0.9 = 10^(5·0.9) = 10^4.5 = 31622.776601
        //   5.74/31622.776601 = 1.8151370e-4
        //   0.0001/3.7 = 2.7027027e-5
        //   somme = 2.0854073e-4
        //   log₁₀(2.0854073e-4) = -3.6808104
        //   (-3.6808104)² = 13.548365
        //   f = 0.25 / 13.548365 = 0.018452.
        let f = dp_friction_factor_swamee_jain(100000.0_f64, 0.0001_f64);
        assert_relative_eq!(f, 0.018452, max_relative = 1e-3);
    }

    #[test]
    fn darcy_weisbach_realistic_case_and_quadratic_in_velocity() {
        // f = 0.02, L = 100, D = 0.1, ρ = 1000, v = 2 :
        //   L/D = 1000 ; ρv²/2 = 1000·4/2 = 2000
        //   Δp = 0.02·1000·2000 = 40000 Pa.
        let dp = dp_darcy_weisbach(0.02_f64, 100.0_f64, 0.1_f64, 1000.0_f64, 2.0_f64);
        assert_relative_eq!(dp, 40000.0, max_relative = 1e-12);
        // Δp ∝ v² : doubler la vitesse quadruple la perte de charge.
        let dp2 = dp_darcy_weisbach(0.02_f64, 100.0_f64, 0.1_f64, 1000.0_f64, 4.0_f64);
        assert_relative_eq!(dp2, 4.0 * dp, max_relative = 1e-12);
    }

    #[test]
    fn equivalent_length_reproduces_minor_loss() {
        // Identité de définition : la perte régulière sur la longueur
        // équivalente L_eq = K·D/f égale exactement la perte singulière K·ρv²/2.
        let (k, f, d, rho, v) = (0.5_f64, 0.02_f64, 0.1_f64, 1000.0_f64, 2.0_f64);
        // L_eq = 0.5·0.1/0.02 = 2.5 m.
        let l_eq = dp_equivalent_length(k, f, d);
        assert_relative_eq!(l_eq, 2.5, max_relative = 1e-12);
        let regular = dp_darcy_weisbach(f, l_eq, d, rho, v);
        let minor = dp_minor_loss(k, rho, v);
        assert_relative_eq!(regular, minor, max_relative = 1e-12);
    }

    #[test]
    fn minor_loss_realistic_case_and_linear_in_coefficient() {
        // K = 0.5, ρ = 1000, v = 2 : Δp_s = 0.5·1000·4/2 = 1000 Pa.
        let dps = dp_minor_loss(0.5_f64, 1000.0_f64, 2.0_f64);
        assert_relative_eq!(dps, 1000.0, max_relative = 1e-12);
        // Δp_s ∝ K : doubler K double la perte singulière.
        let dps2 = dp_minor_loss(1.0_f64, 1000.0_f64, 2.0_f64);
        assert_relative_eq!(dps2, 2.0 * dps, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "μ > 0 requis")]
    fn reynolds_panics_on_nonpositive_viscosity() {
        // μ = 0 ⇒ division par zéro ⇒ entrée rejetée.
        let _ = dp_reynolds(1000.0_f64, 2.0_f64, 0.1_f64, 0.0_f64);
    }
}
