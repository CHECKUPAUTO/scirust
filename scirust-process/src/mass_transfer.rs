//! Transfert de matière interphase selon la **théorie du double film** de
//! Whitman — flux molaire, coefficient global côté gaz (résistances de film en
//! série), force motrice logarithmique moyenne et nombres adimensionnels de
//! **Sherwood** et de **Schmidt**.
//!
//! ```text
//! flux molaire              N   = k_c · ΔC                              [mol·m⁻²·s⁻¹]
//! coeff. global gaz (série) 1/K_G = 1/k_G + m/k_L   ⇒  K_G              [m·s⁻¹]
//! force motrice log. moy.   ΔC_lm = (ΔC₁ − ΔC₂) / ln(ΔC₁/ΔC₂)          [mol·m⁻³]
//! nombre de Sherwood        Sh  = k_c · L / D                           [-]
//! nombre de Schmidt         Sc  = ν / D                                 [-]
//! ```
//!
//! `N` densité de flux molaire [mol·m⁻²·s⁻¹], `k_c` coefficient de transfert de
//! matière (film) [m·s⁻¹], `ΔC` force motrice en concentration [mol·m⁻³],
//! `K_G` coefficient global de transfert côté gaz [m·s⁻¹], `k_G`/`k_L`
//! coefficients de film côté gaz/liquide [m·s⁻¹], `m` pente de la droite de
//! Henry y = m·x reliant les concentrations à l'interface [sans dimension],
//! `ΔC_lm` force motrice logarithmique moyenne [mol·m⁻³], `ΔC₁`/`ΔC₂` forces
//! motrices aux deux extrémités [mol·m⁻³], `L` longueur caractéristique [m],
//! `D` diffusivité (coefficient de diffusion) [m²·s⁻¹], `ν` viscosité
//! cinématique [m²·s⁻¹].
//!
//! **Limite honnête** : modèle du **double film** (deux films stagnants en
//! série de part et d'autre de l'interface, l'**interface étant supposée à
//! l'équilibre**), en **régime permanent** et pour des **solutions diluées**.
//! Les **coefficients de film** (`k_c`, `k_G`, `k_L`), la **diffusivité** `D`,
//! la **viscosité cinématique** `ν` et la **pente de Henry** `m` sont **FOURNIS
//! par l'appelant** (jamais de valeur « par défaut » inventée : ils proviennent
//! de corrélations, de tables ou d'essais). Ce module reste au niveau de
//! l'**opération de transfert interphase** ; il se distingue de la **diffusion
//! moléculaire pure** (loi de Fick), traitée ailleurs.

/// Densité de flux molaire `N = k_c · ΔC` (mol·m⁻²·s⁻¹), premier film.
///
/// `mass_transfer_coefficient` (k_c) [m·s⁻¹] ; `concentration_driving_force`
/// (ΔC) [mol·m⁻³].
///
/// Panique si `mass_transfer_coefficient < 0`.
pub fn masstr_flux(mass_transfer_coefficient: f64, concentration_driving_force: f64) -> f64 {
    assert!(
        mass_transfer_coefficient >= 0.0,
        "k_c ≥ 0 requis (coefficient de transfert de matière)"
    );
    mass_transfer_coefficient * concentration_driving_force
}

/// Coefficient global de transfert de matière côté gaz `K_G` par mise en
/// **série des résistances de film** : `1/K_G = 1/k_G + m/k_L`, d'où
/// `K_G = 1/(1/k_G + m/k_L)` (m·s⁻¹).
///
/// `gas_film_coefficient` (k_G) et `liquid_film_coefficient` (k_L) [m·s⁻¹] ;
/// `henry_slope` (m) pente de la droite d'équilibre y = m·x [sans dimension].
///
/// Panique si `gas_film_coefficient <= 0`, `liquid_film_coefficient <= 0` ou
/// `henry_slope < 0`.
pub fn masstr_overall_coefficient_gas(
    gas_film_coefficient: f64,
    liquid_film_coefficient: f64,
    henry_slope: f64,
) -> f64 {
    assert!(
        gas_film_coefficient > 0.0 && liquid_film_coefficient > 0.0,
        "k_G > 0 et k_L > 0 requis (coefficients de film)"
    );
    assert!(henry_slope >= 0.0, "m ≥ 0 requis (pente de Henry)");
    1.0 / (1.0 / gas_film_coefficient + henry_slope / liquid_film_coefficient)
}

/// Force motrice **logarithmique moyenne**
/// `ΔC_lm = (ΔC₁ − ΔC₂) / ln(ΔC₁/ΔC₂)` (mol·m⁻³), pour une force motrice qui
/// varie entre les deux extrémités du contacteur.
///
/// `driving_force_1` (ΔC₁) et `driving_force_2` (ΔC₂) forces motrices aux deux
/// bouts [mol·m⁻³], toutes deux strictement positives.
///
/// Panique si `driving_force_1 <= 0`, `driving_force_2 <= 0` ou si les deux
/// forces motrices sont trop proches (`ΔC₁ ≈ ΔC₂`, formule singulière : le
/// logarithme s'annule ; utiliser alors leur valeur commune).
pub fn masstr_log_mean_driving_force(driving_force_1: f64, driving_force_2: f64) -> f64 {
    assert!(
        driving_force_1 > 0.0 && driving_force_2 > 0.0,
        "ΔC₁ > 0 et ΔC₂ > 0 requis"
    );
    assert!(
        (driving_force_1 - driving_force_2).abs() > 1.0e-12 * driving_force_1,
        "ΔC₁ ≠ ΔC₂ requis (formule singulière ; sinon ΔC_lm = ΔC commune)"
    );
    (driving_force_1 - driving_force_2) / (driving_force_1 / driving_force_2).ln()
}

/// Nombre de **Sherwood** `Sh = k_c · L / D` (sans dimension), rapport du
/// transfert de matière convectif à la diffusion moléculaire.
///
/// `mass_transfer_coefficient` (k_c) [m·s⁻¹] ; `characteristic_length` (L)
/// [m] ; `diffusivity` (D) [m²·s⁻¹].
///
/// Panique si `mass_transfer_coefficient < 0`, `characteristic_length <= 0` ou
/// `diffusivity <= 0`.
pub fn masstr_sherwood(
    mass_transfer_coefficient: f64,
    characteristic_length: f64,
    diffusivity: f64,
) -> f64 {
    assert!(mass_transfer_coefficient >= 0.0, "k_c ≥ 0 requis");
    assert!(
        characteristic_length > 0.0 && diffusivity > 0.0,
        "L > 0 et D > 0 requis"
    );
    mass_transfer_coefficient * characteristic_length / diffusivity
}

/// Nombre de **Schmidt** `Sc = ν / D` (sans dimension), rapport de la diffusion
/// de quantité de mouvement à la diffusion de matière.
///
/// `kinematic_viscosity` (ν) [m²·s⁻¹] ; `diffusivity` (D) [m²·s⁻¹].
///
/// Panique si `kinematic_viscosity < 0` ou `diffusivity <= 0`.
pub fn masstr_schmidt(kinematic_viscosity: f64, diffusivity: f64) -> f64 {
    assert!(kinematic_viscosity >= 0.0, "ν ≥ 0 requise");
    assert!(diffusivity > 0.0, "D > 0 requise");
    kinematic_viscosity / diffusivity
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn flux_definition_and_reciprocity() {
        // k_c = 0.01 m·s⁻¹, ΔC = 100 mol·m⁻³ ⇒ N = 0.01·100 = 1 mol·m⁻²·s⁻¹.
        let n = masstr_flux(0.01_f64, 100.0_f64);
        assert_relative_eq!(n, 1.0, max_relative = 1e-12);
        // Réciprocité : N / ΔC redonne bien k_c.
        assert_relative_eq!(n / 100.0_f64, 0.01, max_relative = 1e-12);
        // Force motrice nulle ⇒ flux nul.
        assert_relative_eq!(masstr_flux(0.01_f64, 0.0_f64), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn overall_coefficient_series_resistances() {
        // k_G = 0.1, k_L = 0.05, m = 0.5 :
        // 1/K_G = 1/0.1 + 0.5/0.05 = 10 + 10 = 20 ⇒ K_G = 0.05 m·s⁻¹.
        let kg = masstr_overall_coefficient_gas(0.1_f64, 0.05_f64, 0.5_f64);
        assert_relative_eq!(kg, 0.05, max_relative = 1e-12);
        // Identité des résistances en série : 1/K_G = 1/k_G + m/k_L.
        assert_relative_eq!(
            1.0 / kg,
            1.0 / 0.1_f64 + 0.5_f64 / 0.05_f64,
            max_relative = 1e-12
        );
    }

    #[test]
    fn overall_coefficient_gas_controlled_limit() {
        // Résistance liquide négligeable (m/k_L → 0 car m = 0) : K_G → k_G.
        let kg = masstr_overall_coefficient_gas(0.2_f64, 0.001_f64, 0.0_f64);
        assert_relative_eq!(kg, 0.2, max_relative = 1e-12);
    }

    #[test]
    fn log_mean_driving_force_value_and_symmetry() {
        // ΔC₁ = 10, ΔC₂ = 5 : ΔC_lm = (10 − 5)/ln(10/5) = 5/ln 2.
        let lm = masstr_log_mean_driving_force(10.0_f64, 5.0_f64);
        assert_relative_eq!(lm, 5.0_f64 / core::f64::consts::LN_2, max_relative = 1e-12);
        // Symétrie : ΔC_lm(ΔC₁, ΔC₂) = ΔC_lm(ΔC₂, ΔC₁).
        assert_relative_eq!(
            lm,
            masstr_log_mean_driving_force(5.0_f64, 10.0_f64),
            max_relative = 1e-12
        );
        // Encadrement : moyenne géométrique ≤ ΔC_lm ≤ moyenne arithmétique.
        let geo = (10.0_f64 * 5.0_f64).sqrt();
        let arith = (10.0_f64 + 5.0_f64) / 2.0_f64;
        assert!(
            lm >= geo && lm <= arith,
            "ΔC_lm entre moyennes géo. et arith."
        );
    }

    #[test]
    fn sherwood_and_schmidt_realistic_case() {
        // Sh : k_c = 0.02 m·s⁻¹, L = 0.05 m, D = 2·10⁻⁵ m²·s⁻¹.
        // Sh = 0.02·0.05 / 2·10⁻⁵ = 0.001 / 2·10⁻⁵ = 50.
        let sh = masstr_sherwood(0.02_f64, 0.05_f64, 2.0e-5_f64);
        assert_relative_eq!(sh, 50.0, max_relative = 1e-12);
        // Sc pour un gaz : ν = 1.5·10⁻⁵ m²·s⁻¹, D = 2·10⁻⁵ m²·s⁻¹.
        // Sc = 1.5·10⁻⁵ / 2·10⁻⁵ = 0.75 (ordre de grandeur d'un gaz).
        let sc = masstr_schmidt(1.5e-5_f64, 2.0e-5_f64);
        assert_relative_eq!(sc, 0.75, max_relative = 1e-12);
    }

    #[test]
    fn sherwood_proportional_to_length() {
        // À k_c et D fixés, Sh est proportionnel à L : doubler L double Sh.
        let sh1 = masstr_sherwood(0.02_f64, 0.05_f64, 2.0e-5_f64);
        let sh2 = masstr_sherwood(0.02_f64, 0.10_f64, 2.0e-5_f64);
        assert_relative_eq!(sh2, 2.0_f64 * sh1, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "ΔC₁ ≠ ΔC₂ requis")]
    fn log_mean_panics_when_equal() {
        // ΔC₁ = ΔC₂ rend ln(ΔC₁/ΔC₂) = 0 (formule singulière) ⇒ panique.
        let _ = masstr_log_mean_driving_force(7.0_f64, 7.0_f64);
    }
}
