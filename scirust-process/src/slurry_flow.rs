//! Transport hydraulique de solides en conduite horizontale (boues) — masse
//! volumique du mélange, conversion fraction massique → fraction volumique,
//! vitesse critique de dépôt de Durand et excès relatif de perte de charge.
//!
//! ```text
//! masse volumique du mélange
//!   ρ_m  = ρ_s · C_v + ρ_l · (1 − C_v)                          [kg·m⁻³]
//! fraction volumique depuis la fraction massique
//!   C_v  = (w/ρ_s) / [ (w/ρ_s) + (1 − w)/ρ_l ]                  [-]
//! vitesse critique de dépôt (Durand)
//!   V_c  = F_L · sqrt( 2·g·D·(ρ_s − ρ_l)/ρ_l )                  [m·s⁻¹]
//! excès relatif de perte de charge (Durand)
//!   φ_D  = K · [ V_m² / ( g·D·((ρ_s − ρ_l)/ρ_l)·sqrt(C_D) ) ]^(−3/2)  [-]
//! ```
//!
//! `ρ_s` masse volumique du solide [kg·m⁻³], `ρ_l` masse volumique du liquide
//! porteur [kg·m⁻³], `C_v` fraction volumique en solides [sans dimension,
//! 0 ≤ C_v ≤ 1], `w` fraction massique en solides [sans dimension, 0 ≤ w ≤ 1],
//! `ρ_m` masse volumique du mélange (boue) [kg·m⁻³] ; `F_L` facteur empirique de
//! Durand [sans dimension], `D` diamètre intérieur de conduite [m], `g`
//! accélération de la pesanteur [m·s⁻²], `V_c` vitesse critique de dépôt
//! [m·s⁻¹] ; `K` constante empirique de Durand [sans dimension], `V_m` vitesse
//! moyenne (débitante) du mélange [m·s⁻¹], `C_D` coefficient de traînée de la
//! particule [sans dimension], `φ_D` excès relatif de perte de charge du mélange
//! par rapport à celle de l'eau claire [sans dimension].
//!
//! **Limite honnête** : corrélations **empiriques** de Durand pour le transport
//! de boue en conduite **horizontale**. Le facteur `F_L` et la constante `K`
//! dépendent de la **granulométrie** et de la concentration : ils sont
//! **FOURNIS** par l'appelant selon l'abaque de Durand, jamais supposés « par
//! défaut ». Les masses volumiques (`ρ_s`, `ρ_l`) et le coefficient de traînée
//! `C_D` sont eux aussi **FOURNIS** : aucune propriété physique n'est calculée
//! ni inventée ici. La vitesse débitante `V_m` doit rester **supérieure** à la
//! vitesse critique de dépôt `V_c` pour éviter la sédimentation et l'obstruction
//! de la conduite ; ce module ne vérifie pas cette condition, il fournit `V_c`
//! pour la contrôler. Suspension supposée **non colloïdale** (particules
//! grossières, ségrégeables), régime turbulent établi.

/// Masse volumique d'un mélange solide–liquide (boue)
/// `ρ_m = ρ_s · C_v + ρ_l · (1 − C_v)` (kg·m⁻³), moyenne des masses volumiques
/// pondérée par les fractions volumiques.
///
/// `solid_density` (ρ_s) [kg·m⁻³], `liquid_density` (ρ_l) [kg·m⁻³],
/// `solid_volume_fraction` (C_v) fraction volumique en solides [sans dimension].
///
/// Panique si `ρ_s ≤ 0`, si `ρ_l ≤ 0`, ou si `C_v` hors de `[0, 1]`.
pub fn slurry_mixture_density(
    solid_density: f64,
    liquid_density: f64,
    solid_volume_fraction: f64,
) -> f64 {
    assert!(
        solid_density > 0.0,
        "ρ_s > 0 requis (masse volumique du solide)"
    );
    assert!(
        liquid_density > 0.0,
        "ρ_l > 0 requis (masse volumique du liquide)"
    );
    assert!(
        (0.0..=1.0).contains(&solid_volume_fraction),
        "0 ≤ C_v ≤ 1 requis (fraction volumique en solides)"
    );
    solid_density * solid_volume_fraction + liquid_density * (1.0 - solid_volume_fraction)
}

/// Fraction volumique en solides déduite de la fraction massique
/// `C_v = (w/ρ_s) / [ (w/ρ_s) + (1 − w)/ρ_l ]` (sans dimension).
///
/// `mass_fraction` (w) fraction massique en solides [sans dimension],
/// `solid_density` (ρ_s) [kg·m⁻³], `liquid_density` (ρ_l) [kg·m⁻³].
///
/// Panique si `w` hors de `[0, 1]`, si `ρ_s ≤ 0`, ou si `ρ_l ≤ 0`.
pub fn slurry_volume_fraction_from_mass(
    mass_fraction: f64,
    solid_density: f64,
    liquid_density: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&mass_fraction),
        "0 ≤ w ≤ 1 requis (fraction massique en solides)"
    );
    assert!(
        solid_density > 0.0,
        "ρ_s > 0 requis (masse volumique du solide)"
    );
    assert!(
        liquid_density > 0.0,
        "ρ_l > 0 requis (masse volumique du liquide)"
    );
    let solid_term = mass_fraction / solid_density;
    solid_term / (solid_term + (1.0 - mass_fraction) / liquid_density)
}

/// Vitesse critique de dépôt de Durand
/// `V_c = F_L · sqrt( 2·g·D·(ρ_s − ρ_l)/ρ_l )` (m·s⁻¹), vitesse débitante en
/// dessous de laquelle les solides se déposent et forment un lit au fond de la
/// conduite.
///
/// `durand_factor` (F_L) facteur empirique de Durand [sans dimension],
/// `pipe_diameter` (D) diamètre intérieur [m], `gravity` (g) [m·s⁻²],
/// `solid_density` (ρ_s) [kg·m⁻³], `liquid_density` (ρ_l) [kg·m⁻³].
///
/// Panique si `F_L ≤ 0`, si `D ≤ 0`, si `g ≤ 0`, si `ρ_l ≤ 0`, ou si
/// `ρ_s ≤ ρ_l` (aucune force motrice de sédimentation).
pub fn slurry_durand_critical_velocity(
    durand_factor: f64,
    pipe_diameter: f64,
    gravity: f64,
    solid_density: f64,
    liquid_density: f64,
) -> f64 {
    assert!(durand_factor > 0.0, "F_L > 0 requis (facteur de Durand)");
    assert!(pipe_diameter > 0.0, "D > 0 requis (diamètre de conduite)");
    assert!(gravity > 0.0, "g > 0 requis (pesanteur)");
    assert!(
        liquid_density > 0.0,
        "ρ_l > 0 requis (masse volumique du liquide)"
    );
    assert!(
        solid_density > liquid_density,
        "ρ_s > ρ_l requis (solide plus dense que le liquide)"
    );
    durand_factor
        * (2.0 * gravity * pipe_diameter * (solid_density - liquid_density) / liquid_density).sqrt()
}

/// Excès relatif de perte de charge d'une boue en conduite horizontale
/// (corrélation de Durand)
/// `φ_D = K · [ V_m² / ( g·D·((ρ_s − ρ_l)/ρ_l)·sqrt(C_D) ) ]^(−3/2)`
/// (sans dimension), rapport `(i_m − i_l)/(C_v · i_l)` de l'excès de gradient
/// hydraulique de la boue sur celui de l'eau claire.
///
/// `durand_constant` (K) constante empirique de Durand [sans dimension],
/// `mixture_velocity` (V_m) vitesse débitante du mélange [m·s⁻¹],
/// `pipe_diameter` (D) [m], `gravity` (g) [m·s⁻²], `solid_density` (ρ_s)
/// [kg·m⁻³], `liquid_density` (ρ_l) [kg·m⁻³], `drag_coefficient` (C_D)
/// coefficient de traînée de la particule [sans dimension].
///
/// Panique si `K ≤ 0`, si `V_m ≤ 0`, si `D ≤ 0`, si `g ≤ 0`, si `ρ_l ≤ 0`, si
/// `ρ_s ≤ ρ_l`, ou si `C_D ≤ 0`.
pub fn slurry_relative_excess_pressure_gradient(
    durand_constant: f64,
    mixture_velocity: f64,
    pipe_diameter: f64,
    gravity: f64,
    solid_density: f64,
    liquid_density: f64,
    drag_coefficient: f64,
) -> f64 {
    assert!(durand_constant > 0.0, "K > 0 requis (constante de Durand)");
    assert!(
        mixture_velocity > 0.0,
        "V_m > 0 requis (vitesse débitante du mélange)"
    );
    assert!(pipe_diameter > 0.0, "D > 0 requis (diamètre de conduite)");
    assert!(gravity > 0.0, "g > 0 requis (pesanteur)");
    assert!(
        liquid_density > 0.0,
        "ρ_l > 0 requis (masse volumique du liquide)"
    );
    assert!(
        solid_density > liquid_density,
        "ρ_s > ρ_l requis (solide plus dense que le liquide)"
    );
    assert!(
        drag_coefficient > 0.0,
        "C_D > 0 requis (coefficient de traînée)"
    );
    let argument = mixture_velocity * mixture_velocity
        / (gravity
            * pipe_diameter
            * ((solid_density - liquid_density) / liquid_density)
            * drag_coefficient.sqrt());
    durand_constant * argument.powf(-1.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mixture_density_bounds_match_pure_phases() {
        // C_v = 0 ⇒ ρ_m = ρ_l ; C_v = 1 ⇒ ρ_m = ρ_s.
        assert_relative_eq!(
            slurry_mixture_density(2650.0_f64, 1000.0_f64, 0.0_f64),
            1000.0,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            slurry_mixture_density(2650.0_f64, 1000.0_f64, 1.0_f64),
            2650.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn mixture_density_realistic_case() {
        // ρ_s = 2650, ρ_l = 1000, C_v = 0.30 ⇒
        //   ρ_m = 2650·0.30 + 1000·0.70 = 795 + 700 = 1495 kg/m³.
        let rho_m = slurry_mixture_density(2650.0_f64, 1000.0_f64, 0.30_f64);
        assert_relative_eq!(rho_m, 1495.0, max_relative = 1e-12);
    }

    #[test]
    fn volume_fraction_recovers_mass_fraction() {
        // w = 0.5, ρ_s = 2650, ρ_l = 1000 :
        //   C_v = (0.5/2650) / (0.5/2650 + 0.5/1000)
        //       = 1.8867925e-4 / 6.8867925e-4 ≈ 0.273966.
        let w = 0.5_f64;
        let c_v = slurry_volume_fraction_from_mass(w, 2650.0_f64, 1000.0_f64);
        assert_relative_eq!(c_v, 0.273966, max_relative = 1e-3);
        // Réciprocité : la fraction massique se reconstruit par
        //   w = C_v·ρ_s / ρ_m, avec ρ_m la masse volumique du mélange.
        let rho_m = slurry_mixture_density(2650.0_f64, 1000.0_f64, c_v);
        assert_relative_eq!(c_v * 2650.0 / rho_m, w, max_relative = 1e-9);
    }

    #[test]
    fn volume_fraction_extremes() {
        // w = 0 ⇒ C_v = 0 ; w = 1 ⇒ C_v = 1 (indépendant des masses volumiques).
        assert_relative_eq!(
            slurry_volume_fraction_from_mass(0.0_f64, 2650.0_f64, 1000.0_f64),
            0.0,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            slurry_volume_fraction_from_mass(1.0_f64, 2650.0_f64, 1000.0_f64),
            1.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn critical_velocity_realistic_case_and_proportionality() {
        // F_L = 1.34, D = 0.20 m, g = 9.81, ρ_s = 2650, ρ_l = 1000 :
        //   sous racine : 2·9.81·0.20·1650/1000 = 6.4746
        //   sqrt = 2.544523 ; V_c = 1.34·2.544523 ≈ 3.409661 m/s.
        let v_c =
            slurry_durand_critical_velocity(1.34_f64, 0.20_f64, 9.81_f64, 2650.0_f64, 1000.0_f64);
        assert_relative_eq!(v_c, 3.409661, max_relative = 1e-3);
        // V_c ∝ F_L : doubler le facteur double la vitesse critique.
        let v_c2 =
            slurry_durand_critical_velocity(2.68_f64, 0.20_f64, 9.81_f64, 2650.0_f64, 1000.0_f64);
        assert_relative_eq!(v_c2, 2.0 * v_c, max_relative = 1e-12);
    }

    #[test]
    fn excess_gradient_realistic_case_and_linearity_in_constant() {
        // K = 81, V_m = 3, D = 0.20, g = 9.81, ρ_s = 2650, ρ_l = 1000, C_D = 0.5 :
        //   (ρ_s−ρ_l)/ρ_l = 1.65 ; sqrt(C_D) = 0.7071068
        //   dénominateur = 9.81·0.20·1.65·0.7071068 = 2.289119
        //   argument = 9 / 2.289119 = 3.931647
        //   argument^(−1.5) = 1/3.931647^1.5 = 1/7.796627 ≈ 0.1282605
        //   φ_D = 81·0.1282605 ≈ 10.38910.
        let phi_d = slurry_relative_excess_pressure_gradient(
            81.0_f64, 3.0_f64, 0.20_f64, 9.81_f64, 2650.0_f64, 1000.0_f64, 0.5_f64,
        );
        assert_relative_eq!(phi_d, 10.38910, max_relative = 1e-3);
        // φ_D ∝ K : tripler la constante triple l'excès relatif.
        let phi_d3 = slurry_relative_excess_pressure_gradient(
            243.0_f64, 3.0_f64, 0.20_f64, 9.81_f64, 2650.0_f64, 1000.0_f64, 0.5_f64,
        );
        assert_relative_eq!(phi_d3, 3.0 * phi_d, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "0 ≤ C_v ≤ 1 requis")]
    fn mixture_density_panics_on_invalid_fraction() {
        // C_v = 1.5 hors de [0, 1] ⇒ entrée rejetée.
        let _ = slurry_mixture_density(2650.0_f64, 1000.0_f64, 1.5_f64);
    }
}
