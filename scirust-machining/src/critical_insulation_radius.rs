//! **Rayon critique d'isolation** d'un cylindre (fil, tube) ou d'une sphère en
//! **régime permanent** : au-dessous de ce rayon, ajouter de l'isolant
//! **augmente** les pertes thermiques au lieu de les réduire.
//!
//! ```text
//! rayon critique cylindre   r_c = k / h
//! rayon critique sphère     r_c = 2·k / h
//! sous le rayon critique    r_ext < r_c  ⇒  isoler accroît le flux
//! résistance manchon        R = ln(r_ext / r_int) / (2·π·k·L)
//! ```
//!
//! `k` conductivité de l'isolant (W/(m·K)), `h` coefficient de convection
//! extérieur (W/(m²·K)), `r_c` rayon critique (m), `r_int`/`r_ext` rayons
//! intérieur/extérieur du manchon (m), `L` longueur du cylindre (m),
//! `R` résistance conductive du manchon (K/W).
//!
//! **Convention** : SI cohérent. **Limite honnête** : régime **permanent**, la
//! conductivité `k` de l'isolant et le coefficient de convection extérieur `h`
//! sont **fournis par l'appelant** (supposés **constants**, aucune valeur
//! matériau ou procédé « par défaut » inventée) ; géométrie **cylindrique** ou
//! **sphérique** idéale ; le rayon critique explique pourquoi isoler un fin
//! conducteur (petit rayon) peut **augmenter** la dissipation, car la baisse de
//! la résistance de convection (surface extérieure accrue) l'emporte alors sur
//! la hausse de la résistance de conduction.

use core::f64::consts::PI;

/// Rayon critique d'isolation d'un **cylindre** `r_c = k / h` (m).
///
/// En dessous de `r_c`, augmenter le rayon extérieur de l'isolant fait
/// **croître** le flux de chaleur dissipé.
///
/// Panique si `thermal_conductivity <= 0` ou `convection_coefficient <= 0`.
pub fn insulation_critical_radius_cylinder(
    thermal_conductivity: f64,
    convection_coefficient: f64,
) -> f64 {
    assert!(
        thermal_conductivity > 0.0,
        "la conductivité de l'isolant doit être strictement positive"
    );
    assert!(
        convection_coefficient > 0.0,
        "le coefficient de convection doit être strictement positif"
    );
    thermal_conductivity / convection_coefficient
}

/// Rayon critique d'isolation d'une **sphère** `r_c = 2·k / h` (m).
///
/// Panique si `thermal_conductivity <= 0` ou `convection_coefficient <= 0`.
pub fn insulation_critical_radius_sphere(
    thermal_conductivity: f64,
    convection_coefficient: f64,
) -> f64 {
    assert!(
        thermal_conductivity > 0.0,
        "la conductivité de l'isolant doit être strictement positive"
    );
    assert!(
        convection_coefficient > 0.0,
        "le coefficient de convection doit être strictement positif"
    );
    2.0 * thermal_conductivity / convection_coefficient
}

/// Indique si le rayon extérieur est **sous** le rayon critique (`r_ext < r_c`),
/// c'est-à-dire si ajouter de l'isolant **accroît** encore le flux dissipé.
///
/// Panique si `outer_radius <= 0` ou `critical_radius <= 0`.
pub fn insulation_is_below_critical(outer_radius: f64, critical_radius: f64) -> bool {
    assert!(
        outer_radius > 0.0,
        "le rayon extérieur doit être strictement positif"
    );
    assert!(
        critical_radius > 0.0,
        "le rayon critique doit être strictement positif"
    );
    outer_radius < critical_radius
}

/// Résistance conductive d'un **manchon cylindrique**
/// `R = ln(r_ext / r_int) / (2·π·k·L)` (K/W).
///
/// Panique si `inner_radius <= 0`, `outer_radius <= inner_radius`,
/// `thermal_conductivity <= 0` ou `length <= 0`.
pub fn insulation_cylinder_resistance(
    inner_radius: f64,
    outer_radius: f64,
    thermal_conductivity: f64,
    length: f64,
) -> f64 {
    assert!(
        inner_radius > 0.0,
        "le rayon intérieur doit être strictement positif"
    );
    assert!(
        outer_radius > inner_radius,
        "le rayon extérieur doit être strictement supérieur au rayon intérieur"
    );
    assert!(
        thermal_conductivity > 0.0,
        "la conductivité de l'isolant doit être strictement positive"
    );
    assert!(length > 0.0, "la longueur doit être strictement positive");
    (outer_radius / inner_radius).ln() / (2.0 * PI * thermal_conductivity * length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::E;

    #[test]
    fn rayon_critique_cylindre_cas_chiffre() {
        // k = 0,05 W/(m·K) ; h = 10 W/(m²·K) -> r_c = 0,05/10 = 0,005 m.
        let r_c = insulation_critical_radius_cylinder(0.05, 10.0);
        assert_relative_eq!(r_c, 0.005, max_relative = 1e-12);
    }

    #[test]
    fn sphere_vaut_deux_fois_le_cylindre() {
        // r_c,sphère = 2·k/h = 2·r_c,cylindre pour les mêmes k et h.
        let (k, h) = (0.04_f64, 8.0_f64);
        let cyl = insulation_critical_radius_cylinder(k, h);
        let sph = insulation_critical_radius_sphere(k, h);
        assert_relative_eq!(sph, 2.0 * cyl, max_relative = 1e-12);
    }

    #[test]
    fn proportionnalite_a_la_conductivite() {
        // r_c = k/h est linéaire en k : doubler k double r_c.
        let h = 12.0;
        let base = insulation_critical_radius_cylinder(0.03, h);
        let double = insulation_critical_radius_cylinder(0.06, h);
        assert_relative_eq!(double, 2.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn position_relative_au_rayon_critique() {
        // r_c = 0,005 m : un rayon de 0,003 m est en dessous, 0,010 m au-dessus.
        let r_c = insulation_critical_radius_cylinder(0.05, 10.0);
        assert!(insulation_is_below_critical(0.003, r_c));
        assert!(!insulation_is_below_critical(0.010, r_c));
    }

    #[test]
    fn resistance_manchon_rapport_e() {
        // r_ext/r_int = e -> ln(e) = 1 -> R = 1/(2·π·k·L).
        let (r_int, k, length) = (0.01_f64, 0.05_f64, 2.0_f64);
        let r_ext = r_int * E;
        let r = insulation_cylinder_resistance(r_int, r_ext, k, length);
        assert_relative_eq!(r, 1.0 / (2.0 * PI * k * length), max_relative = 1e-12);
    }

    #[test]
    fn resistance_manchon_additive_par_couches() {
        // Deux manchons concentriques de même k et L : R(a→b) + R(b→c) = R(a→c),
        // car ln(b/a) + ln(c/b) = ln(c/a).
        let (a, b, c) = (0.01_f64, 0.02_f64, 0.05_f64);
        let (k, length) = (0.04_f64, 1.5_f64);
        let r_ab = insulation_cylinder_resistance(a, b, k, length);
        let r_bc = insulation_cylinder_resistance(b, c, k, length);
        let r_ac = insulation_cylinder_resistance(a, c, k, length);
        assert_relative_eq!(r_ab + r_bc, r_ac, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le coefficient de convection doit être strictement positif")]
    fn convection_nulle_panique() {
        let _ = insulation_critical_radius_cylinder(0.05, 0.0);
    }
}
