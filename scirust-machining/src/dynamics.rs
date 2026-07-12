//! Dynamique du solide en rotation — énergie cinétique, moments d'inertie des
//! solides usuels, théorème de Huygens, principe fondamental (`M = J·α`) et
//! puissance.
//!
//! Grandeurs de base (unités SI) :
//!
//! ```text
//! énergie cinétique   Ec = ½·m·v²   (translation)   ,   ½·J·ω²   (rotation)
//! moment cinétique    L = J·ω
//! PFD (rotation)      M = J·α
//! puissance           P = M·ω
//! Huygens (Steiner)   J = J_G + m·d²
//! ```
//!
//! `J` moment d'inertie (kg·m²) par rapport à l'axe considéré, `m` masse (kg),
//! `ω` vitesse angulaire (rad/s), `α` accélération angulaire (rad/s²), `M`
//! moment/couple (N·m), `d` distance entre axes parallèles (m).
//!
//! Les moments d'inertie fournis sont ceux des solides homogènes usuels par
//! rapport à leur axe naturel ; le théorème de **Huygens** ([`parallel_axis`])
//! les transporte vers tout axe parallèle.
//!
//! **Convention** : SI cohérent (kg, m, s, rad). **Limite honnête** : solides
//! **homogènes** de géométrie idéale ; pour un solide réel/composite, l'appelant
//! somme les contributions (les inerties par rapport à un même axe s'ajoutent) et
//! recentre chacune par Huygens.

/// Énergie cinétique de translation `Ec = ½·m·v²` (J), masse `mass` (kg) et
/// vitesse `velocity` (m/s).
pub fn kinetic_energy_translation(mass_kg: f64, velocity_m_s: f64) -> f64 {
    0.5 * mass_kg * velocity_m_s * velocity_m_s
}

/// Énergie cinétique de rotation `Ec = ½·J·ω²` (J), inertie `inertia` (kg·m²)
/// et vitesse angulaire `omega` (rad/s).
pub fn kinetic_energy_rotation(inertia_kgm2: f64, omega_rad_s: f64) -> f64 {
    0.5 * inertia_kgm2 * omega_rad_s * omega_rad_s
}

/// Moment cinétique `L = J·ω` (kg·m²/s).
pub fn angular_momentum(inertia_kgm2: f64, omega_rad_s: f64) -> f64 {
    inertia_kgm2 * omega_rad_s
}

/// Moment (couple) requis par le principe fondamental en rotation `M = J·α`
/// (N·m), inertie `inertia` (kg·m²) et accélération angulaire `alpha` (rad/s²).
pub fn torque_from_angular_accel(inertia_kgm2: f64, alpha_rad_s2: f64) -> f64 {
    inertia_kgm2 * alpha_rad_s2
}

/// Puissance en rotation `P = M·ω` (W), moment `torque` (N·m) et vitesse
/// angulaire `omega` (rad/s).
pub fn rotational_power(torque_nm: f64, omega_rad_s: f64) -> f64 {
    torque_nm * omega_rad_s
}

/// Théorème de Huygens (Steiner) : moment d'inertie `J = J_G + m·d²` (kg·m²)
/// par rapport à un axe parallèle à l'axe barycentrique, distant de `distance`
/// (m).
///
/// Panique si `mass < 0`.
pub fn parallel_axis(inertia_center_kgm2: f64, mass_kg: f64, distance_m: f64) -> f64 {
    assert!(mass_kg >= 0.0, "la masse doit être positive ou nulle");
    inertia_center_kgm2 + mass_kg * distance_m * distance_m
}

/// Moment d'inertie d'un **cylindre plein** homogène par rapport à son axe :
/// `J = ½·m·r²` (kg·m²).
pub fn inertia_solid_cylinder(mass_kg: f64, radius_m: f64) -> f64 {
    0.5 * mass_kg * radius_m * radius_m
}

/// Moment d'inertie d'un **cylindre creux** (tube) homogène par rapport à son
/// axe : `J = ½·m·(r_int² + r_ext²)` (kg·m²).
///
/// Panique si `outer < inner`.
pub fn inertia_hollow_cylinder(mass_kg: f64, inner_radius_m: f64, outer_radius_m: f64) -> f64 {
    assert!(
        outer_radius_m >= inner_radius_m,
        "le rayon extérieur doit dépasser l'intérieur"
    );
    0.5 * mass_kg * (inner_radius_m * inner_radius_m + outer_radius_m * outer_radius_m)
}

/// Moment d'inertie d'un **anneau mince** (cerceau) par rapport à son axe :
/// `J = m·r²` (kg·m²).
pub fn inertia_thin_ring(mass_kg: f64, radius_m: f64) -> f64 {
    mass_kg * radius_m * radius_m
}

/// Moment d'inertie d'une **sphère pleine** homogène par rapport à un diamètre :
/// `J = (2/5)·m·r²` (kg·m²).
pub fn inertia_solid_sphere(mass_kg: f64, radius_m: f64) -> f64 {
    0.4 * mass_kg * radius_m * radius_m
}

/// Moment d'inertie d'une **barre mince** par rapport à un axe transversal
/// passant par son **centre** : `J = (1/12)·m·L²` (kg·m²).
pub fn inertia_rod_center(mass_kg: f64, length_m: f64) -> f64 {
    mass_kg * length_m * length_m / 12.0
}

/// Moment d'inertie d'une **barre mince** par rapport à un axe transversal
/// passant par une **extrémité** : `J = (1/3)·m·L²` (kg·m²).
pub fn inertia_rod_end(mass_kg: f64, length_m: f64) -> f64 {
    mass_kg * length_m * length_m / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn translation_kinetic_energy() {
        // m=2 kg, v=3 m/s → ½·2·9 = 9 J.
        assert_relative_eq!(kinetic_energy_translation(2.0, 3.0), 9.0, epsilon = 1e-12);
    }

    #[test]
    fn rotation_kinetic_energy() {
        // J=0,01 kg·m², ω=100 rad/s → ½·0,01·10000 = 50 J.
        assert_relative_eq!(kinetic_energy_rotation(0.01, 100.0), 50.0, epsilon = 1e-12);
    }

    #[test]
    fn solid_cylinder_inertia() {
        // m=2, r=0,1 → ½·2·0,01 = 0,01 kg·m².
        assert_relative_eq!(inertia_solid_cylinder(2.0, 0.1), 0.01, epsilon = 1e-12);
    }

    #[test]
    fn sphere_and_ring_and_hollow_forms() {
        // sphère (2/5)mr² ; anneau mr² ; tube ½m(ri²+re²).
        assert_relative_eq!(
            inertia_solid_sphere(5.0, 0.2),
            0.4 * 5.0 * 0.04,
            epsilon = 1e-12
        );
        assert_relative_eq!(inertia_thin_ring(3.0, 0.5), 3.0 * 0.25, epsilon = 1e-12);
        assert_relative_eq!(
            inertia_hollow_cylinder(4.0, 0.1, 0.2),
            0.5 * 4.0 * (0.01 + 0.04),
            epsilon = 1e-12
        );
    }

    #[test]
    fn huygens_maps_rod_center_to_rod_end() {
        // J_end = J_center + m·(L/2)² : (1/12 + 1/4)mL² = (1/3)mL².
        let (m, l) = (2.0, 1.0);
        let j_end_via_huygens = parallel_axis(inertia_rod_center(m, l), m, l / 2.0);
        assert_relative_eq!(j_end_via_huygens, inertia_rod_end(m, l), epsilon = 1e-12);
    }

    #[test]
    fn pfd_torque_momentum_and_power() {
        // M = J·α ; L = J·ω ; P = M·ω.
        assert_relative_eq!(torque_from_angular_accel(0.01, 50.0), 0.5, epsilon = 1e-12);
        assert_relative_eq!(angular_momentum(0.01, 100.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(rotational_power(0.5, 100.0), 50.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "rayon extérieur")]
    fn hollow_cylinder_rejects_inverted_radii() {
        inertia_hollow_cylinder(4.0, 0.2, 0.1);
    }
}
