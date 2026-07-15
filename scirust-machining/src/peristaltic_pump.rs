//! Pompe **péristaltique** (à galets, tube écrasé) — volume d'un bolus emprisonné,
//! cylindrée par tour et débit volumétrique moyen.
//!
//! ```text
//! volume de bolus   Vb = (π/4)·d²·L        (m³)
//! cylindrée         Vd = Vb·nr             (m³/tr)
//! débit moyen       Q  = Vd·N/60           (m³/s)
//! ```
//!
//! `Vb` volume d'un bolus emprisonné entre deux galets (m³), `d` diamètre intérieur
//! du tube (m), `L` longueur d'occlusion — distance parcourue entre deux galets
//! successifs (m), `Vd` cylindrée (volume refoulé par tour, m³/tr), `nr` nombre de
//! galets, `N` vitesse de rotation (tr/min), `Q` débit volumétrique moyen (m³/s).
//!
//! **Convention** : SI cohérent (mètres, secondes) sauf la vitesse exprimée en
//! **tr/min** (d'où le facteur 60). **Limite honnête** : c'est une pompe
//! **volumétrique à déplacement positif** — le débit est fixé par la géométrie et
//! la vitesse, indépendant de la contre-pression **tant que le tube s'occlut
//! complètement**. Le débit obtenu est un débit **moyen** (l'écoulement réel est
//! pulsé, saccadé au passage de chaque galet). La fatigue et la reprise élastique
//! du tube (« spallation », dérive du débit dans le temps) sont **ignorées**. Le
//! diamètre intérieur, la longueur d'occlusion, le nombre de galets et la vitesse
//! sont **fournis par l'appelant** — aucune valeur « par défaut » n'est inventée.

use core::f64::consts::PI;

/// Volume d'un bolus emprisonné entre deux galets `Vb = (π/4)·d²·L` (m³).
///
/// `tube_inner_diameter` diamètre intérieur du tube `d` (m),
/// `occlusion_length` longueur d'occlusion `L` (m).
///
/// Panique si `tube_inner_diameter < 0` ou `occlusion_length < 0`.
pub fn peri_occlusion_volume(tube_inner_diameter: f64, occlusion_length: f64) -> f64 {
    assert!(
        tube_inner_diameter >= 0.0,
        "le diamètre intérieur du tube ne peut pas être négatif"
    );
    assert!(
        occlusion_length >= 0.0,
        "la longueur d'occlusion ne peut pas être négative"
    );
    PI / 4.0 * tube_inner_diameter * tube_inner_diameter * occlusion_length
}

/// Cylindrée refoulée par tour `Vd = Vb·nr` (m³/tr).
///
/// `occlusion_volume` volume d'un bolus `Vb` (m³), `roller_count` nombre de galets
/// `nr` (un bolus est refoulé par galet et par tour).
///
/// Panique si `occlusion_volume < 0` ou `roller_count == 0`.
pub fn peri_displacement_per_revolution(occlusion_volume: f64, roller_count: u32) -> f64 {
    assert!(
        occlusion_volume >= 0.0,
        "le volume de bolus ne peut pas être négatif"
    );
    assert!(
        roller_count > 0,
        "le nombre de galets doit être strictement positif"
    );
    occlusion_volume * roller_count as f64
}

/// Débit volumétrique moyen `Q = Vd·N/60` (m³/s).
///
/// `displacement_per_revolution` cylindrée `Vd` (m³/tr),
/// `rotational_speed_rpm` vitesse de rotation `N` (tr/min).
///
/// Panique si `displacement_per_revolution < 0` ou `rotational_speed_rpm < 0`.
pub fn peri_flow_rate(displacement_per_revolution: f64, rotational_speed_rpm: f64) -> f64 {
    assert!(
        displacement_per_revolution >= 0.0,
        "la cylindrée ne peut pas être négative"
    );
    assert!(
        rotational_speed_rpm >= 0.0,
        "la vitesse de rotation ne peut pas être négative"
    );
    displacement_per_revolution * rotational_speed_rpm / 60.0
}

/// Débit volumétrique moyen directement à partir de la géométrie et de la vitesse
/// `Q = (π/4)·d²·L·nr·N/60` (m³/s) — composition des trois relations précédentes.
///
/// `tube_inner_diameter` diamètre intérieur du tube `d` (m),
/// `occlusion_length` longueur d'occlusion `L` (m),
/// `roller_count` nombre de galets `nr`,
/// `rotational_speed_rpm` vitesse de rotation `N` (tr/min).
///
/// Panique si `tube_inner_diameter < 0`, `occlusion_length < 0`,
/// `roller_count == 0` ou `rotational_speed_rpm < 0`.
pub fn peri_flow_from_geometry(
    tube_inner_diameter: f64,
    occlusion_length: f64,
    roller_count: u32,
    rotational_speed_rpm: f64,
) -> f64 {
    let bolus = peri_occlusion_volume(tube_inner_diameter, occlusion_length);
    let displacement = peri_displacement_per_revolution(bolus, roller_count);
    peri_flow_rate(displacement, rotational_speed_rpm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bolus_volume_matches_cylinder_formula() {
        // Le bolus est un cylindre de diamètre d et de longueur L :
        // Vb = (π/4)·d²·L. Vérification directe de la formule.
        let d = 0.006_f64;
        let l = 0.030_f64;
        let vb = peri_occlusion_volume(d, l);
        assert_relative_eq!(vb, PI / 4.0 * d * d * l, epsilon = 1e-18);
    }

    #[test]
    fn bolus_scales_with_diameter_squared_and_length() {
        // Vb ∝ d² : doubler d quadruple le volume. Vb ∝ L : doubler L le double.
        let base = peri_occlusion_volume(0.004, 0.020);
        assert_relative_eq!(
            peri_occlusion_volume(0.008, 0.020),
            4.0 * base,
            epsilon = 1e-18
        );
        assert_relative_eq!(
            peri_occlusion_volume(0.004, 0.040),
            2.0 * base,
            epsilon = 1e-18
        );
    }

    #[test]
    fn displacement_scales_linearly_with_rollers() {
        // Vd = Vb·nr : passer de 2 à 4 galets double la cylindrée par tour.
        let vb = 8.5e-7_f64;
        let two = peri_displacement_per_revolution(vb, 2);
        let four = peri_displacement_per_revolution(vb, 4);
        assert_relative_eq!(four, 2.0 * two, epsilon = 1e-18);
        assert_relative_eq!(two, 2.0 * vb, epsilon = 1e-18);
    }

    #[test]
    fn flow_rate_reciprocity_with_speed() {
        // Q = Vd·N/60 : le débit est proportionnel à la vitesse.
        // Réciproquement Q·60/N = Vd.
        let vd = 1.7e-6_f64;
        let n = 100.0_f64;
        let q = peri_flow_rate(vd, n);
        assert_relative_eq!(q, vd * n / 60.0, epsilon = 1e-18);
        assert_relative_eq!(q * 60.0 / n, vd, epsilon = 1e-18);
    }

    #[test]
    fn geometry_composition_equals_step_by_step() {
        // peri_flow_from_geometry doit reproduire exactement la composition
        // des trois fonctions élémentaires.
        let d = 0.006_f64;
        let l = 0.030_f64;
        let nr = 3;
        let n = 120.0_f64;
        let bolus = peri_occlusion_volume(d, l);
        let vd = peri_displacement_per_revolution(bolus, nr);
        let q_step = peri_flow_rate(vd, n);
        let q_direct = peri_flow_from_geometry(d, l, nr, n);
        assert_relative_eq!(q_direct, q_step, epsilon = 1e-18);
    }

    #[test]
    fn realistic_flow_value() {
        // Cas chiffré : d = 6 mm, L = 30 mm, 2 galets, 100 tr/min.
        // Vb = (π/4)·(0,006)²·0,030 = (π/4)·1,08×10⁻⁶ ≈ 8,4823×10⁻⁷ m³.
        // Vd = 2·Vb ≈ 1,69646×10⁻⁶ m³/tr.
        // Q  = Vd·100/60 ≈ 2,8274×10⁻⁶ m³/s (≈ 0,1696 L/min).
        let q = peri_flow_from_geometry(0.006, 0.030, 2, 100.0);
        assert_relative_eq!(q, 2.827433_f64 * 1e-6, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "nombre de galets doit être strictement positif")]
    fn zero_rollers_panics() {
        peri_displacement_per_revolution(8.5e-7, 0);
    }
}
