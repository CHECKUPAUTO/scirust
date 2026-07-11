//! Engrenages cylindriques droits — géométrie de la denture à développante de
//! cercle (système métrique au **module**, proportions normalisées ISO 53 /
//! ISO 21771) et contrainte de flexion en pied de dent (modèle de **Lewis**).
//!
//! Le module `m` (mm) est le paramètre dimensionnant : le diamètre primitif
//! d'une roue à `z` dents vaut `d = m·z`. Les proportions de la denture
//! normale à développante (angle de pression `α`, saillie `ha = m`, creux
//! `hf = 1,25·m`) en découlent.
//!
//! L'effort tangentiel transmis se déduit du couple ou de la puissance :
//!
//! ```text
//! Ft = 2·T / d           (T en N·m, d en m ⇒ Ft en N)
//! Ft = P / v             (P en W, v vitesse tangentielle en m/s)
//! ```
//!
//! La contrainte de flexion en pied de dent (Lewis) :
//!
//! ```text
//! σ = Ft / (b · m · Y)
//! ```
//!
//! avec `b` la largeur de denture (mm) et `Y` le facteur de forme de Lewis
//! (sans dimension, tabulé selon `z` et `α`).
//!
//! **Limite honnête** : la géométrie est exacte pour une denture droite
//! standard sans déport. Le facteur de forme `Y` est une donnée tabulée que
//! l'appelant fournit (il dépend de `z` et de la géométrie d'outil) — la crate
//! ne l'invente pas. Le modèle de Lewis est un premier dimensionnement statique :
//! il ignore la concentration de contrainte en pied (facteur de Lewis modifié
//! `J`), les effets dynamiques et l'usure/grippage (pression de Hertz) que
//! traite un calcul complet type ISO 6336.

use core::f64::consts::PI;

/// Roue cylindrique droite à développante, définie par son module, son nombre
/// de dents et son angle de pression.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpurGear {
    /// Module `m` (mm).
    pub module_mm: f64,
    /// Nombre de dents `z`.
    pub teeth: u32,
    /// Angle de pression `α` (degrés), typiquement 20°.
    pub pressure_angle_deg: f64,
}

impl SpurGear {
    /// Diamètre primitif `d = m·z` (mm).
    pub fn pitch_diameter(&self) -> f64 {
        self.module_mm * self.teeth as f64
    }

    /// Saillie `ha = m` (mm).
    pub fn addendum(&self) -> f64 {
        self.module_mm
    }

    /// Creux `hf = 1,25·m` (mm).
    pub fn dedendum(&self) -> f64 {
        1.25 * self.module_mm
    }

    /// Hauteur de dent `h = ha + hf = 2,25·m` (mm).
    pub fn tooth_depth(&self) -> f64 {
        self.addendum() + self.dedendum()
    }

    /// Diamètre de tête `da = d + 2·ha = m·(z + 2)` (mm).
    pub fn tip_diameter(&self) -> f64 {
        self.pitch_diameter() + 2.0 * self.addendum()
    }

    /// Diamètre de pied `df = d − 2·hf = m·(z − 2,5)` (mm).
    pub fn root_diameter(&self) -> f64 {
        self.pitch_diameter() - 2.0 * self.dedendum()
    }

    /// Diamètre de base `db = d·cos α` (mm).
    pub fn base_diameter(&self) -> f64 {
        self.pitch_diameter() * self.pressure_angle_deg.to_radians().cos()
    }

    /// Pas primitif `p = π·m` (mm).
    pub fn circular_pitch(&self) -> f64 {
        PI * self.module_mm
    }

    /// Pas de base `pb = p·cos α` (mm).
    pub fn base_pitch(&self) -> f64 {
        self.circular_pitch() * self.pressure_angle_deg.to_radians().cos()
    }
}

/// Entraxe `a = m·(z₁ + z₂)/2` (mm) de deux roues **de même module**.
///
/// Panique si les modules diffèrent (l'engrènement l'exige).
pub fn center_distance(g1: &SpurGear, g2: &SpurGear) -> f64 {
    assert!(
        (g1.module_mm - g2.module_mm).abs() < 1e-9,
        "l'engrènement exige un module commun"
    );
    g1.module_mm * (g1.teeth + g2.teeth) as f64 / 2.0
}

/// Rapport de réduction `i = z_menée / z_menante` (sans dimension).
///
/// Panique si `driver_teeth == 0`.
pub fn gear_ratio(driver_teeth: u32, driven_teeth: u32) -> f64 {
    assert!(driver_teeth > 0, "la roue menante a au moins une dent");
    driven_teeth as f64 / driver_teeth as f64
}

/// Vitesse tangentielle au primitif `v = π·d·n / 60000` (m/s), diamètre
/// primitif `pitch_diameter` (mm) et rotation `n` (tr/min).
pub fn pitch_line_velocity_m_s(pitch_diameter_mm: f64, n_rpm: f64) -> f64 {
    PI * pitch_diameter_mm * n_rpm / 60_000.0
}

/// Effort tangentiel `Ft` (N) transmis pour un couple `torque` (N·m) au
/// diamètre primitif `pitch_diameter` (mm) : `Ft = 2·T / d` (d converti en m).
///
/// Panique si `pitch_diameter <= 0`.
pub fn tangential_force_from_torque(torque_nm: f64, pitch_diameter_mm: f64) -> f64 {
    assert!(
        pitch_diameter_mm > 0.0,
        "le diamètre primitif doit être strictement positif"
    );
    2.0 * torque_nm / (pitch_diameter_mm / 1000.0)
}

/// Effort tangentiel `Ft` (N) pour une puissance `power` (kW) et une vitesse
/// tangentielle `velocity` (m/s) : `Ft = P / v`.
///
/// Panique si `velocity <= 0`.
pub fn tangential_force_from_power(power_kw: f64, velocity_m_s: f64) -> f64 {
    assert!(
        velocity_m_s > 0.0,
        "la vitesse tangentielle doit être positive"
    );
    power_kw * 1000.0 / velocity_m_s
}

/// Contrainte de flexion en pied de dent (Lewis), en **MPa** :
/// `σ = Ft / (b·m·Y)`, effort tangentiel `ft` (N), largeur de denture
/// `face_width` (mm), module `module` (mm) et facteur de forme `lewis_y`.
///
/// Comme `Ft` est en N et `b·m` en mm², le résultat est directement en
/// N/mm² = MPa. Panique si un dénominateur est non strictement positif.
pub fn lewis_bending_stress(ft_n: f64, face_width_mm: f64, module_mm: f64, lewis_y: f64) -> f64 {
    assert!(
        face_width_mm > 0.0 && module_mm > 0.0 && lewis_y > 0.0,
        "largeur, module et facteur de Lewis doivent être strictement positifs"
    );
    ft_n / (face_width_mm * module_mm * lewis_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn pinion() -> SpurGear {
        SpurGear {
            module_mm: 2.0,
            teeth: 20,
            pressure_angle_deg: 20.0,
        }
    }

    #[test]
    fn pitch_and_tip_and_root_follow_the_module() {
        let g = pinion();
        // d = 2·20 = 40 mm.
        assert_relative_eq!(g.pitch_diameter(), 40.0, epsilon = 1e-12);
        // da = m(z+2) = 2·22 = 44 mm.
        assert_relative_eq!(g.tip_diameter(), 44.0, epsilon = 1e-12);
        // df = m(z−2,5) = 2·17,5 = 35 mm.
        assert_relative_eq!(g.root_diameter(), 35.0, epsilon = 1e-12);
        // h = 2,25·m = 4,5 mm.
        assert_relative_eq!(g.tooth_depth(), 4.5, epsilon = 1e-12);
    }

    #[test]
    fn base_diameter_uses_the_pressure_angle() {
        // db = 40·cos20° ≈ 37,588 mm.
        assert_relative_eq!(
            pinion().base_diameter(),
            40.0 * 20f64.to_radians().cos(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn center_distance_is_the_half_sum_of_pitch_diameters() {
        let g1 = pinion();
        let g2 = SpurGear {
            module_mm: 2.0,
            teeth: 40,
            pressure_angle_deg: 20.0,
        };
        // a = 2·(20+40)/2 = 60 mm = (40+80)/2.
        assert_relative_eq!(center_distance(&g1, &g2), 60.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "module commun")]
    fn center_distance_rejects_mismatched_modules() {
        let g1 = pinion();
        let g2 = SpurGear {
            module_mm: 3.0,
            teeth: 40,
            pressure_angle_deg: 20.0,
        };
        center_distance(&g1, &g2);
    }

    #[test]
    fn ratio_is_driven_over_driver() {
        // 20 → 40 dents : rapport 2.
        assert_relative_eq!(gear_ratio(20, 40), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn tangential_force_from_torque_and_power_agree() {
        // 40 mm primitif à 1000 tr/min : v = π·40·1000/60000 ≈ 2,094 m/s.
        let d = 40.0;
        let n = 1000.0;
        let v = pitch_line_velocity_m_s(d, n);
        // Couple exact pour 5 kW : T = P/ω, ω = 2πn/60. Les deux voies (couple
        // et puissance) doivent alors donner exactement le même Ft.
        let omega = 2.0 * PI * n / 60.0;
        let torque = 5000.0 / omega;
        let ft_torque = tangential_force_from_torque(torque, d);
        let ft_power = tangential_force_from_power(5.0, v);
        assert_relative_eq!(ft_torque, ft_power, epsilon = 1e-9);
    }

    #[test]
    fn lewis_stress_is_force_over_b_m_y() {
        // Ft=2387 N, b=20 mm, m=2 mm, Y=0,32 → σ ≈ 186,5 MPa.
        assert_relative_eq!(
            lewis_bending_stress(2387.0, 20.0, 2.0, 0.32),
            2387.0 / (20.0 * 2.0 * 0.32),
            epsilon = 1e-9
        );
    }
}
