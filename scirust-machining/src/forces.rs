//! Effort et puissance de coupe — modèle de **Kienzle** (loi de puissance de
//! l'effort spécifique), le modèle de référence en productique pour estimer
//! l'effort de coupe `Fc` à partir de la section de copeau.
//!
//! L'effort spécifique de coupe `kc` (N/mm²) suit une loi de puissance de
//! l'épaisseur de copeau non déformée `h` (mm) :
//!
//! ```text
//! kc(h) = kc1.1 · h^(-mc)
//! ```
//!
//! `kc1.1` est l'effort spécifique de référence (à `h = b = 1 mm`) et `mc` (ou
//! `zc`) l'exposant de Kienzle, tous deux propres au couple outil/matière. La
//! force de coupe est alors l'effort spécifique multiplié par la section de
//! copeau `A = b·h` (largeur × épaisseur) :
//!
//! ```text
//! Fc = kc(h) · b · h = kc1.1 · b · h^(1 - mc)         (N)
//! ```
//!
//! La puissance de coupe s'en déduit par `Pc = Fc · Vc` :
//!
//! ```text
//! Pc = Fc · Vc / 60000        (kW, avec Fc en N et Vc en m/min)
//! ```
//!
//! **Limite honnête** : `kc1.1` et `mc` ne sont **pas** fournis par ce module —
//! ce sont des données matériau tabulées (par ex. ~1500–2200 N/mm² et
//! `mc ≈ 0,2–0,3` pour les aciers) que l'appelant renseigne d'après un
//! catalogue ou des essais. Le modèle de Kienzle ne rend compte que de la
//! composante principale `Fc` ; les composantes d'avance et de pénétration, la
//! géométrie d'arête réelle, l'usure et l'écrouissage exigent des modèles plus
//! riches, hors périmètre ici.

/// Modèle de Kienzle d'un couple outil/matière : effort spécifique de référence
/// `kc1.1` (N/mm²) et exposant `mc` (sans dimension, typiquement 0,15–0,35).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KienzleModel {
    /// Effort spécifique de coupe de référence `kc1.1` (N/mm²), à h = b = 1 mm.
    pub kc11: f64,
    /// Exposant de Kienzle `mc` (parfois noté `zc`), sans dimension.
    pub mc: f64,
}

impl KienzleModel {
    /// Effort spécifique de coupe `kc` (N/mm²) pour une épaisseur de copeau
    /// `chip_thickness` (mm) : `kc = kc1.1 · h^(-mc)`.
    ///
    /// Panique si `chip_thickness <= 0`.
    pub fn specific_cutting_force(&self, chip_thickness_mm: f64) -> f64 {
        assert!(
            chip_thickness_mm > 0.0,
            "l'épaisseur de copeau doit être strictement positive"
        );
        self.kc11 * chip_thickness_mm.powf(-self.mc)
    }

    /// Effort de coupe `Fc` (N) pour une largeur de copeau `width` (mm) et une
    /// épaisseur `chip_thickness` (mm) : `Fc = kc1.1 · b · h^(1 - mc)`.
    pub fn cutting_force(&self, width_mm: f64, chip_thickness_mm: f64) -> f64 {
        self.specific_cutting_force(chip_thickness_mm) * width_mm * chip_thickness_mm
    }

    /// Effort de coupe `Fc` (N) en **tournage** à partir des paramètres
    /// machine et de l'angle de direction d'arête `kappa` (κr, en degrés).
    ///
    /// La section de copeau `A = ap·f` est décomposée en largeur et épaisseur
    /// réelles vues par l'arête :
    ///
    /// ```text
    /// h = f · sin(κr)          b = ap / sin(κr)          A = b·h = ap·f
    /// ```
    ///
    /// Pour l'outil « couteau » usuel (κr = 90°) on retrouve `h = f`,
    /// `b = ap`. Panique si `kappa` n'est pas dans `]0°, 180°[`.
    pub fn cutting_force_turning(
        &self,
        depth_of_cut_mm: f64,
        feed_per_rev_mm: f64,
        kappa_deg: f64,
    ) -> f64 {
        assert!(
            kappa_deg > 0.0 && kappa_deg < 180.0,
            "l'angle de direction d'arête doit être dans ]0°, 180°["
        );
        let s = kappa_deg.to_radians().sin();
        let h = feed_per_rev_mm * s;
        let b = depth_of_cut_mm / s;
        self.cutting_force(b, h)
    }
}

/// Puissance de coupe `Pc` (kW) pour un effort `fc` (N) et une vitesse de coupe
/// `vc` (m/min) : `Pc = Fc·Vc / 60000`.
pub fn cutting_power_kw(fc_n: f64, vc_m_min: f64) -> f64 {
    fc_n * vc_m_min / 60_000.0
}

/// Puissance moteur `Pm` (kW) requise pour délivrer une puissance de coupe
/// `pc` (kW) à travers une transmission de rendement `efficiency` (0 < η ≤ 1) :
/// `Pm = Pc / η`.
///
/// Panique si `efficiency` n'est pas dans `]0, 1]`.
pub fn motor_power_kw(pc_kw: f64, efficiency: f64) -> f64 {
    assert!(
        efficiency > 0.0 && efficiency <= 1.0,
        "le rendement doit être dans ]0, 1]"
    );
    pc_kw / efficiency
}

/// Couple de broche `Mc` (N·m) délivrant une puissance `power` (kW) à une
/// fréquence de rotation `n` (tr/min) : `Mc = 9550 · P / N`.
///
/// La constante 9550 = 60000/(2π) convertit kW et tr/min en N·m. Panique si
/// `n <= 0`.
pub fn spindle_torque_nm(power_kw: f64, n_rpm: f64) -> f64 {
    assert!(n_rpm > 0.0, "la fréquence de rotation doit être positive");
    9550.0 * power_kw / n_rpm
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn steel() -> KienzleModel {
        // Ordre de grandeur d'un acier de construction.
        KienzleModel {
            kc11: 1700.0,
            mc: 0.25,
        }
    }

    #[test]
    fn specific_force_at_unit_thickness_is_kc11() {
        // À h = 1 mm, h^(-mc) = 1 → kc = kc1.1.
        assert_relative_eq!(steel().specific_cutting_force(1.0), 1700.0, epsilon = 1e-9);
    }

    #[test]
    fn specific_force_rises_as_chip_thins() {
        // kc(0,1) = 1700 · 0,1^(-0,25) = 1700 · 10^0,25 ≈ 3022,9 N/mm².
        assert_relative_eq!(
            steel().specific_cutting_force(0.1),
            1700.0 * 10f64.powf(0.25),
            epsilon = 1e-6
        );
    }

    #[test]
    fn cutting_force_equals_specific_force_times_section() {
        let m = steel();
        let b = 3.0;
        let h = 0.2;
        let expected = m.specific_cutting_force(h) * b * h;
        assert_relative_eq!(m.cutting_force(b, h), expected, epsilon = 1e-9);
    }

    #[test]
    fn turning_at_90_degrees_uses_feed_as_thickness() {
        // À κr = 90°, sin = 1 → h = f, b = ap : identique à cutting_force(ap, f).
        let m = steel();
        let direct = m.cutting_force(3.0, 0.2);
        let turning = m.cutting_force_turning(3.0, 0.2, 90.0);
        assert_relative_eq!(turning, direct, epsilon = 1e-9);
    }

    #[test]
    fn turning_conserves_the_chip_section() {
        // Quel que soit κr, b·h = ap·f, donc Fc = kc(h)·ap·f reste cohérent.
        let m = steel();
        let f = m.cutting_force_turning(4.0, 0.25, 60.0);
        // Reconstruction manuelle : h = 0,25·sin60, b = 4/sin60.
        let s = 60f64.to_radians().sin();
        let expected = m.specific_cutting_force(0.25 * s) * (4.0 / s) * (0.25 * s);
        assert_relative_eq!(f, expected, epsilon = 1e-9);
    }

    #[test]
    fn cutting_power_scales_force_and_speed() {
        // Fc=1000 N, Vc=120 m/min → 1000·120/60000 = 2 kW.
        assert_relative_eq!(cutting_power_kw(1000.0, 120.0), 2.0, epsilon = 1e-9);
    }

    #[test]
    fn motor_power_divides_by_efficiency() {
        // 2 kW de coupe à 80 % de rendement → 2,5 kW moteur.
        assert_relative_eq!(motor_power_kw(2.0, 0.8), 2.5, epsilon = 1e-9);
    }

    #[test]
    fn torque_matches_the_9550_relation() {
        // 5 kW à 1000 tr/min → 9550·5/1000 = 47,75 N·m.
        assert_relative_eq!(spindle_torque_nm(5.0, 1000.0), 47.75, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "rendement")]
    fn efficiency_above_one_panics() {
        motor_power_kw(2.0, 1.2);
    }
}
