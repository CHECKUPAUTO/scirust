//! Freins et embrayages — couple transmissible d'un embrayage à disques
//! (frottement) et couple de freinage d'un frein à sangle (bande).
//!
//! ```text
//! embrayage à disques, usure uniforme    C = n·µ·F·(ro + ri)/2
//! embrayage à disques, pression uniforme C = n·(2/3)·µ·F·(ro³ − ri³)/(ro² − ri²)
//! frein à sangle (Euler-Eytelwein)       T1/T2 = e^{µ·θ}
//! couple de freinage                     C = (T1 − T2)·r
//! ```
//!
//! `µ` coefficient de frottement, `F` effort presseur axial (N), `ro`/`ri` rayons
//! extérieur/intérieur de la garniture (m), `n` nombre de surfaces de frottement,
//! `θ` angle d'enroulement de la sangle (rad), `T1`/`T2` tensions brin tendu/mou,
//! `r` rayon du tambour (m).
//!
//! **Convention** : SI cohérent. **Limite honnête** : couples de frottement
//! idéalisés. L'hypothèse d'**usure uniforme** (plus prudente) sous-estime le
//! couple par rapport à la **pression uniforme** (garniture neuve) ; le vrai
//! comportement est intermédiaire. `µ` est fourni par l'appelant.

/// Couple d'un embrayage à disques, hypothèse d'**usure uniforme**
/// `C = n·µ·F·(ro + ri)/2` (N·m).
pub fn disc_clutch_torque_uniform_wear(
    mu: f64,
    axial_force: f64,
    outer_radius: f64,
    inner_radius: f64,
    surfaces: u32,
) -> f64 {
    surfaces as f64 * mu * axial_force * (outer_radius + inner_radius) / 2.0
}

/// Couple d'un embrayage à disques, hypothèse de **pression uniforme**
/// `C = n·(2/3)·µ·F·(ro³ − ri³)/(ro² − ri²)` (N·m).
///
/// Panique si `ro <= ri`.
pub fn disc_clutch_torque_uniform_pressure(
    mu: f64,
    axial_force: f64,
    outer_radius: f64,
    inner_radius: f64,
    surfaces: u32,
) -> f64 {
    assert!(
        outer_radius > inner_radius,
        "le rayon extérieur doit dépasser le rayon intérieur"
    );
    let num = outer_radius.powi(3) - inner_radius.powi(3);
    let den = outer_radius * outer_radius - inner_radius * inner_radius;
    surfaces as f64 * (2.0 / 3.0) * mu * axial_force * num / den
}

/// Rapport des tensions d'une sangle `T1/T2 = e^{µ·θ}` (Euler-Eytelwein).
pub fn band_tension_ratio(mu: f64, wrap_angle_rad: f64) -> f64 {
    (mu * wrap_angle_rad).exp()
}

/// Couple de freinage d'un frein à sangle `C = (T1 − T2)·r` (N·m).
pub fn band_brake_torque(tight_tension: f64, slack_tension: f64, drum_radius: f64) -> f64 {
    (tight_tension - slack_tension) * drum_radius
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn uniform_wear_is_conservative() {
        // À mêmes données, l'usure uniforme donne un couple ≤ pression uniforme.
        let (mu, f, ro, ri) = (0.3, 5000.0, 0.15, 0.10);
        let wear = disc_clutch_torque_uniform_wear(mu, f, ro, ri, 1);
        let press = disc_clutch_torque_uniform_pressure(mu, f, ro, ri, 1);
        assert!(wear <= press);
        // usure : 0,3·5000·(0,25)/2 = 187,5 N·m.
        assert_relative_eq!(wear, 0.3 * 5000.0 * 0.25 / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn multiple_surfaces_scale_linearly() {
        // Un embrayage à 4 surfaces transmet 4× le couple d'une surface.
        let one = disc_clutch_torque_uniform_wear(0.3, 5000.0, 0.15, 0.10, 1);
        let four = disc_clutch_torque_uniform_wear(0.3, 5000.0, 0.15, 0.10, 4);
        assert_relative_eq!(four, 4.0 * one, epsilon = 1e-9);
    }

    #[test]
    fn band_brake_from_capstan_relation() {
        // µ=0,25, θ=π (180°) → T1/T2 = e^{0,25π} ≈ 2,193.
        let ratio = band_tension_ratio(0.25, core::f64::consts::PI);
        assert_relative_eq!(ratio, (0.25 * core::f64::consts::PI).exp(), epsilon = 1e-12);
        // T2=1000, T1=2193, r=0,2 → C = (1193)·0,2 ≈ 238,6 N·m.
        let t1 = 1000.0 * ratio;
        assert_relative_eq!(
            band_brake_torque(t1, 1000.0, 0.2),
            (t1 - 1000.0) * 0.2,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "rayon extérieur")]
    fn inverted_radii_panic() {
        disc_clutch_torque_uniform_pressure(0.3, 5000.0, 0.10, 0.15, 1);
    }
}
