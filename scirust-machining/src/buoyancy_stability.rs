//! Stabilité d'un corps flottant — poussée d'Archimède, hauteur métacentrique et
//! moment de redressement aux faibles angles de gîte.
//!
//! ```text
//! poussée d'Archimède      F_b = rho·V·g
//! rayon métacentrique      BM  = I / V
//! hauteur métacentrique    GM  = BM − BG = I/V − BG
//! stabilité initiale       GM > 0  ⇒  équilibre stable
//! moment de redressement   M_r = W·GM·sin(theta)
//! ```
//!
//! `rho` masse volumique du fluide (kg/m³), `V` volume de carène déplacé (m³),
//! `g` accélération de la pesanteur (m/s²), `F_b` poussée d'Archimède (N),
//! `I` moment quadratique du plan de flottaison autour de l'axe de gîte (m⁴),
//! `BM` rayon métacentrique = distance centre de carène → métacentre (m),
//! `BG` distance centre de carène `B` → centre de gravité `G`, positive quand
//! `G` est au-dessus de `B` (m), `GM` hauteur métacentrique (m), `W` poids du
//! corps flottant (N), `theta` angle de gîte (rad), `M_r` moment de redressement
//! ramenant le corps vers sa position d'équilibre (N·m).
//!
//! **Convention** : SI cohérent — masses volumiques en kg/m³, volumes en m³,
//! longueurs en m, moment quadratique en m⁴, forces en N, moments en N·m, angles
//! en radians. `BG > 0` signifie `G` au-dessus de `B`.
//!
//! **Limite honnête** : hydrostatique pure d'un corps flottant **en équilibre** aux
//! **faibles angles de gîte** (théorie métacentrique, `GM` supposé constant). La
//! masse volumique `rho`, le volume déplacé `V`, la pesanteur `g`, le moment
//! quadratique du plan de flottaison `I` et les positions relatives des centres
//! `BG` sont **fournis par l'appelant** — aucune géométrie de coque, densité de
//! fluide ni valeur de `g` « par défaut » n'est inventée ici. Le modèle **néglige
//! toute dynamique** (houle, roulis amorti, effets de carène liquide, grands
//! angles où la courbe `GZ` cesse d'être linéaire).

/// Poussée d'Archimède `F_b = rho·V·g`.
///
/// `fluid_density` = `rho` (kg/m³), `displaced_volume` = `V` (m³),
/// `gravity` = `g` (m/s²) ; renvoie une force verticale ascendante (N).
///
/// Panique si `fluid_density < 0`, `displaced_volume < 0` ou `gravity < 0`.
pub fn buoyancy_force(fluid_density: f64, displaced_volume: f64, gravity: f64) -> f64 {
    assert!(
        fluid_density >= 0.0 && displaced_volume >= 0.0 && gravity >= 0.0,
        "rho ≥ 0, V ≥ 0 et g ≥ 0 requis"
    );
    fluid_density * displaced_volume * gravity
}

/// Hauteur métacentrique `GM = I/V − BG` (avec `BM = I/V`).
///
/// `waterplane_second_moment` = `I` moment quadratique du plan de flottaison (m⁴),
/// `displaced_volume` = `V` volume de carène (m³), `center_of_gravity_to_buoyancy`
/// = `BG` distance `B → G` positive quand `G` est au-dessus de `B` (m) ; renvoie
/// la hauteur métacentrique `GM` (m), positive si le corps est stable.
///
/// Panique si `waterplane_second_moment < 0` ou `displaced_volume <= 0`.
pub fn buoyancy_metacentric_height(
    waterplane_second_moment: f64,
    displaced_volume: f64,
    center_of_gravity_to_buoyancy: f64,
) -> f64 {
    assert!(
        waterplane_second_moment >= 0.0 && displaced_volume > 0.0,
        "I ≥ 0 et V > 0 requis"
    );
    waterplane_second_moment / displaced_volume - center_of_gravity_to_buoyancy
}

/// Critère de stabilité initiale : `true` si `GM > 0` (métacentre au-dessus de `G`).
///
/// `metacentric_height` = `GM` (m) ; renvoie `true` pour un équilibre stable,
/// `false` pour un équilibre indifférent (`GM = 0`) ou instable (`GM < 0`).
///
/// Panique si `metacentric_height` n'est pas un nombre fini (NaN ou infini).
pub fn buoyancy_is_stable(metacentric_height: f64) -> bool {
    assert!(
        metacentric_height.is_finite(),
        "GM doit être un nombre fini"
    );
    metacentric_height > 0.0
}

/// Moment de redressement `M_r = W·GM·sin(theta)`.
///
/// `weight` = `W` poids du corps flottant (N), `metacentric_height` = `GM` (m),
/// `heel_angle_rad` = `theta` angle de gîte (rad) ; renvoie le moment (N·m)
/// tendant à ramener le corps vers l'équilibre lorsque `GM > 0`.
///
/// Panique si `weight < 0`.
pub fn buoyancy_righting_moment(weight: f64, metacentric_height: f64, heel_angle_rad: f64) -> f64 {
    assert!(weight >= 0.0, "W ≥ 0 requis");
    weight * metacentric_height * heel_angle_rad.sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn buoyancy_force_is_trilinear() {
        // F_b ∝ rho, ∝ V et ∝ g : doubler chacun multiplie la poussée par 8.
        let base = buoyancy_force(1000.0, 2.0, 9.81);
        let scaled = buoyancy_force(2000.0, 4.0, 19.62);
        assert_relative_eq!(scaled, 8.0 * base, epsilon = 1e-6);
    }

    #[test]
    fn metacentric_height_matches_bm_minus_bg() {
        // GM = BM − BG avec BM = I/V : identité de définition.
        let (i, v, bg) = (5.0_f64, 10.0_f64, 0.3_f64);
        let bm = i / v;
        let gm = buoyancy_metacentric_height(i, v, bg);
        assert_relative_eq!(gm, bm - bg, epsilon = 1e-12);
        assert_relative_eq!(gm, 0.2, epsilon = 1e-12);
    }

    #[test]
    fn stability_boundary_at_zero_gm() {
        // GM > 0 stable, GM = 0 (indifférent) et GM < 0 non stables.
        assert!(buoyancy_is_stable(0.15));
        assert!(!buoyancy_is_stable(0.0));
        assert!(!buoyancy_is_stable(-0.1));
    }

    #[test]
    fn righting_moment_vanishes_upright() {
        // À gîte nulle, sin(0) = 0 : aucun moment de redressement.
        let m = buoyancy_righting_moment(1.0e5, 0.2, 0.0);
        assert_relative_eq!(m, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn righting_moment_small_angle_linearity() {
        // Aux faibles angles sin(theta) ≈ theta : M_r ≈ W·GM·theta.
        let (w, gm, theta) = (1.0e5_f64, 0.25_f64, 1.0e-4_f64);
        let exact = buoyancy_righting_moment(w, gm, theta);
        assert_relative_eq!(exact, w * gm * theta, max_relative = 1e-6);
    }

    #[test]
    fn realistic_floating_pontoon() {
        // Ponton en eau de mer : rho = 1025 kg/m³, V = 10 m³, g = 9.81 m/s².
        // F_b = 1025·10·9.81 = 100 552.5 N.
        let f_b = buoyancy_force(1025.0, 10.0, 9.81);
        assert_relative_eq!(f_b, 100_552.5, epsilon = 1e-6);
        // Plan de flottaison I = 8 m⁴, V = 10 m³ → BM = 0.8 m ; BG = 0.5 m.
        // GM = 0.8 − 0.5 = 0.3 m (> 0 ⇒ stable).
        let gm = buoyancy_metacentric_height(8.0, 10.0, 0.5);
        assert_relative_eq!(gm, 0.3, epsilon = 1e-12);
        assert!(buoyancy_is_stable(gm));
        // Poids W = F_b (flottaison libre), gîte theta = 5° = PI/36 rad.
        // M_r = 100552.5·0.3·sin(PI/36) = 30165.75·0.0871557427… = 2629.118… N·m.
        let theta = PI / 36.0;
        let m_r = buoyancy_righting_moment(f_b, gm, theta);
        assert_relative_eq!(m_r, f_b * 0.3 * theta.sin(), epsilon = 1e-9);
        assert_relative_eq!(m_r, 2629.118, epsilon = 1e-2);
    }

    #[test]
    #[should_panic(expected = "V > 0")]
    fn zero_displaced_volume_panics() {
        buoyancy_metacentric_height(8.0, 0.0, 0.5);
    }
}
