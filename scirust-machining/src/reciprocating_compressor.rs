//! **Compresseur alternatif à piston** : cylindrée, rendement volumétrique avec
//! réexpansion de l'espace mort, débit aspiré (FAD) et puissance indiquée
//! polytropique, pour une compression réversible mono-étagée d'un gaz parfait.
//!
//! ```text
//! cylindrée (volume balayé par le piston) :
//!   Vs = (π/4) · D² · L
//! rendement volumétrique (réexpansion de l'espace mort, indice n) :
//!   ηv = 1 + c − c · r^(1/n)
//! débit aspiré aux conditions d'entrée (FAD) :
//!   Q = Vs · ηv · N / 60
//! puissance indiquée polytropique :
//!   Pi = [n/(n−1)] · p1 · Q · ( r^((n−1)/n) − 1 )
//! ```
//!
//! `D` alésage (m), `L` course (m), `Vs` cylindrée (m³), `c` taux d'espace mort
//! (rapport volume mort / cylindrée, adimensionnel), `r` taux de compression
//! `p2/p1` (adimensionnel), `n` indice polytropique (adimensionnel), `ηv`
//! rendement volumétrique (adimensionnel), `N` fréquence de rotation (tr/min),
//! `Q` débit volumique aspiré (m³/s), `p1` pression d'aspiration (Pa), `Pi`
//! puissance indiquée (W).
//!
//! **Convention** : unités SI cohérentes (m, m³, Pa, W) ; la vitesse de rotation
//! est exprimée en tr/min et convertie en tr/s par le facteur 60.
//!
//! **Limite honnête** : modèle de compression **polytropique réversible avec
//! espace mort**, gaz parfait, **une seule étape**. Le taux d'espace mort `c` et
//! l'indice polytropique `n` sont **fournis par l'appelant** ; aucune valeur
//! « par défaut » (gaz, matériau, procédé) n'est inventée. Ce modèle ne prend en
//! compte **ni les pertes aux soupapes**, **ni le refroidissement intermédiaire**,
//! ni les fuites au segment ou l'échauffement à l'aspiration.

use core::f64::consts::PI;

/// Cylindrée (volume balayé) `Vs = (π/4)·D²·L` d'un cylindre à piston (m³).
///
/// `bore` alésage `D` (m), `stroke` course `L` (m).
///
/// Panique si `bore <= 0` ou `stroke <= 0`.
pub fn recipcomp_swept_volume(bore: f64, stroke: f64) -> f64 {
    assert!(bore > 0.0, "l'alésage doit être strictement positif");
    assert!(stroke > 0.0, "la course doit être strictement positive");
    PI / 4.0 * bore * bore * stroke
}

/// Rendement volumétrique avec réexpansion de l'espace mort
/// `ηv = 1 + c − c·r^(1/n)` (adimensionnel).
///
/// `clearance_ratio` taux d'espace mort `c` (≥ 0), `pressure_ratio` taux de
/// compression `r = p2/p1` (≥ 1), `polytropic_index` indice polytropique `n`
/// de la réexpansion.
///
/// Panique si `clearance_ratio < 0`, `pressure_ratio < 1`
/// ou `polytropic_index <= 0`.
pub fn recipcomp_volumetric_efficiency(
    clearance_ratio: f64,
    pressure_ratio: f64,
    polytropic_index: f64,
) -> f64 {
    assert!(
        clearance_ratio >= 0.0,
        "le taux d'espace mort ne peut pas être négatif"
    );
    assert!(
        pressure_ratio >= 1.0,
        "le taux de compression doit être supérieur ou égal à 1"
    );
    assert!(
        polytropic_index > 0.0,
        "l'indice polytropique doit être strictement positif"
    );
    1.0 + clearance_ratio - clearance_ratio * pressure_ratio.powf(1.0 / polytropic_index)
}

/// Débit aspiré aux conditions d'entrée (FAD)
/// `Q = Vs·ηv·N/60` (m³/s) pour un compresseur à simple effet.
///
/// `swept_volume` cylindrée `Vs` (m³), `volumetric_efficiency` rendement
/// volumétrique `ηv` (≥ 0), `rotational_speed_rpm` fréquence de rotation `N`
/// (tr/min).
///
/// Panique si `swept_volume < 0`, `volumetric_efficiency < 0`
/// ou `rotational_speed_rpm < 0`.
pub fn recipcomp_free_air_delivery(
    swept_volume: f64,
    volumetric_efficiency: f64,
    rotational_speed_rpm: f64,
) -> f64 {
    assert!(
        swept_volume >= 0.0,
        "la cylindrée ne peut pas être négative"
    );
    assert!(
        volumetric_efficiency >= 0.0,
        "le rendement volumétrique ne peut pas être négatif"
    );
    assert!(
        rotational_speed_rpm >= 0.0,
        "la vitesse de rotation ne peut pas être négative"
    );
    swept_volume * volumetric_efficiency * rotational_speed_rpm / 60.0
}

/// Puissance indiquée polytropique
/// `Pi = [n/(n−1)]·p1·Q·(r^((n−1)/n) − 1)` (W).
///
/// `inlet_pressure` pression d'aspiration `p1` (Pa), `volume_flow` débit aspiré
/// `Q` (m³/s), `pressure_ratio` taux de compression `r = p2/p1` (≥ 1),
/// `polytropic_index` indice polytropique `n` (> 1).
///
/// Panique si `inlet_pressure <= 0`, `volume_flow < 0`, `pressure_ratio < 1`
/// ou `polytropic_index <= 1`.
pub fn recipcomp_indicated_power(
    inlet_pressure: f64,
    volume_flow: f64,
    pressure_ratio: f64,
    polytropic_index: f64,
) -> f64 {
    assert!(
        inlet_pressure > 0.0,
        "la pression d'aspiration doit être strictement positive"
    );
    assert!(volume_flow >= 0.0, "le débit ne peut pas être négatif");
    assert!(
        pressure_ratio >= 1.0,
        "le taux de compression doit être supérieur ou égal à 1"
    );
    assert!(
        polytropic_index > 1.0,
        "l'indice polytropique doit être strictement supérieur à 1"
    );
    (polytropic_index / (polytropic_index - 1.0))
        * inlet_pressure
        * volume_flow
        * (pressure_ratio.powf((polytropic_index - 1.0) / polytropic_index) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn swept_volume_worked_case() {
        // D = 2 m, L = 1 m : Vs = (π/4)·2²·1 = (π/4)·4 = π (exact).
        assert_relative_eq!(recipcomp_swept_volume(2.0, 1.0), PI, epsilon = 1e-12);
    }

    #[test]
    fn swept_volume_scales_with_square_of_bore() {
        // Vs ∝ D² : doubler l'alésage quadruple la cylindrée (course fixe).
        let base = recipcomp_swept_volume(0.05, 0.08);
        assert_relative_eq!(
            recipcomp_swept_volume(0.10, 0.08),
            4.0 * base,
            epsilon = 1e-12
        );
    }

    #[test]
    fn volumetric_efficiency_unit_when_no_compression() {
        // Cas limite r = 1 : r^(1/n) = 1 donc ηv = 1 + c − c = 1, quel que soit c.
        assert_relative_eq!(
            recipcomp_volumetric_efficiency(0.07, 1.0, 1.3),
            1.0,
            epsilon = 1e-12
        );
        // Cas limite c = 0 (pas d'espace mort) : ηv = 1 quel que soit r.
        assert_relative_eq!(
            recipcomp_volumetric_efficiency(0.0, 8.0, 1.3),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn volumetric_efficiency_worked_case() {
        // c = 0,05 ; r = 8 ; n = 1,3.
        // r^(1/n) = 8^(1/1,3) ≈ 4,9509 ; ηv = 1 + 0,05 − 0,05·4,9509 ≈ 0,80246.
        assert_relative_eq!(
            recipcomp_volumetric_efficiency(0.05, 8.0, 1.3),
            0.8025,
            epsilon = 1e-3
        );
    }

    #[test]
    fn free_air_delivery_worked_case() {
        // Vs = 0,001 m³ ; ηv = 0,8 ; N = 600 tr/min.
        // Q = 0,001·0,8·600/60 = 0,001·0,8·10 = 0,008 m³/s (exact).
        assert_relative_eq!(
            recipcomp_free_air_delivery(0.001, 0.8, 600.0),
            0.008,
            epsilon = 1e-12
        );
    }

    #[test]
    fn indicated_power_worked_case_and_zero_limit() {
        // p1 = 1e5 Pa ; Q = 0,008 m³/s ; r = 8 ; n = 1,3.
        // (n−1)/n = 0,3/1,3 ≈ 0,23077 ; r^… = 8^0,23077 ≈ 1,61587.
        // Pi = (1,3/0,3)·1e5·0,008·(1,61587 − 1) ≈ 4,33333·800·0,61587 ≈ 2135 W.
        assert_relative_eq!(
            recipcomp_indicated_power(1.0e5, 0.008, 8.0, 1.3),
            2135.0,
            max_relative = 1e-3
        );
        // Cas limite r = 1 : r^((n−1)/n) − 1 = 0 donc Pi = 0.
        assert_relative_eq!(
            recipcomp_indicated_power(1.0e5, 0.008, 1.0, 1.3),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn indicated_power_scales_linearly_with_flow_and_pressure() {
        // Pi ∝ Q et Pi ∝ p1 à taux et indice fixés.
        let base = recipcomp_indicated_power(1.0e5, 0.008, 8.0, 1.3);
        assert_relative_eq!(
            recipcomp_indicated_power(1.0e5, 0.016, 8.0, 1.3),
            2.0 * base,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            recipcomp_indicated_power(2.0e5, 0.008, 8.0, 1.3),
            2.0 * base,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "l'indice polytropique doit être strictement supérieur à 1")]
    fn indicated_power_unit_index_panics() {
        recipcomp_indicated_power(1.0e5, 0.008, 8.0, 1.0);
    }
}
