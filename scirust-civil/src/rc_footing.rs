//! **Béton armé — semelle isolée (Eurocode 2 / géotechnique)** : contrainte de
//! sol sous charge centrée ou excentrée (résultante dans le tiers central), aire
//! de semelle requise vis-à-vis de la pression admissible, et moment de flexion
//! en console à la face du poteau à partir de la pression nette du sol.
//!
//! ```text
//! pression centrée    σ    = N / A
//! pression max excentrée   σmax = (N / (L·B)) · (1 + 6·e/L)
//! aire requise        Areq = N / σadm
//! moment console      M    = σnet · B · a² / 2
//! ```
//!
//! `σ` contrainte de sol sous charge centrée (Pa), `N` = `axial_load` effort
//! axial transmis par le poteau (N), `A` = `foundation_area` aire de la semelle
//! (m²), `σmax` = contrainte maximale sous charge excentrée (Pa), `L` =
//! `foundation_length` longueur de la semelle dans le sens de l'excentricité (m),
//! `B` = `foundation_width` largeur de la semelle (m), `e` = `eccentricity`
//! excentricité de la résultante (m), `Areq` aire de semelle requise (m²), `σadm`
//! = `allowable_bearing_pressure` pression admissible du sol (Pa), `M` moment de
//! flexion à la face du poteau (N·m), `σnet` = `net_soil_pressure` pression nette
//! du sol (Pa), `a` = `cantilever_length` porte-à-faux de la console mesuré depuis
//! la face du poteau (m).
//!
//! **Convention** : SI strict — **N, m, Pa** (avec `1 Pa = 1 N/m²`). Les efforts
//! sont en **newtons**, les longueurs et aires en **mètres** et **mètres carrés**,
//! les contraintes et pressions en **pascals**, les moments en **newton-mètres**.
//!
//! **Limite honnête** : semelle supposée **rigide** avec une répartition
//! **linéaire** de la contrainte de sol, valable uniquement lorsque la résultante
//! reste dans le **tiers central** (`e ≤ L/6`, absence de soulèvement). L'effort
//! axial `N` et la pression admissible `σadm` sont **fournis par l'appelant** ; le
//! moment de flexion est calculé en **console** à partir de la pression nette du
//! sol. Ce module **ne dimensionne pas** les armatures (voir `rc_beam_flexure`) ni
//! le **poinçonnement** (voir `rc_punching`). Les résistances caractéristiques et
//! les coefficients partiels de sécurité (`γc`, `γs`, `γM`… ou le coefficient
//! global de sol) sont **fournis par l'appelant** d'après l'Eurocode 2, l'Eurocode
//! 7 et leurs Annexes Nationales ; aucune valeur « par défaut » n'est inventée.

/// Contrainte de sol sous charge centrée `σ = N / A` (Pa), avec `N` en N et `A`
/// en m².
///
/// Panique si `axial_load < 0` ou si `foundation_area <= 0` (division par zéro).
pub fn rcfoot_soil_pressure_centric(axial_load: f64, foundation_area: f64) -> f64 {
    assert!(axial_load >= 0.0, "l'effort axial N doit être ≥ 0");
    assert!(
        foundation_area > 0.0,
        "l'aire de semelle A doit être strictement positive"
    );
    axial_load / foundation_area
}

/// Contrainte maximale de sol sous charge excentrée (résultante dans le tiers
/// central) `σmax = (N / (L·B)) · (1 + 6·e/L)` (Pa), avec `N` en N, `L` et `B` en
/// m et `e` en m.
///
/// Panique si `axial_load < 0`, si `foundation_length <= 0`, si
/// `foundation_width <= 0`, si `eccentricity < 0` ou si `eccentricity` sort du
/// tiers central (`eccentricity > foundation_length / 6`, soulèvement).
pub fn rcfoot_soil_pressure_eccentric_max(
    axial_load: f64,
    foundation_length: f64,
    foundation_width: f64,
    eccentricity: f64,
) -> f64 {
    assert!(axial_load >= 0.0, "l'effort axial N doit être ≥ 0");
    assert!(
        foundation_length > 0.0,
        "la longueur de semelle L doit être strictement positive"
    );
    assert!(
        foundation_width > 0.0,
        "la largeur de semelle B doit être strictement positive"
    );
    assert!(eccentricity >= 0.0, "l'excentricité e doit être ≥ 0");
    assert!(
        eccentricity <= foundation_length / 6.0,
        "l'excentricité e doit rester dans le tiers central (e ≤ L/6)"
    );
    (axial_load / (foundation_length * foundation_width))
        * (1.0 + 6.0 * eccentricity / foundation_length)
}

/// Aire de semelle requise `Areq = N / σadm` (m²), avec `N` en N et `σadm` en Pa.
///
/// Panique si `axial_load < 0` ou si `allowable_bearing_pressure <= 0` (division
/// par zéro).
pub fn rcfoot_required_area(axial_load: f64, allowable_bearing_pressure: f64) -> f64 {
    assert!(axial_load >= 0.0, "l'effort axial N doit être ≥ 0");
    assert!(
        allowable_bearing_pressure > 0.0,
        "la pression admissible σadm doit être strictement positive"
    );
    axial_load / allowable_bearing_pressure
}

/// Moment de flexion à la face du poteau, en console `M = σnet · B · a² / 2`
/// (N·m), avec `σnet` en Pa, `B` en m et `a` en m.
///
/// Panique si `net_soil_pressure < 0`, si `cantilever_length < 0` ou si
/// `width <= 0`.
pub fn rcfoot_cantilever_moment(net_soil_pressure: f64, cantilever_length: f64, width: f64) -> f64 {
    assert!(
        net_soil_pressure >= 0.0,
        "la pression nette du sol σnet doit être ≥ 0"
    );
    assert!(
        cantilever_length >= 0.0,
        "le porte-à-faux a de la console doit être ≥ 0"
    );
    assert!(width > 0.0, "la largeur B doit être strictement positive");
    net_soil_pressure * width * cantilever_length * cantilever_length / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn centric_pressure_is_load_over_area() {
        // Cas chiffré : N = 1 000 kN sur une semelle 2 m × 2 m (A = 4 m²).
        //   σ = 1 000 000 / 4 = 250 000 Pa = 250 kPa.
        let sigma = rcfoot_soil_pressure_centric(1_000_000.0, 4.0);
        assert_relative_eq!(sigma, 250_000.0, epsilon = 1e-6);
    }

    #[test]
    fn centric_pressure_is_proportional_to_load() {
        // Proportionnalité : doubler l'effort double la contrainte.
        let a = 3.5_f64;
        let s1 = rcfoot_soil_pressure_centric(600_000.0, a);
        let s2 = rcfoot_soil_pressure_centric(1_200_000.0, a);
        assert_relative_eq!(s2, 2.0 * s1, epsilon = 1e-9);
    }

    #[test]
    fn eccentric_reduces_to_centric_when_no_eccentricity() {
        // Identité : avec e = 0, σmax = N/(L·B), soit la pression centrée sur
        // l'aire A = L·B.
        let (n, l, b) = (1_000_000.0, 2.0, 2.0);
        let smax = rcfoot_soil_pressure_eccentric_max(n, l, b, 0.0);
        let scentric = rcfoot_soil_pressure_centric(n, l * b);
        assert_relative_eq!(smax, scentric, epsilon = 1e-9);
    }

    #[test]
    fn eccentric_max_matches_hand_calculation() {
        // Cas chiffré : N = 1 000 kN, L = 2 m, B = 2 m, e = 0,2 m (≤ L/6 = 0,333 m).
        //   σmax = (1 000 000 / 4) · (1 + 6·0,2/2)
        //        = 250 000 · (1 + 0,6) = 250 000 · 1,6 = 400 000 Pa.
        let smax = rcfoot_soil_pressure_eccentric_max(1_000_000.0, 2.0, 2.0, 0.2);
        assert_relative_eq!(smax, 400_000.0, epsilon = 1e-6);
    }

    #[test]
    fn required_area_is_reciprocal_of_centric_pressure() {
        // Réciprocité : la pression centrée sur l'aire requise redonne σadm.
        let (n, sadm) = (1_000_000.0, 200_000.0);
        let area = rcfoot_required_area(n, sadm);
        assert_relative_eq!(area, 5.0, epsilon = 1e-9);
        assert_relative_eq!(rcfoot_soil_pressure_centric(n, area), sadm, epsilon = 1e-6);
    }

    #[test]
    fn cantilever_moment_matches_hand_calculation_and_scales_quadratically() {
        // Cas chiffré : σnet = 250 kPa, a = 0,75 m, B = 2 m.
        //   M = 250 000 · 2 · 0,75² / 2 = 250 000 · 2 · 0,562 5 / 2 = 140 625 N·m.
        let m = rcfoot_cantilever_moment(250_000.0, 0.75, 2.0);
        assert_relative_eq!(m, 140_625.0, epsilon = 1e-6);
        // Dépendance quadratique : doubler le porte-à-faux quadruple le moment.
        let m2 = rcfoot_cantilever_moment(250_000.0, 1.5, 2.0);
        assert_relative_eq!(m2, 4.0 * m, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "l'excentricité e doit rester dans le tiers central")]
    fn eccentric_rejects_eccentricity_outside_middle_third() {
        // e = 0,4 m > L/6 = 0,333 m : soulèvement, répartition linéaire invalide.
        rcfoot_soil_pressure_eccentric_max(1_000_000.0, 2.0, 2.0, 0.4);
    }
}
