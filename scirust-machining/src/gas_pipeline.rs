//! **Écoulement de gaz en conduite** — débit volumique par l'équation de Weymouth
//! (forme générale de l'écoulement de gaz permanent) et facteur de transmission associé.
//!
//! ```text
//! chute² de pression   Δ = p1² − p2²
//! débit (base)         Q = E·(Tb/Pb)·sqrt((p1²−p2²)·D⁵ / (G·Tf·L·Z))
//! transmission Weymouth E = 11,19·D_mm^(1/6)                 (D_mm en mm)
//! ```
//!
//! `E` facteur de transmission (sans dimension), `Pb` pression de base (Pa), `Tb`
//! température de base (K), `p1` pression amont (Pa), `p2` pression aval (Pa), `G`
//! gravité spécifique du gaz (air = 1, sans dimension), `Tf` température d'écoulement
//! (K), `L` longueur de conduite (m), `D` diamètre intérieur (m), `Z` facteur de
//! compressibilité (sans dimension), `Q` débit volumique ramené aux conditions de base.
//!
//! **Limite honnête** : écoulement **permanent, isotherme et horizontal**, régime
//! **turbulent pleinement rugueux** (facteur de transmission de Weymouth). La gravité
//! spécifique `G`, la température d'écoulement `Tf`, le facteur `Z` et les conditions
//! de base `(Pb, Tb)` sont des **données fournies par l'appelant** — aucune valeur
//! par défaut n'est inventée. La constante `11,19` et l'homogénéité de `Q` dépendent
//! du **système d'unités** : ce module suppose des unités SI cohérentes (Pa, K, m),
//! `D_mm` en **mm** pour le facteur de transmission.

/// Carré de la chute de pression `Δ = p1² − p2²` (Pa²), moteur de l'écoulement.
///
/// Panique si `upstream_pressure <= 0` ou `downstream_pressure <= 0`.
pub fn gaspipe_pressure_drop_squared(upstream_pressure: f64, downstream_pressure: f64) -> f64 {
    assert!(
        upstream_pressure > 0.0 && downstream_pressure > 0.0,
        "les pressions amont et aval doivent être strictement positives"
    );
    upstream_pressure * upstream_pressure - downstream_pressure * downstream_pressure
}

/// Facteur de transmission de Weymouth `E = 11,19·D_mm^(1/6)` (diamètre en mm).
///
/// Panique si `diameter_mm <= 0`.
pub fn gaspipe_weymouth_transmission_factor(diameter_mm: f64) -> f64 {
    assert!(
        diameter_mm > 0.0,
        "le diamètre (mm) doit être strictement positif"
    );
    11.19 * diameter_mm.powf(1.0 / 6.0)
}

/// Débit volumique aux conditions de base par la forme générale de l'écoulement de gaz
/// `Q = E·(Tb/Pb)·sqrt((p1²−p2²)·D⁵ / (G·Tf·L·Z))`.
///
/// Panique si un paramètre positif est `<= 0`, ou si `upstream_pressure < downstream_pressure`.
#[allow(clippy::too_many_arguments)]
pub fn gaspipe_weymouth_flow(
    transmission_factor: f64,
    base_pressure: f64,
    base_temperature: f64,
    upstream_pressure: f64,
    downstream_pressure: f64,
    specific_gravity: f64,
    flowing_temperature: f64,
    length: f64,
    diameter: f64,
    compressibility: f64,
) -> f64 {
    assert!(
        transmission_factor > 0.0,
        "le facteur de transmission doit être strictement positif"
    );
    assert!(
        base_pressure > 0.0 && base_temperature > 0.0,
        "les conditions de base (Pb, Tb) doivent être strictement positives"
    );
    assert!(
        upstream_pressure > 0.0 && downstream_pressure > 0.0,
        "les pressions amont et aval doivent être strictement positives"
    );
    assert!(
        upstream_pressure >= downstream_pressure,
        "la pression amont doit être supérieure ou égale à la pression aval"
    );
    assert!(
        specific_gravity > 0.0 && flowing_temperature > 0.0,
        "la gravité spécifique et la température d'écoulement doivent être > 0"
    );
    assert!(
        length > 0.0 && diameter > 0.0 && compressibility > 0.0,
        "longueur, diamètre et facteur de compressibilité doivent être > 0"
    );

    let delta = upstream_pressure * upstream_pressure - downstream_pressure * downstream_pressure;
    let numerator = delta * diameter.powi(5);
    let denominator = specific_gravity * flowing_temperature * length * compressibility;
    transmission_factor * (base_temperature / base_pressure) * (numerator / denominator).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn transmission_factor_reference_diameter() {
        // D = 1 mm → E = 11,19 ; D = 64 mm = 2⁶ mm → E = 11,19·2 = 22,38.
        assert_relative_eq!(
            gaspipe_weymouth_transmission_factor(1.0),
            11.19,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            gaspipe_weymouth_transmission_factor(64.0),
            22.38,
            epsilon = 1e-9
        );
    }

    #[test]
    fn pressure_drop_squared_identity() {
        // p1 = 10, p2 = 6 → 100 − 36 = 64 ; antisymétrie du signe.
        assert_relative_eq!(
            gaspipe_pressure_drop_squared(10.0, 6.0),
            64.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            gaspipe_pressure_drop_squared(6.0, 10.0),
            -gaspipe_pressure_drop_squared(10.0, 6.0),
            epsilon = 1e-9
        );
    }

    #[test]
    fn flow_vanishes_without_pressure_difference() {
        // p1 = p2 → sqrt(0) = 0, donc débit nul quels que soient les autres paramètres.
        let q = gaspipe_weymouth_flow(
            30.0, 100_000.0, 300.0, 5.0e6, 5.0e6, 0.6, 280.0, 50_000.0, 0.5, 0.9,
        );
        assert_relative_eq!(q, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn flow_is_linear_in_transmission_factor() {
        // Q ∝ E : doubler le facteur de transmission double le débit.
        let base = gaspipe_weymouth_flow(
            30.0, 100_000.0, 300.0, 5.0e6, 3.0e6, 0.6, 280.0, 50_000.0, 0.5, 0.9,
        );
        let doubled = gaspipe_weymouth_flow(
            60.0, 100_000.0, 300.0, 5.0e6, 3.0e6, 0.6, 280.0, 50_000.0, 0.5, 0.9,
        );
        assert_relative_eq!(doubled / base, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn flow_scales_with_diameter_to_the_five_halves() {
        // Q ∝ sqrt(D⁵) = D^2,5 : doubler le diamètre multiplie le débit par 2^2,5.
        let d_small = 0.5_f64;
        let q_small = gaspipe_weymouth_flow(
            30.0, 100_000.0, 300.0, 5.0e6, 3.0e6, 0.6, 280.0, 50_000.0, d_small, 0.9,
        );
        let q_big = gaspipe_weymouth_flow(
            30.0,
            100_000.0,
            300.0,
            5.0e6,
            3.0e6,
            0.6,
            280.0,
            50_000.0,
            2.0 * d_small,
            0.9,
        );
        assert_relative_eq!(q_big / q_small, 2.0_f64.powf(2.5), epsilon = 1e-9);
    }

    #[test]
    fn realistic_case_gives_expected_flow() {
        // Cas chiffré construit pour un radicande égal à 10⁴ (racine = 100) :
        //   Δ = (5e6)² − (3e6)² = 16e12 ; D⁵ = 0,5⁵ = 0,03125 → num = 5e11.
        //   dénom = G·Tf·L·Z = 0,5·250·500000·0,8 = 5e7 → ratio = 1e4, sqrt = 100.
        //   Tb/Pb = 300/100000 = 0,003 ; Q = 30·0,003·100 = 9,0 m³/s.
        let q = gaspipe_weymouth_flow(
            30.0, 100_000.0, 300.0, 5.0e6, 3.0e6, 0.5, 250.0, 500_000.0, 0.5, 0.8,
        );
        assert_relative_eq!(q, 9.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "la pression amont doit être supérieure")]
    fn reversed_pressures_panic() {
        gaspipe_weymouth_flow(
            30.0, 100_000.0, 300.0, 3.0e6, 5.0e6, 0.6, 280.0, 50_000.0, 0.5, 0.9,
        );
    }
}
