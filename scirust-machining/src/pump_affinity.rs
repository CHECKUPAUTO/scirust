//! **Lois de similitude (affinité)** des pompes centrifuges : elles relient débit,
//! hauteur manométrique et puissance d'un même point de fonctionnement lorsqu'on
//! change la vitesse de rotation ou le diamètre de roue à l'intérieur d'une même
//! famille géométrique.
//!
//! ```text
//! variation de vitesse (diamètre constant), rapport n = N2/N1 :
//!   débit       Q2 = Q1 · n
//!   hauteur     H2 = H1 · n²
//!   puissance   P2 = P1 · n³
//! variation de diamètre (vitesse constante), rapport k = D2/D1 :
//!   débit       Q2 = Q1 · k
//! ```
//!
//! `Q` débit volumique (m³/s), `H` hauteur manométrique (m), `P` puissance
//! hydraulique/arbre (W), `N` fréquence de rotation (tr/min ou rad/s, seul le
//! rapport compte), `D` diamètre de la roue (m), `n = N2/N1` rapport de vitesses
//! (sans dimension), `k = D2/D1` rapport de diamètres (sans dimension). Les indices
//! `1`/`2` désignent le point de référence et le point recherché.
//!
//! **Convention** : SI cohérent. Les rapports `n` et `k` sont adimensionnels, donc
//! l'unité de `N` (tr/min, tr/s, rad/s) est indifférente tant qu'elle est la même
//! au numérateur et au dénominateur ; de même l'unité de sortie est celle de la
//! grandeur de référence fournie.
//!
//! **Limite honnête** : les lois d'affinité supposent la **similitude géométrique**
//! (roue de la même famille) et un **rendement constant** entre les deux points ;
//! elles ne valent rigoureusement que pour de **faibles variations** de vitesse ou
//! de diamètre (le rognage de roue et les effets visqueux, de fuite et de NPSH
//! dégradent l'exactitude). Aucune propriété du fluide, de la pompe ou du procédé
//! n'est supposée : les grandeurs de référence et les rapports sont **fournis par
//! l'appelant**, jamais de valeur « par défaut » inventée.

/// Débit après variation de vitesse `Q2 = Q1 · (N2/N1)` (même unité que `reference_flow`).
///
/// Panique si `reference_flow < 0` ou `speed_ratio < 0`.
pub fn pump_affinity_flow(reference_flow: f64, speed_ratio: f64) -> f64 {
    assert!(
        reference_flow >= 0.0,
        "le débit de référence ne peut pas être négatif"
    );
    assert!(
        speed_ratio >= 0.0,
        "le rapport de vitesses ne peut pas être négatif"
    );
    reference_flow * speed_ratio
}

/// Hauteur manométrique après variation de vitesse `H2 = H1 · (N2/N1)²`
/// (même unité que `reference_head`).
///
/// Panique si `reference_head < 0` ou `speed_ratio < 0`.
pub fn pump_affinity_head(reference_head: f64, speed_ratio: f64) -> f64 {
    assert!(
        reference_head >= 0.0,
        "la hauteur manométrique de référence ne peut pas être négative"
    );
    assert!(
        speed_ratio >= 0.0,
        "le rapport de vitesses ne peut pas être négatif"
    );
    reference_head * speed_ratio.powi(2)
}

/// Puissance après variation de vitesse `P2 = P1 · (N2/N1)³`
/// (même unité que `reference_power`).
///
/// Panique si `reference_power < 0` ou `speed_ratio < 0`.
pub fn pump_affinity_power(reference_power: f64, speed_ratio: f64) -> f64 {
    assert!(
        reference_power >= 0.0,
        "la puissance de référence ne peut pas être négative"
    );
    assert!(
        speed_ratio >= 0.0,
        "le rapport de vitesses ne peut pas être négatif"
    );
    reference_power * speed_ratio.powi(3)
}

/// Débit après variation de diamètre de roue à vitesse constante
/// `Q2 = Q1 · (D2/D1)` (même unité que `reference_flow`).
///
/// Panique si `reference_flow < 0` ou `diameter_ratio < 0`.
pub fn pump_affinity_impeller_flow(reference_flow: f64, diameter_ratio: f64) -> f64 {
    assert!(
        reference_flow >= 0.0,
        "le débit de référence ne peut pas être négatif"
    );
    assert!(
        diameter_ratio >= 0.0,
        "le rapport de diamètres ne peut pas être négatif"
    );
    reference_flow * diameter_ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn unit_ratio_is_identity() {
        // Rapport de vitesses n = 1 : aucun point ne change (cas limite).
        assert_relative_eq!(pump_affinity_flow(0.05, 1.0), 0.05, epsilon = 1e-15);
        assert_relative_eq!(pump_affinity_head(32.0, 1.0), 32.0, epsilon = 1e-15);
        assert_relative_eq!(pump_affinity_power(4200.0, 1.0), 4200.0, epsilon = 1e-15);
        assert_relative_eq!(
            pump_affinity_impeller_flow(0.05, 1.0),
            0.05,
            epsilon = 1e-15
        );
    }

    #[test]
    fn head_and_power_track_flow_exponents() {
        // Identité entre exposants : à vitesse variable, si le débit est multiplié
        // par n, la hauteur l'est par n² et la puissance par n³.
        let (q1, h1, p1) = (0.04_f64, 25.0_f64, 3000.0_f64);
        let n = 1.15_f64;
        let q2 = pump_affinity_flow(q1, n);
        let h2 = pump_affinity_head(h1, n);
        let p2 = pump_affinity_power(p1, n);
        assert_relative_eq!(h2 / h1, (q2 / q1).powi(2), epsilon = 1e-12);
        assert_relative_eq!(p2 / p1, (q2 / q1).powi(3), epsilon = 1e-12);
    }

    #[test]
    fn reciprocity_forward_then_back() {
        // Réciprocité : passer de N1 à N2 (rapport n) puis revenir (rapport 1/n)
        // restitue le point de départ, pour chacune des trois grandeurs.
        let n = 1.3_f64;
        let inv = 1.0 / n;
        assert_relative_eq!(
            pump_affinity_flow(pump_affinity_flow(0.02, n), inv),
            0.02,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            pump_affinity_head(pump_affinity_head(18.0, n), inv),
            18.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            pump_affinity_power(pump_affinity_power(1500.0, n), inv),
            1500.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn power_ratio_is_cube_of_speed_ratio() {
        // Proportionnalité cubique explicite : P2/P1 = n³.
        let n = 0.8_f64;
        let ratio = pump_affinity_power(2000.0, n) / 2000.0;
        assert_relative_eq!(ratio, n.powi(3), epsilon = 1e-12);
    }

    #[test]
    fn worked_case_speed_increase() {
        // Point de référence à N1 = 1450 tr/min : Q1 = 0,050 m³/s, H1 = 30 m,
        // P1 = 20 000 W. Nouveau régime N2 = 1740 tr/min, soit n = 1740/1450 = 1,2.
        // Q2 = 0,050·1,2 = 0,060 m³/s
        // H2 = 30·1,2² = 30·1,44 = 43,2 m
        // P2 = 20000·1,2³ = 20000·1,728 = 34 560 W
        let n = 1740.0 / 1450.0;
        assert_relative_eq!(pump_affinity_flow(0.050, n), 0.060, epsilon = 1e-12);
        assert_relative_eq!(pump_affinity_head(30.0, n), 43.2, epsilon = 1e-12);
        assert_relative_eq!(pump_affinity_power(20_000.0, n), 34_560.0, epsilon = 1e-9);
    }

    #[test]
    fn impeller_flow_scales_linearly_with_diameter() {
        // À vitesse constante, le débit est proportionnel au rapport de diamètres :
        // roue rognée de 250 mm à 225 mm, k = 0,9 → Q2 = 0,040·0,9 = 0,036 m³/s.
        let k = 225.0 / 250.0;
        assert_relative_eq!(
            pump_affinity_impeller_flow(0.040, k),
            0.036,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "rapport de vitesses ne peut pas être négatif")]
    fn negative_speed_ratio_panics() {
        pump_affinity_head(30.0, -0.5);
    }
}
