//! **Vanne de fond** — écoulement dénoyé sous une vanne plane : débit soutiré,
//! veine contractée aval, coefficient de décharge et poussée sur la vanne.
//!
//! ```text
//! débit sous la vanne        Q  = Cd·b·a·√(2·g·H)
//! profondeur contractée      yc = Cc·a
//! coefficient de décharge    Cd = Cc / √(1 + Cc·a/H)
//! poussée nette sur vanne    F  = (1/2)·ρ·g·b·(H₁² − H₂²)
//! ```
//!
//! `Q` débit volumique (m³/s), `Cd` coefficient de décharge (sans dimension),
//! `Cc` coefficient de contraction (sans dimension), `b` largeur de la vanne (m),
//! `a` ouverture de la vanne (m), `H` charge amont mesurée hors zone
//! d'abaissement au-dessus du seuil (m), `yc` profondeur de la veine contractée
//! aval (m), `H₁` charge amont et `H₂` charge aval sur la vanne (m), `ρ` masse
//! volumique du fluide (kg/m³), `g` accélération de la pesanteur (m/s²),
//! `F` poussée hydrostatique nette sur la vanne (N).
//!
//! **Convention** : SI. **Limite honnête** : vanne **plane à écoulement
//! dénoyé** (jet libre en aval), **écoulement permanent**, charge amont `H`
//! mesurée **hors de la zone d'abaissement** de la surface libre. Les
//! coefficients empiriques de décharge `Cd` et de contraction `Cc`, ainsi que
//! la masse volumique `ρ` et la pesanteur `g`, sont **fournis par l'appelant**
//! et ne sont jamais supposés. La poussée `F` est la résultante hydrostatique
//! nette des colonnes d'eau amont et aval (répartition de pression triangulaire,
//! composante dynamique du jet négligée). Distinct de [`crate::weir_flow`]
//! (écoulement par-dessus un seuil, à surface libre).

/// Débit soutiré **sous la vanne** `Q = Cd·b·a·√(2·g·H)` (m³/s).
///
/// Le débit est proportionnel à l'ouverture et croît comme la racine de la
/// charge amont (application de Torricelli à la section ouverte).
///
/// Panique si `discharge_coefficient <= 0`, `gate_width <= 0`,
/// `gate_opening <= 0`, `upstream_head < 0` ou `gravity <= 0`.
pub fn sluice_discharge(
    discharge_coefficient: f64,
    gate_width: f64,
    gate_opening: f64,
    upstream_head: f64,
    gravity: f64,
) -> f64 {
    assert!(
        discharge_coefficient > 0.0,
        "le coefficient de décharge Cd doit être > 0"
    );
    assert!(gate_width > 0.0, "la largeur de vanne b doit être > 0");
    assert!(gate_opening > 0.0, "l'ouverture de vanne a doit être > 0");
    assert!(upstream_head >= 0.0, "la charge amont H doit être ≥ 0");
    assert!(gravity > 0.0, "la pesanteur g doit être > 0");
    discharge_coefficient * gate_width * gate_opening * (2.0 * gravity * upstream_head).sqrt()
}

/// Profondeur de la **veine contractée** aval `yc = Cc·a` (m).
///
/// La contraction verticale du jet sous la vanne réduit l'ouverture géométrique
/// `a` d'un facteur `Cc < 1`.
///
/// Panique si `contraction_coefficient <= 0`, `contraction_coefficient > 1`
/// ou `gate_opening <= 0`.
pub fn sluice_contracted_depth(contraction_coefficient: f64, gate_opening: f64) -> f64 {
    assert!(
        contraction_coefficient > 0.0 && contraction_coefficient <= 1.0,
        "le coefficient de contraction Cc doit être dans ]0, 1]"
    );
    assert!(gate_opening > 0.0, "l'ouverture de vanne a doit être > 0");
    contraction_coefficient * gate_opening
}

/// Coefficient de décharge `Cd = Cc / √(1 + Cc·a/H)` à partir de la contraction (sans dimension).
///
/// Tient compte de la profondeur de la veine contractée dans le bilan de
/// Bernoulli : lorsque l'ouverture devient négligeable devant la charge
/// (`a/H → 0`), `Cd` tend vers `Cc`.
///
/// Panique si `contraction_coefficient` n'est pas dans `]0, 1]`,
/// `gate_opening <= 0` ou `upstream_head <= 0`.
pub fn sluice_discharge_coefficient(
    contraction_coefficient: f64,
    gate_opening: f64,
    upstream_head: f64,
) -> f64 {
    assert!(
        contraction_coefficient > 0.0 && contraction_coefficient <= 1.0,
        "le coefficient de contraction Cc doit être dans ]0, 1]"
    );
    assert!(gate_opening > 0.0, "l'ouverture de vanne a doit être > 0");
    assert!(upstream_head > 0.0, "la charge amont H doit être > 0");
    contraction_coefficient / (1.0 + contraction_coefficient * gate_opening / upstream_head).sqrt()
}

/// Poussée hydrostatique **nette sur la vanne** `F = (1/2)·ρ·g·b·(H₁² − H₂²)` (N).
///
/// Différence des résultantes hydrostatiques des colonnes amont et aval ; nulle
/// lorsque les charges s'équilibrent (`H₁ = H₂`).
///
/// Panique si `fluid_density <= 0`, `gravity <= 0`, `upstream_head < 0`,
/// `downstream_head < 0`, `gate_width <= 0` ou `downstream_head > upstream_head`.
pub fn sluice_force_on_gate(
    fluid_density: f64,
    gravity: f64,
    upstream_head: f64,
    downstream_head: f64,
    gate_width: f64,
) -> f64 {
    assert!(fluid_density > 0.0, "la masse volumique ρ doit être > 0");
    assert!(gravity > 0.0, "la pesanteur g doit être > 0");
    assert!(upstream_head >= 0.0, "la charge amont H₁ doit être ≥ 0");
    assert!(downstream_head >= 0.0, "la charge aval H₂ doit être ≥ 0");
    assert!(gate_width > 0.0, "la largeur de vanne b doit être > 0");
    assert!(
        downstream_head <= upstream_head,
        "la charge aval H₂ ne peut dépasser la charge amont H₁"
    );
    0.5 * fluid_density
        * gravity
        * gate_width
        * (upstream_head * upstream_head - downstream_head * downstream_head)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn contracted_depth_is_linear_in_opening() {
        // yc = Cc·a : doubler l'ouverture double la profondeur contractée.
        let cc = 0.61_f64;
        let a = 0.10_f64;
        let y1 = sluice_contracted_depth(cc, a);
        let y2 = sluice_contracted_depth(cc, 2.0 * a);
        assert_relative_eq!(y2, 2.0 * y1, epsilon = 1e-12);
    }

    #[test]
    fn discharge_is_proportional_to_opening() {
        // Q ∝ a à Cd, b, H, g fixés : tripler l'ouverture triple le débit.
        let (cd, b, a, h, g) = (0.60_f64, 2.0_f64, 0.10_f64, 2.0_f64, 9.81_f64);
        let q1 = sluice_discharge(cd, b, a, h, g);
        let q3 = sluice_discharge(cd, b, 3.0 * a, h, g);
        assert_relative_eq!(q3, 3.0 * q1, epsilon = 1e-12);
    }

    #[test]
    fn discharge_scales_as_sqrt_of_head() {
        // Q ∝ √H : quadrupler la charge double le débit.
        let (cd, b, a, h, g) = (0.60_f64, 1.5_f64, 0.08_f64, 1.0_f64, 9.81_f64);
        let q1 = sluice_discharge(cd, b, a, h, g);
        let q4 = sluice_discharge(cd, b, a, 4.0 * h, g);
        assert_relative_eq!(q4, 2.0 * q1, epsilon = 1e-12);
    }

    #[test]
    fn discharge_coefficient_tends_to_contraction_for_small_opening() {
        // Pour a/H → 0, Cd = Cc/√(1 + Cc·a/H) → Cc.
        let cc = 0.61_f64;
        let cd = sluice_discharge_coefficient(cc, 1.0e-6_f64, 5.0_f64);
        assert_relative_eq!(cd, cc, epsilon = 1e-6);
    }

    #[test]
    fn force_vanishes_when_heads_balance() {
        // F = 0 quand H₁ = H₂ (résultantes hydrostatiques égales).
        let f = sluice_force_on_gate(1000.0_f64, 9.81_f64, 1.7_f64, 1.7_f64, 2.0_f64);
        assert_relative_eq!(f, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn realistic_values_match_hand_calculation() {
        // Cas chiffré : Cd = 0.60, b = 2 m, a = 0.10 m, H = 2 m, g = 9.81 m/s².
        // √(2·9.81·2) = √39.24 = 6.2641839…
        // Q = 0.60·2·0.10·6.2641839 = 0.12·6.2641839 = 0.7517021 m³/s.
        let q = sluice_discharge(0.60_f64, 2.0_f64, 0.10_f64, 2.0_f64, 9.81_f64);
        assert_relative_eq!(q, 0.751_702_06, epsilon = 1e-6);

        // Cd depuis Cc = 0.61 : Cc·a/H = 0.061/2 = 0.0305 ;
        // Cd = 0.61/√1.0305 = 0.61/1.0151354 = 0.6009058.
        let cd = sluice_discharge_coefficient(0.61_f64, 0.10_f64, 2.0_f64);
        assert_relative_eq!(cd, 0.600_905_8, epsilon = 1e-6);

        // Poussée : (1/2)·1000·9.81·2·(2² − 0.5²) = 4905·2·3.75 = 36 787.5 N.
        let f = sluice_force_on_gate(1000.0_f64, 9.81_f64, 2.0_f64, 0.5_f64, 2.0_f64);
        assert_relative_eq!(f, 36_787.5, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "l'ouverture de vanne a doit être > 0")]
    fn discharge_rejects_nonpositive_opening() {
        let _ = sluice_discharge(0.60_f64, 2.0_f64, 0.0_f64, 2.0_f64, 9.81_f64);
    }
}
