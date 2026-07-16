//! Vanne de régulation — coefficient de débit `Cv`/`Kv` et caractéristiques.
//!
//! Dimensionnement hydraulique d'une vanne de régulation en **écoulement
//! liquide turbulent non cavitant** : coefficient de débit `Cv` (ou `Kv`),
//! débit déduit du `Cv`, conversion impérial↔métrique, **autorité** de la vanne
//! et **caractéristique intrinsèque** égal pourcentage.
//!
//! ```text
//! coefficient de débit (liquide)
//!   Cv = Q · √(G / ΔP)                                    [Q en gpm, ΔP en psi]
//! débit à partir du Cv (relation inverse)
//!   Q  = Cv · √(ΔP / G)                                   [Q en gpm, ΔP en psi]
//! conversion impérial → métrique
//!   Kv = Cv / 1.156                                       [Kv en m³·h⁻¹·bar^-½]
//! autorité de la vanne
//!   N  = ΔP_vanne / ΔP_total                              [–]
//! caractéristique intrinsèque égal pourcentage (fraction du Cv max)
//!   φ(x) = R^(x − 1)                                      [–]
//! ```
//!
//! `Q` débit volumique de liquide, `G` densité relative du liquide (adimensionnelle,
//! `G = ρ/ρ_eau` à conditions de référence), `ΔP` perte de charge aux bornes de la
//! vanne, `Cv` coefficient de débit **impérial** (`Q` en gallons US par minute
//! [gpm], `ΔP` en livres par pouce carré [psi]), `Kv` coefficient de débit
//! **métrique** (`Q` en m³·h⁻¹, `ΔP` en bar). `N` autorité de la vanne (rapport de
//! la perte de charge dans la vanne ouverte à la perte de charge totale du circuit,
//! vanne + reste). `R` rangeabilité (rapport `Cv_max/Cv_min`, typiquement 30 à 50),
//! `x` course fractionnaire de l'obturateur (`0` fermé → `1` pleine ouverture),
//! `φ` fraction du `Cv` maximal restituée.
//!
//! Les formules `Cv`/`Q` sont **cohérentes en unités** : `Cv` en unités impériales
//! avec `Q` [gpm] et `ΔP` [psi] ; pour la forme SI (`Kv`), utiliser `Q` [m³·h⁻¹] et
//! `ΔP` [bar] et interpréter la valeur comme un `Kv`. La densité relative `G` étant
//! adimensionnelle, la même expression vaut dans les deux systèmes.
//!
//! **Limite honnête** : écoulement **liquide monophasique, turbulent et NON
//! cavitant** (ni cavitation ni flashing, régime non étranglé). La **densité
//! relative** `G` et les **pertes de charge** (`ΔP` vanne, `ΔP` total) sont
//! **FOURNIES** par l'appelant — jamais calculées ni inventées ici (aucune
//! corrélation de masse volumique, de pression de vapeur, de coefficient de vanne
//! ni de facteur de récupération `F_L`). L'**autorité** `N` et la **caractéristique
//! intrinsèque** `φ` déterminent la caractéristique **installée** de la boucle,
//! qui n'est pas calculée ici. Ce module **ne traite pas** les gaz/vapeurs
//! (compressibilité), la **cavitation/flashing**, ni le régime étranglé.

/// Coefficient de débit liquide `Cv = Q · √(G / ΔP)` (impérial : `Q` [gpm],
/// `ΔP` [psi] ; ou forme `Kv` SI avec `Q` [m³·h⁻¹], `ΔP` [bar]).
///
/// `flow_rate` (Q) débit volumique de liquide, `specific_gravity` (G) densité
/// relative adimensionnelle, `pressure_drop` (ΔP) perte de charge aux bornes de
/// la vanne. Régime turbulent non cavitant supposé.
///
/// Panique si `Q < 0`, si `G ≤ 0` ou si `ΔP ≤ 0`.
pub fn cv_liquid(flow_rate: f64, specific_gravity: f64, pressure_drop: f64) -> f64 {
    assert!(
        flow_rate >= 0.0,
        "Q ≥ 0 requis (débit volumique de liquide)"
    );
    assert!(
        specific_gravity > 0.0,
        "G > 0 requis (densité relative du liquide)"
    );
    assert!(
        pressure_drop > 0.0,
        "ΔP > 0 requis (perte de charge aux bornes de la vanne)"
    );
    flow_rate * (specific_gravity / pressure_drop).sqrt()
}

/// Débit liquide déduit du coefficient `Q = Cv · √(ΔP / G)` (relation inverse de
/// [`cv_liquid`], mêmes conventions d'unités).
///
/// `valve_cv` (Cv) coefficient de débit de la vanne, `specific_gravity` (G)
/// densité relative adimensionnelle, `pressure_drop` (ΔP) perte de charge aux
/// bornes de la vanne. Régime turbulent non cavitant supposé.
///
/// Panique si `Cv < 0`, si `G ≤ 0` ou si `ΔP < 0`.
pub fn cv_flow_from_cv(valve_cv: f64, specific_gravity: f64, pressure_drop: f64) -> f64 {
    assert!(valve_cv >= 0.0, "Cv ≥ 0 requis (coefficient de débit)");
    assert!(
        specific_gravity > 0.0,
        "G > 0 requis (densité relative du liquide)"
    );
    assert!(
        pressure_drop >= 0.0,
        "ΔP ≥ 0 requis (perte de charge aux bornes de la vanne)"
    );
    valve_cv * (pressure_drop / specific_gravity).sqrt()
}

/// Conversion du coefficient impérial vers le coefficient métrique
/// `Kv = Cv / 1.156` (facteur de conversion normalisé, `Cv` en gpm/psi^½ →
/// `Kv` en m³·h⁻¹·bar^-½).
///
/// `valve_cv` (Cv) coefficient de débit impérial de la vanne.
///
/// Panique si `Cv < 0`.
pub fn cv_kv_from_cv(valve_cv: f64) -> f64 {
    assert!(
        valve_cv >= 0.0,
        "Cv ≥ 0 requis (coefficient de débit impérial)"
    );
    valve_cv / 1.156
}

/// Autorité de la vanne `N = ΔP_vanne / ΔP_total` (fraction, `0 < N ≤ 1`),
/// rapport de la perte de charge dans la vanne ouverte à la perte de charge
/// totale du circuit.
///
/// `valve_pressure_drop` (ΔP_vanne) perte de charge dans la vanne (vanne
/// grande ouverte), `total_pressure_drop` (ΔP_total) perte de charge totale du
/// circuit (vanne + reste), mêmes unités.
///
/// Panique si `ΔP_vanne < 0`, si `ΔP_total ≤ 0` ou si `ΔP_vanne > ΔP_total`.
pub fn cv_authority(valve_pressure_drop: f64, total_pressure_drop: f64) -> f64 {
    assert!(
        valve_pressure_drop >= 0.0,
        "ΔP_vanne ≥ 0 requis (perte de charge dans la vanne)"
    );
    assert!(
        total_pressure_drop > 0.0,
        "ΔP_total > 0 requis (perte de charge totale du circuit)"
    );
    assert!(
        valve_pressure_drop <= total_pressure_drop,
        "ΔP_vanne ≤ ΔP_total requis (la vanne fait partie du circuit)"
    );
    valve_pressure_drop / total_pressure_drop
}

/// Caractéristique intrinsèque égal pourcentage `φ(x) = R^(x − 1)` (fraction du
/// `Cv` maximal restituée à la course `x`).
///
/// `rangeability` (R) rangeabilité `Cv_max/Cv_min` (typiquement 30–50),
/// `fractional_travel` (x) course fractionnaire de l'obturateur (`0` fermé →
/// `1` pleine ouverture). À `x = 1`, `φ = 1` (Cv maximal) ; à `x = 0`,
/// `φ = 1/R` (Cv minimal non nul).
///
/// Panique si `R ≤ 0` ou si `x` hors de `[0, 1]`.
pub fn cv_equal_percentage_opening(rangeability: f64, fractional_travel: f64) -> f64 {
    assert!(
        rangeability > 0.0,
        "R > 0 requis (rangeabilité Cv_max/Cv_min)"
    );
    assert!(
        (0.0..=1.0).contains(&fractional_travel),
        "0 ≤ x ≤ 1 requis (course fractionnaire de l'obturateur)"
    );
    rangeability.powf(fractional_travel - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cv_and_flow_are_reciprocal() {
        // Q → Cv → Q : cv_flow_from_cv est l'inverse exact de cv_liquid pour le
        // même couple (G, ΔP), car √(G/ΔP) · √(ΔP/G) = 1.
        let (q, g, dp) = (120.0_f64, 0.85_f64, 30.0_f64);
        let cv = cv_liquid(q, g, dp);
        let q_back = cv_flow_from_cv(cv, g, dp);
        assert_relative_eq!(q_back, q, max_relative = 1e-12);
    }

    #[test]
    fn cv_liquid_numeric_case() {
        // Q = 100 gpm, G = 0.9, ΔP = 25 psi.
        // Cv = 100 · √(0.9/25) = 100 · √0.036 = 100 · 0.1897366596 = 18.97366596.
        // Recalcul indépendant : 0.9/25 = 0.036 ; √0.036 = 0.18973665961 ;
        // ×100 = 18.973665961.
        assert_relative_eq!(
            cv_liquid(100.0_f64, 0.9_f64, 25.0_f64),
            18.973665961,
            max_relative = 1e-3
        );
    }

    #[test]
    fn cv_liquid_scales_linearly_with_flow() {
        // Cv ∝ Q à (G, ΔP) fixés : doubler le débit double le coefficient requis.
        let base = cv_liquid(50.0_f64, 1.0_f64, 10.0_f64);
        let double = cv_liquid(100.0_f64, 1.0_f64, 10.0_f64);
        assert_relative_eq!(double, 2.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn kv_from_cv_numeric_case() {
        // Cv = 10 ⇒ Kv = 10/1.156 = 8.650519031.
        // Recalcul : 1/1.156 = 0.8650519031 ; ×10 = 8.650519031.
        assert_relative_eq!(cv_kv_from_cv(10.0_f64), 8.650519031, max_relative = 1e-3);
    }

    #[test]
    fn authority_is_a_bounded_fraction() {
        // ΔP_vanne = 15, ΔP_total = 30 ⇒ N = 15/30 = 0.5 (autorité correcte).
        // À ΔP_vanne = ΔP_total, N = 1 (toute la perte est dans la vanne).
        assert_relative_eq!(cv_authority(15.0_f64, 30.0_f64), 0.5, max_relative = 1e-12);
        assert_relative_eq!(cv_authority(30.0_f64, 30.0_f64), 1.0, max_relative = 1e-12);
    }

    #[test]
    fn equal_percentage_limits_and_numeric() {
        // Limites : x = 1 ⇒ R^0 = 1 (Cv max) ; x = 0 ⇒ R^(−1) = 1/R (Cv min).
        assert_relative_eq!(
            cv_equal_percentage_opening(50.0_f64, 1.0_f64),
            1.0,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            cv_equal_percentage_opening(50.0_f64, 0.0_f64),
            1.0 / 50.0,
            max_relative = 1e-12
        );
        // Cas chiffré : R = 50, x = 0.5 ⇒ 50^(−0.5) = 1/√50 = 0.1414213562.
        // Recalcul : √50 = 7.0710678119 ; 1/7.0710678119 = 0.14142135624.
        assert_relative_eq!(
            cv_equal_percentage_opening(50.0_f64, 0.5_f64),
            0.14142135624,
            max_relative = 1e-3
        );
    }

    #[test]
    #[should_panic(expected = "ΔP > 0 requis")]
    fn cv_liquid_panics_on_zero_pressure_drop() {
        // ΔP = 0 ⇒ √(G/0) indéfini (division par zéro) ⇒ entrée rejetée.
        let _ = cv_liquid(100.0_f64, 0.9_f64, 0.0_f64);
    }
}
