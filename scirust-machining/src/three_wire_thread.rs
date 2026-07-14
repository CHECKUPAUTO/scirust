//! Mesure de filetage aux **trois piges** (*three-wire method*) — détermination
//! du diamètre sur flancs d'un filet symétrique à partir de la cote mesurée au
//! micromètre sur trois fils calibrés logés dans les creux du filet.
//!
//! Trois piges identiques (deux d'un côté, une de l'autre) sont posées dans les
//! sillons ; le micromètre lit la cote `M` par-dessus. Le **diamètre de pige
//! optimal** `Dw` tangente le flanc à mi-hauteur du filet, ce qui rend `M`
//! insensible aux petits défauts de forme du fond. La cote sur piges relie `M`
//! au **diamètre sur flancs** `E` (pitch diameter) :
//!
//! ```text
//! diamètre de pige optimal   Dw = p / (2·cos α)
//! cote sur piges             M  = E − (p/2)/tan α + Dw·(1 + 1/sin α)
//! diamètre sur flancs        E  = M + (p/2)/tan α − Dw·(1 + 1/sin α)   (inverse)
//! ```
//!
//! Légende (unités SI cohérentes — longueurs toutes dans la même unité, ici m) :
//! - `p` : pas du filet (m), `p > 0`.
//! - `α` : **demi-angle** du filet, entre un flanc et la normale à l'axe (rad).
//!   Pour un filet métrique ISO 60°, `α = 30° = π/6`.
//! - `Dw` : diamètre d'une pige (m), `Dw > 0`.
//! - `E` : diamètre sur flancs (pitch diameter) recherché (m).
//! - `M` : cote lue au micromètre sur les trois piges (m).
//!
//! **Limite honnête** : ce module suppose un **filet symétrique parfait**
//! (flancs droits, deux flancs au même angle `α`), un contact ponctuel pige/flanc
//! et **néglige l'angle d'hélice** (correction d'hélice non appliquée : valable
//! pour les pas faibles / filets à un seul début, sinon `M` est légèrement
//! surestimée). Les piges sont supposées rigides et exactement calibrées. Aucune
//! valeur normalisée de pas, d'angle ou de diamètre de pige n'est imposée :
//! l'angle de flanc `α`, le pas et les diamètres sont **fournis par l'appelant**,
//! ce module n'invente aucune constante « par défaut ».

use core::f64::consts::PI;

/// Diamètre de pige optimal `Dw = p / (2·cos α)` (m) — pige tangentant le flanc
/// à mi-hauteur du filet de pas `pitch` et de demi-angle de flanc
/// `thread_half_angle_rad`.
///
/// Panique si `pitch <= 0` ou si `thread_half_angle_rad` sort de `]0, π/2[`
/// (`cos α = 0` interdit).
pub fn best_wire_diameter(pitch: f64, thread_half_angle_rad: f64) -> f64 {
    assert!(pitch > 0.0, "le pas p doit être strictement positif");
    assert!(
        thread_half_angle_rad > 0.0 && thread_half_angle_rad < PI / 2.0,
        "le demi-angle de flanc α doit être dans ]0, π/2[ rad"
    );
    pitch / (2.0 * thread_half_angle_rad.cos())
}

/// Cote sur piges `M = E − (p/2)/tan α + Dw·(1 + 1/sin α)` (m) lue au micromètre
/// sur trois piges de diamètre `wire_diameter` posées dans un filet de diamètre
/// sur flancs `pitch_diameter`, de pas `pitch` et de demi-angle de flanc
/// `thread_half_angle_rad`.
///
/// Panique si `pitch_diameter <= 0`, `wire_diameter <= 0`, `pitch <= 0`, ou si
/// `thread_half_angle_rad` sort de `]0, π/2[` (`tan α` ou `sin α` nul interdit).
pub fn measurement_over_wires(
    pitch_diameter: f64,
    wire_diameter: f64,
    pitch: f64,
    thread_half_angle_rad: f64,
) -> f64 {
    assert!(
        pitch_diameter > 0.0,
        "le diamètre sur flancs E doit être strictement positif"
    );
    assert!(
        wire_diameter > 0.0,
        "le diamètre de pige Dw doit être strictement positif"
    );
    assert!(pitch > 0.0, "le pas p doit être strictement positif");
    assert!(
        thread_half_angle_rad > 0.0 && thread_half_angle_rad < PI / 2.0,
        "le demi-angle de flanc α doit être dans ]0, π/2[ rad"
    );
    pitch_diameter - (pitch / 2.0) / thread_half_angle_rad.tan()
        + wire_diameter * (1.0 + 1.0 / thread_half_angle_rad.sin())
}

/// Diamètre sur flancs `E = M + (p/2)/tan α − Dw·(1 + 1/sin α)` (m) reconstruit
/// depuis la cote mesurée `measurement`, les piges `wire_diameter`, le pas
/// `pitch` et le demi-angle de flanc `thread_half_angle_rad` ; inverse exact de
/// [`measurement_over_wires`].
///
/// Panique si `measurement <= 0`, `wire_diameter <= 0`, `pitch <= 0`, ou si
/// `thread_half_angle_rad` sort de `]0, π/2[`.
pub fn pitch_diameter_from_measurement(
    measurement: f64,
    wire_diameter: f64,
    pitch: f64,
    thread_half_angle_rad: f64,
) -> f64 {
    assert!(
        measurement > 0.0,
        "la cote mesurée M doit être strictement positive"
    );
    assert!(
        wire_diameter > 0.0,
        "le diamètre de pige Dw doit être strictement positif"
    );
    assert!(pitch > 0.0, "le pas p doit être strictement positif");
    assert!(
        thread_half_angle_rad > 0.0 && thread_half_angle_rad < PI / 2.0,
        "le demi-angle de flanc α doit être dans ]0, π/2[ rad"
    );
    measurement + (pitch / 2.0) / thread_half_angle_rad.tan()
        - wire_diameter * (1.0 + 1.0 / thread_half_angle_rad.sin())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn best_wire_matches_definition_for_metric_thread() {
        // Filet métrique ISO 60° (α = 30°), pas 1 mm : Dw = p/(2·cos30°).
        let p = 1.0e-3_f64;
        let alpha = PI / 6.0;
        let dw = best_wire_diameter(p, alpha);
        assert_relative_eq!(dw, p / (2.0 * alpha.cos()), max_relative = 1e-12);
        // Contrôle chiffré indépendant : 1/(2·0,866025…) ≈ 0,577350 mm.
        assert_relative_eq!(dw, 0.577_350_3e-3, max_relative = 1e-6);
    }

    #[test]
    fn measurement_and_pitch_diameter_are_reciprocal() {
        // Réciprocité : E(M(E)) = E pour plusieurs jeux de paramètres.
        let alpha = PI / 6.0;
        for &(e, p) in &[
            (9.026e-3_f64, 1.5e-3_f64),
            (5.35e-3, 0.8e-3),
            (18.376e-3, 2.5e-3),
        ]
        {
            let dw = best_wire_diameter(p, alpha);
            let m = measurement_over_wires(e, dw, p, alpha);
            assert_relative_eq!(
                pitch_diameter_from_measurement(m, dw, p, alpha),
                e,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn measurement_grows_linearly_with_pitch_diameter() {
        // M − E est constant à pas, pige et angle fixés (terme géométrique pur) :
        // deux diamètres sur flancs décalés de Δ produisent des cotes décalées de Δ.
        let (p, alpha) = (1.5e-3_f64, PI / 6.0);
        let dw = best_wire_diameter(p, alpha);
        let m1 = measurement_over_wires(9.0e-3, dw, p, alpha);
        let m2 = measurement_over_wires(9.0e-3 + 0.2e-3, dw, p, alpha);
        assert_relative_eq!(m2 - m1, 0.2e-3, max_relative = 1e-12);
    }

    #[test]
    fn measurement_increases_with_wire_diameter() {
        // Monotonie physique : à E, p, α fixés, ∂M/∂Dw = 1 + 1/sin α > 0,
        // donc une pige plus grosse relève la cote.
        let (e, p, alpha) = (9.026e-3_f64, 1.5e-3_f64, PI / 6.0);
        let dw = best_wire_diameter(p, alpha);
        let m_ref = measurement_over_wires(e, dw, p, alpha);
        let m_big = measurement_over_wires(e, dw * 1.01, p, alpha);
        assert!(m_big > m_ref);
        // Pente exacte : ΔM / ΔDw = 1 + 1/sin α (ici 1 + 1/sin30° = 3).
        assert_relative_eq!(
            (m_big - m_ref) / (dw * 0.01),
            1.0 + 1.0 / alpha.sin(),
            max_relative = 1e-9
        );
    }

    #[test]
    fn realistic_m10x1_5_case() {
        // M10×1,5 : diamètre sur flancs nominal E = 9,026 mm, α = 30°.
        let (e, p, alpha) = (9.026e-3_f64, 1.5e-3_f64, PI / 6.0);
        let dw = best_wire_diameter(p, alpha);
        let m = measurement_over_wires(e, dw, p, alpha);
        // Reconstruction manuelle du même résultat pour verrouiller la formule.
        let expected = e - (p / 2.0) / alpha.tan() + dw * (1.0 + 1.0 / alpha.sin());
        assert_relative_eq!(m, expected, max_relative = 1e-12);
        // La cote sur piges dépasse le diamètre sur flancs (piges saillantes).
        assert!(m > e);
    }

    #[test]
    #[should_panic(expected = "demi-angle de flanc α doit être dans ]0, π/2[")]
    fn right_angle_flank_panics() {
        best_wire_diameter(1.0e-3, PI / 2.0);
    }
}
