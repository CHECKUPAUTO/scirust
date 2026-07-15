//! Diamètre de **foret d'avant-trou de taraudage** (*tap drill size*) pour filet
//! métrique ISO 60° — dimensionne le perçage préalable au taraudage en fonction
//! du **pourcentage d'engagement de filet** visé.
//!
//! Avant de tarauder un trou, on perce un avant-trou dont le diamètre fixe la
//! hauteur de filet réellement formée : plus le foret est petit, plus le filet est
//! « plein » (engagement élevé, filet résistant mais taraud plus sollicité) ; plus
//! il est gros, plus le filet est tronqué (engagement faible, taraudage facile).
//! Pour un profil ISO 60°, la hauteur de filet interne à 100 % vaut `(5√3/8)·p`,
//! d'où le facteur géométrique `1,0825` reliant diamètre et engagement :
//!
//! ```text
//! diamètre de foret     drill = D − (n/100)·k·p
//! pourcentage engagé    n     = (D − drill) / (k·p) · 100     (inverse exacte)
//! facteur de hauteur    k     = 5·√3/8 ≈ 1,0825   (profil ISO 60°)
//! ```
//!
//! Légende (unités SI cohérentes — longueurs toutes dans la même unité, ici m) :
//! - `D` : diamètre nominal (extérieur) du filet (m), `D > 0`.
//! - `p` : pas du filet (m), `p > 0`.
//! - `n` : pourcentage d'engagement de filet (sans dimension, en %) ; usuel 60–75 %.
//! - `drill` : diamètre du foret d'avant-trou (m).
//! - `k` : facteur de hauteur de filet ISO 60°, constante géométrique exacte
//!   `5√3/8`, exposée par [`TAP_DRILL_HEIGHT_FACTOR_ISO60`].
//!
//! **Limite honnête** : ce module suppose un **filet métrique ISO 60° symétrique
//! parfait** ; le facteur `k = 1,0825` découle exactement de cette géométrie et
//! n'est valable que pour elle (Whitworth 55°, UNC/UNF pouces, etc. utilisent une
//! autre convention). Le **pourcentage d'engagement** visé (60–75 % usuel) est un
//! choix de bureau des méthodes **fourni par l'appelant** ; ce module n'impose
//! aucune valeur « par défaut », ni tolérance de foret normalisée, ni correction de
//! sur-perçage lié à l'expansion de matière au taraudage.

/// Facteur de hauteur de filet du profil métrique ISO 60° : `k = 5·√3/8 ≈ 1,0825`
/// (sans dimension). Relie le diamètre du foret d'avant-trou à l'engagement de
/// filet. Valeur exacte issue de la hauteur de filet interne `(5/8)·H` avec
/// `H = (√3/2)·p`.
pub const TAP_DRILL_HEIGHT_FACTOR_ISO60: f64 = 1.0825;

/// Diamètre de foret d'avant-trou `drill = D − (n/100)·k·p` (m) pour un filet
/// métrique ISO 60° de diamètre nominal `nominal_diameter`, de pas `pitch`, visant
/// un pourcentage d'engagement `thread_percentage` (en %).
///
/// À `n = 100 %` on retrouve le diamètre intérieur (minor diameter) théorique
/// `D − k·p` ; à `n = 0 %` le foret vaut `D` (aucun filet).
///
/// Panique si `nominal_diameter <= 0`, `pitch <= 0`, si `thread_percentage` sort de
/// `]0, 100]`, ou si le diamètre de foret obtenu est négatif ou nul (pas trop grand
/// devant le diamètre nominal).
pub fn tap_drill_diameter(nominal_diameter: f64, pitch: f64, thread_percentage: f64) -> f64 {
    assert!(
        nominal_diameter > 0.0,
        "le diamètre nominal D doit être strictement positif"
    );
    assert!(pitch > 0.0, "le pas p doit être strictement positif");
    assert!(
        thread_percentage > 0.0 && thread_percentage <= 100.0,
        "le pourcentage d'engagement n doit être dans ]0, 100] %"
    );
    let drill =
        nominal_diameter - (thread_percentage / 100.0) * TAP_DRILL_HEIGHT_FACTOR_ISO60 * pitch;
    assert!(
        drill > 0.0,
        "diamètre de foret non physique (<= 0) : pas trop grand devant D"
    );
    drill
}

/// Pourcentage d'engagement de filet `n = (D − drill)/(k·p)·100` (en %) réellement
/// obtenu en perçant un avant-trou de diamètre `drill_diameter` avant taraudage
/// d'un filet métrique ISO 60° de diamètre nominal `nominal_diameter` et de pas
/// `pitch` ; inverse exacte de [`tap_drill_diameter`].
///
/// Panique si `nominal_diameter <= 0`, `pitch <= 0`, ou si `drill_diameter` sort de
/// `]0, nominal_diameter[` (un foret `>= D` ne forme aucun filet).
pub fn thread_engagement_percent(nominal_diameter: f64, drill_diameter: f64, pitch: f64) -> f64 {
    assert!(
        nominal_diameter > 0.0,
        "le diamètre nominal D doit être strictement positif"
    );
    assert!(pitch > 0.0, "le pas p doit être strictement positif");
    assert!(
        drill_diameter > 0.0 && drill_diameter < nominal_diameter,
        "le diamètre de foret drill doit être dans ]0, D["
    );
    (nominal_diameter - drill_diameter) / (TAP_DRILL_HEIGHT_FACTOR_ISO60 * pitch) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn drill_matches_definition_for_m6x1() {
        // M6×1,0 visant 75 % : drill = 6 − 0,75·1,0825·1 = 5,188125 mm.
        let (d, p, n) = (6.0e-3_f64, 1.0e-3_f64, 75.0_f64);
        let drill = tap_drill_diameter(d, p, n);
        let expected = d - (n / 100.0) * TAP_DRILL_HEIGHT_FACTOR_ISO60 * p;
        assert_relative_eq!(drill, expected, max_relative = 1e-12);
        assert_relative_eq!(drill, 5.188_125e-3, max_relative = 1e-9);
    }

    #[test]
    fn full_engagement_gives_minor_diameter() {
        // À 100 % d'engagement, le foret vaut le diamètre intérieur D − k·p.
        let (d, p) = (10.0e-3_f64, 1.5e-3_f64);
        let drill = tap_drill_diameter(d, p, 100.0);
        assert_relative_eq!(
            drill,
            d - TAP_DRILL_HEIGHT_FACTOR_ISO60 * p,
            max_relative = 1e-12
        );
    }

    #[test]
    fn diameter_and_engagement_are_reciprocal() {
        // Réciprocité : n(drill(D, p, n)) = n sur plusieurs jeux de paramètres.
        for &(d, p, n) in &[
            (6.0e-3_f64, 1.0e-3_f64, 60.0_f64),
            (10.0e-3, 1.5e-3, 75.0),
            (12.0e-3, 1.75e-3, 68.0),
        ]
        {
            let drill = tap_drill_diameter(d, p, n);
            assert_relative_eq!(
                thread_engagement_percent(d, drill, p),
                n,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn material_removed_is_proportional_to_engagement() {
        // D − drill = (n/100)·k·p est linéaire en n : doubler n double la matière.
        let (d, p) = (8.0e-3_f64, 1.25e-3_f64);
        let gap_37 = d - tap_drill_diameter(d, p, 37.5);
        let gap_75 = d - tap_drill_diameter(d, p, 75.0);
        assert_relative_eq!(gap_75, 2.0 * gap_37, max_relative = 1e-12);
    }

    #[test]
    fn bigger_drill_lowers_engagement() {
        // Monotonie physique : à D, p fixés, ∂n/∂drill < 0.
        let (d, p) = (10.0e-3_f64, 1.5e-3_f64);
        let n_small = thread_engagement_percent(d, 8.5e-3, p);
        let n_big = thread_engagement_percent(d, 8.7e-3, p);
        assert!(n_big < n_small);
        // Pente exacte : Δn/Δdrill = −100/(k·p).
        assert_relative_eq!(
            (n_big - n_small) / 0.2e-3,
            -100.0 / (TAP_DRILL_HEIGHT_FACTOR_ISO60 * p),
            max_relative = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "pourcentage d'engagement n doit être dans ]0, 100]")]
    fn engagement_above_hundred_percent_panics() {
        tap_drill_diameter(6.0e-3, 1.0e-3, 120.0);
    }
}
