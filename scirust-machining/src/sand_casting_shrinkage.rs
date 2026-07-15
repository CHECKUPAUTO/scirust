//! Fonderie sable — **surdimensionnement du modèle** (surcotes de retrait,
//! d'usinage et de dépouille).
//!
//! ```text
//! cote modèle (retrait)   Lp = L·(1 + s)
//! surcote d'usinage       Lm = L + a
//! surépaisseur dépouille  Δ  = h·tan(θ)
//! cote modèle complète    Lp = L·(1 + s) + a
//! ```
//!
//! `L` cote nominale de la pièce brute (m), `s` retrait linéaire de
//! solidification (sans dimension, ex. fonte grise ≈ 0,01), `a` surcote
//! d'usinage laissée sur la face (m), `h` hauteur de la paroi dépouillée (m),
//! `θ` angle de dépouille (rad), `Δ` surépaisseur ajoutée en pied de paroi (m),
//! `Lp` cote du modèle (m), `Lm` cote après ajout d'usinage (m).
//!
//! **Convention** : SI cohérent (mètres, radians), toutes cotes en f64.
//! **Limite honnête** : modèle de **retrait linéaire uniforme** et isotrope ;
//! le coefficient de retrait `s`, les surcotes d'usinage `a` et les angles de
//! dépouille `θ` dépendent du matériau, de la géométrie et du procédé et sont
//! **fournis par l'appelant** — aucune valeur « par défaut » n'est inventée
//! ici. Ne traite ni le retrait volumique (retassure), ni les retraits
//! anisotropes ou entravés.

/// Cote du modèle intégrant le retrait de solidification `Lp = L·(1 + s)` (m).
///
/// Panique si `casting_dimension < 0` ou si `shrinkage_allowance <= -1`
/// (retrait rendant la cote nulle ou négative).
pub fn pattern_dimension(casting_dimension: f64, shrinkage_allowance: f64) -> f64 {
    assert!(
        casting_dimension >= 0.0,
        "la cote de la pièce doit être positive"
    );
    assert!(
        shrinkage_allowance > -1.0,
        "le retrait doit être supérieur à -1 (cote résultante strictement positive)"
    );
    casting_dimension * (1.0 + shrinkage_allowance)
}

/// Retrait linéaire implicite entre pièce et modèle `s = Lp/L − 1` (—).
///
/// Réciproque de [`pattern_dimension`].
///
/// Panique si `casting_dimension <= 0`.
pub fn casting_shrinkage_ratio(pattern_dimension: f64, casting_dimension: f64) -> f64 {
    assert!(
        casting_dimension > 0.0,
        "la cote de la pièce doit être strictement positive"
    );
    pattern_dimension / casting_dimension - 1.0
}

/// Cote intégrant la surcote d'usinage `Lm = L + a` (m).
///
/// Panique si `dimension < 0` ou `allowance < 0`.
pub fn pattern_machining_allowance_added(dimension: f64, allowance: f64) -> f64 {
    assert!(dimension >= 0.0, "la cote doit être positive");
    assert!(allowance >= 0.0, "la surcote d'usinage doit être positive");
    dimension + allowance
}

/// Surépaisseur ajoutée en pied de paroi par la dépouille `Δ = h·tan(θ)` (m).
///
/// Panique si `height < 0` ou si `draft_angle_rad` n'est pas dans `[0, π/2)`.
pub fn pattern_draft_added_dimension(height: f64, draft_angle_rad: f64) -> f64 {
    use core::f64::consts::FRAC_PI_2;
    assert!(height >= 0.0, "la hauteur doit être positive");
    assert!(
        (0.0..FRAC_PI_2).contains(&draft_angle_rad),
        "l'angle de dépouille doit être dans [0, π/2)"
    );
    height * draft_angle_rad.tan()
}

/// Cote complète du modèle : retrait puis surcote d'usinage
/// `Lp = L·(1 + s) + a` (m).
///
/// Panique via [`pattern_dimension`] et [`pattern_machining_allowance_added`].
pub fn pattern_full_dimension(
    casting_dimension: f64,
    shrinkage_allowance: f64,
    machining_allowance: f64,
) -> f64 {
    pattern_machining_allowance_added(
        pattern_dimension(casting_dimension, shrinkage_allowance),
        machining_allowance,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::FRAC_PI_4;

    #[test]
    fn shrinkage_ratio_is_inverse_of_pattern_dimension() {
        // Réciprocité : retrouver s à partir de la cote modèle.
        let l = 0.250_f64;
        let s = 0.01_f64;
        let lp = pattern_dimension(l, s);
        assert_relative_eq!(casting_shrinkage_ratio(lp, l), s, epsilon = 1e-12);
    }

    #[test]
    fn one_percent_shrinkage_realistic_case() {
        // Fonte grise ~1 % : une cote pièce de 200 mm impose 202 mm au modèle.
        let lp = pattern_dimension(0.200, 0.01);
        assert_relative_eq!(lp, 0.202, epsilon = 1e-12);
    }

    #[test]
    fn machining_allowance_is_additive() {
        // Surcote purement additive et cumulative.
        let base = pattern_machining_allowance_added(0.100, 0.003);
        assert_relative_eq!(base, 0.103, epsilon = 1e-12);
        // Ajouter deux fois a/2 équivaut à ajouter a.
        let split = pattern_machining_allowance_added(
            pattern_machining_allowance_added(0.100, 0.0015),
            0.0015,
        );
        assert_relative_eq!(split, base, epsilon = 1e-12);
    }

    #[test]
    fn draft_at_45_degrees_equals_height() {
        // tan(π/4) = 1 → la surépaisseur vaut exactement la hauteur.
        let h = 0.050_f64;
        assert_relative_eq!(
            pattern_draft_added_dimension(h, FRAC_PI_4),
            h,
            epsilon = 1e-12
        );
    }

    #[test]
    fn draft_is_proportional_to_height() {
        // À angle fixé, Δ est linéaire en h.
        let a = pattern_draft_added_dimension(0.02, 0.05);
        let b = pattern_draft_added_dimension(0.06, 0.05);
        assert_relative_eq!(b / a, 3.0, epsilon = 1e-9);
        // Angle nul → aucune surépaisseur.
        assert_relative_eq!(
            pattern_draft_added_dimension(0.10, 0.0),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn full_dimension_combines_shrinkage_then_machining() {
        // Lp = L·(1+s) + a, ici 0,200·1,01 + 0,003 = 0,205.
        let lp = pattern_full_dimension(0.200, 0.01, 0.003);
        assert_relative_eq!(lp, 0.202 + 0.003, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "dépouille")]
    fn draft_beyond_right_angle_panics() {
        use core::f64::consts::FRAC_PI_2;
        pattern_draft_added_dimension(0.05, FRAC_PI_2);
    }
}
