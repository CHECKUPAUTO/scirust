//! Roulage de tôle sur cintreuse à trois rouleaux — **rayon cintré** par
//! approximation en arc de cercle, **retour élastique** (ratio de rayons) et
//! **rayon minimal** admissible.
//!
//! ```text
//! rayon cintré (corde)   R = a²/(8·h) + h/2
//! ratio de retour élast. Ks = Ri / Rf
//! rayon minimal          Rmin = t / (2·εmax)
//! ```
//!
//! `a` espacement entre rouleaux inférieurs (m), `h` abaissement (course) du
//! rouleau supérieur (m), `R` rayon de courbure obtenu sous charge (m). La
//! relation corde/flèche est exacte pour un arc de cercle dont la corde vaut `a`
//! et la flèche (sagitta) vaut `h`. `Ri` rayon sous charge (m) et `Rf` rayon
//! final après relâchement (m) donnent le ratio de retour élastique `Ks`
//! (adimensionnel, ∈ ]0, 1] puisque le rayon augmente au déchargement).
//! `t` épaisseur de la tôle (m), `εmax` déformation en fibre extrême maximale
//! admissible (adimensionnelle, ∈ ]0, 1]), `Rmin` rayon de cintrage minimal (m)
//! tiré de la déformation de flexion `ε = t/(2R)`.
//!
//! **Convention** : SI cohérent (m, adimensionnel). **Limite honnête** :
//! approximation géométrique par un **arc de cercle** unique (corde/flèche),
//! valable pour une **tôle mince** (fibre neutre à mi-épaisseur, déformation
//! plane négligée) ; le **retour élastique** est traité séparément via un ratio
//! de rayons **fourni par l'appelant** (aucun modèle module d'Young / limite
//! élastique n'est câblé ici). La déformation maximale admissible `εmax`, les
//! propriétés matériau et les paramètres procédé sont **fournis par
//! l'appelant** — aucune valeur « par défaut » n'est inventée.

/// Rayon cintré `R = a²/(8·h) + h/2` (m) par approximation en arc de cercle.
///
/// Relation exacte corde/flèche : pour un arc dont la corde vaut l'espacement
/// des rouleaux inférieurs `a` et la flèche vaut l'abaissement `h` du rouleau
/// supérieur, le rayon vaut `a²/(8·h) + h/2`.
///
/// Panique si `roller_spacing <= 0` ou `top_roller_offset <= 0`.
pub fn roll_bend_radius_from_geometry(roller_spacing: f64, top_roller_offset: f64) -> f64 {
    assert!(
        roller_spacing > 0.0,
        "l'espacement des rouleaux inférieurs doit être strictement positif"
    );
    assert!(
        top_roller_offset > 0.0,
        "l'abaissement du rouleau supérieur doit être strictement positif"
    );
    roller_spacing.powi(2) / (8.0 * top_roller_offset) + top_roller_offset / 2.0
}

/// Ratio de retour élastique `Ks = Ri / Rf` (adimensionnel).
///
/// Rapport du rayon sous charge `Ri` au rayon final `Rf` après relâchement.
/// Comme le rayon augmente au déchargement (`Rf >= Ri`), on a `Ks ∈ ]0, 1]`.
///
/// Panique si `initial_radius <= 0` ou `final_radius <= 0`.
pub fn roll_bend_springback_ratio(initial_radius: f64, final_radius: f64) -> f64 {
    assert!(
        initial_radius > 0.0,
        "le rayon sous charge doit être strictement positif"
    );
    assert!(
        final_radius > 0.0,
        "le rayon final doit être strictement positif"
    );
    initial_radius / final_radius
}

/// Rayon de cintrage minimal `Rmin = t / (2·εmax)` (m).
///
/// Tiré de la déformation de flexion en fibre extrême `ε = t/(2R)` : le rayon
/// minimal correspond à la déformation maximale admissible `εmax`.
///
/// Panique si `thickness <= 0` ou si `max_strain ∉ ]0, 1]`.
pub fn roll_bend_minimum_radius(thickness: f64, max_strain: f64) -> f64 {
    assert!(
        thickness > 0.0,
        "l'épaisseur doit être strictement positive"
    );
    assert!(
        max_strain > 0.0 && max_strain <= 1.0,
        "la déformation maximale admissible doit être dans ]0, 1]"
    );
    thickness / (2.0 * max_strain)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn geometry_recovers_radius_from_exact_chord_sagitta() {
        // Réciprocité corde/flèche : pour un rayon R et une corde a donnés,
        // la flèche exacte est h = R - sqrt(R² - a²/4) ; la formule doit
        // redonner R.
        let r = 1.0_f64;
        let a = 0.4_f64;
        let h = r - (r.powi(2) - a.powi(2) / 4.0).sqrt();
        let r_rec = roll_bend_radius_from_geometry(a, h);
        assert_relative_eq!(r_rec, r, epsilon = 1e-12);
    }

    #[test]
    fn geometry_realistic_value() {
        // a = 0,5 m, h = 0,05 m → R = a²/(8h) + h/2 = 0,25/0,4 + 0,025 = 0,65 m.
        let r = roll_bend_radius_from_geometry(0.5, 0.05);
        assert_relative_eq!(r, 0.65, epsilon = 1e-12);
    }

    #[test]
    fn geometry_shallow_offset_dominated_by_first_term() {
        // Pour h très petit, R ≈ a²/(8·h) : le terme h/2 devient négligeable.
        let a = 0.6_f64;
        let h = 1e-4_f64;
        let r = roll_bend_radius_from_geometry(a, h);
        assert_relative_eq!(r, a.powi(2) / (8.0 * h), epsilon = 1e-3);
    }

    #[test]
    fn springback_ratio_is_unity_without_relaxation() {
        // Ri = Rf ⇒ Ks = 1 (pas de retour élastique).
        assert_relative_eq!(roll_bend_springback_ratio(0.5, 0.5), 1.0, epsilon = 1e-12);
        // Rf > Ri ⇒ Ks < 1.
        assert!(roll_bend_springback_ratio(0.5, 0.55) < 1.0);
    }

    #[test]
    fn minimum_radius_reciprocal_of_bending_strain() {
        // Rmin = t/(2·εmax) et ε = t/(2·R) sont réciproques : réinjecter Rmin
        // dans la déformation redonne εmax. Cas réaliste t = 2 mm, εmax = 2 %.
        let t = 0.002_f64;
        let eps = 0.02_f64;
        let r_min = roll_bend_minimum_radius(t, eps);
        assert_relative_eq!(r_min, 0.05, epsilon = 1e-12);
        let eps_back = t / (2.0 * r_min);
        assert_relative_eq!(eps_back, eps, epsilon = 1e-12);
    }

    #[test]
    fn minimum_radius_scales_with_thickness() {
        // Rmin ∝ t : doubler l'épaisseur double le rayon minimal.
        let r1 = roll_bend_minimum_radius(0.001, 0.03);
        let r2 = roll_bend_minimum_radius(0.002, 0.03);
        assert_relative_eq!(r2 / r1, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "déformation maximale admissible")]
    fn out_of_range_strain_panics() {
        roll_bend_minimum_radius(0.002, 1.5);
    }
}
