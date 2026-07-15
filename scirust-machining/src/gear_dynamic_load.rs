//! Charge dynamique de denture d'engrenage — **formule de Buckingham** : à
//! l'effort tangentiel transmis s'ajoute un incrément dynamique dû à l'erreur de
//! denture et à la vitesse au primitif ; on fournit aussi le facteur de vitesse
//! de Barth et la majoration par facteur de service.
//!
//! ```text
//! charge dynamique            F_d = Ft + F_i
//! incrément de Buckingham     F_i = 21·v·(C·b + Ft) / (21·v + sqrt(C·b + Ft))
//! facteur de vitesse (Barth)  Kv  = 3 / (3 + v)
//! charge majorée de service   F_s = W·Ks
//! ```
//!
//! `Ft` effort tangentiel transmis à la denture (N), `F_i` incrément dynamique
//! (N), `F_d` charge dynamique totale (N), `v` vitesse linéaire au cercle primitif
//! (m/s), `C` facteur de déformation de Buckingham (N/m) traduisant l'erreur de
//! denture selon la classe de précision et le module, `b` largeur de denture (m),
//! de sorte que `C·b` est un effort (N), `Kv` facteur de vitesse de Barth
//! (adimensionnel), `W` charge transmise (N), `Ks` facteur de service
//! (adimensionnel), `F_s` charge majorée (N).
//!
//! **Convention** : SI cohérent (N, m, m/s). Le facteur numérique `21` de la
//! formule de Buckingham suppose `v` en **m/s** ; il vaut `6` pour `v` en pi/min
//! dans la forme impériale d'origine — cette variante n'est pas traitée ici.
//! **Limite honnête** : la formule dynamique de Buckingham est une **estimation
//! empirique** de la surcharge dynamique. Le facteur de déformation `C` (erreur de
//! denture selon la classe de précision et le module), l'effort tangentiel `Ft`,
//! la vitesse au primitif `v` et le facteur de service `Ks` sont des données de
//! **procédé/matériau fournies par l'appelant** — aucune valeur n'est inventée
//! ici. Le dimensionnement (comparaison à une charge admissible d'usure ou de
//! flexion) relève de l'appelant.

/// Charge dynamique totale `F_d = Ft + F_i` (N).
///
/// Somme de l'effort tangentiel transmis et de l'incrément dynamique de
/// Buckingham `dynamic_factor_term` (typiquement issu de
/// [`geardyn_buckingham_increment`]).
///
/// Panique si `tangential_load <= 0` ou `dynamic_factor_term < 0`.
pub fn geardyn_dynamic_load(tangential_load: f64, dynamic_factor_term: f64) -> f64 {
    assert!(
        tangential_load > 0.0,
        "l'effort tangentiel doit être positif"
    );
    assert!(
        dynamic_factor_term >= 0.0,
        "l'incrément dynamique doit être positif ou nul"
    );
    tangential_load + dynamic_factor_term
}

/// Incrément dynamique de Buckingham
/// `F_i = 21·v·(C·b + Ft) / (21·v + sqrt(C·b + Ft))` (N).
///
/// `v` est la vitesse au primitif (m/s) et `C` le facteur de déformation (N/m),
/// de sorte que `C·b` (N) est l'effort d'erreur de denture. À vitesse nulle
/// l'incrément est nul ; il croît avec `v` vers l'asymptote `C·b + Ft`.
///
/// Panique si `pitch_line_velocity < 0`, `face_width <= 0`,
/// `tangential_load <= 0` ou `deformation_factor < 0`.
pub fn geardyn_buckingham_increment(
    pitch_line_velocity: f64,
    face_width: f64,
    tangential_load: f64,
    deformation_factor: f64,
) -> f64 {
    assert!(
        pitch_line_velocity >= 0.0,
        "la vitesse au primitif doit être positive ou nulle"
    );
    assert!(face_width > 0.0, "la largeur de denture doit être positive");
    assert!(
        tangential_load > 0.0,
        "l'effort tangentiel doit être positif"
    );
    assert!(
        deformation_factor >= 0.0,
        "le facteur de déformation doit être positif ou nul"
    );
    let error_load = deformation_factor * face_width + tangential_load;
    21.0 * pitch_line_velocity * error_load / (21.0 * pitch_line_velocity + error_load.sqrt())
}

/// Facteur de vitesse de Barth `Kv = 3 / (3 + v)` (adimensionnel).
///
/// Décroît de 1 (vitesse nulle) vers 0 quand la vitesse au primitif `v` (m/s)
/// augmente ; utilisé comme facteur de service dynamique dans l'équation de Lewis.
///
/// Panique si `pitch_line_velocity < 0`.
pub fn geardyn_velocity_factor_barth(pitch_line_velocity: f64) -> f64 {
    assert!(
        pitch_line_velocity >= 0.0,
        "la vitesse au primitif doit être positive ou nulle"
    );
    3.0 / (3.0 + pitch_line_velocity)
}

/// Charge majorée par facteur de service `F_s = W·Ks` (N).
///
/// Applique le facteur de service `Ks` (surcharge de fonctionnement, chocs) à la
/// charge transmise `W`.
///
/// Panique si `transmitted_load <= 0` ou `service_factor < 1`.
pub fn geardyn_service_factor_load(transmitted_load: f64, service_factor: f64) -> f64 {
    assert!(
        transmitted_load > 0.0,
        "la charge transmise doit être positive"
    );
    assert!(
        service_factor >= 1.0,
        "le facteur de service doit être supérieur ou égal à 1"
    );
    transmitted_load * service_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dynamic_load_is_sum_of_static_and_increment() {
        // Identité F_d = Ft + F_i : la charge dynamique est la somme exacte.
        let ft = 3_000.0_f64;
        let fi = 4_320.0_f64;
        let fd = geardyn_dynamic_load(ft, fi);
        assert_relative_eq!(fd, ft + fi, epsilon = 1e-12);
    }

    #[test]
    fn increment_vanishes_at_zero_velocity() {
        // Cas limite v = 0 : numérateur nul → incrément dynamique nul.
        let fi = geardyn_buckingham_increment(0.0, 0.05, 3_000.0, 1.0e5);
        assert_relative_eq!(fi, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn increment_tends_to_error_load_at_high_velocity() {
        // v ≫ 1 : 21·v domine sqrt(C·b + Ft), donc F_i → C·b + Ft.
        let (b, ft, c) = (0.05_f64, 3_000.0_f64, 1.0e5_f64);
        let error_load = c * b + ft; // 5000 + 3000 = 8000 N
        let fi = geardyn_buckingham_increment(1.0e9, b, ft, c);
        assert!(fi < error_load);
        assert_relative_eq!(fi, error_load, epsilon = 1e-3);
    }

    #[test]
    fn increment_realistic_value() {
        // v = 5 m/s, b = 50 mm, Ft = 3000 N, C = 1e5 N/m.
        // C·b + Ft = 5000 + 3000 = 8000 N ; sqrt(8000) = 89,442719...
        // F_i = 21·5·8000 / (21·5 + 89,442719) = 840000 / 194,442719 ≈ 4320,0383 N.
        let fi = geardyn_buckingham_increment(5.0, 0.05, 3_000.0, 1.0e5);
        assert_relative_eq!(fi, 4_320.038_334, epsilon = 1e-3);
    }

    #[test]
    fn barth_factor_limits() {
        // Kv = 1 à v = 0 ; Kv = 0,5 à v = 3 m/s (v = 3 → 3/(3+3)).
        assert_relative_eq!(geardyn_velocity_factor_barth(0.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(geardyn_velocity_factor_barth(3.0), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn service_factor_scales_load_linearly() {
        // F_s = W·Ks : Ks = 1 laisse la charge inchangée, Ks = 1,5 la majore de 50 %.
        let w = 4_000.0_f64;
        assert_relative_eq!(geardyn_service_factor_load(w, 1.0), w, epsilon = 1e-12);
        assert_relative_eq!(
            geardyn_service_factor_load(w, 1.5),
            6_000.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "largeur de denture")]
    fn zero_face_width_panics() {
        geardyn_buckingham_increment(5.0, 0.0, 3_000.0, 1.0e5);
    }
}
