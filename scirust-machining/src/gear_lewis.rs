//! Résistance à la flexion de denture — **équation de Lewis** : la dent est
//! traitée comme une poutre en porte-à-faux d'égale résistance et l'on en déduit
//! la contrainte à la base de la dent, l'effort tangentiel admissible et la
//! correction dynamique de Barth.
//!
//! ```text
//! contrainte de flexion     σ = Ft / (b·m·Y)
//! effort admissible         Ft_adm = σ_adm·b·m·Y
//! correction dynamique      σ_dyn = σ / Kv
//! ```
//!
//! `Ft` effort tangentiel transmis au diamètre primitif (N), `b` largeur de
//! denture (m), `m` module métrique **exprimé en mètres** (m), `Y` facteur de
//! forme de Lewis (adimensionnel, dépend du nombre de dents et du profil),
//! `σ`/`σ_adm` contrainte de flexion à la base de la dent / contrainte admissible
//! du matériau (Pa), `Kv` facteur de vitesse de Barth (adimensionnel, ∈ ]0, 1]).
//! La contrainte est l'effort tangentiel rapporté à l'aire modulée `b·m·Y` ;
//! l'effort admissible en est l'inverse à contrainte imposée ; la correction
//! divise par `Kv` pour majorer la contrainte due aux effets dynamiques.
//!
//! **Convention** : SI cohérent (N, m, Pa) ; le module `m` est donné en mètres
//! (un module de 5 mm vaut `0,005`). **Limite honnête** : équation de Lewis pure
//! (dent = poutre en porte-à-faux, charge appliquée en pointe, sans concentration
//! de contrainte) ; le facteur de forme `Y`, la contrainte admissible `σ_adm`
//! (nombre de dents, matériau, traitement) et le facteur de vitesse `Kv` (procédé
//! de taillage, vitesse périphérique) sont **fournis par l'appelant** — aucune
//! valeur n'est inventée ici. Ne couvre **pas** la pression de contact
//! (Hertz / grippage-piqûre AGMA) ni les facteurs correctifs AGMA
//! (surcharge, distribution de charge, etc.).

/// Contrainte de flexion à la base de la dent `σ = Ft / (b·m·Y)` (Pa).
///
/// `b·m·Y` est l'aire de flexion modulée par le facteur de forme de Lewis.
///
/// Panique si `tangential_force < 0`, `face_width <= 0`, `module_metric <= 0`
/// ou `lewis_form_factor <= 0`.
pub fn gear_lewis_bending_stress(
    tangential_force: f64,
    face_width: f64,
    module_metric: f64,
    lewis_form_factor: f64,
) -> f64 {
    assert!(
        tangential_force >= 0.0,
        "l'effort tangentiel doit être positif"
    );
    assert!(face_width > 0.0, "la largeur de denture doit être positive");
    assert!(module_metric > 0.0, "le module doit être positif");
    assert!(
        lewis_form_factor > 0.0,
        "le facteur de forme de Lewis doit être positif"
    );
    tangential_force / (face_width * module_metric * lewis_form_factor)
}

/// Effort tangentiel admissible `Ft_adm = σ_adm·b·m·Y` (N).
///
/// Inverse de l'équation de Lewis à contrainte admissible imposée : effort
/// tangentiel maximal supportable par la denture en flexion.
///
/// Panique si `allowable_stress < 0`, `face_width <= 0`, `module_metric <= 0`
/// ou `lewis_form_factor <= 0`.
pub fn gear_lewis_allowable_force(
    allowable_stress: f64,
    face_width: f64,
    module_metric: f64,
    lewis_form_factor: f64,
) -> f64 {
    assert!(
        allowable_stress >= 0.0,
        "la contrainte admissible doit être positive"
    );
    assert!(face_width > 0.0, "la largeur de denture doit être positive");
    assert!(module_metric > 0.0, "le module doit être positif");
    assert!(
        lewis_form_factor > 0.0,
        "le facteur de forme de Lewis doit être positif"
    );
    allowable_stress * face_width * module_metric * lewis_form_factor
}

/// Contrainte corrigée par le facteur de vitesse de Barth `σ_dyn = σ / Kv` (Pa).
///
/// Majore la contrainte de flexion pour tenir compte des effets dynamiques ; le
/// facteur de vitesse `Kv ∈ ]0, 1]` est **fourni par l'appelant** (loi de Barth
/// selon le procédé de taillage et la vitesse périphérique).
///
/// Panique si `bending_stress < 0` ou si `velocity_factor ∉ ]0, 1]`.
pub fn gear_lewis_with_velocity_factor(bending_stress: f64, velocity_factor: f64) -> f64 {
    assert!(
        bending_stress >= 0.0,
        "la contrainte de flexion doit être positive"
    );
    assert!(
        velocity_factor > 0.0 && velocity_factor <= 1.0,
        "le facteur de vitesse doit être dans ]0, 1]"
    );
    bending_stress / velocity_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bending_stress_realistic_value() {
        // Ft = 3000 N, b = 40 mm, m = 5 mm, Y = 0,35.
        // σ = 3000 / (0,040·0,005·0,35) = 3000 / 7e-5 ≈ 42,857 MPa.
        let sigma = gear_lewis_bending_stress(3000.0, 0.040, 0.005, 0.35);
        assert_relative_eq!(sigma, 3000.0 / (0.040 * 0.005 * 0.35), epsilon = 1e-3);
        assert_relative_eq!(sigma, 42_857_142.857_142_86, epsilon = 1.0);
    }

    #[test]
    fn allowable_force_is_inverse_of_bending_stress() {
        // Réciprocité : Ft_adm(σ(Ft)) = Ft à géométrie identique.
        let (b, m, y) = (0.050, 0.006, 0.40);
        let ft = 4200.0;
        let sigma = gear_lewis_bending_stress(ft, b, m, y);
        let ft_back = gear_lewis_allowable_force(sigma, b, m, y);
        assert_relative_eq!(ft_back, ft, epsilon = 1e-9);
    }

    #[test]
    fn allowable_force_direct_value() {
        // σ_adm = 100 MPa, b = 40 mm, m = 5 mm, Y = 0,35.
        // Ft_adm = 100e6·0,040·0,005·0,35 = 7000 N.
        let ft = gear_lewis_allowable_force(100e6, 0.040, 0.005, 0.35);
        assert_relative_eq!(ft, 7000.0, epsilon = 1e-6);
    }

    #[test]
    fn bending_stress_scales_inversely_with_face_width() {
        // σ ∝ 1/b : doubler la largeur de denture divise la contrainte par deux.
        let s1 = gear_lewis_bending_stress(2500.0, 0.030, 0.004, 0.32);
        let s2 = gear_lewis_bending_stress(2500.0, 0.060, 0.004, 0.32);
        assert_relative_eq!(s1 / s2, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn velocity_factor_amplifies_stress() {
        // σ_dyn = σ / Kv : avec Kv < 1 la contrainte corrigée dépasse la statique.
        let sigma = gear_lewis_bending_stress(3000.0, 0.040, 0.005, 0.35);
        let dyn_stress = gear_lewis_with_velocity_factor(sigma, 0.6);
        assert_relative_eq!(dyn_stress, sigma / 0.6, epsilon = 1e-3);
        assert!(dyn_stress > sigma);
    }

    #[test]
    fn velocity_factor_unity_leaves_stress_unchanged() {
        // Cas limite Kv = 1 : aucune majoration dynamique.
        let sigma = gear_lewis_bending_stress(1800.0, 0.025, 0.003, 0.30);
        let dyn_stress = gear_lewis_with_velocity_factor(sigma, 1.0);
        assert_relative_eq!(dyn_stress, sigma, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "facteur de vitesse")]
    fn out_of_range_velocity_factor_panics() {
        gear_lewis_with_velocity_factor(50e6, 1.5);
    }
}
