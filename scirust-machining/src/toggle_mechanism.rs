//! Genouillère (toggle) — amplification d'effort d'un mécanisme à biellettes
//! symétriques, quand celles-ci approchent l'alignement (angle `θ` → 0).
//!
//! ```text
//! rapport d'effort (idéal)   R = F_out / F_in = 1 / (2·tan θ)
//! effort de sortie           F_out = F_in / (2·tan θ)
//! ```
//!
//! `θ` demi-angle que chaque biellette fait avec l'horizontale d'alignement
//! (rad), `F_in` effort moteur appliqué au genou (N), `F_out` effort de
//! serrage/poussée obtenu (N). Le facteur `2` traduit les **deux** biellettes
//! symétriques qui reprennent l'effort d'entrée. Quand `θ` diminue, le levier
//! s'amplifie : `R → ∞` lorsque `θ → 0` (biellettes alignées, point mort).
//!
//! **Convention** : SI cohérent, angle en rad, efforts en N. **Limite honnête** :
//! mécanisme **idéal sans frottement** (articulations parfaites, biellettes
//! rigides et de masse négligeable), efforts d'entrée et de sortie
//! perpendiculaires au sens usuel de la genouillère. L'amplification tend vers
//! l'infini au voisinage du point mort : c'est une idéalisation, la valeur réelle
//! est bornée par le frottement et la raideur — à introduire par l'appelant.
//! Aucune constante matériau ou coefficient de frottement n'est supposé ici ;
//! l'appelant fournit toutes les données physiques.

/// Rapport d'amplification d'effort idéal de la genouillère
/// `R = F_out / F_in = 1 / (2·tan θ)` (sans dimension).
///
/// Panique si `half_angle_rad <= 0` (au point mort `θ = 0` le rapport diverge)
/// ou si `half_angle_rad >= π/2` (`tan θ` n'est plus défini physiquement).
pub fn toggle_force_ratio(half_angle_rad: f64) -> f64 {
    assert!(
        half_angle_rad > 0.0,
        "le demi-angle θ doit être strictement positif (θ = 0 est le point mort, rapport infini)"
    );
    assert!(
        half_angle_rad < core::f64::consts::FRAC_PI_2,
        "le demi-angle θ doit rester inférieur à π/2"
    );
    1.0 / (2.0 * half_angle_rad.tan())
}

/// Effort de sortie de la genouillère `F_out = F_in / (2·tan θ)` (N).
///
/// Panique si `input_force < 0`, si `half_angle_rad <= 0` (point mort) ou si
/// `half_angle_rad >= π/2`.
pub fn toggle_output_force(input_force: f64, half_angle_rad: f64) -> f64 {
    assert!(
        input_force >= 0.0,
        "l'effort d'entrée F_in ne peut pas être négatif"
    );
    input_force * toggle_force_ratio(half_angle_rad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn output_force_is_input_times_ratio() {
        // Identité de définition : F_out = F_in · R.
        let theta = 12.0_f64.to_radians();
        let f_in = 350.0;
        assert_relative_eq!(
            toggle_output_force(f_in, theta),
            f_in * toggle_force_ratio(theta),
            epsilon = 1e-12
        );
    }

    #[test]
    fn ratio_is_unity_at_the_natural_angle() {
        // R = 1 quand 2·tan θ = 1, soit θ = atan(1/2) ≈ 26.565°.
        let theta = (0.5_f64).atan();
        assert_relative_eq!(toggle_force_ratio(theta), 1.0, epsilon = 1e-12);
        // Cas chiffré réaliste : à 45° l'effort de sortie vaut F_in/2 (tan 45° = 1).
        let theta45 = core::f64::consts::FRAC_PI_4;
        assert_relative_eq!(toggle_output_force(1000.0, theta45), 500.0, epsilon = 1e-9);
    }

    #[test]
    fn amplification_grows_as_angle_shrinks() {
        // Plus θ diminue, plus l'amplification augmente (approche du point mort).
        let big = toggle_force_ratio(5.0_f64.to_radians());
        let small = toggle_force_ratio(25.0_f64.to_radians());
        assert!(big > small);
        // Et le rapport tend vers l'infini : θ très petit ⇒ R très grand.
        assert!(toggle_force_ratio(0.001) > 100.0);
    }

    #[test]
    fn ratio_scales_inversely_with_tan_theta() {
        // Proportionnalité : R·tan θ = 1/2 pour tout θ valide.
        for deg in [3.0_f64, 15.0, 40.0, 70.0]
        {
            let theta = deg.to_radians();
            assert_relative_eq!(
                toggle_force_ratio(theta) * theta.tan(),
                0.5,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn output_force_scales_linearly_with_input() {
        // Linéarité en F_in à angle fixé : doubler l'entrée double la sortie.
        let theta = 18.0_f64.to_radians();
        let base = toggle_output_force(200.0, theta);
        assert_relative_eq!(
            toggle_output_force(400.0, theta),
            2.0 * base,
            epsilon = 1e-9
        );
        // Un effort d'entrée nul donne une sortie nulle.
        assert_relative_eq!(toggle_output_force(0.0, theta), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "point mort")]
    fn dead_center_panics() {
        // θ = 0 : biellettes alignées, rapport infini interdit.
        toggle_force_ratio(0.0);
    }
}
