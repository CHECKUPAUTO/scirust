//! Table élévatrice à ciseaux (un étage) — effort d'un vérin horizontal, avantage
//! mécanique du mécanisme et hauteur de levage en fonction de l'angle des bras.
//!
//! ```text
//! effort du vérin      F = k · W / (2·tan θ)        (vérin horizontal)
//! avantage mécanique   MA = W / F = 2·tan θ / k
//! hauteur de levage    H = n · 2·L·sin θ
//! ```
//!
//! `W` charge verticale sur la table (N), `θ` angle des bras avec l'horizontale
//! (rad, `0 < θ < π/2`), `k` facteur géométrique adimensionnel qui regroupe les
//! rapports de bras de levier propres à l'attelage du vérin (`k = 1` pour le cas
//! canonique symétrique), `F` effort du vérin (N), `MA` avantage mécanique
//! (adimensionnel), `L` longueur d'un bras (m), `n` nombre d'étages de ciseaux,
//! `H` hauteur de la table (m). Quand la table descend, `θ → 0` et `tan θ → 0` :
//! l'effort du vérin diverge, ce qui traduit le point dur bien connu en position
//! basse.
//!
//! **Convention** : SI cohérent, angles en rad. **Limite honnête** : modèle
//! statique idéal d'un **seul étage** de ciseaux à vérin horizontal, frottement
//! des articulations et masse propre des bras négligés ; `k` synthétise la
//! géométrie exacte de l'attelage et **doit être fourni par l'appelant** — aucune
//! valeur « par défaut » n'est inventée ici. Pour un empilage de `n` étages
//! identiques, la hauteur s'additionne linéairement mais l'effort du vérin dépend
//! de l'architecture réelle et n'est pas modélisé au-delà de l'étage de base.

use core::f64::consts::FRAC_PI_2;

/// Effort du vérin horizontal `F = k · W / (2·tan θ)` (N).
///
/// Croît sans borne quand la table descend (`θ → 0`), ce qui reproduit le point
/// dur en position basse.
///
/// Panique si `load < 0`, si `geometry_factor <= 0`, ou si `arm_angle_rad`
/// n'est pas dans `]0, π/2[` (sinon `tan θ ≤ 0` et le dénominateur est invalide).
pub fn scissor_actuator_force(load: f64, arm_angle_rad: f64, geometry_factor: f64) -> f64 {
    assert!(load >= 0.0, "la charge doit être positive ou nulle");
    assert!(
        geometry_factor > 0.0,
        "le facteur géométrique doit être strictement positif"
    );
    assert!(
        arm_angle_rad > 0.0 && arm_angle_rad < FRAC_PI_2,
        "l'angle des bras doit vérifier 0 < θ < π/2"
    );
    geometry_factor * load / (2.0 * arm_angle_rad.tan())
}

/// Avantage mécanique du mécanisme `MA = W / F = 2·tan θ / k` (adimensionnel).
///
/// Réciproque du gain en effort : tend vers 0 en position basse (`θ → 0`) et
/// croît quand la table monte.
///
/// Panique si `geometry_factor <= 0` ou si `arm_angle_rad` n'est pas dans
/// `]0, π/2[`.
pub fn scissor_mechanical_advantage(arm_angle_rad: f64, geometry_factor: f64) -> f64 {
    assert!(
        geometry_factor > 0.0,
        "le facteur géométrique doit être strictement positif"
    );
    assert!(
        arm_angle_rad > 0.0 && arm_angle_rad < FRAC_PI_2,
        "l'angle des bras doit vérifier 0 < θ < π/2"
    );
    2.0 * arm_angle_rad.tan() / geometry_factor
}

/// Hauteur de la table `H = n · 2·L·sin θ` (m).
///
/// Somme des ouvertures verticales de `n` étages de ciseaux identiques.
///
/// Panique si `arm_length < 0`, si `n_stages == 0`, ou si `arm_angle_rad` n'est
/// pas dans `[0, π/2]` (angle physique d'un bras).
pub fn scissor_height(arm_length: f64, arm_angle_rad: f64, n_stages: u32) -> f64 {
    assert!(arm_length >= 0.0, "la longueur de bras doit être positive");
    assert!(n_stages > 0, "le nombre d'étages doit être au moins 1");
    assert!(
        (0.0..=FRAC_PI_2).contains(&arm_angle_rad),
        "l'angle des bras doit vérifier 0 ≤ θ ≤ π/2"
    );
    f64::from(n_stages) * 2.0 * arm_length * arm_angle_rad.sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn force_and_advantage_are_reciprocal() {
        // Identité F = W/MA : l'effort du vérin est l'inverse de l'avantage
        // mécanique appliqué à la charge, pour tout k et tout θ.
        let (w, theta, k) = (5000.0, 25.0_f64.to_radians(), 1.4);
        let f = scissor_actuator_force(w, theta, k);
        let ma = scissor_mechanical_advantage(theta, k);
        assert_relative_eq!(f, w / ma, epsilon = 1e-9);
    }

    #[test]
    fn actuator_force_grows_as_table_descends() {
        // θ plus petit (table basse) ⇒ effort du vérin plus grand : point dur.
        let (w, k) = (5000.0, 1.0);
        let high = scissor_actuator_force(w, 40.0_f64.to_radians(), k);
        let low = scissor_actuator_force(w, 10.0_f64.to_radians(), k);
        assert!(low > high);
    }

    #[test]
    fn force_is_proportional_to_load() {
        // À géométrie fixée, F est linéaire en W (doubler la charge double l'effort).
        let (theta, k) = (30.0_f64.to_radians(), 1.2);
        let f1 = scissor_actuator_force(2000.0, theta, k);
        let f2 = scissor_actuator_force(4000.0, theta, k);
        assert_relative_eq!(f2, 2.0 * f1, epsilon = 1e-9);
    }

    #[test]
    fn height_adds_up_over_stages_and_realistic_case() {
        // Additivité : n étages donnent n fois la hauteur d'un seul.
        let (l, theta) = (0.8, 35.0_f64.to_radians());
        let one = scissor_height(l, theta, 1);
        let three = scissor_height(l, theta, 3);
        assert_relative_eq!(three, 3.0 * one, epsilon = 1e-9);
        // Cas chiffré : bras L=0.8 m à 30°, un étage → 2·0.8·sin30° = 0.8 m.
        assert_relative_eq!(
            scissor_height(0.8, 30.0_f64.to_radians(), 1),
            0.8,
            epsilon = 1e-9
        );
    }

    #[test]
    fn height_is_maximal_at_ninety_degrees() {
        // θ = π/2 (bras verticaux) : H = n·2·L, ouverture maximale.
        let l = 0.5;
        assert_relative_eq!(
            scissor_height(l, FRAC_PI_2, 2),
            2.0 * 2.0 * l,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "0 < θ < π/2")]
    fn zero_angle_panics_on_actuator_force() {
        // θ = 0 : tan θ = 0, effort infini, entrée interdite.
        scissor_actuator_force(5000.0, 0.0, 1.0);
    }
}
