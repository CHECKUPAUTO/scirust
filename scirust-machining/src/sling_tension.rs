//! Élingues de levage — tension dans les brins d'une élingue à plusieurs
//! brins symétriques et effort horizontal induit par l'angle d'ouverture.
//!
//! ```text
//! tension d'un brin      T = W/(n·cos θ)
//! facteur de charge      k = 1/cos θ
//! effort horizontal      H = T·sin θ
//! ```
//!
//! `W` charge totale suspendue (N), `n` nombre de brins porteurs (–), `θ`
//! angle du brin par rapport à la **verticale** (rad), `T` tension dans un
//! brin (N), `k` facteur de charge (multiplicateur de tension dû à l'angle,
//! –), `H` composante horizontale de la tension d'un brin (N), tendant à
//! comprimer/rapprocher les points d'accrochage.
//!
//! **Convention** : SI cohérent, angles en radians mesurés depuis la
//! verticale (θ = 0 → brins verticaux). **Limite honnête** : modèle statique
//! d'une élingue à brins **symétriques** et charge **également répartie**
//! entre les `n` brins (hypothèse idéalisée : en pratique une élingue à 3 ou
//! 4 brins ne répartit pas rigoureusement la charge). Angle limité à
//! `θ < 90°` (`cos θ > 0`) ; la tension diverge quand `θ → 90°`. Aucune prise
//! en compte du poids propre des brins, de l'élasticité ni des effets
//! dynamiques (levage brusque). Les charges admissibles (CMU/WLL) et
//! coefficients de sécurité sont **fournis par l'appelant**.

use core::f64::consts::PI;

/// Tension dans un brin `T = W/(n·cos θ)` (N).
///
/// Répartition égale de la charge `W` entre `n` brins symétriques, chaque
/// brin faisant l'angle `angle_from_vertical_rad` avec la verticale.
///
/// Panique si `total_load < 0`, si `n_legs == 0`, ou si l'angle n'est pas
/// dans `[0, π/2[`.
pub fn sling_leg_tension(total_load: f64, n_legs: u32, angle_from_vertical_rad: f64) -> f64 {
    assert!(total_load >= 0.0, "la charge totale doit être positive");
    assert!(
        n_legs > 0,
        "le nombre de brins doit être strictement positif"
    );
    assert!(
        (0.0..PI / 2.0).contains(&angle_from_vertical_rad),
        "l'angle depuis la verticale doit être dans [0, π/2["
    );
    total_load / (f64::from(n_legs) * angle_from_vertical_rad.cos())
}

/// Facteur de charge `k = 1/cos θ` (–).
///
/// Multiplicateur appliqué à la tension d'un brin vertical équivalent du fait
/// de l'ouverture angulaire de l'élingue (`k ≥ 1`).
///
/// Panique si l'angle n'est pas dans `[0, π/2[`.
pub fn sling_load_factor(angle_from_vertical_rad: f64) -> f64 {
    assert!(
        (0.0..PI / 2.0).contains(&angle_from_vertical_rad),
        "l'angle depuis la verticale doit être dans [0, π/2["
    );
    1.0 / angle_from_vertical_rad.cos()
}

/// Composante horizontale de la tension d'un brin `H = T·sin θ` (N).
///
/// Effort de rapprochement des points d'accrochage induit par l'inclinaison
/// du brin de tension `leg_tension`.
///
/// Panique si `leg_tension < 0` ou si l'angle n'est pas dans `[0, π/2[`.
pub fn sling_horizontal_force(leg_tension: f64, angle_from_vertical_rad: f64) -> f64 {
    assert!(leg_tension >= 0.0, "la tension du brin doit être positive");
    assert!(
        (0.0..PI / 2.0).contains(&angle_from_vertical_rad),
        "l'angle depuis la verticale doit être dans [0, π/2["
    );
    leg_tension * angle_from_vertical_rad.sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn vertical_legs_share_load_equally() {
        // θ = 0 : chaque brin porte exactement W/n, sans majoration.
        let w = 12_000.0;
        let n = 4;
        let t = sling_leg_tension(w, n, 0.0);
        assert_relative_eq!(t, w / f64::from(n), epsilon = 1e-9);
        // Somme des composantes verticales = W.
        assert_relative_eq!(f64::from(n) * t * 0.0_f64.cos(), w, epsilon = 1e-6);
    }

    #[test]
    fn tension_is_factor_times_vertical_share() {
        // Identité : T = (W/n)·k  avec k = 1/cos θ.
        let w = 8_000.0;
        let n = 2;
        let theta = PI / 6.0; // 30°
        let t = sling_leg_tension(w, n, theta);
        let k = sling_load_factor(theta);
        assert_relative_eq!(t, (w / f64::from(n)) * k, epsilon = 1e-9);
    }

    #[test]
    fn vertical_equilibrium_holds() {
        // La somme des composantes verticales des n brins équilibre W :
        // n·T·cos θ = W, quel que soit θ.
        let w = 5_000.0;
        let n = 3;
        let theta = PI / 5.0; // 36°
        let t = sling_leg_tension(w, n, theta);
        assert_relative_eq!(f64::from(n) * t * theta.cos(), w, epsilon = 1e-6);
    }

    #[test]
    fn load_factor_known_sixty_degrees() {
        // À 60° depuis la verticale : k = 1/cos60° = 2 (cas classique de
        // sécurité levage : la tension double).
        assert_relative_eq!(sling_load_factor(PI / 3.0), 2.0, epsilon = 1e-9);
    }

    #[test]
    fn horizontal_force_matches_pythagoras() {
        // H = T·sin θ et la composante verticale V = T·cos θ vérifient
        // H² + V² = T².
        let t = 10_000.0;
        let theta = PI / 4.0; // 45°
        let h = sling_horizontal_force(t, theta);
        let v = t * theta.cos();
        assert_relative_eq!(h * h + v * v, t * t, epsilon = 1e-3);
        // À 45° la composante horizontale égale la verticale.
        assert_relative_eq!(h, v, epsilon = 1e-9);
    }

    #[test]
    fn wider_angle_increases_tension() {
        // Ouvrir l'angle augmente strictement la tension (T ∝ 1/cos θ).
        let w = 6_000.0;
        let n = 2;
        let t_narrow = sling_leg_tension(w, n, PI / 12.0); // 15°
        let t_wide = sling_leg_tension(w, n, PI / 3.0); // 60°
        assert!(t_wide > t_narrow);
    }

    #[test]
    #[should_panic(expected = "verticale")]
    fn right_angle_panics() {
        // θ = 90° : cos θ = 0, tension infinie → interdit.
        sling_leg_tension(1_000.0, 2, PI / 2.0);
    }
}
