//! **Accouplement hydrodynamique** (coupleur à fluide) — glissement, vitesse de
//! sortie, rendement et couple transmis en régime permanent, à partir des
//! vitesses de rotation entrée/sortie et de la loi de similitude du couple.
//!
//! ```text
//! glissement       s   = (N_in - N_out) / N_in
//! vitesse sortie   N_out = N_in·(1 - s)
//! rendement        η   = N_out / N_in   ( = 1 - s )
//! couple transmis  T   = λ·ρ·N_in²·D⁵
//! ```
//!
//! `N_in` vitesse de l'arbre menant (tr/s ou rad/s, unité cohérente), `N_out`
//! vitesse de l'arbre mené (même unité que `N_in`), `s` glissement (sans unité,
//! 0 ≤ s ≤ 1), `η` rendement de transmission (sans unité, 0 ≤ η ≤ 1),
//! `λ` coefficient de couple (sans unité), `ρ` masse volumique du fluide
//! (kg/m³), `D` diamètre de la roue à aubes / impulseur (m), `T` couple transmis
//! (N·m si `N_in` est en tr/s et que `λ` intègre le facteur associé — unité
//! cohérente avec la convention de `λ`).
//!
//! **Convention** : unités SI cohérentes. Dans un coupleur hydrodynamique, le
//! couple d'entrée est **égal** au couple de sortie (pas de multiplication de
//! couple, contrairement à un convertisseur) ; le rendement se réduit donc au
//! rapport des vitesses `η = N_out / N_in = 1 - s`, les pertes étant dissipées en
//! chaleur dans le glissement.
//!
//! **Limite honnête** : modèle de **régime permanent** (pas de transitoire,
//! d'inertie ni de remplissage partiel). Le coefficient de couple `λ` et la masse
//! volumique `ρ` du fluide sont des données **fournies par l'appelant** (elles
//! dépendent du point de fonctionnement, de la géométrie et de la température) ;
//! aucune valeur « par défaut » n'est supposée. La loi `T = λ·ρ·N²·D⁵` est la loi
//! de similitude des turbomachines, valable à coefficient `λ` fixé.

/// Glissement `s = (N_in - N_out)/N_in` d'un coupleur hydrodynamique.
///
/// Panique si `input_speed <= 0`, `output_speed < 0`, ou si
/// `output_speed > input_speed` (le mené ne peut pas dépasser le menant).
pub fn fluidcoup_slip(input_speed: f64, output_speed: f64) -> f64 {
    assert!(
        input_speed > 0.0,
        "la vitesse d'entrée N_in doit être strictement positive"
    );
    assert!(
        output_speed >= 0.0,
        "la vitesse de sortie N_out ne peut pas être négative"
    );
    assert!(
        output_speed <= input_speed,
        "la vitesse de sortie N_out ne peut pas dépasser la vitesse d'entrée N_in"
    );
    (input_speed - output_speed) / input_speed
}

/// Vitesse de sortie `N_out = N_in·(1 - s)` à partir du glissement.
///
/// Panique si `input_speed < 0` ou si `slip` n'est pas dans `[0, 1]`.
pub fn fluidcoup_output_speed(input_speed: f64, slip: f64) -> f64 {
    assert!(
        input_speed >= 0.0,
        "la vitesse d'entrée N_in ne peut pas être négative"
    );
    assert!(
        (0.0..=1.0).contains(&slip),
        "le glissement s doit être dans [0, 1]"
    );
    input_speed * (1.0 - slip)
}

/// Rendement de transmission `η = N_out/N_in` d'un coupleur hydrodynamique
/// (égal à `1 - s`, le couple entrée/sortie étant identique en régime).
///
/// Panique si `input_speed <= 0`, `output_speed < 0`, ou si
/// `output_speed > input_speed`.
pub fn fluidcoup_efficiency(input_speed: f64, output_speed: f64) -> f64 {
    assert!(
        input_speed > 0.0,
        "la vitesse d'entrée N_in doit être strictement positive"
    );
    assert!(
        output_speed >= 0.0,
        "la vitesse de sortie N_out ne peut pas être négative"
    );
    assert!(
        output_speed <= input_speed,
        "la vitesse de sortie N_out ne peut pas dépasser la vitesse d'entrée N_in"
    );
    output_speed / input_speed
}

/// Couple transmis `T = λ·ρ·N_in²·D⁵` (loi de similitude : couple ∝ N²).
///
/// Panique si `torque_coefficient < 0`, `density <= 0`, `input_speed < 0`, ou
/// `impeller_diameter <= 0`.
pub fn fluidcoup_torque(
    torque_coefficient: f64,
    density: f64,
    input_speed: f64,
    impeller_diameter: f64,
) -> f64 {
    assert!(
        torque_coefficient >= 0.0,
        "le coefficient de couple λ ne peut pas être négatif"
    );
    assert!(
        density > 0.0,
        "la masse volumique ρ doit être strictement positive"
    );
    assert!(
        input_speed >= 0.0,
        "la vitesse d'entrée N_in ne peut pas être négative"
    );
    assert!(
        impeller_diameter > 0.0,
        "le diamètre d'impulseur D doit être strictement positif"
    );
    torque_coefficient * density * input_speed * input_speed * impeller_diameter.powi(5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn slip_and_output_speed_are_reciprocal() {
        // Réciprocité : N_out(N_in, s(N_in, N_out)) = N_out.
        let (n_in, n_out) = (25.0, 24.0);
        let s = fluidcoup_slip(n_in, n_out);
        assert_relative_eq!(fluidcoup_output_speed(n_in, s), n_out, epsilon = 1e-12);
    }

    #[test]
    fn efficiency_equals_one_minus_slip() {
        // Identité η = 1 - s (couple identique entrée/sortie en régime).
        let (n_in, n_out) = (25.0, 23.5);
        let s = fluidcoup_slip(n_in, n_out);
        let eta = fluidcoup_efficiency(n_in, n_out);
        assert_relative_eq!(eta, 1.0 - s, epsilon = 1e-12);
    }

    #[test]
    fn zero_slip_gives_unit_efficiency_and_equal_speeds() {
        // Cas limite s = 0 : N_out = N_in et η = 1.
        let n_in = 30.0;
        assert_relative_eq!(fluidcoup_output_speed(n_in, 0.0), n_in, epsilon = 1e-12);
        assert_relative_eq!(fluidcoup_efficiency(n_in, n_in), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn torque_is_proportional_to_speed_squared() {
        // T ∝ N² : doubler N_in quadruple le couple (λ, ρ, D fixés).
        let (lambda, rho, d) = (2.5e-4, 860.0, 0.30);
        let t1 = fluidcoup_torque(lambda, rho, 20.0, d);
        let t2 = fluidcoup_torque(lambda, rho, 40.0, d);
        assert_relative_eq!(t2, 4.0 * t1, epsilon = 1e-9);
    }

    #[test]
    fn torque_is_proportional_to_diameter_to_the_fifth() {
        // T ∝ D⁵ : doubler le diamètre multiplie le couple par 2⁵ = 32.
        let (lambda, rho, n) = (3.0e-4, 900.0, 15.0);
        let t1 = fluidcoup_torque(lambda, rho, n, 0.20);
        let t2 = fluidcoup_torque(lambda, rho, n, 0.40);
        assert_relative_eq!(t2, 32.0 * t1, epsilon = 1e-9);
    }

    #[test]
    fn realistic_torque_case() {
        // λ = 2,5e-4 ; ρ = 860 kg/m³ ; N_in = 24 ; D = 0,30 m :
        // T = 2,5e-4·860·24²·0,30⁵
        //   = 0,215·576·0,00243 = 0,300 931 2 (unité cohérente).
        let t = fluidcoup_torque(2.5e-4, 860.0, 24.0, 0.30);
        let expected = 2.5e-4 * 860.0 * 24.0 * 24.0 * 0.30_f64.powi(5);
        assert_relative_eq!(t, expected, epsilon = 1e-9);
        assert!(t > 0.300 && t < 0.301);
    }

    #[test]
    #[should_panic(
        expected = "la vitesse de sortie N_out ne peut pas dépasser la vitesse d'entrée N_in"
    )]
    fn output_faster_than_input_panics() {
        fluidcoup_slip(20.0, 25.0);
    }
}
