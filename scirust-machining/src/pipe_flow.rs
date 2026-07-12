//! Pertes de charge en conduite — perte régulière de **Darcy-Weisbach**, facteur
//! de frottement (laminaire, **Colebrook-White** implicite, **Swamee-Jain**
//! explicite) et pertes singulières.
//!
//! ```text
//! perte régulière (m)   h_f = f·(L/D)·v²/(2g)
//! laminaire (Re<2300)   f = 64/Re
//! Colebrook (turbulent) 1/√f = −2·log10(ε/(3,7D) + 2,51/(Re·√f))
//! Swamee-Jain (explicite) f = 0,25/[log10(ε/(3,7D) + 5,74/Re^0,9)]²
//! perte singulière (m)  h_s = K·v²/(2g)
//! ```
//!
//! `f` facteur de frottement de Darcy (sans dimension), `L` longueur (m), `D`
//! diamètre (m), `v` vitesse débitante (m/s), `ε/D` rugosité relative, `K`
//! coefficient de perte singulière, `Re` nombre de Reynolds.
//!
//! **Convention** : SI cohérent, pertes en mètres de colonne. **Limite honnête** :
//! conduite circulaire en charge, régime **permanent** ; le facteur de Colebrook
//! vaut pour le régime turbulent (`Re ≳ 4000`). `Re` et `ε/D` sont fournis par
//! l'appelant (voir [`crate::bernoulli::reynolds_number`]).

/// Perte de charge régulière de Darcy-Weisbach `h_f = f·(L/D)·v²/(2g)` (m).
///
/// Panique si `diameter <= 0` ou `g <= 0`.
pub fn darcy_head_loss(f: f64, length: f64, diameter: f64, velocity: f64, g: f64) -> f64 {
    assert!(diameter > 0.0 && g > 0.0, "D > 0 et g > 0 requis");
    f * (length / diameter) * velocity * velocity / (2.0 * g)
}

/// Perte singulière `h_s = K·v²/(2g)` (m).
///
/// Panique si `g <= 0`.
pub fn minor_loss(k: f64, velocity: f64, g: f64) -> f64 {
    assert!(g > 0.0, "g doit être strictement positif");
    k * velocity * velocity / (2.0 * g)
}

/// Facteur de frottement **laminaire** `f = 64/Re`.
///
/// Panique si `re <= 0`.
pub fn laminar_friction_factor(re: f64) -> f64 {
    assert!(
        re > 0.0,
        "le nombre de Reynolds doit être strictement positif"
    );
    64.0 / re
}

/// Facteur de frottement turbulent, approximation **explicite de Swamee-Jain**
/// `f = 0,25/[log10(ε/(3,7D) + 5,74/Re^0,9)]²`.
///
/// Panique si `re <= 0` ou `relative_roughness < 0`.
pub fn swamee_jain_friction(re: f64, relative_roughness: f64) -> f64 {
    assert!(
        re > 0.0 && relative_roughness >= 0.0,
        "Re > 0 et ε/D ≥ 0 requis"
    );
    let arg = relative_roughness / 3.7 + 5.74 / re.powf(0.9);
    let log = arg.log10();
    0.25 / (log * log)
}

/// Facteur de frottement turbulent par résolution **implicite de Colebrook-White**
/// `1/√f = −2·log10(ε/(3,7D) + 2,51/(Re·√f))`.
///
/// Résolu par itération de point fixe amorcée sur Swamee-Jain (convergence en
/// quelques itérations). Panique si `re <= 0` ou `relative_roughness < 0`.
pub fn colebrook_friction(re: f64, relative_roughness: f64) -> f64 {
    assert!(
        re > 0.0 && relative_roughness >= 0.0,
        "Re > 0 et ε/D ≥ 0 requis"
    );
    // Amorce explicite puis itération sur 1/√f.
    let mut inv_sqrt_f = 1.0 / swamee_jain_friction(re, relative_roughness).sqrt();
    for _ in 0..50
    {
        let next = -2.0 * (relative_roughness / 3.7 + 2.51 * inv_sqrt_f / re).log10();
        if (next - inv_sqrt_f).abs() < 1e-12
        {
            inv_sqrt_f = next;
            break;
        }
        inv_sqrt_f = next;
    }
    1.0 / (inv_sqrt_f * inv_sqrt_f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn darcy_and_minor_losses() {
        // f=0,02, L=100, D=0,1, v=2, g=9,81 → h_f = 0,02·1000·4/19,62 ≈ 4,077 m.
        let hf = darcy_head_loss(0.02, 100.0, 0.1, 2.0, 9.81);
        assert_relative_eq!(hf, 0.02 * 1000.0 * 4.0 / 19.62, epsilon = 1e-9);
        // singulière K=0,5 (entrée) : h_s = 0,5·4/19,62 ≈ 0,102 m.
        assert_relative_eq!(
            minor_loss(0.5, 2.0, 9.81),
            0.5 * 4.0 / 19.62,
            epsilon = 1e-9
        );
    }

    #[test]
    fn laminar_factor_matches_definition() {
        // Re=2000 → f = 64/2000 = 0,032.
        assert_relative_eq!(laminar_friction_factor(2000.0), 0.032, epsilon = 1e-12);
    }

    #[test]
    fn colebrook_agrees_with_swamee_jain() {
        // Les deux corrélations turbulentes doivent être proches (<3%).
        let (re, rr) = (1e5, 0.001);
        let f_c = colebrook_friction(re, rr);
        let f_sj = swamee_jain_friction(re, rr);
        assert_relative_eq!(f_c, f_sj, max_relative = 0.03);
        // valeur physique plausible (0,02–0,025 dans ce régime).
        assert!(f_c > 0.018 && f_c < 0.026);
    }

    #[test]
    fn colebrook_solves_its_own_implicit_equation() {
        // Le f renvoyé doit satisfaire 1/√f = −2·log10(ε/3,7D + 2,51/(Re√f)).
        let (re, rr) = (5e4, 0.0005);
        let f = colebrook_friction(re, rr);
        let lhs = 1.0 / f.sqrt();
        let rhs = -2.0 * (rr / 3.7 + 2.51 / (re * f.sqrt())).log10();
        assert_relative_eq!(lhs, rhs, epsilon = 1e-9);
    }

    #[test]
    fn smoother_pipe_has_lower_friction() {
        // À Re égal, une conduite plus lisse a un f plus faible.
        let re = 1e5;
        assert!(colebrook_friction(re, 0.0001) < colebrook_friction(re, 0.01));
    }

    #[test]
    #[should_panic(expected = "Reynolds")]
    fn zero_reynolds_laminar_panics() {
        laminar_friction_factor(0.0);
    }
}
