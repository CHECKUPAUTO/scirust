//! Poteau-poutre (« beam-column ») : amplification des efforts de flexion par la
//! compression axiale (effet **P-delta** du second ordre).
//!
//! ```text
//! facteur d'amplification   Af = 1 / (1 − P/Pcr)
//! moment amplifié           M  = M0 · Af = M0 / (1 − P/Pcr)
//! flèche amplifiée          δ  = δ0 · Af = δ0 / (1 − P/Pcr)
//! ```
//!
//! `P` charge axiale de compression (N), `Pcr` charge critique d'Euler (N),
//! `M0` moment de flexion du premier ordre (N·m), `δ0` flèche du premier ordre
//! (m). Le rapport `P/Pcr` est sans dimension et doit rester dans `[0, 1[`.
//!
//! **Convention** : SI cohérent. **Limite honnête** : amplificateur **élastique**
//! du second ordre (approximation classique de l'effet P-delta pour une barre
//! droite en compression + flexion). La charge critique d'Euler `Pcr` est
//! **fournie par l'appelant** (voir [`crate::buckling`]) : aucune valeur de
//! module, d'inertie ou de longueur n'est supposée ici. La formule diverge
//! quand `P → Pcr` ; on exige strictement `P < Pcr`. Complète
//! [`crate::buckling`].

/// Rapport de charge `P/Pcr` (sans dimension), commun aux trois amplificateurs.
///
/// Panique si `euler_critical_load <= 0`, si `axial_load < 0`, ou si
/// `axial_load >= euler_critical_load` (la barre a atteint ou dépassé le
/// flambement, l'amplificateur diverge).
pub fn beam_column_load_ratio(axial_load: f64, euler_critical_load: f64) -> f64 {
    assert!(
        euler_critical_load > 0.0,
        "la charge critique d'Euler doit être strictement positive"
    );
    assert!(
        axial_load >= 0.0,
        "la charge axiale de compression doit être positive ou nulle"
    );
    assert!(
        axial_load < euler_critical_load,
        "la charge axiale doit rester strictement inférieure à la charge critique (P < Pcr)"
    );
    axial_load / euler_critical_load
}

/// Facteur d'amplification P-delta `Af = 1 / (1 − P/Pcr)` (sans dimension).
///
/// Vaut `1` sans charge axiale et tend vers l'infini quand `P → Pcr`.
///
/// Panique si `euler_critical_load <= 0`, si `axial_load < 0`, ou si
/// `axial_load >= euler_critical_load`.
pub fn beam_column_amplification_factor(axial_load: f64, euler_critical_load: f64) -> f64 {
    1.0 / (1.0 - beam_column_load_ratio(axial_load, euler_critical_load))
}

/// Moment de flexion amplifié `M = M0 / (1 − P/Pcr)` (N·m).
///
/// Panique si `euler_critical_load <= 0`, si `axial_load < 0`, ou si
/// `axial_load >= euler_critical_load`.
pub fn beam_column_amplified_moment(
    first_order_moment: f64,
    axial_load: f64,
    euler_critical_load: f64,
) -> f64 {
    first_order_moment * beam_column_amplification_factor(axial_load, euler_critical_load)
}

/// Flèche amplifiée `δ = δ0 / (1 − P/Pcr)` (m).
///
/// Panique si `euler_critical_load <= 0`, si `axial_load < 0`, ou si
/// `axial_load >= euler_critical_load`.
pub fn beam_column_amplified_deflection(
    first_order_deflection: f64,
    axial_load: f64,
    euler_critical_load: f64,
) -> f64 {
    first_order_deflection * beam_column_amplification_factor(axial_load, euler_critical_load)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn no_axial_load_gives_unit_amplification() {
        // P = 0 → Af = 1, aucun effet du second ordre.
        assert_relative_eq!(
            beam_column_amplification_factor(0.0, 400e3),
            1.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            beam_column_amplified_moment(12.0, 0.0, 400e3),
            12.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn realistic_amplification_case() {
        // P = 100 kN, Pcr = 400 kN → P/Pcr = 0,25 → Af = 1/0,75 = 4/3.
        let (p, pcr) = (100e3, 400e3);
        assert_relative_eq!(beam_column_load_ratio(p, pcr), 0.25, epsilon = 1e-12);
        assert_relative_eq!(
            beam_column_amplification_factor(p, pcr),
            4.0 / 3.0,
            epsilon = 1e-12
        );
        // M0 = 10 kN·m → M = 10·4/3 ≈ 13,333 kN·m.
        assert_relative_eq!(
            beam_column_amplified_moment(10e3, p, pcr),
            10e3 * 4.0 / 3.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn moment_and_deflection_share_the_same_amplifier() {
        // Le même facteur multiplie moment et flèche → M/M0 = δ/δ0.
        let (p, pcr) = (150e3, 500e3);
        let af = beam_column_amplification_factor(p, pcr);
        assert_relative_eq!(
            beam_column_amplified_moment(7.5, p, pcr),
            7.5 * af,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            beam_column_amplified_deflection(3.0e-3, p, pcr),
            3.0e-3 * af,
            epsilon = 1e-12
        );
    }

    #[test]
    fn amplification_is_proportional_in_the_input() {
        // Linéarité en M0 : doubler M0 double M (même charge axiale).
        let (p, pcr) = (80e3, 320e3);
        let m1 = beam_column_amplified_moment(5.0, p, pcr);
        let m2 = beam_column_amplified_moment(10.0, p, pcr);
        assert_relative_eq!(m2, 2.0 * m1, epsilon = 1e-12);
    }

    #[test]
    fn factor_grows_towards_infinity_near_critical_load() {
        // Plus P s'approche de Pcr, plus l'amplificateur augmente (monotone).
        let pcr = 1.0e6;
        let low = beam_column_amplification_factor(0.10 * pcr, pcr);
        let high = beam_column_amplification_factor(0.90 * pcr, pcr);
        assert!(high > low);
        // À 90 % de Pcr : Af = 1/0,1 = 10.
        assert_relative_eq!(high, 10.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "strictement inférieure à la charge critique")]
    fn axial_load_reaching_critical_load_panics() {
        beam_column_amplification_factor(400e3, 400e3);
    }
}
