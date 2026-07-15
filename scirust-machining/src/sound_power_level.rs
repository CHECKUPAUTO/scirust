//! Acoustique — **niveau de puissance sonore** `Lw` (en décibels, dB) d'une
//! source, et conversion en niveau de pression sonore `Lp` en champ libre.
//!
//! ```text
//! niveau puissance        Lw   = 10·log10(W / W_ref)               [dB]
//! puissance (inverse)     W    = W_ref · 10^(Lw / 10)              [W]
//! puissance → pression    Lp   = Lw − 10·log10(4·π·r²) + 10·log10(Q)
//!  (champ libre, ponctuelle)
//! pression → puissance    Lw   = Lp + 10·log10(4·π·r²) − 10·log10(Q)
//! ```
//!
//! `W` puissance acoustique rayonnée [W], `W_ref` puissance de référence [W]
//! (1 pW = 1e-12 W dans l'air), `Lw` niveau de puissance sonore [dB], `Lp`
//! niveau de pression sonore [dB], `r` distance source→point [m], `Q` facteur
//! de directivité [sans dimension] (Q = 1 rayonnement sphérique, Q = 2
//! hémisphérique sur plan réfléchissant, etc.).
//!
//! **Limite honnête** : la puissance de référence `W_ref` (usuellement 1 pW
//! dans l'air) et le facteur de directivité `Q` sont **fournis par l'appelant** ;
//! aucune valeur n'est supposée par défaut. La conversion puissance↔pression
//! suppose une **source ponctuelle** en **champ libre** (pas d'absorption
//! atmosphérique, de réflexions ni de champ réverbérant) ; toute la directivité
//! est portée par `Q`, fourni à part.

use core::f64::consts::PI;

/// Niveau de puissance sonore `Lw = 10·log10(W / W_ref)` [dB].
///
/// `sound_power` et `reference_power` en watts (W) ; le niveau est en dB.
///
/// Panique si `sound_power <= 0` ou `reference_power <= 0`.
pub fn swl_from_power(sound_power: f64, reference_power: f64) -> f64 {
    assert!(
        sound_power > 0.0,
        "la puissance acoustique doit être strictement positive (W)"
    );
    assert!(
        reference_power > 0.0,
        "la puissance de référence doit être strictement positive (W)"
    );
    10.0 * (sound_power / reference_power).log10()
}

/// Puissance acoustique `W = W_ref · 10^(Lw / 10)` [W] (inverse de
/// [`swl_from_power`]).
///
/// `swl` en dB, `reference_power` en W ; la puissance rendue est en W.
///
/// Panique si `reference_power <= 0` ou si `swl` n'est pas fini.
pub fn swl_to_power(swl: f64, reference_power: f64) -> f64 {
    assert!(
        reference_power > 0.0,
        "la puissance de référence doit être strictement positive (W)"
    );
    assert!(swl.is_finite(), "le niveau doit être fini (dB)");
    reference_power * 10.0_f64.powf(swl / 10.0)
}

/// Niveau de pression sonore en champ libre pour une source ponctuelle
/// `Lp = Lw − 10·log10(4·π·r²) + 10·log10(Q)` [dB].
///
/// `swl` niveau de puissance en dB, `distance` en mètres, `directivity_factor`
/// facteur de directivité `Q` (sans dimension) ; le résultat est en dB.
///
/// Panique si `distance <= 0`, `directivity_factor <= 0` ou `swl` non fini.
pub fn swl_to_spl(swl: f64, distance: f64, directivity_factor: f64) -> f64 {
    assert!(
        distance > 0.0,
        "la distance doit être strictement positive (m)"
    );
    assert!(
        directivity_factor > 0.0,
        "le facteur de directivité doit être strictement positif"
    );
    assert!(
        swl.is_finite(),
        "le niveau de puissance doit être fini (dB)"
    );
    swl - 10.0 * (4.0 * PI * distance * distance).log10() + 10.0 * directivity_factor.log10()
}

/// Niveau de puissance sonore reconstitué depuis une pression mesurée en champ
/// libre `Lw = Lp + 10·log10(4·π·r²) − 10·log10(Q)` [dB] (inverse de
/// [`swl_to_spl`]).
///
/// `spl` niveau de pression en dB, `distance` en mètres, `directivity_factor`
/// facteur de directivité `Q` (sans dimension) ; le résultat est en dB.
///
/// Panique si `distance <= 0`, `directivity_factor <= 0` ou `spl` non fini.
pub fn spl_to_swl(spl: f64, distance: f64, directivity_factor: f64) -> f64 {
    assert!(
        distance > 0.0,
        "la distance doit être strictement positive (m)"
    );
    assert!(
        directivity_factor > 0.0,
        "le facteur de directivité doit être strictement positif"
    );
    assert!(spl.is_finite(), "le niveau de pression doit être fini (dB)");
    spl + 10.0 * (4.0 * PI * distance * distance).log10() - 10.0 * directivity_factor.log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const W_REF_AIR: f64 = 1e-12; // 1 pW, référence de l'air.

    #[test]
    fn reference_power_gives_zero_db() {
        // W = W_ref  ⇒  Lw = 10·log10(1) = 0 dB.
        assert_relative_eq!(swl_from_power(W_REF_AIR, W_REF_AIR), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn known_power_values() {
        // 1 mW ⇒ 10·log10(1e-3 / 1e-12) = 10·log10(1e9) = 90 dB exactement.
        assert_relative_eq!(swl_from_power(1e-3, W_REF_AIR), 90.0, epsilon = 1e-9);
        // 1 W ⇒ 10·log10(1e12) = 120 dB exactement.
        assert_relative_eq!(swl_from_power(1.0, W_REF_AIR), 120.0, epsilon = 1e-9);
    }

    #[test]
    fn power_and_level_are_reciprocal() {
        // swl_to_power ∘ swl_from_power = identité.
        let w = 3.7e-5; // W
        let level = swl_from_power(w, W_REF_AIR);
        assert_relative_eq!(swl_to_power(level, W_REF_AIR), w, epsilon = 1e-18);
    }

    #[test]
    fn doubling_power_adds_three_db() {
        // Doubler la puissance ajoute 10·log10(2) ≈ 3,0103 dB.
        let single = swl_from_power(1e-6, W_REF_AIR);
        let doubled = swl_from_power(2e-6, W_REF_AIR);
        assert_relative_eq!(doubled - single, 10.0 * 2.0_f64.log10(), epsilon = 1e-12);
    }

    #[test]
    fn spl_and_swl_are_reciprocal() {
        // spl_to_swl ∘ swl_to_spl = identité, pour Q quelconque et r quelconque.
        let lw = 95.0;
        let (r, q) = (3.5, 2.0);
        let lp = swl_to_spl(lw, r, q);
        assert_relative_eq!(spl_to_swl(lp, r, q), lw, epsilon = 1e-9);
    }

    #[test]
    fn free_field_reference_cases() {
        // Sphérique (Q = 1) à 1 m : Lp = Lw − 10·log10(4·π).
        // 4·π = 12,566370614… ⇒ 10·log10(4·π) = 10,99209864… dB.
        let lw = 90.0;
        let expected_sphere = lw - 10.0 * (4.0 * PI).log10();
        assert_relative_eq!(swl_to_spl(lw, 1.0, 1.0), expected_sphere, epsilon = 1e-12);
        assert_relative_eq!(swl_to_spl(lw, 1.0, 1.0), 79.007_901_36, epsilon = 1e-6);

        // Hémisphérique (Q = 2) à 1 m : Lp = Lw − 10·log10(2·π), soit +3,0103 dB
        // par rapport au cas sphérique.
        let expected_hemi = lw - 10.0 * (2.0 * PI).log10();
        assert_relative_eq!(swl_to_spl(lw, 1.0, 2.0), expected_hemi, epsilon = 1e-12);

        // Chaque doublement de distance retire 20·log10(2) ≈ 6,0206 dB.
        let drop = swl_to_spl(lw, 1.0, 1.0) - swl_to_spl(lw, 2.0, 1.0);
        assert_relative_eq!(drop, 20.0 * 2.0_f64.log10(), epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "strictement positive")]
    fn zero_power_panics() {
        swl_from_power(0.0, W_REF_AIR);
    }
}
