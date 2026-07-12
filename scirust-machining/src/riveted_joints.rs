//! Assemblages **rivés** (ou boulonnés travaillant au cisaillement) — modes de
//! ruine d'un joint à recouvrement et rendement.
//!
//! ```text
//! cisaillement des rivets  Fs = n·(π·d²/4)·τ_adm
//! matage (pression)        Fp = n·d·t·σ_p_adm
//! déchirure de la tôle     Ft = (p − d)·t·σ_t_adm    (section nette d'un pas)
//! tôle pleine (référence)  F0 = p·t·σ_t_adm
//! rendement                η = min(Fs, Fp, Ft)/F0
//! ```
//!
//! `d` diamètre du rivet (m), `t` épaisseur de tôle (m), `p` pas (entraxe des
//! rivets, m), `n` nombre de rivets par pas, `τ_adm`, `σ_p_adm`, `σ_t_adm`
//! contraintes admissibles au cisaillement, au matage et en traction.
//!
//! **Convention** : unités cohérentes de l'appelant. **Limite honnête** :
//! résistances **admissibles** des trois modes classiques (rivets, matage,
//! déchirure) sur un motif d'un pas ; le rendement compare la ruine la plus
//! faible à la tôle pleine. Ne traite ni les rangées multiples couplées, ni la
//! précharge des boulons HR.

use core::f64::consts::PI;

/// Résistance au **cisaillement des rivets** `Fs = n·(π·d²/4)·τ_adm`.
pub fn rivet_shear_strength(diameter: f64, count: u32, allowable_shear: f64) -> f64 {
    count as f64 * (PI * diameter * diameter / 4.0) * allowable_shear
}

/// Résistance au **matage** `Fp = n·d·t·σ_p_adm`.
pub fn bearing_strength(diameter: f64, thickness: f64, count: u32, allowable_bearing: f64) -> f64 {
    count as f64 * diameter * thickness * allowable_bearing
}

/// Résistance à la **déchirure** de la tôle sur un pas `Ft = (p − d)·t·σ_t_adm`.
///
/// Panique si `pitch <= diameter` (section nette nulle ou négative).
pub fn tearing_strength(pitch: f64, diameter: f64, thickness: f64, allowable_tension: f64) -> f64 {
    assert!(
        pitch > diameter,
        "le pas doit être supérieur au diamètre du rivet"
    );
    (pitch - diameter) * thickness * allowable_tension
}

/// Résistance de la **tôle pleine** sur un pas `F0 = p·t·σ_t_adm` (référence).
pub fn solid_plate_strength(pitch: f64, thickness: f64, allowable_tension: f64) -> f64 {
    pitch * thickness * allowable_tension
}

/// Rendement du joint `η = résistance minimale / tôle pleine` (dans `]0, 1]`).
///
/// Panique si `solid_strength <= 0`.
pub fn joint_efficiency(weakest_strength: f64, solid_plate_strength: f64) -> f64 {
    assert!(
        solid_plate_strength > 0.0,
        "la résistance de la tôle pleine doit être strictement positive"
    );
    weakest_strength / solid_plate_strength
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rivet_shear_and_bearing() {
        // d=20 mm, τ_adm=80 MPa, 1 rivet : Fs = π·400/4·80 ≈ 25 133 N (unités N, mm, MPa).
        let fs = rivet_shear_strength(20.0, 1, 80.0);
        assert_relative_eq!(fs, PI * 400.0 / 4.0 * 80.0, epsilon = 1e-6);
        // matage : Fp = 20·10·160 = 32 000 N.
        assert_relative_eq!(
            bearing_strength(20.0, 10.0, 1, 160.0),
            32_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn tearing_uses_net_section() {
        // p=60, d=20, t=10, σ_t=100 → Ft = (60−20)·10·100 = 40 000 N.
        assert_relative_eq!(
            tearing_strength(60.0, 20.0, 10.0, 100.0),
            40_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn efficiency_is_weakest_over_solid() {
        // Tôle pleine : F0 = 60·10·100 = 60 000 N.
        let f0 = solid_plate_strength(60.0, 10.0, 100.0);
        assert_relative_eq!(f0, 60_000.0, epsilon = 1e-6);
        // Mode le plus faible parmi rivet(25133), matage(32000), déchirure(40000).
        let fs = rivet_shear_strength(20.0, 1, 80.0);
        let weakest = fs
            .min(bearing_strength(20.0, 10.0, 1, 160.0))
            .min(tearing_strength(60.0, 20.0, 10.0, 100.0));
        let eta = joint_efficiency(weakest, f0);
        assert_relative_eq!(eta, fs / f0, epsilon = 1e-9); // le rivet gouverne
        assert!(eta > 0.0 && eta < 1.0);
    }

    #[test]
    #[should_panic(expected = "pas doit être supérieur")]
    fn pitch_below_diameter_panics() {
        tearing_strength(15.0, 20.0, 10.0, 100.0);
    }
}
