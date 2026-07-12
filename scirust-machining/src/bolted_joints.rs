//! Assemblages **boulonnés précontraints** (esprit **VDI 2230**) — précharge au
//! serrage, facteur de charge et répartition d'un effort extérieur entre le
//! boulon et les pièces serrées.
//!
//! ```text
//! précharge au couple   F0 = T/(K·d)
//! facteur de charge     Φ = Cb/(Cb + Cp)        (Cb boulon, Cp pièces)
//! effort boulon max     Fb = F0 + Φ·Fe
//! serrage résiduel      Fr = F0 − (1 − Φ)·Fe
//! effort de décollement Fsep = F0/(1 − Φ)
//! ```
//!
//! `T` couple de serrage (N·m), `K` coefficient de couple (nut factor, ~0,2), `d`
//! diamètre nominal (m), `Cb`/`Cp` raideurs du boulon et des pièces (N/m), `Fe`
//! effort extérieur axial (N), `F0` précharge. Le boulon ne reprend qu'une
//! **fraction** `Φ` de l'effort extérieur ; le reste décharge les pièces.
//!
//! **Convention** : SI cohérent, efforts de traction positifs. **Limite
//! honnête** : modèle **linéaire** de raideurs en série (VDI 2230 simplifié),
//! chargement axial concentrique ; ne traite ni l'excentration, ni le tassement
//! (perte de précharge), ni la fatigue. `K`, `Cb`, `Cp` sont fournis par
//! l'appelant.

/// Précharge obtenue au serrage `F0 = T/(K·d)` (N).
///
/// Panique si `k*d <= 0`.
pub fn preload_from_torque(torque: f64, k_factor: f64, diameter: f64) -> f64 {
    assert!(
        k_factor * diameter > 0.0,
        "K·d doit être strictement positif"
    );
    torque / (k_factor * diameter)
}

/// Facteur de charge `Φ = Cb/(Cb + Cp)` (fraction de l'effort extérieur reprise
/// par le boulon).
///
/// Panique si `cb + cp <= 0`.
pub fn load_factor(bolt_stiffness: f64, member_stiffness: f64) -> f64 {
    let sum = bolt_stiffness + member_stiffness;
    assert!(
        sum > 0.0,
        "la raideur totale doit être strictement positive"
    );
    bolt_stiffness / sum
}

/// Effort **total** dans le boulon `Fb = F0 + Φ·Fe` (N).
pub fn bolt_working_load(preload: f64, load_factor: f64, external_load: f64) -> f64 {
    preload + load_factor * external_load
}

/// Serrage **résiduel** sur les pièces `Fr = F0 − (1 − Φ)·Fe` (N).
///
/// Devient négatif (décollement) si l'effort extérieur dépasse [`separation_load`].
pub fn residual_clamp_load(preload: f64, load_factor: f64, external_load: f64) -> f64 {
    preload - (1.0 - load_factor) * external_load
}

/// Effort extérieur provoquant le **décollement** du joint `Fsep = F0/(1 − Φ)` (N).
///
/// Panique si `load_factor >= 1`.
pub fn separation_load(preload: f64, load_factor: f64) -> f64 {
    assert!(load_factor < 1.0, "Φ doit être strictement inférieur à 1");
    preload / (1.0 - load_factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn preload_from_tightening_torque() {
        // M10 (d=0,010), K=0,2, T=40 N·m → F0 = 40/(0,2·0,010) = 20 000 N.
        assert_relative_eq!(
            preload_from_torque(40.0, 0.2, 0.010),
            20_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn load_factor_bounds() {
        // Boulon souple / pièces raides : Φ petit (le boulon voit peu l'effort).
        // Cb=1e8, Cp=9e8 → Φ = 0,1.
        assert_relative_eq!(load_factor(1e8, 9e8), 0.1, epsilon = 1e-12);
    }

    #[test]
    fn bolt_sees_only_a_fraction_of_external_load() {
        // F0=20000, Φ=0,2, Fe=5000 → Fb = 20000 + 1000 = 21000 N.
        assert_relative_eq!(
            bolt_working_load(20_000.0, 0.2, 5000.0),
            21_000.0,
            epsilon = 1e-9
        );
        // Le serrage résiduel tombe de (1−Φ)·Fe = 4000 → 16000 N.
        assert_relative_eq!(
            residual_clamp_load(20_000.0, 0.2, 5000.0),
            16_000.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn separation_when_external_exceeds_threshold() {
        // Fsep = F0/(1−Φ) = 20000/0,8 = 25000 N. À cet effort, serrage résiduel = 0.
        let fsep = separation_load(20_000.0, 0.2);
        assert_relative_eq!(fsep, 25_000.0, epsilon = 1e-9);
        assert_relative_eq!(
            residual_clamp_load(20_000.0, 0.2, fsep),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "Φ doit être strictement inférieur à 1")]
    fn unit_load_factor_panics() {
        separation_load(20_000.0, 1.0);
    }
}
