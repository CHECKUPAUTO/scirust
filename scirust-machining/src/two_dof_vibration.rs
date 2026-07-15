//! Vibration libre d'un système **non amorti à 2 degrés de liberté** (chaîne
//! masse-ressort) : deux pulsations propres et mesure du couplage.
//!
//! Topologie : masse `m1` reliée au bâti par `k1` et à la masse `m2` par `k2`
//! (bâti–k1–m1–k2–m2). L'équation caractéristique en `ω²` s'écrit :
//!
//! ```text
//! ω⁴ − ω²·b + c = 0
//! b = (k1 + k2)/m1 + k2/m2          (somme des ω² des deux modes)
//! c = k1·k2 / (m1·m2)               (produit des ω² des deux modes)
//! ω²        = ( b ∓ √(b² − 4c) ) / 2
//! ω_basse   = √( (b − √(b² − 4c)) / 2 )
//! ω_haute   = √( (b + √(b² − 4c)) / 2 )
//! couplage  = k2 / k1
//! ```
//!
//! `m1`, `m2` masses (kg) ; `k1`, `k2` raideurs (N/m) ; `b` (rad²/s²) ;
//! `c` (rad⁴/s⁴) ; pulsations `ω` (rad/s) ; couplage sans dimension.
//!
//! **Convention** : SI. **Limite honnête** : système linéaire **à 2 ddl NON
//! amorti**, pour la seule topologie décrite (bâti–k1–m1–k2–m2) ; masses et
//! raideurs sont **fournies par l'appelant** (aucune valeur de matériau, de
//! procédé ou de « défaut » n'est inventée). Le discriminant `b² − 4c` est
//! toujours ≥ 0 pour ce système : les deux modes sont réels. Distinct de
//! [`crate::vibrations`] qui traite le cas à 1 ddl.

/// Discriminant `b² − 4c` de l'équation caractéristique en `ω²` (rad⁴/s⁴).
///
/// Avec `b = (k1+k2)/m1 + k2/m2` et `c = k1·k2/(m1·m2)`. Toujours ≥ 0 pour
/// cette topologie, garantissant deux modes réels.
///
/// Panique si `mass1 <= 0`, `mass2 <= 0`, `stiffness1 < 0` ou `stiffness2 < 0`.
pub fn twodof_frequency_equation_discriminant(
    mass1: f64,
    mass2: f64,
    stiffness1: f64,
    stiffness2: f64,
) -> f64 {
    assert!(
        mass1 > 0.0 && mass2 > 0.0,
        "les masses doivent être strictement positives"
    );
    assert!(
        stiffness1 >= 0.0 && stiffness2 >= 0.0,
        "les raideurs doivent être positives ou nulles"
    );
    let b = (stiffness1 + stiffness2) / mass1 + stiffness2 / mass2;
    let c = stiffness1 * stiffness2 / (mass1 * mass2);
    b * b - 4.0 * c
}

/// Pulsations propres `(ω_basse, ω_haute)` en rad/s, racines de l'équation
/// caractéristique `ω⁴ − ω²·b + c = 0`.
///
/// `ω² = (b ∓ √(b² − 4c)) / 2` avec `b = (k1+k2)/m1 + k2/m2` et
/// `c = k1·k2/(m1·m2)`. Renvoie d'abord la pulsation basse, puis la haute.
///
/// Panique si `mass1 <= 0`, `mass2 <= 0`, `stiffness1 < 0` ou `stiffness2 < 0`.
pub fn twodof_natural_frequencies_rad(
    mass1: f64,
    mass2: f64,
    stiffness1: f64,
    stiffness2: f64,
) -> (f64, f64) {
    assert!(
        mass1 > 0.0 && mass2 > 0.0,
        "les masses doivent être strictement positives"
    );
    assert!(
        stiffness1 >= 0.0 && stiffness2 >= 0.0,
        "les raideurs doivent être positives ou nulles"
    );
    let b = (stiffness1 + stiffness2) / mass1 + stiffness2 / mass2;
    let c = stiffness1 * stiffness2 / (mass1 * mass2);
    // Discriminant borné à 0 pour absorber d'éventuelles erreurs d'arrondi.
    let root = (b * b - 4.0 * c).max(0.0).sqrt();
    let omega_low = ((b - root) / 2.0).max(0.0).sqrt();
    let omega_high = ((b + root) / 2.0).sqrt();
    (omega_low, omega_high)
}

/// Facteur de couplage `k2 / k1` (sans dimension), mesure de l'intensité du
/// couplage inter-masses relativement à l'ancrage au bâti.
///
/// Panique si `stiffness1 <= 0` ou `stiffness2 < 0`.
pub fn twodof_coupling_factor(stiffness2: f64, stiffness1: f64) -> f64 {
    assert!(
        stiffness1 > 0.0,
        "la raideur d'ancrage k1 doit être strictement positive"
    );
    assert!(
        stiffness2 >= 0.0,
        "la raideur de couplage k2 doit être positive ou nulle"
    );
    stiffness2 / stiffness1
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cas_chiffre_realiste() {
        // m1=2 kg, m2=1 kg, k1=100 N/m, k2=50 N/m.
        // b = (100+50)/2 + 50/1 = 75 + 50 = 125.
        // c = 100·50/(2·1) = 2500.
        // disc = 125² − 4·2500 = 15625 − 10000 = 5625 ; √ = 75.
        // ω²_bas = (125−75)/2 = 25 → ω = 5 ; ω²_haut = (125+75)/2 = 100 → ω = 10.
        let disc = twodof_frequency_equation_discriminant(2.0, 1.0, 100.0, 50.0);
        assert_relative_eq!(disc, 5625.0, epsilon = 1e-9);
        let (lo, hi) = twodof_natural_frequencies_rad(2.0, 1.0, 100.0, 50.0);
        assert_relative_eq!(lo, 5.0, max_relative = 1e-9);
        assert_relative_eq!(hi, 10.0, max_relative = 1e-9);
    }

    #[test]
    fn identites_somme_et_produit_des_carres() {
        // ω²_bas + ω²_haut = b ; ω²_bas · ω²_haut = c (relations de Viète).
        let (m1, m2, k1, k2) = (3.0_f64, 1.5_f64, 220.0_f64, 90.0_f64);
        let b = (k1 + k2) / m1 + k2 / m2;
        let c = k1 * k2 / (m1 * m2);
        let (lo, hi) = twodof_natural_frequencies_rad(m1, m2, k1, k2);
        assert_relative_eq!(lo * lo + hi * hi, b, max_relative = 1e-12);
        assert_relative_eq!(lo * lo * hi * hi, c, max_relative = 1e-12);
    }

    #[test]
    fn mode_bas_inferieur_au_mode_haut() {
        let (lo, hi) = twodof_natural_frequencies_rad(1.0, 1.0, 1.0, 1.0);
        assert!(lo < hi);
        // m=k=1 → nombre d'or : ω_bas = 1/φ ≈ 0,618, ω_haut = φ ≈ 1,618.
        assert_relative_eq!(lo, 0.618_033_988_75, max_relative = 1e-3);
        assert_relative_eq!(hi, 1.618_033_988_75, max_relative = 1e-3);
    }

    #[test]
    fn discriminant_toujours_positif_modes_reels() {
        // Pour cette topologie, b² − 4c ≥ 0 quelles que soient les valeurs.
        let disc = twodof_frequency_equation_discriminant(4.0, 0.5, 300.0, 120.0);
        assert!(disc >= 0.0);
    }

    #[test]
    fn couplage_est_le_rapport_des_raideurs() {
        assert_relative_eq!(twodof_coupling_factor(50.0, 100.0), 0.5, epsilon = 1e-12);
        // k2 = k1 → couplage unitaire.
        assert_relative_eq!(twodof_coupling_factor(80.0, 80.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "strictement positives")]
    fn masse_nulle_panique() {
        twodof_natural_frequencies_rad(0.0, 1.0, 100.0, 50.0);
    }
}
