//! Sévérité vibratoire selon ISO 10816 — vitesse vibratoire efficace (RMS) et
//! classement en zones d'évaluation A/B/C/D d'une machine tournante.
//!
//! ```text
//! valeur efficace d'un signal sinusoïdal   v_rms = v_pic / √2
//! valeur crête depuis l'efficace           v_pic = v_rms · √2
//! RMS global de N échantillons/composantes  v_rms = √( (1/N) · Σ vᵢ² )
//! classement (limites fournies) :
//!   v_rms < L_ab            → zone A
//!   L_ab ≤ v_rms < L_bc     → zone B
//!   L_bc ≤ v_rms < L_cd     → zone C
//!   v_rms ≥ L_cd            → zone D
//! ```
//!
//! `v_rms` vitesse vibratoire efficace (mm/s RMS), `v_pic` amplitude crête de la
//! vitesse (mm/s), `vᵢ` valeurs instantanées ou composantes de vitesse (mm/s),
//! `N` nombre d'échantillons, `L_ab`, `L_bc`, `L_cd` limites de zones A/B, B/C,
//! C/D (mm/s RMS). Les zones se lisent : A machine neuve, B exploitation sans
//! restriction, C surveillance rapprochée, D risque de dommage.
//!
//! **Convention** : toutes les vitesses en mm/s ; les limites de zones sont en
//! mm/s RMS et croissantes (`0 < L_ab < L_bc < L_cd`). **Limite honnête** : la
//! relation crête/efficace `v_pic = v_rms · √2` n'est exacte que pour un signal
//! sinusoïdal pur. Les limites de zones A/B/C/D dépendent de la CLASSE de machine
//! (puissance, type de support rigide ou souple, montage) et la norme ne fournit
//! PAS une valeur unique : elles sont FOURNIES par l'appelant selon la partie
//! d'ISO 10816 et la classe applicables — aucune valeur « par défaut » n'est
//! inventée ici.

use core::f64::consts::{FRAC_1_SQRT_2, SQRT_2};

/// Zone d'évaluation de sévérité vibratoire selon ISO 10816.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VibrationZone {
    /// Zone A : machine typiquement neuve, faible sévérité.
    A,
    /// Zone B : exploitation acceptable sans restriction de durée.
    B,
    /// Zone C : niveau tolérable pour une durée limitée, surveillance requise.
    C,
    /// Zone D : sévérité pouvant causer des dommages, action nécessaire.
    D,
}

/// Valeur efficace d'un signal sinusoïdal `v_rms = v_pic / √2` (mm/s RMS).
///
/// Convertit une amplitude crête de vitesse en valeur efficace, exacte pour un
/// signal sinusoïdal pur.
///
/// Panique si `peak_value < 0`.
pub fn iso10816_rms_from_peak(peak_value: f64) -> f64 {
    assert!(
        peak_value >= 0.0,
        "l'amplitude crête doit être positive ou nulle"
    );
    peak_value * FRAC_1_SQRT_2
}

/// Valeur crête depuis l'efficace `v_pic = v_rms · √2` (mm/s crête).
///
/// Inverse de [`iso10816_rms_from_peak`], exacte pour un signal sinusoïdal pur.
///
/// Panique si `rms_value < 0`.
pub fn iso10816_peak_from_rms(rms_value: f64) -> f64 {
    assert!(
        rms_value >= 0.0,
        "la valeur efficace doit être positive ou nulle"
    );
    rms_value * SQRT_2
}

/// RMS global `v_rms = √( (1/N) · Σ vᵢ² )` d'un jeu de vitesses (mm/s RMS).
///
/// Racine de la moyenne des carrés des `velocities` (échantillons temporels ou
/// composantes de vitesse, en mm/s).
///
/// Panique si `velocities` est vide.
pub fn iso10816_velocity_rms_from_components(velocities: &[f64]) -> f64 {
    assert!(
        !velocities.is_empty(),
        "le jeu de vitesses ne doit pas être vide"
    );
    let sum_squares: f64 = velocities.iter().map(|v| v * v).sum();
    (sum_squares / velocities.len() as f64).sqrt()
}

/// Classe une vitesse efficace en zone A/B/C/D selon des limites fournies.
///
/// Renvoie la [`VibrationZone`] telle que `v_rms < L_ab → A`,
/// `L_ab ≤ v_rms < L_bc → B`, `L_bc ≤ v_rms < L_cd → C`, `v_rms ≥ L_cd → D`.
/// Les limites `zone_ab_limit`, `zone_bc_limit`, `zone_cd_limit` (mm/s RMS)
/// dépendent de la classe de machine et sont fournies par l'appelant.
///
/// Panique si `velocity_rms_mm_s < 0` ou si les limites ne sont pas strictement
/// croissantes et positives (`0 < L_ab < L_bc < L_cd`).
pub fn iso10816_zone(
    velocity_rms_mm_s: f64,
    zone_ab_limit: f64,
    zone_bc_limit: f64,
    zone_cd_limit: f64,
) -> VibrationZone {
    assert!(
        velocity_rms_mm_s >= 0.0,
        "la vitesse efficace doit être positive ou nulle"
    );
    assert!(
        0.0 < zone_ab_limit && zone_ab_limit < zone_bc_limit && zone_bc_limit < zone_cd_limit,
        "les limites de zones doivent vérifier 0 < L_ab < L_bc < L_cd"
    );
    if velocity_rms_mm_s < zone_ab_limit
    {
        VibrationZone::A
    }
    else if velocity_rms_mm_s < zone_bc_limit
    {
        VibrationZone::B
    }
    else if velocity_rms_mm_s < zone_cd_limit
    {
        VibrationZone::C
    }
    else
    {
        VibrationZone::D
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rms_from_peak_realistic_case() {
        // Amplitude crête de 10 mm/s → efficace 10/√2 = 7,071067811... mm/s.
        assert_relative_eq!(
            iso10816_rms_from_peak(10.0),
            7.071_067_811_865_476,
            epsilon = 1e-12
        );
    }

    #[test]
    fn peak_rms_are_reciprocal() {
        // Réciprocité : v_pic → v_rms → v_pic redonne la valeur de départ.
        let peak = 3.7;
        let rms = iso10816_rms_from_peak(peak);
        assert_relative_eq!(iso10816_peak_from_rms(rms), peak, epsilon = 1e-12);
    }

    #[test]
    fn global_rms_of_constant_set_equals_that_constant() {
        // Cas limite : toutes les composantes égales à c → RMS global = |c|.
        let c = 2.8;
        let rms = iso10816_velocity_rms_from_components(&[c, c, c, c]);
        assert_relative_eq!(rms, c, epsilon = 1e-12);
    }

    #[test]
    fn global_rms_known_case() {
        // √( (3² + 4²)/2 ) = √(25/2) = √12,5 = 3,535533905... mm/s.
        let rms = iso10816_velocity_rms_from_components(&[3.0, 4.0]);
        assert_relative_eq!(rms, 3.535_533_905_932_738, epsilon = 1e-12);
    }

    #[test]
    fn global_rms_is_positively_homogeneous() {
        // Homogénéité : multiplier toutes les vitesses par k multiplie le RMS par k.
        let base = [1.0, -2.0, 3.5, 0.5];
        let k = 4.0;
        let scaled: Vec<f64> = base.iter().map(|v| k * v).collect();
        let r_base = iso10816_velocity_rms_from_components(&base);
        let r_scaled = iso10816_velocity_rms_from_components(&scaled);
        assert_relative_eq!(r_scaled, k * r_base, epsilon = 1e-12);
    }

    #[test]
    fn zone_classification_covers_all_bands() {
        // Limites fournies (exemple classe machine) : L_ab=1,4 ; L_bc=2,8 ; L_cd=4,5.
        let (l_ab, l_bc, l_cd) = (1.4, 2.8, 4.5);
        assert_eq!(iso10816_zone(1.0, l_ab, l_bc, l_cd), VibrationZone::A);
        assert_eq!(iso10816_zone(2.0, l_ab, l_bc, l_cd), VibrationZone::B);
        assert_eq!(iso10816_zone(3.5, l_ab, l_bc, l_cd), VibrationZone::C);
        assert_eq!(iso10816_zone(6.0, l_ab, l_bc, l_cd), VibrationZone::D);
        // Bornes inclusives à gauche : v = L_ab tombe en zone B, pas A.
        assert_eq!(iso10816_zone(l_ab, l_ab, l_bc, l_cd), VibrationZone::B);
    }

    #[test]
    #[should_panic(expected = "les limites de zones doivent vérifier 0 < L_ab < L_bc < L_cd")]
    fn non_increasing_limits_panics() {
        iso10816_zone(2.0, 2.8, 1.4, 4.5);
    }
}
