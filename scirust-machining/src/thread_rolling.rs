//! Productique — **roulage de filet** (filetage par déformation plastique à
//! froid, sans enlèvement de matière) : diamètre du lopin par conservation de
//! volume, diamètre sur flancs ISO, pénétration par tour et effort de roulage estimé.
//!
//! ```text
//! diamètre du lopin      d_b = d − 0.6495·p              (mm)
//! diamètre sur flancs    d_2 = d − 0.6495·p              (mm)
//! pénétration par tour   a   = f_r / n                   (mm/tr)
//! effort de roulage      F   = K·A·k_f                   (N)
//! ```
//!
//! `d` diamètre extérieur (nominal) du filet (mm), `p` pas du filet (mm), `d_b`
//! diamètre du lopin avant roulage (mm), `d_2` diamètre sur flancs du filet ISO
//! (mm), `f_r` pénétration radiale totale des molettes (mm), `n` nombre de tours
//! de la pièce pendant la pénétration (tr), `a` pénétration radiale par tour de
//! pièce (mm/tr), `F` effort de roulage (N), `K` facteur de forme du procédé (–),
//! `A` aire projetée des flancs en contact (mm²), `k_f` contrainte d'écoulement du
//! matériau (N/mm² = MPa).
//!
//! **Convention** : longueurs en mm, contrainte en N/mm² (MPa), aire en mm² →
//! effort en N. Le facteur `0.6495 = (√3·5)/8·(2/√3)`… en pratique la constante
//! ISO reliant pas et diamètre sur flancs, `d_2 = d − 3√3/(8)·p ≈ d − 0.6495·p`.
//!
//! **Limite honnête** : filetage par **déformation plastique à froid** (sans copeau) ;
//! le diamètre de lopin découle d'une **conservation de volume approchée** propre au
//! filet ISO (le métal refoulé remonte pour former la crête, d'où `d_b ≈ d_2`). La
//! **contrainte d'écoulement** `k_f`, le **facteur de forme** `K` et l'aire projetée
//! `A` sont des données FOURNIES par l'appelant : aucune valeur « par défaut » n'est
//! inventée. L'effort `F` est une **ESTIMATION** ; l'écrouissage réel, la friction
//! molette/pièce et le remplissage progressif du profil font varier l'effort effectif.

/// Constante ISO reliant le pas au retrait sur flancs : `d − d_2 = 0.6495·p`.
///
/// Sans dimension ; multiplie un pas (mm) pour donner un retrait radial diamétral (mm).
pub const THREADROLL_ISO_FLANK_FACTOR: f64 = 0.6495;

/// Diamètre du lopin `d_b = d − 0.6495·p` (mm) avant roulage : par conservation de
/// volume approchée du filet ISO, le lopin part sensiblement au diamètre sur flancs,
/// le métal refoulé remontant pour former la crête du filet.
///
/// Panique si `major_diameter <= 0`, si `pitch <= 0`, ou si le diamètre de lopin
/// résultant est négatif ou nul (pas trop grossier pour le diamètre nominal).
pub fn threadroll_blank_diameter(major_diameter: f64, pitch: f64) -> f64 {
    assert!(
        major_diameter > 0.0,
        "le diamètre extérieur doit être strictement positif"
    );
    assert!(pitch > 0.0, "le pas doit être strictement positif");
    let blank = major_diameter - THREADROLL_ISO_FLANK_FACTOR * pitch;
    assert!(
        blank > 0.0,
        "le pas est trop grand pour ce diamètre extérieur (lopin non physique)"
    );
    blank
}

/// Diamètre sur flancs `d_2 = d − 0.6495·p` (mm) d'un filet ISO : diamètre du
/// cylindre fictif coupant le profil là où pleins et vides sont égaux. Identique
/// en valeur au diamètre du lopin visé par le roulage.
///
/// Panique si `major_diameter <= 0`, si `pitch <= 0`, ou si le diamètre sur flancs
/// résultant est négatif ou nul.
pub fn threadroll_pitch_diameter(major_diameter: f64, pitch: f64) -> f64 {
    assert!(
        major_diameter > 0.0,
        "le diamètre extérieur doit être strictement positif"
    );
    assert!(pitch > 0.0, "le pas doit être strictement positif");
    let pitch_diameter = major_diameter - THREADROLL_ISO_FLANK_FACTOR * pitch;
    assert!(
        pitch_diameter > 0.0,
        "le pas est trop grand pour ce diamètre extérieur (diamètre sur flancs non physique)"
    );
    pitch_diameter
}

/// Pénétration radiale par tour de pièce `a = f_r / n` (mm/tr) : la pénétration
/// radiale totale des molettes `f_r` répartie sur `n` tours de la pièce.
///
/// Panique si `radial_feed < 0` ou si `workpiece_revolutions <= 0`.
pub fn threadroll_penetration_per_revolution(radial_feed: f64, workpiece_revolutions: f64) -> f64 {
    assert!(
        radial_feed >= 0.0,
        "la pénétration radiale doit être positive ou nulle"
    );
    assert!(
        workpiece_revolutions > 0.0,
        "le nombre de tours de la pièce doit être strictement positif"
    );
    radial_feed / workpiece_revolutions
}

/// Effort de roulage estimé `F = K·A·k_f` (N) : produit du facteur de forme du
/// procédé `K`, de l'aire projetée des flancs en contact `A` (mm²) et de la
/// contrainte d'écoulement du matériau `k_f` (N/mm² = MPa).
///
/// Panique si `projected_flank_area < 0`, si `flow_stress < 0`, ou si
/// `form_factor <= 0`.
pub fn threadroll_rolling_force(
    projected_flank_area: f64,
    flow_stress: f64,
    form_factor: f64,
) -> f64 {
    assert!(
        projected_flank_area >= 0.0,
        "l'aire projetée des flancs doit être positive ou nulle"
    );
    assert!(
        flow_stress >= 0.0,
        "la contrainte d'écoulement doit être positive ou nulle"
    );
    assert!(
        form_factor > 0.0,
        "le facteur de forme doit être strictement positif"
    );
    form_factor * projected_flank_area * flow_stress
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn blank_equals_pitch_diameter() {
        // Les deux formules sont identiques (le lopin part au diamètre sur flancs).
        let (major, pitch) = (10.0_f64, 1.5_f64);
        let blank = threadroll_blank_diameter(major, pitch);
        let d2 = threadroll_pitch_diameter(major, pitch);
        assert_relative_eq!(blank, d2, epsilon = 1e-12);
    }

    #[test]
    fn pitch_diameter_iso_m10x1_5() {
        // M10×1,5 : d_2 = 10 − 0,6495·1,5 = 10 − 0,97425 = 9,02575 mm
        // (conforme au diamètre sur flancs normalisé ≈ 9,026 mm).
        let d2 = threadroll_pitch_diameter(10.0, 1.5);
        assert_relative_eq!(d2, 9.025_75, epsilon = 1e-6);
    }

    #[test]
    fn blank_decreases_linearly_with_pitch() {
        // d_b(2p) − d_b(p) = −0,6495·p : la pente vaut exactement le facteur ISO.
        let major = 20.0_f64;
        let d1 = threadroll_blank_diameter(major, 1.0);
        let d2 = threadroll_blank_diameter(major, 2.0);
        assert_relative_eq!(d1 - d2, THREADROLL_ISO_FLANK_FACTOR, epsilon = 1e-12);
    }

    #[test]
    fn penetration_reciprocal_with_revolutions() {
        // a = f_r/n, et réciproquement a·n = f_r (identité de définition).
        let (radial_feed, revolutions) = (1.0_f64, 20.0_f64);
        let a = threadroll_penetration_per_revolution(radial_feed, revolutions);
        assert_relative_eq!(a, 0.05, epsilon = 1e-12);
        assert_relative_eq!(a * revolutions, radial_feed, epsilon = 1e-12);
    }

    #[test]
    fn rolling_force_realistic_case_and_proportional() {
        // K=2, A=50 mm², k_f=500 MPa (=500 N/mm²) → F = 2·50·500 = 50 000 N.
        let f = threadroll_rolling_force(50.0, 500.0, 2.0);
        assert_relative_eq!(f, 50_000.0, epsilon = 1e-9);
        // F ∝ k_f : doubler la contrainte d'écoulement double l'effort.
        let f2 = threadroll_rolling_force(50.0, 1000.0, 2.0);
        assert_relative_eq!(f2 / f, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "pas est trop grand")]
    fn pitch_diameter_rejects_oversize_pitch() {
        // p = 20 mm sur d = 10 mm → d_2 = 10 − 12,99 < 0 : non physique.
        threadroll_pitch_diameter(10.0, 20.0);
    }
}
