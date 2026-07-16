//! **Charpente métallique — résistance en flexion et cisaillement** d'une section
//! transversale (Eurocode 3, EN 1993-1-1) : moment résistant plastique (classes 1
//! et 2), moment résistant élastique (classe 3), aire de cisaillement d'un profilé
//! en I laminé, résistance plastique à l'effort tranchant et taux de travail.
//!
//! ```text
//! moment résistant plastique  Mc,Rd = Wpl · fy / γ_M0        (classe 1 ou 2)
//! moment résistant élastique  Mc,Rd = Wel · fy / γ_M0        (classe 3)
//! aire de cisaillement (I)    Av    = A − 2·b·tf + (tw + 2·r)·tf
//! résistance au tranchant     Vpl,Rd = Av · fy / (√3 · γ_M0)
//! taux de travail             η     = E_d / R_d
//! ```
//!
//! `Wpl` module de flexion plastique de la section (mm³), `Wel` module de flexion
//! élastique (mm³), `fy` limite d'élasticité de l'acier (MPa), `γ_M0` coefficient
//! partiel de sécurité pour la résistance des sections (sans dimension), `Mc,Rd`
//! moment résistant de calcul de la section (N·mm) ; `A` aire brute totale de la
//! section (mm²), `b` largeur de semelle (mm), `tf` épaisseur de semelle (mm),
//! `tw` épaisseur d'âme (mm), `r` rayon de congé âme-semelle (mm), `Av` aire de
//! cisaillement (mm²), `Vpl,Rd` résistance plastique de calcul à l'effort
//! tranchant (N) ; `E_d` sollicitation de calcul (N·mm pour un moment, N pour un
//! effort), `R_d` résistance de calcul correspondante, `η` taux de travail (sans
//! dimension, `η ≤ 1` = section vérifiée).
//!
//! **Convention** : unités **N, mm, MPa** (1 MPa = 1 N/mm²), cohérentes entre
//! elles (Eurocode) ; les moments s'expriment donc en **N·mm** (1 kN·m = 10⁶
//! N·mm), les modules de flexion en **mm³** et les aires en **mm²**. Types `f64`.
//!
//! **Limite honnête** : flexion **sans déversement** — la section est supposée
//! **tenue latéralement** (ou de classe adaptée) de sorte que la résistance vaut
//! celle de la section droite ; le déversement (moment critique `Mcr`, courbe de
//! flambement latéral) n'est **pas** traité ici. L'aire de cisaillement `Av`
//! retenue est celle d'un **profilé en I ou H laminé chargé parallèlement à
//! l'âme**. L'**interaction M-V** n'est à considérer que si `V_Ed > 0,5·Vpl,Rd`
//! (réduction de la limite d'élasticité de l'âme) : cette réduction est **à la
//! charge de l'appelant** et n'est pas appliquée ici. Les **résistances
//! caractéristiques** (`fy`) et **tous les coefficients partiels de sécurité**
//! (`γ_M0`) sont **fournis par l'appelant** d'après l'**Eurocode 3 (EN 1993-1-1)**
//! et son **Annexe Nationale** — aucune valeur « par défaut » n'est inventée.

/// Moment résistant plastique de calcul `Mc,Rd = Wpl · fy / γ_M0` (N·mm), pour
/// une section de **classe 1 ou 2** (Eurocode 3, EN 1993-1-1 §6.2.5).
///
/// `plastic_modulus` = `Wpl` (mm³), `fy` limite d'élasticité (MPa), `gamma_m0`
/// = `γ_M0` (sans dimension) fourni par l'Eurocode 3 et son Annexe Nationale ;
/// renvoie un moment (N·mm).
///
/// Panique si `plastic_modulus < 0`, si `fy < 0` ou si `gamma_m0 <= 0` (division
/// par zéro).
pub fn steelbend_plastic_moment_resistance(plastic_modulus: f64, fy: f64, gamma_m0: f64) -> f64 {
    assert!(
        plastic_modulus >= 0.0,
        "le module plastique Wpl doit être ≥ 0"
    );
    assert!(fy >= 0.0, "la limite d'élasticité fy doit être ≥ 0");
    assert!(
        gamma_m0 > 0.0,
        "le coefficient partiel γ_M0 doit être strictement positif"
    );
    plastic_modulus * fy / gamma_m0
}

/// Moment résistant élastique de calcul `Mc,Rd = Wel · fy / γ_M0` (N·mm), pour
/// une section de **classe 3** (Eurocode 3, EN 1993-1-1 §6.2.5).
///
/// `elastic_modulus` = `Wel` (mm³), `fy` limite d'élasticité (MPa), `gamma_m0`
/// = `γ_M0` (sans dimension) fourni par l'Eurocode 3 et son Annexe Nationale ;
/// renvoie un moment (N·mm).
///
/// Panique si `elastic_modulus < 0`, si `fy < 0` ou si `gamma_m0 <= 0` (division
/// par zéro).
pub fn steelbend_elastic_moment_resistance(elastic_modulus: f64, fy: f64, gamma_m0: f64) -> f64 {
    assert!(
        elastic_modulus >= 0.0,
        "le module élastique Wel doit être ≥ 0"
    );
    assert!(fy >= 0.0, "la limite d'élasticité fy doit être ≥ 0");
    assert!(
        gamma_m0 > 0.0,
        "le coefficient partiel γ_M0 doit être strictement positif"
    );
    elastic_modulus * fy / gamma_m0
}

/// Aire de cisaillement d'un **profilé en I ou H laminé chargé parallèlement à
/// l'âme** `Av = A − 2·b·tf + (tw + 2·r)·tf` (mm²) (Eurocode 3, EN 1993-1-1
/// §6.2.6(3)).
///
/// `total_area` = `A` aire brute totale (mm²), `flange_width` = `b` (mm),
/// `flange_thickness` = `tf` (mm), `web_thickness` = `tw` (mm), `root_radius` =
/// `r` rayon de congé (mm) ; renvoie l'aire de cisaillement `Av` (mm²).
///
/// Panique si `total_area <= 0`, ou si l'une des dimensions `flange_width`,
/// `flange_thickness`, `web_thickness`, `root_radius` est `< 0`.
pub fn steelbend_shear_area_rolled_i(
    total_area: f64,
    flange_width: f64,
    flange_thickness: f64,
    web_thickness: f64,
    root_radius: f64,
) -> f64 {
    assert!(
        total_area > 0.0,
        "l'aire totale A doit être strictement positive"
    );
    assert!(flange_width >= 0.0, "la largeur de semelle b doit être ≥ 0");
    assert!(
        flange_thickness >= 0.0,
        "l'épaisseur de semelle tf doit être ≥ 0"
    );
    assert!(web_thickness >= 0.0, "l'épaisseur d'âme tw doit être ≥ 0");
    assert!(root_radius >= 0.0, "le rayon de congé r doit être ≥ 0");
    total_area - 2.0 * flange_width * flange_thickness
        + (web_thickness + 2.0 * root_radius) * flange_thickness
}

/// Résistance plastique de calcul à l'effort tranchant
/// `Vpl,Rd = Av · fy / (√3 · γ_M0)` (N) (Eurocode 3, EN 1993-1-1 §6.2.6(2)).
///
/// `shear_area` = `Av` (mm²), `fy` limite d'élasticité (MPa), `gamma_m0` =
/// `γ_M0` (sans dimension) fourni par l'Eurocode 3 et son Annexe Nationale ;
/// renvoie un effort (N).
///
/// Panique si `shear_area < 0`, si `fy < 0` ou si `gamma_m0 <= 0` (division par
/// zéro).
pub fn steelbend_plastic_shear_resistance(shear_area: f64, fy: f64, gamma_m0: f64) -> f64 {
    assert!(shear_area >= 0.0, "l'aire de cisaillement Av doit être ≥ 0");
    assert!(fy >= 0.0, "la limite d'élasticité fy doit être ≥ 0");
    assert!(
        gamma_m0 > 0.0,
        "le coefficient partiel γ_M0 doit être strictement positif"
    );
    shear_area * fy / (3.0_f64.sqrt() * gamma_m0)
}

/// Taux de travail `η = E_d / R_d` (sans dimension), rapport de la sollicitation
/// de calcul à la résistance de calcul (`η ≤ 1` = section vérifiée).
///
/// `design_value` = `E_d` sollicitation (N·mm pour un moment, N pour un effort),
/// `resistance` = `R_d` résistance de calcul dans la même unité ; renvoie un taux
/// adimensionnel.
///
/// Panique si `design_value < 0` ou si `resistance <= 0` (division par zéro).
pub fn steelbend_utilisation(design_value: f64, resistance: f64) -> f64 {
    assert!(
        design_value >= 0.0,
        "la sollicitation de calcul E_d doit être ≥ 0"
    );
    assert!(
        resistance > 0.0,
        "la résistance de calcul R_d doit être strictement positive"
    );
    design_value / resistance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn plastic_moment_reciprocity() {
        // Réciprocité : Mc,Rd · γ_M0 / fy restitue le module plastique Wpl.
        let wpl = 628_400.0_f64;
        let fy = 355.0_f64;
        let gamma_m0 = 1.0_f64;
        let m = steelbend_plastic_moment_resistance(wpl, fy, gamma_m0);
        assert_relative_eq!(m * gamma_m0 / fy, wpl, epsilon = 1e-6);
    }

    #[test]
    fn shape_factor_equals_modulus_ratio() {
        // À fy et γ_M0 fixés, Mpl/Mel = Wpl/Wel (facteur de forme de la section).
        let (wpl, wel, fy, gamma_m0) = (628_400.0_f64, 557_100.0, 235.0, 1.0);
        let mpl = steelbend_plastic_moment_resistance(wpl, fy, gamma_m0);
        let mel = steelbend_elastic_moment_resistance(wel, fy, gamma_m0);
        assert_relative_eq!(mpl / mel, wpl / wel, epsilon = 1e-12);
    }

    #[test]
    fn shear_resistance_reciprocity() {
        // Réciprocité : Vpl,Rd · √3 · γ_M0 restitue le produit Av · fy.
        let (av, fy, gamma_m0) = (2_566.97_f64, 235.0, 1.0);
        let v = steelbend_plastic_shear_resistance(av, fy, gamma_m0);
        assert_relative_eq!(v * 3.0_f64.sqrt() * gamma_m0, av * fy, epsilon = 1e-3);
    }

    #[test]
    fn shear_resistance_proportional_to_area() {
        // Vpl,Rd ∝ Av : doubler l'aire de cisaillement double la résistance.
        let single = steelbend_plastic_shear_resistance(1_200.0, 355.0, 1.0);
        let double = steelbend_plastic_shear_resistance(2_400.0, 355.0, 1.0);
        assert_relative_eq!(double, 2.0 * single, epsilon = 1e-9);
    }

    #[test]
    fn utilisation_at_resistance_and_half() {
        // η = 1 quand E_d = R_d, et η = 0,5 à la moitié de la résistance.
        assert_relative_eq!(
            steelbend_utilisation(150.0e6, 150.0e6),
            1.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(steelbend_utilisation(75.0e6, 150.0e6), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn realistic_ipe300_s235_case() {
        // Profilé IPE 300, acier S235, γ_M0 = 1,0 (EN 1993-1-1) :
        //   A = 5380 mm², b = 150 mm, tf = 10,7 mm, tw = 7,1 mm, r = 15 mm
        //   Wpl = 628 400 mm³, Wel = 557 100 mm³, fy = 235 MPa
        //   Av  = 5380 − 2·150·10,7 + (7,1 + 2·15)·10,7
        //       = 5380 − 3210 + 37,1·10,7
        //       = 5380 − 3210 + 396,97          = 2566,97 mm²
        //   Mpl,Rd = 628 400 · 235 / 1,0        = 147 674 000 N·mm  (147,674 kN·m)
        //   Mel,Rd = 557 100 · 235 / 1,0        = 130 918 500 N·mm  (130,919 kN·m)
        //   Vpl,Rd = 2566,97 · 235 / (√3 · 1,0) = 348 279,59 N       (348,28 kN)
        let av = steelbend_shear_area_rolled_i(5_380.0, 150.0, 10.7, 7.1, 15.0);
        assert_relative_eq!(av, 2_566.97, epsilon = 1e-6);
        let mpl = steelbend_plastic_moment_resistance(628_400.0, 235.0, 1.0);
        assert_relative_eq!(mpl, 147_674_000.0, epsilon = 1.0);
        let mel = steelbend_elastic_moment_resistance(557_100.0, 235.0, 1.0);
        assert_relative_eq!(mel, 130_918_500.0, epsilon = 1.0);
        let vpl = steelbend_plastic_shear_resistance(av, 235.0, 1.0);
        assert_relative_eq!(vpl, 348_279.592_817_898, epsilon = 1e-3);
        // Un effort tranchant de calcul égal à la moitié de Vpl,Rd donne η = 0,5,
        // seuil au-delà duquel l'interaction M-V est à considérer (à la charge de
        // l'appelant).
        assert_relative_eq!(steelbend_utilisation(vpl / 2.0, vpl), 0.5, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le coefficient partiel γ_M0 doit être strictement positif")]
    fn zero_gamma_m0_panics() {
        steelbend_plastic_moment_resistance(628_400.0, 235.0, 0.0);
    }
}
