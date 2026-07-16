//! **Charpente métallique — assemblage boulonné** (Eurocode 3, EN 1993-1-8) :
//! résistance au cisaillement d'un boulon par plan de cisaillement, résistance à
//! la pression diamétrale (portance) d'une plaque, résistance à la traction d'un
//! boulon, et résistance d'un groupe de boulons à répartition égale.
//!
//! ```text
//! cisaillement (par plan)   Fv,Rd = αv · fub · As / γ_M2
//! pression diamétrale       Fb,Rd = k1 · αb · fu · d · t / γ_M2
//! traction (k2 = 0,9)       Ft,Rd = k2 · fub · As / γ_M2
//! groupe (répartition égale) Fgr,Rd = Fbolt,Rd · n
//! ```
//!
//! `αv` coefficient de cisaillement (0,6 ou 0,5 selon la classe et la position du
//! plan de cisaillement, sans dimension), `fub` résistance ultime à la traction du
//! boulon (MPa), `As` aire résistante (de la partie filetée) du boulon (mm²),
//! `γ_M2` coefficient partiel de sécurité des assemblages (sans dimension) ;
//! `k1` coefficient de bord/pince (sans dimension), `αb` coefficient de pince/pas
//! (sans dimension), `fu` résistance ultime de la plaque assemblée (MPa), `d`
//! diamètre nominal du boulon (mm), `t` épaisseur de la plaque en portance (mm) ;
//! `k2` coefficient de traction (`0,9` pour un boulon courant, sans dimension) ;
//! `Fbolt,Rd` résistance d'un boulon individuel (N), `n` nombre de boulons du
//! groupe (sans dimension) ; `Fv,Rd`, `Fb,Rd`, `Ft,Rd`, `Fgr,Rd` résistances de
//! calcul (N).
//!
//! **Convention** : unités **N, mm, MPa** (1 MPa = 1 N/mm²), cohérentes entre
//! elles (Eurocode) ; les aires s'expriment en **mm²**, les résistances en **N**
//! (1 kN = 10³ N). Types `f64`.
//!
//! **Limite honnête** : ce module traite les boulons en **cisaillement**, en
//! **pression diamétrale** et en **traction** de façon **indépendante**. Tous les
//! **coefficients** (`αv`, `k1`, `αb`, `k2`) et toutes les **résistances
//! caractéristiques** (`fub`, `fu`) sont **fournis par l'appelant** d'après
//! l'**Eurocode 3 (EN 1993-1-8, tableau 3.4)** et son **Annexe Nationale** ; le
//! **coefficient partiel** `γ_M2` est **fourni** de la même manière — aucune
//! valeur « par défaut » n'est inventée. Le **groupe** suppose une **répartition
//! égale** de l'effort entre boulons (**pas d'excentricité** : un chargement
//! excentré introduit un moment à combiner par l'appelant). Ne sont **pas**
//! vérifiés ici : le **poinçonnement** (`Bp,Rd`), l'**interaction
//! cisaillement-traction** (`Fv,Ed/Fv,Rd + Ft,Ed/(1,4·Ft,Rd) ≤ 1`), ni la
//! **résistance des sections nettes** des pièces assemblées.

/// Résistance de calcul au cisaillement d'un boulon **par plan de cisaillement**
/// `Fv,Rd = αv · fub · As / γ_M2` (N) (Eurocode 3, EN 1993-1-8, tableau 3.4).
///
/// `alpha_v` = `αv` coefficient de cisaillement (0,6 ou 0,5 selon la classe et la
/// position du plan, sans dimension), `ultimate_strength_bolt` = `fub` résistance
/// ultime du boulon (MPa), `stress_area` = `As` aire résistante (mm²), `gamma_m2`
/// = `γ_M2` (sans dimension) fourni par l'Eurocode 3 et son Annexe Nationale ;
/// renvoie un effort (N) **par plan de cisaillement**.
///
/// Panique si `alpha_v < 0`, si `ultimate_strength_bolt < 0`, si `stress_area < 0`
/// ou si `gamma_m2 <= 0` (division par zéro).
pub fn steelbolt_shear_resistance(
    alpha_v: f64,
    ultimate_strength_bolt: f64,
    stress_area: f64,
    gamma_m2: f64,
) -> f64 {
    assert!(alpha_v >= 0.0, "le coefficient αv doit être ≥ 0");
    assert!(
        ultimate_strength_bolt >= 0.0,
        "la résistance ultime du boulon fub doit être ≥ 0"
    );
    assert!(stress_area >= 0.0, "l'aire résistante As doit être ≥ 0");
    assert!(
        gamma_m2 > 0.0,
        "le coefficient partiel γ_M2 doit être strictement positif"
    );
    alpha_v * ultimate_strength_bolt * stress_area / gamma_m2
}

/// Résistance de calcul à la **pression diamétrale** (portance) d'une plaque
/// `Fb,Rd = k1 · αb · fu · d · t / γ_M2` (N) (Eurocode 3, EN 1993-1-8,
/// tableau 3.4).
///
/// `k1` = `k1` coefficient de bord/pince (sans dimension), `alpha_b` = `αb`
/// coefficient de pince/pas (sans dimension), `ultimate_strength_plate` = `fu`
/// résistance ultime de la plaque (MPa), `bolt_diameter` = `d` diamètre nominal du
/// boulon (mm), `plate_thickness` = `t` épaisseur de la plaque (mm), `gamma_m2` =
/// `γ_M2` (sans dimension) fourni par l'Eurocode 3 et son Annexe Nationale ;
/// renvoie un effort (N).
///
/// Panique si `k1 < 0`, si `alpha_b < 0`, si `ultimate_strength_plate < 0`, si
/// `bolt_diameter < 0`, si `plate_thickness < 0` ou si `gamma_m2 <= 0` (division
/// par zéro).
pub fn steelbolt_bearing_resistance(
    k1: f64,
    alpha_b: f64,
    ultimate_strength_plate: f64,
    bolt_diameter: f64,
    plate_thickness: f64,
    gamma_m2: f64,
) -> f64 {
    assert!(k1 >= 0.0, "le coefficient k1 doit être ≥ 0");
    assert!(alpha_b >= 0.0, "le coefficient αb doit être ≥ 0");
    assert!(
        ultimate_strength_plate >= 0.0,
        "la résistance ultime de la plaque fu doit être ≥ 0"
    );
    assert!(
        bolt_diameter >= 0.0,
        "le diamètre du boulon d doit être ≥ 0"
    );
    assert!(
        plate_thickness >= 0.0,
        "l'épaisseur de la plaque t doit être ≥ 0"
    );
    assert!(
        gamma_m2 > 0.0,
        "le coefficient partiel γ_M2 doit être strictement positif"
    );
    k1 * alpha_b * ultimate_strength_plate * bolt_diameter * plate_thickness / gamma_m2
}

/// Résistance de calcul à la **traction** d'un boulon
/// `Ft,Rd = k2 · fub · As / γ_M2` (N) avec `k2 = 0,9` pour un boulon courant
/// (Eurocode 3, EN 1993-1-8, tableau 3.4).
///
/// `k2` = `k2` coefficient de traction (`0,9` en règle générale, sans dimension),
/// `ultimate_strength_bolt` = `fub` résistance ultime du boulon (MPa),
/// `stress_area` = `As` aire résistante (mm²), `gamma_m2` = `γ_M2` (sans
/// dimension) fourni par l'Eurocode 3 et son Annexe Nationale ; renvoie un effort
/// (N).
///
/// Panique si `k2 < 0`, si `ultimate_strength_bolt < 0`, si `stress_area < 0` ou
/// si `gamma_m2 <= 0` (division par zéro).
pub fn steelbolt_tension_resistance(
    k2: f64,
    ultimate_strength_bolt: f64,
    stress_area: f64,
    gamma_m2: f64,
) -> f64 {
    assert!(k2 >= 0.0, "le coefficient k2 doit être ≥ 0");
    assert!(
        ultimate_strength_bolt >= 0.0,
        "la résistance ultime du boulon fub doit être ≥ 0"
    );
    assert!(stress_area >= 0.0, "l'aire résistante As doit être ≥ 0");
    assert!(
        gamma_m2 > 0.0,
        "le coefficient partiel γ_M2 doit être strictement positif"
    );
    k2 * ultimate_strength_bolt * stress_area / gamma_m2
}

/// Résistance de calcul d'un **groupe de boulons à répartition égale**
/// `Fgr,Rd = Fbolt,Rd · n` (N) : somme des résistances individuelles lorsque
/// l'effort se répartit également entre boulons (**sans excentricité**).
///
/// `single_bolt_resistance` = `Fbolt,Rd` résistance d'un boulon individuel (N, par
/// exemple le minimum de `Fv,Rd` et `Fb,Rd`), `bolt_count` = `n` nombre de boulons
/// (sans dimension) ; renvoie la résistance du groupe (N).
///
/// Panique si `single_bolt_resistance < 0` ou si `bolt_count < 0`.
pub fn steelbolt_group_shear_resistance(single_bolt_resistance: f64, bolt_count: f64) -> f64 {
    assert!(
        single_bolt_resistance >= 0.0,
        "la résistance d'un boulon Fbolt,Rd doit être ≥ 0"
    );
    assert!(bolt_count >= 0.0, "le nombre de boulons n doit être ≥ 0");
    single_bolt_resistance * bolt_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn shear_resistance_reciprocity() {
        // Réciprocité : Fv,Rd · γ_M2 / (αv · fub) restitue l'aire résistante As.
        let (alpha_v, fub, as_bolt, gamma_m2) = (0.6_f64, 800.0, 245.0, 1.25);
        let fv = steelbolt_shear_resistance(alpha_v, fub, as_bolt, gamma_m2);
        assert_relative_eq!(fv * gamma_m2 / (alpha_v * fub), as_bolt, epsilon = 1e-9);
    }

    #[test]
    fn bearing_resistance_proportional_to_thickness() {
        // Fb,Rd ∝ t : doubler l'épaisseur de la plaque double la portance.
        let single = steelbolt_bearing_resistance(2.5, 0.7, 360.0, 20.0, 10.0, 1.25);
        let double = steelbolt_bearing_resistance(2.5, 0.7, 360.0, 20.0, 20.0, 1.25);
        assert_relative_eq!(double, 2.0 * single, epsilon = 1e-6);
    }

    #[test]
    fn tension_matches_shear_ratio() {
        // À fub, As et γ_M2 fixés, Ft,Rd/Fv,Rd = k2/αv (mêmes facteurs communs).
        let (fub, as_bolt, gamma_m2) = (800.0_f64, 245.0, 1.25);
        let (k2, alpha_v) = (0.9_f64, 0.6);
        let ft = steelbolt_tension_resistance(k2, fub, as_bolt, gamma_m2);
        let fv = steelbolt_shear_resistance(alpha_v, fub, as_bolt, gamma_m2);
        assert_relative_eq!(ft / fv, k2 / alpha_v, epsilon = 1e-12);
    }

    #[test]
    fn group_resistance_scales_with_count() {
        // Fgr,Rd = Fbolt,Rd · n : proportionnalité au nombre de boulons.
        let single = 94_080.0_f64;
        let g2 = steelbolt_group_shear_resistance(single, 2.0);
        let g6 = steelbolt_group_shear_resistance(single, 6.0);
        assert_relative_eq!(g2, 2.0 * single, epsilon = 1e-9);
        assert_relative_eq!(g6, 3.0 * g2, epsilon = 1e-9);
    }

    #[test]
    fn realistic_m20_class88_s235_case() {
        // Boulon M20 classe 8.8, plaque S235, γ_M2 = 1,25 (EN 1993-1-8) :
        //   fub = 800 MPa, As = 245 mm², αv = 0,6 (plan dans la partie filetée)
        //   fu(plaque) = 360 MPa, d = 20 mm, t = 10 mm, k1 = 2,5, αb = 0,7
        //   k2 = 0,9, groupe de n = 4 boulons.
        //   Fv,Rd = 0,6 · 800 · 245 / 1,25 = 117 600 / 1,25       = 94 080 N
        //   Fb,Rd = 2,5 · 0,7 · 360 · 20 · 10 / 1,25 = 126 000 / 1,25 = 100 800 N
        //   Ft,Rd = 0,9 · 800 · 245 / 1,25 = 176 400 / 1,25       = 141 120 N
        //   Fgr,Rd = 94 080 · 4                                   = 376 320 N
        let fv = steelbolt_shear_resistance(0.6, 800.0, 245.0, 1.25);
        assert_relative_eq!(fv, 94_080.0, epsilon = 1e-3);
        let fb = steelbolt_bearing_resistance(2.5, 0.7, 360.0, 20.0, 10.0, 1.25);
        assert_relative_eq!(fb, 100_800.0, epsilon = 1e-3);
        let ft = steelbolt_tension_resistance(0.9, 800.0, 245.0, 1.25);
        assert_relative_eq!(ft, 141_120.0, epsilon = 1e-3);
        // La résistance d'un boulon en cisaillement (94 080 N) est inférieure à sa
        // portance (100 800 N) : le cisaillement gouverne, base du groupe.
        let governing = fv.min(fb);
        assert_relative_eq!(governing, 94_080.0, epsilon = 1e-3);
        let group = steelbolt_group_shear_resistance(governing, 4.0);
        assert_relative_eq!(group, 376_320.0, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "le coefficient partiel γ_M2 doit être strictement positif")]
    fn zero_gamma_m2_panics() {
        steelbolt_shear_resistance(0.6, 800.0, 245.0, 0.0);
    }
}
