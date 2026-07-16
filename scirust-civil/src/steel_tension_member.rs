//! **Charpente métallique — barre tendue axialement** (Eurocode 3,
//! EN 1993-1-1 §6.2.3) : résistance plastique de la section brute `Npl,Rd`,
//! résistance ultime de la section nette `Nu,Rd` (au droit des trous),
//! calcul de l'aire nette `Anet` et résistance de calcul en traction prise
//! comme minimum des deux.
//!
//! ```text
//! aire nette (trous alignés) Anet  = A − n·d0·t
//! section brute (plastique)  Npl,Rd = A·fy / γM0
//! section nette (ultime)     Nu,Rd  = 0,9·Anet·fu / γM2
//! résistance de calcul       Nt,Rd  = min(Npl,Rd ; Nu,Rd)
//! ```
//!
//! `A` = `gross_area` aire brute de la section (mm²), `n` = `hole_count` nombre
//! de trous dans la coupe considérée (sans dimension), `d0` = `hole_diameter`
//! diamètre des trous (mm), `t` = `thickness` épaisseur de la pièce percée
//! (mm), `Anet` = `net_area` aire nette (mm²), `fy` = `yield_strength` limite
//! d'élasticité de l'acier (MPa), `fu` = `ultimate_strength` résistance ultime
//! à la traction de l'acier (MPa), `γM0` = `gamma_m0` et `γM2` = `gamma_m2`
//! coefficients partiels de sécurité (sans dimension), `Npl,Rd` = `gross_resistance`,
//! `Nu,Rd` = `net_resistance` et `Nt,Rd` résistances de calcul (N).
//!
//! **Convention** : unités **N, mm, MPa** (avec `1 MPa = 1 N/mm²`, donc les
//! produits `A·fy` et `Anet·fu` sont en N) ; `n`, `γM0` et `γM2` sont sans
//! dimension. Types `f64`.
//!
//! **Limite honnête** : ce module traite la seule **traction axiale** d'une
//! barre. Les résistances caractéristiques `fy` et `fu`, ainsi que les
//! **coefficients partiels** `γM0` et `γM2`, sont **fournis par l'appelant**
//! d'après l'**Eurocode 3** et son **Annexe Nationale** ; aucune valeur « par
//! défaut » n'est inventée. L'aire nette déduit ici des **trous alignés** dans
//! la coupe : pour des **trous en quinconce**, l'appelant doit corriger l'aire
//! déduite par le terme `s²/(4·p)` (EN 1993-1-1 §6.2.2.2). Ne sont **pas**
//! couverts : la **traction avec moment**, le **cisaillement de bloc**
//! (`Veff,Rd`), ni les **cornières attachées par une seule aile** (`Nu,Rd`
//! réduit selon EN 1993-1-8 §3.10.3).

/// Aire nette au droit des trous `Anet = A − n·d0·t` (mm²) pour des trous
/// **alignés** dans la coupe (EN 1993-1-1 §6.2.2.2), avec `gross_area` = `A`
/// l'aire brute (mm²), `hole_count` = `n` le nombre de trous (sans dimension),
/// `hole_diameter` = `d0` le diamètre des trous (mm) et `thickness` = `t`
/// l'épaisseur de la pièce percée (mm).
///
/// Panique si `gross_area <= 0`, si `hole_count < 0`, si `hole_diameter < 0`,
/// si `thickness < 0`, ou si l'aire déduite dépasse l'aire brute (`Anet < 0`,
/// géométrie incohérente).
pub fn steelten_net_area(
    gross_area: f64,
    hole_count: f64,
    hole_diameter: f64,
    thickness: f64,
) -> f64 {
    assert!(
        gross_area > 0.0,
        "l'aire brute A doit être strictement positive (mm²)"
    );
    assert!(hole_count >= 0.0, "le nombre de trous n doit être ≥ 0");
    assert!(
        hole_diameter >= 0.0,
        "le diamètre des trous d0 doit être ≥ 0 (mm)"
    );
    assert!(thickness >= 0.0, "l'épaisseur t doit être ≥ 0 (mm)");
    let net_area = gross_area - hole_count * hole_diameter * thickness;
    assert!(
        net_area >= 0.0,
        "l'aire déduite des trous dépasse l'aire brute (Anet < 0) : géométrie incohérente"
    );
    net_area
}

/// Résistance plastique de la **section brute** `Npl,Rd = A·fy / γM0` (N)
/// (EN 1993-1-1 §6.2.3), avec `gross_area` = `A` l'aire brute (mm²),
/// `yield_strength` = `fy` la limite d'élasticité (MPa) et `gamma_m0` = `γM0`
/// le coefficient partiel (sans dimension) fourni par l'Eurocode et son Annexe
/// Nationale.
///
/// Panique si `gross_area <= 0`, si `yield_strength < 0` ou si `gamma_m0 <= 0`
/// (division par zéro).
pub fn steelten_gross_section_resistance(
    gross_area: f64,
    yield_strength: f64,
    gamma_m0: f64,
) -> f64 {
    assert!(
        gross_area > 0.0,
        "l'aire brute A doit être strictement positive (mm²)"
    );
    assert!(
        yield_strength >= 0.0,
        "la limite d'élasticité fy doit être ≥ 0 (MPa)"
    );
    assert!(
        gamma_m0 > 0.0,
        "le coefficient partiel γM0 doit être strictement positif"
    );
    gross_area * yield_strength / gamma_m0
}

/// Résistance ultime de la **section nette** `Nu,Rd = 0,9·Anet·fu / γM2` (N)
/// au droit des trous (EN 1993-1-1 §6.2.3), le coefficient `0,9` traduisant la
/// concentration de contraintes autour des trous. `net_area` = `Anet` est
/// l'aire nette (mm²), `ultimate_strength` = `fu` la résistance ultime (MPa)
/// et `gamma_m2` = `γM2` le coefficient partiel (sans dimension) fourni par
/// l'Eurocode et son Annexe Nationale.
///
/// Panique si `net_area < 0`, si `ultimate_strength < 0` ou si `gamma_m2 <= 0`
/// (division par zéro).
pub fn steelten_net_section_resistance(
    net_area: f64,
    ultimate_strength: f64,
    gamma_m2: f64,
) -> f64 {
    assert!(net_area >= 0.0, "l'aire nette Anet doit être ≥ 0 (mm²)");
    assert!(
        ultimate_strength >= 0.0,
        "la résistance ultime fu doit être ≥ 0 (MPa)"
    );
    assert!(
        gamma_m2 > 0.0,
        "le coefficient partiel γM2 doit être strictement positif"
    );
    0.9 * net_area * ultimate_strength / gamma_m2
}

/// Résistance de calcul en traction `Nt,Rd = min(Npl,Rd ; Nu,Rd)` (N)
/// (EN 1993-1-1 §6.2.3(2)) : le plus petit de la résistance plastique de la
/// section brute et de la résistance ultime de la section nette. `gross_resistance`
/// = `Npl,Rd` et `net_resistance` = `Nu,Rd` sont des efforts (N).
///
/// Panique si `gross_resistance < 0` ou si `net_resistance < 0`.
pub fn steelten_design_resistance(gross_resistance: f64, net_resistance: f64) -> f64 {
    assert!(
        gross_resistance >= 0.0,
        "la résistance de section brute Npl,Rd doit être ≥ 0 (N)"
    );
    assert!(
        net_resistance >= 0.0,
        "la résistance de section nette Nu,Rd doit être ≥ 0 (N)"
    );
    gross_resistance.min(net_resistance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn net_area_deducts_aligned_holes() {
        // Anet = A − n·d0·t : deux trous d0 = 22 mm dans une plaque t = 10 mm
        // retirent 2·22·10 = 440 mm² d'une aire brute de 2000 mm² → 1560 mm².
        let anet = steelten_net_area(2000.0, 2.0, 22.0, 10.0);
        assert_relative_eq!(anet, 1560.0, epsilon = 1e-9);
        // Sans trou (n = 0), l'aire nette égale l'aire brute.
        assert_relative_eq!(
            steelten_net_area(2000.0, 0.0, 22.0, 10.0),
            2000.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn gross_resistance_reciprocity() {
        // Réciprocité : Npl,Rd · γM0 / fy restitue l'aire brute A.
        let (area, fy, gamma_m0) = (2000.0_f64, 355.0, 1.0);
        let npl = steelten_gross_section_resistance(area, fy, gamma_m0);
        assert_relative_eq!(npl * gamma_m0 / fy, area, epsilon = 1e-9);
    }

    #[test]
    fn net_resistance_proportional_to_area() {
        // Nu,Rd ∝ Anet : doubler l'aire nette double la résistance ultime.
        let single = steelten_net_section_resistance(1000.0, 490.0, 1.25);
        let double = steelten_net_section_resistance(2000.0, 490.0, 1.25);
        assert_relative_eq!(double, 2.0 * single, epsilon = 1e-9);
    }

    #[test]
    fn design_resistance_takes_minimum() {
        // Nt,Rd = min(Npl,Rd ; Nu,Rd), symétrique en ses arguments.
        assert_relative_eq!(
            steelten_design_resistance(710_000.0, 611_000.0),
            611_000.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            steelten_design_resistance(611_000.0, 710_000.0),
            611_000.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn realistic_flat_bar_s355_case() {
        // Plat 200 × 10 mm en S355, deux trous d0 = 22 mm, γM0 = 1,0, γM2 = 1,25 :
        //   A = 200·10 = 2000 mm² ; fy = 355 MPa ; fu = 490 MPa (S355).
        //   Anet = 2000 − 2·22·10 = 2000 − 440 = 1560 mm².
        //   Npl,Rd = 2000·355 / 1,0 = 710 000 N.
        //   Nu,Rd  = 0,9·1560·490 / 1,25 = 687 960 / 1,25 = 550 368 N.
        //   Nt,Rd  = min(710 000 ; 550 368) = 550 368 N (la section nette gouverne).
        let area = 2000.0_f64;
        let anet = steelten_net_area(area, 2.0, 22.0, 10.0);
        assert_relative_eq!(anet, 1560.0, epsilon = 1e-9);
        let npl = steelten_gross_section_resistance(area, 355.0, 1.0);
        assert_relative_eq!(npl, 710_000.0, epsilon = 1e-3);
        let nu = steelten_net_section_resistance(anet, 490.0, 1.25);
        assert_relative_eq!(nu, 550_368.0, epsilon = 1e-3);
        let nt = steelten_design_resistance(npl, nu);
        assert_relative_eq!(nt, 550_368.0, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "le coefficient partiel γM0 doit être strictement positif")]
    fn zero_gamma_m0_panics() {
        steelten_gross_section_resistance(2000.0, 355.0, 0.0);
    }
}
