//! **Structure bois — assemblage par organe** métallique en simple cisaillement
//! (Eurocode 5, EN 1995-1-1) : résistance à l'enfoncement `fh,k`, moment
//! d'écoulement plastique `My,Rk` de l'organe, et capacités par mode de rupture
//! de la théorie de **Johansen**.
//!
//! ```text
//! enfoncement   fh,k = 0,082 · (1 - 0,01·d) · ρk
//! moment plast. My,Rk = 0,3 · fu · d^2,6
//! mode plaque mince    Fv,Rk = fh,k · t · d
//! mode plaque épaisse  Fv,Rk = 2,3 · sqrt( My,Rk · fh,k · d )
//! ```
//!
//! `ρk` = `density` masse volumique caractéristique du bois (kg/m³), `d` =
//! `fastener_diameter` diamètre de l'organe (mm), `fh,k` =
//! `embedment_strength` résistance caractéristique à l'enfoncement (MPa), `fu` =
//! `ultimate_tensile_strength` résistance ultime en traction de l'acier de
//! l'organe (MPa), `My,Rk` = `yield_moment` moment d'écoulement plastique
//! (N·mm), `t` = `timber_thickness` épaisseur du bois pénétré (mm), `Fv,Rk`
//! capacité caractéristique par organe et par plan de cisaillement (N).
//!
//! **Convention** : unités **N, mm, MPa** (1 MPa = 1 N/mm²), cohérentes entre
//! elles (Eurocode) ; la masse volumique est en **kg/m³** (unité de la formule
//! empirique d'enfoncement), les moments en **N·mm**, les capacités en **N**.
//! Types `f64`.
//!
//! **Limite honnête** : organe métallique en **simple cisaillement** selon la
//! **théorie de Johansen** (modes de rupture de l'EN 1995-1-1 §8.2.2) ; la
//! résistance à l'enfoncement `fh,k` (bois massif, effort perpendiculaire aux
//! fibres, organe **non prédétré**) et le moment d'écoulement `My,Rk` suivent
//! les **formules empiriques** de l'Eurocode 5, avec la **masse volumique** `ρk`
//! **et la résistance ultime** `fu` **fournies par l'appelant** — aucune valeur
//! « par défaut » n'est inventée. L'**effet de corde** (arrachement, `Fax,Rk`),
//! les **distances aux rives et entre organes**, le **nombre efficace** `nef`,
//! le **coefficient partiel** `γ_M` et le **coefficient** `kmod` sont à vérifier
//! et à appliquer par l'appelant d'après l'Eurocode 5 et son Annexe Nationale.
//! La **capacité de calcul** de l'assemblage est le **minimum** des capacités de
//! tous les modes de rupture pertinents.

/// Résistance caractéristique à l'enfoncement
/// `fh,k = 0,082 · (1 - 0,01·d) · ρk` (MPa) pour un organe **non prédétré** dans
/// du **bois massif**, effort **perpendiculaire** aux fibres
/// (Eurocode 5, EN 1995-1-1 §8.3.1.1, éq. 8.15).
///
/// `density` = `ρk` masse volumique caractéristique du bois (kg/m³, fournie),
/// `fastener_diameter` = `d` diamètre de l'organe (mm, formule valable pour
/// `d ≤ 8 mm` selon l'Eurocode) ; renvoie `fh,k` (MPa).
///
/// Panique si `density < 0`, si `fastener_diameter < 0` ou si
/// `fastener_diameter > 100` (facteur `1 - 0,01·d` négatif, hors domaine).
pub fn timberconn_embedment_strength(density: f64, fastener_diameter: f64) -> f64 {
    assert!(density >= 0.0, "la masse volumique ρk doit être ≥ 0");
    assert!(
        fastener_diameter >= 0.0,
        "le diamètre de l'organe d doit être ≥ 0"
    );
    assert!(
        fastener_diameter <= 100.0,
        "le diamètre d doit rester ≤ 100 mm (facteur 1 - 0,01·d ≥ 0)"
    );
    0.082 * (1.0 - 0.01 * fastener_diameter) * density
}

/// Moment d'écoulement plastique caractéristique d'un organe (pointe, broche ou
/// boulon) `My,Rk = 0,3 · fu · d^2,6` (N·mm)
/// (Eurocode 5, EN 1995-1-1 §8.3.1.1, éq. 8.14).
///
/// `ultimate_tensile_strength` = `fu` résistance ultime en traction de l'acier
/// de l'organe (MPa, fournie), `fastener_diameter` = `d` diamètre de l'organe
/// (mm) ; renvoie `My,Rk` (N·mm, car MPa · mm^2,6 = N/mm² · mm^2,6 = N·mm^0,6…,
/// la constante empirique 0,3 étant dimensionnée pour donner des N·mm).
///
/// Panique si `ultimate_tensile_strength < 0` ou si `fastener_diameter < 0`
/// (base de la puissance non réelle).
pub fn timberconn_yield_moment(ultimate_tensile_strength: f64, fastener_diameter: f64) -> f64 {
    assert!(
        ultimate_tensile_strength >= 0.0,
        "la résistance ultime fu doit être ≥ 0"
    );
    assert!(
        fastener_diameter >= 0.0,
        "le diamètre de l'organe d doit être ≥ 0"
    );
    0.3 * ultimate_tensile_strength * fastener_diameter.powf(2.6)
}

/// Capacité caractéristique du mode de rupture « **plaque mince** » (écrasement
/// du bois sur toute l'épaisseur, sans rotule plastique dans l'organe)
/// `Fv,Rk = fh,k · t · d` (N) (Eurocode 5, EN 1995-1-1 §8.2.3, mode a).
///
/// `embedment_strength` = `fh,k` résistance à l'enfoncement (MPa),
/// `fastener_diameter` = `d` diamètre de l'organe (mm), `timber_thickness` = `t`
/// épaisseur du bois pénétré (mm) ; renvoie la capacité (N, car MPa · mm · mm =
/// N/mm² · mm² = N).
///
/// Panique si `embedment_strength < 0`, si `fastener_diameter < 0` ou si
/// `timber_thickness < 0`.
pub fn timberconn_capacity_thin_plate_single_shear(
    embedment_strength: f64,
    fastener_diameter: f64,
    timber_thickness: f64,
) -> f64 {
    assert!(
        embedment_strength >= 0.0,
        "la résistance à l'enfoncement fh,k doit être ≥ 0"
    );
    assert!(
        fastener_diameter >= 0.0,
        "le diamètre de l'organe d doit être ≥ 0"
    );
    assert!(
        timber_thickness >= 0.0,
        "l'épaisseur du bois t doit être ≥ 0"
    );
    embedment_strength * timber_thickness * fastener_diameter
}

/// Capacité caractéristique du mode de rupture « **plaque épaisse** » (rotule
/// plastique dans l'organe, plastification du bois)
/// `Fv,Rk = 2,3 · sqrt( My,Rk · fh,k · d )` (N)
/// (Eurocode 5, EN 1995-1-1 §8.2.3, mode b).
///
/// `embedment_strength` = `fh,k` résistance à l'enfoncement (MPa),
/// `fastener_diameter` = `d` diamètre de l'organe (mm), `yield_moment` =
/// `My,Rk` moment d'écoulement plastique (N·mm) ; renvoie la capacité (N, car
/// sqrt(N·mm · N/mm² · mm) = sqrt(N²) = N).
///
/// Panique si `yield_moment < 0`, si `embedment_strength < 0` ou si
/// `fastener_diameter < 0` (racine d'un produit négatif).
pub fn timberconn_capacity_thick_plate_yield(
    embedment_strength: f64,
    fastener_diameter: f64,
    yield_moment: f64,
) -> f64 {
    assert!(
        yield_moment >= 0.0,
        "le moment d'écoulement My,Rk doit être ≥ 0"
    );
    assert!(
        embedment_strength >= 0.0,
        "la résistance à l'enfoncement fh,k doit être ≥ 0"
    );
    assert!(
        fastener_diameter >= 0.0,
        "le diamètre de l'organe d doit être ≥ 0"
    );
    2.3 * (yield_moment * embedment_strength * fastener_diameter).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn embedment_strength_reference_value() {
        // Bois massif ρk = 350 kg/m³, pointe d = 4 mm (non prédétré) :
        //   fh,k = 0,082·(1 - 0,01·4)·350 = 0,082·0,96·350
        //        = 0,082·336 = 27,552 MPa.
        let fhk = timberconn_embedment_strength(350.0, 4.0);
        assert_relative_eq!(fhk, 27.552, epsilon = 1e-9);
    }

    #[test]
    fn embedment_strength_proportional_to_density() {
        // fh,k est linéaire en ρk : doubler la densité double fh,k.
        let f1 = timberconn_embedment_strength(350.0, 6.0);
        let f2 = timberconn_embedment_strength(700.0, 6.0);
        assert_relative_eq!(f2, 2.0 * f1, epsilon = 1e-9);
    }

    #[test]
    fn yield_moment_reciprocity() {
        // Réciprocité : My,Rk / (0,3·fu) restitue d^2,6.
        let (fu, d) = (600.0_f64, 5.0_f64);
        let my = timberconn_yield_moment(fu, d);
        assert_relative_eq!(my / (0.3 * fu), d.powf(2.6), epsilon = 1e-9);
    }

    #[test]
    fn thin_plate_capacity_proportionalities() {
        // Fv,Rk = fh,k·t·d : linéaire en chacun des trois facteurs.
        let base = timberconn_capacity_thin_plate_single_shear(27.552, 4.0, 20.0);
        // 27,552·20·4 = 2204,16 N.
        assert_relative_eq!(base, 2204.16, epsilon = 1e-9);
        // Doubler l'épaisseur double la capacité.
        let dbl_t = timberconn_capacity_thin_plate_single_shear(27.552, 4.0, 40.0);
        assert_relative_eq!(dbl_t, 2.0 * base, epsilon = 1e-9);
        // Doubler le diamètre double la capacité.
        let dbl_d = timberconn_capacity_thin_plate_single_shear(27.552, 8.0, 20.0);
        assert_relative_eq!(dbl_d, 2.0 * base, epsilon = 1e-9);
    }

    #[test]
    fn thick_plate_capacity_scaling() {
        // Fv,Rk = 2,3·sqrt(My·fh,k·d) : multiplier le produit interne par 4
        // double la capacité (racine carrée).
        let base = timberconn_capacity_thick_plate_yield(27.552, 4.0, 6616.5);
        let scaled = timberconn_capacity_thick_plate_yield(4.0 * 27.552, 4.0, 6616.5);
        assert_relative_eq!(scaled, 2.0 * base, epsilon = 1e-9);
    }

    #[test]
    fn realistic_nailed_joint_governing_mode() {
        // Assemblage cloué, simple cisaillement, plaque métallique + bois massif :
        //   ρk = 350 kg/m³ ; pointe d = 4 mm ; fu = 600 MPa ; t = 20 mm.
        //   fh,k  = 0,082·0,96·350                = 27,552 MPa
        //   My,Rk = 0,3·600·4^2,6 = 180·36,758347 = 6616,50 N·mm
        //   plaque mince   : 27,552·20·4          = 2204,16 N
        //   plaque épaisse : 2,3·sqrt(6616,50·27,552·4)
        //                  = 2,3·sqrt(729191,7)
        //                  = 2,3·853,928           = 1964,03 N
        //   capacité retenue = min(2204,16 ; 1964,03) = 1964,03 N (plaque épaisse).
        let fhk = timberconn_embedment_strength(350.0, 4.0);
        assert_relative_eq!(fhk, 27.552, epsilon = 1e-9);

        let my = timberconn_yield_moment(600.0, 4.0);
        assert_relative_eq!(my, 6616.5025, epsilon = 1e-3);

        let thin = timberconn_capacity_thin_plate_single_shear(fhk, 4.0, 20.0);
        assert_relative_eq!(thin, 2204.16, epsilon = 1e-3);

        let thick = timberconn_capacity_thick_plate_yield(fhk, 4.0, my);
        assert_relative_eq!(thick, 1964.032, epsilon = 1e-3);

        let governing = thin.min(thick);
        assert_relative_eq!(governing, 1964.032, epsilon = 1e-3);
        assert!(thick < thin); // le mode plaque épaisse gouverne
    }

    #[test]
    #[should_panic(expected = "le diamètre d doit rester ≤ 100 mm")]
    fn oversized_diameter_panics() {
        timberconn_embedment_strength(350.0, 150.0);
    }
}
