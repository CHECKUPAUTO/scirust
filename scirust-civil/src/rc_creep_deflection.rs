//! **Béton armé — flèche différée par fluage (Eurocode 2)** : module effectif
//! du béton `Ec,eff`, courbure `1/r` sous moment de service, coefficient de
//! distribution `ζ` entre état non fissuré et fissuré, flèche interpolée et
//! flèche totale par l'approche simplifiée `f · (1 + φ)`.
//!
//! ```text
//! module effectif     Ec,eff = Ecm / (1 + φ)
//! courbure            1/r    = M / (Ec,eff · Ic)
//! coefficient distrib ζ      = 1 − β · (Mcr / M)²
//! flèche interpolée   f      = (1 − ζ) · fI + ζ · fII
//! flèche totale       ftot   = finst · (1 + φ)
//! ```
//!
//! `Ecm` module d'élasticité sécant du béton (Pa), `φ` coefficient de fluage
//! (sans dimension), `Ec,eff` module effectif à long terme (Pa), `M` moment
//! fléchissant de service appliqué (N·m), `Ic` inertie de la section (m⁴,
//! fissurée ou brute selon l'état considéré), `1/r` courbure (1/m), `β`
//! coefficient tenant compte de la durée/répétition du chargement (sans
//! dimension), `Mcr` moment de fissuration (N·m), `ζ` coefficient de
//! distribution/répartition (sans dimension, entre 0 et 1), `fI` flèche en état
//! non fissuré (état I, m), `fII` flèche en état fissuré (état II, m), `finst`
//! flèche instantanée (m), `ftot` flèche totale incluant le fluage (m).
//!
//! **Convention** : unités SI cohérentes — N, m, Pa (`1 Pa = 1 N/m²`) ; les
//! modules sont en **Pa**, les moments en **N·m**, les inerties en **m⁴**, les
//! courbures en **1/m**, les flèches en **m** ; les coefficients (`φ`, `β`, `ζ`)
//! sont **sans dimension**. Toute échelle homogène (p. ex. MPa/mm/N·mm) donne
//! les mêmes coefficients tant qu'elle reste cohérente pour la courbure et la
//! flèche.
//!
//! **Limite honnête** : interpolation **simplifiée** de l'Eurocode 2. Le
//! coefficient de fluage `φ` (fonction de l'humidité, de l'âge de mise en
//! charge et de la géométrie), le module sécant `Ecm`, les inerties `Ic`
//! (fissurée en état II, brute en état I) et le moment de fissuration `Mcr` sont
//! **fournis par l'appelant** d'après l'EC2 et son Annexe Nationale ; aucune
//! valeur n'est inventée. Le **retrait** (dont la courbure de retrait) est
//! traité **séparément** et n'est pas inclus ici. La flèche totale
//! `finst · (1 + φ)` est une **approximation d'ingénieur** ; l'intégration
//! rigoureuse des courbures le long de l'élément et le choix de la combinaison
//! d'actions quasi-permanente restent à la charge de l'ingénieur.

/// Module d'élasticité effectif à long terme `Ec,eff = Ecm / (1 + φ)` (Pa),
/// avec le module sécant `Ecm` en Pa et le coefficient de fluage `φ` sans
/// dimension. Plus le fluage est important, plus le module effectif est faible.
///
/// Panique si `elastic_modulus <= 0` ou si `creep_coefficient < 0`.
pub fn creep_effective_modulus(elastic_modulus: f64, creep_coefficient: f64) -> f64 {
    assert!(
        elastic_modulus > 0.0,
        "le module d'élasticité Ecm doit être strictement positif"
    );
    assert!(
        creep_coefficient >= 0.0,
        "le coefficient de fluage φ doit être ≥ 0"
    );
    elastic_modulus / (1.0 + creep_coefficient)
}

/// Courbure sous moment de service `1/r = M / (Ec,eff · Ic)` (1/m), avec le
/// moment `M` en N·m, le module effectif `Ec,eff` en Pa et l'inertie `Ic` en
/// m⁴.
///
/// Panique si `effective_modulus <= 0` ou si `cracked_inertia <= 0` (division).
pub fn creep_curvature(moment: f64, effective_modulus: f64, cracked_inertia: f64) -> f64 {
    assert!(
        effective_modulus > 0.0,
        "le module effectif Ec,eff doit être strictement positif"
    );
    assert!(
        cracked_inertia > 0.0,
        "l'inertie Ic doit être strictement positive"
    );
    moment / (effective_modulus * cracked_inertia)
}

/// Coefficient de distribution `ζ = 1 − β · (Mcr / M)²` (sans dimension), avec
/// `β` coefficient de durée/répétition de charge, `Mcr` moment de fissuration
/// et `M` moment appliqué (mêmes unités de moment). Tant que `M ≤ Mcr` la
/// section n'est pas fissurée et `ζ` peut être négatif : il est alors borné à
/// `0` (état non fissuré pur) selon l'EC2.
///
/// Panique si `beta < 0`, si `cracking_moment < 0` ou si `applied_moment <= 0`
/// (division).
pub fn creep_distribution_coefficient(beta: f64, cracking_moment: f64, applied_moment: f64) -> f64 {
    assert!(beta >= 0.0, "le coefficient β doit être ≥ 0");
    assert!(
        cracking_moment >= 0.0,
        "le moment de fissuration Mcr doit être ≥ 0"
    );
    assert!(
        applied_moment > 0.0,
        "le moment appliqué M doit être strictement positif"
    );
    let ratio = cracking_moment / applied_moment;
    let zeta = 1.0 - beta * ratio.powi(2);
    if zeta < 0.0 { 0.0 } else { zeta }
}

/// Flèche interpolée `f = (1 − ζ) · fI + ζ · fII` (m), entre la flèche en état
/// non fissuré `fI` (état I) et la flèche en état fissuré `fII` (état II), le
/// coefficient de distribution `ζ` étant sans dimension.
///
/// Panique si `distribution_coefficient` n'est pas dans `[0, 1]`.
pub fn creep_interpolated_deflection(
    uncracked_deflection: f64,
    cracked_deflection: f64,
    distribution_coefficient: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&distribution_coefficient),
        "le coefficient de distribution ζ doit être compris entre 0 et 1"
    );
    (1.0 - distribution_coefficient) * uncracked_deflection
        + distribution_coefficient * cracked_deflection
}

/// Flèche totale incluant le fluage `ftot = finst · (1 + φ)` (m), par l'approche
/// simplifiée, avec la flèche instantanée `finst` en m et le coefficient de
/// fluage `φ` sans dimension.
///
/// Panique si `instantaneous_deflection < 0` ou si `creep_coefficient < 0`.
pub fn creep_total_deflection(instantaneous_deflection: f64, creep_coefficient: f64) -> f64 {
    assert!(
        instantaneous_deflection >= 0.0,
        "la flèche instantanée finst doit être ≥ 0"
    );
    assert!(
        creep_coefficient >= 0.0,
        "le coefficient de fluage φ doit être ≥ 0"
    );
    instantaneous_deflection * (1.0 + creep_coefficient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn effective_modulus_halves_when_phi_unity() {
        // φ = 1 : Ec,eff = Ecm / (1 + 1) = Ecm / 2.
        // Ecm = 33 GPa = 33e9 Pa → Ec,eff = 16,5e9 Pa.
        let e_eff = creep_effective_modulus(33.0e9, 1.0);
        assert_relative_eq!(e_eff, 16.5e9, epsilon = 1.0);
        // Limite φ = 0 : aucun fluage, module inchangé.
        assert_relative_eq!(creep_effective_modulus(33.0e9, 0.0), 33.0e9, epsilon = 1e-3);
    }

    #[test]
    fn curvature_clean_case_and_inertia_inverse() {
        // Cas chiffré : M = 100 000 N·m, Ec,eff = 20e9 Pa, Ic = 5e-4 m⁴.
        //   1/r = 100000 / (20e9 · 5e-4) = 100000 / 1,0e7 = 1,0e-2 1/m
        // Recalcul : dénominateur 20e9 · 5e-4 = 1,0e7 ; 1e5 / 1e7 = 0,01.
        let curv = creep_curvature(100_000.0, 20.0e9, 5.0e-4);
        assert_relative_eq!(curv, 0.01, epsilon = 1e-9);
        // Inertie doublée → courbure moitié.
        let curv2 = creep_curvature(100_000.0, 20.0e9, 1.0e-3);
        assert_relative_eq!(curv2 / curv, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn distribution_coefficient_case_and_clamp() {
        // Cas chiffré : β = 1, Mcr = 60 kN·m, M = 100 kN·m.
        //   ζ = 1 − 1 · (60/100)² = 1 − 0,36 = 0,64
        let zeta = creep_distribution_coefficient(1.0, 60.0e3, 100.0e3);
        assert_relative_eq!(zeta, 0.64, epsilon = 1e-9);
        // M ≤ Mcr : la formule brute donnerait ζ < 0, on borne à 0.
        let zeta0 = creep_distribution_coefficient(1.0, 100.0e3, 80.0e3);
        assert_relative_eq!(zeta0, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn interpolated_deflection_endpoints_and_case() {
        // ζ = 0 → flèche = état I ; ζ = 1 → flèche = état II.
        let fi = 5.0e-3;
        let fii = 20.0e-3;
        assert_relative_eq!(
            creep_interpolated_deflection(fi, fii, 0.0),
            fi,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            creep_interpolated_deflection(fi, fii, 1.0),
            fii,
            epsilon = 1e-12
        );
        // Cas chiffré : ζ = 0,64.
        //   f = (1 − 0,64)·5 + 0,64·20 = 0,36·5 + 0,64·20 = 1,8 + 12,8 = 14,6 mm
        let f = creep_interpolated_deflection(fi, fii, 0.64);
        assert_relative_eq!(f, 14.6e-3, epsilon = 1e-9);
    }

    #[test]
    fn total_deflection_case_and_no_creep_limit() {
        // Cas chiffré : finst = 10 mm, φ = 2 → ftot = 10 · (1 + 2) = 30 mm.
        let ftot = creep_total_deflection(10.0e-3, 2.0);
        assert_relative_eq!(ftot, 30.0e-3, epsilon = 1e-12);
        // Limite φ = 0 : flèche totale = flèche instantanée.
        assert_relative_eq!(
            creep_total_deflection(10.0e-3, 0.0),
            10.0e-3,
            epsilon = 1e-12
        );
    }

    #[test]
    fn effective_modulus_reduces_curvature_consistently() {
        // Cohérence : à M et Ic fixés, un module effectif plus faible (fluage
        // plus fort) augmente la courbure d'un facteur (1 + φ).
        // Ecm = 30e9, φ = 2 → Ec,eff = 10e9.
        let e_eff = creep_effective_modulus(30.0e9, 2.0);
        let curv_long = creep_curvature(50_000.0, e_eff, 4.0e-4);
        let curv_short = creep_curvature(50_000.0, 30.0e9, 4.0e-4);
        assert_relative_eq!(curv_long / curv_short, 3.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "le coefficient de distribution ζ doit être compris entre 0 et 1")]
    fn interpolated_deflection_rejects_out_of_range_zeta() {
        // ζ = 1,2 hors de l'intervalle [0, 1] : interdit.
        creep_interpolated_deflection(5.0e-3, 20.0e-3, 1.2);
    }
}
