//! **Béton précontraint — pertes de précontrainte et effort (Eurocode 2, ELS)** :
//! perte par raccourcissement élastique du béton, perte par relaxation de
//! l'acier, contrainte de précontrainte effective après pertes, effort de
//! précontrainte `P` et contrainte en fibre inférieure d'une section non
//! fissurée.
//!
//! ```text
//! raccourcissement élastique  Δσel   = αe · σc
//! relaxation de l'acier       Δσr    = relaxation_fraction · σp0
//! précontrainte effective     σp,eff = σp0 − Δσtot
//! effort de précontrainte     P      = σp,eff · Ap
//! contrainte fibre inférieure σinf   = P/A + P·e/Winf − M/Winf
//! ```
//!
//! `αe` coefficient d'équivalence acier/béton `Ep/Ecm` (sans dimension), `σc`
//! contrainte de compression du béton au niveau du câble (MPa), `Δσel` perte de
//! contrainte par raccourcissement élastique (MPa), `relaxation_fraction`
//! fraction de perte par relaxation de l'acier (sans dimension, p. ex. `0,025`
//! pour 2,5 %), `σp0` contrainte de précontrainte initiale dans le câble (MPa),
//! `Δσr` perte par relaxation (MPa), `Δσtot` somme des pertes fournies
//! (raccourcissement élastique, relaxation, fluage, retrait, frottement…) (MPa),
//! `σp,eff` contrainte de précontrainte effective après pertes (MPa), `Ap` aire
//! de la section des câbles (mm²), `P` effort de précontrainte (N), `A` aire de
//! la section brute de béton (mm²), `e` excentricité du câble par rapport au
//! centre de gravité de la section (mm), `Winf` module de flexion en fibre
//! inférieure (mm³), `M` moment fléchissant appliqué (N·mm), `σinf` contrainte
//! normale en fibre inférieure (MPa, compression comptée positive).
//!
//! **Convention** : N, mm, MPa (avec `1 MPa = 1 N/mm²`) ; les contraintes sont
//! en **MPa**, les aires en **mm²**, l'excentricité en **mm**, le module de
//! flexion en **mm³**, le moment en **N·mm** et l'effort en **N**.
//!
//! **Limite honnête** : la précontrainte initiale `σp0`, le coefficient
//! d'équivalence `αe`, la contrainte de béton au câble `σc`, la fraction de
//! relaxation `relaxation_fraction` **et** les pertes différées de fluage et de
//! retrait (incluses dans `Δσtot`) sont **fournies par l'appelant** d'après
//! l'Eurocode 2 et son Annexe Nationale ; aucune valeur « par défaut » n'est
//! inventée et les pertes de fluage/retrait **ne sont pas** recalculées en
//! détail. Le calcul de contrainte suppose une section **non fissurée** (ELS) et
//! un **comportement élastique linéaire** (superposition des effets). La
//! convention de signe (compression positive) et la conclusion réglementaire
//! restent à la charge de l'ingénieur.

/// Perte de contrainte par raccourcissement élastique `Δσel = αe · σc` (MPa),
/// avec `αe = Ep/Ecm` le coefficient d'équivalence (sans dimension) et `σc` la
/// contrainte de compression du béton au niveau du câble (MPa).
///
/// Panique si `modular_ratio <= 0` ou si `concrete_stress_at_tendon < 0`.
pub fn psc_elastic_shortening_loss(modular_ratio: f64, concrete_stress_at_tendon: f64) -> f64 {
    assert!(
        modular_ratio > 0.0,
        "le coefficient d'équivalence αe doit être strictement positif"
    );
    assert!(
        concrete_stress_at_tendon >= 0.0,
        "la contrainte de béton au câble σc doit être ≥ 0"
    );
    modular_ratio * concrete_stress_at_tendon
}

/// Perte de contrainte par relaxation de l'acier
/// `Δσr = relaxation_fraction · σp0` (MPa), avec `relaxation_fraction` la
/// fraction de relaxation fournie (sans dimension, `0 ≤ fraction ≤ 1`) et `σp0`
/// la précontrainte initiale (MPa).
///
/// Panique si `relaxation_fraction < 0`, si `relaxation_fraction > 1` ou si
/// `initial_prestress <= 0`.
pub fn psc_relaxation_loss(relaxation_fraction: f64, initial_prestress: f64) -> f64 {
    assert!(
        relaxation_fraction >= 0.0,
        "la fraction de relaxation doit être ≥ 0"
    );
    assert!(
        relaxation_fraction <= 1.0,
        "la fraction de relaxation doit être ≤ 1"
    );
    assert!(
        initial_prestress > 0.0,
        "la précontrainte initiale σp0 doit être strictement positive"
    );
    relaxation_fraction * initial_prestress
}

/// Contrainte de précontrainte effective après pertes
/// `σp,eff = σp0 − Δσtot` (MPa), avec `σp0` la précontrainte initiale (MPa) et
/// `Δσtot` la somme des pertes fournies (MPa).
///
/// Panique si `initial_prestress <= 0`, si `total_losses < 0` ou si
/// `total_losses > initial_prestress` (précontrainte effective négative).
pub fn psc_effective_prestress(initial_prestress: f64, total_losses: f64) -> f64 {
    assert!(
        initial_prestress > 0.0,
        "la précontrainte initiale σp0 doit être strictement positive"
    );
    assert!(
        total_losses >= 0.0,
        "les pertes totales Δσtot doivent être ≥ 0"
    );
    assert!(
        total_losses <= initial_prestress,
        "les pertes totales Δσtot ne peuvent dépasser la précontrainte initiale σp0"
    );
    initial_prestress - total_losses
}

/// Effort de précontrainte `P = σp,eff · Ap` (N), avec `σp,eff` la contrainte de
/// précontrainte effective (MPa) et `Ap` l'aire des câbles (mm²).
///
/// Panique si `effective_prestress <= 0` ou si `tendon_area <= 0`.
pub fn psc_prestress_force(effective_prestress: f64, tendon_area: f64) -> f64 {
    assert!(
        effective_prestress > 0.0,
        "la précontrainte effective σp,eff doit être strictement positive"
    );
    assert!(
        tendon_area > 0.0,
        "l'aire des câbles Ap doit être strictement positive"
    );
    effective_prestress * tendon_area
}

/// Contrainte normale en fibre inférieure d'une section non fissurée
/// `σinf = P/A + P·e/Winf − M/Winf` (MPa, compression positive), avec `P` en N,
/// `A` en mm², `e` en mm, `Winf` en mm³ et `M` en N·mm.
///
/// Panique si `prestress_force < 0`, si `section_area <= 0` ou si
/// `section_modulus_bottom <= 0`.
pub fn psc_concrete_stress_bottom(
    prestress_force: f64,
    section_area: f64,
    eccentricity: f64,
    section_modulus_bottom: f64,
    applied_moment: f64,
) -> f64 {
    assert!(
        prestress_force >= 0.0,
        "l'effort de précontrainte P doit être ≥ 0"
    );
    assert!(
        section_area > 0.0,
        "l'aire de la section A doit être strictement positive"
    );
    assert!(
        section_modulus_bottom > 0.0,
        "le module de flexion Winf doit être strictement positif"
    );
    prestress_force / section_area + prestress_force * eccentricity / section_modulus_bottom
        - applied_moment / section_modulus_bottom
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn elastic_shortening_case_and_proportionality() {
        // Cas usuel : αe = 6, σc = 5 MPa → Δσel = 6 · 5 = 30 MPa.
        let loss = psc_elastic_shortening_loss(6.0, 5.0);
        assert_relative_eq!(loss, 30.0, epsilon = 1e-12);
        // Proportionnalité : doubler σc double la perte.
        let loss2 = psc_elastic_shortening_loss(6.0, 10.0);
        assert_relative_eq!(loss2 / loss, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn relaxation_loss_case() {
        // Relaxation 2,5 % sur σp0 = 1360 MPa → Δσr = 0,025 · 1360 = 34 MPa.
        let dr = psc_relaxation_loss(0.025, 1360.0);
        assert_relative_eq!(dr, 34.0, epsilon = 1e-9);
        // Une fraction nulle donne une perte nulle.
        assert_relative_eq!(psc_relaxation_loss(0.0, 1360.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn effective_prestress_reciprocity() {
        // Réciprocité : σp,eff + Δσtot = σp0.
        let sigma0 = 1360.0;
        let losses = 260.0;
        let eff = psc_effective_prestress(sigma0, losses);
        assert_relative_eq!(eff, 1100.0, epsilon = 1e-9);
        assert_relative_eq!(eff + losses, sigma0, epsilon = 1e-9);
    }

    #[test]
    fn prestress_force_proportionality() {
        // P = σp,eff · Ap. σp,eff = 1000 MPa, Ap = 1000 mm² → P = 1 000 000 N.
        let p = psc_prestress_force(1000.0, 1000.0);
        assert_relative_eq!(p, 1_000_000.0, epsilon = 1e-6);
        // Doubler l'aire double l'effort.
        let p2 = psc_prestress_force(1000.0, 2000.0);
        assert_relative_eq!(p2 / p, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn concrete_stress_bottom_worked_case() {
        // Cas chiffré vérifié :
        //   P/A       = 1e6 / 1e5             = 10 MPa
        //   P·e/Winf  = 1e6 · 200 / 2e7 = 2e8 / 2e7 = 10 MPa
        //   M/Winf    = 3e8 / 2e7             = 15 MPa
        //   σinf      = 10 + 10 − 15          = 5 MPa
        let sigma = psc_concrete_stress_bottom(1.0e6, 1.0e5, 200.0, 2.0e7, 3.0e8);
        assert_relative_eq!(sigma, 5.0, epsilon = 1e-6);
    }

    #[test]
    fn concrete_stress_bottom_zero_moment_and_eccentricity() {
        // Sans moment ni excentricité, σinf = P/A (compression uniforme).
        //   P/A = 1e6 / 1e5 = 10 MPa.
        let sigma = psc_concrete_stress_bottom(1.0e6, 1.0e5, 0.0, 2.0e7, 0.0);
        assert_relative_eq!(sigma, 10.0, epsilon = 1e-9);
    }

    #[test]
    fn full_chain_case() {
        // Chaîne complète : pertes → précontrainte effective → effort.
        //   Δσel = 6 · 5 = 30 MPa
        //   Δσr  = 0,025 · 1360 = 34 MPa
        //   Δσtot (avec fluage/retrait fournis = 236) = 30 + 34 + 236 = 300 MPa
        //   σp,eff = 1360 − 300 = 1060 MPa
        //   P = 1060 · 1000 = 1 060 000 N
        let dsel = psc_elastic_shortening_loss(6.0, 5.0);
        let dsr = psc_relaxation_loss(0.025, 1360.0);
        let total = dsel + dsr + 236.0;
        let eff = psc_effective_prestress(1360.0, total);
        assert_relative_eq!(eff, 1060.0, epsilon = 1e-9);
        let p = psc_prestress_force(eff, 1000.0);
        assert_relative_eq!(p, 1_060_000.0, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(
        expected = "les pertes totales Δσtot ne peuvent dépasser la précontrainte initiale σp0"
    )]
    fn effective_prestress_rejects_excessive_losses() {
        // Pertes supérieures à la précontrainte initiale : σp,eff serait négatif.
        psc_effective_prestress(1000.0, 1200.0);
    }
}
