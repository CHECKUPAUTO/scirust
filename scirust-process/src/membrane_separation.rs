//! Séparation membranaire — flux à travers une membrane dense (modèle
//! **solution-diffusion**), taux de rejet et de récupération, facteur de
//! concentration du rétentat et pression osmotique (loi de **van't Hoff**).
//!
//! ```text
//! flux solution-diffusion   J   = P·Δp / δ                               [mol·m⁻²·s⁻¹] ou [m·s⁻¹]
//! taux de rejet observé     R   = 1 − c_p / c_f                          [-]
//! taux de récupération      Y   = Q_p / Q_f                              [-]
//! facteur de concentration  CF  = (1 − Y·(1 − R)) / (1 − Y)              [-]
//! pression osmotique        π   = i·c·R_g·T                              [Pa]
//! ```
//!
//! `J` flux transmembranaire (densité de flux ; unité selon la perméabilité
//! fournie) ; `P` perméabilité de la membrane [(unité de J)·m·Pa⁻¹, FOURNIE] ;
//! `Δp` pression motrice **effective** [Pa] ; `δ` épaisseur de la couche active
//! [m] ; `R` taux de rejet observé [sans dimension] ; `c_f`/`c_p` concentrations
//! côté alimentation/perméat [même unité, p. ex. mol·m⁻³] ; `Y` taux de
//! récupération (conversion) [sans dimension] ; `Q_p`/`Q_f` débits de
//! perméat/alimentation [même unité, p. ex. m³·s⁻¹] ; `CF` facteur de
//! concentration du rétentat [sans dimension] ; `π` pression osmotique [Pa] ;
//! `i` facteur de van't Hoff [sans dimension, FOURNI] ; `c` concentration
//! molaire de soluté [mol·m⁻³] ; `R_g` constante des gaz parfaits [J·mol⁻¹·K⁻¹] ;
//! `T` température [K].
//!
//! **Limite honnête** : modèle **solution-diffusion** avec **perméabilité `P`
//! FOURNIE** par l'appelant (jamais inventée ; issue d'essais ou de fiches
//! fabricant). En osmose inverse, la **pression motrice effective** est la
//! différence de pression **moins** la différence de pression osmotique
//! (`Δp = Δp_hydraulique − Δπ`) : cette composition est à réaliser par
//! l'appelant avant d'appeler [`memb_solution_diffusion_flux`]. Le rejet et la
//! récupération résultent d'un **bilan de matière** ; le facteur de
//! concentration suppose un rejet `R` **constant** sur le module. La
//! **polarisation de concentration est NÉGLIGÉE** — si elle compte, son facteur
//! est à FOURNIR et à appliquer par l'appelant. La pression osmotique suit la
//! **loi de van't Hoff** (solutions diluées), avec un **facteur `i` FOURNI** ;
//! la **température est en KELVIN**.

/// Flux transmembranaire (modèle **solution-diffusion**) `J = P·Δp / δ`.
///
/// `permeability` (P) perméabilité FOURNIE [(unité de J)·m·Pa⁻¹] ;
/// `thickness` (δ) épaisseur de la couche active [m] ; `driving_pressure` (Δp)
/// pression motrice **effective** [Pa] (en osmose inverse : différence de
/// pression moins différence de pression osmotique, composée par l'appelant).
///
/// Panique si `permeability < 0`, `thickness <= 0` ou `driving_pressure < 0`.
pub fn memb_solution_diffusion_flux(
    permeability: f64,
    thickness: f64,
    driving_pressure: f64,
) -> f64 {
    assert!(permeability >= 0.0, "perméabilité P ≥ 0 requise");
    assert!(thickness > 0.0, "épaisseur δ > 0 requise");
    assert!(
        driving_pressure >= 0.0,
        "pression motrice effective Δp ≥ 0 requise"
    );
    permeability * driving_pressure / thickness
}

/// Taux de rejet observé `R = 1 − c_p / c_f` (sans dimension).
///
/// `feed_concentration` (c_f) et `permeate_concentration` (c_p) concentrations
/// côté alimentation/perméat, dans la **même unité** (p. ex. mol·m⁻³). `R = 1`
/// pour un rejet total ; `R` peut être négatif si `c_p > c_f`.
///
/// Panique si `feed_concentration <= 0` ou `permeate_concentration < 0`.
pub fn memb_rejection(feed_concentration: f64, permeate_concentration: f64) -> f64 {
    assert!(
        feed_concentration > 0.0,
        "concentration d'alimentation c_f > 0 requise"
    );
    assert!(
        permeate_concentration >= 0.0,
        "concentration de perméat c_p ≥ 0 requise"
    );
    1.0 - permeate_concentration / feed_concentration
}

/// Taux de récupération (conversion) `Y = Q_p / Q_f` (sans dimension).
///
/// `permeate_flow` (Q_p) et `feed_flow` (Q_f) débits de perméat/alimentation
/// dans la **même unité** (p. ex. m³·s⁻¹). Par bilan, `0 ≤ Y ≤ 1`.
///
/// Panique si `feed_flow <= 0`, `permeate_flow < 0` ou `permeate_flow > feed_flow`.
pub fn memb_recovery(permeate_flow: f64, feed_flow: f64) -> f64 {
    assert!(feed_flow > 0.0, "débit d'alimentation Q_f > 0 requis");
    assert!(permeate_flow >= 0.0, "débit de perméat Q_p ≥ 0 requis");
    assert!(
        permeate_flow <= feed_flow,
        "Q_p ≤ Q_f requis (perméat ≤ alimentation)"
    );
    permeate_flow / feed_flow
}

/// Facteur de concentration du rétentat `CF = (1 − Y·(1 − R)) / (1 − Y)`
/// (sans dimension), à rejet `R` supposé **constant** sur le module.
///
/// `recovery` (Y) taux de récupération [sans dimension] ; `rejection` (R) taux
/// de rejet [sans dimension]. À rejet total (`R = 1`), `CF = 1/(1 − Y)`.
///
/// Panique si `recovery` n'est pas dans `[0, 1[` ou si `rejection` n'est pas
/// dans `[0, 1]`.
pub fn memb_concentration_factor(recovery: f64, rejection: f64) -> f64 {
    assert!(
        (0.0..1.0).contains(&recovery),
        "taux de récupération Y dans [0, 1[ requis"
    );
    assert!(
        (0.0..=1.0).contains(&rejection),
        "taux de rejet R dans [0, 1] requis"
    );
    (1.0 - recovery * (1.0 - rejection)) / (1.0 - recovery)
}

/// Pression osmotique `π = i·c·R_g·T` (Pa), loi de **van't Hoff**.
///
/// `van_t_hoff_factor` (i) facteur de van't Hoff FOURNI [sans dimension] ;
/// `molar_concentration` (c) concentration molaire de soluté [mol·m⁻³] ;
/// `gas_constant` (R_g) constante des gaz parfaits [J·mol⁻¹·K⁻¹] ;
/// `temperature` (T) température **en kelvin** [K].
///
/// Panique si `van_t_hoff_factor <= 0`, `molar_concentration < 0`,
/// `gas_constant <= 0` ou `temperature <= 0`.
pub fn memb_osmotic_pressure(
    van_t_hoff_factor: f64,
    molar_concentration: f64,
    gas_constant: f64,
    temperature: f64,
) -> f64 {
    assert!(
        van_t_hoff_factor > 0.0,
        "facteur de van't Hoff i > 0 requis"
    );
    assert!(
        molar_concentration >= 0.0,
        "concentration molaire c ≥ 0 requise"
    );
    assert!(gas_constant > 0.0, "constante des gaz R_g > 0 requise");
    assert!(temperature > 0.0, "température T > 0 K requise (kelvin)");
    van_t_hoff_factor * molar_concentration * gas_constant * temperature
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn flux_definition_and_proportionality() {
        // P = 1e-12, δ = 1e-4, Δp = 1e6 ⇒ J = 1e-12·1e6/1e-4 = 1e-6/1e-4 = 1e-2.
        let j = memb_solution_diffusion_flux(1.0e-12_f64, 1.0e-4_f64, 1.0e6_f64);
        assert_relative_eq!(j, 1.0e-2, max_relative = 1e-9);
        // Le flux est proportionnel à la pression motrice : doubler Δp double J.
        let j2 = memb_solution_diffusion_flux(1.0e-12_f64, 1.0e-4_f64, 2.0e6_f64);
        assert_relative_eq!(j2, 2.0 * j, max_relative = 1e-9);
        // Pression motrice nulle ⇒ flux nul.
        assert_relative_eq!(
            memb_solution_diffusion_flux(1.0e-12_f64, 1.0e-4_f64, 0.0_f64),
            0.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn rejection_realistic_case() {
        // c_f = 1000, c_p = 50 ⇒ R = 1 − 50/1000 = 1 − 0.05 = 0.95.
        assert_relative_eq!(
            memb_rejection(1000.0_f64, 50.0_f64),
            0.95,
            max_relative = 1e-12
        );
        // Perméat nul ⇒ rejet total R = 1.
        assert_relative_eq!(
            memb_rejection(1000.0_f64, 0.0_f64),
            1.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn recovery_definition_and_bounds() {
        // Q_p = 30, Q_f = 100 ⇒ Y = 0.3.
        assert_relative_eq!(
            memb_recovery(30.0_f64, 100.0_f64),
            0.3,
            max_relative = 1e-12
        );
        // Tout le débit récupéré ⇒ Y = 1.
        assert_relative_eq!(
            memb_recovery(100.0_f64, 100.0_f64),
            1.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn concentration_factor_perfect_rejection() {
        // R = 1 : CF = (1 − Y·0)/(1 − Y) = 1/(1 − Y). Y = 0.75 ⇒ CF = 1/0.25 = 4.
        assert_relative_eq!(
            memb_concentration_factor(0.75_f64, 1.0_f64),
            4.0,
            max_relative = 1e-12
        );
        // R = 0.9, Y = 0.5 : CF = (1 − 0.5·0.1)/(1 − 0.5) = 0.95/0.5 = 1.9.
        assert_relative_eq!(
            memb_concentration_factor(0.5_f64, 0.9_f64),
            1.9,
            max_relative = 1e-12
        );
        // Récupération nulle ⇒ pas de concentration (CF = 1) quel que soit R.
        assert_relative_eq!(
            memb_concentration_factor(0.0_f64, 0.8_f64),
            1.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn osmotic_pressure_realistic_case() {
        // i = 2, c = 1000 mol·m⁻³, R_g = 8.314, T = 298 K :
        // π = 2·1000·8.314·298 = 2·8.314 = 16.628 ; ·1000 = 16628 ; ·298 = 4 955 144 Pa.
        let pi = memb_osmotic_pressure(2.0_f64, 1000.0_f64, 8.314_f64, 298.0_f64);
        assert_relative_eq!(pi, 4_955_144.0, max_relative = 1e-9);
        // Concentration nulle ⇒ pression osmotique nulle.
        assert_relative_eq!(
            memb_osmotic_pressure(2.0_f64, 0.0_f64, 8.314_f64, 298.0_f64),
            0.0,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "taux de récupération Y dans [0, 1[ requis")]
    fn concentration_factor_panics_at_full_recovery() {
        // Y = 1 rend le dénominateur (1 − Y) nul ⇒ formule singulière, panique.
        let _ = memb_concentration_factor(1.0_f64, 0.95_f64);
    }
}
