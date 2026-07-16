//! Absorption / stripping **étagé** par la méthode de **Kremser** — facteur
//! d'absorption `A`, facteur de stripping `S`, fraction de soluté récupérée sur
//! `N` étages, nombre d'étages requis (forme logarithmique) et débit de liquide
//! minimal pour une récupération fixée (pincement à l'entrée du gaz).
//!
//! ```text
//! facteur d'absorption   A   = L / (K·G)                                    [-]
//! facteur de stripping   S   = 1 / A                                        [-]
//! fraction absorbée      φ_A = (A^(N+1) − A) / (A^(N+1) − 1)      (A ≠ 1)    [-]
//! fraction absorbée      φ_A = N / (N + 1)                        (A = 1)    [-]
//! étages requis          N   = ln[ (Δ_in/Δ_out)·(1 − 1/A) + 1/A ] / ln(A)   [-]
//!   avec Δ_in = y_in − y*,  Δ_out = y_out − y*,  y* = K·x_in
//! liquide minimal        Lₘ  = G·K·φ_A                            (pincement)[mol·s⁻¹]
//! ```
//!
//! `A` facteur d'absorption [sans dimension], `S` facteur de stripping [sans
//! dimension], `L` débit molaire de liquide (solvant) [mol·s⁻¹], `G` débit
//! molaire de gaz porteur [mol·s⁻¹], `K` constante d'équilibre de la droite
//! `y = K·x` [sans dimension], `N` nombre d'étages théoriques [sans dimension,
//! réel], `φ_A` fraction du soluté absorbée [sans dimension, 0 ≤ φ_A ≤ 1],
//! `y_in`/`y_out` fractions molaires du soluté dans le gaz à l'entrée/à la
//! sortie [sans dimension], `y* = K·x_in` composition du gaz en équilibre avec
//! le liquide entrant [sans dimension], `Lₘ` débit molaire de liquide minimal
//! [mol·s⁻¹]. Toute paire de débits molaires dans la **même unité** convient
//! (les rapports `A`, `S`, `φ_A` sont sans dimension).
//!
//! **Limite honnête** : la méthode de **Kremser** décrit une **cascade à
//! contre-courant d'étages théoriques à l'équilibre** en **solution diluée**
//! (débits molaires `L` et `G` supposés **constants** le long de la colonne).
//! La **droite d'équilibre `y = K·x` de pente `K` constante** et les **débits**
//! sont **FOURNIS par l'appelant** — jamais inventés : `K` provient de la loi
//! de Henry, de tables VLE ou d'essais ; les débits d'un bilan de matière. Les
//! **coefficients de transfert de matière, diffusivités et corrélations** ne
//! sont **pas** l'objet de ce module (approche par étages, non par unités de
//! transfert) ; le **rendement d'étage réel** (Murphree/global), **fourni**,
//! convertit les étages théoriques en étages réels. Le **débit minimal**
//! suppose le **pincement à l'entrée du gaz** avec **solvant pur** (`x_in = 0`).
//! Hypothèse **isotherme** (enthalpies de dissolution négligées). Ce module
//! **complète** l'absorption sans la dupliquer : il expose les facteurs `A`/`S`
//! et l'inversion `N`(récupération), au niveau opération unitaire.

/// Facteur d'absorption `A = L / (K·G)` (sans dimension).
///
/// `liquid_flow` (L) et `gas_flow` (G) débits molaires dans une **même unité**
/// (p. ex. mol·s⁻¹) ; `equilibrium_constant` (K) pente de la droite d'équilibre
/// `y = K·x` [sans dimension]. `A > 1` favorise l'absorption, `A < 1` le
/// stripping.
///
/// Panique si `gas_flow <= 0`, `equilibrium_constant <= 0` ou `liquid_flow < 0`.
pub fn krem_absorption_factor(liquid_flow: f64, gas_flow: f64, equilibrium_constant: f64) -> f64 {
    assert!(
        gas_flow > 0.0 && equilibrium_constant > 0.0 && liquid_flow >= 0.0,
        "G > 0, K > 0 et L ≥ 0 requis"
    );
    liquid_flow / (equilibrium_constant * gas_flow)
}

/// Facteur de stripping `S = 1 / A` (sans dimension), réciproque du facteur
/// d'absorption.
///
/// `absorption_factor` (A) facteur d'absorption [sans dimension].
///
/// Panique si `absorption_factor <= 0`.
pub fn krem_stripping_factor(absorption_factor: f64) -> f64 {
    assert!(absorption_factor > 0.0, "A > 0 requis");
    1.0 / absorption_factor
}

/// Fraction de soluté absorbée sur `N` étages théoriques (équation de
/// **Kremser**) :
/// `φ_A = (A^(N+1) − A) / (A^(N+1) − 1)` pour `A ≠ 1`, et la limite
/// `φ_A = N / (N + 1)` pour `A = 1` (droite d'équilibre parallèle à
/// l'opérante). La fraction est rapportée au maximum absorbable `y_in − y*`.
///
/// `absorption_factor` (A) sans dimension ; `stages` (N) nombre d'étages
/// théoriques réel (≥ 0). Pour `A > 1`, `φ_A → 1` quand `N → ∞`.
///
/// Panique si `absorption_factor <= 0` ou `stages < 0`.
pub fn krem_fraction_absorbed(absorption_factor: f64, stages: f64) -> f64 {
    assert!(absorption_factor > 0.0, "A > 0 requis");
    assert!(stages >= 0.0, "N ≥ 0 étage requis");
    if (absorption_factor - 1.0).abs() < 1.0e-9
    {
        stages / (stages + 1.0)
    }
    else
    {
        let power = absorption_factor.powf(stages + 1.0);
        (power - absorption_factor) / (power - 1.0)
    }
}

/// Nombre d'étages théoriques requis (forme **logarithmique** de Kremser) :
/// `N = ln[ (Δ_in/Δ_out)·(1 − 1/A) + 1/A ] / ln(A)`
/// avec `Δ_in = y_in − y*`, `Δ_out = y_out − y*` et `y* = K·x_in` la
/// composition du gaz en équilibre avec le liquide entrant.
///
/// `absorption_factor` (A) sans dimension ; `inlet_ratio` (y_in) et
/// `outlet_ratio` (y_out) fractions molaires du soluté dans le gaz à
/// l'entrée/à la sortie [sans dimension] ; `equilibrium_ratio` (y* = K·x_in)
/// composition du gaz en équilibre avec le liquide entrant [sans dimension]
/// (0 pour un solvant pur). C'est l'inverse exact de [`krem_fraction_absorbed`].
///
/// Panique si `absorption_factor <= 0`, si `absorption_factor ≈ 1` (ln A
/// singulier), si `inlet_ratio − equilibrium_ratio <= 0` ou
/// `outlet_ratio − equilibrium_ratio <= 0` (forces motrices non positives), si
/// `outlet_ratio > inlet_ratio` (le gaz doit céder du soluté), ou si
/// l'argument du logarithme est `<= 0` (récupération infaisable pour ce `A`).
pub fn krem_stages_required(
    absorption_factor: f64,
    inlet_ratio: f64,
    outlet_ratio: f64,
    equilibrium_ratio: f64,
) -> f64 {
    assert!(absorption_factor > 0.0, "A > 0 requis");
    assert!(
        (absorption_factor - 1.0).abs() > 1.0e-9,
        "A ≠ 1 requis (ln A singulier en A = 1)"
    );
    let driving_in = inlet_ratio - equilibrium_ratio;
    let driving_out = outlet_ratio - equilibrium_ratio;
    assert!(
        driving_in > 0.0 && driving_out > 0.0,
        "y_in − y* > 0 et y_out − y* > 0 requis (forces motrices positives)"
    );
    assert!(
        inlet_ratio >= outlet_ratio,
        "y_in ≥ y_out requis (le gaz cède du soluté)"
    );
    let inv_a = 1.0 / absorption_factor;
    let argument = (driving_in / driving_out) * (1.0 - inv_a) + inv_a;
    assert!(
        argument > 0.0,
        "argument du logarithme > 0 requis (A trop faible pour cette récupération)"
    );
    argument.ln() / absorption_factor.ln()
}

/// Débit molaire de liquide **minimal** pour une récupération fixée
/// `Lₘ = G·K·φ_A` (mol·s⁻¹), correspondant au **pincement à l'entrée du gaz**
/// avec **solvant pur** (`x_in = 0`) et un nombre d'étages infini : le facteur
/// d'absorption minimal vaut alors `A_min = φ_A`, d'où `Lₘ = A_min·K·G`.
///
/// `gas_flow` (G) débit molaire de gaz [mol·s⁻¹] ; `equilibrium_constant` (K)
/// pente de la droite d'équilibre [sans dimension] ; `recovery_fraction` (φ_A)
/// fraction de soluté à récupérer [sans dimension, 0 < φ_A ≤ 1]. Un débit réel
/// `L > Lₘ` (facteur `A > A_min`) est ensuite **fourni** pour dimensionner un
/// nombre fini d'étages.
///
/// Panique si `gas_flow <= 0`, `equilibrium_constant <= 0`, ou si
/// `recovery_fraction` n'est pas dans `]0, 1]`.
pub fn krem_minimum_liquid(
    gas_flow: f64,
    equilibrium_constant: f64,
    recovery_fraction: f64,
) -> f64 {
    assert!(
        gas_flow > 0.0 && equilibrium_constant > 0.0,
        "G > 0 et K > 0 requis"
    );
    assert!(
        recovery_fraction > 0.0 && recovery_fraction <= 1.0,
        "récupération φ_A dans ]0, 1] requise"
    );
    gas_flow * equilibrium_constant * recovery_fraction
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn factor_and_stripping_are_reciprocal() {
        // L = 120, G = 40, K = 1.5 ⇒ A = 120/(1.5·40) = 120/60 = 2.
        let a = krem_absorption_factor(120.0_f64, 40.0_f64, 1.5_f64);
        assert_relative_eq!(a, 2.0, max_relative = 1e-12);
        // S = 1/A et A·S = 1 (réciprocité).
        let s = krem_stripping_factor(a);
        assert_relative_eq!(s, 0.5, max_relative = 1e-12);
        assert_relative_eq!(a * s, 1.0, max_relative = 1e-12);
    }

    #[test]
    fn fraction_absorbed_unit_factor_limit() {
        // A = 1 : limite φ_A = N/(N+1). Pour N = 4 ⇒ 4/5 = 0.8.
        assert_relative_eq!(
            krem_fraction_absorbed(1.0_f64, 4.0_f64),
            0.8,
            max_relative = 1e-12
        );
        // Continuité : A très proche de 1 redonne la même limite.
        assert_relative_eq!(
            krem_fraction_absorbed(1.0_f64 + 1e-11, 4.0_f64),
            0.8,
            max_relative = 1e-6
        );
    }

    #[test]
    fn fraction_absorbed_numeric_case() {
        // A = 2, N = 3 : A^(N+1) = 2^4 = 16 ; φ_A = (16 − 2)/(16 − 1) = 14/15.
        assert_relative_eq!(
            krem_fraction_absorbed(2.0_f64, 3.0_f64),
            14.0_f64 / 15.0_f64,
            max_relative = 1e-12
        );
        // A > 1 : un très grand N absorbe la quasi-totalité (φ_A → 1, sans l'atteindre).
        let phi = krem_fraction_absorbed(2.0_f64, 40.0_f64);
        assert!(phi < 1.0, "φ_A reste < 1 pour N fini");
        assert_relative_eq!(phi, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn stages_required_and_fraction_are_reciprocal() {
        // Solvant pur (y* = 0), A = 2, y_in = 0.1, y_out = 0.01.
        // Fraction absorbée attendue φ_A = (y_in − y_out)/(y_in − y*) = 0.09/0.1 = 0.9.
        let n = krem_stages_required(2.0_f64, 0.1_f64, 0.01_f64, 0.0_f64);
        let phi = krem_fraction_absorbed(2.0_f64, n);
        assert_relative_eq!(phi, 0.9, max_relative = 1e-9);
    }

    #[test]
    fn stages_required_numeric_case() {
        // A = 2, y_in = 0.1, y_out = 0.01, y* = 0 :
        // argument = (0.1/0.01)·(1 − 0.5) + 0.5 = 10·0.5 + 0.5 = 5.5.
        // N = ln(5.5)/ln(2). ln(5.5) = 1.7047480922384254, ln(2) = 0.6931471805599453,
        // N = 1.7047480922384254 / 0.6931471805599453 = 2.4594316186372978.
        assert_relative_eq!(
            krem_stages_required(2.0_f64, 0.1_f64, 0.01_f64, 0.0_f64),
            2.4594316186372978,
            max_relative = 1e-9
        );
    }

    #[test]
    fn minimum_liquid_numeric_and_proportionality() {
        // G = 10, K = 2, φ_A = 0.9 ⇒ Lₘ = 10·2·0.9 = 18.
        let lmin = krem_minimum_liquid(10.0_f64, 2.0_f64, 0.9_f64);
        assert_relative_eq!(lmin, 18.0, max_relative = 1e-12);
        // Récupération totale (φ_A = 1) ⇒ Lₘ = G·K, soit A_min = 1 exactement.
        let lfull = krem_minimum_liquid(10.0_f64, 2.0_f64, 1.0_f64);
        assert_relative_eq!(lfull, 20.0, max_relative = 1e-12);
        assert_relative_eq!(
            krem_absorption_factor(lfull, 10.0_f64, 2.0_f64),
            1.0,
            max_relative = 1e-12
        );
        // Proportionnalité en G : doubler le gaz double Lₘ.
        assert_relative_eq!(
            krem_minimum_liquid(20.0_f64, 2.0_f64, 0.9_f64),
            2.0 * lmin,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "A ≠ 1 requis")]
    fn stages_required_panics_at_unit_factor() {
        // A = 1 rend ln A = 0 : la forme logarithmique est singulière ⇒ panique.
        let _ = krem_stages_required(1.0_f64, 0.1_f64, 0.01_f64, 0.0_f64);
    }
}
