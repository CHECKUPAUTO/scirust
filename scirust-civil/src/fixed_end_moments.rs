//! **Moments d'encastrement parfait** (« fixed-end moments ») pour une **poutre
//! bi-encastrée prismatique** sous les cas de charge usuels de la résistance des
//! matériaux : charge **uniformément répartie**, charge **concentrée au milieu**,
//! charge **concentrée quelconque**, charge **triangulaire** et **tassement
//! d'appui différentiel**. Ces moments constituent les termes de départ de la
//! **méthode de Cross** (distribution des moments) et de la méthode des
//! déplacements.
//!
//! ```text
//! charge répartie          |M| = w·L² / 12
//! charge au milieu         |M| = P·L / 8
//! charge quelconque        Mab = P·a·b² / L²   (appui côté a)
//! charge triangulaire      |M| = w·L² / 30   (extrémité chargée)
//!                          |M| = w·L² / 20   (extrémité déchargée)
//! tassement d'appui        |M| = 6·E·I·δ / L²
//! ```
//!
//! `M` moment d'encastrement à un appui (N·m), `w` charge linéique
//! (N/m — valeur de crête pour la charge triangulaire), `L` portée de la poutre
//! (m), `P` charge concentrée (N), `a` distance de la charge à l'appui considéré
//! (m), `b` distance de la charge à l'appui opposé (m, avec `a + b = L`), `E`
//! module d'élasticité longitudinal du matériau (Pa), `I` moment d'inertie de
//! flexion de la section (m⁴), `δ` tassement d'appui différentiel (m).
//!
//! **Convention** : **SI cohérent** — charges linéiques en **N/m**, charges
//! concentrées en **N**, longueurs en **m**, module `E` en **Pa**, inertie `I`
//! en **m⁴**, tassement `δ` en **m** ; tous les moments sont alors renvoyés en
//! **N·m**. Les fonctions renvoient la **valeur absolue** du moment
//! d'encastrement : c'est à l'appelant d'affecter le **signe** selon la
//! convention adoptée (par exemple moments positifs comprimant la fibre
//! supérieure) et selon le sens du tassement. Types `f64`.
//!
//! **Limite honnête** : **valeurs classiques de la RDM** pour une poutre
//! **bi-encastrée prismatique** (section et rigidité `E·I` constantes) en
//! **comportement élastique linéaire**, petites déformations. Les **charges**
//! (`w`, `P`), leurs **positions** (`a`, `b`), la **portée** `L`, la **rigidité**
//! (`E`, `I`) et le **tassement** `δ` sont **fournis par l'appelant** — aucune
//! valeur « par défaut » n'est inventée. Ces moments servent d'**entrée** à la
//! méthode de Cross ; ce module **ne réalise pas** l'équilibrage aux nœuds ni la
//! distribution des moments. Les effets du **second ordre**, du **cisaillement
//! (poutre de Timoshenko)**, de la **non-prismaticité** et des **appuis
//! élastiques** ne sont pas traités.

/// Moment d'encastrement parfait d'une **charge uniformément répartie** sur toute
/// la portée d'une poutre bi-encastrée : `|M| = w·L² / 12`.
///
/// `load_per_length` = `w` charge linéique (N/m), `span` = `L` portée (m) ;
/// renvoie la valeur absolue du moment d'encastrement (N·m), identique aux deux
/// appuis.
///
/// Panique si `load_per_length` est non fini, ou si `span` est non fini ou n'est
/// pas strictement positif (une portée physique est strictement positive).
pub fn fem_udl(load_per_length: f64, span: f64) -> f64 {
    assert!(
        load_per_length.is_finite(),
        "la charge linéique doit être finie"
    );
    assert!(
        span.is_finite() && span > 0.0,
        "la portée doit être finie et strictement positive"
    );
    load_per_length * span * span / 12.0
}

/// Moment d'encastrement parfait d'une **charge concentrée appliquée au milieu**
/// d'une poutre bi-encastrée : `|M| = P·L / 8`.
///
/// `point_load` = `P` charge concentrée (N), `span` = `L` portée (m) ; renvoie la
/// valeur absolue du moment d'encastrement (N·m), identique aux deux appuis.
///
/// Panique si `point_load` est non fini, ou si `span` est non fini ou n'est pas
/// strictement positif (une portée physique est strictement positive).
pub fn fem_point_center(point_load: f64, span: f64) -> f64 {
    assert!(
        point_load.is_finite(),
        "la charge concentrée doit être finie"
    );
    assert!(
        span.is_finite() && span > 0.0,
        "la portée doit être finie et strictement positive"
    );
    point_load * span / 8.0
}

/// Moment d'encastrement parfait d'une **charge concentrée en position
/// quelconque** sur une poutre bi-encastrée, à l'appui situé du côté de la
/// distance `a` : `Mab = P·a·b² / L²`.
///
/// `point_load` = `P` charge concentrée (N), `distance_a` = `a` distance de la
/// charge à l'appui considéré (m), `distance_b` = `b` distance de la charge à
/// l'appui opposé (m, avec `a + b = L`), `span` = `L` portée (m) ; renvoie la
/// valeur absolue du moment d'encastrement à cet appui (N·m). Le moment à
/// l'appui opposé s'obtient en permutant `a` et `b` : `Mba = P·a²·b / L²`.
///
/// Panique si `point_load` est non fini, si `distance_a` ou `distance_b` est non
/// fini ou négatif, ou si `span` est non fini ou n'est pas strictement positif
/// (les distances sont physiquement ≥ 0 et la portée strictement positive).
pub fn fem_point_general(point_load: f64, distance_a: f64, distance_b: f64, span: f64) -> f64 {
    assert!(
        point_load.is_finite(),
        "la charge concentrée doit être finie"
    );
    assert!(
        distance_a.is_finite() && distance_a >= 0.0,
        "la distance a doit être finie et positive ou nulle"
    );
    assert!(
        distance_b.is_finite() && distance_b >= 0.0,
        "la distance b doit être finie et positive ou nulle"
    );
    assert!(
        span.is_finite() && span > 0.0,
        "la portée doit être finie et strictement positive"
    );
    point_load * distance_a * distance_b * distance_b / (span * span)
}

/// Moment d'encastrement parfait d'une **charge triangulaire** dont l'intensité
/// varie linéairement de zéro à une valeur de crête `w`, sur une poutre
/// bi-encastrée. La fonction renvoie le moment à l'**extrémité chargée**
/// (côté crête) : `|M| = w·L² / 30`. À l'**extrémité déchargée** (côté nul), le
/// moment d'encastrement vaut `|M| = w·L² / 20` ; il s'obtient en multipliant le
/// résultat par `30/20 = 1,5`.
///
/// `peak_load_per_length` = `w` valeur de crête de la charge linéique (N/m),
/// `span` = `L` portée (m) ; renvoie la valeur absolue du moment d'encastrement à
/// l'extrémité chargée (N·m).
///
/// Panique si `peak_load_per_length` est non fini, ou si `span` est non fini ou
/// n'est pas strictement positif (une portée physique est strictement positive).
pub fn fem_triangular(peak_load_per_length: f64, span: f64) -> f64 {
    assert!(
        peak_load_per_length.is_finite(),
        "la charge de crête doit être finie"
    );
    assert!(
        span.is_finite() && span > 0.0,
        "la portée doit être finie et strictement positive"
    );
    peak_load_per_length * span * span / 30.0
}

/// Moment d'encastrement induit par un **tassement d'appui différentiel** `δ`
/// entre les deux appuis d'une poutre bi-encastrée prismatique :
/// `|M| = 6·E·I·δ / L²`.
///
/// `elastic_modulus` = `E` module d'élasticité longitudinal (Pa), `inertia` =
/// `I` moment d'inertie de flexion de la section (m⁴), `settlement` = `δ`
/// tassement différentiel (m), `span` = `L` portée (m) ; renvoie la valeur
/// absolue du moment d'encastrement engendré (N·m), identique aux deux appuis.
///
/// Panique si `settlement` est non fini, si `elastic_modulus` ou `inertia` est
/// non fini ou n'est pas strictement positif, ou si `span` est non fini ou n'est
/// pas strictement positif (module, inertie et portée physiquement strictement
/// positifs).
pub fn fem_support_settlement(
    elastic_modulus: f64,
    inertia: f64,
    settlement: f64,
    span: f64,
) -> f64 {
    assert!(
        elastic_modulus.is_finite() && elastic_modulus > 0.0,
        "le module d'élasticité doit être fini et strictement positif"
    );
    assert!(
        inertia.is_finite() && inertia > 0.0,
        "l'inertie doit être finie et strictement positive"
    );
    assert!(settlement.is_finite(), "le tassement doit être fini");
    assert!(
        span.is_finite() && span > 0.0,
        "la portée doit être finie et strictement positive"
    );
    6.0 * elastic_modulus * inertia * settlement / (span * span)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Proportionnalité de la charge répartie : le moment croît linéairement avec
    /// la charge et comme le carré de la portée. Doubler `w` double `|M|` ;
    /// doubler `L` quadruple `|M|`.
    #[test]
    fn udl_proportionality() {
        let base = fem_udl(10_000.0, 6.0);
        assert_relative_eq!(fem_udl(20_000.0, 6.0), 2.0 * base, epsilon = 1e-6);
        assert_relative_eq!(fem_udl(10_000.0, 12.0), 4.0 * base, epsilon = 1e-6);
    }

    /// Cas chiffré de la charge répartie : `w = 12 000 N/m`, `L = 5 m`.
    /// `|M| = 12 000 · 25 / 12 = 300 000 / 12 = 25 000 N·m`.
    #[test]
    fn udl_computed_case() {
        // Recalcul indépendant : 5^2 = 25 ; 12000 * 25 = 300000 ; /12 = 25000.
        assert_relative_eq!(fem_udl(12_000.0, 5.0), 25_000.0, epsilon = 1e-6);
    }

    /// Cas chiffré de la charge concentrée au milieu : `P = 40 000 N`, `L = 8 m`.
    /// `|M| = 40 000 · 8 / 8 = 40 000 N·m`.
    #[test]
    fn point_center_computed_case() {
        // Recalcul indépendant : 40000 * 8 = 320000 ; /8 = 40000.
        assert_relative_eq!(fem_point_center(40_000.0, 8.0), 40_000.0, epsilon = 1e-6);
    }

    /// Cohérence de la charge quelconque avec le cas milieu : pour `a = b = L/2`,
    /// `Mab = P·(L/2)·(L/2)²/L² = P·L/8`, qui coïncide avec `fem_point_center`.
    #[test]
    fn point_general_matches_center_at_midspan() {
        let span = 8.0;
        let load = 40_000.0;
        assert_relative_eq!(
            fem_point_general(load, span / 2.0, span / 2.0, span),
            fem_point_center(load, span),
            epsilon = 1e-6
        );
    }

    /// Cas chiffré de la charge quelconque : `P = 60 000 N`, `a = 2 m`,
    /// `b = 4 m`, `L = 6 m`. `Mab = 60 000 · 2 · 16 / 36 = 1 920 000 / 36 =
    /// 53 333,333… N·m`. On vérifie aussi que `Mab + Mba = P·a·b/L` (relation
    /// classique) : `Mba = 60 000 · 4 · 4 / 36 = 26 666,666… N·m`, et
    /// `Mab + Mba = 80 000 = P·a·b/L = 60 000·2·4/6`.
    #[test]
    fn point_general_computed_case_and_identity() {
        // Recalcul indépendant : b^2 = 16 ; 60000*2*16 = 1_920_000 ; /36 = 53333.3333...
        let m_ab = fem_point_general(60_000.0, 2.0, 4.0, 6.0);
        assert_relative_eq!(m_ab, 1_920_000.0 / 36.0, epsilon = 1e-3);
        // Somme des deux moments d'encastrement = P·a·b/L.
        let m_ba = fem_point_general(60_000.0, 4.0, 2.0, 6.0);
        assert_relative_eq!(m_ab + m_ba, 60_000.0 * 2.0 * 4.0 / 6.0, epsilon = 1e-3);
    }

    /// Rapport des deux moments de la charge triangulaire : l'extrémité déchargée
    /// (`w·L²/20`) porte un moment `30/20 = 1,5` fois celui de l'extrémité chargée
    /// (`w·L²/30`). Cas chiffré : `w = 9 000 N/m`, `L = 10 m` ⇒
    /// `|M| = 9 000 · 100 / 30 = 30 000 N·m` à l'extrémité chargée.
    #[test]
    fn triangular_ratio_and_computed_case() {
        // Recalcul indépendant : 10^2 = 100 ; 9000 * 100 = 900000 ; /30 = 30000.
        let loaded = fem_triangular(9_000.0, 10.0);
        assert_relative_eq!(loaded, 30_000.0, epsilon = 1e-6);
        // Moment à l'extrémité déchargée = w·L²/20 = 1,5 · (w·L²/30).
        let unloaded = 9_000.0 * 10.0 * 10.0 / 20.0;
        assert_relative_eq!(unloaded / loaded, 1.5, epsilon = 1e-9);
    }

    /// Cas chiffré du tassement d'appui : `E = 200 GPa = 2,0e11 Pa`,
    /// `I = 8,0e-5 m⁴`, `δ = 0,010 m`, `L = 4 m`.
    /// `|M| = 6 · 2,0e11 · 8,0e-5 · 0,010 / 16 = 960 000 / 16 = 60 000 N·m`.
    #[test]
    fn support_settlement_computed_case() {
        // Recalcul indépendant : 6 * 2e11 = 1.2e12 ; *8e-5 = 9.6e7 ; *0.01 = 9.6e5.
        // L^2 = 16 ; 960000 / 16 = 60000.
        let m = fem_support_settlement(2.0e11, 8.0e-5, 0.010, 4.0);
        assert_relative_eq!(m, 60_000.0, epsilon = 1e-3);
    }

    /// Une portée nulle est physiquement inadmissible et doit être rejetée.
    #[test]
    #[should_panic(expected = "la portée doit être finie et strictement positive")]
    fn udl_rejects_zero_span() {
        let _ = fem_udl(10_000.0, 0.0);
    }
}
