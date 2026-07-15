//! Règle des bras de levier sur un **diagramme de phases binaire** — **fraction
//! de phase**, **fraction complémentaire** de l'autre phase et **longueur de la
//! conode** (tie line) reliant les deux phases à l'équilibre.
//!
//! ```text
//! fraction de la phase 2   W2 = (C0 − C1)/(C2 − C1)
//! fraction complémentaire  W1 = 1 − W2   (= (C2 − C0)/(C2 − C1))
//! longueur de la conode    Lc = |C2 − C1|
//! ```
//!
//! `C0` composition globale de l'alliage, `C1` composition de la phase 1
//! (extrémité de la conode), `C2` composition de la phase 2 (autre extrémité),
//! `W2` fraction (massique ou molaire) de la phase 2, `W1` fraction de la
//! phase 1, `Lc` longueur de la conode. Toutes les compositions sont exprimées
//! dans **la même unité** au choix de l'appelant (fraction ∈ [0, 1] ou
//! pourcentage massique/atomique) ; `W1`, `W2` sont adimensionnels.
//!
//! **Convention** : compositions homogènes entre elles (même grandeur, même
//! unité), `C0` compris entre `C1` et `C2`. **Limite honnête** : système
//! binaire **à l'équilibre thermodynamique**, à la température de la conode ;
//! les compositions des phases `C1` et `C2` (extrémités de la conode) sont
//! **lues sur le diagramme de phases** et **fournies par l'appelant** — aucune
//! valeur « par défaut » de matériau ou de procédé n'est inventée ici. La règle
//! des bras de levier suppose l'équilibre atteint (elle ne dit rien de la
//! cinétique de transformation ni des états hors équilibre).

/// Fraction de la phase 2 par la règle des bras de levier
/// `W2 = (C0 − C1)/(C2 − C1)` (adimensionnel).
///
/// `overall_composition` = `C0`, `phase1_composition` = `C1`,
/// `phase2_composition` = `C2` (mêmes grandeur et unité). Le bras de levier
/// pris est celui **opposé** à la phase 2 (segment `C0`–`C1`) rapporté à la
/// longueur totale de la conode.
///
/// Panique si `C1 == C2` (conode dégénérée, division par zéro) ou si `C0`
/// n'est pas compris entre `C1` et `C2`.
pub fn lever_phase_fraction(
    overall_composition: f64,
    phase1_composition: f64,
    phase2_composition: f64,
) -> f64 {
    assert!(
        phase1_composition != phase2_composition,
        "les compositions des deux phases doivent être distinctes (conode non dégénérée)"
    );
    let lower = phase1_composition.min(phase2_composition);
    let upper = phase1_composition.max(phase2_composition);
    assert!(
        overall_composition >= lower && overall_composition <= upper,
        "la composition globale C0 doit être comprise entre C1 et C2"
    );
    (overall_composition - phase1_composition) / (phase2_composition - phase1_composition)
}

/// Fraction complémentaire de l'autre phase `W1 = 1 − W2` (adimensionnel).
///
/// Les fractions des deux phases d'un système binaire à l'équilibre somment à
/// l'unité ; cette fonction renvoie donc la fraction de la phase restante.
///
/// Panique si `fraction ∉ [0, 1]`.
pub fn lever_complementary_fraction(fraction: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&fraction),
        "la fraction de phase doit être dans [0, 1]"
    );
    1.0 - fraction
}

/// Longueur de la conode `Lc = |C2 − C1|` (même unité que les compositions).
///
/// Distance entre les deux extrémités de la conode (tie line) sur l'axe des
/// compositions, à la température considérée.
///
/// Panique si `C1 == C2` (conode dégénérée, longueur nulle).
pub fn lever_tie_line_length(phase1_composition: f64, phase2_composition: f64) -> f64 {
    assert!(
        phase1_composition != phase2_composition,
        "les compositions des deux phases doivent être distinctes (conode non dégénérée)"
    );
    (phase2_composition - phase1_composition).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn fraction_at_endpoints() {
        // C0 = C1 → toute la matière est en phase 1, donc W2 = 0.
        assert_relative_eq!(lever_phase_fraction(10.0, 10.0, 60.0), 0.0, epsilon = 1e-15);
        // C0 = C2 → toute la matière est en phase 2, donc W2 = 1.
        assert_relative_eq!(lever_phase_fraction(60.0, 10.0, 60.0), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn fraction_realistic_value() {
        // Conode C1 = 10 %, C2 = 60 %, alliage C0 = 40 %.
        // W2 = (40 − 10)/(60 − 10) = 30/50 = 0,6.
        let w2 = lever_phase_fraction(40.0, 10.0, 60.0);
        assert_relative_eq!(w2, 0.6, epsilon = 1e-15);
    }

    #[test]
    fn swapping_phases_gives_complement() {
        // Identité : échanger les rôles des deux phases donne la fraction
        // complémentaire, W2(C0,C1,C2) + W2(C0,C2,C1) = 1.
        let (c0, c1, c2) = (40.0_f64, 10.0_f64, 60.0_f64);
        let w2 = lever_phase_fraction(c0, c1, c2);
        let w1 = lever_phase_fraction(c0, c2, c1);
        assert_relative_eq!(w1 + w2, 1.0, epsilon = 1e-15);
        // et la fonction complémentaire reproduit ce même échange.
        assert_relative_eq!(lever_complementary_fraction(w2), w1, epsilon = 1e-15);
    }

    #[test]
    fn midpoint_gives_half() {
        // Proportionnalité : C0 au milieu de la conode → W2 = 0,5.
        let (c1, c2) = (10.0_f64, 60.0_f64);
        let mid = 0.5 * (c1 + c2);
        assert_relative_eq!(lever_phase_fraction(mid, c1, c2), 0.5, epsilon = 1e-15);
    }

    #[test]
    fn tie_line_length_is_symmetric() {
        // La longueur de la conode ne dépend pas de l'ordre des extrémités
        // et vaut |C2 − C1|.
        let (c1, c2) = (10.0_f64, 60.0_f64);
        assert_relative_eq!(lever_tie_line_length(c1, c2), 50.0, epsilon = 1e-15);
        assert_relative_eq!(
            lever_tie_line_length(c2, c1),
            lever_tie_line_length(c1, c2),
            epsilon = 1e-15
        );
    }

    #[test]
    fn complementary_is_involutive() {
        // Appliquer deux fois la complémentation redonne la valeur initiale.
        let w2 = 0.6_f64;
        assert_relative_eq!(
            lever_complementary_fraction(lever_complementary_fraction(w2)),
            w2,
            epsilon = 1e-15
        );
    }

    #[test]
    #[should_panic(expected = "comprise entre C1 et C2")]
    fn overall_outside_tie_line_panics() {
        lever_phase_fraction(70.0, 10.0, 60.0);
    }
}
