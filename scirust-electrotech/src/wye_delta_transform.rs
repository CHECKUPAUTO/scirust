//! **Transformation étoile–triangle (Kennelly)** — conversion des trois
//! résistances d'un réseau résistif à trois bornes entre sa configuration en
//! étoile (Y) et sa configuration en triangle (Δ), y compris le cas équilibré.
//!
//! ```text
//! triangle → étoile   R1 = Rab·Rca / (Rab + Rbc + Rca)
//!                     R2 = Rab·Rbc / (Rab + Rbc + Rca)
//!                     R3 = Rbc·Rca / (Rab + Rbc + Rca)
//!
//! étoile → triangle   Rab = (R1·R2 + R2·R3 + R3·R1) / R3
//!                     Rbc = (R1·R2 + R2·R3 + R3·R1) / R1
//!                     Rca = (R1·R2 + R2·R3 + R3·R1) / R2
//!
//! cas équilibré       Ry = RΔ / 3      RΔ = 3·Ry
//! ```
//!
//! Réseau à trois bornes A, B, C. Côtés du triangle : `Rab` (Ω) entre A et B,
//! `Rbc` (Ω) entre B et C, `Rca` (Ω) entre C et A. Branches de l'étoile :
//! `R1` (Ω) reliant le nœud A au point neutre, `R2` (Ω) reliant B au neutre,
//! `R3` (Ω) reliant C au neutre. Le paramètre `ra_delta` désigne `Rab`,
//! `rb_delta` désigne `Rbc`, `rc_delta` désigne `Rca` ; `r1_wye`, `r2_wye`,
//! `r3_wye` désignent `R1`, `R2`, `R3`. `RΔ` (Ω) résistance d'un côté du
//! triangle équilibré, `Ry` (Ω) résistance d'une branche de l'étoile
//! équilibrée. Toutes les grandeurs sont des résistances en ohms (f64).
//!
//! **Convention** : SI ; toutes les résistances en Ω. Ordre des triplets
//! renvoyés : `ydelta_delta_to_wye` renvoie `(R1, R2, R3)` (branches aux nœuds
//! A, B, C) ; `ydelta_wye_to_delta` renvoie `(Rab, Rbc, Rca)` (côtés A-B, B-C,
//! C-A). **Limite honnête** : transformation **exacte** de Kennelly pour les
//! réseaux **purement résistifs** à trois bornes (elle s'applique aussi aux
//! impédances complexes, mais ce module reste en arithmétique **réelle** f64) ;
//! elle préserve la résistance vue de chaque paire de bornes mais **ne conserve
//! pas** les tensions/courants internes du nœud neutre. Les valeurs des
//! résistances sont **fournies par l'appelant** (schéma, mesures) — aucune
//! valeur n'est inventée. Le cas équilibré donne le facteur 3.

/// Conversion triangle → étoile : renvoie `(R1, R2, R3)` (Ω), les trois
/// résistances de branche de l'étoile équivalente aux nœuds A, B, C.
///
/// `R1 = Rab·Rca/S`, `R2 = Rab·Rbc/S`, `R3 = Rbc·Rca/S` avec
/// `S = Rab + Rbc + Rca` et `ra_delta = Rab`, `rb_delta = Rbc`, `rc_delta = Rca`.
///
/// Panique si `ra_delta < 0`, `rb_delta < 0`, `rc_delta < 0`, ou si la somme
/// `ra_delta + rb_delta + rc_delta <= 0` (division par zéro).
pub fn ydelta_delta_to_wye(ra_delta: f64, rb_delta: f64, rc_delta: f64) -> (f64, f64, f64) {
    assert!(ra_delta >= 0.0, "la résistance Rab doit être ≥ 0");
    assert!(rb_delta >= 0.0, "la résistance Rbc doit être ≥ 0");
    assert!(rc_delta >= 0.0, "la résistance Rca doit être ≥ 0");
    let sum = ra_delta + rb_delta + rc_delta;
    assert!(
        sum > 0.0,
        "la somme des résistances du triangle doit être strictement positive"
    );
    let r1_wye = ra_delta * rc_delta / sum;
    let r2_wye = ra_delta * rb_delta / sum;
    let r3_wye = rb_delta * rc_delta / sum;
    (r1_wye, r2_wye, r3_wye)
}

/// Conversion étoile → triangle : renvoie `(Rab, Rbc, Rca)` (Ω), les trois
/// résistances de côté du triangle équivalent (côtés A-B, B-C, C-A).
///
/// `Rab = P/R3`, `Rbc = P/R1`, `Rca = P/R2` avec
/// `P = R1·R2 + R2·R3 + R3·R1` et `r1_wye = R1`, `r2_wye = R2`, `r3_wye = R3`.
///
/// Panique si `r1_wye <= 0`, `r2_wye <= 0` ou `r3_wye <= 0` (division par zéro).
pub fn ydelta_wye_to_delta(r1_wye: f64, r2_wye: f64, r3_wye: f64) -> (f64, f64, f64) {
    assert!(
        r1_wye > 0.0,
        "la résistance de branche R1 doit être strictement positive"
    );
    assert!(
        r2_wye > 0.0,
        "la résistance de branche R2 doit être strictement positive"
    );
    assert!(
        r3_wye > 0.0,
        "la résistance de branche R3 doit être strictement positive"
    );
    let sum_of_products = r1_wye * r2_wye + r2_wye * r3_wye + r3_wye * r1_wye;
    let ra_delta = sum_of_products / r3_wye;
    let rb_delta = sum_of_products / r1_wye;
    let rc_delta = sum_of_products / r2_wye;
    (ra_delta, rb_delta, rc_delta)
}

/// Cas équilibré triangle → étoile : `Ry = RΔ / 3` (Ω).
///
/// Panique si `delta_resistance < 0`.
pub fn ydelta_balanced_delta_to_wye(delta_resistance: f64) -> f64 {
    assert!(
        delta_resistance >= 0.0,
        "la résistance de triangle RΔ doit être ≥ 0"
    );
    delta_resistance / 3.0
}

/// Cas équilibré étoile → triangle : `RΔ = 3·Ry` (Ω).
///
/// Panique si `wye_resistance < 0`.
pub fn ydelta_balanced_wye_to_delta(wye_resistance: f64) -> f64 {
    assert!(
        wye_resistance >= 0.0,
        "la résistance d'étoile Ry doit être ≥ 0"
    );
    3.0 * wye_resistance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn balanced_factor_three_reciprocity() {
        // Cas équilibré : RΔ = 30 Ω ⇒ Ry = 30/3 = 10 Ω, et réciproquement
        // 3·10 = 30 Ω. Les deux fonctions équilibrées sont inverses l'une de
        // l'autre.
        let ry = ydelta_balanced_delta_to_wye(30.0);
        assert_relative_eq!(ry, 10.0, epsilon = 1e-12);
        assert_relative_eq!(ydelta_balanced_wye_to_delta(ry), 30.0, epsilon = 1e-12);
    }

    #[test]
    fn balanced_matches_general_symmetric_case() {
        // Cohérence : sur un triangle symétrique (Rab = Rbc = Rca = R), la
        // formule générale doit redonner exactement R/3 sur chaque branche.
        let r = 45.0_f64;
        let (r1, r2, r3) = ydelta_delta_to_wye(r, r, r);
        let expected = ydelta_balanced_delta_to_wye(r);
        assert_relative_eq!(r1, expected, epsilon = 1e-12);
        assert_relative_eq!(r2, expected, epsilon = 1e-12);
        assert_relative_eq!(r3, expected, epsilon = 1e-12);
    }

    #[test]
    fn round_trip_delta_wye_delta() {
        // Réciprocité de Kennelly : Δ → Y → Δ redonne le triangle initial.
        // Triangle de départ : Rab = 10, Rbc = 20, Rca = 30 Ω.
        let (rab0, rbc0, rca0) = (10.0_f64, 20.0_f64, 30.0_f64);
        let (r1, r2, r3) = ydelta_delta_to_wye(rab0, rbc0, rca0);
        let (rab, rbc, rca) = ydelta_wye_to_delta(r1, r2, r3);
        assert_relative_eq!(rab, rab0, epsilon = 1e-9);
        assert_relative_eq!(rbc, rbc0, epsilon = 1e-9);
        assert_relative_eq!(rca, rca0, epsilon = 1e-9);
    }

    #[test]
    fn realistic_delta_to_wye_case() {
        // Cas chiffré réaliste, triangle Rab = 10, Rbc = 20, Rca = 30 Ω :
        //   S  = 10 + 20 + 30 = 60 Ω
        //   R1 = Rab·Rca / S = 10·30 / 60 = 300/60 = 5 Ω
        //   R2 = Rab·Rbc / S = 10·20 / 60 = 200/60 = 3,333 333… Ω
        //   R3 = Rbc·Rca / S = 20·30 / 60 = 600/60 = 10 Ω
        let (r1, r2, r3) = ydelta_delta_to_wye(10.0, 20.0, 30.0);
        assert_relative_eq!(r1, 5.0, epsilon = 1e-9);
        assert_relative_eq!(r2, 10.0 / 3.0, epsilon = 1e-9);
        assert_relative_eq!(r3, 10.0, epsilon = 1e-9);
    }

    #[test]
    fn realistic_wye_to_delta_case() {
        // Cas chiffré réaliste, étoile R1 = 5, R2 = 10/3, R3 = 10 Ω :
        //   P   = R1·R2 + R2·R3 + R3·R1
        //       = 5·(10/3) + (10/3)·10 + 10·5
        //       = 50/3 + 100/3 + 50 = 150/3 + 50 = 50 + 50 = 100 Ω²
        //   Rab = P/R3 = 100/10 = 10 Ω
        //   Rbc = P/R1 = 100/5  = 20 Ω
        //   Rca = P/R2 = 100/(10/3) = 30 Ω
        let (rab, rbc, rca) = ydelta_wye_to_delta(5.0, 10.0 / 3.0, 10.0);
        assert_relative_eq!(rab, 10.0, epsilon = 1e-9);
        assert_relative_eq!(rbc, 20.0, epsilon = 1e-9);
        assert_relative_eq!(rca, 30.0, epsilon = 1e-9);
    }

    #[test]
    fn proportionality_of_delta_to_wye() {
        // Homogénéité : multiplier toutes les résistances du triangle par un
        // facteur k multiplie chaque branche de l'étoile par le même k
        // (les formules sont homogènes de degré 1).
        let k = 4.0_f64;
        let (r1, r2, r3) = ydelta_delta_to_wye(3.0, 6.0, 9.0);
        let (s1, s2, s3) = ydelta_delta_to_wye(k * 3.0, k * 6.0, k * 9.0);
        assert_relative_eq!(s1, k * r1, epsilon = 1e-9);
        assert_relative_eq!(s2, k * r2, epsilon = 1e-9);
        assert_relative_eq!(s3, k * r3, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "la résistance de branche R1 doit être strictement positive")]
    fn zero_wye_branch_panics() {
        ydelta_wye_to_delta(0.0, 5.0, 3.0);
    }
}
