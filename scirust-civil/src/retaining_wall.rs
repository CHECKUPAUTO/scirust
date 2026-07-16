//! **Géotechnique — stabilité d'un mur-poids de soutènement** : coefficients de
//! sécurité au **renversement** et au **glissement**, **excentricité** de la
//! résultante sur la base et **contrainte de sol maximale** sous la semelle
//! (répartition trapézoïdale), en équilibre statique 2D par mètre linéaire.
//!
//! ```text
//! sécurité renversement   Fr = Mstab / Mrenv
//! sécurité glissement     Fg = (V · μ) / H
//! excentricité résultante e  = B/2 − Mnet / V
//! contrainte sol maximale σmax = (V / B) · (1 + 6·e / B)
//! ```
//!
//! `Mstab` moment stabilisateur autour du pied aval (N·m par mètre de mur),
//! `Mrenv` moment de renversement autour du même pied (N·m/m), `Fr` sécurité au
//! renversement (sans dimension), `V` charge verticale totale reprise par la
//! base (N/m), `μ` coefficient de frottement base-sol (sans dimension), `H`
//! poussée horizontale résultante (N/m), `Fg` sécurité au glissement (sans
//! dimension), `B` largeur de la semelle (m), `Mnet` moment net des forces
//! autour du pied aval, `Mnet = Mstab − Mrenv` (N·m/m), `e` excentricité de la
//! résultante par rapport au centre de la base (m), `σmax` contrainte verticale
//! maximale du sol sous la semelle (Pa).
//!
//! **Convention** : **SI strict** — longueurs en **m**, forces en **N** (par
//! mètre linéaire de mur : N/m), moments en **N·m** (par mètre : N·m/m),
//! contraintes en **Pa** (1 Pa = 1 N/m²), coefficient de frottement sans
//! dimension. Types `f64`.
//!
//! **Limite honnête** : **mur-poids rigide**, **équilibre statique 2D par mètre
//! linéaire**. La **poussée horizontale** `H`, les **poids et moments**
//! (`V`, `Mstab`, `Mrenv`, `Mnet`) et le **coefficient de frottement base-sol**
//! `μ` sont des **données fournies par l'appelant** (la poussée se calcule avec
//! le module `earth_pressure`) — aucune valeur « par défaut » n'est inventée.
//! La contrainte de sol suppose une **résultante dans le tiers central**
//! (`e ≤ B/6`, diagramme trapézoïdal entièrement en compression) ; au-delà, la
//! semelle décolle et la formule ne s'applique plus. Ne sont **pas** vérifiées
//! la **stabilité générale** (grand glissement) ni la **butée aval**. Les
//! **résistances caractéristiques** géotechniques et les **coefficients partiels
//! de sécurité** (γφ, γγ, γc, γR… de l'Eurocode 7 / EN 1997 et de son Annexe
//! Nationale) sont appliqués par l'appelant sur les actions ou les résistances
//! selon l'approche de calcul retenue ; les sécurités renvoyées ici comparent
//! des grandeurs déjà pondérées ou non, au choix de l'appelant.

/// Coefficient de sécurité au renversement `Fr = Mstab / Mrenv` (sans
/// dimension), rapport du moment stabilisateur au moment de renversement autour
/// du pied aval du mur.
///
/// `stabilizing_moment` = `Mstab` moment stabilisateur (N·m/m),
/// `overturning_moment` = `Mrenv` moment de renversement (N·m/m) ; renvoie la
/// sécurité au renversement `Fr` (sans dimension). `Fr > 1` traduit un mur
/// stable vis-à-vis du basculement.
///
/// Panique si `stabilizing_moment < 0` ou si `overturning_moment <= 0` (moment
/// de renversement strictement positif requis, division).
pub fn retwall_overturning_safety(stabilizing_moment: f64, overturning_moment: f64) -> f64 {
    assert!(
        stabilizing_moment >= 0.0,
        "le moment stabilisateur Mstab doit être ≥ 0"
    );
    assert!(
        overturning_moment > 0.0,
        "le moment de renversement Mrenv doit être > 0 (division)"
    );
    stabilizing_moment / overturning_moment
}

/// Coefficient de sécurité au glissement `Fg = (V · μ) / H` (sans dimension),
/// rapport de la résistance au frottement mobilisable sur la base à la poussée
/// horizontale.
///
/// `vertical_load` = `V` charge verticale totale sur la base (N/m),
/// `base_friction_coefficient` = `μ` coefficient de frottement base-sol (sans
/// dimension), `horizontal_thrust` = `H` poussée horizontale résultante (N/m) ;
/// renvoie la sécurité au glissement `Fg` (sans dimension). `Fg > 1` traduit un
/// mur stable vis-à-vis du glissement sur sa base.
///
/// Panique si `vertical_load < 0`, si `base_friction_coefficient < 0` ou si
/// `horizontal_thrust <= 0` (poussée strictement positive requise, division).
pub fn retwall_sliding_safety(
    vertical_load: f64,
    base_friction_coefficient: f64,
    horizontal_thrust: f64,
) -> f64 {
    assert!(vertical_load >= 0.0, "la charge verticale V doit être ≥ 0");
    assert!(
        base_friction_coefficient >= 0.0,
        "le coefficient de frottement μ doit être ≥ 0"
    );
    assert!(
        horizontal_thrust > 0.0,
        "la poussée horizontale H doit être > 0 (division)"
    );
    vertical_load * base_friction_coefficient / horizontal_thrust
}

/// Contrainte de sol maximale sous la semelle `σmax = (V / B) · (1 + 6·e / B)`
/// (Pa), pointe de compression du diagramme trapézoïdal (au pied aval côté
/// excentricité).
///
/// `vertical_load` = `V` charge verticale totale (N/m), `base_width` = `B`
/// largeur de la semelle (m), `eccentricity` = `e` excentricité de la résultante
/// (m) ; renvoie la contrainte maximale `σmax` (Pa, car (N/m)/m = N/m² = Pa).
/// À `e = 0` la répartition est uniforme et `σmax = V/B`.
///
/// Panique si `vertical_load < 0`, si `base_width <= 0`, si `eccentricity < 0`
/// ou si `eccentricity > base_width / 6` (résultante hors du tiers central : la
/// semelle décolle et le diagramme trapézoïdal ne s'applique plus).
pub fn retwall_base_pressure_max(vertical_load: f64, base_width: f64, eccentricity: f64) -> f64 {
    assert!(vertical_load >= 0.0, "la charge verticale V doit être ≥ 0");
    assert!(base_width > 0.0, "la largeur de base B doit être > 0");
    assert!(eccentricity >= 0.0, "l'excentricité e doit être ≥ 0");
    assert!(
        eccentricity <= base_width / 6.0,
        "l'excentricité e doit être ≤ B/6 (résultante dans le tiers central)"
    );
    (vertical_load / base_width) * (1.0 + 6.0 * eccentricity / base_width)
}

/// Excentricité de la résultante par rapport au centre de la base
/// `e = B/2 − Mnet / V` (m), où `Mnet / V` est l'abscisse du point d'application
/// de la résultante mesurée depuis le pied aval.
///
/// `net_moment` = `Mnet` moment net des forces autour du pied aval (N·m/m,
/// typiquement `Mstab − Mrenv`), `vertical_load` = `V` charge verticale totale
/// (N/m), `base_width` = `B` largeur de la semelle (m) ; renvoie l'excentricité
/// `e` (m). Une valeur `e ≤ B/6` place la résultante dans le tiers central.
///
/// Panique si `net_moment < 0`, si `vertical_load <= 0` (division) ou si
/// `base_width <= 0` (grandeurs physiquement positives).
pub fn retwall_resultant_eccentricity(net_moment: f64, vertical_load: f64, base_width: f64) -> f64 {
    assert!(net_moment >= 0.0, "le moment net Mnet doit être ≥ 0");
    assert!(
        vertical_load > 0.0,
        "la charge verticale V doit être > 0 (division)"
    );
    assert!(base_width > 0.0, "la largeur de base B doit être > 0");
    base_width / 2.0 - net_moment / vertical_load
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn overturning_safety_unity_when_balanced() {
        // Fr = Mstab/Mrenv : à moments égaux la sécurité vaut exactement 1
        // (équilibre limite au basculement).
        for &m in &[50_000.0_f64, 125_000.0, 400_000.0]
        {
            assert_relative_eq!(retwall_overturning_safety(m, m), 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn overturning_safety_is_proportional_to_stabilizing_moment() {
        // Fr est linéaire en Mstab à Mrenv fixé : doubler Mstab double Fr.
        let mrenv = 125_000.0_f64;
        let f1 = retwall_overturning_safety(200_000.0, mrenv);
        let f2 = retwall_overturning_safety(400_000.0, mrenv);
        assert_relative_eq!(f2, 2.0 * f1, epsilon = 1e-9);
    }

    #[test]
    fn sliding_safety_known_value_and_thrust_scaling() {
        // V = 200 kN/m, μ = 0,5, H = 75 kN/m :
        //   Fg = V·μ/H = 200 000·0,5/75 000 = 100 000/75 000 = 1,333333…
        let fg = retwall_sliding_safety(200_000.0, 0.5, 75_000.0);
        assert_relative_eq!(fg, 4.0 / 3.0, epsilon = 1e-9);
        // Fg est inversement proportionnelle à H : doubler H divise Fg par 2.
        let fg2 = retwall_sliding_safety(200_000.0, 0.5, 150_000.0);
        assert_relative_eq!(fg2, fg / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn base_pressure_uniform_at_zero_eccentricity() {
        // À e = 0 la résultante est centrée : σmax = V/B (répartition uniforme).
        let (v, b) = (200_000.0_f64, 3.0_f64);
        assert_relative_eq!(retwall_base_pressure_max(v, b, 0.0), v / b, epsilon = 1e-9);
    }

    #[test]
    fn eccentricity_and_pressure_are_consistent() {
        // Réciprocité : construire Mnet à partir d'une excentricité voulue
        //   e_cible = 0,375 m, V = 200 kN/m, B = 3 m
        //   Mnet = V·(B/2 − e) = 200 000·(1,5 − 0,375) = 200 000·1,125 = 225 000
        // puis retrouver e à partir de Mnet doit redonner e_cible.
        let (v, b, e_target) = (200_000.0_f64, 3.0_f64, 0.375_f64);
        let net_moment = v * (b / 2.0 - e_target);
        assert_relative_eq!(net_moment, 225_000.0, epsilon = 1e-6);
        let e = retwall_resultant_eccentricity(net_moment, v, b);
        assert_relative_eq!(e, e_target, epsilon = 1e-12);
    }

    #[test]
    fn realistic_gravity_wall_case() {
        // Mur-poids par mètre linéaire, moments pris autour du pied aval :
        //   V = 200 kN/m, B = 3 m, μ = 0,5, H = 75 kN/m
        //   Mstab = 350 kN·m/m, Mrenv = 125 kN·m/m
        // Renversement : Fr = 350/125 = 2,8
        // Glissement   : Fg = 200·0,5/75 = 1,33333…
        // Excentricité : Mnet = 350 000 − 125 000 = 225 000
        //                e = B/2 − Mnet/V = 1,5 − 225 000/200 000 = 0,375 m
        //                (≤ B/6 = 0,5 m : résultante dans le tiers central)
        // Contrainte   : σmax = (V/B)·(1 + 6e/B)
        //                     = (200 000/3)·(1 + 6·0,375/3)
        //                     = 66 666,667·1,75 = 116 666,667 Pa
        let (v, b, mu, h) = (200_000.0_f64, 3.0_f64, 0.5_f64, 75_000.0_f64);
        let (mstab, mrenv) = (350_000.0_f64, 125_000.0_f64);

        let fr = retwall_overturning_safety(mstab, mrenv);
        assert_relative_eq!(fr, 2.8, epsilon = 1e-9);

        let fg = retwall_sliding_safety(v, mu, h);
        assert_relative_eq!(fg, 4.0 / 3.0, epsilon = 1e-9);

        let net_moment = mstab - mrenv;
        let e = retwall_resultant_eccentricity(net_moment, v, b);
        assert_relative_eq!(e, 0.375, epsilon = 1e-9);
        assert!(e <= b / 6.0);

        let sigma_max = retwall_base_pressure_max(v, b, e);
        assert_relative_eq!(sigma_max, 116_666.667, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(
        expected = "l'excentricité e doit être ≤ B/6 (résultante dans le tiers central)"
    )]
    fn eccentricity_outside_middle_third_panics() {
        // e = 0,6 m > B/6 = 0,5 m pour B = 3 m : la semelle décolle, la formule
        // trapézoïdale ne s'applique plus.
        retwall_base_pressure_max(200_000.0, 3.0, 0.6);
    }
}
