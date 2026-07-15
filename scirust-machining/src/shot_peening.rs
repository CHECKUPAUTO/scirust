//! Grenaillage de précontrainte — **couverture** (loi d'Avrami/exponentielle) et
//! **intensité Almen** (saturation empirique de la hauteur de flèche).
//!
//! ```text
//! couverture (Avrami)   C = 1 - exp(-t / tau)
//! calage sur 98 %       tau = t98 / ln(50)      ⇒  C = 1 - 0,02^(t / t98)
//! passes multiples      C = 1 - (1 - c)^n
//! flèche Almen          h = h_sat · (1 - exp(-t / tau_A))
//! critère de saturation r = (h(2t) - h(t)) / h(t)   (saturation si r ≤ 0,10)
//! ```
//!
//! `C` couverture (fraction 0..1, ou % quand indiqué), `t` temps d'exposition (s),
//! `tau` constante de temps de couverture (s), `t98` temps pour atteindre 98 % de
//! couverture (s), `c` couverture d'une passe unique (fraction 0..1), `n` nombre de
//! passes (adimensionnel), `h` hauteur de flèche Almen (m), `h_sat` flèche
//! asymptotique de saturation (m), `tau_A` constante de temps de la courbe Almen (s),
//! `r` accroissement relatif de flèche entre `t` et `2t` (adimensionnel).
//!
//! **Convention** : SI cohérent (temps en s, flèche en m), les fractions de
//! couverture sont dans `[0, 1]` sauf `peening_coverage_percent` qui renvoie des
//! pour-cent. **Limite honnête** : la couverture suit une loi **exponentielle
//! (Avrami)** phénoménologique et la saturation Almen est un **critère empirique**
//! (règle des 10 % sur un doublement d'exposition). Les paramètres de procédé
//! (`t98`, `tau`, `tau_A`, `h_sat`, couverture par passe) dépendent du média, du
//! débit, de l'angle et de la pièce : ils sont **fournis par l'appelant** — aucune
//! valeur « par défaut » n'est inventée.

/// Couverture par la loi d'Avrami calée sur 98 % : `C% = 100·(1 - 0,02^(t/t98))`.
///
/// Modèle `C = 1 - exp(-t/tau)` avec `tau = t98/ln(50)`, de sorte que
/// `C = 98 %` exactement lorsque `exposure_time == time_for_98_percent`.
/// `exposure_time` = `t` (s), `time_for_98_percent` = `t98` (s). Renvoie des %.
///
/// Panique si `exposure_time < 0` ou `time_for_98_percent <= 0`.
pub fn peening_coverage_percent(exposure_time: f64, time_for_98_percent: f64) -> f64 {
    assert!(
        exposure_time >= 0.0,
        "le temps d'exposition doit être positif"
    );
    assert!(
        time_for_98_percent > 0.0,
        "le temps pour 98 % doit être strictement positif"
    );
    let ln50 = 50.0_f64.ln();
    100.0 * (1.0 - (-exposure_time * ln50 / time_for_98_percent).exp())
}

/// Temps d'exposition pour atteindre une couverture visée `t = -t98·ln(1-C)/ln(50)`.
///
/// Réciproque exacte de [`peening_coverage_percent`] (mêmes hypothèses d'Avrami).
/// `target_coverage` = `C` (fraction 0..1, exclu 1), `time_for_98_percent` = `t98` (s).
/// Renvoie le temps d'exposition (s).
///
/// Panique si `target_coverage` hors de `[0, 1[` ou `time_for_98_percent <= 0`.
pub fn peening_time_for_coverage(target_coverage: f64, time_for_98_percent: f64) -> f64 {
    assert!(
        (0.0..1.0).contains(&target_coverage),
        "la couverture visée doit être dans [0, 1["
    );
    assert!(
        time_for_98_percent > 0.0,
        "le temps pour 98 % doit être strictement positif"
    );
    let ln50 = 50.0_f64.ln();
    -time_for_98_percent * (1.0 - target_coverage).ln() / ln50
}

/// Couverture après `n` passes indépendantes `C = 1 - (1 - c)^n` (fraction 0..1).
///
/// Chaque passe ajoute une couverture `c` sur la surface encore vierge.
/// `passes` = `n` (adimensionnel, ≥ 0), `single_pass_coverage` = `c` (fraction 0..1).
///
/// Panique si `passes < 0` ou `single_pass_coverage` hors de `[0, 1]`.
pub fn peening_coverage_from_passes(passes: f64, single_pass_coverage: f64) -> f64 {
    assert!(passes >= 0.0, "le nombre de passes doit être positif");
    assert!(
        (0.0..=1.0).contains(&single_pass_coverage),
        "la couverture par passe doit être dans [0, 1]"
    );
    1.0 - (1.0 - single_pass_coverage).powf(passes)
}

/// Nombre de passes pour une couverture visée `n = ln(1-C)/ln(1-c)`.
///
/// Réciproque de [`peening_coverage_from_passes`] (nombre de passes, réel non entier).
/// `target_coverage` = `C` (fraction 0..1, exclu 1), `single_pass_coverage` = `c`
/// (fraction 0..1, exclus 0 et 1).
///
/// Panique si `target_coverage` hors de `[0, 1[` ou `single_pass_coverage` hors de `]0, 1[`.
pub fn peening_passes_for_coverage(target_coverage: f64, single_pass_coverage: f64) -> f64 {
    assert!(
        (0.0..1.0).contains(&target_coverage),
        "la couverture visée doit être dans [0, 1["
    );
    assert!(
        single_pass_coverage > 0.0 && single_pass_coverage < 1.0,
        "la couverture par passe doit être dans ]0, 1["
    );
    (1.0 - target_coverage).ln() / (1.0 - single_pass_coverage).ln()
}

/// Hauteur de flèche Almen `h = h_sat·(1 - exp(-t/tau_A))` (m).
///
/// Approche exponentielle vers la flèche de saturation `h_sat`.
/// `saturation_height` = `h_sat` (m), `time_constant` = `tau_A` (s),
/// `exposure_time` = `t` (s).
///
/// Panique si `saturation_height < 0`, `time_constant <= 0` ou `exposure_time < 0`.
pub fn almen_arc_height(saturation_height: f64, time_constant: f64, exposure_time: f64) -> f64 {
    assert!(
        saturation_height >= 0.0,
        "la flèche de saturation doit être positive"
    );
    assert!(
        time_constant > 0.0,
        "la constante de temps doit être strictement positive"
    );
    assert!(
        exposure_time >= 0.0,
        "le temps d'exposition doit être positif"
    );
    saturation_height * (1.0 - (-exposure_time / time_constant).exp())
}

/// Accroissement relatif de flèche sur un doublement d'exposition
/// `r = (h(2t) - h(t)) / h(t)` (adimensionnel).
///
/// Critère de saturation Almen (SAE J443) : la saturation est atteinte lorsque
/// `r ≤ 0,10` (doubler le temps n'ajoute pas plus de 10 % de flèche).
/// `arc_height` = `h(t)` (m), `arc_height_double_time` = `h(2t)` (m).
///
/// Panique si `arc_height <= 0` ou `arc_height_double_time < arc_height`.
pub fn almen_saturation_increase_ratio(arc_height: f64, arc_height_double_time: f64) -> f64 {
    assert!(
        arc_height > 0.0,
        "la flèche de référence doit être strictement positive"
    );
    assert!(
        arc_height_double_time >= arc_height,
        "la flèche à double exposition ne peut pas décroître"
    );
    (arc_height_double_time - arc_height) / arc_height
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn coverage_and_time_are_reciprocal() {
        // t = f(C) doit inverser exactement C = f(t) (loi d'Avrami).
        let t98 = 12.0_f64; // s
        let c = 0.90_f64;
        let t = peening_time_for_coverage(c, t98);
        assert_relative_eq!(peening_coverage_percent(t, t98) / 100.0, c, epsilon = 1e-12);
    }

    #[test]
    fn coverage_reaches_98_at_reference_time() {
        // Par construction, l'exposition t98 donne 98 % de couverture.
        let t98 = 20.0_f64;
        assert_relative_eq!(peening_coverage_percent(t98, t98), 98.0, epsilon = 1e-9);
    }

    #[test]
    fn passes_and_coverage_are_reciprocal() {
        // n = g(C) doit inverser C = 1 - (1-c)^n.
        let c = 0.25_f64; // couverture par passe
        let target = 0.95_f64;
        let n = peening_passes_for_coverage(target, c);
        assert_relative_eq!(peening_coverage_from_passes(n, c), target, epsilon = 1e-12);
    }

    #[test]
    fn single_pass_matches_its_coverage() {
        // Une seule passe rend exactement la couverture par passe.
        let c = 0.4_f64;
        assert_relative_eq!(peening_coverage_from_passes(1.0, c), c, epsilon = 1e-15);
    }

    #[test]
    fn arc_height_rises_from_zero_towards_saturation() {
        // À t=0 : flèche nulle ; asymptote vers h_sat aux temps longs.
        let h_sat = 0.30e-3_f64; // 0,30 mm
        let tau = 15.0_f64;
        assert_relative_eq!(almen_arc_height(h_sat, tau, 0.0), 0.0, epsilon = 1e-18);
        // À t = 5·tau, on est à moins de 1 % de la saturation.
        let h = almen_arc_height(h_sat, tau, 5.0 * tau);
        assert_relative_eq!(h / h_sat, 1.0 - (-5.0_f64).exp(), epsilon = 1e-12);
    }

    #[test]
    fn saturation_ratio_matches_exponential_model() {
        // Pour le modèle exponentiel, r = (h(2t)-h(t))/h(t) = exp(-t/tau).
        let h_sat = 0.25e-3_f64;
        let tau = 10.0_f64;
        let t = 10.0_f64; // t = tau
        let h1 = almen_arc_height(h_sat, tau, t);
        let h2 = almen_arc_height(h_sat, tau, 2.0 * t);
        let r = almen_saturation_increase_ratio(h1, h2);
        assert_relative_eq!(r, (-t / tau).exp(), epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "couverture par passe doit être dans ]0, 1[")]
    fn full_single_pass_coverage_panics() {
        peening_passes_for_coverage(0.9, 1.0);
    }
}
