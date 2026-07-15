//! Usinage — **taillage d'engrenage par fraise-mère** (hobbing) : rapport
//! cinématique de génération, avance de table et temps de taillage.
//!
//! ```text
//! rotation pièce   n_w = n_h·s/z          (rapport de génération)
//! avance de table  vf  = f_a·n_w          (mm/min)
//! temps de coupe   t   = b/vf             (min)
//! temps combiné    t   = b·z/(n_h·s·f_a)  (min)
//! ```
//!
//! `n_h` fréquence de rotation de la fraise-mère (tr/min), `s` nombre de filets
//! (starts) de la fraise, `z` nombre de dents à tailler, `n_w` fréquence de
//! rotation de la pièce/table (tr/min), `f_a` avance axiale par tour de pièce
//! (mm/tr), `vf` avance de table (mm/min), `b` largeur de denture à parcourir
//! (mm), `t` temps de taillage (min).
//!
//! **Convention** : unités de fiche outil (mm, tr/min, mm/min, min). Une fraise
//! à `s` filets engendre `s` dents par tour, d'où `n_w = n_h·s/z`.
//! **Limite honnête** : génération continue idéalisée ; l'avance axiale, les
//! fréquences de rotation et la largeur utile sont **fournies par l'appelant**
//! (aucune valeur de procédé, de matériau ou de marge inventée). On ignore les
//! courses d'approche et de dégagement de la fraise ainsi que la profondeur
//! d'engagement liée au diamètre outil ; ajoutez-les à `b` si nécessaire.

/// Fréquence de rotation de la pièce `n_w = n_h·s/z` (tr/min) imposée par le
/// rapport cinématique de génération.
///
/// Panique si `hob_rpm <= 0`, `hob_starts == 0` ou `gear_teeth == 0`.
pub fn hobbing_work_rpm_from_hob(hob_rpm: f64, hob_starts: u32, gear_teeth: u32) -> f64 {
    assert!(
        hob_rpm > 0.0,
        "la fréquence de rotation de la fraise doit être strictement positive"
    );
    assert!(hob_starts > 0, "le nombre de filets doit être au moins 1");
    assert!(gear_teeth > 0, "le nombre de dents doit être au moins 1");
    hob_rpm * hob_starts as f64 / gear_teeth as f64
}

/// Avance de table `vf = f_a·n_w` (mm/min) à partir de l'avance axiale par tour
/// de pièce et de la fréquence de rotation de la pièce.
///
/// Panique si `axial_feed_per_rev < 0` ou `work_rpm < 0`.
pub fn hobbing_table_feed(axial_feed_per_rev: f64, work_rpm: f64) -> f64 {
    assert!(
        axial_feed_per_rev >= 0.0,
        "l'avance axiale par tour doit être positive ou nulle"
    );
    assert!(
        work_rpm >= 0.0,
        "la fréquence de rotation de la pièce doit être positive ou nulle"
    );
    axial_feed_per_rev * work_rpm
}

/// Temps de coupe `t = b/vf` (min) pour parcourir la largeur de denture à
/// l'avance de table donnée.
///
/// Panique si `gear_width < 0` ou `feed_rate <= 0`.
pub fn hobbing_cutting_time(gear_width: f64, feed_rate: f64) -> f64 {
    assert!(
        gear_width >= 0.0,
        "la largeur de denture doit être positive ou nulle"
    );
    assert!(
        feed_rate > 0.0,
        "l'avance de table doit être strictement positive"
    );
    gear_width / feed_rate
}

/// Temps de taillage combiné `t = b·z/(n_h·s·f_a)` (min), équivalent à la
/// composition de [`hobbing_work_rpm_from_hob`], [`hobbing_table_feed`] et
/// [`hobbing_cutting_time`].
///
/// Panique si `gear_width < 0`, `gear_teeth == 0`, `hob_rpm <= 0`,
/// `hob_starts == 0` ou `axial_feed_per_rev <= 0`.
pub fn hobbing_generating_time(
    gear_width: f64,
    gear_teeth: u32,
    hob_rpm: f64,
    hob_starts: u32,
    axial_feed_per_rev: f64,
) -> f64 {
    assert!(
        gear_width >= 0.0,
        "la largeur de denture doit être positive ou nulle"
    );
    assert!(gear_teeth > 0, "le nombre de dents doit être au moins 1");
    assert!(
        hob_rpm > 0.0,
        "la fréquence de rotation de la fraise doit être strictement positive"
    );
    assert!(hob_starts > 0, "le nombre de filets doit être au moins 1");
    assert!(
        axial_feed_per_rev > 0.0,
        "l'avance axiale par tour doit être strictement positive"
    );
    gear_width * gear_teeth as f64 / (hob_rpm * hob_starts as f64 * axial_feed_per_rev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn work_rpm_follows_generation_ratio() {
        // n_h=120 tr/min, s=1, z=30 → n_w = 120·1/30 = 4 tr/min.
        assert_relative_eq!(
            hobbing_work_rpm_from_hob(120.0, 1, 30),
            4.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn work_rpm_proportional_to_starts() {
        // Doubler le nombre de filets double la rotation pièce (mêmes n_h, z).
        let one = hobbing_work_rpm_from_hob(120.0, 1, 30);
        let two = hobbing_work_rpm_from_hob(120.0, 2, 30);
        assert_relative_eq!(two, 2.0 * one, epsilon = 1e-12);
    }

    #[test]
    fn cutting_time_inversely_proportional_to_feed() {
        // t = b/vf : doubler l'avance de table divise le temps par deux.
        let t1 = hobbing_cutting_time(40.0, 8.0);
        let t2 = hobbing_cutting_time(40.0, 16.0);
        assert_relative_eq!(t1, 2.0 * t2, epsilon = 1e-12);
    }

    #[test]
    fn combined_time_matches_step_by_step() {
        // Identité : le temps combiné = composition des trois étapes.
        let (b, z, n_h, s, f_a) = (40.0, 30_u32, 120.0, 1_u32, 2.0);
        let n_w = hobbing_work_rpm_from_hob(n_h, s, z);
        let vf = hobbing_table_feed(f_a, n_w);
        let step = hobbing_cutting_time(b, vf);
        let combined = hobbing_generating_time(b, z, n_h, s, f_a);
        assert_relative_eq!(step, combined, epsilon = 1e-12);
        // Cas chiffré : n_w=4, vf=8 mm/min, t = 40/8 = 5 min.
        assert_relative_eq!(combined, 5.0, epsilon = 1e-12);
    }

    #[test]
    fn table_feed_definition() {
        // vf = f_a·n_w = 2·4 = 8 mm/min.
        assert_relative_eq!(hobbing_table_feed(2.0, 4.0), 8.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "l'avance de table doit être strictement positive")]
    fn zero_feed_rate_panics() {
        hobbing_cutting_time(40.0, 0.0);
    }
}
