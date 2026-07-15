//! Usinage — **fraisage de filet** par interpolation hélicoïdale : nombre de
//! tours hélicoïdaux, correction d'avance centre-outil → arête et temps d'usinage
//! en une passe.
//!
//! ```text
//! tours hélicoïdaux      n  = L/p                         (–)
//! correction d'avance    fe = f·D_t/(D_t − d_o)           (mm/min)   (filet intérieur)
//! temps d'une orbite     t1 = C/vf                        (min)
//! temps total            t  = n·C/vf = (L/p)·(C/vf)       (min)
//! avance de table        vf = fr·N                        (mm/min)
//! ```
//!
//! `L` longueur de filet à usiner (mm), `p` pas du filet (mm), `n` nombre d'orbites
//! hélicoïdales, `f` avance programmée au **centre** de l'outil (mm/min), `fe`
//! avance à l'**arête** de coupe (mm/min), `D_t` diamètre du filet (mm), `d_o`
//! diamètre de l'outil (mm), `C` circonférence de la trajectoire du **centre** de
//! l'outil (mm), `vf` avance de table le long de l'hélice (mm/min), `fr` avance par
//! tour de broche (mm/tr), `N` fréquence de rotation de la broche (tr/min), `t1`
//! temps d'une orbite (min), `t` temps total (min).
//!
//! **Convention** : unités de fiche outil (mm, tr/min, mm/min) ; temps en minutes.
//! L'avance de table `vf = fr·N` : `fr` synthétise l'avance par dent et le nombre
//! de dents (`fr = fz·z`), qui n'apparaissent donc pas séparément dans le temps.
//! **Limite honnête** : interpolation hélicoïdale **idéale** en **une seule passe**,
//! trajectoire du centre supposée circulaire de circonférence `C` fournie. La
//! correction d'avance est celle d'un filet **intérieur** (l'arête tourne plus vite
//! que le centre). Les avances (`f`, `fr`, `vf`), la circonférence `C`, les diamètres
//! et la vitesse de broche sont des données de gamme FOURNIES par l'appelant : aucune
//! valeur « par défaut » n'est inventée. Ne modélise ni les entrées/sorties d'arc,
//! ni la reprise, ni l'effort de coupe ni l'usure.

/// Nombre de tours hélicoïdaux `n = L/p` pour usiner un filet de longueur `L` et
/// de pas `p` (sans dimension).
///
/// Panique si `thread_length < 0` ou si `pitch <= 0`.
pub fn thread_mill_helical_revolutions(thread_length: f64, pitch: f64) -> f64 {
    assert!(
        thread_length >= 0.0,
        "la longueur de filet doit être positive ou nulle"
    );
    assert!(pitch > 0.0, "le pas doit être strictement positif");
    thread_length / pitch
}

/// Avance corrigée à l'arête de coupe `fe = f·D_t/(D_t − d_o)` pour un filet
/// **intérieur** : l'arête décrit un cercle plus grand que le centre de l'outil et
/// avance donc plus vite (mm/min si `f` en mm/min). Le facteur `D_t/(D_t − d_o)`
/// est toujours ≥ 1.
///
/// Panique si `programmed_feed < 0`, si `tool_diameter < 0`, ou si
/// `tool_diameter >= thread_diameter`.
pub fn thread_mill_peripheral_feed_compensation(
    programmed_feed: f64,
    tool_diameter: f64,
    thread_diameter: f64,
) -> f64 {
    assert!(
        programmed_feed >= 0.0,
        "l'avance programmée doit être positive ou nulle"
    );
    assert!(
        tool_diameter >= 0.0,
        "le diamètre de l'outil doit être positif ou nul"
    );
    assert!(
        thread_diameter > tool_diameter,
        "le diamètre de l'outil doit être inférieur au diamètre du filet"
    );
    programmed_feed * thread_diameter / (thread_diameter - tool_diameter)
}

/// Temps d'une orbite hélicoïdale `t1 = C/vf` (min si `C` en mm et `feed_rate` en
/// mm/min), soit le temps de parcourir la circonférence `C` de la trajectoire du
/// centre de l'outil à l'avance de table `feed_rate`.
///
/// Panique si `circumference < 0` ou si `feed_rate <= 0`.
pub fn thread_mill_pass_time(circumference: f64, feed_rate: f64) -> f64 {
    assert!(
        circumference >= 0.0,
        "la circonférence doit être positive ou nulle"
    );
    assert!(
        feed_rate > 0.0,
        "l'avance de table doit être strictement positive"
    );
    circumference / feed_rate
}

/// Temps total de fraisage de filet `t = (L/p)·(C/vf)` avec `vf = fr·N`
/// (min), composition de [`thread_mill_helical_revolutions`] et de
/// [`thread_mill_pass_time`] : `n` orbites de circonférence `C` parcourues à
/// l'avance de table `vf`.
///
/// Panique si `thread_length < 0`, si `pitch <= 0`, si `tool_center_circumference < 0`,
/// si `feed_per_rev <= 0`, ou si `spindle_speed_rpm <= 0`.
pub fn thread_mill_time(
    thread_length: f64,
    pitch: f64,
    tool_center_circumference: f64,
    feed_per_rev: f64,
    spindle_speed_rpm: f64,
) -> f64 {
    assert!(
        feed_per_rev > 0.0,
        "l'avance par tour doit être strictement positive"
    );
    assert!(
        spindle_speed_rpm > 0.0,
        "la fréquence de rotation doit être strictement positive"
    );
    let table_feed = feed_per_rev * spindle_speed_rpm;
    let revolutions = thread_mill_helical_revolutions(thread_length, pitch);
    revolutions * thread_mill_pass_time(tool_center_circumference, table_feed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn revolutions_reciprocal_with_pitch() {
        // n = L/p, et réciproquement n·p = L (identité de définition).
        let (length, pitch) = (15.0_f64, 1.5_f64);
        let n = thread_mill_helical_revolutions(length, pitch);
        assert_relative_eq!(n, 10.0, epsilon = 1e-12);
        assert_relative_eq!(n * pitch, length, epsilon = 1e-12);
    }

    #[test]
    fn feed_compensation_amplifies_and_is_proportional() {
        // D_t=20, d_o=10 → facteur 20/(20−10)=2 : l'arête avance deux fois plus vite.
        let fe = thread_mill_peripheral_feed_compensation(100.0, 10.0, 20.0);
        assert_relative_eq!(fe, 200.0, epsilon = 1e-12);
        // fe ∝ f : doubler l'avance programmée double l'avance à l'arête.
        let fe2 = thread_mill_peripheral_feed_compensation(200.0, 10.0, 20.0);
        assert_relative_eq!(fe2 / fe, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn feed_compensation_vanishes_for_thin_tool() {
        // Outil infiniment fin (d_o → 0) → facteur → 1 : arête ≈ centre.
        let fe = thread_mill_peripheral_feed_compensation(120.0, 1e-9, 20.0);
        assert_relative_eq!(fe, 120.0, epsilon = 1e-6);
    }

    #[test]
    fn pass_time_reciprocal_with_feed() {
        // t1 = C/vf, et réciproquement C = t1·vf.
        let (circumference, feed) = (50.0_f64, 100.0_f64);
        let t1 = thread_mill_pass_time(circumference, feed);
        assert_relative_eq!(t1, 0.5, epsilon = 1e-12);
        assert_relative_eq!(t1 * feed, circumference, epsilon = 1e-12);
    }

    #[test]
    fn total_time_composes_and_scales_with_length() {
        // L=15, p=1,5 → n=10 ; vf = fr·N = 0,2·500 = 100 mm/min ; C=100 mm.
        // t = 10·(100/100) = 10 min.
        let t = thread_mill_time(15.0, 1.5, 100.0, 0.2, 500.0);
        assert_relative_eq!(t, 10.0, epsilon = 1e-12);
        // t ∝ L : doubler la longueur de filet double le temps.
        let t2 = thread_mill_time(30.0, 1.5, 100.0, 0.2, 500.0);
        assert_relative_eq!(t2 / t, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "diamètre de l'outil doit être inférieur")]
    fn feed_compensation_rejects_oversize_tool() {
        thread_mill_peripheral_feed_compensation(100.0, 20.0, 20.0);
    }
}
