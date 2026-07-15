//! GD&T — tolérancement de **position** : conversion entre l'écart mesuré du
//! centre réel d'un élément et la **zone de tolérance diamétrale** cylindrique,
//! avec prise en compte du modificateur au **maximum de matière** (MMC).
//!
//! ```text
//! écart diamétral   d = 2·√(dx² + dy²)              (diamètre de la zone occupée)
//! conformité        conforme ⇔ 2·√(dx² + dy²) ≤ t   (zone Ø t centrée sur le vrai profil)
//! tolérance bonus   b = |taille_réelle − taille_MMC| (gain autorisé par l'écart au MMC)
//! tolérance totale  T = t + b                        (zone effective sous modificateur Ⓜ)
//! ```
//!
//! `dx`, `dy` composantes de l'écart du centre réel par rapport à la position
//! théorique vraie (m) ; `d` diamètre de la zone cylindrique réellement occupée
//! (m) ; `t` tolérance de position spécifiée au cadre (m) ; `taille_réelle` et
//! `taille_MMC` diamètres effectif et au maximum de matière de l'élément (m) ;
//! `b` tolérance bonus (m) ; `T` tolérance totale disponible (m).
//!
//! **Convention** : SI cohérent ; il suffit que toutes les longueurs partagent
//! la même unité pour des ratios et sommes corrects.
//! **Limite honnête** : on suppose un **référentiel parfait** (datum exact,
//! profil théorique vrai sans erreur) et une **zone cylindrique** (modificateur
//! Ø). La tolérance bonus modélise le seul modificateur au maximum de matière Ⓜ
//! sur l'élément régulé, sans effet de datum mobile. Aucune tolérance nominale,
//! taille MMC ni classe d'ajustement n'est imposée : ces valeurs sont
//! **fournies par l'appelant** (jamais de « défaut » inventé).

/// Diamètre de la zone occupée `d = 2·√(dx² + dy²)` (m) à partir des composantes
/// planes `x_deviation`, `y_deviation` de l'écart du centre réel.
///
/// C'est la réciproque partielle de [`gdt_position_is_within`] : l'élément est
/// conforme ssi ce diamètre ne dépasse pas la tolérance spécifiée.
///
/// Panique si `x_deviation` ou `y_deviation` n'est pas fini.
pub fn gdt_position_diametral_deviation(x_deviation: f64, y_deviation: f64) -> f64 {
    assert!(
        x_deviation.is_finite() && y_deviation.is_finite(),
        "les écarts dx et dy doivent être finis"
    );
    2.0 * (x_deviation * x_deviation + y_deviation * y_deviation).sqrt()
}

/// Conformité de position : renvoie `true` ssi la zone diamétrale occupée
/// `2·√(dx² + dy²)` tient dans la tolérance `tolerance` (m).
///
/// Équivaut à `gdt_position_diametral_deviation(dx, dy) <= tolerance`.
///
/// Panique si `dx` ou `dy` n'est pas fini, ou si `tolerance < 0`.
pub fn gdt_position_is_within(x_deviation: f64, y_deviation: f64, tolerance: f64) -> bool {
    assert!(
        tolerance >= 0.0,
        "la tolérance de position t ne peut être négative"
    );
    gdt_position_diametral_deviation(x_deviation, y_deviation) <= tolerance
}

/// Tolérance bonus `b = |taille_réelle − taille_MMC|` (m) accordée par l'écart
/// de l'élément à son maximum de matière, sous modificateur Ⓜ.
///
/// Panique si `actual_size <= 0` ou si `mmc_size <= 0`.
pub fn gdt_position_bonus_tolerance(actual_size: f64, mmc_size: f64) -> f64 {
    assert!(
        actual_size > 0.0,
        "la taille réelle de l'élément doit être strictement positive"
    );
    assert!(
        mmc_size > 0.0,
        "la taille au maximum de matière (MMC) doit être strictement positive"
    );
    (actual_size - mmc_size).abs()
}

/// Tolérance totale de position `T = t + b` (m) : tolérance spécifiée `stated`
/// augmentée de la tolérance bonus `bonus`.
///
/// Panique si `stated < 0` ou si `bonus < 0`.
pub fn gdt_position_total_tolerance(stated: f64, bonus: f64) -> f64 {
    assert!(
        stated >= 0.0,
        "la tolérance spécifiée t ne peut être négative"
    );
    assert!(bonus >= 0.0, "la tolérance bonus b ne peut être négative");
    stated + bonus
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn deviation_is_diameter_not_radius() {
        // Écart purement radial r : la zone diamétrale vaut 2·r.
        let r = 0.003_f64;
        assert_relative_eq!(
            gdt_position_diametral_deviation(r, 0.0),
            2.0 * r,
            max_relative = 1e-12
        );
        // Pythagore : (3,4) → rayon 5 → diamètre 10.
        assert_relative_eq!(
            gdt_position_diametral_deviation(0.03, 0.04),
            2.0 * 0.05,
            max_relative = 1e-12
        );
    }

    #[test]
    fn deviation_scales_linearly() {
        // Homogénéité de degré 1 : doubler l'écart double le diamètre de zone.
        let d1 = gdt_position_diametral_deviation(0.002, 0.005);
        let d2 = gdt_position_diametral_deviation(0.004, 0.010);
        assert_relative_eq!(d2, 2.0 * d1, max_relative = 1e-12);
    }

    #[test]
    fn within_matches_diametral_deviation() {
        // Cohérence stricte entre le booléen et la grandeur diamétrale.
        let (dx, dy) = (0.006_f64, 0.008_f64); // diamètre = 2·0,01 = 0,02
        let d = gdt_position_diametral_deviation(dx, dy);
        assert!(gdt_position_is_within(dx, dy, d)); // égalité = conforme
        assert!(gdt_position_is_within(dx, dy, d + 1e-9));
        assert!(!gdt_position_is_within(dx, dy, d - 1e-9));
    }

    #[test]
    fn bonus_is_symmetric_gap_to_mmc() {
        // |a − m| est symétrique : l'écart au MMC ne dépend pas de l'ordre.
        assert_relative_eq!(
            gdt_position_bonus_tolerance(0.0121, 0.0120),
            gdt_position_bonus_tolerance(0.0120, 0.0121),
            max_relative = 1e-12
        );
        // Élément pile au MMC : aucun bonus.
        assert_relative_eq!(
            gdt_position_bonus_tolerance(0.0120, 0.0120),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn total_tolerance_adds_bonus() {
        // T = t + b, et le bonus élargit la zone acceptable.
        // Perçage Ø12 régulé à Ⓜ, tolérance 0,1 mm, alésage réel Ø12,05 :
        // bonus = 0,05 mm, tolérance totale = 0,15 mm.
        let t = 0.000_1_f64;
        let bonus = gdt_position_bonus_tolerance(0.012_05, 0.012_00);
        let total = gdt_position_total_tolerance(t, bonus);
        assert_relative_eq!(bonus, 0.000_05, max_relative = 1e-9);
        assert_relative_eq!(total, 0.000_15, max_relative = 1e-9);
        // Un écart refusé sous t seul devient conforme sous la tolérance totale.
        let (dx, dy) = (0.000_06_f64, 0.0);
        assert!(!gdt_position_is_within(dx, dy, t));
        assert!(gdt_position_is_within(dx, dy, total));
    }

    #[test]
    #[should_panic(expected = "maximum de matière")]
    fn non_positive_mmc_panics() {
        gdt_position_bonus_tolerance(0.012, 0.0);
    }
}
