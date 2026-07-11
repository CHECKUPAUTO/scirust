//! Durée de vie de l'outil — loi de **Taylor** (1907) reliant la vitesse de
//! coupe à la durée de vie de l'arête, et sa forme étendue intégrant l'avance
//! et la profondeur de passe.
//!
//! Forme classique à deux paramètres :
//!
//! ```text
//! Vc · T^n = C
//! ```
//!
//! `Vc` vitesse de coupe (m/min), `T` durée de vie de l'arête (min), `n`
//! exposant de Taylor (sans dimension, ~0,1 pour l'ARS/HSS, ~0,25 pour le
//! carbure, ~0,5–0,7 pour la céramique) et `C` la vitesse (m/min) donnant une
//! durée de vie d'une minute. En échelle log-log, `log Vc` est affine en
//! `log T`.
//!
//! Forme étendue à quatre paramètres, tenant compte de l'avance `f` (mm/tr) et
//! de la profondeur de passe `ap` (mm) :
//!
//! ```text
//! Vc · T^n · f^a · ap^b = C
//! ```
//!
//! **Limite honnête** : `n`, `C`, `a`, `b` sont des constantes empiriques
//! issues d'essais de coupe pour un couple outil/matière donné — ce module ne
//! les invente pas. La loi de Taylor décrit la tendance dominante de l'usure ;
//! elle ignore les phénomènes transitoires (rodage, écaillage brutal) et n'est
//! valable que dans la plage de vitesses ayant servi à l'ajuster.

/// Durée de vie de l'arête `T` (min) pour une vitesse de coupe `vc` (m/min)
/// selon `Vc·T^n = C` : `T = (C / Vc)^(1/n)`.
///
/// Panique si `vc <= 0`, `c <= 0` ou `n <= 0`.
pub fn taylor_tool_life(vc_m_min: f64, c: f64, n: f64) -> f64 {
    assert!(vc_m_min > 0.0, "la vitesse de coupe doit être positive");
    assert!(c > 0.0, "la constante C de Taylor doit être positive");
    assert!(n > 0.0, "l'exposant n de Taylor doit être positif");
    (c / vc_m_min).powf(1.0 / n)
}

/// Vitesse de coupe `Vc` (m/min) donnant une durée de vie `t` (min) selon
/// `Vc·T^n = C` : `Vc = C / T^n`. Réciproque de [`taylor_tool_life`].
///
/// Panique si `t <= 0`, `c <= 0` ou `n <= 0`.
pub fn taylor_cutting_speed(t_min: f64, c: f64, n: f64) -> f64 {
    assert!(t_min > 0.0, "la durée de vie doit être positive");
    assert!(c > 0.0, "la constante C de Taylor doit être positive");
    assert!(n > 0.0, "l'exposant n de Taylor doit être positif");
    c / t_min.powf(n)
}

/// Loi de Taylor étendue `Vc · T^n · f^a · ap^b = C`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtendedTaylor {
    /// Exposant `n` sur la durée de vie.
    pub n: f64,
    /// Exposant `a` sur l'avance `f`.
    pub a: f64,
    /// Exposant `b` sur la profondeur de passe `ap`.
    pub b: f64,
    /// Constante `C` (m/min).
    pub c: f64,
}

impl ExtendedTaylor {
    /// Durée de vie `T` (min) pour une vitesse `vc` (m/min), une avance `feed`
    /// (mm/tr) et une profondeur `depth` (mm) :
    /// `T = (C / (Vc·f^a·ap^b))^(1/n)`.
    ///
    /// Panique si une entrée ou une constante est non strictement positive.
    pub fn tool_life(&self, vc_m_min: f64, feed_mm: f64, depth_mm: f64) -> f64 {
        assert!(
            vc_m_min > 0.0 && feed_mm > 0.0 && depth_mm > 0.0,
            "vitesse, avance et profondeur doivent être positives"
        );
        assert!(self.c > 0.0 && self.n > 0.0, "C et n doivent être positifs");
        let denom = vc_m_min * feed_mm.powf(self.a) * depth_mm.powf(self.b);
        (self.c / denom).powf(1.0 / self.n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn tool_life_and_speed_are_reciprocal() {
        // C=300, n=0,25, Vc=150 → T = (300/150)^4 = 2^4 = 16 min.
        let t = taylor_tool_life(150.0, 300.0, 0.25);
        assert_relative_eq!(t, 16.0, epsilon = 1e-9);
        // et la réciproque redonne 150 m/min.
        assert_relative_eq!(taylor_cutting_speed(t, 300.0, 0.25), 150.0, epsilon = 1e-9);
    }

    #[test]
    fn constant_c_is_the_one_minute_speed() {
        // T = 1 min ⇒ Vc = C, quelle que soit la valeur de n.
        assert_relative_eq!(taylor_cutting_speed(1.0, 250.0, 0.3), 250.0, epsilon = 1e-9);
    }

    #[test]
    fn higher_speed_shortens_tool_life() {
        let slow = taylor_tool_life(100.0, 300.0, 0.25);
        let fast = taylor_tool_life(200.0, 300.0, 0.25);
        assert!(fast < slow);
    }

    #[test]
    fn extended_taylor_reduces_to_simple_form() {
        // Avec a=b=0, f et ap n'interviennent plus : forme classique.
        let ext = ExtendedTaylor {
            n: 0.25,
            a: 0.0,
            b: 0.0,
            c: 300.0,
        };
        let t = ext.tool_life(150.0, 0.2, 3.0);
        assert_relative_eq!(t, taylor_tool_life(150.0, 300.0, 0.25), epsilon = 1e-9);
    }

    #[test]
    fn extended_taylor_penalises_feed_and_depth() {
        // Avec a,b>0, augmenter f ou ap augmente le dénominateur → T diminue.
        let ext = ExtendedTaylor {
            n: 0.25,
            a: 0.5,
            b: 0.2,
            c: 400.0,
        };
        let light = ext.tool_life(150.0, 0.1, 1.0);
        let heavy = ext.tool_life(150.0, 0.4, 4.0);
        assert!(heavy < light);
    }
}
