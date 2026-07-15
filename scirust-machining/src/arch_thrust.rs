//! Statique d'un **arc à trois articulations** (isostatique) : poussée horizontale
//! aux appuis, réaction verticale et effort au sommet, sous charge symétrique.
//!
//! ```text
//! charge répartie w (symétrique)   poussée horizontale  H = w·L²/(8·f)
//! charge ponctuelle P au centre    poussée horizontale  H = P·L/(4·f)
//! réaction verticale (répartie)    V = w·L/2   (chaque appui)
//! effort au sommet (clé)           N = H       (purement horizontal)
//! ```
//!
//! `w` charge répartie (N/m), `P` charge ponctuelle (N), `L` portée (m), `f` flèche
//! (rise, m), `H` poussée horizontale (N), `V` réaction verticale (N), `N` effort
//! normal au sommet (N). La poussée est reprise par les appuis (ou un tirant).
//!
//! **Convention** : SI cohérent. **Limite honnête** : arc à **trois articulations**
//! (statiquement déterminé), géométrie et chargement **symétriques fournis par
//! l'appelant**, portée et flèche **fournies** ; aucune valeur « par défaut »
//! n'est inventée. Les efforts sont statiquement déterminés à partir de l'équilibre
//! global et de l'articulation à la clé ; on ne calcule pas la ligne de pression
//! complète ni les arcs hyperstatiques.

/// Poussée horizontale d'un arc trois-articulations sous charge répartie symétrique :
/// `H = w·L²/(8·f)`.
///
/// Panique si `span <= 0` ou `rise <= 0`.
pub fn arch_horizontal_thrust_udl(distributed_load: f64, span: f64, rise: f64) -> f64 {
    assert!(span > 0.0, "portée span > 0 requise");
    assert!(rise > 0.0, "flèche rise > 0 requise");
    distributed_load * span * span / (8.0 * rise)
}

/// Poussée horizontale d'un arc trois-articulations sous charge ponctuelle au centre :
/// `H = P·L/(4·f)`.
///
/// Panique si `span <= 0` ou `rise <= 0`.
pub fn arch_horizontal_thrust_point_center(load: f64, span: f64, rise: f64) -> f64 {
    assert!(span > 0.0, "portée span > 0 requise");
    assert!(rise > 0.0, "flèche rise > 0 requise");
    load * span / (4.0 * rise)
}

/// Réaction verticale à chaque appui sous charge répartie symétrique : `V = w·L/2`.
///
/// Panique si `span <= 0`.
pub fn arch_support_reaction_vertical_udl(distributed_load: f64, span: f64) -> f64 {
    assert!(span > 0.0, "portée span > 0 requise");
    distributed_load * span / 2.0
}

/// Effort normal au sommet (clé) d'un arc trois-articulations : `N = H`.
///
/// À la clé articulée, l'effort tranchant et le moment sont nuls ; l'effort est
/// purement horizontal et égal à la poussée. La fonction renvoie donc directement
/// la poussée horizontale fournie.
pub fn arch_axial_thrust_at_crown(horizontal_thrust: f64) -> f64 {
    horizontal_thrust
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn udl_thrust_realistic_value() {
        // w = 10000 N/m, L = 20 m, f = 4 m → H = 10000·400/(8·4) = 4e6/32 = 125000 N.
        assert_relative_eq!(
            arch_horizontal_thrust_udl(10000.0, 20.0, 4.0),
            125_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn point_center_thrust_realistic_value() {
        // P = 8000 N, L = 20 m, f = 4 m → H = 8000·20/(4·4) = 160000/16 = 10000 N.
        assert_relative_eq!(
            arch_horizontal_thrust_point_center(8000.0, 20.0, 4.0),
            10_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn vertical_reaction_is_half_total() {
        // w = 10000 N/m sur 20 m → total 200000 N → 100000 N par appui.
        assert_relative_eq!(
            arch_support_reaction_vertical_udl(10000.0, 20.0),
            100_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn thrust_scales_inversely_with_rise() {
        // Doubler la flèche divise la poussée par deux (répartie et ponctuelle).
        let h1 = arch_horizontal_thrust_udl(5000.0, 12.0, 3.0);
        let h2 = arch_horizontal_thrust_udl(5000.0, 12.0, 6.0);
        assert_relative_eq!(h1, 2.0 * h2, epsilon = 1e-9);

        let p1 = arch_horizontal_thrust_point_center(5000.0, 12.0, 3.0);
        let p2 = arch_horizontal_thrust_point_center(5000.0, 12.0, 6.0);
        assert_relative_eq!(p1, 2.0 * p2, epsilon = 1e-9);
    }

    #[test]
    fn crown_axial_equals_horizontal_thrust() {
        // L'effort au sommet est identiquement la poussée horizontale.
        let h = arch_horizontal_thrust_udl(7000.0, 15.0, 5.0);
        assert_relative_eq!(arch_axial_thrust_at_crown(h), h, epsilon = 1e-12);
    }

    #[test]
    fn point_equals_udl_with_equivalent_total_load() {
        // Charge ponctuelle P au centre ↔ charge répartie de résultante 2P donne
        // la même poussée : H_udl(w=2P/L) = (2P/L)·L²/(8f) = P·L/(4f) = H_point(P).
        let load = 6000.0;
        let span = 18.0;
        let rise = 4.5;
        let equivalent_w = 2.0 * load / span;
        assert_relative_eq!(
            arch_horizontal_thrust_udl(equivalent_w, span, rise),
            arch_horizontal_thrust_point_center(load, span, rise),
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "flèche rise > 0 requise")]
    fn zero_rise_panics() {
        arch_horizontal_thrust_udl(10000.0, 20.0, 0.0);
    }
}
