//! Câbles paraboliques — câble tendu sous charge **uniformément répartie
//! horizontalement** (ponts suspendus) : tensions et longueur développée.
//!
//! ```text
//! tension horizontale  H = w·L²/(8·d)
//! tension maximale     T = √( H² + (w·L/2)² )        (aux appuis)
//! longueur du câble    s ≈ L·(1 + (8/3)·(d/L)²)      (approx. flèche faible)
//! ```
//!
//! `w` charge répartie par unité de **longueur horizontale** (N/m), `L` portée
//! (m), `d` flèche au milieu (m), `H` tension horizontale (constante le long du
//! câble), `T` tension maximale (aux appuis). La flèche `d` fixe le compromis
//! tension/longueur.
//!
//! **Convention** : SI cohérent. **Limite honnête** : approximation
//! **parabolique** (charge uniforme en projection horizontale, flèche faible
//! `d/L ≲ 0,1`) — un câble sous son propre poids suit une **chaînette**
//! (caténaire) que ce module n'implémente pas. Appuis de même niveau.

/// Tension horizontale `H = w·L²/(8·d)` (N), constante le long du câble.
///
/// Panique si `sag <= 0`.
pub fn horizontal_tension(load_per_length: f64, span: f64, sag: f64) -> f64 {
    assert!(sag > 0.0, "la flèche doit être strictement positive");
    load_per_length * span * span / (8.0 * sag)
}

/// Tension maximale aux appuis `T = √(H² + (w·L/2)²)` (N).
pub fn max_tension(horizontal_tension: f64, load_per_length: f64, span: f64) -> f64 {
    let v = load_per_length * span / 2.0;
    (horizontal_tension * horizontal_tension + v * v).sqrt()
}

/// Longueur développée du câble (approximation parabolique)
/// `s ≈ L·(1 + (8/3)·(d/L)²)` (m).
///
/// Panique si `span <= 0`.
pub fn parabolic_length(span: f64, sag: f64) -> f64 {
    assert!(span > 0.0, "la portée doit être strictement positive");
    let ratio = sag / span;
    span * (1.0 + (8.0 / 3.0) * ratio * ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn horizontal_tension_definition() {
        // w=1000 N/m, L=100 m, d=10 m → H = 1000·10000/(80) = 125 000 N.
        assert_relative_eq!(
            horizontal_tension(1000.0, 100.0, 10.0),
            125_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn max_tension_exceeds_horizontal() {
        // T = √(H² + V²) > H (composante verticale aux appuis).
        let h = horizontal_tension(1000.0, 100.0, 10.0);
        let t = max_tension(h, 1000.0, 100.0);
        assert!(t > h);
        assert_relative_eq!(
            t,
            (h * h + (1000.0f64 * 50.0).powi(2)).sqrt(),
            epsilon = 1e-3
        );
    }

    #[test]
    fn smaller_sag_raises_tension() {
        // Réduire la flèche augmente la tension horizontale (H ∝ 1/d).
        let h1 = horizontal_tension(1000.0, 100.0, 10.0);
        let h2 = horizontal_tension(1000.0, 100.0, 5.0);
        assert_relative_eq!(h2 / h1, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn cable_is_longer_than_span() {
        // La longueur développée dépasse la portée.
        let s = parabolic_length(100.0, 10.0);
        assert!(s > 100.0);
        assert_relative_eq!(s, 100.0 * (1.0 + (8.0 / 3.0) * 0.01), epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "flèche")]
    fn zero_sag_panics() {
        horizontal_tension(1000.0, 100.0, 0.0);
    }
}
