//! Quadrature de Romberg : extrapolation de Richardson sur la règle du
//! trapèze. Converge à l'ordre 2k pour k niveaux d'extrapolation. Très
//! efficace sur fonctions très lisses (analytiques).

/// Intègre f sur [a,b] par Romberg. `max_levels` ~ 10-15 suffit en pratique.
/// `tol` est la tolérance absolue sur l'erreur estimée (différence entre deux
/// niveaux successifs).
pub fn romberg<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, tol: f64, max_levels: usize) -> f64 {
    let mut r = vec![vec![0.0_f64; max_levels]; max_levels];

    // Niveau 0 : trapèze simple
    r[0][0] = 0.5 * (b - a) * (f(a) + f(b));

    for i in 1..max_levels
    {
        // Trapèze composé avec 2^i intervalles
        let n = 1usize << (i - 1); // nombre de NOUVEAUX points
        let h = (b - a) / (1usize << i) as f64;
        let mut sum = 0.0;
        for k in 0..n
        {
            sum += f(a + (2 * k + 1) as f64 * h);
        }
        r[i][0] = 0.5 * r[i - 1][0] + h * sum;

        // Extrapolation de Richardson
        for j in 1..=i
        {
            let denom = (1usize << (2 * j)) as f64 - 1.0;
            r[i][j] = r[i][j - 1] + (r[i][j - 1] - r[i - 1][j - 1]) / denom;
        }

        // Convergence : différence entre deux niveaux d'extrapolation
        if i >= 2 && (r[i][i] - r[i - 1][i - 1]).abs() < tol
        {
            return r[i][i];
        }
    }
    r[max_levels - 1][max_levels - 1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn romberg_sin() {
        let v = romberg(|x| x.sin(), 0.0, PI, 1e-14, 15);
        assert_relative_eq!(v, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn romberg_exp() {
        // ∫₀¹ exp(x) dx = e - 1
        let v = romberg(|x: f64| x.exp(), 0.0, 1.0, 1e-14, 15);
        assert_relative_eq!(v, std::f64::consts::E - 1.0, epsilon = 1e-13);
    }
}
