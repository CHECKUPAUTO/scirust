//! Accélération d'Anderson (Walker & Ni, « Anderson Acceleration for
//! Fixed-Point Iterations », SIAM J. Numer. Anal. 49(4), 2011) : accélère
//! une itération à point fixe `x_{k+1} = g(x_k)` en recombinant les
//! dernières itérations plutôt qu'en n'utiliser que la plus récente —
//! utile pour les boucles de couplage de type Picard (multi-physique) ou
//! pour accélérer la convergence lente de Newton/Broyden
//! (`crate::nonlinear`) sur des problèmes mal conditionnés.
//!
//! ## Algorithme (formulation « Type-I » à contrainte de somme unitaire)
//! Avec `f_i = g(x_i) - x_i` le résidu à l'itération `i`, la version
//! Anderson(m) choisit, parmi une fenêtre des `m+1` dernières itérations,
//! les coefficients `α_i` (Σα_i = 1) minimisant `‖Σ α_i f_i‖`, puis pose
//! `x_{k+1} = Σ α_i g(x_i)`. En éliminant `α_0 = 1 - Σ_{i≥1} α_i`, ce
//! problème sous contrainte devient les moindres carrés sans contrainte
//! `min_γ ‖ΔF·γ + f_base‖` (résolus ici par la QR déjà présente dans ce
//! crate — `crate::linalg::solve_qr_least_squares`), avec
//! `x_{k+1} = g_base + ΔG·γ`. C'est mathématiquement équivalent à la
//! formulation par différences la plus souvent citée, mais se lit
//! directement comme « combinaison affine des sorties passées de `g` qui
//! annule au mieux le résidu combiné ».
//!
//! ## Déterminisme
//! Fenêtre de mémoire `m` fixe, moindres carrés résolus par une QR
//! déterministe (pas d'aléa), nombre max d'itérations fixe.

use crate::linalg::{Matrix, norm2, qr_decompose, solve_qr_least_squares};
use crate::{ConvergenceInfo, Solution, SolverError, SolverResult, Tolerance};

fn check_finite_slice(v: &[f64], iter: usize) -> SolverResult<()> {
    for &x in v
    {
        if !x.is_finite()
        {
            return Err(SolverError::NanDetected { iter, value: x });
        }
    }
    Ok(())
}

/// Accélère l'itération à point fixe `x_{k+1} = g(x_k)` par Anderson(m).
///
/// `m` est la taille de la fenêtre de mémoire (nombre d'itérations passées
/// recombinées ; typiquement 3 à 10). Automatiquement plafonnée à la
/// dimension de `x0` (au-delà, le système de moindres carrés serait
/// sous-déterminé).
pub fn anderson_accelerate<G>(
    g: G,
    x0: Vec<f64>,
    m: usize,
    tol: Tolerance,
) -> SolverResult<Solution<Vec<f64>>>
where
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0
    {
        return Err(SolverError::InvalidInput(
            "anderson_accelerate: x0 must be non-empty".to_string(),
        ));
    }
    if m == 0
    {
        return Err(SolverError::InvalidInput(
            "anderson_accelerate: m must be >= 1".to_string(),
        ));
    }
    check_finite_slice(&x0, 0)?;
    let window = m.min(n);

    let mut x = x0;
    let mut gx = g(&x);
    check_finite_slice(&gx, 0)?;
    let mut fx: Vec<f64> = gx.iter().zip(&x).map(|(gi, xi)| gi - xi).collect();

    // Historique (x, g(x), f(x)) de la fenêtre courante, du plus ancien au
    // plus récent.
    let mut history: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> =
        vec![(x.clone(), gx.clone(), fx.clone())];

    for k in 0..tol.max_iter
    {
        let residual = norm2(&fx);
        if residual <= tol.abs + tol.rel * norm2(&x).max(1.0)
        {
            return Ok(Solution {
                value: x,
                info: ConvergenceInfo {
                    iterations: k,
                    residual,
                    converged: true,
                },
            });
        }

        if history.len() == 1
        {
            // Pas encore d'historique : itération à point fixe ordinaire.
            x = gx.clone();
        }
        else
        {
            let (base_x, base_g, base_f) = &history[0];
            let _ = base_x;
            let cols = history.len() - 1;
            let mut delta_f = Matrix::zeros(n, cols);
            let mut delta_g = Matrix::zeros(n, cols);
            for (c, (_, g_i, f_i)) in history[1..].iter().enumerate()
            {
                for i in 0..n
                {
                    delta_f[(i, c)] = f_i[i] - base_f[i];
                    delta_g[(i, c)] = g_i[i] - base_g[i];
                }
            }
            let neg_base_f: Vec<f64> = base_f.iter().map(|v| -v).collect();
            let qr = qr_decompose(delta_f)?;
            let gamma = solve_qr_least_squares(&qr, &neg_base_f)?;

            let mut x_next = base_g.clone();
            for c in 0..cols
            {
                for i in 0..n
                {
                    x_next[i] += delta_g[(i, c)] * gamma[c];
                }
            }
            x = x_next;
        }

        gx = g(&x);
        check_finite_slice(&gx, k + 1)?;
        fx = gx.iter().zip(&x).map(|(gi, xi)| gi - xi).collect();
        check_finite_slice(&fx, k + 1)?;

        history.push((x.clone(), gx.clone(), fx.clone()));
        if history.len() > window + 1
        {
            history.remove(0);
        }
    }

    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: norm2(&fx),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn accelerates_scalar_cosine_fixed_point() {
        // x = cos(x) -> le nombre de Dottie, ≈0.7390851332151607.
        let sol = anderson_accelerate(
            |x: &[f64]| vec![x[0].cos()],
            vec![0.5],
            5,
            Tolerance::new(1e-12, 1e-12, 200),
        )
        .unwrap();
        assert_relative_eq!(sol.value[0], 0.739_085_133_215_160_7, epsilon = 1e-9);
    }

    #[test]
    fn converges_to_the_true_fixed_point_of_a_linear_map() {
        // g(x) = A x + b, rayon spectral < 1 -> point fixe (I-A)^-1 b.
        // A = [[0.4, 0.1], [0.2, 0.3]], b = [1, 2].
        let g = |x: &[f64]| vec![0.4 * x[0] + 0.1 * x[1] + 1.0, 0.2 * x[0] + 0.3 * x[1] + 2.0];
        let sol = anderson_accelerate(g, vec![0.0, 0.0], 4, Tolerance::default()).unwrap();
        // Résolu à la main : (I-A) x = b.
        let x0 = sol.value[0];
        let x1 = sol.value[1];
        assert_relative_eq!(0.6 * x0 - 0.1 * x1, 1.0, epsilon = 1e-6);
        assert_relative_eq!(-0.2 * x0 + 0.7 * x1, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn accelerates_convergence_versus_plain_fixed_point_iteration() {
        // Itération de Picard nue pour comparaison : x_{k+1} = g(x_k).
        fn plain_iterations(
            g: impl Fn(&[f64]) -> Vec<f64>,
            mut x: Vec<f64>,
            tol: f64,
            max_iter: usize,
        ) -> usize {
            for k in 0..max_iter
            {
                let gx = g(&x);
                let res: f64 = gx
                    .iter()
                    .zip(&x)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if res <= tol
                {
                    return k;
                }
                x = gx;
            }
            max_iter
        }
        let g = |x: &[f64]| vec![0.9 * x[0].cos() + 0.1];
        let plain = plain_iterations(g, vec![0.0], 1e-10, 500);
        let sol = anderson_accelerate(g, vec![0.0], 5, Tolerance::new(1e-10, 1e-10, 500)).unwrap();
        assert!(
            sol.info.iterations <= plain,
            "Anderson ({}) should not need more iterations than plain Picard ({plain})",
            sol.info.iterations
        );
    }

    #[test]
    fn rejects_zero_window() {
        assert!(
            anderson_accelerate(|x: &[f64]| x.to_vec(), vec![0.0], 0, Tolerance::default())
                .is_err()
        );
    }

    #[test]
    fn rejects_empty_initial_point() {
        assert!(
            anderson_accelerate(|x: &[f64]| x.to_vec(), vec![], 3, Tolerance::default()).is_err()
        );
    }
}
