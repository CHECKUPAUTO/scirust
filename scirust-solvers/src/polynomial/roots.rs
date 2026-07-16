//! Calcul des racines d'un polynôme par l'algorithme de Durand-Kerner
//! (a.k.a. Weierstrass).
//!
//! Itère sur toutes les racines simultanément. Chaque itération :
//!   z_i ← z_i - p(z_i) / Π_{j≠i} (z_i - z_j)
//!
//! Convergence quadratique près de racines simples ; converge globalement
//! depuis presque tous les points de départ, sur des polynômes modérément
//! mal conditionnés (racines entières bien séparées de degré ~5-10, voir
//! [`tests::degree_5`]).
//!
//! **Limite connue** : cette implémentation travaille sur les coefficients
//! sous forme développée (base monomiale), qui est numériquement instable
//! pour des degrés élevés — le polynôme de Wilkinson classique (degré 20,
//! racines 1..20) a des coefficients allant jusqu'à ~2.4×10¹⁸, et
//! l'évaluation de Horner en `f64` y produit des annulations catastrophiques
//! dès les premières itérations. [`durand_kerner`]/[`durand_kerner_strict`]
//! détectent ce cas (pas non fini) et renvoient
//! [`SolverError::NanDetected`] plutôt que des racines silencieusement
//! fausses — mais ne le résolvent pas : un tel polynôme a besoin d'une
//! méthode dédiée (déflation, arithmétique étendue, ou passage par la
//! matrice compagnon avec un solveur d'eigenvalues, comme le fait LAPACK).

use super::Polynomial;
use crate::{SolverError, SolverResult};

/// Trouve toutes les racines (complexes) via Durand-Kerner. Best-effort : si
/// `max_iter` est épuisé sans que le pas maximal descende sous `tol`, renvoie
/// quand même la meilleure estimation courante (voir [`durand_kerner_strict`]
/// pour une variante qui signale l'échec au lieu de le masquer).
/// Renvoie un Vec<(re, im)> de longueur `degree`.
pub fn durand_kerner(p: &Polynomial, max_iter: usize, tol: f64) -> SolverResult<Vec<(f64, f64)>> {
    durand_kerner_table(p, max_iter, tol).map(|(z, _)| z)
}

/// Variante stricte de [`durand_kerner`] : renvoie
/// [`SolverError::NoConvergence`] si le pas maximal n'est pas descendu sous
/// `tol` après `max_iter` itérations, plutôt que de renvoyer silencieusement
/// la meilleure estimation disponible.
pub fn durand_kerner_strict(
    p: &Polynomial,
    max_iter: usize,
    tol: f64,
) -> SolverResult<Vec<(f64, f64)>> {
    let (z, last_step) = durand_kerner_table(p, max_iter, tol)?;
    if last_step < tol
    {
        Ok(z)
    }
    else
    {
        Err(SolverError::NoConvergence {
            iterations: max_iter,
            residual: last_step,
        })
    }
}

/// Runs Durand-Kerner and returns `(roots, last_max_step)`. A `last_max_step`
/// of `0.0` for a degree-0 polynomial (no roots) or a first-iteration
/// all-coincident-roots skip is treated as converged by both callers above.
fn durand_kerner_table(
    p: &Polynomial,
    max_iter: usize,
    tol: f64,
) -> SolverResult<(Vec<(f64, f64)>, f64)> {
    let n = p.degree();
    if n == 0
    {
        return Ok((Vec::new(), 0.0));
    }
    // Normalise pour que le coefficient dominant vaille 1
    let lead = *p.coeffs.last().unwrap();
    if lead.abs() < 1e-30
    {
        return Err(SolverError::InvalidInput(
            "leading coefficient is zero".into(),
        ));
    }
    let monic: Vec<f64> = p.coeffs.iter().map(|c| c / lead).collect();

    // Initialisation : racines de l'unité multipliées par 0.4 + 0.9i
    // (classique, évite les coïncidences avec des racines réelles)
    let mut z: Vec<(f64, f64)> = Vec::with_capacity(n);
    let base = (0.4_f64, 0.9_f64);
    let theta_step = 2.0 * std::f64::consts::PI / n as f64;
    for i in 0..n
    {
        let angle = theta_step * i as f64;
        let r = base.0.hypot(base.1);
        let phi = base.1.atan2(base.0);
        let total = phi + angle;
        z.push((r * total.cos(), r * total.sin()));
    }

    // Évaluation de p(z) (complexe) par Horner
    let eval_complex = |z: (f64, f64)| -> (f64, f64) {
        let mut acc = (0.0_f64, 0.0_f64);
        for &c in monic.iter().rev()
        {
            // acc = acc * z + c
            let (ar, ai) = acc;
            let (zr, zi) = z;
            let nr = ar * zr - ai * zi + c;
            let ni = ar * zi + ai * zr;
            acc = (nr, ni);
        }
        acc
    };

    let mut last_max_step = f64::INFINITY;
    for _ in 0..max_iter
    {
        let mut max_step = 0.0_f64;
        for i in 0..n
        {
            // Calcule p(z_i)
            let pz = eval_complex(z[i]);
            // Calcule le produit Π_{j != i} (z_i - z_j)
            let mut denom = (1.0_f64, 0.0_f64);
            for j in 0..n
            {
                if i == j
                {
                    continue;
                }
                let dr = z[i].0 - z[j].0;
                let di = z[i].1 - z[j].1;
                // denom *= (dr, di)
                let (or_, oi) = denom;
                denom = (or_ * dr - oi * di, or_ * di + oi * dr);
            }
            let dmag = denom.0.hypot(denom.1);
            if dmag < 1e-30
            {
                continue; // racines confondues — saute cette itération
            }
            // step = p(z) / denom  (division complexe)
            let nr = (pz.0 * denom.0 + pz.1 * denom.1) / (dmag * dmag);
            let ni = (pz.1 * denom.0 - pz.0 * denom.1) / (dmag * dmag);
            let step_mag = nr.hypot(ni);
            // `f64::max` treats NaN as the *smaller* operand (IEEE 754
            // minNum/maxNum semantics), so `max_step.max(step_mag)` would
            // silently drop a NaN step instead of propagating it — a
            // catastrophic-cancellation breakdown (huge coefficients, e.g.
            // the classic degree-20 Wilkinson polynomial expanded to
            // monomial form) would then report `last_max_step` near 0 and
            // look "converged" while every root is NaN. Reject explicitly.
            if !step_mag.is_finite()
            {
                return Err(SolverError::NanDetected {
                    iter: i,
                    value: step_mag,
                });
            }
            z[i].0 -= nr;
            z[i].1 -= ni;
            max_step = max_step.max(step_mag);
        }
        last_max_step = max_step;
        if max_step < tol
        {
            return Ok((z, max_step));
        }
    }
    // Hors-tolérance — `durand_kerner` renvoie quand même la meilleure
    // estimation ; `durand_kerner_strict` la rejette via `last_max_step`.
    Ok((z, last_max_step))
}

/// Alias par défaut pour `durand_kerner` avec tolérances raisonnables.
pub fn roots(p: &Polynomial) -> SolverResult<Vec<(f64, f64)>> {
    durand_kerner(p, 200, 1e-12)
}

/// Filtre les racines réelles : celles dont la partie imaginaire est < eps.
/// Trie par ordre croissant.
pub fn real_roots(p: &Polynomial, eps: f64) -> SolverResult<Vec<f64>> {
    let all = roots(p)?;
    let reals: Vec<f64> = all
        .into_iter()
        .filter(|&(_, im)| im.abs() < eps)
        .map(|(re, _)| re)
        .collect();
    Ok(sort_and_dedup_reals(reals))
}

/// Trie par ordre croissant puis dédupe les racines réelles.
///
/// Utilise `total_cmp` plutôt que `partial_cmp().unwrap()` afin de garder un
/// ordre total déterministe même si une racine dégénère en `NaN` (ce qui peut
/// arriver sur des polynômes mal conditionnés) : `unwrap` paniquerait alors.
fn sort_and_dedup_reals(mut reals: Vec<f64>) -> Vec<f64> {
    reals.sort_by(|a, b| a.total_cmp(b));
    // Dédupe (deux racines complexes conjuguées peuvent donner deux versions
    // de la même racine réelle si la partie imaginaire est sous epsilon)
    reals.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    reals
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn linear_root() {
        // 2x + 4 = 0  →  x = -2
        let p = Polynomial::new(vec![4.0, 2.0]);
        let r = real_roots(&p, 1e-8).unwrap();
        assert_eq!(r.len(), 1);
        assert_relative_eq!(r[0], -2.0, epsilon = 1e-10);
    }

    #[test]
    fn quadratic_two_real_roots() {
        // x² - 3x + 2 = 0  →  x = 1 et x = 2
        let p = Polynomial::new(vec![2.0, -3.0, 1.0]);
        let r = real_roots(&p, 1e-8).unwrap();
        assert_eq!(r.len(), 2);
        assert_relative_eq!(r[0], 1.0, epsilon = 1e-9);
        assert_relative_eq!(r[1], 2.0, epsilon = 1e-9);
    }

    #[test]
    fn quadratic_complex_roots() {
        // x² + 1 = 0  →  pas de racine réelle, racines ±i
        let p = Polynomial::new(vec![1.0, 0.0, 1.0]);
        let r = real_roots(&p, 1e-8).unwrap();
        assert!(r.is_empty());
        // Mais en complexe
        let rc = roots(&p).unwrap();
        assert_eq!(rc.len(), 2);
        // Une racine doit avoir une partie imaginaire proche de +1, l'autre -1
        let imags: Vec<f64> = rc.iter().map(|(_, im)| *im).collect();
        let has_pos_i = imags.iter().any(|&im| (im - 1.0).abs() < 1e-6);
        let has_neg_i = imags.iter().any(|&im| (im + 1.0).abs() < 1e-6);
        assert!(has_pos_i && has_neg_i);
    }

    #[test]
    fn cubic_three_real() {
        // (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6
        let p = Polynomial::from_descending(vec![1.0, -6.0, 11.0, -6.0]);
        let r = real_roots(&p, 1e-6).unwrap();
        assert_eq!(r.len(), 3);
        assert_relative_eq!(r[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(r[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(r[2], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn cubic_irrational_root() {
        // x³ - 2x - 5 = 0, une seule racine réelle ≈ 2.0945514815
        let p = Polynomial::from_descending(vec![1.0, 0.0, -2.0, -5.0]);
        let r = real_roots(&p, 1e-6).unwrap();
        assert_eq!(r.len(), 1);
        assert_relative_eq!(r[0], 2.094_551_481_542_326_6, epsilon = 1e-6);
    }

    #[test]
    fn sort_with_nan_does_not_panic() {
        // Régression : avec `partial_cmp().unwrap()`, un NaN dans la liste des
        // racines réelles (possible sur un polynôme mal conditionné, où la
        // partie imaginaire ~0 mais la partie réelle dégénère) faisait paniquer
        // le tri. `total_cmp` garantit un ordre total sans panique.
        let sorted = sort_and_dedup_reals(vec![3.0, f64::NAN, 1.0, 2.0]);
        // Les valeurs finies restent triées ; NaN est ordonné de façon
        // déterministe (en fin de liste) sans panique.
        assert!(sorted.len() >= 3);
        let finite: Vec<f64> = sorted.iter().copied().filter(|x| x.is_finite()).collect();
        assert_eq!(finite, vec![1.0, 2.0, 3.0]);
        assert!(sorted.iter().any(|x| x.is_nan()));
    }

    #[test]
    fn degree_5() {
        // (x-1)(x-2)(x-3)(x-4)(x-5)
        let mut p = Polynomial::from_descending(vec![1.0, -1.0]);
        for k in 2..=5
        {
            // Multiplie p par (x - k)
            let q = Polynomial::from_descending(vec![1.0, -(k as f64)]);
            let mut new_coeffs = vec![0.0; p.coeffs.len() + q.coeffs.len() - 1];
            for (i, &a) in p.coeffs.iter().enumerate()
            {
                for (j, &b) in q.coeffs.iter().enumerate()
                {
                    new_coeffs[i + j] += a * b;
                }
            }
            p = Polynomial::new(new_coeffs);
        }
        let r = real_roots(&p, 1e-5).unwrap();
        assert_eq!(r.len(), 5);
        for (i, expected) in (1..=5).enumerate()
        {
            assert_relative_eq!(r[i], expected as f64, epsilon = 1e-4);
        }
    }

    /// Builds `Π_{k=1}^{degree} (x - k)` in monomial form (the Wilkinson
    /// polynomial family) — huge, ill-conditioned coefficients at higher
    /// degree, which is exactly what breaks a monomial-basis root finder.
    fn wilkinson_style(degree: i64) -> Polynomial {
        let mut p = Polynomial::from_descending(vec![1.0, -1.0]);
        for k in 2..=degree
        {
            let q = Polynomial::from_descending(vec![1.0, -(k as f64)]);
            let mut new_coeffs = vec![0.0; p.coeffs.len() + q.coeffs.len() - 1];
            for (i, &a) in p.coeffs.iter().enumerate()
            {
                for (j, &b) in q.coeffs.iter().enumerate()
                {
                    new_coeffs[i + j] += a * b;
                }
            }
            p = Polynomial::new(new_coeffs);
        }
        p
    }

    /// Regression test for a P2 audit finding: `f64::max` treats NaN as the
    /// *smaller* operand, so a catastrophic-cancellation breakdown used to
    /// get silently absorbed into `max_step` and reported as "converged"
    /// with every root equal to NaN. The classic degree-20 Wilkinson
    /// polynomial — famous for defeating naive root-finders on its monomial
    /// coefficients — reproduces the breakdown; it must now surface as
    /// `NanDetected` instead of fake convergence.
    #[test]
    fn wilkinson_degree_20_reports_nan_instead_of_fake_convergence() {
        let p = wilkinson_style(20);
        let result = durand_kerner(&p, 500, 1e-6);
        assert!(
            matches!(result, Err(SolverError::NanDetected { .. })),
            "expected NanDetected on the classic Wilkinson breakdown, got {result:?}"
        );
    }

    /// `durand_kerner` stays best-effort on ordinary (non-NaN) non-convergence;
    /// `durand_kerner_strict` reports the same case as `NoConvergence`.
    #[test]
    fn durand_kerner_strict_reports_non_convergence() {
        let p = wilkinson_style(6);
        // A single iteration is nowhere near enough to converge, but stays
        // numerically tame (unlike degree 20): no NaN breakdown, just an
        // ordinary "ran out of iterations" case.
        let lenient = durand_kerner(&p, 1, 1e-12);
        assert!(
            lenient.is_ok(),
            "best-effort variant must not error: {lenient:?}"
        );

        let strict = durand_kerner_strict(&p, 1, 1e-12);
        assert!(
            matches!(strict, Err(SolverError::NoConvergence { .. })),
            "expected NoConvergence, got {strict:?}"
        );
    }
}
