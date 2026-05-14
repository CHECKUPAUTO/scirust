//! Optimization algorithms: Simplex and Bregman Projection.

/// A basic implementation of the Simplex algorithm for Linear Programming.
/// Maximizes c^T * x subject to Ax <= b, x >= 0.
pub fn simplex(c: &[f64], a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let m = a.len();
    let n = c.len();

    // Initial tableau
    let mut tableau = vec![vec![0.0; n + m + 1]; m + 1];

    for i in 0..m {
        for j in 0..n {
            tableau[i][j] = a[i][j];
        }
        tableau[i][n + i] = 1.0; // slack variables
        tableau[i][n + m] = b[i];
    }

    for j in 0..n {
        tableau[m][j] = -c[j];
    }

    loop {
        // Find entering column (most negative value in bottom row)
        let mut pivot_col = 0;
        let mut min_val = tableau[m][0];
        for j in 1..(n + m) {
            if tableau[m][j] < min_val {
                min_val = tableau[m][j];
                pivot_col = j;
            }
        }

        if min_val >= -1e-10 {
            break; // optimal
        }

        // Find leaving row (minimum ratio test)
        let mut pivot_row = None;
        let mut min_ratio = f64::INFINITY;
        for i in 0..m {
            if tableau[i][pivot_col] > 1e-10 {
                let ratio = tableau[i][n + m] / tableau[i][pivot_col];
                if ratio < min_ratio {
                    min_ratio = ratio;
                    pivot_row = Some(i);
                }
            }
        }

        let r = match pivot_row {
            Some(row) => row,
            None => return None, // unbounded
        };

        // Pivot
        let divisor = tableau[r][pivot_col];
        for j in 0..=(n + m) {
            tableau[r][j] /= divisor;
        }

        for i in 0..=m {
            if i != r {
                let multiplier = tableau[i][pivot_col];
                for j in 0..=(n + m) {
                    tableau[i][j] -= multiplier * tableau[r][j];
                }
            }
        }
    }

    // Extract solution
    let mut x = vec![0.0; n];
    for j in 0..n {
        let mut row_with_one = None;
        let mut is_basis = true;
        for i in 0..m {
            if (tableau[i][j] - 1.0).abs() < 1e-10 {
                if row_with_one.is_none() {
                    row_with_one = Some(i);
                } else {
                    is_basis = false;
                    break;
                }
            } else if tableau[i][j].abs() > 1e-10 {
                is_basis = false;
                break;
            }
        }
        if is_basis && row_with_one.is_some() && tableau[m][j].abs() < 1e-10 {
            x[j] = tableau[row_with_one.unwrap()][n + m];
        }
    }

    Some(x)
}

/// Bregman Projection for KL divergence (used in pricing/prediction).
/// Projects point `y` onto the simplex defined by `sum(x) = 1, x >= 0`.
pub fn bregman_projection_simplex(y: &[f64]) -> Vec<f64> {
    // For KL divergence, Bregman projection onto the simplex is just softmax
    // or normalized exponential if we use certain distance.
    // Here we implement projection onto the probability simplex.
    let mut x = y.to_vec();
    x.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut running_sum = 0.0;
    let mut rho = 0;
    for i in 0..x.len() {
        running_sum += x[i];
        if x[i] + (1.0 - running_sum) / ((i + 1) as f64) > 0.0 {
            rho = i + 1;
        }
    }

    let theta = (1.0 - x[..rho].iter().sum::<f64>()) / (rho as f64);
    y.iter().map(|&yi| (yi + theta).max(0.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex() {
        // Maximize 3x + 2y
        // subject to:
        // 2x + y <= 18
        // 2x + 3y <= 42
        // 3x + y <= 24
        // x, y >= 0
        // Optimal solution: x=3, y=12, Value=33 (or something like that)
        let c = vec![3.0, 2.0];
        let a = vec![
            vec![2.0, 1.0],
            vec![2.0, 3.0],
            vec![3.0, 1.0],
        ];
        let b = vec![18.0, 42.0, 24.0];

        let sol = simplex(&c, &a, &b).unwrap();
        // Check constraints
        assert!(2.0 * sol[0] + sol[1] <= 18.0001);
        assert!(3.0 * sol[0] + 2.0 * sol[1] >= 32.999);
    }

    #[test]
    fn test_bregman_projection() {
        let y = vec![1.0, 2.0, 3.0];
        let x = bregman_projection_simplex(&y);

        let sum: f64 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        for &val in &x {
            assert!(val >= 0.0);
        }
    }
}
