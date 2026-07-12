//! # Backward pass — gradients des noyaux (autodiff manuel)
//!
//! Rétropropagation des briques d'entraînement, complément des `forward` du
//! crate. Chaque fonction reçoit le gradient de la sortie (`dy`, même forme que
//! la sortie du `forward`) et produit les gradients des entrées/paramètres.
//!
//! * **`linear_backward`** — pour `Y = X·W + b` : `dX = dY·Wᵀ`, `dW = Xᵀ·dY`,
//!   `db = Σ_lignes dY`. Les deux produits matriciels réutilisent le **GEMM
//!   tuilé/packé** ([`crate::gemm::sgemm_tiled`]) — le backward hérite donc de
//!   l'accélération AVX-512/… du forward.
//! * **`relu_backward`**, **`silu_backward`** — dérivées élément par élément.
//! * **`rmsnorm_backward`** — gradient par ligne de la RMSNorm.
//!
//! Tous les gradients sont **vérifiés par différences finies centrées**
//! (gradcheck) dans les tests, la garantie de justesse d'un backward.

use crate::gemm::sgemm_tiled;
use crate::matrix::view::{MatrixView, MatrixViewMut};

/// Transpose `a` (`rows×cols`, row-major) → `cols×rows`.
fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * cols];
    for i in 0..rows
    {
        for j in 0..cols
        {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

/// Backward d'une couche linéaire `Y = X·W + b` (broadcast de `b` par ligne).
///
/// Formes : `X` `m×k`, `W` `k×n`, `b`/`db` `n`, `dY` `m×n`, `dX` `m×k`,
/// `dW` `k×n`. Calcule `dX = dY·Wᵀ`, `dW = Xᵀ·dY`, `db = Σ_i dY[i,:]`.
#[allow(clippy::too_many_arguments)]
pub fn linear_backward(
    x: &[f32],
    m: usize,
    k: usize,
    w: &[f32],
    n: usize,
    dy: &[f32],
    dx: &mut [f32],
    dw: &mut [f32],
    db: &mut [f32],
) {
    assert_eq!(x.len(), m * k, "linear_backward: X shape");
    assert_eq!(w.len(), k * n, "linear_backward: W shape");
    assert_eq!(dy.len(), m * n, "linear_backward: dY shape");
    assert_eq!(dx.len(), m * k, "linear_backward: dX shape");
    assert_eq!(dw.len(), k * n, "linear_backward: dW shape");
    assert_eq!(db.len(), n, "linear_backward: db shape");

    // dX = dY(m×n) · Wᵀ(n×k)
    let wt = transpose(w, k, n); // n×k
    sgemm_tiled(
        1.0,
        MatrixView::new(dy, m, n),
        MatrixView::new(&wt, n, k),
        0.0,
        MatrixViewMut::new(dx, m, k),
    );

    // dW = Xᵀ(k×m) · dY(m×n)
    let xt = transpose(x, m, k); // k×m
    sgemm_tiled(
        1.0,
        MatrixView::new(&xt, k, m),
        MatrixView::new(dy, m, n),
        0.0,
        MatrixViewMut::new(dw, k, n),
    );

    // db[j] = Σ_i dY[i,j]
    for (j, dbj) in db.iter_mut().enumerate()
    {
        let mut acc = 0.0f32;
        for i in 0..m
        {
            acc += dy[i * n + j];
        }
        *dbj = acc;
    }
}

/// Backward de ReLU : `dx[i] = dy[i]` si `x[i] > 0`, sinon `0`.
pub fn relu_backward(x: &[f32], dy: &[f32], dx: &mut [f32]) {
    assert_eq!(x.len(), dy.len(), "relu_backward: length");
    assert_eq!(x.len(), dx.len(), "relu_backward: length");
    for i in 0..x.len()
    {
        dx[i] = if x[i] > 0.0 { dy[i] } else { 0.0 };
    }
}

/// Backward de SiLU (`f(x) = x·σ(x)`) : `dx = dy · f'(x)` avec
/// `f'(x) = σ(x) · (1 + x·(1 − σ(x)))`.
pub fn silu_backward(x: &[f32], dy: &[f32], dx: &mut [f32]) {
    assert_eq!(x.len(), dy.len(), "silu_backward: length");
    assert_eq!(x.len(), dx.len(), "silu_backward: length");
    for i in 0..x.len()
    {
        let s = 1.0 / (1.0 + (-x[i]).exp());
        let deriv = s * (1.0 + x[i] * (1.0 - s));
        dx[i] = dy[i] * deriv;
    }
}

/// Backward de RMSNorm (`y = x / √(moyenne(x²)+eps) · γ`), par ligne
/// (`rows × d`). Produit `dx` (gradient de l'entrée) et **accumule** `dg`
/// (gradient du gain `γ`, sommé sur les lignes — mets `dg` à zéro avant si tu
/// ne veux pas d'accumulation).
///
/// Formule : soit `inv = (moyenne(x²)+eps)^(−1/2)` et `c = Σ_i dY_i·γ_i·x_i`.
/// Alors `dx_j = inv·γ_j·dY_j − (inv³/d)·x_j·c` et `dg_j += dY_j·x_j·inv`.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_backward(
    x: &[f32],
    rows: usize,
    d: usize,
    gamma: &[f32],
    eps: f32,
    dy: &[f32],
    dx: &mut [f32],
    dg: &mut [f32],
) {
    assert_eq!(x.len(), rows * d, "rmsnorm_backward: X shape");
    assert_eq!(gamma.len(), d, "rmsnorm_backward: gamma shape");
    assert_eq!(dy.len(), rows * d, "rmsnorm_backward: dY shape");
    assert_eq!(dx.len(), rows * d, "rmsnorm_backward: dX shape");
    assert_eq!(dg.len(), d, "rmsnorm_backward: dg shape");

    for r in 0..rows
    {
        let xr = &x[r * d..r * d + d];
        let dyr = &dy[r * d..r * d + d];
        let dxr = &mut dx[r * d..r * d + d];

        let ss: f32 = xr.iter().map(|&v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        let inv3 = inv * inv * inv;
        // c = Σ dY_i γ_i x_i
        let c: f32 = (0..d).map(|i| dyr[i] * gamma[i] * xr[i]).sum();

        for j in 0..d
        {
            dxr[j] = inv * gamma[j] * dyr[j] - (inv3 / d as f32) * xr[j] * c;
            dg[j] += dyr[j] * xr[j] * inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Références forward (indépendantes) pour le gradcheck ----

    fn linear_fwd(x: &[f32], m: usize, k: usize, w: &[f32], n: usize, b: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; m * n];
        for i in 0..m
        {
            for j in 0..n
            {
                let mut acc = b[j];
                for p in 0..k
                {
                    acc += x[i * k + p] * w[p * n + j];
                }
                y[i * n + j] = acc;
            }
        }
        y
    }

    fn silu_fwd(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
    }

    fn relu_fwd(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| v.max(0.0)).collect()
    }

    fn rmsnorm_fwd(x: &[f32], rows: usize, d: usize, g: &[f32], eps: f32) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * d];
        for r in 0..rows
        {
            let row = &x[r * d..r * d + d];
            let inv = 1.0 / (row.iter().map(|&v| v * v).sum::<f32>() / d as f32 + eps).sqrt();
            for j in 0..d
            {
                y[r * d + j] = row[j] * inv * g[j];
            }
        }
        y
    }

    /// Gradient numérique de `L = Σ f(input) · seed` par rapport à `input[i]`,
    /// par différences finies centrées.
    fn num_grad(input: &[f32], seed: &[f32], h: f32, f: impl Fn(&[f32]) -> Vec<f32>) -> Vec<f32> {
        let n = input.len();
        let mut g = vec![0.0f32; n];
        let mut buf = input.to_vec();
        for i in 0..n
        {
            let orig = buf[i];
            buf[i] = orig + h;
            let yp = f(&buf);
            buf[i] = orig - h;
            let ym = f(&buf);
            buf[i] = orig;
            let lp: f32 = yp.iter().zip(seed).map(|(a, b)| a * b).sum();
            let lm: f32 = ym.iter().zip(seed).map(|(a, b)| a * b).sum();
            g[i] = (lp - lm) / (2.0 * h);
        }
        g
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: len");
        for i in 0..a.len()
        {
            assert!(
                (a[i] - b[i]).abs() <= tol * (1.0 + b[i].abs()),
                "{ctx}: idx {i}: analytique {} vs numérique {}",
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn linear_backward_gradcheck() {
        let (m, k, n) = (4usize, 5usize, 3usize);
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.11).sin()).collect();
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 0.2).collect();
        // seed = gradient de la sortie (dY).
        let seed: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.3).sin() + 0.2).collect();

        let mut dx = vec![0.0f32; m * k];
        let mut dw = vec![0.0f32; k * n];
        let mut db = vec![0.0f32; n];
        linear_backward(&x, m, k, &w, n, &seed, &mut dx, &mut dw, &mut db);

        let h = 1e-3;
        // dX
        let g_x = num_grad(&x, &seed, h, |xx| linear_fwd(xx, m, k, &w, n, &b));
        assert_close(&dx, &g_x, 2e-2, "dX");
        // dW
        let g_w = num_grad(&w, &seed, h, |ww| linear_fwd(&x, m, k, ww, n, &b));
        assert_close(&dw, &g_w, 2e-2, "dW");
        // db
        let g_b = num_grad(&b, &seed, h, |bb| linear_fwd(&x, m, k, &w, n, bb));
        assert_close(&db, &g_b, 2e-2, "db");
    }

    #[test]
    fn relu_backward_gradcheck() {
        // On évite x proche de 0 (non différentiable).
        let x: Vec<f32> = (0..40)
            .map(|i| (i as f32 * 0.37).sin() * 3.0 + 0.5)
            .collect();
        let seed: Vec<f32> = (0..40).map(|i| (i as f32 * 0.2).cos() + 0.3).collect();
        let mut dx = vec![0.0f32; x.len()];
        relu_backward(&x, &seed, &mut dx);
        let g = num_grad(&x, &seed, 1e-3, relu_fwd);
        assert_close(&dx, &g, 2e-2, "relu dX");
    }

    #[test]
    fn silu_backward_gradcheck() {
        let x: Vec<f32> = (0..50).map(|i| (i as f32 * 0.23).sin() * 4.0).collect();
        let seed: Vec<f32> = (0..50).map(|i| (i as f32 * 0.15).cos() + 0.4).collect();
        let mut dx = vec![0.0f32; x.len()];
        silu_backward(&x, &seed, &mut dx);
        let g = num_grad(&x, &seed, 1e-3, silu_fwd);
        assert_close(&dx, &g, 2e-2, "silu dX");
    }

    #[test]
    fn rmsnorm_backward_gradcheck() {
        let (rows, d) = (3usize, 7usize);
        let x: Vec<f32> = (0..rows * d)
            .map(|i| (i as f32 * 0.29).sin() * 2.0 + 1.0)
            .collect();
        let g: Vec<f32> = (0..d).map(|i| 0.5 + i as f32 * 0.1).collect();
        let eps = 1e-5f32;
        let seed: Vec<f32> = (0..rows * d)
            .map(|i| (i as f32 * 0.17).cos() + 0.2)
            .collect();

        let mut dx = vec![0.0f32; rows * d];
        let mut dg = vec![0.0f32; d];
        rmsnorm_backward(&x, rows, d, &g, eps, &seed, &mut dx, &mut dg);

        // dX vs numérique (on fait varier x).
        let gx = num_grad(&x, &seed, 1e-3, |xx| rmsnorm_fwd(xx, rows, d, &g, eps));
        assert_close(&dx, &gx, 3e-2, "rmsnorm dX");
        // dγ vs numérique (on fait varier g).
        let gg = num_grad(&g, &seed, 1e-3, |gg| rmsnorm_fwd(&x, rows, d, gg, eps));
        assert_close(&dg, &gg, 3e-2, "rmsnorm dgamma");
    }
}
