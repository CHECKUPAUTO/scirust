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

/// Backward de GELU (approximation tanh, comme le forward de
/// [`crate::activations::gelu_scalar`]) : `dx = dy · gelu'(x)`.
///
/// Avec `g(x) = √(2/π)·(x + 0.044715·x³)` et `t = tanh(g)` :
/// `gelu'(x) = 0.5·(1 + t) + 0.5·x·(1 − t²)·g'(x)`,
/// `g'(x) = √(2/π)·(1 + 3·0.044715·x²)`.
pub fn gelu_backward(x: &[f32], dy: &[f32], dx: &mut [f32]) {
    assert_eq!(x.len(), dy.len(), "gelu_backward: length");
    assert_eq!(x.len(), dx.len(), "gelu_backward: length");
    const C0: f32 = 0.797_884_6; // √(2/π)
    const C1: f32 = 0.044_715;
    for i in 0..x.len()
    {
        let xi = x[i];
        let g = C0 * (xi + C1 * xi * xi * xi);
        let t = g.tanh();
        let gp = C0 * (1.0 + 3.0 * C1 * xi * xi);
        let deriv = 0.5 * (1.0 + t) + 0.5 * xi * (1.0 - t * t) * gp;
        dx[i] = dy[i] * deriv;
    }
}

/// Backward de LayerNorm (`y = (x − μ)/√(σ²+eps)·γ + β`), par ligne (`rows × d`).
/// Produit `dx` et **accumule** `dgamma`, `dbeta` (sommés sur les lignes ; mets-
/// les à zéro avant si tu ne veux pas d'accumulation).
///
/// Soit `x̂ = (x − μ)/σ` et `dŷ_i = dY_i·γ_i`. Alors, par ligne :
/// `dβ_i += dY_i`, `dγ_i += dY_i·x̂_i`, et
/// `dx_i = (1/(d·σ))·[ d·dŷ_i − Σ_k dŷ_k − x̂_i·Σ_k dŷ_k·x̂_k ]`.
#[allow(clippy::too_many_arguments)]
pub fn layernorm_backward(
    x: &[f32],
    rows: usize,
    d: usize,
    gamma: &[f32],
    eps: f32,
    dy: &[f32],
    dx: &mut [f32],
    dgamma: &mut [f32],
    dbeta: &mut [f32],
) {
    assert_eq!(x.len(), rows * d, "layernorm_backward: X shape");
    assert_eq!(gamma.len(), d, "layernorm_backward: gamma shape");
    assert_eq!(dy.len(), rows * d, "layernorm_backward: dY shape");
    assert_eq!(dx.len(), rows * d, "layernorm_backward: dX shape");
    assert_eq!(dgamma.len(), d, "layernorm_backward: dgamma shape");
    assert_eq!(dbeta.len(), d, "layernorm_backward: dbeta shape");

    let df = d as f32;
    for r in 0..rows
    {
        let xr = &x[r * d..r * d + d];
        let dyr = &dy[r * d..r * d + d];
        let dxr = &mut dx[r * d..r * d + d];

        let mean = xr.iter().sum::<f32>() / df;
        let var = xr.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / df;
        let sigma = (var + eps).sqrt();
        let inv = 1.0 / sigma;

        // Sommes de réduction sur la ligne.
        let mut sum_dyh = 0.0f32; // Σ dŷ_k
        let mut sum_dyh_xh = 0.0f32; // Σ dŷ_k·x̂_k
        for k in 0..d
        {
            let xh = (xr[k] - mean) * inv;
            let dyh = dyr[k] * gamma[k];
            sum_dyh += dyh;
            sum_dyh_xh += dyh * xh;
        }

        for j in 0..d
        {
            let xh = (xr[j] - mean) * inv;
            let dyh = dyr[j] * gamma[j];
            dxr[j] = inv / df * (df * dyh - sum_dyh - xh * sum_dyh_xh);
            dgamma[j] += dyr[j] * xh;
            dbeta[j] += dyr[j];
        }
    }
}

/// Backward de softmax **par ligne** : reçoit la **sortie** `y = softmax(x)`
/// (`rows × d`) et `dy` (gradient de la sortie), écrit `dx`.
/// `dx_i = y_i·(dY_i − Σ_k dY_k·y_k)` (produit par la jacobienne du softmax).
pub fn softmax_backward(y: &[f32], rows: usize, d: usize, dy: &[f32], dx: &mut [f32]) {
    assert_eq!(y.len(), rows * d, "softmax_backward: Y shape");
    assert_eq!(dy.len(), rows * d, "softmax_backward: dY shape");
    assert_eq!(dx.len(), rows * d, "softmax_backward: dX shape");
    for r in 0..rows
    {
        let yr = &y[r * d..r * d + d];
        let dyr = &dy[r * d..r * d + d];
        let dxr = &mut dx[r * d..r * d + d];
        let dot: f32 = (0..d).map(|k| dyr[k] * yr[k]).sum();
        for j in 0..d
        {
            dxr[j] = yr[j] * (dyr[j] - dot);
        }
    }
}

/// Backward de l'attention **une tête** `O = softmax(scale·Q·Kᵀ)·V`.
///
/// À partir de `dout` (gradient de `O`, `s×d`), écrit `dq` (`s×d`), `dk`
/// (`t×d`), `dv` (`t×d`). `Q` est `s×d`, `K`/`V` sont `t×d`. Recompute
/// `P = softmax(scale·Q·Kᵀ)` puis applique la chaîne :
/// `dV = Pᵀ·dO`, `dP = dO·Vᵀ`, `dScores = softmax'(P, dP)`,
/// `dQ = scale·dScores·K`, `dK = scale·dScoresᵀ·Q`. Les produits passent par le
/// GEMM tuilé ; le softmax backward est [`softmax_backward`].
#[allow(clippy::too_many_arguments)]
pub fn attention_backward(
    q: &[f32],
    s: usize,
    d: usize,
    k: &[f32],
    t: usize,
    v: &[f32],
    scale: f32,
    dout: &[f32],
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
) {
    assert_eq!(q.len(), s * d, "attention_backward: Q shape");
    assert_eq!(k.len(), t * d, "attention_backward: K shape");
    assert_eq!(v.len(), t * d, "attention_backward: V shape");
    assert_eq!(dout.len(), s * d, "attention_backward: dO shape");
    assert_eq!(dq.len(), s * d, "attention_backward: dQ shape");
    assert_eq!(dk.len(), t * d, "attention_backward: dK shape");
    assert_eq!(dv.len(), t * d, "attention_backward: dV shape");

    // P = softmax(scale · Q·Kᵀ)  (s×t).
    let kt = transpose(k, t, d); // d×t
    let mut p = vec![0.0f32; s * t];
    sgemm_tiled(
        scale,
        MatrixView::new(q, s, d),
        MatrixView::new(&kt, d, t),
        0.0,
        MatrixViewMut::new(&mut p, s, t),
    );
    crate::attention::softmax_rows(&mut p, s, t);

    // dV = Pᵀ · dO  (t×d).
    let pt = transpose(&p, s, t); // t×s
    sgemm_tiled(
        1.0,
        MatrixView::new(&pt, t, s),
        MatrixView::new(dout, s, d),
        0.0,
        MatrixViewMut::new(dv, t, d),
    );

    // dP = dO · Vᵀ  (s×t).
    let vt = transpose(v, t, d); // d×t
    let mut dp = vec![0.0f32; s * t];
    sgemm_tiled(
        1.0,
        MatrixView::new(dout, s, d),
        MatrixView::new(&vt, d, t),
        0.0,
        MatrixViewMut::new(&mut dp, s, t),
    );

    // dScores = softmax'(P, dP)  (s×t).
    let mut dscores = vec![0.0f32; s * t];
    softmax_backward(&p, s, t, &dp, &mut dscores);

    // dQ = scale · dScores · K  (s×d).
    sgemm_tiled(
        scale,
        MatrixView::new(&dscores, s, t),
        MatrixView::new(k, t, d),
        0.0,
        MatrixViewMut::new(dq, s, d),
    );

    // dK = scale · dScoresᵀ · Q  (t×d).
    let dst = transpose(&dscores, s, t); // t×s
    sgemm_tiled(
        scale,
        MatrixView::new(&dst, t, s),
        MatrixView::new(q, s, d),
        0.0,
        MatrixViewMut::new(dk, t, d),
    );
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

    // ---- Références forward supplémentaires ----

    fn gelu_fwd(x: &[f32]) -> Vec<f32> {
        const C0: f32 = 0.797_884_6;
        const C1: f32 = 0.044_715;
        x.iter()
            .map(|&v| 0.5 * v * (1.0 + (C0 * (v + C1 * v * v * v)).tanh()))
            .collect()
    }

    fn layernorm_fwd(x: &[f32], rows: usize, d: usize, g: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * d];
        for r in 0..rows
        {
            let row = &x[r * d..r * d + d];
            let mean = row.iter().sum::<f32>() / d as f32;
            let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for j in 0..d
            {
                y[r * d + j] = (row[j] - mean) * inv * g[j] + b[j];
            }
        }
        y
    }

    fn softmax_fwd(x: &[f32], rows: usize, d: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * d];
        for r in 0..rows
        {
            let row = &x[r * d..r * d + d];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..d
            {
                let e = (row[j] - m).exp();
                y[r * d + j] = e;
                sum += e;
            }
            for j in 0..d
            {
                y[r * d + j] /= sum;
            }
        }
        y
    }

    #[test]
    fn gelu_backward_gradcheck() {
        let x: Vec<f32> = (0..50).map(|i| (i as f32 * 0.23).sin() * 3.0).collect();
        let seed: Vec<f32> = (0..50).map(|i| (i as f32 * 0.15).cos() + 0.4).collect();
        let mut dx = vec![0.0f32; x.len()];
        gelu_backward(&x, &seed, &mut dx);
        let g = num_grad(&x, &seed, 1e-3, gelu_fwd);
        assert_close(&dx, &g, 2e-2, "gelu dX");
    }

    #[test]
    fn layernorm_backward_gradcheck() {
        let (rows, d) = (3usize, 7usize);
        let x: Vec<f32> = (0..rows * d)
            .map(|i| (i as f32 * 0.29).sin() * 2.0 + 1.0)
            .collect();
        let g: Vec<f32> = (0..d).map(|i| 0.5 + i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..d).map(|i| i as f32 * 0.05 - 0.1).collect();
        let eps = 1e-5f32;
        let seed: Vec<f32> = (0..rows * d)
            .map(|i| (i as f32 * 0.17).cos() + 0.2)
            .collect();

        let mut dx = vec![0.0f32; rows * d];
        let mut dg = vec![0.0f32; d];
        let mut db = vec![0.0f32; d];
        layernorm_backward(&x, rows, d, &g, eps, &seed, &mut dx, &mut dg, &mut db);

        let gx = num_grad(&x, &seed, 1e-3, |xx| {
            layernorm_fwd(xx, rows, d, &g, &b, eps)
        });
        assert_close(&dx, &gx, 3e-2, "layernorm dX");
        let gg = num_grad(&g, &seed, 1e-3, |gg| {
            layernorm_fwd(&x, rows, d, gg, &b, eps)
        });
        assert_close(&dg, &gg, 3e-2, "layernorm dgamma");
        let gb = num_grad(&b, &seed, 1e-3, |bb| {
            layernorm_fwd(&x, rows, d, &g, bb, eps)
        });
        assert_close(&db, &gb, 3e-2, "layernorm dbeta");
    }

    #[test]
    fn softmax_backward_gradcheck() {
        let (rows, d) = (4usize, 6usize);
        let x: Vec<f32> = (0..rows * d)
            .map(|i| (i as f32 * 0.31).sin() * 2.0)
            .collect();
        let seed: Vec<f32> = (0..rows * d)
            .map(|i| (i as f32 * 0.19).cos() + 0.3)
            .collect();

        let y = softmax_fwd(&x, rows, d);
        let mut dx = vec![0.0f32; rows * d];
        softmax_backward(&y, rows, d, &seed, &mut dx);

        let gx = num_grad(&x, &seed, 1e-3, |xx| softmax_fwd(xx, rows, d));
        assert_close(&dx, &gx, 2e-2, "softmax dX");
    }

    #[test]
    fn attention_backward_gradcheck() {
        use crate::attention::attention;
        let (s, d, t) = (3usize, 4usize, 5usize);
        let scale = 1.0 / (d as f32).sqrt();
        let q: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.11).sin()).collect();
        let k: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.07).cos()).collect();
        let v: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.05) - 0.3).collect();
        let seed: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.13).cos() + 0.2).collect(); // = dO

        let mut dq = vec![0.0f32; s * d];
        let mut dk = vec![0.0f32; t * d];
        let mut dv = vec![0.0f32; t * d];
        attention_backward(&q, s, d, &k, t, &v, scale, &seed, &mut dq, &mut dk, &mut dv);

        // Perte L = Σ O·seed ; O = attention(Q,K,V).
        let fwd = |qq: &[f32], kk: &[f32], vv: &[f32]| -> Vec<f32> {
            let mut o = vec![0.0f32; s * d];
            attention(qq, s, d, kk, t, vv, scale, &mut o);
            o
        };
        let num = |input: &[f32], f: &dyn Fn(&[f32]) -> Vec<f32>| -> Vec<f32> {
            num_grad(input, &seed, 1e-3, f)
        };

        let gq = num(&q, &|qq| fwd(qq, &k, &v));
        assert_close(&dq, &gq, 3e-2, "attention dQ");
        let gk = num(&k, &|kk| fwd(&q, kk, &v));
        assert_close(&dk, &gk, 3e-2, "attention dK");
        let gv = num(&v, &|vv| fwd(&q, &k, vv));
        assert_close(&dv, &gv, 3e-2, "attention dV");
    }
}
