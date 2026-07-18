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
//! * **`conv2d_backward`** — convolution 2D **valide** (sans remplissage),
//!   **stride 1** (première brique CNN entraînable de ce module ; une
//!   extension à `stride > 1` suivrait le même schéma de dispersion pour
//!   `dX`, limite documentée plutôt qu'une lacune silencieuse). `dX` est
//!   dispersée exactement comme l'adjointe de la convolution
//!   ([`crate::fixed::conv2d::conv2d_transpose`], son pendant virgule fixe
//!   déterministe) ; `dW` est une corrélation croisée entre `X` et `dY` ;
//!   `db` somme `dY` sur les positions spatiales.
//! * **`max_pool2d_backward`**, **`avg_pool2d_backward`** — `dY` est routée
//!   vers l'unique position maximale de chaque fenêtre (recalculée depuis
//!   `X`, comme `relu_backward` redérive son propre routage) pour le max,
//!   ou distribuée uniformément sur toute la fenêtre pour la moyenne.
//! * **`batch_norm_backward`** — BatchNorm **entraînement** : la
//!   moyenne/variance de chaque canal sont recalculées sur le lot courant,
//!   à la différence de [`crate::fixed::norm::batch_norm`] (inférence,
//!   statistiques figées `running_mean`/`running_var`). Même forme
//!   fermée que `layernorm_backward` (réduction à travers une statistique
//!   partagée), mais réduite sur les `batch·spatial` éléments **dispersés**
//!   de chaque canal plutôt que sur les `d` éléments contigus d'une ligne.
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
///
/// Nécessite `crate::attention` (softmax par ligne) — seule fonction de ce
/// module dépendant de la pile transformer-inference, d'où le `cfg` dédié
/// plutôt que de gater tout le fichier.
#[cfg(feature = "transformer-inference")]
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

/// Backward de la convolution 2D **valide** (sans remplissage), **stride
/// 1** : `Y = conv2d(X, W) + b` (cf. en-tête de module pour la limite au
/// stride 1). `X` : `in_channels×height×width` ; `W` :
/// `out_channels×in_channels×kernel_h×kernel_w` (convention PyTorch
/// `Conv2d`, comme [`crate::fixed::conv2d::conv2d`]) ; `b`/`db` :
/// `out_channels` ; `dY` : `out_channels×height_out×width_out`
/// (`height_out = height − kernel_h + 1`, idem largeur).
///
/// `dX[ci,ih,iw] = Σ_{co,kh,kw} dY[co,oh,ow]·W[co,ci,kh,kw]` sur les
/// positions `(oh,ow)` telles que `oh+kh=ih`, `ow+kw=iw` — dispersion
/// identique à l'adjointe de la convolution (cf. en-tête de module).
/// `dW[co,ci,kh,kw] = Σ_{oh,ow} X[ci,oh+kh,ow+kw]·dY[co,oh,ow]` (corrélation
/// croisée). `db[co] = Σ_{oh,ow} dY[co,oh,ow]`.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward(
    x: &[f32],
    in_channels: usize,
    height: usize,
    width: usize,
    w: &[f32],
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    dy: &[f32],
    dx: &mut [f32],
    dw: &mut [f32],
    db: &mut [f32],
) {
    assert!(
        height >= kernel_h && width >= kernel_w,
        "conv2d_backward: kernel larger than input"
    );
    let height_out = height - kernel_h + 1;
    let width_out = width - kernel_w + 1;
    assert_eq!(
        x.len(),
        in_channels * height * width,
        "conv2d_backward: X shape"
    );
    assert_eq!(
        w.len(),
        out_channels * in_channels * kernel_h * kernel_w,
        "conv2d_backward: W shape"
    );
    assert_eq!(
        dy.len(),
        out_channels * height_out * width_out,
        "conv2d_backward: dY shape"
    );
    assert_eq!(dx.len(), x.len(), "conv2d_backward: dX shape");
    assert_eq!(dw.len(), w.len(), "conv2d_backward: dW shape");
    assert_eq!(db.len(), out_channels, "conv2d_backward: db shape");

    dx.fill(0.0);
    dw.fill(0.0);

    let channel_size = height * width;
    let kernel_size = kernel_h * kernel_w;
    let spatial_out = height_out * width_out;

    for co in 0..out_channels
    {
        let mut b_acc = 0.0f32;
        for oh in 0..height_out
        {
            for ow in 0..width_out
            {
                let dy_val = dy[co * spatial_out + oh * width_out + ow];
                b_acc += dy_val;
                for ci in 0..in_channels
                {
                    let x_ci = &x[ci * channel_size..(ci + 1) * channel_size];
                    let w_co_ci = &w[(co * in_channels + ci) * kernel_size
                        ..(co * in_channels + ci + 1) * kernel_size];
                    let dx_ci = &mut dx[ci * channel_size..(ci + 1) * channel_size];
                    let dw_co_ci = &mut dw[(co * in_channels + ci) * kernel_size
                        ..(co * in_channels + ci + 1) * kernel_size];
                    for kh in 0..kernel_h
                    {
                        for kw in 0..kernel_w
                        {
                            let ih = oh + kh;
                            let iw = ow + kw;
                            dx_ci[ih * width + iw] += dy_val * w_co_ci[kh * kernel_w + kw];
                            dw_co_ci[kh * kernel_w + kw] += x_ci[ih * width + iw] * dy_val;
                        }
                    }
                }
            }
        }
        db[co] = b_acc;
    }
}

/// Backward de max_pool2d (`channels×height×width`, fenêtre
/// `window_h×window_w`, pas `stride_h×stride_w`, valide/sans remplissage) :
/// chaque `dY[c,oh,ow]` est routée **entièrement** vers la position d'entrée
/// qui réalise le maximum de sa fenêtre (recalculé depuis `X` — cf. en-tête
/// de module) ; les autres positions de la fenêtre reçoivent un gradient
/// nul. En cas d'égalité exacte, la **première** position rencontrée (ordre
/// ligne puis colonne) reçoit le gradient (convention arbitraire mais
/// déterministe). Fenêtres chevauchantes (`stride < window`) : `dX`
/// **accumule** les contributions de toutes les fenêtres concernées.
#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_backward(
    x: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    window_h: usize,
    window_w: usize,
    stride_h: usize,
    stride_w: usize,
    dy: &[f32],
    dx: &mut [f32],
) {
    assert!(
        height >= window_h && width >= window_w,
        "max_pool2d_backward: window larger than input"
    );
    let height_out = (height - window_h) / stride_h + 1;
    let width_out = (width - window_w) / stride_w + 1;
    assert_eq!(
        x.len(),
        channels * height * width,
        "max_pool2d_backward: X shape"
    );
    assert_eq!(
        dy.len(),
        channels * height_out * width_out,
        "max_pool2d_backward: dY shape"
    );
    assert_eq!(dx.len(), x.len(), "max_pool2d_backward: dX shape");

    dx.fill(0.0);
    let channel_size = height * width;
    let spatial_out = height_out * width_out;
    for c in 0..channels
    {
        let x_c = &x[c * channel_size..(c + 1) * channel_size];
        let dx_c = &mut dx[c * channel_size..(c + 1) * channel_size];
        for oh in 0..height_out
        {
            for ow in 0..width_out
            {
                let mut best_val = f32::NEG_INFINITY;
                let mut best_idx = 0usize;
                for kh in 0..window_h
                {
                    for kw in 0..window_w
                    {
                        let idx = (oh * stride_h + kh) * width + (ow * stride_w + kw);
                        if x_c[idx] > best_val
                        {
                            best_val = x_c[idx];
                            best_idx = idx;
                        }
                    }
                }
                dx_c[best_idx] += dy[c * spatial_out + oh * width_out + ow];
            }
        }
    }
}

/// Backward d'avg_pool2d (mêmes conventions que [`max_pool2d_backward`]) :
/// `dY[c,oh,ow]/(window_h·window_w)` est distribuée **uniformément** sur
/// toute la fenêtre (aucune donnée d'entrée nécessaire — contrairement au
/// max, la moyenne ne dépend pas de quelle position a la plus grande
/// valeur). Fenêtres chevauchantes : `dX` accumule.
#[allow(clippy::too_many_arguments)]
pub fn avg_pool2d_backward(
    channels: usize,
    height: usize,
    width: usize,
    window_h: usize,
    window_w: usize,
    stride_h: usize,
    stride_w: usize,
    dy: &[f32],
    dx: &mut [f32],
) {
    assert!(
        height >= window_h && width >= window_w,
        "avg_pool2d_backward: window larger than input"
    );
    let height_out = (height - window_h) / stride_h + 1;
    let width_out = (width - window_w) / stride_w + 1;
    assert_eq!(
        dx.len(),
        channels * height * width,
        "avg_pool2d_backward: dX shape"
    );
    assert_eq!(
        dy.len(),
        channels * height_out * width_out,
        "avg_pool2d_backward: dY shape"
    );

    dx.fill(0.0);
    let inv_window = 1.0 / (window_h * window_w) as f32;
    let channel_size = height * width;
    let spatial_out = height_out * width_out;
    for c in 0..channels
    {
        let dx_c = &mut dx[c * channel_size..(c + 1) * channel_size];
        for oh in 0..height_out
        {
            for ow in 0..width_out
            {
                let g = dy[c * spatial_out + oh * width_out + ow] * inv_window;
                for kh in 0..window_h
                {
                    for kw in 0..window_w
                    {
                        dx_c[(oh * stride_h + kh) * width + (ow * stride_w + kw)] += g;
                    }
                }
            }
        }
    }
}

/// Backward de BatchNorm **entraînement** (cf. en-tête de module pour la
/// distinction avec [`crate::fixed::norm::batch_norm`], inférence) :
/// `y[b,c,s] = (x[b,c,s] − μ_c)/√(σ²_c+eps)·γ_c + β_c`, où `μ_c`/`σ²_c`
/// (variance **biaisée**) sont la moyenne/variance du canal `c` sur les
/// `N = batch·spatial` éléments `x[:,c,:]`. `x`/`dy`/`dx` :
/// `batch × channels × spatial` (même convention que
/// [`crate::fixed::norm::batch_norm_batched`]) ; `gamma`/`dgamma`/`dbeta` :
/// `channels`. Produit `dx` et **accumule** `dgamma`, `dbeta` (mets-les à
/// zéro avant si tu ne veux pas d'accumulation).
///
/// Soit `x̂ = (x − μ_c)/σ_c` et `dŷ = dY·γ_c`. Par canal :
/// `dβ_c += Σ dY`, `dγ_c += Σ dY·x̂`, et
/// `dx = (1/(N·σ_c))·[ N·dŷ − Σ dŷ − x̂·Σ dŷ·x̂ ]` — les sommes portant sur
/// les `N` positions `(b,s)` du canal, exactement comme dans
/// `layernorm_backward` (jamais simplifié en `Σ(x−μ)=0` : cette somme est
/// calculée naïvement pour ne pas masquer une éventuelle erreur ailleurs
/// dans la formule).
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_backward(
    x: &[f32],
    batch: usize,
    channels: usize,
    spatial: usize,
    gamma: &[f32],
    eps: f32,
    dy: &[f32],
    dx: &mut [f32],
    dgamma: &mut [f32],
    dbeta: &mut [f32],
) {
    let sample_len = channels * spatial;
    assert_eq!(x.len(), batch * sample_len, "batch_norm_backward: X shape");
    assert_eq!(gamma.len(), channels, "batch_norm_backward: gamma shape");
    assert_eq!(dy.len(), x.len(), "batch_norm_backward: dY shape");
    assert_eq!(dx.len(), x.len(), "batch_norm_backward: dX shape");
    assert_eq!(dgamma.len(), channels, "batch_norm_backward: dgamma shape");
    assert_eq!(dbeta.len(), channels, "batch_norm_backward: dbeta shape");

    let n = (batch * spatial) as f32;
    for c in 0..channels
    {
        // Passe 1 : moyenne puis variance (biaisée) du canal sur `N` éléments.
        let mut mean = 0.0f32;
        for b in 0..batch
        {
            let start = b * sample_len + c * spatial;
            mean += x[start..start + spatial].iter().sum::<f32>();
        }
        mean /= n;

        let mut var = 0.0f32;
        for b in 0..batch
        {
            let start = b * sample_len + c * spatial;
            var += x[start..start + spatial]
                .iter()
                .map(|&v| (v - mean) * (v - mean))
                .sum::<f32>();
        }
        var /= n;
        let sigma = (var + eps).sqrt();
        let inv = 1.0 / sigma;
        let g = gamma[c];

        // Passe 2 : sommes de réduction (dŷ = dY·γ_c, x̂ = (x−μ_c)·inv).
        let mut sum_dyh = 0.0f32;
        let mut sum_dyh_xh = 0.0f32;
        for b in 0..batch
        {
            let start = b * sample_len + c * spatial;
            let xr = &x[start..start + spatial];
            let dyr = &dy[start..start + spatial];
            for s in 0..spatial
            {
                let xh = (xr[s] - mean) * inv;
                let dyh = dyr[s] * g;
                sum_dyh += dyh;
                sum_dyh_xh += dyh * xh;
            }
        }

        // Passe 3 : dx, et accumulation de dgamma/dbeta.
        let mut dgamma_c = 0.0f32;
        let mut dbeta_c = 0.0f32;
        for b in 0..batch
        {
            let start = b * sample_len + c * spatial;
            let xr = &x[start..start + spatial];
            let dyr = &dy[start..start + spatial];
            let dxr = &mut dx[start..start + spatial];
            for s in 0..spatial
            {
                let xh = (xr[s] - mean) * inv;
                let dyh = dyr[s] * g;
                dxr[s] = inv / n * (n * dyh - sum_dyh - xh * sum_dyh_xh);
                dgamma_c += dyr[s] * xh;
                dbeta_c += dyr[s];
            }
        }
        dgamma[c] += dgamma_c;
        dbeta[c] += dbeta_c;
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
    #[cfg(feature = "transformer-inference")]
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

    // ---- Références forward CNN (indépendantes) pour le gradcheck ----

    #[allow(clippy::too_many_arguments)]
    fn conv2d_fwd(
        x: &[f32],
        in_channels: usize,
        height: usize,
        width: usize,
        w: &[f32],
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        b: &[f32],
    ) -> Vec<f32> {
        let height_out = height - kernel_h + 1;
        let width_out = width - kernel_w + 1;
        let mut y = vec![0.0f32; out_channels * height_out * width_out];
        for co in 0..out_channels
        {
            for oh in 0..height_out
            {
                for ow in 0..width_out
                {
                    let mut acc = b[co];
                    for ci in 0..in_channels
                    {
                        for kh in 0..kernel_h
                        {
                            for kw in 0..kernel_w
                            {
                                acc += x[ci * height * width + (oh + kh) * width + (ow + kw)]
                                    * w[((co * in_channels + ci) * kernel_h + kh) * kernel_w + kw];
                            }
                        }
                    }
                    y[co * height_out * width_out + oh * width_out + ow] = acc;
                }
            }
        }
        y
    }

    #[allow(clippy::too_many_arguments)]
    fn max_pool2d_fwd(
        x: &[f32],
        channels: usize,
        height: usize,
        width: usize,
        window_h: usize,
        window_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Vec<f32> {
        let height_out = (height - window_h) / stride_h + 1;
        let width_out = (width - window_w) / stride_w + 1;
        let mut y = vec![0.0f32; channels * height_out * width_out];
        for c in 0..channels
        {
            for oh in 0..height_out
            {
                for ow in 0..width_out
                {
                    let mut best = f32::NEG_INFINITY;
                    for kh in 0..window_h
                    {
                        for kw in 0..window_w
                        {
                            let v = x[c * height * width
                                + (oh * stride_h + kh) * width
                                + (ow * stride_w + kw)];
                            if v > best
                            {
                                best = v;
                            }
                        }
                    }
                    y[c * height_out * width_out + oh * width_out + ow] = best;
                }
            }
        }
        y
    }

    #[allow(clippy::too_many_arguments)]
    fn avg_pool2d_fwd(
        x: &[f32],
        channels: usize,
        height: usize,
        width: usize,
        window_h: usize,
        window_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Vec<f32> {
        let height_out = (height - window_h) / stride_h + 1;
        let width_out = (width - window_w) / stride_w + 1;
        let inv_window = 1.0 / (window_h * window_w) as f32;
        let mut y = vec![0.0f32; channels * height_out * width_out];
        for c in 0..channels
        {
            for oh in 0..height_out
            {
                for ow in 0..width_out
                {
                    let mut acc = 0.0f32;
                    for kh in 0..window_h
                    {
                        for kw in 0..window_w
                        {
                            acc += x[c * height * width
                                + (oh * stride_h + kh) * width
                                + (ow * stride_w + kw)];
                        }
                    }
                    y[c * height_out * width_out + oh * width_out + ow] = acc * inv_window;
                }
            }
        }
        y
    }

    #[test]
    fn conv2d_backward_gradcheck() {
        let (in_channels, height, width) = (2usize, 5usize, 5usize);
        let (out_channels, kernel_h, kernel_w) = (3usize, 2usize, 3usize);
        let height_out = height - kernel_h + 1;
        let width_out = width - kernel_w + 1;

        let x: Vec<f32> = (0..in_channels * height * width)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();
        let w: Vec<f32> = (0..out_channels * in_channels * kernel_h * kernel_w)
            .map(|i| (i as f32 * 0.07).cos())
            .collect();
        let b: Vec<f32> = (0..out_channels).map(|i| i as f32 * 0.1 - 0.2).collect();
        let seed: Vec<f32> = (0..out_channels * height_out * width_out)
            .map(|i| (i as f32 * 0.3).sin() + 0.2)
            .collect();

        let mut dx = vec![0.0f32; x.len()];
        let mut dw = vec![0.0f32; w.len()];
        let mut db = vec![0.0f32; out_channels];
        conv2d_backward(
            &x,
            in_channels,
            height,
            width,
            &w,
            out_channels,
            kernel_h,
            kernel_w,
            &seed,
            &mut dx,
            &mut dw,
            &mut db,
        );

        let h = 1e-3;
        let g_x = num_grad(&x, &seed, h, |xx| {
            conv2d_fwd(
                xx,
                in_channels,
                height,
                width,
                &w,
                out_channels,
                kernel_h,
                kernel_w,
                &b,
            )
        });
        assert_close(&dx, &g_x, 2e-2, "conv2d dX");
        let g_w = num_grad(&w, &seed, h, |ww| {
            conv2d_fwd(
                &x,
                in_channels,
                height,
                width,
                ww,
                out_channels,
                kernel_h,
                kernel_w,
                &b,
            )
        });
        assert_close(&dw, &g_w, 2e-2, "conv2d dW");
        let g_b = num_grad(&b, &seed, h, |bb| {
            conv2d_fwd(
                &x,
                in_channels,
                height,
                width,
                &w,
                out_channels,
                kernel_h,
                kernel_w,
                bb,
            )
        });
        assert_close(&db, &g_b, 2e-2, "conv2d db");
    }

    #[test]
    fn max_pool2d_backward_gradcheck() {
        let (channels, height, width) = (2usize, 6usize, 6usize);
        let (window_h, window_w, stride_h, stride_w) = (2usize, 2usize, 2usize, 2usize);
        let height_out = (height - window_h) / stride_h + 1;
        let width_out = (width - window_w) / stride_w + 1;

        // Valeurs strictement croissantes (écart 0.7, très supérieur au pas de
        // différences finies 1e-3) : le maximum de chaque fenêtre reste
        // toujours la même position, quelle que soit la petite perturbation —
        // le max n'est pas différentiable près d'une égalité, cf. en-tête de
        // fonction, ce choix évite un tel cas limite dans le gradcheck.
        let x: Vec<f32> = (0..channels * height * width)
            .map(|i| i as f32 * 0.7)
            .collect();
        let seed: Vec<f32> = (0..channels * height_out * width_out)
            .map(|i| (i as f32 * 0.3).sin() + 0.2)
            .collect();

        let mut dx = vec![0.0f32; x.len()];
        max_pool2d_backward(
            &x, channels, height, width, window_h, window_w, stride_h, stride_w, &seed, &mut dx,
        );

        let h = 1e-3;
        let g_x = num_grad(&x, &seed, h, |xx| {
            max_pool2d_fwd(
                xx, channels, height, width, window_h, window_w, stride_h, stride_w,
            )
        });
        assert_close(&dx, &g_x, 2e-2, "max_pool2d dX");
    }

    #[test]
    fn avg_pool2d_backward_gradcheck() {
        let (channels, height, width) = (2usize, 6usize, 6usize);
        let (window_h, window_w, stride_h, stride_w) = (2usize, 3usize, 2usize, 3usize);
        let height_out = (height - window_h) / stride_h + 1;
        let width_out = (width - window_w) / stride_w + 1;

        let seed: Vec<f32> = (0..channels * height_out * width_out)
            .map(|i| (i as f32 * 0.3).sin() + 0.2)
            .collect();

        let mut dx = vec![0.0f32; channels * height * width];
        avg_pool2d_backward(
            channels, height, width, window_h, window_w, stride_h, stride_w, &seed, &mut dx,
        );

        // avg_pool2d_fwd ne dépend de x que par sa forme ici (référence
        // indépendante) : x arbitraire, seul le gradient (linéaire en x)
        // importe pour le gradcheck.
        let x0: Vec<f32> = (0..channels * height * width)
            .map(|i| (i as f32 * 0.13).cos())
            .collect();
        let h = 1e-3;
        let g_x = num_grad(&x0, &seed, h, |xx| {
            avg_pool2d_fwd(
                xx, channels, height, width, window_h, window_w, stride_h, stride_w,
            )
        });
        assert_close(&dx, &g_x, 2e-2, "avg_pool2d dX");
    }

    // ---- Référence forward BatchNorm entraînement (indépendante) ----

    #[allow(clippy::too_many_arguments)]
    fn batch_norm_train_fwd(
        x: &[f32],
        batch: usize,
        channels: usize,
        spatial: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        let sample_len = channels * spatial;
        let n = (batch * spatial) as f32;
        let mut y = vec![0.0f32; batch * sample_len];
        for c in 0..channels
        {
            let mut mean = 0.0f32;
            for b in 0..batch
            {
                let start = b * sample_len + c * spatial;
                mean += x[start..start + spatial].iter().sum::<f32>();
            }
            mean /= n;

            let mut var = 0.0f32;
            for b in 0..batch
            {
                let start = b * sample_len + c * spatial;
                var += x[start..start + spatial]
                    .iter()
                    .map(|&v| (v - mean) * (v - mean))
                    .sum::<f32>();
            }
            var /= n;
            let inv = 1.0 / (var + eps).sqrt();

            for b in 0..batch
            {
                let start = b * sample_len + c * spatial;
                for s in 0..spatial
                {
                    y[start + s] = (x[start + s] - mean) * inv * gamma[c] + beta[c];
                }
            }
        }
        y
    }

    #[test]
    fn batch_norm_backward_gradcheck() {
        let (batch, channels, spatial) = (3usize, 2usize, 4usize);
        let sample_len = channels * spatial;
        let x: Vec<f32> = (0..batch * sample_len)
            .map(|i| (i as f32 * 0.13).sin() * 2.0 + 0.5)
            .collect();
        let gamma: Vec<f32> = (0..channels).map(|i| 0.6 + i as f32 * 0.2).collect();
        let beta: Vec<f32> = (0..channels).map(|i| i as f32 * 0.1 - 0.1).collect();
        let eps = 1e-5f32;
        let seed: Vec<f32> = (0..batch * sample_len)
            .map(|i| (i as f32 * 0.19).cos() + 0.2)
            .collect();

        let mut dx = vec![0.0f32; batch * sample_len];
        let mut dgamma = vec![0.0f32; channels];
        let mut dbeta = vec![0.0f32; channels];
        batch_norm_backward(
            &x,
            batch,
            channels,
            spatial,
            &gamma,
            eps,
            &seed,
            &mut dx,
            &mut dgamma,
            &mut dbeta,
        );

        let h = 1e-3;
        let gx = num_grad(&x, &seed, h, |xx| {
            batch_norm_train_fwd(xx, batch, channels, spatial, &gamma, &beta, eps)
        });
        assert_close(&dx, &gx, 3e-2, "batch_norm dX");
        let gg = num_grad(&gamma, &seed, h, |gg| {
            batch_norm_train_fwd(&x, batch, channels, spatial, gg, &beta, eps)
        });
        assert_close(&dgamma, &gg, 3e-2, "batch_norm dgamma");
        let gb = num_grad(&beta, &seed, h, |bb| {
            batch_norm_train_fwd(&x, batch, channels, spatial, &gamma, bb, eps)
        });
        assert_close(&dbeta, &gb, 3e-2, "batch_norm dbeta");
    }
}
