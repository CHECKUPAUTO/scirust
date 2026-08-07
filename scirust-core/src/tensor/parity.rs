// scirust-core/src/tensor/parity.rs
//
// Tensor Parity Profile 1.0 — noyaux de parité sémantique avec PyTorch 2.13.0.
//
// Ce module est le seul endroit où la parité est CLAIMED : chaque fonction est
// vérifiée par tests/parity/ (harness Rust-only, fixtures générées hors-ligne
// contre le baseline figé cf30153c4c131c8164ee7798e5022d810682e2cb).
//
// Disciplines (SCOPE.md) :
//   - erreurs structurées `SciRustError` (jamais de panic sur le chemin public),
//   - arithmétique de shape/index *checked* (checked_add/mul),
//   - ordre d'accumulation déterministe (séquentiel),
//   - f32, layout row-major contigu (strided views via TensorND existant).

use crate::error::{Result, SciRustError, check_axis};
use crate::tensor::tensor_nd::TensorND;

/// Applique un mapping élémentaire `f` et retourne un nouveau tenseur contigu.
fn map(t: &TensorND, f: impl Fn(f32) -> f32) -> Result<TensorND> {
    let out: Vec<f32> = t.data.iter().map(|&x| f(x)).collect();
    Ok(TensorND::new(out, t.shape.clone()))
}

/// Applique un mapping élémentaire binaire sur deux tenseurs de même shape.
fn map2(a: &TensorND, b: &TensorND, f: impl Fn(f32, f32) -> f32) -> Result<TensorND> {
    if a.shape() != b.shape()
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::map2",
            expected: (a.numel(), 1),
            got: (b.numel(), 1),
        });
    }
    let out: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| f(x, y))
        .collect();
    Ok(TensorND::new(out, a.shape.clone()))
}

fn checked_prod(shape: &[usize]) -> Result<usize> {
    let mut n = 1usize;
    for &d in shape
    {
        n = n
            .checked_mul(d)
            .ok_or_else(|| SciRustError::InvalidConfig(format!("shape overflow: {shape:?}")))?;
    }
    Ok(n)
}

// ------------------------------------------------------------------ //
//  Elementwise — unaires                                          //
// ------------------------------------------------------------------ //

pub fn neg(t: &TensorND) -> Result<TensorND> {
    map(t, |x| -x)
}

pub fn reciprocal(t: &TensorND) -> Result<TensorND> {
    map(t, |x| 1.0 / x)
}

pub fn exp(t: &TensorND) -> Result<TensorND> {
    map(t, f32::exp)
}

pub fn log(t: &TensorND) -> Result<TensorND> {
    map(t, f32::ln)
}

pub fn log10(t: &TensorND) -> Result<TensorND> {
    map(t, f32::log10)
}

pub fn sqrt(t: &TensorND) -> Result<TensorND> {
    map(t, f32::sqrt)
}

pub fn rsqrt(t: &TensorND) -> Result<TensorND> {
    map(t, |x| 1.0 / x.sqrt())
}

pub fn sin(t: &TensorND) -> Result<TensorND> {
    map(t, f32::sin)
}

pub fn cos(t: &TensorND) -> Result<TensorND> {
    map(t, f32::cos)
}

pub fn tan(t: &TensorND) -> Result<TensorND> {
    map(t, f32::tan)
}

pub fn asin(t: &TensorND) -> Result<TensorND> {
    map(t, f32::asin)
}

pub fn acos(t: &TensorND) -> Result<TensorND> {
    map(t, f32::acos)
}

pub fn atan(t: &TensorND) -> Result<TensorND> {
    map(t, f32::atan)
}

pub fn sinh(t: &TensorND) -> Result<TensorND> {
    map(t, f32::sinh)
}

pub fn cosh(t: &TensorND) -> Result<TensorND> {
    map(t, f32::cosh)
}

pub fn tanh(t: &TensorND) -> Result<TensorND> {
    map(t, f32::tanh)
}

pub fn sigmoid(t: &TensorND) -> Result<TensorND> {
    map(t, |x| 1.0 / (1.0 + (-x).exp()))
}

pub fn relu(t: &TensorND) -> Result<TensorND> {
    map(t, |x| if x > 0.0 { x } else { 0.0 })
}

pub fn silu(t: &TensorND) -> Result<TensorND> {
    map(t, |x| x * (1.0 / (1.0 + (-x).exp())))
}

/// gelu au sens `approximate="tanh"` de PyTorch 2.13 (défaut du baseline
/// pour cette comparaison ; cf. fixture elementwise/gelu.json).
pub fn gelu(t: &TensorND) -> Result<TensorND> {
    #[allow(clippy::excessive_precision)]
    const C: f32 = 0.797_884_560_802_865_4; // sqrt(2/pi)
    map(t, |x| {
        0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
    })
}

pub fn pow_scalar(t: &TensorND, e: f32) -> Result<TensorND> {
    map(t, |x| x.powf(e))
}

// ------------------------------------------------------------------ //
//  Elementwise — binaires                                          //
// ------------------------------------------------------------------ //

pub fn add(a: &TensorND, b: &TensorND) -> Result<TensorND> {
    map2(a, b, |x, y| x + y)
}

pub fn sub(a: &TensorND, b: &TensorND) -> Result<TensorND> {
    map2(a, b, |x, y| x - y)
}

pub fn mul(a: &TensorND, b: &TensorND) -> Result<TensorND> {
    map2(a, b, |x, y| x * y)
}

pub fn div(a: &TensorND, b: &TensorND) -> Result<TensorND> {
    map2(a, b, |x, y| x / y)
}

pub fn atan2(a: &TensorND, b: &TensorND) -> Result<TensorND> {
    map2(a, b, f32::atan2)
}

// ------------------------------------------------------------------ //
//  Normalisation (axe = dernière dimension)                        //
// ------------------------------------------------------------------ //

/// softmax sur la dernière dimension (stableshift max), contigu.
pub fn softmax_last(t: &TensorND) -> Result<TensorND> {
    if t.ndim() == 0
    {
        return Err(SciRustError::RankMismatch {
            op: "parity::softmax_last",
            expected: 1,
            got: 0,
        });
    }
    let last = t.ndim() - 1;
    let axis_len = t.shape[last];
    let outer = t.numel() / axis_len;
    let mut out = vec![0.0f32; t.numel()];
    for i in 0..outer
    {
        let base = i * axis_len;
        let mut max = f32::NEG_INFINITY;
        for j in 0..axis_len
        {
            let v = t.data[base + j];
            if v > max
            {
                max = v;
            }
        }
        let mut sum = 0.0f32;
        for j in 0..axis_len
        {
            let e = (t.data[base + j] - max).exp();
            out[base + j] = e;
            sum += e;
        }
        for j in 0..axis_len
        {
            out[base + j] /= sum;
        }
    }
    Ok(TensorND::new(out, t.shape.clone()))
}

pub fn log_softmax_last(t: &TensorND) -> Result<TensorND> {
    if t.ndim() == 0
    {
        return Err(SciRustError::RankMismatch {
            op: "parity::log_softmax_last",
            expected: 1,
            got: 0,
        });
    }
    let last = t.ndim() - 1;
    let axis_len = t.shape[last];
    let outer = t.numel() / axis_len;
    let mut out = vec![0.0f32; t.numel()];
    for i in 0..outer
    {
        let base = i * axis_len;
        let mut max = f32::NEG_INFINITY;
        for j in 0..axis_len
        {
            let v = t.data[base + j];
            if v > max
            {
                max = v;
            }
        }
        let mut sum = 0.0f32;
        for j in 0..axis_len
        {
            sum += (t.data[base + j] - max).exp();
        }
        let logsum = max + sum.ln();
        for j in 0..axis_len
        {
            out[base + j] = t.data[base + j] - logsum;
        }
    }
    Ok(TensorND::new(out, t.shape.clone()))
}

// ------------------------------------------------------------------ //
//  Normalisation affine (layer_norm / rms_norm)                     //
// ------------------------------------------------------------------ //

/// Vérifie que `normalized_shape` correspond aux dims de queue de `x` et
/// retourne (rows, group) où `group` est le nombre d'éléments par groupe.
fn norm_groups(
    x: &TensorND,
    op: &'static str,
    normalized_shape: &[usize],
) -> Result<(usize, usize)> {
    if normalized_shape.is_empty()
    {
        return Err(SciRustError::InvalidConfig(format!(
            "{op}: normalized_shape ne peut pas être vide"
        )));
    }
    if normalized_shape.len() > x.ndim()
    {
        return Err(SciRustError::RankMismatch {
            op,
            expected: x.ndim(),
            got: normalized_shape.len(),
        });
    }
    let rank = x.ndim();
    for (d, &want) in normalized_shape.iter().enumerate()
    {
        let got = x.shape[rank - normalized_shape.len() + d];
        if got != want
        {
            return Err(SciRustError::ShapeMismatch {
                op,
                expected: (1, want),
                got: (1, got),
            });
        }
    }
    let group = checked_prod(normalized_shape)?;
    if group == 0
    {
        return Err(SciRustError::InvalidConfig(format!("{op}: group nul")));
    }
    Ok((x.numel() / group, group))
}

/// layer_norm affine : y = ((x - mean)/sqrt(var+eps)) * w + b,
/// normalisé sur les dims de queue `normalized_shape` (var non biaisée).
pub fn layer_norm(
    x: &TensorND,
    w: &TensorND,
    b: &TensorND,
    normalized_shape: &[usize],
    eps: f32,
) -> Result<TensorND> {
    let (rows, group) = norm_groups(x, "parity::layer_norm", normalized_shape)?;
    if w.numel() != group || b.numel() != group
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::layer_norm",
            expected: (1, group),
            got: (1, w.numel()),
        });
    }
    let mut out = vec![0.0f32; x.numel()];
    for i in 0..rows
    {
        let base = i * group;
        let mean: f32 = (0..group).map(|j| x.data[base + j]).sum::<f32>() / group as f32;
        let var: f32 = (0..group)
            .map(|j| {
                let d = x.data[base + j] - mean;
                d * d
            })
            .sum::<f32>()
            / group as f32;
        let std = (var + eps).sqrt();
        for j in 0..group
        {
            let xhat = (x.data[base + j] - mean) / std;
            out[base + j] = xhat * w.data[j] + b.data[j];
        }
    }
    Ok(TensorND::new(out, x.shape.clone()))
}

/// Gradient layer_norm (affine) : gx, gw, gb.
/// Formules standard par groupe (ligne de `group` éléments) :
///   std = sqrt(var + eps), xhat = (x - mean)/std, dxhat = gout * w
///   gx  = (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)) / std
///   gw  = Σ_rows gout * xhat          (par dim normalisée)
///   gb  = Σ_rows gout                 (par dim normalisée)
pub fn d_layer_norm(
    gout: &TensorND,
    x: &TensorND,
    w: &TensorND,
    normalized_shape: &[usize],
    eps: f32,
) -> Result<(TensorND, TensorND, TensorND)> {
    let (rows, group) = norm_groups(x, "parity::d_layer_norm", normalized_shape)?;
    if gout.numel() != x.numel() || w.numel() != group
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::d_layer_norm",
            expected: (1, x.numel()),
            got: (1, gout.numel()),
        });
    }
    let mut gx = vec![0.0f32; x.numel()];
    let mut gw = vec![0.0f32; group];
    let mut gb = vec![0.0f32; group];
    for i in 0..rows
    {
        let base = i * group;
        let mean: f32 = (0..group).map(|j| x.data[base + j]).sum::<f32>() / group as f32;
        let var: f32 = (0..group)
            .map(|j| {
                let d = x.data[base + j] - mean;
                d * d
            })
            .sum::<f32>()
            / group as f32;
        let std = (var + eps).sqrt();
        let mut xhat = vec![0.0f32; group];
        let mut dxhat = vec![0.0f32; group];
        for j in 0..group
        {
            xhat[j] = (x.data[base + j] - mean) / std;
            dxhat[j] = gout.data[base + j] * w.data[j];
        }
        let mean_dxhat: f32 = dxhat.iter().sum::<f32>() / group as f32;
        let mean_dxhat_xhat: f32 =
            (0..group).map(|j| dxhat[j] * xhat[j]).sum::<f32>() / group as f32;
        for j in 0..group
        {
            gx[base + j] = (dxhat[j] - mean_dxhat - xhat[j] * mean_dxhat_xhat) / std;
            gw[j] += gout.data[base + j] * xhat[j];
            gb[j] += gout.data[base + j];
        }
    }
    Ok((
        TensorND::new(gx, x.shape.clone()),
        TensorND::new(gw, vec![group]),
        TensorND::new(gb, vec![group]),
    ))
}

/// rms_norm : y = (x / sqrt(mean(x²) + eps)) * w, normalisé sur les dims de
/// queue `normalized_shape`.
pub fn rms_norm(
    x: &TensorND,
    w: &TensorND,
    normalized_shape: &[usize],
    eps: f32,
) -> Result<TensorND> {
    let (rows, group) = norm_groups(x, "parity::rms_norm", normalized_shape)?;
    if w.numel() != group
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::rms_norm",
            expected: (1, group),
            got: (1, w.numel()),
        });
    }
    let mut out = vec![0.0f32; x.numel()];
    for i in 0..rows
    {
        let base = i * group;
        let ms: f32 = (0..group)
            .map(|j| x.data[base + j] * x.data[base + j])
            .sum::<f32>()
            / group as f32;
        let r = (ms + eps).sqrt();
        for j in 0..group
        {
            out[base + j] = x.data[base + j] / r * w.data[j];
        }
    }
    Ok(TensorND::new(out, x.shape.clone()))
}

/// Gradient rms_norm : gx, gw.
/// Par groupe : r = sqrt(ms + eps), xhat = x/r, dxhat = gout * w
///   gx = dxhat/r - xhat * Σ(dxhat * xhat) / r
///   gw = Σ_rows gout * xhat
pub fn d_rms_norm(
    gout: &TensorND,
    x: &TensorND,
    w: &TensorND,
    normalized_shape: &[usize],
    eps: f32,
) -> Result<(TensorND, TensorND)> {
    let (rows, group) = norm_groups(x, "parity::d_rms_norm", normalized_shape)?;
    if gout.numel() != x.numel() || w.numel() != group
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::d_rms_norm",
            expected: (1, x.numel()),
            got: (1, gout.numel()),
        });
    }
    let mut gx = vec![0.0f32; x.numel()];
    let mut gw = vec![0.0f32; group];
    for i in 0..rows
    {
        let base = i * group;
        let ms: f32 = (0..group)
            .map(|j| x.data[base + j] * x.data[base + j])
            .sum::<f32>()
            / group as f32;
        let r = (ms + eps).sqrt();
        let dot: f32 = (0..group)
            .map(|j| {
                let dxhat = gout.data[base + j] * w.data[j];
                dxhat * x.data[base + j]
            })
            .sum::<f32>();
        let n = group as f32;
        for j in 0..group
        {
            let xhat = x.data[base + j] / r;
            let dxhat = gout.data[base + j] * w.data[j];
            // gx = dxhat/r - x·Σ(dxhat·x)/(n·r³)
            gx[base + j] = dxhat / r - x.data[base + j] * dot / (n * r * r * r);
            gw[j] += gout.data[base + j] * xhat;
        }
    }
    Ok((
        TensorND::new(gx, x.shape.clone()),
        TensorND::new(gw, vec![group]),
    ))
}

// ------------------------------------------------------------------ //
//  Réductions (déterministes, séquentielles)                       //
// ------------------------------------------------------------------ //

fn reduce_axis_shape(t: &TensorND, axis: usize) -> Result<Vec<usize>> {
    check_axis("parity::reduce", axis, t.ndim())?;
    let mut shape = t.shape.clone();
    shape.remove(axis);
    Ok(shape)
}

/// Réduit un axe en appliquant `f` sur chaque ligne le long de l'axe
/// (ordre déterministe : séquentiel de l'indice 0 au dernier).
#[allow(clippy::needless_range_loop)]
fn reduce_axis(t: &TensorND, axis: usize, f: impl Fn(&[f32]) -> f32) -> Result<TensorND> {
    check_axis("parity::reduce", axis, t.ndim())?;
    let out_shape = reduce_axis_shape(t, axis)?;
    let out_numel = checked_prod(&out_shape)?;
    let axis_len = t.shape[axis];

    // Strides row-major du tenseur d'entrée (avec l'axe).
    let mut strides = vec![0usize; t.ndim()];
    let mut acc = 1usize;
    for d in (0..t.ndim()).rev()
    {
        strides[d] = acc;
        acc = acc
            .checked_mul(t.shape[d])
            .ok_or_else(|| SciRustError::InvalidConfig("shape overflow".into()))?;
    }

    let mut out = vec![0.0f32; out_numel];
    let mut line = vec![0.0f32; axis_len];
    let mut t_idx = vec![0usize; t.ndim()];
    for i in 0..out_numel
    {
        // Décodage row-major de l'index de sortie en coordonnées d'entrée
        // (l'axe reçoit 0) : on itère les dims de out_shape de la dernière
        // à la première, en insérant l'axe.
        let mut rem = i;
        for k in (0..out_shape.len()).rev()
        {
            let pos = if k < axis { k } else { k + 1 };
            let dim = out_shape[k];
            t_idx[pos] = rem % dim;
            rem /= dim;
        }
        debug_assert_eq!(rem, 0);

        for j in 0..axis_len
        {
            t_idx[axis] = j;
            let mut off = 0usize;
            for (k, &idx) in t_idx.iter().enumerate()
            {
                off += idx * strides[k];
            }
            line[j] = t.data[off];
        }
        out[i] = f(&line);
    }
    Ok(TensorND::new(out, out_shape))
}

pub fn sum_axis(t: &TensorND, axis: usize) -> Result<TensorND> {
    reduce_axis(t, axis, |line| line.iter().fold(0.0f32, |a, &b| a + b))
}

pub fn mean_axis(t: &TensorND, axis: usize) -> Result<TensorND> {
    let n = t.shape[axis] as f32;
    reduce_axis(t, axis, |line| line.iter().fold(0.0f32, |a, &b| a + b) / n)
}

/// Variance (population, `unbiased=false`) — fixture générée avec
/// `torch.var(x, dim=axis, unbiased=False)`.
pub fn var_axis(t: &TensorND, axis: usize) -> Result<TensorND> {
    let n = t.shape[axis] as f32;
    reduce_axis(t, axis, |line| {
        let mean = line.iter().fold(0.0f32, |a, &b| a + b) / n;
        line.iter()
            .fold(0.0f32, |a, &b| a + (b - mean) * (b - mean))
            / n
    })
}

/// max le long d'un axe (keepdim=false) : valeurs + indices du premier
/// maximum (sémantique torch : premier index en cas d'égalité).
pub fn max_axis(t: &TensorND, axis: usize) -> Result<(TensorND, Vec<usize>)> {
    check_axis("parity::max_axis", axis, t.ndim())?;
    let out_shape = reduce_axis_shape(t, axis)?;
    let out_numel = checked_prod(&out_shape)?;
    let axis_len = t.shape[axis];
    let mut strides = vec![0usize; t.ndim()];
    let mut acc = 1usize;
    for d in (0..t.ndim()).rev()
    {
        strides[d] = acc;
        acc = acc
            .checked_mul(t.shape[d])
            .ok_or_else(|| SciRustError::InvalidConfig("shape overflow".into()))?;
    }
    let mut out = vec![0.0f32; out_numel];
    let mut indices = vec![0usize; out_numel];
    let mut t_idx = vec![0usize; t.ndim()];
    for i in 0..out_numel
    {
        let mut rem = i;
        for k in (0..out_shape.len()).rev()
        {
            let pos = if k < axis { k } else { k + 1 };
            let dim = out_shape[k];
            t_idx[pos] = rem % dim;
            rem /= dim;
        }
        debug_assert_eq!(rem, 0);
        let mut best = f32::NEG_INFINITY;
        let mut best_j = 0usize;
        for j in 0..axis_len
        {
            t_idx[axis] = j;
            let mut off = 0usize;
            for (k, &idx) in t_idx.iter().enumerate()
            {
                off += idx * strides[k];
            }
            let v = t.data[off];
            if v > best
            {
                best = v;
                best_j = j;
            }
        }
        out[i] = best;
        indices[i] = best_j;
    }
    Ok((TensorND::new(out, out_shape), indices))
}

/// Gradient de max(dim) : le gout de chaque ligne est routé vers l'élément
/// max (indices du premier maximum), 0 ailleurs.
pub fn d_max_axis(
    gout: &TensorND,
    t: &TensorND,
    axis: usize,
    indices: &[usize],
) -> Result<TensorND> {
    check_axis("parity::d_max_axis", axis, t.ndim())?;
    let out_shape = reduce_axis_shape(t, axis)?;
    if gout.numel() != checked_prod(&out_shape)? || indices.len() != gout.numel()
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::d_max_axis",
            expected: (1, checked_prod(&out_shape)?),
            got: (1, gout.numel()),
        });
    }
    let axis_len = t.shape[axis];
    let mut strides = vec![0usize; t.ndim()];
    let mut acc = 1usize;
    for d in (0..t.ndim()).rev()
    {
        strides[d] = acc;
        acc = acc
            .checked_mul(t.shape[d])
            .ok_or_else(|| SciRustError::InvalidConfig("shape overflow".into()))?;
    }
    let mut gx = vec![0.0f32; t.numel()];
    let mut t_idx = vec![0usize; t.ndim()];
    for (i, &j) in indices.iter().enumerate()
    {
        let mut rem = i;
        for k in (0..out_shape.len()).rev()
        {
            let pos = if k < axis { k } else { k + 1 };
            let dim = out_shape[k];
            t_idx[pos] = rem % dim;
            rem /= dim;
        }
        debug_assert_eq!(rem, 0);
        debug_assert!(j < axis_len, "d_max_axis: index {j} >= axis_len {axis_len}");
        t_idx[axis] = j;
        let mut off = 0usize;
        for (k, &idx) in t_idx.iter().enumerate()
        {
            off += idx * strides[k];
        }
        gx[off] = gout.data[i];
    }
    Ok(TensorND::new(gx, t.shape.clone()))
}

/// Norme de Frobenius (p=2) le long d'un axe : y = sqrt(Σ x²).
pub fn norm_axis_p2(t: &TensorND, axis: usize) -> Result<TensorND> {
    check_axis("parity::norm_axis_p2", axis, t.ndim())?;
    let out_shape = reduce_axis_shape(t, axis)?;
    let out_numel = checked_prod(&out_shape)?;
    let axis_len = t.shape[axis];
    let mut strides = vec![0usize; t.ndim()];
    let mut acc = 1usize;
    for d in (0..t.ndim()).rev()
    {
        strides[d] = acc;
        acc = acc
            .checked_mul(t.shape[d])
            .ok_or_else(|| SciRustError::InvalidConfig("shape overflow".into()))?;
    }
    let mut out = vec![0.0f32; out_numel];
    let mut t_idx = vec![0usize; t.ndim()];
    for (i, slot) in out.iter_mut().enumerate()
    {
        let mut rem = i;
        for k in (0..out_shape.len()).rev()
        {
            let pos = if k < axis { k } else { k + 1 };
            let dim = out_shape[k];
            t_idx[pos] = rem % dim;
            rem /= dim;
        }
        debug_assert_eq!(rem, 0);
        let mut sum = 0.0f32;
        for j in 0..axis_len
        {
            t_idx[axis] = j;
            let mut off = 0usize;
            for (k, &idx) in t_idx.iter().enumerate()
            {
                off += idx * strides[k];
            }
            sum += t.data[off] * t.data[off];
        }
        *slot = sum.sqrt();
    }
    Ok(TensorND::new(out, out_shape))
}

/// Gradient norme p=2 le long d'un axe : gx = gout * x / ||x||₂ (0 si nul).
pub fn d_norm_axis_p2(gout: &TensorND, t: &TensorND, axis: usize) -> Result<TensorND> {
    check_axis("parity::d_norm_axis_p2", axis, t.ndim())?;
    let out_shape = reduce_axis_shape(t, axis)?;
    if gout.numel() != checked_prod(&out_shape)?
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::d_norm_axis_p2",
            expected: (1, checked_prod(&out_shape)?),
            got: (1, gout.numel()),
        });
    }
    let axis_len = t.shape[axis];
    let mut strides = vec![0usize; t.ndim()];
    let mut acc = 1usize;
    for d in (0..t.ndim()).rev()
    {
        strides[d] = acc;
        acc = acc
            .checked_mul(t.shape[d])
            .ok_or_else(|| SciRustError::InvalidConfig("shape overflow".into()))?;
    }
    let mut gx = vec![0.0f32; t.numel()];
    let mut t_idx = vec![0usize; t.ndim()];
    for i in 0..gout.numel()
    {
        let mut rem = i;
        for k in (0..out_shape.len()).rev()
        {
            let pos = if k < axis { k } else { k + 1 };
            let dim = out_shape[k];
            t_idx[pos] = rem % dim;
            rem /= dim;
        }
        debug_assert_eq!(rem, 0);
        let mut sum = 0.0f32;
        for j in 0..axis_len
        {
            t_idx[axis] = j;
            let mut off = 0usize;
            for (k, &idx) in t_idx.iter().enumerate()
            {
                off += idx * strides[k];
            }
            sum += t.data[off] * t.data[off];
        }
        let norm = sum.sqrt();
        if norm == 0.0
        {
            continue;
        }
        let g = gout.data[i] / norm;
        for j in 0..axis_len
        {
            t_idx[axis] = j;
            let mut off = 0usize;
            for (k, &idx) in t_idx.iter().enumerate()
            {
                off += idx * strides[k];
            }
            gx[off] = g * t.data[off];
        }
    }
    Ok(TensorND::new(gx, t.shape.clone()))
}

// ------------------------------------------------------------------ //
//  Dérivées élémentaires (utilisées par le harness pour gradcheck) //
// ------------------------------------------------------------------ //

/// Dérivée de l'op élémentaire unaire par rapport à son entrée, en `x`.
pub fn d_unary(op: &str, x: f32) -> Result<f32> {
    let d = match op
    {
        "neg" => -1.0,
        "reciprocal" => -1.0 / (x * x),
        "exp" => x.exp(),
        "log" => 1.0 / x,
        "log10" => 1.0 / (x * 10.0f32.ln()),
        "sqrt" => 0.5 / x.sqrt(),
        "rsqrt" => -0.5 / (x * x.sqrt()),
        "sin" => x.cos(),
        "cos" => -x.sin(),
        "tan" => 1.0 + x.tan() * x.tan(),
        "asin" => 1.0 / (1.0 - x * x).sqrt(),
        "acos" => -1.0 / (1.0 - x * x).sqrt(),
        "atan" => 1.0 / (1.0 + x * x),
        "sinh" => x.cosh(),
        "cosh" => x.sinh(),
        "tanh" => 1.0 - x.tanh() * x.tanh(),
        "sigmoid" =>
        {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 - s)
        },
        "relu" =>
        {
            if x > 0.0
            {
                1.0
            }
            else
            {
                0.0
            }
        },
        "silu" =>
        {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 + x * (1.0 - s))
        },
        "gelu" =>
        {
            #[allow(clippy::excessive_precision)]
            const C: f32 = 0.797_884_560_802_865_4; // sqrt(2/pi)
            let u = C * (x + 0.044715 * x * x * x);
            let t = u.tanh();
            0.5 * (1.0 + t + x * (1.0 - t * t) * C * (1.0 + 3.0 * 0.044715 * x * x))
        },
        "pow" => 2.0 * x, // exponent fixé à 2.0 (fixture)
        _ =>
        {
            return Err(SciRustError::InvalidConfig(format!(
                "d_unary: unknown op '{op}'"
            )));
        },
    };
    Ok(d)
}

pub fn d_binary(op: &str, x: f32, y: f32) -> Result<(f32, f32)> {
    let d = match op
    {
        "add" => (1.0, 1.0),
        "sub" => (1.0, -1.0),
        "mul" => (y, x),
        "div" => (1.0 / y, -x / (y * y)),
        "atan2" =>
        {
            let r2 = x * x + y * y;
            (y / r2, -x / r2)
        },
        _ =>
        {
            return Err(SciRustError::InvalidConfig(format!(
                "d_binary: unknown op '{op}'"
            )));
        },
    };
    Ok(d)
}

// ------------------------------------------------------------------ //
//  Dérivées de réduction (utilisées par le harness pour gradcheck) //
// ------------------------------------------------------------------ //

pub fn d_sum(gout: f32) -> f32 {
    gout
}

pub fn d_mean(gout: f32, n: f32) -> f32 {
    gout / n
}

pub fn d_var(x: f32, mean: f32, gout: f32, n: f32) -> f32 {
    2.0 * (x - mean) * gout / n
}

// ------------------------------------------------------------------ //
//  Shape (reshape / transpose / permute / broadcast / slice / flatten)
// ------------------------------------------------------------------ //

fn validate_perm(op: &'static str, ndim: usize, dims: &[usize]) -> Result<()> {
    if dims.len() != ndim
    {
        return Err(SciRustError::RankMismatch {
            op,
            expected: ndim,
            got: dims.len(),
        });
    }
    let mut seen = vec![false; ndim];
    for &d in dims
    {
        if d >= ndim
        {
            return Err(SciRustError::AxisOutOfBounds {
                op,
                axis: d,
                rank: ndim,
            });
        }
        if seen[d]
        {
            return Err(SciRustError::InvalidConfig(format!(
                "{op}: duplicate axis {d} in permutation {dims:?}"
            )));
        }
        seen[d] = true;
    }
    Ok(())
}

pub fn reshape(t: &TensorND, new_shape: &[usize]) -> Result<TensorND> {
    t.reshape(new_shape)
}

pub fn permute(t: &TensorND, dims: &[usize]) -> Result<TensorND> {
    validate_perm("parity::permute", t.ndim(), dims)?;
    t.transpose(dims)
}

/// `torch.transpose(x, d0, d1)` : échange deux axes (validation identique à
/// PyTorch pour le cas 2-D ; au-delà, le swap est vérifié).
pub fn transpose2(t: &TensorND, d0: usize, d1: usize) -> Result<TensorND> {
    let ndim = t.ndim();
    if d0 >= ndim || d1 >= ndim
    {
        return Err(SciRustError::AxisOutOfBounds {
            op: "parity::transpose2",
            axis: if d0 >= ndim { d0 } else { d1 },
            rank: ndim,
        });
    }
    let mut dims: Vec<usize> = (0..ndim).collect();
    dims.swap(d0, d1);
    t.transpose(&dims)
}

pub fn broadcast_to(t: &TensorND, target: &[usize]) -> Result<TensorND> {
    t.broadcast_to(target)
}

pub fn slice_axis(t: &TensorND, axis: usize, start: usize, end: usize) -> Result<TensorND> {
    check_axis("parity::slice_axis", axis, t.ndim())?;
    if start > end || end > t.shape[axis]
    {
        return Err(SciRustError::InvalidConfig(format!(
            "parity::slice_axis: invalid range [{start}, {end}) for axis {axis} (dim {})",
            t.shape[axis]
        )));
    }
    t.slice_axis(axis, start, end)
}

pub fn flatten(t: &TensorND) -> Result<TensorND> {
    Ok(t.flatten())
}

/// Gradient de reshape : reboucher dans la shape d'entrée.
pub fn g_reshape(gout: &TensorND, in_shape: &[usize]) -> Result<TensorND> {
    gout.reshape(in_shape)
}

/// Gradient de transpose/permute : permutation inverse.
pub fn g_permute(gout: &TensorND, dims: &[usize]) -> Result<TensorND> {
    validate_perm("parity::g_permute", gout.ndim(), dims)?;
    let mut inv = vec![0usize; dims.len()];
    for (i, &d) in dims.iter().enumerate()
    {
        inv[d] = i;
    }
    gout.transpose(&inv)
}

/// Gradient de broadcast_to : somme-réduction du gradient vers la shape
/// d'entrée (règle torch : grad broadcast = sum sur les dims broadcastées).
pub fn g_broadcast(gout: &TensorND, in_shape: &[usize]) -> Result<TensorND> {
    let gout_rank = gout.shape().len();
    let in_rank = in_shape.len();
    if gout_rank < in_rank
    {
        return Err(SciRustError::RankMismatch {
            op: "parity::g_broadcast",
            expected: in_rank,
            got: gout_rank,
        });
    }
    let in_numel = checked_prod(in_shape)?;
    let out_numel = checked_prod(gout.shape())?;
    let offset = gout_rank - in_rank;
    for (r, &in_dim) in in_shape.iter().enumerate()
    {
        let out_dim = gout.shape()[offset + r];
        if in_dim != out_dim && in_dim != 1
        {
            return Err(SciRustError::BroadcastIncompatible {
                op: "parity::g_broadcast",
                from: in_shape.to_vec(),
                to: gout.shape().to_vec(),
            });
        }
    }
    let mut acc = vec![0.0f32; in_numel];
    let mut out_coords = vec![0usize; gout_rank];
    for oi in 0..out_numel
    {
        let mut rem = oi;
        for d in (0..gout_rank).rev()
        {
            out_coords[d] = rem % gout.shape()[d];
            rem /= gout.shape()[d];
        }
        debug_assert_eq!(rem, 0);
        let mut in_linear = 0usize;
        let mut stride = 1usize;
        for r in (0..in_rank).rev()
        {
            let in_dim = in_shape[r];
            let c = if in_dim == 1
            {
                0
            }
            else
            {
                out_coords[offset + r]
            };
            in_linear += c * stride;
            stride *= in_dim;
        }
        acc[in_linear] += gout.data[oi];
    }
    Ok(TensorND::new(acc, in_shape.to_vec()))
}

/// Gradient de slice : scatter du gradient dans un zéro-padding à la shape
/// d'entrée (grad torch : zero outside the slice).
pub fn g_slice(
    gout: &TensorND,
    in_shape: &[usize],
    axis: usize,
    start: usize,
    end: usize,
) -> Result<TensorND> {
    check_axis("parity::g_slice", axis, in_shape.len())?;
    if start > end || end > in_shape[axis]
    {
        return Err(SciRustError::InvalidConfig(format!(
            "parity::g_slice: invalid range [{start}, {end}) for axis {axis} (dim {})",
            in_shape[axis]
        )));
    }
    let in_numel = checked_prod(in_shape)?;
    let _ = in_numel;
    let mut strides = vec![0usize; in_shape.len()];
    let mut acc = 1usize;
    for d in (0..in_shape.len()).rev()
    {
        strides[d] = acc;
        acc = acc
            .checked_mul(in_shape[d])
            .ok_or_else(|| SciRustError::InvalidConfig("shape overflow".into()))?;
    }
    // gout garde le même rang que l'entrée (seule la dim de l'axe change).
    let g_shape = gout.shape();
    let mut out = vec![0.0f32; checked_prod(in_shape)?];
    let mut g_idx = vec![0usize; in_shape.len()];
    for i in 0..gout.numel()
    {
        let mut rem = i;
        for d in (0..g_shape.len()).rev()
        {
            g_idx[d] = rem % g_shape[d];
            rem /= g_shape[d];
        }
        debug_assert_eq!(rem, 0);
        let mut off = 0usize;
        for (k, &dim) in in_shape.iter().enumerate()
        {
            let idx = if k == axis
            {
                g_idx[k] + start
            }
            else
            {
                g_idx[k]
            };
            debug_assert!(idx < dim, "g_slice: index {idx} >= dim {dim}");
            off += idx * strides[k];
        }
        out[off] = gout.data[i];
    }
    Ok(TensorND::new(out, in_shape.to_vec()))
}

// ------------------------------------------------------------------ //
//  Linear (matmul 2-D, bmm 3-D, linear + bias)                      //
// ------------------------------------------------------------------ //

fn matmul_2d_impl(a: &TensorND, b: &TensorND) -> Result<Vec<f32>> {
    if a.ndim() != 2 || b.ndim() != 2
    {
        return Err(SciRustError::RankMismatch {
            op: "parity::matmul2",
            expected: 2,
            got: if a.ndim() != 2 { a.ndim() } else { b.ndim() },
        });
    }
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    if k != k2
    {
        return Err(SciRustError::DimMismatch {
            op: "parity::matmul2",
            a_cols: k,
            b_rows: k2,
        });
    }
    let numel = m
        .checked_mul(n)
        .ok_or_else(|| SciRustError::InvalidConfig("matmul output size overflow".into()))?;
    let mut out = vec![0.0f32; numel];
    for i in 0..m
    {
        for j in 0..n
        {
            let mut acc = 0.0f32;
            for l in 0..k
            {
                acc += a.data[i * k + l] * b.data[l * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    Ok(out)
}

pub fn matmul2(a: &TensorND, b: &TensorND) -> Result<TensorND> {
    if a.ndim() != 2 || b.ndim() != 2
    {
        return Err(SciRustError::RankMismatch {
            op: "parity::matmul2",
            expected: 2,
            got: if a.ndim() != 2 { a.ndim() } else { b.ndim() },
        });
    }
    let out_shape = vec![a.shape[0], b.shape[1]];
    Ok(TensorND::new(matmul_2d_impl(a, b)?, out_shape))
}

pub fn bmm(a: &TensorND, b: &TensorND) -> Result<TensorND> {
    if a.ndim() != 3 || b.ndim() != 3
    {
        return Err(SciRustError::RankMismatch {
            op: "parity::bmm",
            expected: 3,
            got: if a.ndim() != 3 { a.ndim() } else { b.ndim() },
        });
    }
    let (ba, m, k) = (a.shape[0], a.shape[1], a.shape[2]);
    let (bb, k2, n) = (b.shape[0], b.shape[1], b.shape[2]);
    if ba != bb
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::bmm",
            expected: (ba, 1),
            got: (bb, 1),
        });
    }
    if k != k2
    {
        return Err(SciRustError::DimMismatch {
            op: "parity::bmm",
            a_cols: k,
            b_rows: k2,
        });
    }
    let numel = ba
        .checked_mul(m)
        .and_then(|v| v.checked_mul(n))
        .ok_or_else(|| SciRustError::InvalidConfig("bmm output size overflow".into()))?;
    let mut out = vec![0.0f32; numel];
    for bt in 0..ba
    {
        let aoff = bt * m * k;
        let boff = bt * k * n;
        let ooff = bt * m * n;
        for i in 0..m
        {
            for j in 0..n
            {
                let mut acc = 0.0f32;
                for l in 0..k
                {
                    acc += a.data[aoff + i * k + l] * b.data[boff + l * n + j];
                }
                out[ooff + i * n + j] = acc;
            }
        }
    }
    Ok(TensorND::new(out, vec![ba, m, n]))
}

pub fn linear(x: &TensorND, w: &TensorND, bias: Option<&TensorND>) -> Result<TensorND> {
    if x.ndim() != 2 || w.ndim() != 2
    {
        return Err(SciRustError::RankMismatch {
            op: "parity::linear",
            expected: 2,
            got: if x.ndim() != 2 { x.ndim() } else { w.ndim() },
        });
    }
    let (m, k) = (x.shape[0], x.shape[1]);
    let (n, k2) = (w.shape[0], w.shape[1]);
    if k != k2
    {
        return Err(SciRustError::DimMismatch {
            op: "parity::linear",
            a_cols: k,
            b_rows: k2,
        });
    }
    // F.linear(x, W, b) = x·Wᵀ + b  (W est (out_features, in_features))
    let w_t = w.transpose(&[1, 0])?;
    let mut out = matmul_2d_impl(x, &w_t)?;
    if let Some(b) = bias
    {
        if b.numel() != n
        {
            return Err(SciRustError::ShapeMismatch {
                op: "parity::linear",
                expected: (n, 1),
                got: (b.numel(), 1),
            });
        }
        for i in 0..m
        {
            for j in 0..n
            {
                out[i * n + j] += b.data[j];
            }
        }
    }
    Ok(TensorND::new(out, vec![m, n]))
}

/// Gradients matmul 2-D : ga = gout·bᵀ, gb = aᵀ·gout.
pub fn d_matmul2(gout: &TensorND, a: &TensorND, b: &TensorND) -> Result<(TensorND, TensorND)> {
    let b_t = b.transpose(&[1, 0])?;
    let a_t = a.transpose(&[1, 0])?;
    let ga = matmul2(gout, &b_t)?;
    let gb = matmul2(&a_t, gout)?;
    Ok((ga, gb))
}

/// Gradients bmm : mêmes règles par batch.
pub fn d_bmm(gout: &TensorND, a: &TensorND, b: &TensorND) -> Result<(TensorND, TensorND)> {
    let b_t = b.transpose(&[0, 2, 1])?;
    let a_t = a.transpose(&[0, 2, 1])?;
    let ga = bmm(gout, &b_t)?;
    let gb = bmm(&a_t, gout)?;
    Ok((ga, gb))
}

/// Gradients linear : gx = gout·wᵀ, gw = goutᵀ·x, gb = Σ_r gout (par colonne).
pub fn d_linear(
    gout: &TensorND,
    x: &TensorND,
    w: &TensorND,
    with_bias: bool,
) -> Result<(TensorND, TensorND, TensorND)> {
    // gx = gout·W   (pas Wᵀ : out = x·Wᵀ)
    let gx = matmul2(gout, w)?;
    let gw = matmul2(&gout.transpose(&[1, 0])?, x)?;
    let n = w.shape[0];
    let mut gb = vec![0.0f32; n];
    if with_bias
    {
        let rows = gout.shape[0];
        for i in 0..rows
        {
            for (j, gbj) in gb.iter_mut().enumerate()
            {
                *gbj += gout.data[i * n + j];
            }
        }
    }
    Ok((gx, gw, TensorND::new(gb, vec![n])))
}

// ------------------------------------------------------------------ //
//  Loss (mse_loss, cross_entropy — reduction="mean", sortie scalaire)
// ------------------------------------------------------------------ //

pub fn mse_loss_mean(pred: &TensorND, target: &TensorND) -> Result<TensorND> {
    if pred.shape() != target.shape()
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::mse_loss_mean",
            expected: (pred.numel(), 1),
            got: (target.numel(), 1),
        });
    }
    let n = pred.numel() as f32;
    let sum = pred
        .data
        .iter()
        .zip(target.data.iter())
        .fold(0.0f32, |acc, (&p, &t)| {
            let d = p - t;
            acc + d * d
        });
    Ok(TensorND::new(vec![sum / n], vec![]))
}

pub fn d_mse_loss_mean(pred: &TensorND, target: &TensorND, gout: f32) -> Result<TensorND> {
    if pred.shape() != target.shape()
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::d_mse_loss_mean",
            expected: (pred.numel(), 1),
            got: (target.numel(), 1),
        });
    }
    let n = pred.numel() as f32;
    let out: Vec<f32> = pred
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(&p, &t)| 2.0 * (p - t) * gout / n)
        .collect();
    Ok(TensorND::new(out, pred.shape.clone()))
}

/// cross_entropy (reduction="mean") : moyenne sur les lignes de
/// -log_softmax(logits)[target]. Algorithme logsumexp (stable).
pub fn cross_entropy_mean(logits: &TensorND, targets: &[usize]) -> Result<TensorND> {
    if logits.ndim() != 2
    {
        return Err(SciRustError::RankMismatch {
            op: "parity::cross_entropy_mean",
            expected: 2,
            got: logits.ndim(),
        });
    }
    let (rows, cols) = (logits.shape[0], logits.shape[1]);
    if targets.len() != rows
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::cross_entropy_mean",
            expected: (rows, 1),
            got: (targets.len(), 1),
        });
    }
    let mut total = 0.0f32;
    for (i, &target) in targets.iter().enumerate()
    {
        let base = i * cols;
        if target >= cols
        {
            return Err(SciRustError::IndexOutOfBounds {
                what: "cross_entropy target",
                index: target,
                bound: cols,
            });
        }
        let mut max = f32::NEG_INFINITY;
        for j in 0..cols
        {
            let v = logits.data[base + j];
            if v > max
            {
                max = v;
            }
        }
        let mut sum = 0.0f32;
        for j in 0..cols
        {
            sum += (logits.data[base + j] - max).exp();
        }
        let lse = max + sum.ln();
        total += lse - logits.data[base + target];
    }
    Ok(TensorND::new(vec![total / rows as f32], vec![]))
}

/// Gradient cross_entropy (mean) : (softmax(logits) - onehot)/rows · gout.
pub fn d_cross_entropy_mean(logits: &TensorND, targets: &[usize], gout: f32) -> Result<TensorND> {
    let (rows, cols) = (logits.shape[0], logits.shape[1]);
    if targets.len() != rows
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::d_cross_entropy_mean",
            expected: (rows, 1),
            got: (targets.len(), 1),
        });
    }
    let mut out = vec![0.0f32; rows * cols];
    for (i, &target) in targets.iter().enumerate()
    {
        let base = i * cols;
        if target >= cols
        {
            return Err(SciRustError::IndexOutOfBounds {
                what: "cross_entropy target",
                index: target,
                bound: cols,
            });
        }
        let mut max = f32::NEG_INFINITY;
        for j in 0..cols
        {
            let v = logits.data[base + j];
            if v > max
            {
                max = v;
            }
        }
        let mut sum = 0.0f32;
        for j in 0..cols
        {
            sum += (logits.data[base + j] - max).exp();
        }
        for j in 0..cols
        {
            let p = (logits.data[base + j] - max).exp() / sum;
            let onehot = if j == target { 1.0 } else { 0.0 };
            out[base + j] = (p - onehot) * gout / rows as f32;
        }
    }
    Ok(TensorND::new(out, logits.shape.clone()))
}

// ------------------------------------------------------------------ //
//  Fonctions spéciales f64 (lgamma / digamma / trigamma)            //
//  Rows special du registre : dtypes f64, tol 1e-10. Ces noyaux sont
//  indépendants de TensorND (f32) : le harness les appelle sur des
//  tranches &[f64] extraites des fixtures.
// ------------------------------------------------------------------ //

/// ln Γ(x) par l'approximation de Lanczos (g=7, coefficients Godfrey).
/// Précision ~1e-14 relative pour x > 0.
pub fn lgamma_f64(x: f64) -> f64 {
    const G: f64 = 7.0;
    // Coefficients Godfrey (g=7) — précision délibérée, chaque chiffre compte.
    #[allow(clippy::excessive_precision)]
    const C: [f64; 9] = [
        0.999999999999809_93,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5
    {
        // Réflexion : ln Γ(x) = ln π - ln Γ(1-x) - ln(sin πx)
        let s = (std::f64::consts::PI * x).sin().abs();
        let lg = std::f64::consts::PI.ln() - lgamma_f64(1.0 - x) - s.ln();
        return if x > 0.0 { lg } else { f64::NAN };
    }
    let z = x - 1.0;
    let mut series = C[0];
    for (i, &c) in C.iter().enumerate().skip(1)
    {
        series += c / (z + i as f64);
    }
    let t = z + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + series.ln()
}

/// ψ(x) — digamma par récurrence (remontée à x ≥ 6) + série asymptotique
/// (Bernoulli jusqu'à B20). Précision ~1e-13 relative pour x > 0.
pub fn digamma_f64(x: f64) -> f64 {
    debug_assert!(x > 0.0, "digamma_f64: x doit être > 0, got {x}");
    let mut v = x;
    let mut acc = 0.0f64;
    while v < 6.0
    {
        acc -= 1.0 / v;
        v += 1.0;
    }
    let i = 1.0 / v;
    let i2 = i * i;
    let i4 = i2 * i2;
    let i6 = i4 * i2;
    let i8 = i4 * i4;
    let i10 = i8 * i2;
    let i12 = i8 * i4;
    let i14 = i8 * i6;
    let i16 = i8 * i8;
    let i18 = i10 * i8;
    let i20 = i10 * i10;
    acc + v.ln() - 0.5 * i - i2 / 12.0 + i4 / 120.0 - i6 / 252.0 + i8 / 240.0 - i10 / 132.0
        + 691.0 * i12 / 32760.0
        - i14 / 12.0
        + 3617.0 * i16 / 8160.0
        - 43867.0 * i18 / 143640.0
        + 174611.0 * i20 / 6600.0
}

/// ψ₁(x) — trigamma par récurrence + série asymptotique (B20).
/// Précision ~1e-13 relative pour x > 0. Grad de digamma (autograd torch).
pub fn trigamma_f64(x: f64) -> f64 {
    debug_assert!(x > 0.0, "trigamma_f64: x doit être > 0, got {x}");
    let mut v = x;
    let mut acc = 0.0f64;
    while v < 6.0
    {
        acc += 1.0 / (v * v);
        v += 1.0;
    }
    let i = 1.0 / v;
    let i2 = i * i;
    let i3 = i2 * i;
    let i5 = i3 * i2;
    let i7 = i5 * i2;
    let i9 = i7 * i2;
    let i11 = i9 * i2;
    let i13 = i11 * i2;
    let i15 = i13 * i2;
    let i17 = i15 * i2;
    let i19 = i17 * i2;
    let i21 = i19 * i2;
    acc + i + 0.5 * i2 + i3 / 6.0 - i5 / 30.0 + i7 / 42.0 - i9 / 30.0 + 5.0 * i11 / 66.0
        - 691.0 * i13 / 2730.0
        + 7.0 * i15 / 6.0
        - 3617.0 * i17 / 510.0
        + 43867.0 * i19 / 798.0
        - 174611.0 * i21 / 330.0
}

// ------------------------------------------------------------------ //
//  Shape — cat / gather / unfold                                   //
// ------------------------------------------------------------------ //

fn row_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for d in (0..shape.len()).rev()
    {
        strides[d] = acc;
        acc *= shape[d];
    }
    strides
}

fn linear_off(shape: &[usize], coords: &[usize], strides: &[usize]) -> usize {
    let _ = shape;
    coords.iter().zip(strides).map(|(&c, &s)| c * s).sum()
}

/// cat de deux tenseurs 2-D le long de `dim` (0 ou 1), même rang.
pub fn cat2(a: &TensorND, b: &TensorND, dim: usize) -> Result<TensorND> {
    if a.ndim() != 2 || b.ndim() != 2
    {
        return Err(SciRustError::RankMismatch {
            op: "parity::cat2",
            expected: 2,
            got: a.ndim(),
        });
    }
    if dim >= 2
    {
        return Err(SciRustError::AxisOutOfBounds {
            op: "parity::cat2",
            axis: dim,
            rank: 2,
        });
    }
    for d in 0..2
    {
        if d != dim && a.shape[d] != b.shape[d]
        {
            return Err(SciRustError::ShapeMismatch {
                op: "parity::cat2",
                expected: (a.shape[0], a.shape[1]),
                got: (b.shape[0], b.shape[1]),
            });
        }
    }
    let mut out_shape = a.shape.clone();
    out_shape[dim] = a.shape[dim] + b.shape[dim];
    let mut out = vec![0.0f32; checked_prod(&out_shape)?];
    for (src, start) in [(&a.data, 0usize), (&b.data, a.shape[dim])]
    {
        let mut coords = vec![0usize; 2];
        for i in 0..src.len()
        {
            let mut rem = i;
            for d in (0..2).rev()
            {
                coords[d] = rem % a.shape[d];
                rem /= a.shape[d];
            }
            coords[dim] += start;
            out[linear_off(&out_shape, &coords, &row_strides(&out_shape))] = src[i];
        }
    }
    Ok(TensorND::new(out, out_shape))
}

/// Gradients de cat : tranches de gout.
pub fn d_cat2(
    gout: &TensorND,
    a_shape: &[usize],
    b_shape: &[usize],
    dim: usize,
) -> Result<(TensorND, TensorND)> {
    let ga_shape = a_shape.to_vec();
    let gb_shape = b_shape.to_vec();
    let mut ga = vec![0.0f32; checked_prod(&ga_shape)?];
    let mut gb = vec![0.0f32; checked_prod(&gb_shape)?];
    let mut coords = vec![0usize; 2];
    for i in 0..gout.numel()
    {
        let mut rem = i;
        for d in (0..2).rev()
        {
            coords[d] = rem % gout.shape[d];
            rem /= gout.shape[d];
        }
        let v = gout.data[i];
        if coords[dim] < a_shape[dim]
        {
            ga[linear_off(&ga_shape, &coords, &row_strides(&ga_shape))] = v;
        }
        else
        {
            coords[dim] -= a_shape[dim];
            gb[linear_off(&gb_shape, &coords, &row_strides(&gb_shape))] = v;
        }
    }
    let _ = b_shape;
    Ok((TensorND::new(ga, ga_shape), TensorND::new(gb, gb_shape)))
}

/// gather le long d'un axe (indices même shape que x) — 2-D.
pub fn gather2(x: &TensorND, axis: usize, indices: &[usize]) -> Result<TensorND> {
    if x.ndim() != 2
    {
        return Err(SciRustError::RankMismatch {
            op: "parity::gather2",
            expected: 2,
            got: x.ndim(),
        });
    }
    if axis >= 2
    {
        return Err(SciRustError::AxisOutOfBounds {
            op: "parity::gather2",
            axis,
            rank: 2,
        });
    }
    if indices.len() != x.numel()
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::gather2",
            expected: (1, x.numel()),
            got: (1, indices.len()),
        });
    }
    let mut out = vec![0.0f32; x.numel()];
    let mut coords = vec![0usize; 2];
    for i in 0..x.numel()
    {
        let mut rem = i;
        for d in (0..2).rev()
        {
            coords[d] = rem % x.shape[d];
            rem /= x.shape[d];
        }
        let idx = indices[i];
        if idx >= x.shape[axis]
        {
            return Err(SciRustError::IndexOutOfBounds {
                what: "gather index",
                index: idx,
                bound: x.shape[axis],
            });
        }
        coords[axis] = idx;
        out[i] = x.data[linear_off(x.shape(), &coords, &row_strides(x.shape()))];
    }
    Ok(TensorND::new(out, x.shape.clone()))
}

/// Gradient gather : scatter-add de gout vers les indices.
pub fn d_gather2(
    gout: &TensorND,
    x_shape: &[usize],
    axis: usize,
    indices: &[usize],
) -> Result<TensorND> {
    if axis >= 2
    {
        return Err(SciRustError::AxisOutOfBounds {
            op: "parity::d_gather2",
            axis,
            rank: 2,
        });
    }
    if indices.len() != gout.numel()
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::d_gather2",
            expected: (1, gout.numel()),
            got: (1, indices.len()),
        });
    }
    let mut gx = vec![0.0f32; checked_prod(x_shape)?];
    let mut coords = [0usize; 2];
    for (i, (&idx, &g)) in indices.iter().zip(gout.data.iter()).enumerate()
    {
        let mut rem = i;
        for d in (0..2).rev()
        {
            coords[d] = rem % gout.shape[d];
            rem /= gout.shape[d];
        }
        coords[axis] = idx;
        gx[linear_off(x_shape, &coords, &row_strides(x_shape))] += g;
    }
    Ok(TensorND::new(gx, x_shape.to_vec()))
}

/// unfold (sliding windows) le long d'un axe — 2-D, sortie rang+1.
pub fn unfold2(x: &TensorND, axis: usize, size: usize, step: usize) -> Result<TensorND> {
    if x.ndim() != 2
    {
        return Err(SciRustError::RankMismatch {
            op: "parity::unfold2",
            expected: 2,
            got: x.ndim(),
        });
    }
    if axis >= 2 || size == 0 || step == 0
    {
        return Err(SciRustError::InvalidConfig(format!(
            "parity::unfold2: axis {axis}, size {size}, step {step}"
        )));
    }
    let dim = x.shape[axis];
    if size > dim
    {
        return Err(SciRustError::InvalidConfig(format!(
            "parity::unfold2: size {size} > dim {dim}"
        )));
    }
    let l = (dim - size) / step + 1;
    let mut out_shape = x.shape.clone();
    out_shape[axis] = l;
    out_shape.push(size);
    let strides = row_strides(x.shape());
    let mut out = vec![0.0f32; checked_prod(&out_shape)?];
    // décodage : coordonnées de sortie sur 3 dims (2 remplacées par l + size)
    let mut in_coords = [0usize; 2];
    for (i, o) in out.iter_mut().enumerate()
    {
        let mut rem = i;
        let mut o_coords = [0usize; 3];
        for d in (0..3).rev()
        {
            o_coords[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }
        in_coords.copy_from_slice(&o_coords[..2]);
        in_coords[axis] = o_coords[axis] * step + o_coords[2];
        *o = x.data[linear_off(x.shape(), &in_coords, &strides)];
    }
    Ok(TensorND::new(out, out_shape))
}

/// Gradient unfold : somme des contributions de chaque fenêtre.
pub fn d_unfold2(
    gout: &TensorND,
    x_shape: &[usize],
    axis: usize,
    size: usize,
    step: usize,
) -> Result<TensorND> {
    if axis >= 2 || size == 0 || step == 0
    {
        return Err(SciRustError::InvalidConfig(format!(
            "parity::d_unfold2: axis {axis}, size {size}, step {step}"
        )));
    }
    let strides = row_strides(x_shape);
    let mut gx = vec![0.0f32; checked_prod(x_shape)?];
    let mut in_coords = [0usize; 2];
    let mut o_coords = [0usize; 3];
    for (i, &g) in gout.data.iter().enumerate()
    {
        let mut rem = i;
        for d in (0..3).rev()
        {
            o_coords[d] = rem % gout.shape[d];
            rem /= gout.shape[d];
        }
        in_coords.copy_from_slice(&o_coords[..2]);
        in_coords[axis] = o_coords[axis] * step + o_coords[2];
        gx[linear_off(x_shape, &in_coords, &strides)] += g;
    }
    Ok(TensorND::new(gx, x_shape.to_vec()))
}

// ------------------------------------------------------------------ //
//  Indexing — embedding                                            //
// ------------------------------------------------------------------ //

/// embedding(indices, weight) : out[i, j, :] = w[indices[i, j], :].
pub fn embed(idx_shape: &[usize], indices: &[usize], w: &TensorND) -> Result<TensorND> {
    let (v, d) = (w.shape[0], w.shape[1]);
    let idx_numel = checked_prod(idx_shape)?;
    if indices.len() != idx_numel
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::embed",
            expected: (1, idx_numel),
            got: (1, indices.len()),
        });
    }
    let mut out_shape = idx_shape.to_vec();
    out_shape.push(d);
    let mut out = vec![0.0f32; idx_numel * d];
    for (i, &idx) in indices.iter().enumerate()
    {
        if idx >= v
        {
            return Err(SciRustError::IndexOutOfBounds {
                what: "embedding index",
                index: idx,
                bound: v,
            });
        }
        for j in 0..d
        {
            out[i * d + j] = w.data[idx * d + j];
        }
    }
    Ok(TensorND::new(out, out_shape))
}

/// Gradient embedding : scatter-add vers w.
pub fn d_embed(gout: &TensorND, indices: &[usize], v: usize, d: usize) -> Result<TensorND> {
    let mut gw = vec![0.0f32; v * d];
    for (i, &idx) in indices.iter().enumerate()
    {
        if idx >= v
        {
            return Err(SciRustError::IndexOutOfBounds {
                what: "embedding index",
                index: idx,
                bound: v,
            });
        }
        for j in 0..d
        {
            gw[idx * d + j] += gout.data[i * d + j];
        }
    }
    Ok(TensorND::new(gw, vec![v, d]))
}

// ------------------------------------------------------------------ //
//  Linear — cosine_similarity / normalize                          //
// ------------------------------------------------------------------ //

/// cosine_similarity(a, b, dim=-1) — lignes de la dernière dim.
pub fn cosine_sim(a: &TensorND, b: &TensorND) -> Result<TensorND> {
    if a.shape() != b.shape() || a.ndim() != 2
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::cosine_sim",
            expected: (a.shape[0], a.shape[1]),
            got: (b.shape[0], b.shape[1]),
        });
    }
    let (rows, cols) = (a.shape[0], a.shape[1]);
    let mut out = vec![0.0f32; rows];
    for (r, o) in out.iter_mut().enumerate()
    {
        let base = r * cols;
        let na: f32 = (0..cols)
            .map(|j| a.data[base + j] * a.data[base + j])
            .sum::<f32>()
            .sqrt();
        let nb: f32 = (0..cols)
            .map(|j| b.data[base + j] * b.data[base + j])
            .sum::<f32>()
            .sqrt();
        let dot: f32 = (0..cols).map(|j| a.data[base + j] * b.data[base + j]).sum();
        *o = if na == 0.0 || nb == 0.0
        {
            0.0
        }
        else
        {
            dot / (na * nb)
        };
    }
    Ok(TensorND::new(out, vec![rows]))
}

/// Gradients cosine_similarity (lignes) :
///   ga_i = gout·(b_i/(na·nb) - a_i·dot/(na³·nb))
///   gb_i = gout·(a_i/(na·nb) - b_i·dot/(na·nb³))
pub fn d_cosine_sim(gout: &TensorND, a: &TensorND, b: &TensorND) -> Result<(TensorND, TensorND)> {
    let (rows, cols) = (a.shape[0], a.shape[1]);
    let mut ga = vec![0.0f32; a.numel()];
    let mut gb = vec![0.0f32; b.numel()];
    for r in 0..rows
    {
        let base = r * cols;
        let na: f32 = (0..cols)
            .map(|j| a.data[base + j] * a.data[base + j])
            .sum::<f32>()
            .sqrt();
        let nb: f32 = (0..cols)
            .map(|j| b.data[base + j] * b.data[base + j])
            .sum::<f32>()
            .sqrt();
        let dot: f32 = (0..cols).map(|j| a.data[base + j] * b.data[base + j]).sum();
        let g = gout.data[r];
        if na == 0.0 || nb == 0.0
        {
            continue;
        }
        let den_ab = na * nb;
        let den_a3b = den_ab * na * na;
        let den_ab3 = den_ab * nb * nb;
        for j in 0..cols
        {
            let ai = a.data[base + j];
            let bi = b.data[base + j];
            ga[base + j] = g * (bi / den_ab - ai * dot / den_a3b);
            gb[base + j] = g * (ai / den_ab - bi * dot / den_ab3);
        }
    }
    Ok((
        TensorND::new(ga, a.shape.clone()),
        TensorND::new(gb, b.shape.clone()),
    ))
}

/// F.normalize(x, p=2, dim=1) — normalisation L2 par ligne.
pub fn normalize2(x: &TensorND) -> Result<TensorND> {
    if x.ndim() != 2
    {
        return Err(SciRustError::RankMismatch {
            op: "parity::normalize2",
            expected: 2,
            got: x.ndim(),
        });
    }
    let (rows, cols) = (x.shape[0], x.shape[1]);
    let mut out = vec![0.0f32; x.numel()];
    for r in 0..rows
    {
        let base = r * cols;
        let n: f32 = (0..cols)
            .map(|j| x.data[base + j] * x.data[base + j])
            .sum::<f32>()
            .sqrt();
        for j in 0..cols
        {
            out[base + j] = if n == 0.0 { 0.0 } else { x.data[base + j] / n };
        }
    }
    Ok(TensorND::new(out, x.shape.clone()))
}

/// Gradient normalize L2 : gx_j = (gout_j - x_j·Σ(gout·x)/n²)/n.
pub fn d_normalize2(gout: &TensorND, x: &TensorND) -> Result<TensorND> {
    let (rows, cols) = (x.shape[0], x.shape[1]);
    let mut gx = vec![0.0f32; x.numel()];
    for r in 0..rows
    {
        let base = r * cols;
        let n: f32 = (0..cols)
            .map(|j| x.data[base + j] * x.data[base + j])
            .sum::<f32>()
            .sqrt();
        if n == 0.0
        {
            continue;
        }
        let dot: f32 = (0..cols)
            .map(|j| gout.data[base + j] * x.data[base + j])
            .sum();
        for j in 0..cols
        {
            gx[base + j] = (gout.data[base + j] - x.data[base + j] * dot / (n * n)) / n;
        }
    }
    Ok(TensorND::new(gx, x.shape.clone()))
}

// ------------------------------------------------------------------ //
//  Normalisation — dropout (masque commité) / batch_norm eval       //
// ------------------------------------------------------------------ //

/// dropout : y = x·mask/(1-p), mask ∈ {0,1} commité dans la fixture.
pub fn dropout_apply(x: &TensorND, p: f32, mask: &[f32]) -> Result<TensorND> {
    if mask.len() != x.numel()
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::dropout_apply",
            expected: (1, x.numel()),
            got: (1, mask.len()),
        });
    }
    if !(0.0..1.0).contains(&p)
    {
        return Err(SciRustError::InvalidConfig(format!(
            "parity::dropout_apply: p={p}"
        )));
    }
    let scale = 1.0 / (1.0 - p);
    let out: Vec<f32> = x
        .data
        .iter()
        .zip(mask)
        .map(|(&xv, &m)| xv * m * scale)
        .collect();
    Ok(TensorND::new(out, x.shape.clone()))
}

pub fn d_dropout(gout: &TensorND, p: f32, mask: &[f32]) -> Result<TensorND> {
    let scale = 1.0 / (1.0 - p);
    let gx: Vec<f32> = gout
        .data
        .iter()
        .zip(mask)
        .map(|(&g, &m)| g * m * scale)
        .collect();
    Ok(TensorND::new(gx, gout.shape.clone()))
}

/// batch_norm en mode eval (stats running, affine) : 2-D, canaux = dim 1.
/// y = (x - rm)/sqrt(rv + eps) * w + b
pub fn batch_norm_eval(
    x: &TensorND,
    w: &TensorND,
    b: &TensorND,
    rm: &TensorND,
    rv: &TensorND,
    eps: f32,
) -> Result<TensorND> {
    if x.ndim() != 2
    {
        return Err(SciRustError::RankMismatch {
            op: "parity::batch_norm_eval",
            expected: 2,
            got: x.ndim(),
        });
    }
    let (rows, c) = (x.shape[0], x.shape[1]);
    if w.numel() != c || b.numel() != c || rm.numel() != c || rv.numel() != c
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::batch_norm_eval",
            expected: (1, c),
            got: (1, w.numel()),
        });
    }
    let std: Vec<f32> = rv.data.iter().map(|&v| (v + eps).sqrt()).collect();
    let mut out = vec![0.0f32; x.numel()];
    for i in 0..rows
    {
        let base = i * c;
        for j in 0..c
        {
            let xhat = (x.data[base + j] - rm.data[j]) / std[j];
            out[base + j] = xhat * w.data[j] + b.data[j];
        }
    }
    Ok(TensorND::new(out, x.shape.clone()))
}

/// Gradient batch_norm eval : gx, gw, gb.
pub fn d_batch_norm_eval(
    gout: &TensorND,
    x: &TensorND,
    w: &TensorND,
    rm: &TensorND,
    rv: &TensorND,
    eps: f32,
) -> Result<(TensorND, TensorND, TensorND)> {
    let (rows, c) = (x.shape[0], x.shape[1]);
    let std: Vec<f32> = rv.data.iter().map(|&v| (v + eps).sqrt()).collect();
    let mut gx = vec![0.0f32; x.numel()];
    let mut gw = vec![0.0f32; c];
    let mut gb = vec![0.0f32; c];
    for i in 0..rows
    {
        let base = i * c;
        for j in 0..c
        {
            let xhat = (x.data[base + j] - rm.data[j]) / std[j];
            gx[base + j] = gout.data[base + j] * w.data[j] / std[j];
            gw[j] += gout.data[base + j] * xhat;
            gb[j] += gout.data[base + j];
        }
    }
    Ok((
        TensorND::new(gx, x.shape.clone()),
        TensorND::new(gw, vec![c]),
        TensorND::new(gb, vec![c]),
    ))
}

// ------------------------------------------------------------------ //
//  Positional — rope (paires, base fixe)                            //
// ------------------------------------------------------------------ //

/// rope : dernières dims (L, H, D) avec D pair ; θ_p,i = p / base^(2i/D).
///   y0 = x0·cosθ - x1·sinθ ; y1 = x0·sinθ + x1·cosθ  (paires adjacentes)
pub fn rope(x: &TensorND, base: f32) -> Result<TensorND> {
    let rank = x.ndim();
    if rank < 2
    {
        return Err(SciRustError::RankMismatch {
            op: "parity::rope",
            expected: 2,
            got: rank,
        });
    }
    let d = x.shape[rank - 1];
    if !d.is_multiple_of(2)
    {
        return Err(SciRustError::InvalidConfig(format!(
            "parity::rope: dernière dim impaire {d}"
        )));
    }
    let l = x.shape[0];
    let mut out = vec![0.0f32; x.numel()];
    let outer = x.numel() / (l * d);
    let pairs = d / 2;
    for o in 0..outer
    {
        for p in 0..l
        {
            let base_off = (p * outer + o) * d;
            for k in 0..pairs
            {
                let theta = p as f32 / base.powf(2.0 * k as f32 / d as f32);
                let (c, s) = (theta.cos(), theta.sin());
                let (x0, x1) = (x.data[base_off + 2 * k], x.data[base_off + 2 * k + 1]);
                out[base_off + 2 * k] = x0 * c - x1 * s;
                out[base_off + 2 * k + 1] = x0 * s + x1 * c;
            }
        }
    }
    Ok(TensorND::new(out, x.shape.clone()))
}

/// Gradient rope :
///   gx0 = gout0·cos + gout1·sin ; gx1 = -gout0·sin + gout1·cos
pub fn d_rope(gout: &TensorND, x: &TensorND, base: f32) -> Result<TensorND> {
    let rank = x.ndim();
    let d = x.shape[rank - 1];
    let l = x.shape[0];
    let mut gx = vec![0.0f32; x.numel()];
    let outer = x.numel() / (l * d);
    let pairs = d / 2;
    for o in 0..outer
    {
        for p in 0..l
        {
            let base_off = (p * outer + o) * d;
            for k in 0..pairs
            {
                let theta = p as f32 / base.powf(2.0 * k as f32 / d as f32);
                let (c, s) = (theta.cos(), theta.sin());
                let (g0, g1) = (gout.data[base_off + 2 * k], gout.data[base_off + 2 * k + 1]);
                gx[base_off + 2 * k] = g0 * c + g1 * s;
                gx[base_off + 2 * k + 1] = -g0 * s + g1 * c;
            }
        }
    }
    Ok(TensorND::new(gx, x.shape.clone()))
}

// ------------------------------------------------------------------ //
//  Attention — scaled_dot_product_attention (1 batch)               //
// ------------------------------------------------------------------ //

/// sdpa sans masque ni dropout : out = softmax(q·kᵀ/√d)·v, shapes (B,L,E).
pub fn sdpa(q: &TensorND, k: &TensorND, v: &TensorND) -> Result<TensorND> {
    if q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3
    {
        return Err(SciRustError::RankMismatch {
            op: "parity::sdpa",
            expected: 3,
            got: q.ndim(),
        });
    }
    if q.shape() != k.shape() || q.shape() != v.shape()
    {
        return Err(SciRustError::ShapeMismatch {
            op: "parity::sdpa",
            expected: (q.shape[0], q.shape[1]),
            got: (k.shape[0], k.shape[1]),
        });
    }
    let (b, l, e) = (q.shape[0], q.shape[1], q.shape[2]);
    let inv = (e as f32).sqrt().recip();
    let mut out = vec![0.0f32; b * l * e];
    for batch in 0..b
    {
        let (qb, kb, vb) = (batch * l * e, batch * l * e, batch * l * e);
        let mut scores = vec![0.0f32; l * l];
        for i in 0..l
        {
            for j in 0..l
            {
                let mut acc = 0.0f32;
                for d in 0..e
                {
                    acc += q.data[qb + i * e + d] * k.data[kb + j * e + d];
                }
                scores[i * l + j] = acc * inv;
            }
        }
        for i in 0..l
        {
            let max = scores[i * l..(i + 1) * l]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..l
            {
                sum += (scores[i * l + j] - max).exp();
            }
            for j in 0..l
            {
                let p = (scores[i * l + j] - max).exp() / sum;
                for d in 0..e
                {
                    out[batch * l * e + i * e + d] += p * v.data[vb + j * e + d];
                }
            }
        }
    }
    Ok(TensorND::new(out, q.shape.clone()))
}

/// Gradients sdpa : (gq, gk, gv).
pub fn d_sdpa(
    gout: &TensorND,
    q: &TensorND,
    k: &TensorND,
    v: &TensorND,
) -> Result<(TensorND, TensorND, TensorND)> {
    let (b, l, e) = (q.shape[0], q.shape[1], q.shape[2]);
    let inv = (e as f32).sqrt().recip();
    let mut gq = vec![0.0f32; q.numel()];
    let mut gk = vec![0.0f32; k.numel()];
    let mut gv = vec![0.0f32; v.numel()];
    for batch in 0..b
    {
        let (qb, kb, vb, ob) = (batch * l * e, batch * l * e, batch * l * e, batch * l * e);
        let mut scores = vec![0.0f32; l * l];
        let mut probs = vec![0.0f32; l * l];
        for i in 0..l
        {
            for j in 0..l
            {
                let mut acc = 0.0f32;
                for d in 0..e
                {
                    acc += q.data[qb + i * e + d] * k.data[kb + j * e + d];
                }
                scores[i * l + j] = acc * inv;
            }
        }
        for i in 0..l
        {
            let max = scores[i * l..(i + 1) * l]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..l
            {
                sum += (scores[i * l + j] - max).exp();
            }
            for j in 0..l
            {
                probs[i * l + j] = (scores[i * l + j] - max).exp() / sum;
            }
        }
        // gv = Pᵀ·gout
        for i in 0..l
        {
            for j in 0..l
            {
                let p = probs[i * l + j];
                for d in 0..e
                {
                    gv[vb + j * e + d] += p * gout.data[ob + i * e + d];
                }
            }
        }
        // gP = gout·vᵀ ; gS = P ⊙ (gP - Σ_j gP·P) ; puis gq/gk
        for i in 0..l
        {
            let mut gp_row = vec![0.0f32; l];
            let mut dot = 0.0f32;
            for j in 0..l
            {
                let mut acc = 0.0f32;
                for d in 0..e
                {
                    acc += gout.data[ob + i * e + d] * v.data[vb + j * e + d];
                }
                gp_row[j] = acc;
                dot += acc * probs[i * l + j];
            }
            for j in 0..l
            {
                let gs = probs[i * l + j] * (gp_row[j] - dot);
                for d in 0..e
                {
                    gq[qb + i * e + d] += gs * k.data[kb + j * e + d] * inv;
                    gk[kb + j * e + d] += gs * q.data[qb + i * e + d] * inv;
                }
            }
        }
    }
    Ok((
        TensorND::new(gq, q.shape.clone()),
        TensorND::new(gk, k.shape.clone()),
        TensorND::new(gv, v.shape.clone()),
    ))
}

// ------------------------------------------------------------------ //
//  Quantization / conversion                                        //
// ------------------------------------------------------------------ //

/// fake_quantize_per_tensor_affine (STE) : arrondi à l'éventuel pair,
/// clamp [qmin, qmax] ; y = (q - zp)·scale. Grad = gout (STE).
pub fn fake_quant(x: f32, scale: f32, zp: i64, qmin: i64, qmax: i64) -> f32 {
    let v = x / scale;
    let q = round_half_even(v) as i64 + zp;
    let q = q.clamp(qmin, qmax);
    (q - zp) as f32 * scale
}

fn round_half_even(v: f32) -> f32 {
    let floor = v.floor();
    let frac = v - floor;
    if frac == 0.5
    {
        if (floor as i64) % 2 == 0
        {
            floor
        }
        else
        {
            floor + 1.0
        }
    }
    else
    {
        v.round()
    }
}

pub fn fake_quant_map(t: &TensorND, scale: f32, zp: i64, qmin: i64, qmax: i64) -> Result<TensorND> {
    Ok(TensorND::new(
        t.data
            .iter()
            .map(|&v| fake_quant(v, scale, zp, qmin, qmax))
            .collect(),
        t.shape.clone(),
    ))
}

/// Gradient fake_quantize (STE) : gout partout où la valeur quantisée
/// PRÉ-CLAMP est dans [qmin, qmax], zéro sinon (masquage torch).
pub fn d_fake_quant_map(
    gout: &TensorND,
    x: &TensorND,
    scale: f32,
    zp: i64,
    qmin: i64,
    qmax: i64,
) -> Result<TensorND> {
    let mut gx = vec![0.0f32; x.numel()];
    for (i, (&v, &g)) in x.data.iter().zip(gout.data.iter()).enumerate()
    {
        let q = round_half_even(v / scale) as i64 + zp;
        gx[i] = if qmin <= q && q <= qmax { g } else { 0.0 };
    }
    Ok(TensorND::new(gx, x.shape.clone()))
}

/// to_bf16 : arrondi round-to-nearest-even de la mantisse f32 à 8 bits.
pub fn to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let round_bias = 0x7FFF + ((bits >> 16) & 1);
    f32::from_bits((bits + round_bias) & 0xFFFF_0000)
}

pub fn to_bf16_map(t: &TensorND) -> Result<TensorND> {
    Ok(TensorND::new(
        t.data.iter().map(|&x| to_bf16(x)).collect(),
        t.shape.clone(),
    ))
}
