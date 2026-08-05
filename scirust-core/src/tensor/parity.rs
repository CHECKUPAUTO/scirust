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
