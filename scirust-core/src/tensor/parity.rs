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
