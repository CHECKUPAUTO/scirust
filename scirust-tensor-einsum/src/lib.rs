//! A general Einstein-summation engine over [`TensorND`].
//!
//! Supports multiple operands, summation over repeated/contracted indices,
//! diagonals (repeated index within one operand), transposition, and both
//! explicit (`"ij,jk->ik"`) and implicit (`"ij,jk"`) output specifications.

use scirust_tensor_core::TensorND;
use std::collections::{BTreeMap, BTreeSet};

/// Evaluate an einsum expression. Returns the resulting tensor or an error
/// describing a malformed pattern / shape mismatch.
pub fn einsum(pattern: &str, inputs: &[&TensorND]) -> Result<TensorND, String> {
    let (in_specs, out_spec) = parse_pattern(pattern, inputs.len())?;

    // Determine the extent of every index label and check consistency.
    let mut sizes: BTreeMap<char, usize> = BTreeMap::new();
    for (spec, t) in in_specs.iter().zip(inputs.iter()) {
        if spec.len() != t.shape.len() {
            return Err(format!(
                "operand '{}' has rank {} but tensor has rank {}",
                spec.iter().collect::<String>(),
                spec.len(),
                t.shape.len()
            ));
        }
        for (p, &lab) in spec.iter().enumerate() {
            let d = t.shape[p];
            match sizes.get(&lab) {
                Some(&prev) if prev != d => {
                    return Err(format!("index '{lab}' has inconsistent sizes {prev} and {d}"))
                }
                _ => {
                    sizes.insert(lab, d);
                }
            }
        }
    }
    for &lab in &out_spec {
        if !sizes.contains_key(&lab) {
            return Err(format!("output index '{lab}' does not appear in any input"));
        }
    }

    // Iterate over output labels first, then the contracted (summed) labels.
    let out_set: BTreeSet<char> = out_spec.iter().copied().collect();
    let mut label_order: Vec<char> = out_spec.clone();
    label_order.extend(sizes.keys().copied().filter(|l| !out_set.contains(l)));
    let dims: Vec<usize> = label_order.iter().map(|l| sizes[l]).collect();
    let label_pos: BTreeMap<char, usize> =
        label_order.iter().enumerate().map(|(i, &l)| (l, i)).collect();

    let out_shape: Vec<usize> = out_spec.iter().map(|l| sizes[l]).collect();
    let out_strides = row_major_strides(&out_shape);
    let out_size: usize = out_shape.iter().product::<usize>().max(1);
    let mut out_data = vec![0.0f32; out_size];

    let total: usize = dims.iter().product::<usize>().max(1);
    let mut idx = vec![0usize; label_order.len()];
    for _ in 0..total {
        let mut prod = 1.0f32;
        for (spec, t) in in_specs.iter().zip(inputs.iter()) {
            let mut off = 0usize;
            for (p, &lab) in spec.iter().enumerate() {
                off += idx[label_pos[&lab]] * t.strides[p];
            }
            prod *= t.data[off];
        }
        let mut o = 0usize;
        for (p, &lab) in out_spec.iter().enumerate() {
            o += idx[label_pos[&lab]] * out_strides[p];
        }
        out_data[o] += prod;
        increment(&mut idx, &dims);
    }

    Ok(TensorND::new(out_data, out_shape))
}

fn parse_pattern(pattern: &str, n_inputs: usize) -> Result<(Vec<Vec<char>>, Vec<char>), String> {
    let pattern: String = pattern.chars().filter(|c| !c.is_whitespace()).collect();
    let (lhs, rhs) = match pattern.split_once("->") {
        Some((l, r)) => (l.to_string(), Some(r.to_string())),
        None => (pattern.clone(), None),
    };
    let in_specs: Vec<Vec<char>> = lhs.split(',').map(|s| s.chars().collect()).collect();
    if in_specs.len() != n_inputs {
        return Err(format!(
            "pattern declares {} operands but {} tensors were given",
            in_specs.len(),
            n_inputs
        ));
    }
    let out_spec: Vec<char> = match rhs {
        Some(r) => r.chars().collect(),
        None => {
            // Implicit mode: indices appearing exactly once, in sorted order.
            let mut counts: BTreeMap<char, usize> = BTreeMap::new();
            for spec in &in_specs {
                for &c in spec {
                    *counts.entry(c).or_insert(0) += 1;
                }
            }
            counts
                .into_iter()
                .filter(|(_, n)| *n == 1)
                .map(|(c, _)| c)
                .collect()
        }
    };
    Ok((in_specs, out_spec))
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut s = vec![1usize; ndim];
    if ndim <= 1 {
        return s;
    }
    for i in (0..ndim - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn increment(idx: &mut [usize], dims: &[usize]) {
    for i in (0..idx.len()).rev() {
        idx[i] += 1;
        if idx[i] < dims[i] {
            return;
        }
        idx[i] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul() {
        // A(2x3) · B(3x2)
        let a = TensorND::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = TensorND::new(vec![7., 8., 9., 10., 11., 12.], vec![3, 2]);
        let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        // [1*7+2*9+3*11, 1*8+2*10+3*12; 4*7+5*9+6*11, 4*8+5*10+6*12]
        assert_eq!(c.data, vec![58., 64., 139., 154.]);
    }

    #[test]
    fn transpose_and_trace_and_sum() {
        let m = TensorND::new(vec![1., 2., 3., 4.], vec![2, 2]);
        assert_eq!(einsum("ij->ji", &[&m]).unwrap().data, vec![1., 3., 2., 4.]);
        assert_eq!(einsum("ii->", &[&m]).unwrap().data, vec![5.0]); // trace
        assert_eq!(einsum("ij->", &[&m]).unwrap().data, vec![10.0]); // full sum
    }

    #[test]
    fn batched_multi_head_attention_scores() {
        // The Documentation.md example: (b,h,i,d),(b,h,j,d) -> (b,h,i,j)
        // Tiny case b=h=1, i=j=1, d=2: scores = Q·Kᵀ over d.
        let q = TensorND::new(vec![1.0, 2.0], vec![1, 1, 1, 2]);
        let k = TensorND::new(vec![3.0, 4.0], vec![1, 1, 1, 2]);
        let s = einsum("bhid,bhjd->bhij", &[&q, &k]).unwrap();
        assert_eq!(s.shape, vec![1, 1, 1, 1]);
        assert_eq!(s.data, vec![11.0]); // 1*3 + 2*4
    }

    #[test]
    fn implicit_output_sorts_free_indices() {
        let a = TensorND::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = TensorND::new(vec![1., 1., 1.], vec![3]);
        // "ij,j" with no -> : free index i ⇒ matrix-vector product.
        let r = einsum("ij,j", &[&a, &b]).unwrap();
        assert_eq!(r.shape, vec![2]);
        assert_eq!(r.data, vec![6.0, 15.0]);
    }
}
