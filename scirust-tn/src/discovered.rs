//! Algorithme decouvert automatiquement par forge (FunSearch/AlphaEvolve-style).
//! Injecte le 2026-06-04T20:11:33Z.
//!
//! model = hf.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:Q8_0
//! params = 1564
//! baseline_params = 4096
//! ratio = 2.62x
//! L2_train = 7.213e-16
//! bytes = 1491
//! verified_holdout = true
//!
//! NE PAS editer a la main : regenere par le binaire `inject_elite`.

use nalgebra::DMatrix;

pub fn compress(flat_tensor: &[f64], shape: &[usize]) -> Vec<f64> {
    let n0 = shape[0];
    let ncols: usize = shape[1..].iter().product();
    let m = DMatrix::from_row_slice(n0, ncols, flat_tensor);
    let svd = m.svd(true, true);
    let s = &svd.singular_values;
    let u = svd.u.as_ref().unwrap();
    let vt = svd.v_t.as_ref().unwrap();
    let smax = s.iter().cloned().fold(0.0_f64, f64::max);
    let tol = 1e-9_f64.max(smax * 1e-7);
    let mut r = 0usize;
    for i in 0..s.len()
    {
        if s[i] > tol
        {
            r += 1;
        }
    }
    if r == 0
    {
        r = 1;
    }
    let mut out = Vec::with_capacity(1 + r * (n0 + ncols + 1));
    out.push(r as f64);
    for j in 0..r
    {
        for i in 0..n0
        {
            out.push(u[(i, j)]);
        }
    }
    for j in 0..r
    {
        out.push(s[j]);
    }
    for j in 0..r
    {
        for c in 0..ncols
        {
            out.push(vt[(j, c)]);
        }
    }
    out
}

pub fn reconstruct(compressed: &[f64], shape: &[usize], rebuilt: &mut [f64]) {
    let n0 = shape[0];
    let ncols: usize = shape[1..].iter().product();
    let r = compressed[0] as usize;
    let mut idx = 1usize;
    let u = DMatrix::from_column_slice(n0, r, &compressed[idx..idx + n0 * r]);
    idx += n0 * r;
    let s: Vec<f64> = compressed[idx..idx + r].to_vec();
    idx += r;
    let vt = DMatrix::from_row_slice(r, ncols, &compressed[idx..idx + r * ncols]);
    let mut us = u.clone();
    for j in 0..r
    {
        for i in 0..n0
        {
            us[(i, j)] *= s[j];
        }
    }
    let m = us * vt;
    for i in 0..n0
    {
        for c in 0..ncols
        {
            rebuilt[i * ncols + c] = m[(i, c)];
        }
    }
}

#[cfg(test)]
mod forge_tests {
    use super::*;
    #[test]
    fn roundtrip_low_rank() {
        let shape = [4usize, 4usize];
        let a = [1.0_f64, 2.0, 3.0, 4.0];
        let b = [0.5_f64, -1.0, 2.0, 0.25];
        let mut flat = vec![0.0_f64; 16];
        for i in 0..4
        {
            for j in 0..4
            {
                flat[i * 4 + j] = a[i] * b[j];
            }
        }
        let comp = compress(&flat, &shape);
        let mut rebuilt = vec![0.0_f64; 16];
        reconstruct(&comp, &shape, &mut rebuilt);
        let err: f64 = flat
            .iter()
            .zip(&rebuilt)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-9, "roundtrip L2 trop grand: {err}");
        assert!(comp.len() < flat.len(), "pas de compression");
    }
}
