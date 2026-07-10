//! Haar wavelet transform — forward and inverse DWT.
//!
//! All operations are deterministic `f64`.  The boundary extension is
//! **periodic** (wraparound), which keeps the number of coefficients
//! exactly equal to the input length at every level — perfect for
//! coefficient-wise thresholding.
//!
//! ## Example
//! ```text
//! s  = [1, 2, 3, 4, 5, 6, 7, 8]
//! cA = [2.12, 4.95, 7.78, 10.6]   (scaling / approximation)
//! cD = [-0.707, -0.707, -0.707, -0.707]  (detail / wavelet)
//! ```

use core::f64::consts::FRAC_1_SQRT_2;

/// Perform a 1-level **Haar discrete wavelet transform** (analysis) in place.
///
/// The first half of the output holds the approximation coefficients; the second
/// half holds the detail coefficients.  Length must be even.
pub fn haar_dwt(data: &mut [f64]) {
    let n = data.len();
    assert!(n % 2 == 0, "length must be even for Haar DWT, got {n}");
    if n < 2
    {
        return;
    }
    // s₀[j] = (data[2j] + data[2j+1]) / √2
    // d₀[j] = (data[2j] - data[2j+1]) / √2
    let half = n / 2;
    let mut tmp = vec![0.0; n];
    for j in 0..half
    {
        let a = data[2 * j];
        let b = data[2 * j + 1];
        tmp[j] = (a + b) * FRAC_1_SQRT_2;
        tmp[half + j] = (a - b) * FRAC_1_SQRT_2;
    }
    data.copy_from_slice(&tmp);
}

/// Perform a 1-level **inverse Haar DWT** (synthesis) in place.
///
/// Input must be in the same arrangement as `haar_dwt` output: first half
/// approximation, second half detail.  Length must be even.
pub fn haar_idwt(data: &mut [f64]) {
    let n = data.len();
    assert!(n % 2 == 0, "length must be even for Haar IDWT, got {n}");
    if n < 2
    {
        return;
    }
    let half = n / 2;
    let mut tmp = vec![0.0; n];
    for j in 0..half
    {
        let c = data[j];
        let d = data[half + j];
        tmp[2 * j] = (c + d) * FRAC_1_SQRT_2;
        tmp[2 * j + 1] = (c - d) * FRAC_1_SQRT_2;
    }
    data.copy_from_slice(&tmp);
}

/// Perform a multi-level **Haar DWT** (pyramid decomposition) in place.
///
/// `levels` is the number of decomposition levels (must be ≥ 1).  After
/// decomposition, the first `n / 2^levels` coefficients are the coarsest
/// approximation; the remaining coefficients are detail coefficients from
/// each level.
///
/// Length must be divisible by `2^levels`.
///
/// A single scratch buffer of size `n` is allocated — zero heap churn per level.
pub fn haar_dwt_multilevel(data: &mut [f64], levels: usize) {
    let n = data.len();
    let divisor = 1 << levels;
    assert!(
        n % divisor == 0,
        "length must be divisible by 2^{levels}, got {n}"
    );
    let mut scratch = vec![0.0; n];
    let mut len = n;
    for _ in 0..levels
    {
        let half = len / 2;
        for j in 0..half
        {
            let a = data[2 * j];
            let b = data[2 * j + 1];
            scratch[j] = (a + b) * FRAC_1_SQRT_2;
            scratch[half + j] = (a - b) * FRAC_1_SQRT_2;
        }
        data[..len].copy_from_slice(&scratch[..len]);
        len = half;
    }
}

/// Perform a multi-level **inverse Haar DWT** in place.
///
/// Must match the number of levels used in `haar_dwt_multilevel`.
///
/// A single scratch buffer of size `n` is allocated — zero heap churn per level.
pub fn haar_idwt_multilevel(data: &mut [f64], levels: usize) {
    let n = data.len();
    let divisor = 1 << levels;
    assert!(
        n % divisor == 0,
        "length must be divisible by 2^{levels}, got {n}"
    );
    let mut scratch = vec![0.0; n];
    let mut len = n / divisor;
    for _ in 0..levels
    {
        let half = len;
        let full = len * 2;
        for j in 0..half
        {
            let c = data[j];
            let d = data[half + j];
            scratch[2 * j] = (c + d) * FRAC_1_SQRT_2;
            scratch[2 * j + 1] = (c - d) * FRAC_1_SQRT_2;
        }
        data[..full].copy_from_slice(&scratch[..full]);
        len = full;
    }
}

/// Build a **Haar DWT matrix** `W` (size `n×n`) for a 1-level transform.
///
/// Multiplying a signal vector by `W` applies the 1-level Haar DWT (approximation
/// then detail coefficients).  `W` is orthogonal, so `Wᵀ` reconstructs.
///
/// Construction: each column `j` is `haar_dwt(e_j)` where `e_j` is the `j`-th
/// standard basis vector.  This guarantees `W · x == haar_dwt(x)`.
pub fn haar_matrix(n: usize) -> Vec<Vec<f64>> {
    assert!(
        n.is_power_of_two(),
        "Haar matrix requires power-of-two, got {n}"
    );
    let mut w = vec![vec![0.0; n]; n];
    let mut col_buf = vec![0.0; n];
    for j in 0..n
    {
        col_buf.fill(0.0);
        col_buf[j] = 1.0;
        haar_dwt(&mut col_buf);
        for i in 0..n
        {
            w[i][j] = col_buf[i];
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haar_round_trip_single_level() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = data.clone();
        haar_dwt(&mut data);
        haar_idwt(&mut data);
        for (a, b) in data.iter().zip(&orig)
        {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn haar_multilevel_round_trip() {
        let mut data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let orig = data.clone();
        haar_dwt_multilevel(&mut data, 3);
        haar_idwt_multilevel(&mut data, 3);
        for (a, b) in data.iter().zip(&orig)
        {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn haar_matrix_is_orthogonal() {
        let n = 8;
        let w = haar_matrix(n);
        // Check W·W^T = I
        for i in 0..n
        {
            for j in 0..n
            {
                let dot: f64 = (0..n).map(|k| w[i][k] * w[j][k]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-12, "W·Wᵀ[{i},{j}] = {dot}");
            }
        }
    }

    #[test]
    fn haar_matrix_matches_in_place_dwt() {
        let n = 8;
        let sig = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        // In-place DWT
        let mut expected = sig.clone();
        haar_dwt(&mut expected);
        // Matrix multiply
        let w = haar_matrix(n);
        let actual: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| w[i][j] * sig[j]).sum())
            .collect();
        for (a, b) in actual.iter().zip(&expected)
        {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }
}
