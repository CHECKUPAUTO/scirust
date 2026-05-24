//! Integration tests: matrix → TT decomposition → reconstructed matrix.

use scirust_tn::tt::decompose::{
    reconstruct_matrix, reconstruct_tensor, tt_decompose_matrix, tt_decompose_tensor,
};
use scirust_tn::TensorND;

fn frob_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

fn frob_norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Round-trip a random tensor of shape (4, 6, 5) at full rank.
#[test]
fn tensor_3mode_full_rank_round_trip() {
    let shape = vec![4, 6, 5];
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i * 37) as f32 * 0.0173).sin() + (i as f32) * 0.01)
        .collect();
    let t = TensorND::new(shape, data.clone());

    let tt = tt_decompose_tensor(&t, 1000, 0.0);
    let recon = reconstruct_tensor(&tt);
    let rel = frob_err(&data, &recon) / frob_norm(&data);
    assert!(rel < 1e-3, "rel err = {rel}");
}

/// Round-trip a synthetic outer-product tensor → expect ranks of 1.
#[test]
fn tensor_outer_product_yields_rank_one() {
    let u = [1.0f32, 2.0, -0.5, 3.0];
    let v = [0.7f32, -1.1, 0.4];
    let w = [2.0f32, 0.5];

    let mut data = vec![0.0f32; 4 * 3 * 2];
    for i in 0..4 {
        for j in 0..3 {
            for k in 0..2 {
                data[(i * 3 + j) * 2 + k] = u[i] * v[j] * w[k];
            }
        }
    }
    let t = TensorND::new(vec![4, 3, 2], data.clone());
    let tt = tt_decompose_tensor(&t, 100, 1e-6);
    assert_eq!(tt.ranks, vec![1, 1, 1, 1]);
    let recon = reconstruct_tensor(&tt);
    assert!(frob_err(&data, &recon) / frob_norm(&data) < 1e-4);
}

/// Matrix → TT-Linear → matrix round-trip. Standard transformer-FFN-sized.
#[test]
fn matrix_round_trip_768x3072() {
    // Smaller version to keep the test fast: 48x192 = 9216 elements
    let in_features = 48;
    let out_features = 192;
    let w: Vec<f32> = (0..(in_features * out_features))
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();

    // Factor: 48 = 4 * 12, 192 = 12 * 16 → d=2
    let in_dims = vec![4, 12];
    let out_dims = vec![12, 16];
    let tt = tt_decompose_matrix(&w, &in_dims, &out_dims, 1000, 0.0);
    let recon = reconstruct_matrix(&tt, &in_dims, &out_dims);
    let rel = frob_err(&w, &recon) / frob_norm(&w);
    assert!(rel < 1e-3, "rel err = {rel}");
}

/// Verify that truncation produces controlled error.
#[test]
fn matrix_truncation_error_bound() {
    // Build a matrix that's mathematically close to low-rank
    let in_features = 32;
    let out_features = 32;
    let mut w = vec![0.0f32; in_features * out_features];
    // Sum of a few outer products → effective rank ~3
    for r in 0..3 {
        for i in 0..in_features {
            for j in 0..out_features {
                w[i * out_features + j] +=
                    ((i as f32) * (r as f32 + 1.0) * 0.1).sin()
                        * ((j as f32) * (r as f32 + 1.0) * 0.13).cos();
            }
        }
    }

    let in_dims = vec![4, 8];
    let out_dims = vec![4, 8];
    let tt_high = tt_decompose_matrix(&w, &in_dims, &out_dims, 32, 0.0);
    let tt_low = tt_decompose_matrix(&w, &in_dims, &out_dims, 4, 0.0);

    let rec_high = reconstruct_matrix(&tt_high, &in_dims, &out_dims);
    let rec_low = reconstruct_matrix(&tt_low, &in_dims, &out_dims);

    let err_high = frob_err(&w, &rec_high) / frob_norm(&w);
    let err_low = frob_err(&w, &rec_low) / frob_norm(&w);

    // Higher rank should be no worse than lower rank.
    assert!(err_high <= err_low + 1e-4, "high {err_high} > low {err_low}");
}
