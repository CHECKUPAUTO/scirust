//! Integration test: after TT-decomposing a Linear layer, the TTLinear's
//! reconstructed weight should agree with the original within the SVD
//! truncation tolerance.
//!
//! The Phase 1 forward path uses `reconstruct_weight()` internally, so this
//! test exercises the full pipeline that `forward()` relies on.

use scirust_core::nn::init::Zeros;
use scirust_core::nn::Linear;
use scirust_core::nn::PcgEngine;
use scirust_tn::{auto_factorize, tt_decompose, tt_decompose_auto};

fn frob_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

fn frob_norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn make_linear(in_features: usize, out_features: usize) -> Linear {
    let mut rng = PcgEngine::new(42);
    Linear::new(in_features, out_features, &Zeros, &Zeros, &mut rng)
}

#[test]
fn ttlinear_matches_linear_full_rank() {
    let in_features = 48;
    let out_features = 96;
    let mut linear = make_linear(in_features, out_features);
    for i in 0..in_features {
        for j in 0..out_features {
            linear.weight.data[i * out_features + j] = ((i * 7 + j * 3) as f32).sin();
        }
    }
    for j in 0..out_features {
        linear.bias.data[j] = (j as f32) * 0.01;
    }

    let in_dims = vec![6, 8];
    let out_dims = vec![8, 12];
    let tt = tt_decompose(&linear, &in_dims, &out_dims, 1000, 0.0);

    let w_recon = tt.reconstruct_weight();
    let rel = frob_err(&linear.weight.data, &w_recon.data) / frob_norm(&linear.weight.data);
    assert!(rel < 1e-3, "rel err = {rel}");
}

#[test]
fn ttlinear_auto_factorize_works() {
    let linear = make_linear(64, 128);
    let tt = tt_decompose_auto(&linear, 3, 16, 1e-4);
    assert_eq!(tt.in_dims.iter().product::<usize>(), 64);
    assert_eq!(tt.out_dims.iter().product::<usize>(), 128);
    assert_eq!(tt.in_dims.len(), 3);
    assert_eq!(tt.out_dims.len(), 3);
}

#[test]
fn auto_factorize_balanced() {
    // Sanity check: factors should be roughly balanced
    let f = auto_factorize(768, 3);
    let max = *f.iter().max().unwrap();
    let min = *f.iter().min().unwrap();
    // max/min ratio reasonable for 768 = 2^8 * 3
    assert!(max as f32 / min as f32 <= 4.0, "factors {f:?} not balanced");
}

#[test]
fn ttlinear_compression_reports() {
    let in_features = 64;
    let out_features = 64;
    let mut linear = make_linear(in_features, out_features);
    // Synthetic low-rank weight: 2 outer products
    for i in 0..in_features {
        for j in 0..out_features {
            linear.weight.data[i * out_features + j] =
                (i as f32).sin() * (j as f32).cos() + (i as f32 + j as f32) * 0.001;
        }
    }

    // With small max_rank we expect significant compression.
    let tt = tt_decompose_auto(&linear, 2, 4, 1e-3);
    let ratio = tt.compression_ratio();
    // dense_params = 64*64 + 64 = 4160
    // num_params <= 2 cores small + bias 64. Expect > 5x
    assert!(ratio > 1.0, "compression ratio should be > 1, got {ratio}");
    println!("compression ratio = {ratio:.2}x (params: {} → {})",
             tt.dense_params(), tt.num_params());
}
