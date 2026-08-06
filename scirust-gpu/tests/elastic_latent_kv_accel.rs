//! Differential accelerator checks for Elastic Latent KV Phase 12.

use scirust_gpu::{BackendResult, CpuBackend, RawComputeBackend};
#[cfg(feature = "cuda")]
use scirust_gpu::{BackendError, CudaBackend};
#[cfg(feature = "wgpu")]
use scirust_gpu::WgpuBackend;

fn project<B: RawComputeBackend>(
    backend: &B,
    dense: &[f32],
    basis: &[f32],
    tokens: usize,
    dimension: usize,
    rank: usize,
) -> BackendResult<Vec<f32>> {
    backend.gemm_f32(dense, basis, tokens, dimension, rank)
}

fn reconstruct<B: RawComputeBackend>(
    backend: &B,
    latent: &[f32],
    basis_transposed: &[f32],
    tokens: usize,
    rank: usize,
    dimension: usize,
) -> BackendResult<Vec<f32>> {
    backend.gemm_f32(latent, basis_transposed, tokens, rank, dimension)
}

fn fixtures() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dense = vec![
        0.5, -0.2, 0.8, 0.1,
        -0.4, 0.7, 0.3, -0.6,
        0.9, 0.2, -0.5, 0.4,
    ];
    // First two columns of the 4x4 identity basis: shape [dimension=4, rank=2].
    let basis = vec![
        1.0, 0.0,
        0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0,
    ];
    // Transpose of the same basis: shape [rank=2, dimension=4].
    let basis_transposed = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    ];
    (dense, basis, basis_transposed)
}

#[cfg(any(feature = "wgpu", feature = "cuda"))]
fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "index {index}: actual={actual}, expected={expected}, error={error}, tolerance={tolerance}"
        );
    }
}

#[test]
fn cpu_latent_projection_matches_hand_oracle() {
    let (dense, basis, basis_transposed) = fixtures();
    let latent = project(&CpuBackend, &dense, &basis, 3, 4, 2).unwrap();
    assert_eq!(latent, vec![0.5, -0.2, -0.4, 0.7, 0.9, 0.2]);
    let reconstructed = reconstruct(&CpuBackend, &latent, &basis_transposed, 3, 2, 4).unwrap();
    assert_eq!(
        reconstructed,
        vec![
            0.5, -0.2, 0.0, 0.0,
            -0.4, 0.7, 0.0, 0.0,
            0.9, 0.2, 0.0, 0.0,
        ]
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_latent_projection_matches_cpu_oracle() {
    let (dense, basis, basis_transposed) = fixtures();
    let cpu_latent = project(&CpuBackend, &dense, &basis, 3, 4, 2).unwrap();
    let gpu_latent = project(&WgpuBackend, &dense, &basis, 3, 4, 2)
        .expect("Phase 12 WGPU validation requires an available WGPU adapter");
    assert_close(&gpu_latent, &cpu_latent, 2.0e-5);

    let cpu_dense = reconstruct(&CpuBackend, &cpu_latent, &basis_transposed, 3, 2, 4).unwrap();
    let gpu_dense = reconstruct(&WgpuBackend, &gpu_latent, &basis_transposed, 3, 2, 4)
        .expect("Phase 12 WGPU validation requires an available WGPU adapter");
    assert_close(&gpu_dense, &cpu_dense, 2.0e-5);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_latent_projection_is_honest_and_bounded_when_available() {
    let (dense, basis, basis_transposed) = fixtures();
    let cpu_latent = project(&CpuBackend, &dense, &basis, 3, 4, 2).unwrap();
    match project(&CudaBackend, &dense, &basis, 3, 4, 2) {
        Ok(cuda_latent) => {
            // CUDA path rounds inputs to bf16 before Tensor-core multiplication.
            assert_close(&cuda_latent, &cpu_latent, 5.0e-2);
            let cpu_dense =
                reconstruct(&CpuBackend, &cpu_latent, &basis_transposed, 3, 2, 4).unwrap();
            let cuda_dense =
                reconstruct(&CudaBackend, &cuda_latent, &basis_transposed, 3, 2, 4).unwrap();
            assert_close(&cuda_dense, &cpu_dense, 7.5e-2);
        }
        Err(BackendError::Unavailable("cuda")) => {}
        Err(error) => panic!("unexpected CUDA latent-projection failure: {error:?}"),
    }
}
