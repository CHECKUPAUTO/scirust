use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_simd::gemm::sgemm_tiled;
use scirust_simd::matrix::view::{MatrixView, MatrixViewMut};
use scirust_simd::matrix::workspace_gemm::{GemmWorkspaceF32, sgemm_tiled_with_workspace};

const M: usize = 256;
const K: usize = 256;
const N: usize = 256;

fn deterministic_data(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (state >> 40) as u32;
            bits as f32 / ((1_u32 << 24) as f32) - 0.5
        })
        .collect()
}

fn bench_gemm_workspace(c: &mut Criterion) {
    let a = deterministic_data(M * K, 0xA11C_E001);
    let b = deterministic_data(K * N, 0xB22C_E002);
    let mut output = vec![0.0f32; M * N];
    let mut workspace = GemmWorkspaceF32::new();

    let mut group = c.benchmark_group("sgemm_workspace_256");
    group.throughput(Throughput::Elements((M * K * N) as u64));

    group.bench_function(BenchmarkId::new("allocation", "one_shot"), |bencher| {
        bencher.iter(|| {
            sgemm_tiled(
                1.0,
                MatrixView::new(black_box(&a), M, K),
                MatrixView::new(black_box(&b), K, N),
                0.0,
                MatrixViewMut::new(black_box(&mut output), M, N),
            );
            black_box(output[0]);
        });
    });

    group.bench_function(BenchmarkId::new("allocation", "reused_workspace"), |bencher| {
        bencher.iter(|| {
            sgemm_tiled_with_workspace(
                1.0,
                MatrixView::new(black_box(&a), M, K),
                MatrixView::new(black_box(&b), K, N),
                0.0,
                MatrixViewMut::new(black_box(&mut output), M, N),
                black_box(&mut workspace),
            );
            black_box(output[0]);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_gemm_workspace);
criterion_main!(benches);
