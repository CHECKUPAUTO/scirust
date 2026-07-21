// Migration note (scirust-bench-schema): like `io_bench.rs`, this file has NO
// random data generator to pin -- there is no seed_from_u64/Lcg::new/PcgEngine::new
// call anywhere below. `bench_bump_allocation`, `bench_aligned_fill`, and
// `bench_slab_reuse` each operate on one fixed-size, fixed-value buffer
// (4096 f32 elements filled with the constant black_box(1.0)/black_box(2.0);
// a 1024-slot, 128-byte Slab written with the constant black_box(3.0)) --
// there is no swept parametrization axis and no randomness to make
// reproducible in the first place. `BenchRecord::seed` is still mandatory,
// so a conversion uses `0` to mean "no RNG -- fixed, non-random input", e.g.
// (after `cargo bench --bench arena_bench`, reading
// target/criterion/arena/alloc_fill_4096_f32/new/estimates.json -- the
// "arena/..." prefix baked into each bench_function name is criterion's
// group, "alloc_fill_4096_f32"/"aligned_fill_4096_f32"/"slab_alloc_write_free"
// are its bench ids):
//   scirust_bench_schema::criterion_estimate_to_record(
//       &estimates_json,
//       "scirust-arena/bump_allocation",  // kernel
//       "fixed_4096_f32",                 // dataset (no swept axis; fixed size)
//       "PinnedArena::alloc_slice_fill",   // method
//       0,                                 // seed: no RNG, constant input
//   )
// See scirust-bench-schema's crate docs ("Migrating criterion targets") for the full pattern.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use scirust_arena::{AlignedVec, PinnedArena, Slab};

fn bench_bump_allocation(c: &mut Criterion) {
    let mut arena = PinnedArena::new_for_type::<f32>(4096);
    c.bench_function("arena/alloc_fill_4096_f32", |b| {
        b.iter(|| {
            {
                let values = arena.alloc_slice_fill(4096, black_box(1.0f32)).unwrap();
                black_box(values[4095]);
            }
            arena.reset();
        });
    });
}

fn bench_aligned_fill(c: &mut Criterion) {
    let mut values = AlignedVec::<f32>::new(4096);
    c.bench_function("arena/aligned_fill_4096_f32", |b| {
        b.iter(|| {
            values.fill(black_box(2.0));
            black_box(values.as_slice()[4095]);
        });
    });
}

fn bench_slab_reuse(c: &mut Criterion) {
    let mut slab = Slab::<f32, 128>::new(1024);
    c.bench_function("arena/slab_alloc_write_free", |b| {
        b.iter(|| {
            let handle = slab.alloc().unwrap();
            slab.data_slice(handle).unwrap()[0] = black_box(3.0);
            slab.free(handle);
        });
    });
}

criterion_group!(
    benches,
    bench_bump_allocation,
    bench_aligned_fill,
    bench_slab_reuse
);
criterion_main!(benches);
