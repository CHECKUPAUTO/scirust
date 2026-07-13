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
