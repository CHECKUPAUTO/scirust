// scirust-simd/benches/geometry_bench.rs
//
// Benchmarks criterion de la géométrie générique : rotation de vecteur et
// construction angle-axe, en `Quaternion<Q16_16>` (virgule fixe déterministe)
// vs `Quaternion<f32>` (référence).
//
// `rotate_vector` n'utilise que des opérations d'anneau (produits croisés) :
// le chemin virgule fixe y est proche du flottant. `from_axis_angle` appelle
// sin/cos/sqrt : on y mesure le surcoût des transcendantes entières.
//
// Lancement (cible AVX2 pour éviter la sur-détection AVX-512 en VM) :
//   RUSTFLAGS="-C target-cpu=x86-64-v3" \
//     cargo bench -p scirust-simd --features portable-simd --bench geometry_bench

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_simd::fixed::Q16_16;
use scirust_simd::geometry::Quaternion;

const N: usize = 1 << 12;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// `N` vecteurs 3D dans `[-1, 1)`, en fixe et en `f32`.
fn vectors(seed: u64) -> (Vec<[Q16_16; 3]>, Vec<[f32; 3]>) {
    let mut rng = Lcg(seed);
    let mut fx = Vec::with_capacity(N);
    let mut ff = Vec::with_capacity(N);
    for _ in 0..N
    {
        let v = [rng.unit(), rng.unit(), rng.unit()];
        fx.push([
            Q16_16::try_from(v[0]).unwrap(),
            Q16_16::try_from(v[1]).unwrap(),
            Q16_16::try_from(v[2]).unwrap(),
        ]);
        ff.push([v[0] as f32, v[1] as f32, v[2] as f32]);
    }
    (fx, ff)
}

/// Rotation d'un flux de vecteurs par un quaternion unitaire fixe.
fn bench_rotate(c: &mut Criterion) {
    let (fx, ff) = vectors(0xA1);
    let qx = Quaternion::<Q16_16>::from_axis_angle(
        [
            Q16_16::try_from(0.3).unwrap(),
            Q16_16::try_from(-0.6).unwrap(),
            Q16_16::try_from(0.75).unwrap(),
        ],
        Q16_16::try_from(0.9).unwrap(),
    );
    let qf = Quaternion::<f32>::from_axis_angle([0.3, -0.6, 0.75], 0.9);

    let mut g = c.benchmark_group("rotate_vector");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut acc = [Q16_16::zero(); 3];
            for &v in black_box(&fx)
            {
                let r = qx.rotate_vector(v);
                acc[0] += r[0];
                acc[1] += r[1];
                acc[2] += r[2];
            }
            acc
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            let mut acc = [0.0f32; 3];
            for &v in black_box(&ff)
            {
                let r = qf.rotate_vector(v);
                acc[0] += r[0];
                acc[1] += r[1];
                acc[2] += r[2];
            }
            acc
        })
    });
    g.finish();
}

/// Construction angle-axe (sin/cos/sqrt sous le capot).
fn bench_from_axis_angle(c: &mut Criterion) {
    let mut rng = Lcg(0xB2);
    let angles_x: Vec<Q16_16> = (0..N)
        .map(|_| Q16_16::try_from(rng.unit() * 3.0).unwrap())
        .collect();
    let mut rng = Lcg(0xB2);
    let angles_f: Vec<f32> = (0..N).map(|_| (rng.unit() * 3.0) as f32).collect();

    let mut g = c.benchmark_group("from_axis_angle");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut acc = Quaternion::<Q16_16>::zero();
            for &a in black_box(&angles_x)
            {
                let q = Quaternion::<Q16_16>::from_axis_angle(
                    [Q16_16::zero(), Q16_16::zero(), Q16_16::one()],
                    a,
                );
                acc = acc + q;
            }
            acc
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            let mut acc = Quaternion::<f32>::zero();
            for &a in black_box(&angles_f)
            {
                let q = Quaternion::<f32>::from_axis_angle([0.0, 0.0, 1.0], a);
                acc = acc + q;
            }
            acc
        })
    });
    g.finish();
}

/// Interpolation sphérique (slerp, via acos + sin) fixe vs f32.
fn bench_slerp(c: &mut Criterion) {
    let ax = Quaternion::<Q16_16>::from_axis_angle(
        [Q16_16::zero(), Q16_16::zero(), Q16_16::one()],
        Q16_16::try_from(0.2).unwrap(),
    );
    let bx = Quaternion::<Q16_16>::from_axis_angle(
        [
            Q16_16::try_from(0.6).unwrap(),
            Q16_16::try_from(0.0).unwrap(),
            Q16_16::try_from(0.8).unwrap(),
        ],
        Q16_16::try_from(1.5).unwrap(),
    );
    let af = Quaternion::<f32>::from_axis_angle([0.0, 0.0, 1.0], 0.2);
    let bf = Quaternion::<f32>::from_axis_angle([0.6, 0.0, 0.8], 1.5);

    let steps = 256u32;
    let mut g = c.benchmark_group("slerp");
    g.throughput(Throughput::Elements(steps as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut acc = Quaternion::<Q16_16>::zero();
            for s in 0..steps
            {
                let t = Q16_16::try_from(s as f64 / steps as f64).unwrap();
                acc = acc + Quaternion::slerp(black_box(ax), black_box(bx), t);
            }
            acc
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            let mut acc = Quaternion::<f32>::zero();
            for s in 0..steps
            {
                let t = s as f32 / steps as f32;
                acc = acc + Quaternion::slerp(black_box(af), black_box(bf), t);
            }
            acc
        })
    });
    g.finish();
}

criterion_group!(benches, bench_rotate, bench_from_axis_angle, bench_slerp);
criterion_main!(benches);
