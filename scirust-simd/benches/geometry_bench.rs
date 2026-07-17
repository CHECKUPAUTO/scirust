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
use scirust_simd::geometry::{DualQuaternion, Quaternion, Transform};

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

/// Reconstruction depuis une matrice de rotation (méthode de Shepperd, sqrt +
/// divisions réelles) fixe vs f32.
fn bench_from_rotation_matrix(c: &mut Criterion) {
    let mut rng = Lcg(0xC3);
    let angles_x: Vec<Q16_16> = (0..N)
        .map(|_| Q16_16::try_from(rng.unit() * 3.0).unwrap())
        .collect();
    let mut rng = Lcg(0xC3);
    let angles_f: Vec<f32> = (0..N).map(|_| (rng.unit() * 3.0) as f32).collect();
    let axis_x = [
        Q16_16::try_from(0.267).unwrap(),
        Q16_16::try_from(0.535).unwrap(),
        Q16_16::try_from(0.802).unwrap(),
    ];
    let axis_f = [0.267f32, 0.535, 0.802];
    let mats_x: Vec<[[Q16_16; 3]; 3]> = angles_x
        .iter()
        .map(|&a| Quaternion::<Q16_16>::from_axis_angle(axis_x, a).to_rotation_matrix())
        .collect();
    let mats_f: Vec<[[f32; 3]; 3]> = angles_f
        .iter()
        .map(|&a| Quaternion::<f32>::from_axis_angle(axis_f, a).to_rotation_matrix())
        .collect();

    let mut g = c.benchmark_group("from_rotation_matrix");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut acc = Quaternion::<Q16_16>::zero();
            for &m in black_box(&mats_x)
            {
                acc = acc + Quaternion::<Q16_16>::from_rotation_matrix(m);
            }
            acc
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            let mut acc = Quaternion::<f32>::zero();
            for &m in black_box(&mats_f)
            {
                acc = acc + Quaternion::<f32>::from_rotation_matrix(m);
            }
            acc
        })
    });
    g.finish();
}

/// Aller-retour angles d'Euler → quaternion → angles d'Euler (`atan2`/`asin`
/// sous le capot) fixe vs f32.
fn bench_euler_roundtrip(c: &mut Criterion) {
    let mut rng = Lcg(0xD4);
    let triples_x: Vec<[Q16_16; 3]> = (0..N)
        .map(|_| {
            [
                Q16_16::try_from(rng.unit()).unwrap(),
                Q16_16::try_from(rng.unit() * 0.5).unwrap(),
                Q16_16::try_from(rng.unit()).unwrap(),
            ]
        })
        .collect();
    let mut rng = Lcg(0xD4);
    let triples_f: Vec<[f32; 3]> = (0..N)
        .map(|_| {
            [
                rng.unit() as f32,
                (rng.unit() * 0.5) as f32,
                rng.unit() as f32,
            ]
        })
        .collect();

    let mut g = c.benchmark_group("euler_roundtrip");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut acc = Quaternion::<Q16_16>::zero();
            for &t in black_box(&triples_x)
            {
                let q = Quaternion::<Q16_16>::from_euler(t[0], t[1], t[2]);
                let (r, p, y) = q.to_euler();
                acc = acc + Quaternion::<Q16_16>::new(r, p, y, Q16_16::zero());
            }
            acc
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| {
            let mut acc = Quaternion::<f32>::zero();
            for &t in black_box(&triples_f)
            {
                let q = Quaternion::<f32>::from_euler(t[0], t[1], t[2]);
                let (r, p, y) = q.to_euler();
                acc = acc + Quaternion::<f32>::new(r, p, y, 0.0);
            }
            acc
        })
    });
    g.finish();
}

/// Composition de `SE(3)` (produit de Hamilton + rotation d'une translation)
/// et transformation d'un flux de points, fixe vs f32.
fn bench_transform(c: &mut Criterion) {
    let (fx, ff) = vectors(0xE5);
    let ax = Quaternion::<Q16_16>::from_axis_angle(
        [
            Q16_16::try_from(0.267).unwrap(),
            Q16_16::try_from(0.535).unwrap(),
            Q16_16::try_from(0.802).unwrap(),
        ],
        Q16_16::try_from(0.9).unwrap(),
    );
    let bx = Quaternion::<Q16_16>::from_axis_angle(
        [
            Q16_16::try_from(0.408).unwrap(),
            Q16_16::try_from(0.408).unwrap(),
            Q16_16::try_from(0.816).unwrap(),
        ],
        Q16_16::try_from(1.6).unwrap(),
    );
    let tx_a = Transform::new(
        ax,
        [
            Q16_16::try_from(0.2).unwrap(),
            Q16_16::try_from(-0.4).unwrap(),
            Q16_16::try_from(0.6).unwrap(),
        ],
    );
    let tx_b = Transform::new(
        bx,
        [
            Q16_16::try_from(-0.5).unwrap(),
            Q16_16::try_from(0.3).unwrap(),
            Q16_16::try_from(0.1).unwrap(),
        ],
    );
    let af = Quaternion::<f32>::from_axis_angle([0.267, 0.535, 0.802], 0.9);
    let bf = Quaternion::<f32>::from_axis_angle([0.408, 0.408, 0.816], 1.6);
    let tf_a = Transform::new(af, [0.2, -0.4, 0.6]);
    let tf_b = Transform::new(bf, [-0.5, 0.3, 0.1]);

    let mut g = c.benchmark_group("transform_compose");
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| black_box(tx_a).compose(black_box(&tx_b)))
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| black_box(tf_a).compose(black_box(&tf_b)))
    });
    g.finish();

    let mut g = c.benchmark_group("transform_point");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut acc = [Q16_16::zero(); 3];
            for &v in black_box(&fx)
            {
                let r = tx_a.transform_point(v);
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
                let r = tf_a.transform_point(v);
                acc[0] += r[0];
                acc[1] += r[1];
                acc[2] += r[2];
            }
            acc
        })
    });
    g.finish();
}

/// Coût du quaternion dual (encodage unifié de `SE(3)`) face à `Transform` :
/// `mul_dual`/`transform_point` sont censés rester du même ordre de grandeur
/// (même travail, juste réparti différemment) ; `sclerp` a un coût propre
/// (extraction de l'axe de vissage, `acos`/`sin`/`cos`) sans équivalent dans
/// `Transform` (qui n'interpole pas).
fn bench_dual_quaternion(c: &mut Criterion) {
    let ax = Quaternion::<Q16_16>::from_axis_angle(
        [
            Q16_16::try_from(0.267).unwrap(),
            Q16_16::try_from(0.535).unwrap(),
            Q16_16::try_from(0.802).unwrap(),
        ],
        Q16_16::try_from(0.9).unwrap(),
    );
    let bx = Quaternion::<Q16_16>::from_axis_angle(
        [
            Q16_16::try_from(0.408).unwrap(),
            Q16_16::try_from(0.408).unwrap(),
            Q16_16::try_from(0.816).unwrap(),
        ],
        Q16_16::try_from(1.6).unwrap(),
    );
    let dqx_a = DualQuaternion::from_rotation_translation(
        ax,
        [
            Q16_16::try_from(0.2).unwrap(),
            Q16_16::try_from(-0.4).unwrap(),
            Q16_16::try_from(0.6).unwrap(),
        ],
    );
    let dqx_b = DualQuaternion::from_rotation_translation(
        bx,
        [
            Q16_16::try_from(-0.5).unwrap(),
            Q16_16::try_from(0.3).unwrap(),
            Q16_16::try_from(0.1).unwrap(),
        ],
    );
    let af = Quaternion::<f32>::from_axis_angle([0.267, 0.535, 0.802], 0.9);
    let bf = Quaternion::<f32>::from_axis_angle([0.408, 0.408, 0.816], 1.6);
    let dqf_a = DualQuaternion::from_rotation_translation(af, [0.2, -0.4, 0.6]);
    let dqf_b = DualQuaternion::from_rotation_translation(bf, [-0.5, 0.3, 0.1]);

    let (fx, ff) = vectors(0xE6);

    let mut g = c.benchmark_group("dual_quaternion_mul");
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| black_box(dqx_a).mul_dual(black_box(dqx_b)))
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| black_box(dqf_a).mul_dual(black_box(dqf_b)))
    });
    g.finish();

    let mut g = c.benchmark_group("dual_quaternion_transform_point");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            let mut acc = [Q16_16::zero(); 3];
            for &v in black_box(&fx)
            {
                let r = dqx_a.transform_point(v);
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
                let r = dqf_a.transform_point(v);
                acc[0] += r[0];
                acc[1] += r[1];
                acc[2] += r[2];
            }
            acc
        })
    });
    g.finish();

    let mut g = c.benchmark_group("dual_quaternion_sclerp");
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |b| {
        b.iter(|| {
            DualQuaternion::sclerp(
                black_box(dqx_a),
                black_box(dqx_b),
                Q16_16::try_from(0.37).unwrap(),
            )
        })
    });
    g.bench_function(BenchmarkId::new("f32", "f32"), |b| {
        b.iter(|| DualQuaternion::sclerp(black_box(dqf_a), black_box(dqf_b), 0.37f32))
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_rotate,
    bench_from_axis_angle,
    bench_slerp,
    bench_from_rotation_matrix,
    bench_euler_roundtrip,
    bench_transform,
    bench_dual_quaternion
);
criterion_main!(benches);
