//! Wall-clock micro-benchmarks for the ADM 3+1 decomposition.
//!
//! These measure elapsed time and are therefore machine-dependent and not
//! bit-reproducible — that is inherent to timing; the decomposition itself is
//! fully deterministic. They cover a full decomposition (lapse, shift, spatial
//! metric, extrinsic curvature, and the spatial Ricci scalar) and the
//! Gauss-Codazzi constraints (including the momentum divergence, the heaviest
//! quantity). Run with `cargo bench -p scirust-relativity --bench adm`.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use scirust_relativity::adm::{AdmSettings, adm_constraints, adm_decomposition};
use scirust_relativity::{ExponentialScaleFactor, Flrw, PainleveGullstrand};
use std::f64::consts::FRAC_PI_2;

fn settings() -> AdmSettings {
    AdmSettings {
        time_step: 1.0e-3,
        spatial_step: 1.0e-3,
    }
}

fn adm(criterion: &mut Criterion) {
    let flrw = Flrw::new(ExponentialScaleFactor::try_new(0.5).expect("valid H"));
    let painleve = PainleveGullstrand::try_new(1.0).expect("valid mass");
    let settings = settings();

    // A full decomposition (time-dependent slice, so a non-zero extrinsic curvature).
    criterion.bench_function("adm_decomposition_flrw", |bencher| {
        bencher.iter(|| {
            black_box(
                adm_decomposition(&flrw, black_box(&[0.0, 0.1, 0.2, 0.3]), &settings)
                    .expect("decomposition"),
            )
        });
    });

    // The constraints, including the momentum divergence (the heaviest quantity),
    // on the non-zero-shift Painlevé–Gullstrand slicing.
    criterion.bench_function("adm_constraints_painleve", |bencher| {
        bencher.iter(|| {
            black_box(
                adm_constraints(
                    &painleve,
                    black_box(&[0.0, 4.0, FRAC_PI_2, 0.0]),
                    0.0,
                    &settings,
                )
                .expect("constraints"),
            )
        });
    });
}

criterion_group!(benches, adm);
criterion_main!(benches);
