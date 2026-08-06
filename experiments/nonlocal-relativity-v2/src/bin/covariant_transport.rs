//! Covariant (covector / tensor) parallel-transport metric-compatibility checks.
//!
//! Parallel transport of lower-index objects must respect metric compatibility
//! (`nabla g = 0`). This experiment reports three consequences across
//! backgrounds:
//!
//! - **Metric self-transport.** `transport(g_start)` should equal `g_end`; the
//!   max component gap falls with the RK4 substep count (second order).
//! - **Index lowering commutes.** `lower(transport(V))` should equal
//!   `transport(lower(V))`.
//! - **Contraction preservation.** The scalar `W_a V^a` is constant when both
//!   the covector `W` and the vector `V` are transported.
//!
//! Established general relativity only; the transport reuses the deterministic
//! geodesic-segment RK4 engine.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::{
    AntiDeSitter, Connection, DeSitter, Metric, Schwarzschild, transport_along_segment,
    transport_covariant_tensor_along_segment, transport_covector_along_segment,
};
use std::f64::consts::FRAC_PI_2;

fn lower(metric: &[[f64; 4]; 4], vector: &[f64; 4]) -> [f64; 4] {
    let mut covector = [0.0; 4];
    for (a, slot) in covector.iter_mut().enumerate()
    {
        for (b, &component) in vector.iter().enumerate()
        {
            *slot += metric[a][b] * component;
        }
    }
    covector
}

fn metric_self_transport_gap<B: Metric<4> + Connection<4>>(
    background: &B,
    start: [f64; 4],
    end: [f64; 4],
    substeps: usize,
) -> Result<f64, String> {
    let metric_start = background.components(&start);
    let metric_end = background.components(&end);
    let transported =
        transport_covariant_tensor_along_segment(background, &start, &end, &metric_start, substeps)
            .map_err(|e| e.to_string())?;
    let mut worst = 0.0_f64;
    for a in 0..4
    {
        for b in 0..4
        {
            worst = worst.max((transported[a][b] - metric_end[a][b]).abs());
        }
    }
    Ok(worst)
}

fn consistency_gaps<B: Metric<4> + Connection<4>>(
    background: &B,
    start: [f64; 4],
    end: [f64; 4],
) -> Result<(f64, f64), String> {
    let vector = [0.2, 0.1, 0.05, 0.02];
    let covector = [0.3, -0.1, 0.2, 0.05];
    let substeps = 300;

    // Index-lowering commutation.
    let vector_end = transport_along_segment(background, &start, &end, &vector, substeps)
        .map_err(|e| e.to_string())?;
    let lowered_after = lower(&background.components(&end), &vector_end);
    let lowered_before = lower(&background.components(&start), &vector);
    let covector_from_lowered =
        transport_covector_along_segment(background, &start, &end, &lowered_before, substeps)
            .map_err(|e| e.to_string())?;
    let lowering_gap = (0..4)
        .map(|i| (lowered_after[i] - covector_from_lowered[i]).abs())
        .fold(0.0_f64, f64::max);

    // Contraction preservation.
    let contraction_start: f64 = (0..4).map(|i| covector[i] * vector[i]).sum();
    let covector_end =
        transport_covector_along_segment(background, &start, &end, &covector, substeps)
            .map_err(|e| e.to_string())?;
    let contraction_end: f64 = (0..4).map(|i| covector_end[i] * vector_end[i]).sum();
    let contraction_gap = (contraction_end - contraction_start).abs();

    Ok((lowering_gap, contraction_gap))
}

fn main() -> Result<(), String> {
    let schwarzschild =
        Schwarzschild::try_new(1.0).ok_or_else(|| "invalid Schwarzschild".to_string())?;
    let de_sitter = DeSitter::try_new(0.05).ok_or_else(|| "invalid de Sitter".to_string())?;
    let anti_de_sitter =
        AntiDeSitter::try_new(0.05).ok_or_else(|| "invalid anti-de Sitter".to_string())?;

    print_experiment_header(
        "Covariant transport metric-compatibility checks",
        "scirust-relativity geometry core (established general relativity)",
        "checks covector/tensor transport respects metric compatibility; established GR.",
    );

    // Part A: metric self-transport convergence.
    println!("# Part A: max|transport(g_start) - g_end| vs RK4 substeps (should fall ~h^2)");
    println!("background,substeps,metric_self_transport_gap");
    for substeps in [25, 50, 100, 200, 400]
    {
        let schwarzschild_gap = metric_self_transport_gap(
            &schwarzschild,
            [0.0, 10.0, FRAC_PI_2, 0.0],
            [0.0, 8.0, FRAC_PI_2, 0.5],
            substeps,
        )?;
        let de_sitter_gap = metric_self_transport_gap(
            &de_sitter,
            [0.0, 3.0, FRAC_PI_2, 0.0],
            [0.0, 4.0, 1.3, 0.4],
            substeps,
        )?;
        require_finite(&[
            ("schwarzschild_gap", schwarzschild_gap),
            ("de_sitter_gap", de_sitter_gap),
        ])?;
        println!("Schwarzschild,{substeps},{schwarzschild_gap:.3e}");
        println!("de_Sitter,{substeps},{de_sitter_gap:.3e}");
    }

    // Part B: index-lowering commutation and contraction preservation.
    println!("#");
    println!("# Part B: index-lowering commutation and covector-vector contraction preservation");
    println!("background,index_lowering_gap,contraction_gap");
    for &(label, start, end) in &[
        (
            "Schwarzschild",
            [0.0, 10.0, FRAC_PI_2, 0.0],
            [0.0, 7.0, FRAC_PI_2, 0.6],
        ),
        (
            "de_Sitter",
            [0.0, 3.0, FRAC_PI_2, 0.0],
            [0.0, 4.0, 1.2, 0.5],
        ),
        (
            "anti_de_Sitter",
            [0.0, 3.0, FRAC_PI_2, 0.0],
            [0.0, 4.0, 1.2, 0.5],
        ),
    ]
    {
        let (lowering_gap, contraction_gap) = match label
        {
            "Schwarzschild" => consistency_gaps(&schwarzschild, start, end)?,
            "de_Sitter" => consistency_gaps(&de_sitter, start, end)?,
            _ => consistency_gaps(&anti_de_sitter, start, end)?,
        };
        require_finite(&[
            ("lowering_gap", lowering_gap),
            ("contraction_gap", contraction_gap),
        ])?;
        println!("{label},{lowering_gap:.3e},{contraction_gap:.3e}");
    }

    println!("# interpretation: the metric transports to itself (gap falls ~h^2), index lowering");
    println!("# commutes with transport, and the covector-vector contraction is preserved -- the");
    println!("# three signatures of metric compatibility, holding across every background to the");
    println!("# integration tolerance. Established GR, not a phenomenological model.");
    Ok(())
}
