//! Reproducible dense quantum-gradient benchmark.
//!
//! Compares the exact dense adjoint Jacobian with the exact parameter-shift
//! oracle over a deterministic grid of:
//!
//! - qubit counts;
//! - symbolic parameter counts;
//! - ordered Pauli-observable counts.
//!
//! The two methods receive the same circuit, bindings, and observables.
//! Workload angles are generated from the reported seed by a local,
//! fully specified SplitMix64 stream. Every controlled sweep family retains
//! one fixed stream so smaller workloads are prefixes of larger workloads.
//! Timing order alternates sample by sample to distribute machine drift,
//! Tukey fences reject isolated timing spikes, and Welch's unequal-variance
//! t-test reports whether the observed timing difference exceeds noise.
//!
//! Run with:
//!
//! ```text
//! cargo run -p scirust-core --release --example quantum_gradient_benchmark
//! ```
//!
//! Human-readable results are followed by CANR §9 `BenchRecord` JSONL rows.

use scirust_bench_schema::{BenchRecord, ConfidenceInterval};
use scirust_core::quantum::{
    Circuit, Observable, Operation, Parameter, ParameterId, ParameterValues, Pauli, PauliTerm,
    QuantumResult, adjoint_jacobian, parameter_shift_gradients,
};
use scirust_stats::describe::{mean, median, quantile, std_error};
use scirust_stats::htest::{Tail, t_test_two_sample};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

const SEED: u64 = 0x5C1_2026_0719;
const MAX_GRADIENT_ERROR: f32 = 1.0e-4;
const WARMUP_SAMPLES: usize = 3;

type Jacobian = BTreeMap<ParameterId, Vec<f32>>;

/// Deterministic generator used only to construct reproducible workloads.
///
/// SplitMix64 has a fully specified integer transition. `next_signed_f32`
/// converts a 24-bit integer to `f32` through an exact power-of-two scale,
/// avoiding external RNGs and platform-specific distributions.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn next_signed_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as i32;
        let centered = bits - 8_388_608;
        centered as f32 / 8_388_608.0
    }
}

/// Returns one fixed stream per controlled sweep family.
///
/// Keeping the stream independent of workload dimensions makes the sweeps
/// nested: when parameter or observable counts increase, all previously
/// generated circuit angles remain unchanged. Qubit sweeps likewise retain
/// identical angle values while changing only state dimension and topology.
fn workload_seed(spec: Spec) -> u64 {
    let family_tag = match spec.family
    {
        "combined" => 0x434F_4D42_494E_4544,
        "qubits" => 0x5155_4249_5453_0001,
        "parameters" => 0x5041_5241_4D53_0002,
        "observables" => 0x4F42_5345_5256_0003,
        family => panic!("unknown benchmark family: {family}"),
    };

    SEED.wrapping_add(family_tag)
}

#[derive(Debug, Clone, Copy)]
struct Spec {
    family: &'static str,
    qubits: usize,
    parameters: usize,
    observables: usize,
    samples: usize,
    repeats: usize,
}

impl Spec {
    fn label(self) -> String {
        format!(
            "{}/q{}_p{}_o{}",
            self.family, self.qubits, self.parameters, self.observables
        )
    }
}

#[derive(Debug)]
struct Workload {
    spec: Spec,
    circuit: Circuit,
    values: ParameterValues,
    observables: Vec<Observable>,
}

#[derive(Debug)]
struct TimingSummary {
    raw_count: usize,
    clean_count: usize,
    dropped: usize,
    mean_ns: f64,
    median_ns: f64,
    ci95_lo_ns: f64,
    ci95_hi_ns: f64,
}

#[derive(Debug)]
struct Comparison {
    adjoint: TimingSummary,
    parameter_shift: TimingSummary,
    speedup: f64,
    max_abs_error: f32,
    welch_p_value: Option<f64>,
    checksum: f64,
}

fn pauli(index: usize) -> Pauli {
    match index % 3
    {
        0 => Pauli::X,
        1 => Pauli::Y,
        _ => Pauli::Z,
    }
}

fn build_workload(spec: Spec) -> QuantumResult<Workload> {
    let mut circuit = Circuit::new(spec.qubits)?;
    let mut rng = SplitMix64::new(workload_seed(spec));

    // Establish a non-trivial complex superposition before the trainable
    // section. Fixed phase gates are intentionally included because the
    // adjoint sweep must reverse them even though they are not differentiated.
    for qubit in 0..spec.qubits
    {
        circuit.push(Operation::H { target: qubit })?;
        if qubit % 2 == 0
        {
            circuit.push(Operation::S { target: qubit })?;
        }
        else
        {
            circuit.push(Operation::T { target: qubit })?;
        }
    }

    let mut values = ParameterValues::new();

    for index in 0..spec.parameters
    {
        let id = ParameterId(index as u32);
        let target = index % spec.qubits;
        let parameter = Parameter::Symbol(id);

        let rotation = match index % 3
        {
            0 => Operation::Rx { target, parameter },
            1 => Operation::Ry { target, parameter },
            _ => Operation::Rz { target, parameter },
        };
        circuit.push(rotation)?;

        // A deterministic non-zero offset produces both local and longer-range
        // entanglers as the parameter index advances.
        let offset = 1 + (index / spec.qubits) % (spec.qubits - 1);
        let other = (target + offset) % spec.qubits;
        if index % 2 == 0
        {
            circuit.push(Operation::Cnot {
                control: target,
                target: other,
            })?;
        }
        else
        {
            circuit.push(Operation::Cz {
                control: target,
                target: other,
            })?;
        }

        // Draw both values on every iteration so the seeded stream does not
        // depend on whether this operation receives a fixed phase gate.
        let fixed_phase = rng.next_signed_f32() * 0.35;
        let angle = rng.next_signed_f32() * 0.9;

        // Exercise inverse propagation through a fixed complex phase without
        // introducing an unsupported symbolic PhaseShift.
        if index % 4 == 3
        {
            circuit.push(Operation::PhaseShift {
                target: other,
                parameter: Parameter::Fixed(fixed_phase),
            })?;
        }

        values.insert(id, angle)?;
    }

    let mut observables = Vec::with_capacity(spec.observables);
    for index in 0..spec.observables
    {
        let first = index % spec.qubits;
        let offset = 1 + (index * 2) % (spec.qubits - 1);
        let second = (first + offset) % spec.qubits;

        observables.push(Observable::new(vec![
            PauliTerm::new(first, pauli(index)),
            PauliTerm::new(second, pauli(index + 1)),
        ])?);
    }

    Ok(Workload {
        spec,
        circuit,
        values,
        observables,
    })
}

fn parameter_shift_jacobian(workload: &Workload) -> QuantumResult<Jacobian> {
    let mut jacobian = BTreeMap::new();
    for parameter in workload.circuit.parameter_ids()
    {
        let gradients = parameter_shift_gradients(
            &workload.circuit,
            &workload.values,
            &workload.observables,
            parameter,
        )?;
        jacobian.insert(parameter, gradients);
    }
    Ok(jacobian)
}

fn jacobian_checksum(jacobian: &Jacobian) -> f64 {
    let mut checksum = 0.0f64;
    for (parameter, gradients) in jacobian
    {
        for (observable, &gradient) in gradients.iter().enumerate()
        {
            let weight = 1.0 + f64::from(parameter.0) * 0.03125 + observable as f64 * 0.0078125;
            checksum += f64::from(gradient) * weight;
        }
    }
    checksum
}

fn max_abs_error(left: &Jacobian, right: &Jacobian) -> f32 {
    assert_eq!(
        left.len(),
        right.len(),
        "the two Jacobians must expose the same parameters"
    );

    let mut maximum = 0.0f32;
    for (parameter, left_gradients) in left
    {
        let right_gradients = right
            .get(parameter)
            .expect("the two Jacobians must expose identical parameter IDs");
        assert_eq!(
            left_gradients.len(),
            right_gradients.len(),
            "the two Jacobians must expose the same observables"
        );
        for (&left_gradient, &right_gradient) in left_gradients.iter().zip(right_gradients)
        {
            maximum = maximum.max((left_gradient - right_gradient).abs());
        }
    }
    maximum
}

fn shared_parameter_values_match(left: &Workload, right: &Workload) -> bool {
    left.values
        .iter()
        .all(|(parameter, value)| right.values.get(parameter) == Some(value))
}

/// Verifies that each controlled sweep changes only its declared axis.
///
/// - parameter sweep: smaller circuits are exact operation prefixes and retain
///   identical observables and shared parameter values;
/// - observable sweep: circuits and values are identical, while smaller
///   observable lists are exact prefixes;
/// - qubit sweep: every shared symbolic parameter retains the same angle.
fn validate_controlled_sweeps(specs: &[Spec]) -> QuantumResult<()> {
    let workloads = specs
        .iter()
        .copied()
        .map(build_workload)
        .collect::<QuantumResult<Vec<_>>>()?;

    let parameters = workloads
        .iter()
        .filter(|workload| workload.spec.family == "parameters")
        .collect::<Vec<_>>();
    assert_eq!(parameters.len(), 4);

    for pair in parameters.windows(2)
    {
        let smaller = pair[0];
        let larger = pair[1];

        assert!(smaller.spec.parameters < larger.spec.parameters);
        assert_eq!(smaller.spec.qubits, larger.spec.qubits);
        assert_eq!(smaller.spec.observables, larger.spec.observables);

        assert!(
            larger
                .circuit
                .operations()
                .starts_with(smaller.circuit.operations()),
            "parameter sweep circuits must be nested operation prefixes"
        );

        assert_eq!(
            smaller.observables, larger.observables,
            "parameter sweep observables must remain identical"
        );

        assert!(
            shared_parameter_values_match(smaller, larger),
            "parameter sweep must retain every shared parameter angle"
        );
    }

    let observables = workloads
        .iter()
        .filter(|workload| workload.spec.family == "observables")
        .collect::<Vec<_>>();
    assert_eq!(observables.len(), 4);

    for pair in observables.windows(2)
    {
        let smaller = pair[0];
        let larger = pair[1];

        assert!(smaller.spec.observables < larger.spec.observables);

        assert_eq!(
            smaller.circuit, larger.circuit,
            "observable sweep circuit must remain identical"
        );

        assert_eq!(
            smaller.values, larger.values,
            "observable sweep parameter values must remain identical"
        );

        assert!(
            larger.observables.starts_with(&smaller.observables),
            "observable sweep lists must be nested prefixes"
        );
    }

    let qubits = workloads
        .iter()
        .filter(|workload| workload.spec.family == "qubits")
        .collect::<Vec<_>>();
    assert_eq!(qubits.len(), 4);

    for pair in qubits.windows(2)
    {
        let smaller = pair[0];
        let larger = pair[1];

        assert!(smaller.spec.qubits < larger.spec.qubits);
        assert_eq!(smaller.spec.parameters, larger.spec.parameters);
        assert_eq!(smaller.spec.observables, larger.spec.observables);

        assert!(
            shared_parameter_values_match(smaller, larger),
            "qubit sweep must retain every shared parameter angle"
        );
    }

    Ok(())
}

fn time_repeated<F>(repeats: usize, mut run: F) -> (f64, f64)
where
    F: FnMut() -> Jacobian,
{
    let start = Instant::now();
    let mut checksum = 0.0f64;

    for _ in 0..repeats
    {
        let jacobian = black_box(run());
        checksum += jacobian_checksum(black_box(&jacobian));
    }

    let elapsed_ns = start.elapsed().as_secs_f64() * 1.0e9;
    (elapsed_ns / repeats as f64, checksum)
}

fn reject_outliers(mut samples: Vec<f64>) -> Vec<f64> {
    if samples.len() < 4
    {
        return samples;
    }

    let q1 = quantile(&samples, 0.25);
    let q3 = quantile(&samples, 0.75);
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;

    samples.retain(|&sample| sample >= lower && sample <= upper);
    samples
}

fn summarize(raw: Vec<f64>) -> TimingSummary {
    let raw_count = raw.len();
    let clean = reject_outliers(raw);
    assert!(
        clean.len() >= 2,
        "at least two clean timing samples are required"
    );

    let clean_mean = mean(&clean);
    let margin = 1.96 * std_error(&clean);

    TimingSummary {
        raw_count,
        clean_count: clean.len(),
        dropped: raw_count - clean.len(),
        mean_ns: clean_mean,
        median_ns: median(&clean),
        ci95_lo_ns: clean_mean - margin,
        ci95_hi_ns: clean_mean + margin,
    }
}

fn compare(workload: &Workload) -> Comparison {
    let adjoint_reference =
        adjoint_jacobian(&workload.circuit, &workload.values, &workload.observables)
            .expect("adjoint Jacobian must succeed");
    let shift_reference =
        parameter_shift_jacobian(workload).expect("parameter-shift Jacobian must succeed");

    let error = max_abs_error(&adjoint_reference, &shift_reference);
    assert!(
        error <= MAX_GRADIENT_ERROR,
        "adjoint/parameter-shift disagreement {error:.9e} exceeds {MAX_GRADIENT_ERROR:.1e}"
    );

    // Warm both paths before collecting measurements.
    for _ in 0..WARMUP_SAMPLES
    {
        black_box(
            adjoint_jacobian(&workload.circuit, &workload.values, &workload.observables)
                .expect("adjoint warmup must succeed"),
        );
        black_box(parameter_shift_jacobian(workload).expect("parameter-shift warmup must succeed"));
    }

    let mut adjoint_samples = Vec::with_capacity(workload.spec.samples);
    let mut shift_samples = Vec::with_capacity(workload.spec.samples);
    let mut checksum = 0.0f64;

    for sample_index in 0..workload.spec.samples
    {
        if sample_index % 2 == 0
        {
            let (adjoint_ns, adjoint_checksum) = time_repeated(workload.spec.repeats, || {
                adjoint_jacobian(&workload.circuit, &workload.values, &workload.observables)
                    .expect("adjoint timing run must succeed")
            });
            let (shift_ns, shift_checksum) = time_repeated(workload.spec.repeats, || {
                parameter_shift_jacobian(workload).expect("parameter-shift timing run must succeed")
            });
            adjoint_samples.push(adjoint_ns);
            shift_samples.push(shift_ns);
            checksum += adjoint_checksum + shift_checksum;
        }
        else
        {
            let (shift_ns, shift_checksum) = time_repeated(workload.spec.repeats, || {
                parameter_shift_jacobian(workload).expect("parameter-shift timing run must succeed")
            });
            let (adjoint_ns, adjoint_checksum) = time_repeated(workload.spec.repeats, || {
                adjoint_jacobian(&workload.circuit, &workload.values, &workload.observables)
                    .expect("adjoint timing run must succeed")
            });
            shift_samples.push(shift_ns);
            adjoint_samples.push(adjoint_ns);
            checksum += shift_checksum + adjoint_checksum;
        }
    }

    assert!(
        checksum.is_finite(),
        "benchmark checksum must remain finite"
    );

    let adjoint_clean = reject_outliers(adjoint_samples.clone());
    let shift_clean = reject_outliers(shift_samples.clone());
    let welch_p_value = t_test_two_sample(&adjoint_clean, &shift_clean, false, Tail::TwoSided)
        .map(|result| result.p_value);

    let adjoint = summarize(adjoint_samples);
    let parameter_shift = summarize(shift_samples);
    let speedup = parameter_shift.median_ns / adjoint.median_ns;

    Comparison {
        adjoint,
        parameter_shift,
        speedup,
        max_abs_error: error,
        welch_p_value,
        checksum,
    }
}

fn push_timing_records(
    records: &mut Vec<BenchRecord>,
    dataset: &str,
    method: &str,
    summary: &TimingSummary,
) {
    records.push(
        BenchRecord::new(
            "quantum_dense_gradient_jacobian",
            dataset,
            method,
            SEED,
            "mean_wall_time_ns",
            summary.mean_ns,
        )
        .with_ci(ConfidenceInterval {
            lo: summary.ci95_lo_ns,
            hi: summary.ci95_hi_ns,
            level: 0.95,
        }),
    );
    records.push(BenchRecord::new(
        "quantum_dense_gradient_jacobian",
        dataset,
        method,
        SEED,
        "median_wall_time_ns",
        summary.median_ns,
    ));
}

fn main() -> QuantumResult<()> {
    let specs = [
        // End-to-end scaling: all three dimensions increase together.
        Spec {
            family: "combined",
            qubits: 4,
            parameters: 4,
            observables: 1,
            samples: 40,
            repeats: 50,
        },
        Spec {
            family: "combined",
            qubits: 6,
            parameters: 8,
            observables: 2,
            samples: 40,
            repeats: 10,
        },
        Spec {
            family: "combined",
            qubits: 8,
            parameters: 16,
            observables: 4,
            samples: 30,
            repeats: 3,
        },
        Spec {
            family: "combined",
            qubits: 10,
            parameters: 32,
            observables: 8,
            samples: 20,
            repeats: 1,
        },
        // Controlled qubit sweep: parameters=8, observables=2.
        Spec {
            family: "qubits",
            qubits: 4,
            parameters: 8,
            observables: 2,
            samples: 40,
            repeats: 20,
        },
        Spec {
            family: "qubits",
            qubits: 6,
            parameters: 8,
            observables: 2,
            samples: 40,
            repeats: 10,
        },
        Spec {
            family: "qubits",
            qubits: 8,
            parameters: 8,
            observables: 2,
            samples: 30,
            repeats: 5,
        },
        Spec {
            family: "qubits",
            qubits: 10,
            parameters: 8,
            observables: 2,
            samples: 25,
            repeats: 2,
        },
        // Controlled parameter sweep: qubits=8, observables=2.
        Spec {
            family: "parameters",
            qubits: 8,
            parameters: 4,
            observables: 2,
            samples: 30,
            repeats: 10,
        },
        Spec {
            family: "parameters",
            qubits: 8,
            parameters: 8,
            observables: 2,
            samples: 30,
            repeats: 5,
        },
        Spec {
            family: "parameters",
            qubits: 8,
            parameters: 16,
            observables: 2,
            samples: 30,
            repeats: 3,
        },
        Spec {
            family: "parameters",
            qubits: 8,
            parameters: 32,
            observables: 2,
            samples: 25,
            repeats: 2,
        },
        // Controlled observable sweep: qubits=8, parameters=16.
        Spec {
            family: "observables",
            qubits: 8,
            parameters: 16,
            observables: 1,
            samples: 30,
            repeats: 3,
        },
        Spec {
            family: "observables",
            qubits: 8,
            parameters: 16,
            observables: 2,
            samples: 30,
            repeats: 3,
        },
        Spec {
            family: "observables",
            qubits: 8,
            parameters: 16,
            observables: 4,
            samples: 30,
            repeats: 3,
        },
        Spec {
            family: "observables",
            qubits: 8,
            parameters: 16,
            observables: 8,
            samples: 25,
            repeats: 2,
        },
    ];

    validate_controlled_sweeps(&specs)?;

    println!("SciRust exact dense quantum-gradient benchmark");
    println!("seed={SEED} warmup={WARMUP_SAMPLES}");
    println!("generator: SplitMix64 with exact signed 24-bit f32 conversion");
    println!("methods: one adjoint Jacobian vs one parameter-shift sweep per symbolic parameter");
    println!("workloads: combined scaling plus controlled qubit, parameter, and observable sweeps");
    println!("timing: alternating order, Tukey outlier rejection, Welch unequal-variance test");
    println!("controlled sweep nesting: validated\n");

    let mut records = Vec::new();

    for spec in specs
    {
        let workload = build_workload(spec)?;
        let comparison = compare(&workload);
        let label = spec.label();
        let operations = workload.circuit.operations().len();

        let significance = match comparison.welch_p_value
        {
            Some(p) if p < 0.05 => "significant",
            Some(_) => "within noise",
            None => "unavailable",
        };

        println!(
            "{label:>12} ops={operations:>3} samples={:>2} repeats={:>2}",
            spec.samples, spec.repeats
        );
        println!(
            "  adjoint        {:>12.3} µs median  {:>12.3} µs mean  n={}/{} drop={}",
            comparison.adjoint.median_ns / 1.0e3,
            comparison.adjoint.mean_ns / 1.0e3,
            comparison.adjoint.clean_count,
            comparison.adjoint.raw_count,
            comparison.adjoint.dropped,
        );
        println!(
            "  parameter-shift{:>12.3} µs median  {:>12.3} µs mean  n={}/{} drop={}",
            comparison.parameter_shift.median_ns / 1.0e3,
            comparison.parameter_shift.mean_ns / 1.0e3,
            comparison.parameter_shift.clean_count,
            comparison.parameter_shift.raw_count,
            comparison.parameter_shift.dropped,
        );
        println!(
            "  speedup={:>9.3}x  max|Δgradient|={:.9e}  Welch p={} ({significance})  checksum={:.9e}\n",
            comparison.speedup,
            comparison.max_abs_error,
            comparison
                .welch_p_value
                .map_or_else(|| "n/a".to_owned(), |p| format!("{p:.4e}")),
            comparison.checksum,
        );

        let dataset = format!(
            "{label}/ops={operations}/samples={}/repeats={}",
            spec.samples, spec.repeats
        );

        push_timing_records(&mut records, &dataset, "adjoint", &comparison.adjoint);
        push_timing_records(
            &mut records,
            &dataset,
            "parameter_shift",
            &comparison.parameter_shift,
        );
        records.push(BenchRecord::new(
            "quantum_dense_gradient_jacobian",
            &dataset,
            "adjoint_vs_parameter_shift",
            SEED,
            "median_speedup_ratio",
            comparison.speedup,
        ));
        records.push(BenchRecord::new(
            "quantum_dense_gradient_jacobian",
            &dataset,
            "adjoint_vs_parameter_shift",
            SEED,
            "max_abs_gradient_error",
            f64::from(comparison.max_abs_error),
        ));
        if let Some(p_value) = comparison.welch_p_value
        {
            records.push(BenchRecord::new(
                "quantum_dense_gradient_jacobian",
                &dataset,
                "adjoint_vs_parameter_shift",
                SEED,
                "welch_two_sided_p_value",
                p_value,
            ));
        }
    }

    println!(
        "=== bench-schema JSONL ({} records, scirust-bench-schema) ===",
        records.len()
    );
    print!("{}", scirust_bench_schema::to_jsonl(&records));

    Ok(())
}
