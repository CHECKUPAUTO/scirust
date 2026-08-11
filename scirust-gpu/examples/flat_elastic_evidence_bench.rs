#[cfg(not(feature = "flat-autotune"))]
fn main() {
    eprintln!("flat_elastic_evidence_bench requires --features flat-autotune");
}

#[cfg(feature = "flat-autotune")]
fn main() {
    bench::run();
}

#[cfg(feature = "flat-autotune")]
mod bench {
    use std::collections::BTreeSet;
    use std::env;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use elastic_autotuner::measurement_protocol::{
        ElasticMeasurementProtocol, ElasticResidenceMode, ElasticSynchronizationBoundary,
        ElasticTimingSource,
    };
    use elastic_autotuner::persistence::ElasticPersistedPlan;
    use flat_attention::{
        AsymmetricGroupedAttentionShape, AsymmetricRotaryEmbeddingConfig, FlatAttentionConfig,
        forward_reference_projection_grouped_rope_asymmetric,
    };
    use scirust_gpu::flat_autotune::{
        ElasticAutoTuner, ElasticCandidate, ElasticConfig, ElasticEvidence, ElasticMode,
        ElasticObjective, FlatElasticPlanner, FlatElasticRequest, FlatKvRepresentation,
    };
    use scirust_gpu::{BackendResult, FlatM11ResidentConfig, GpuMatrix, WgpuFlatM11Bridge};

    const M11_FAMILY: &str = "flat-m11-external-asymmetric-projection";
    const M15_FAMILY: &str = "flat-m15-resident-decode";
    const BENCH_SCHEMA: &[u8] = b"scirust-flat-elastic-m11-m15-bench-v1";
    const FLAT_REVISION: &[u8] = b"flat-attention@24d3340edeb059e40e0fe0c400e814685701d855";
    const DEFAULT_Q_HEADS: usize = 8;
    const DEFAULT_KV_HEADS: usize = 2;
    const DEFAULT_HEAD_DIM: usize = 64;
    const DEFAULT_WARMUPS: usize = 5;
    const DEFAULT_REPEATS: usize = 21;
    const THETA: f32 = 10_000.0;
    const ATOL: f32 = 1.5e-4;
    const RTOL: f32 = 1.0e-3;

    pub fn run() {
        let q_heads = env_usize("SCIRUST_FLAT_ELASTIC_BENCH_Q_HEADS", DEFAULT_Q_HEADS);
        let kv_heads = env_usize("SCIRUST_FLAT_ELASTIC_BENCH_KV_HEADS", DEFAULT_KV_HEADS);
        let head_dim = env_usize("SCIRUST_FLAT_ELASTIC_BENCH_HEAD_DIM", DEFAULT_HEAD_DIM);
        let warmups = env_usize("SCIRUST_FLAT_ELASTIC_BENCH_WARMUPS", DEFAULT_WARMUPS);
        let repeats = env_usize("SCIRUST_FLAT_ELASTIC_BENCH_REPEATS", DEFAULT_REPEATS);
        let kv_lens = env_usize_list(
            "SCIRUST_FLAT_ELASTIC_BENCH_KV_LENS",
            &[1, 2, 5, 17, 65, 257, 1025, 4097],
        );
        let recorded_unix_ns = env_u64("SCIRUST_FLAT_ELASTIC_RECORDED_UNIX_NS", 0);
        let record_dir = env::var_os("SCIRUST_FLAT_ELASTIC_RECORD_DIR").map(PathBuf::from);
        let source_revision = env::var("SCIRUST_FLAT_ELASTIC_SOURCE_REVISION").ok();

        assert!(q_heads > 0, "q_heads must be non-zero");
        assert!(kv_heads > 0, "kv_heads must be non-zero");
        assert!(q_heads.is_multiple_of(kv_heads));
        assert!(head_dim > 0 && head_dim.is_multiple_of(2));
        assert!(repeats > 0, "measured iteration count must be non-zero");
        assert!(!kv_lens.is_empty(), "at least one kv_len is required");
        assert!(kv_lens.iter().all(|&len| len > 0));
        assert_flat_revision_matches_manifest();

        let warmup_iterations = u32::try_from(warmups).expect("warmup count exceeds u32");
        let measured_iterations = u32::try_from(repeats).expect("repeat count exceeds u32");
        let protocol = ElasticMeasurementProtocol::new(
            warmup_iterations,
            measured_iterations,
            ElasticTimingSource::HostWallClock,
            ElasticResidenceMode::Resident,
            ElasticSynchronizationBoundary::PerIteration,
        );
        protocol
            .validate()
            .expect("measurement protocol must be valid");

        let bridge = WgpuFlatM11Bridge::new().unwrap_or_else(|error| {
            panic!("WGPU adapter required for FLAT Elastic evidence benchmark: {error}")
        });
        assert!(
            bridge.m15_available(),
            "FLAT M15 pipeline must compile before evidence collection"
        );

        if let Some(dir) = &record_dir
        {
            assert!(
                recorded_unix_ns > 0,
                "SCIRUST_FLAT_ELASTIC_RECORDED_UNIX_NS must be non-zero when records are written"
            );
            assert!(
                source_revision
                    .as_deref()
                    .is_some_and(|revision| !revision.trim().is_empty()),
                "SCIRUST_FLAT_ELASTIC_SOURCE_REVISION is required when records are written"
            );
            fs::create_dir_all(dir).expect("create Elastic record output directory");
        }

        eprintln!("adapter={}", bridge.adapter_name());
        eprintln!("timing=host-wall-clock,resident,per-iteration-device-poll");
        eprintln!("correctness=scalar-oracle-before-timing");
        println!(
            "adapter,kv_len,candidate,samples,median_ns,p95_ns,p99_ns,mad_ns,max_abs,max_rel,selected"
        );

        let mut recorded_problem_classes = BTreeSet::new();
        for kv_len in kv_lens
        {
            let recorded_problem_classes =
                record_dir.as_ref().map(|_| &mut recorded_problem_classes);
            run_case(
                &bridge,
                q_heads,
                kv_heads,
                head_dim,
                kv_len,
                warmups,
                repeats,
                protocol,
                recorded_unix_ns,
                source_revision.as_deref(),
                record_dir.as_deref(),
                recorded_problem_classes,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_case(
        bridge: &WgpuFlatM11Bridge,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        warmups: usize,
        repeats: usize,
        protocol: ElasticMeasurementProtocol,
        recorded_unix_ns: u64,
        source_revision: Option<&str>,
        record_dir: Option<&Path>,
        recorded_problem_classes: Option<&mut BTreeSet<Vec<u8>>>,
    ) {
        let q_width = q_heads.checked_mul(head_dim).expect("Q width overflow");
        let kv_width = kv_heads.checked_mul(head_dim).expect("KV width overflow");
        let position = kv_len - 1;
        let config = FlatM11ResidentConfig {
            batch: 1,
            q_heads,
            kv_heads,
            query_len: 1,
            kv_len,
            head_dim,
            causal: true,
            softmax_scale: None,
            query_position_offset: position,
            theta: THETA,
            query_rope_position_offset: position,
            kv_rope_position_offset: 0,
        };
        let request = FlatElasticRequest::new(config, FlatKvRepresentation::PreRotated)
            .expect("decode request must satisfy FLAT semantics");
        let planner = FlatElasticPlanner::new(request).expect("build FLAT Elastic planner");
        if let Some(seen) = recorded_problem_classes
        {
            let class_key = planner.problem_class().class_key().to_vec();
            assert!(
                seen.insert(class_key),
                "duplicate H2 problem class for kv_len={kv_len}; write at most one record per validity region"
            );
        }
        let hardware = planner
            .hardware_profile(bridge.context())
            .expect("encode WGPU hardware profile");
        let tuner = ElasticAutoTuner::new(ElasticConfig {
            mode: ElasticMode::Learn,
            objective: ElasticObjective::MinLatency,
            max_ranked_candidates: 0,
        });
        let ranked = planner.rank_candidates(&tuner, &hardware);
        let m11 = candidate_named(&ranked, M11_FAMILY);
        let m15 = candidate_named(&ranked, M15_FAMILY);
        assert_eq!(
            ranked.len(),
            2,
            "fully-visible pre-rotated decode must expose exactly M11 and M15"
        );

        let q = fixture(q_width, 0.25);
        let raw_k = fixture(
            kv_len.checked_mul(kv_width).expect("K length overflow"),
            0.85,
        );
        let v = fixture(
            kv_len.checked_mul(kv_width).expect("V length overflow"),
            1.45,
        );
        let rotated_k = rotate_k_projection(&raw_k, kv_len, kv_heads, head_dim, THETA, 0);
        let q_gpu = bridge.context().upload(&q, 1, q_width);
        let k_gpu = bridge.context().upload(&rotated_k, kv_len, kv_width);
        let v_gpu = bridge.context().upload(&v, kv_len, kv_width);

        let shape = AsymmetricGroupedAttentionShape {
            batch: 1,
            q_heads,
            kv_heads,
            query_len: 1,
            kv_len,
            head_dim,
            query_position_offset: position,
        };
        let expected = forward_reference_projection_grouped_rope_asymmetric(
            &q,
            &raw_k,
            &v,
            shape,
            FlatAttentionConfig {
                causal: true,
                softmax_scale: None,
            },
            AsymmetricRotaryEmbeddingConfig {
                theta: THETA,
                query_position_offset: position,
                kv_position_offset: 0,
            },
        )
        .expect("scalar FLAT oracle must accept benchmark fixture");

        let (m11_abs, m11_rel) = qualify_candidate(
            bridge,
            &m11,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            config,
            &expected.output,
        );
        let (m15_abs, m15_rel) = qualify_candidate(
            bridge,
            &m15,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            config,
            &expected.output,
        );

        for warmup in 0..warmups
        {
            if warmup.is_multiple_of(2)
            {
                warmup_candidate(bridge, &m11, &q_gpu, &k_gpu, &v_gpu, config);
                warmup_candidate(bridge, &m15, &q_gpu, &k_gpu, &v_gpu, config);
            }
            else
            {
                warmup_candidate(bridge, &m15, &q_gpu, &k_gpu, &v_gpu, config);
                warmup_candidate(bridge, &m11, &q_gpu, &k_gpu, &v_gpu, config);
            }
        }

        let mut m11_samples = Vec::with_capacity(repeats);
        let mut m15_samples = Vec::with_capacity(repeats);
        for iteration in 0..repeats
        {
            if iteration.is_multiple_of(2)
            {
                m11_samples.push(measure_candidate(
                    bridge, &m11, &q_gpu, &k_gpu, &v_gpu, config,
                ));
                m15_samples.push(measure_candidate(
                    bridge, &m15, &q_gpu, &k_gpu, &v_gpu, config,
                ));
            }
            else
            {
                m15_samples.push(measure_candidate(
                    bridge, &m15, &q_gpu, &k_gpu, &v_gpu, config,
                ));
                m11_samples.push(measure_candidate(
                    bridge, &m11, &q_gpu, &k_gpu, &v_gpu, config,
                ));
            }
        }

        let mut scratch = vec![0_u64; repeats];
        let m11_measurement = protocol
            .summarize(&m11_samples, &mut scratch)
            .expect("summarize M11 samples");
        let m15_measurement = protocol
            .summarize(&m15_samples, &mut scratch)
            .expect("summarize M15 samples");
        let m11_evidence = ElasticEvidence::validated(
            m11.clone(),
            correctness_bytes(
                planner.problem_class().class_key(),
                &m11,
                protocol,
                expected.output.len(),
                m11_abs,
                m11_rel,
            ),
            m11_measurement,
        )
        .expect("M11 Elastic evidence must validate");
        let m15_evidence = ElasticEvidence::validated(
            m15.clone(),
            correctness_bytes(
                planner.problem_class().class_key(),
                &m15,
                protocol,
                expected.output.len(),
                m15_abs,
                m15_rel,
            ),
            m15_measurement,
        )
        .expect("M15 Elastic evidence must validate");
        let evidence = [m11_evidence, m15_evidence];
        let selected = planner
            .evaluate_measured_evidence(&tuner, hardware.clone(), &evidence)
            .expect("Elastic measured selection must succeed");
        let selected_family = selected.evidence.candidate.kernel_family.as_str();

        print_evidence(
            bridge.adapter_name(),
            kv_len,
            &evidence[0],
            m11_abs,
            m11_rel,
            selected_family,
        );
        print_evidence(
            bridge.adapter_name(),
            kv_len,
            &evidence[1],
            m15_abs,
            m15_rel,
            selected_family,
        );

        if let Some(dir) = record_dir
        {
            let source_revision =
                source_revision.expect("source revision validated before benchmark");
            let provenance = format!(
                "scirust-flat-elastic-evidence-v1;source={source_revision};adapter={};kv_len={kv_len};q_heads={q_heads};kv_heads={kv_heads};head_dim={head_dim}",
                bridge.adapter_name()
            )
            .into_bytes();
            let source_dependency = format!("scirust@{source_revision}").into_bytes();
            let record = ElasticPersistedPlan::new(
                selected,
                protocol,
                true,
                recorded_unix_ns,
                provenance,
                vec![
                    BENCH_SCHEMA.to_vec(),
                    FLAT_REVISION.to_vec(),
                    source_dependency,
                ],
            )
            .expect("build canonical selected-plan record");
            let encoded = record
                .encode()
                .expect("encode canonical selected-plan record");
            let path = dir.join(format!("flat-elastic-kv-{kv_len}.elauto"));
            fs::write(&path, encoded).expect("write selected-plan record");
            eprintln!("record={}", path.display());
        }
    }

    fn assert_flat_revision_matches_manifest() {
        let revision = FLAT_REVISION
            .strip_prefix(b"flat-attention@")
            .expect("benchmark FLAT revision must use flat-attention@<sha>");
        let revision = core::str::from_utf8(revision).expect("FLAT revision SHA must be UTF-8");
        let dependency = include_str!("../Cargo.toml")
            .lines()
            .find(|line| line.trim_start().starts_with("flat-attention = {"))
            .expect("scirust-gpu manifest must declare the FLAT dependency");
        let expected = format!("rev = \"{revision}\"");
        assert!(
            dependency.contains(&expected),
            "benchmark FLAT revision {revision} does not match Cargo pin: {dependency}"
        );
    }

    fn candidate_named(
        ranked: &[scirust_gpu::flat_autotune::RankedCandidate],
        family: &str,
    ) -> ElasticCandidate {
        ranked
            .iter()
            .find(|entry| entry.candidate.kernel_family == family)
            .unwrap_or_else(|| panic!("missing qualified candidate {family}"))
            .candidate
            .clone()
    }

    fn qualify_candidate(
        bridge: &WgpuFlatM11Bridge,
        candidate: &ElasticCandidate,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        config: FlatM11ResidentConfig,
        expected: &[f32],
    ) -> (f32, f32) {
        let output = execute_candidate(bridge, candidate, q, k, v, config)
            .expect("candidate correctness dispatch");
        let actual = bridge
            .context()
            .download(&output)
            .expect("candidate correctness readback");
        assert_close(&actual, expected)
    }

    fn warmup_candidate(
        bridge: &WgpuFlatM11Bridge,
        candidate: &ElasticCandidate,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        config: FlatM11ResidentConfig,
    ) {
        let _output = execute_candidate(bridge, candidate, q, k, v, config)
            .expect("candidate warmup dispatch");
        let _ = bridge.context().device().poll(wgpu::Maintain::Wait);
    }

    fn measure_candidate(
        bridge: &WgpuFlatM11Bridge,
        candidate: &ElasticCandidate,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        config: FlatM11ResidentConfig,
    ) -> u64 {
        let start = Instant::now();
        let _output = execute_candidate(bridge, candidate, q, k, v, config)
            .expect("candidate measured dispatch");
        let _ = bridge.context().device().poll(wgpu::Maintain::Wait);
        u64::try_from(start.elapsed().as_nanos()).expect("timing sample exceeds u64 nanoseconds")
    }

    fn execute_candidate(
        bridge: &WgpuFlatM11Bridge,
        candidate: &ElasticCandidate,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        config: FlatM11ResidentConfig,
    ) -> BackendResult<GpuMatrix> {
        match candidate.kernel_family.as_str()
        {
            M11_FAMILY => bridge.forward_pre_rotated_k(q, k, v, config),
            M15_FAMILY => bridge.forward_pre_rotated_k_m15(q, k, v, config),
            family => Err(scirust_gpu::BackendError::Execution(format!(
                "unsupported FLAT Elastic benchmark candidate {family}"
            ))),
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32]) -> (f32, f32) {
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate()
        {
            let error = (actual - expected).abs();
            let relative = error / expected.abs().max(f32::MIN_POSITIVE);
            max_abs = max_abs.max(error);
            max_rel = max_rel.max(relative);
            let tolerance = ATOL + RTOL * expected.abs();
            assert!(
                error <= tolerance,
                "parity index {index}: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
            );
        }
        (max_abs, max_rel)
    }

    fn correctness_bytes(
        problem_key: &[u8],
        candidate: &ElasticCandidate,
        protocol: ElasticMeasurementProtocol,
        elements: usize,
        max_abs: f32,
        max_rel: f32,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"FLATCOR1");
        push_bytes(&mut out, problem_key);
        push_bytes(&mut out, candidate.kernel_family.as_bytes());
        push_bytes(&mut out, &candidate.kernel_revision);
        out.extend_from_slice(
            &u64::try_from(elements)
                .expect("correctness element count exceeds u64")
                .to_le_bytes(),
        );
        out.extend_from_slice(&max_abs.to_bits().to_le_bytes());
        out.extend_from_slice(&max_rel.to_bits().to_le_bytes());
        out.extend_from_slice(
            &protocol
                .canonical_bytes()
                .expect("encode measurement protocol identity"),
        );
        out
    }

    fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
        let len = u32::try_from(bytes.len()).expect("correctness field exceeds u32 length");
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(bytes);
    }

    fn print_evidence(
        adapter: &str,
        kv_len: usize,
        evidence: &ElasticEvidence,
        max_abs: f32,
        max_rel: f32,
        selected_family: &str,
    ) {
        let measurement = evidence.measurement;
        println!(
            "{},{kv_len},{},{},{},{},{},{},{max_abs:.9},{max_rel:.9},{}",
            csv_field(adapter),
            evidence.candidate.kernel_family,
            measurement.sample_count,
            measurement.median_ns,
            measurement.p95_ns,
            measurement.p99_ns,
            measurement.mad_ns,
            evidence.candidate.kernel_family == selected_family,
        );
    }

    fn fixture(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let x = index as f32 * 0.023 + phase;
                x.sin() * 1.875 + (x * 0.41).cos() * 0.28125
            })
            .collect()
    }

    fn rotate_k_projection(
        raw: &[f32],
        kv_len: usize,
        kv_heads: usize,
        head_dim: usize,
        theta: f32,
        position_offset: usize,
    ) -> Vec<f32> {
        let mut rotated = raw.to_vec();
        let width = kv_heads * head_dim;
        for position in 0..kv_len
        {
            let absolute_position = position_offset + position;
            for head in 0..kv_heads
            {
                let base = position * width + head * head_dim;
                for pair in 0..head_dim / 2
                {
                    let dim = 2 * pair;
                    let exponent = -2.0 * pair as f32 / head_dim as f32;
                    let frequency = theta.powf(exponent);
                    let angle = absolute_position as f32 * frequency;
                    let (sin, cos) = angle.sin_cos();
                    let even = raw[base + dim];
                    let odd = raw[base + dim + 1];
                    rotated[base + dim] = even * cos - odd * sin;
                    rotated[base + dim + 1] = even * sin + odd * cos;
                }
            }
        }
        rotated
    }

    fn env_usize(name: &str, default: usize) -> usize {
        match env::var(name)
        {
            Ok(value) => value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{name} must be an unsigned integer, got {value:?}")),
            Err(env::VarError::NotPresent) => default,
            Err(error) => panic!("failed to read {name}: {error}"),
        }
    }

    fn env_u64(name: &str, default: u64) -> u64 {
        match env::var(name)
        {
            Ok(value) => value
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("{name} must be an unsigned integer, got {value:?}")),
            Err(env::VarError::NotPresent) => default,
            Err(error) => panic!("failed to read {name}: {error}"),
        }
    }

    fn env_usize_list(name: &str, default: &[usize]) -> Vec<usize> {
        match env::var(name)
        {
            Ok(value) => value
                .split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(|item| {
                    item.parse::<usize>()
                        .unwrap_or_else(|_| panic!("{name} contains invalid integer {item:?}"))
                })
                .collect(),
            Err(env::VarError::NotPresent) => default.to_vec(),
            Err(error) => panic!("failed to read {name}: {error}"),
        }
    }

    fn csv_field(value: &str) -> String {
        format!("\"{}\"", value.replace('"', "\"\""))
    }
}
