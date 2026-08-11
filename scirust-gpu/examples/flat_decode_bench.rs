#[cfg(not(feature = "flat-attention"))]
fn main() {
    eprintln!("flat_decode_bench requires --features flat-attention");
}

#[cfg(feature = "flat-attention")]
fn main() {
    bench::run();
}

#[cfg(feature = "flat-attention")]
mod bench {
    use std::env;
    use std::time::{Duration, Instant};

    use scirust_gpu::{
        BackendResult, FlatM11ResidentConfig, GpuChain, GpuMatrix, WgpuFlatM11Bridge,
    };

    const DEFAULT_Q_HEADS: usize = 8;
    const DEFAULT_KV_HEADS: usize = 2;
    const DEFAULT_HEAD_DIM: usize = 64;
    const DEFAULT_WARMUPS: usize = 3;
    const DEFAULT_REPEATS: usize = 11;
    const THETA: f32 = 10_000.0;
    const ATOL: f32 = 1.5e-4;
    const RTOL: f32 = 1.0e-3;

    pub fn run() {
        let q_heads = env_usize("SCIRUST_FLAT_DECODE_BENCH_Q_HEADS", DEFAULT_Q_HEADS);
        let kv_heads = env_usize("SCIRUST_FLAT_DECODE_BENCH_KV_HEADS", DEFAULT_KV_HEADS);
        let head_dim = env_usize("SCIRUST_FLAT_DECODE_BENCH_HEAD_DIM", DEFAULT_HEAD_DIM);
        let warmups = env_usize("SCIRUST_FLAT_DECODE_BENCH_WARMUPS", DEFAULT_WARMUPS);
        let repeats = env_usize("SCIRUST_FLAT_DECODE_BENCH_REPEATS", DEFAULT_REPEATS);
        let kv_lens = env_usize_list("SCIRUST_FLAT_DECODE_BENCH_KV_LENS", &[1, 17, 64, 256, 1024]);

        assert!(q_heads > 0, "q_heads must be non-zero");
        assert!(kv_heads > 0, "kv_heads must be non-zero");
        assert!(
            q_heads.is_multiple_of(kv_heads),
            "q_heads must be a multiple of kv_heads"
        );
        assert!(
            head_dim > 0 && head_dim.is_multiple_of(2),
            "head_dim must be positive and even"
        );
        assert!(repeats > 0, "repeats must be non-zero");
        assert!(!kv_lens.is_empty(), "at least one kv_len is required");
        assert!(
            kv_lens.iter().all(|&len| len > 0),
            "kv_len must be non-zero"
        );

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("WGPU adapter unavailable; paired FLAT decode benchmark cannot run");
            std::process::exit(2);
        };
        let bridge = chain
            .flat_m11_bridge()
            .expect("build FLAT M15 bridge on the benchmark GpuChain");
        assert_eq!(
            chain.adapter_name(),
            bridge.adapter_name(),
            "legacy and FLAT paths must use the same adapter"
        );

        eprintln!("adapter={}", chain.adapter_name());
        eprintln!("measurement=synchronized-final-context-download");
        eprintln!("Q RoPE is inside both timed paths; K RoPE and Q/K/V upload are outside timing");
        println!(
            "adapter,q_heads,kv_heads,head_dim,kv_len,warmups,repeats,legacy_median_us,flat_median_us,legacy_over_flat,parity_max_abs"
        );

        for kv_len in kv_lens
        {
            run_case(
                &chain, &bridge, q_heads, kv_heads, head_dim, kv_len, warmups, repeats,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_case(
        chain: &GpuChain,
        bridge: &WgpuFlatM11Bridge,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        warmups: usize,
        repeats: usize,
    ) {
        let q_width = q_heads.checked_mul(head_dim).expect("Q width overflow");
        let kv_width = kv_heads.checked_mul(head_dim).expect("KV width overflow");
        let q = fixture(q_width, 0.25);
        let raw_k = fixture(
            kv_len.checked_mul(kv_width).expect("K length overflow"),
            0.85,
        );
        let v = fixture(
            kv_len.checked_mul(kv_width).expect("V length overflow"),
            1.45,
        );

        let q_gpu = chain.upload(&q, 1, q_width);
        let raw_k_gpu = chain.upload(&raw_k, kv_len, kv_width);
        let v_gpu = chain.upload(&v, kv_len, kv_width);
        let rotated_k_gpu = chain
            .rope_heads(&raw_k_gpu, kv_heads, kv_len, 0, THETA)
            .expect("pre-rotate resident K once before measurement");
        // This download is deliberately outside the timed region. Besides proving
        // that the fixture exists, it fences uploads and K RoPE so neither path
        // inherits setup work in its first measured iteration.
        let _ = chain
            .download(&rotated_k_gpu)
            .expect("synchronize pre-rotated K fixture");

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

        let legacy_parity = legacy_decode_attention(
            chain,
            &q_gpu,
            &rotated_k_gpu,
            &v_gpu,
            q_heads,
            kv_heads,
            head_dim,
            position,
        )
        .expect("legacy parity dispatch");
        let flat_parity = bridge
            .forward_pre_rotated_k(&q_gpu, &rotated_k_gpu, &v_gpu, config)
            .expect("FLAT M15 parity dispatch");
        let legacy_values = chain
            .download(&legacy_parity)
            .expect("legacy parity readback");
        let flat_values = chain.download(&flat_parity).expect("FLAT parity readback");
        let parity_max_abs = assert_close(&flat_values, &legacy_values);

        for warmup in 0..warmups
        {
            if warmup.is_multiple_of(2)
            {
                let _ = measure_legacy(
                    chain,
                    &q_gpu,
                    &rotated_k_gpu,
                    &v_gpu,
                    q_heads,
                    kv_heads,
                    head_dim,
                    position,
                );
                let _ = measure_flat(bridge, chain, &q_gpu, &rotated_k_gpu, &v_gpu, config);
            }
            else
            {
                let _ = measure_flat(bridge, chain, &q_gpu, &rotated_k_gpu, &v_gpu, config);
                let _ = measure_legacy(
                    chain,
                    &q_gpu,
                    &rotated_k_gpu,
                    &v_gpu,
                    q_heads,
                    kv_heads,
                    head_dim,
                    position,
                );
            }
        }

        let mut legacy_samples = Vec::with_capacity(repeats);
        let mut flat_samples = Vec::with_capacity(repeats);
        for iteration in 0..repeats
        {
            if iteration.is_multiple_of(2)
            {
                legacy_samples.push(measure_legacy(
                    chain,
                    &q_gpu,
                    &rotated_k_gpu,
                    &v_gpu,
                    q_heads,
                    kv_heads,
                    head_dim,
                    position,
                ));
                flat_samples.push(measure_flat(
                    bridge,
                    chain,
                    &q_gpu,
                    &rotated_k_gpu,
                    &v_gpu,
                    config,
                ));
            }
            else
            {
                flat_samples.push(measure_flat(
                    bridge,
                    chain,
                    &q_gpu,
                    &rotated_k_gpu,
                    &v_gpu,
                    config,
                ));
                legacy_samples.push(measure_legacy(
                    chain,
                    &q_gpu,
                    &rotated_k_gpu,
                    &v_gpu,
                    q_heads,
                    kv_heads,
                    head_dim,
                    position,
                ));
            }
        }

        let legacy_median = median(&mut legacy_samples);
        let flat_median = median(&mut flat_samples);
        let legacy_us = micros(legacy_median);
        let flat_us = micros(flat_median);
        assert!(legacy_us.is_finite() && legacy_us > 0.0);
        assert!(flat_us.is_finite() && flat_us > 0.0);
        let ratio = legacy_us / flat_us;
        assert!(ratio.is_finite() && ratio > 0.0);

        println!(
            "{},{q_heads},{kv_heads},{head_dim},{kv_len},{warmups},{repeats},{legacy_us:.3},{flat_us:.3},{ratio:.6},{parity_max_abs:.9}",
            csv_field(chain.adapter_name()),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn legacy_decode_attention(
        chain: &GpuChain,
        q: &GpuMatrix,
        pre_rotated_k: &GpuMatrix,
        v: &GpuMatrix,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        position: usize,
    ) -> BackendResult<GpuMatrix> {
        let qr = chain.rope_heads(q, q_heads, 1, position, THETA)?;
        let d_model = q_heads * head_dim;
        let repeat = q_heads / kv_heads;
        let mut out: Option<GpuMatrix> = None;
        for head in 0..q_heads
        {
            let kv = head / repeat;
            let qs = chain.slice_cols(&qr, head * head_dim, head_dim)?;
            let ks = chain.slice_cols(pre_rotated_k, kv * head_dim, head_dim)?;
            let vs = chain.slice_cols(v, kv * head_dim, head_dim)?;
            // All cached keys are at positions <= the single decode query, so
            // this matches ResidentModel::incr_attention's non-causal head call.
            let ctx = chain.attention(&qs, &ks, &vs, false)?;
            let padded = chain.place_cols(&ctx, head * head_dim, d_model)?;
            out = Some(match out
            {
                None => padded,
                Some(acc) => chain.add(&acc, &padded)?,
            });
        }
        out.ok_or_else(|| scirust_gpu::BackendError::ShapeMismatch("zero query heads".into()))
    }

    #[allow(clippy::too_many_arguments)]
    fn measure_legacy(
        chain: &GpuChain,
        q: &GpuMatrix,
        pre_rotated_k: &GpuMatrix,
        v: &GpuMatrix,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        position: usize,
    ) -> Duration {
        let start = Instant::now();
        let output = legacy_decode_attention(
            chain,
            q,
            pre_rotated_k,
            v,
            q_heads,
            kv_heads,
            head_dim,
            position,
        )
        .expect("legacy decode dispatch");
        let values = chain
            .download(&output)
            .expect("legacy synchronized readback");
        assert!(!values.is_empty());
        start.elapsed()
    }

    fn measure_flat(
        bridge: &WgpuFlatM11Bridge,
        chain: &GpuChain,
        q: &GpuMatrix,
        pre_rotated_k: &GpuMatrix,
        v: &GpuMatrix,
        config: FlatM11ResidentConfig,
    ) -> Duration {
        let start = Instant::now();
        let output = bridge
            .forward_pre_rotated_k(q, pre_rotated_k, v, config)
            .expect("FLAT M15 decode dispatch");
        let values = chain.download(&output).expect("FLAT synchronized readback");
        assert!(!values.is_empty());
        start.elapsed()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) -> f32 {
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0f32;
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate()
        {
            let tolerance = ATOL + RTOL * expected.abs();
            let error = (actual - expected).abs();
            max_abs = max_abs.max(error);
            assert!(
                error <= tolerance,
                "parity index {index}: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
            );
        }
        max_abs
    }

    fn fixture(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let x = index as f32 * 0.023 + phase;
                x.sin() * 1.875 + (x * 0.41).cos() * 0.28125
            })
            .collect()
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

    fn median(samples: &mut [Duration]) -> Duration {
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    fn micros(duration: Duration) -> f64 {
        duration.as_secs_f64() * 1_000_000.0
    }

    fn csv_field(value: &str) -> String {
        format!("\"{}\"", value.replace('"', "\"\""))
    }
}
