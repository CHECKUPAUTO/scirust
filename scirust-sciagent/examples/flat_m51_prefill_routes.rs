#[cfg(feature = "flat-attention")]
mod enabled {
    use std::error::Error;
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use scirust_sciagent::config::SciAgentConfig;
    use scirust_sciagent::gpu::{ResidentModel, ResidentPrefillAttentionRoute};
    use scirust_sciagent::model::SciAgentModel;

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn prompt(len: usize, vocab: usize) -> Vec<u32> {
        (0..len)
            .map(|index| ((index * 17 + 3) % vocab) as u32)
            .collect()
    }

    fn percentile_ns(samples: &[Duration], percentile: usize) -> u128 {
        let mut values: Vec<u128> = samples.iter().map(Duration::as_nanos).collect();
        values.sort_unstable();
        let rank = percentile.saturating_mul(values.len()).div_ceil(100).max(1);
        values[rank - 1]
    }

    fn relative_error(actual: &[f32], expected: &[f32]) -> Result<f32, Box<dyn Error>> {
        if actual.len() != expected.len()
        {
            return Err("prefill routes returned different logit lengths".into());
        }
        let mut numerator = 0.0f64;
        let mut denominator = 0.0f64;
        for (&actual, &expected) in actual.iter().zip(expected)
        {
            if !actual.is_finite() || !expected.is_finite()
            {
                return Err("prefill route emitted a non-finite logit".into());
            }
            numerator += f64::from(actual - expected).powi(2);
            denominator += f64::from(expected).powi(2);
        }
        Ok((numerator.sqrt() / denominator.sqrt().max(f64::MIN_POSITIVE)) as f32)
    }

    fn run_route(
        resident: &mut ResidentModel,
        input: &[u32],
        route: ResidentPrefillAttentionRoute,
    ) -> Vec<f32> {
        resident.set_prefill_attention_route(route);
        resident.prefill_last_logits(input)
    }

    fn time_route(
        resident: &mut ResidentModel,
        input: &[u32],
        route: ResidentPrefillAttentionRoute,
    ) -> Duration {
        resident.set_prefill_attention_route(route);
        let started = Instant::now();
        black_box(resident.prefill_last_logits(input));
        started.elapsed()
    }

    pub fn main() -> Result<(), Box<dyn Error>> {
        let vocab = 256usize;
        let warmups = env_usize("SCIAGENT_FLAT_M51_WARMUPS", 2);
        let repeats = env_usize("SCIAGENT_FLAT_M51_REPEATS", 7);
        if warmups == 0 || repeats < 4
        {
            return Err("M51 requires warmups > 0 and repeats >= 4".into());
        }
        let config = SciAgentConfig {
            vocab_size: vocab,
            d_model: env_usize("SCIAGENT_FLAT_M51_D_MODEL", 512),
            n_layers: env_usize("SCIAGENT_FLAT_M51_LAYERS", 8),
            n_heads: env_usize("SCIAGENT_FLAT_M51_Q_HEADS", 8),
            n_kv_heads: env_usize("SCIAGENT_FLAT_M51_KV_HEADS", 2),
            d_ff: env_usize("SCIAGENT_FLAT_M51_D_FF", 1408),
            max_seq_len: env_usize("SCIAGENT_FLAT_M51_MAX_SEQ", 520),
            rope_theta: 10_000.0,
            tie_embeddings: true,
            use_bias: false,
            eps: 1e-5,
        };
        let model = SciAgentModel::new(&config);
        let Some(mut resident) = ResidentModel::from_model(&model)
        else
        {
            return Err("M51 requires a real WGPU adapter".into());
        };

        let routes = [
            ResidentPrefillAttentionRoute::Legacy,
            ResidentPrefillAttentionRoute::FlatRawK,
            ResidentPrefillAttentionRoute::FlatPreRotatedKReuse,
        ];
        println!(
            "revision,adapter,prompt_len,d_model,layers,q_heads,kv_heads,warmups,repeats,legacy_median_ms,current_median_ms,candidate_median_ms,current_over_candidate,legacy_over_candidate,current_rel_err,candidate_rel_err,performance_claim"
        );
        for prompt_len in [128usize, 512]
        {
            if prompt_len > config.max_seq_len
            {
                return Err(format!("M51 prompt {prompt_len} exceeds max_seq_len").into());
            }
            let input = prompt(prompt_len, vocab);
            let legacy = run_route(&mut resident, &input, routes[0]);
            let current = run_route(&mut resident, &input, routes[1]);
            let candidate = run_route(&mut resident, &input, routes[2]);
            let current_error = relative_error(&current, &legacy)?;
            let candidate_error = relative_error(&candidate, &legacy)?;
            if current_error >= 3.0e-3 || candidate_error >= 3.0e-3
            {
                return Err(format!(
                    "M51 prefill parity failed: current={current_error}, candidate={candidate_error}"
                )
                .into());
            }

            for iteration in 0..warmups
            {
                for offset in 0..routes.len()
                {
                    black_box(time_route(
                        &mut resident,
                        &input,
                        routes[(iteration + offset) % routes.len()],
                    ));
                }
            }
            let mut samples: [Vec<Duration>; 3] =
                std::array::from_fn(|_| Vec::with_capacity(repeats));
            for iteration in 0..repeats
            {
                for offset in 0..routes.len()
                {
                    let index = (iteration + offset) % routes.len();
                    samples[index].push(time_route(&mut resident, &input, routes[index]));
                }
            }
            let medians = samples.map(|values| percentile_ns(&values, 50));
            let revision = std::env::var("SCIRUST_SOURCE_REVISION")
                .or_else(|_| std::env::var("GITHUB_SHA"))
                .unwrap_or_else(|_| "unknown".to_owned());
            println!(
                "{revision},\"{}\",{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.8},{:.8},none",
                resident.adapter_name().replace('"', "\"\""),
                prompt_len,
                config.d_model,
                config.n_layers,
                config.n_heads,
                config.n_kv_heads,
                warmups,
                repeats,
                medians[0] as f64 / 1_000_000.0,
                medians[1] as f64 / 1_000_000.0,
                medians[2] as f64 / 1_000_000.0,
                medians[1] as f64 / medians[2].max(1) as f64,
                medians[0] as f64 / medians[2].max(1) as f64,
                current_error,
                candidate_error,
            );
        }
        Ok(())
    }
}

#[cfg(feature = "flat-attention")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    enabled::main()
}

#[cfg(not(feature = "flat-attention"))]
fn main() {
    eprintln!("enable the flat-attention feature to run M51");
    std::process::exit(2);
}
