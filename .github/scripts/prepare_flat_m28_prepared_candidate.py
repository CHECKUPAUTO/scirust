#!/usr/bin/env python3
"""Prepare an uncommitted M28 candidate using FLAT #74 prepared bindings.

This script is intentionally evidence-only. It rewrites the checked-out SciRust
workspace inside the physical-Thor workflow, updates the FLAT git revision, and
adds a fourth measured path using WgpuGroupedForwardPipeline::prepare /
encode_prepared. The repository pin is not promoted by this script.
"""

from pathlib import Path

OLD_FLAT_REV = "24d3340edeb059e40e0fe0c400e814685701d855"
CANDIDATE_FLAT_REV = "f8e1be6a073596b446d298910eb04c122c7e29b0"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def patch_cargo() -> None:
    path = Path("scirust-gpu/Cargo.toml")
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        f'rev = "{OLD_FLAT_REV}"',
        f'rev = "{CANDIDATE_FLAT_REV}"',
        "FLAT git revision",
    )
    path.write_text(text, encoding="utf-8")


def patch_benchmark() -> None:
    path = Path("scirust-gpu/examples/flat_m28_naive_vs_fused.rs")
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "const DEFAULT_REPEATS: usize = 9;",
        "const DEFAULT_REPEATS: usize = 12;",
        "default repeats",
    )
    text = replace_once(
        text,
        "adapter,backend,causal,seq_len,head_dim,warmups,repeats,naive_median_us,naive_p95_us,flat_fresh_median_us,flat_fresh_p95_us,flat_reused_median_us,flat_reused_p95_us,naive_over_flat_fresh,naive_over_flat_reused,naive_parity_max_abs,flat_parity_max_abs,performance_claim",
        "adapter,backend,causal,seq_len,head_dim,warmups,repeats,naive_median_us,naive_p95_us,flat_fresh_median_us,flat_fresh_p95_us,flat_reused_median_us,flat_reused_p95_us,flat_prepared_median_us,flat_prepared_p95_us,naive_over_flat_fresh,naive_over_flat_reused,naive_over_flat_prepared,flat_reused_over_prepared,naive_parity_max_abs,flat_parity_max_abs,prepared_parity_max_abs,performance_claim",
        "CSV header",
    )

    anchor = '        assert_close("FLAT LSE", &flat_host[layout.lse_offset()..], &expected.lse)?;\n'
    start = text.index(anchor) + len(anchor)
    end = text.index("\n        println!(\n", start)
    timing = r'''

        // Candidate FLAT #74: prepare immutable uniform/bind-group state once.
        // Keep the ordinary encode() path in the same run to isolate host-side
        // preparation overhead from the fused kernel itself.
        let prepared_output = pipeline.create_output_buffer(ctx.device(), shape)?;
        let prepared = pipeline.prepare(
            ctx.device(),
            GroupedForwardPass {
                q: flat_inputs.q,
                k: flat_inputs.k,
                v: flat_inputs.v,
                output: &prepared_output,
                shape,
                config,
            },
        )?;
        let time_flat_prepared = || -> Result<Duration, Box<dyn Error>> {
            let start = Instant::now();
            let mut encoder = ctx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("scirust-m28-flat-prepared-forward"),
                });
            let _ = pipeline.encode_prepared(&mut encoder, &prepared);
            ctx.queue().submit(Some(encoder.finish()));
            let _ = ctx.device().poll(wgpu::Maintain::Wait);
            Ok(start.elapsed())
        };

        let _ = time_flat_prepared()?;
        let prepared_host = ctx.download_buffer(
            &prepared_output,
            layout.output_elements,
            layout.output_bytes,
        )?;
        let prepared_parity = assert_close(
            "FLAT prepared output",
            &prepared_host[..layout.q_elements],
            &expected.output,
        )?;
        assert_close(
            "FLAT prepared LSE",
            &prepared_host[layout.lse_offset()..],
            &expected.lse,
        )?;

        for _ in 0..warmups
        {
            let _ = time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?;
            let _ = time_flat_fresh_output(&ctx, &pipeline, flat_inputs, shape, config)?;
            let _ = time_flat_reused(&ctx, &pipeline, flat_inputs, &reused_output, shape, config)?;
            let _ = time_flat_prepared()?;
        }

        let mut naive_samples = Vec::with_capacity(repeats);
        let mut flat_fresh_samples = Vec::with_capacity(repeats);
        let mut flat_reused_samples = Vec::with_capacity(repeats);
        let mut flat_prepared_samples = Vec::with_capacity(repeats);
        for iteration in 0..repeats
        {
            // Cyclic rotation. The physical gate requests 12 repeats so every
            // path occupies every position exactly three times.
            match iteration % 4
            {
                0 =>
                {
                    naive_samples.push(time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?);
                    flat_fresh_samples.push(time_flat_fresh_output(&ctx, &pipeline, flat_inputs, shape, config)?);
                    flat_reused_samples.push(time_flat_reused(&ctx, &pipeline, flat_inputs, &reused_output, shape, config)?);
                    flat_prepared_samples.push(time_flat_prepared()?);
                },
                1 =>
                {
                    flat_fresh_samples.push(time_flat_fresh_output(&ctx, &pipeline, flat_inputs, shape, config)?);
                    flat_reused_samples.push(time_flat_reused(&ctx, &pipeline, flat_inputs, &reused_output, shape, config)?);
                    flat_prepared_samples.push(time_flat_prepared()?);
                    naive_samples.push(time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?);
                },
                2 =>
                {
                    flat_reused_samples.push(time_flat_reused(&ctx, &pipeline, flat_inputs, &reused_output, shape, config)?);
                    flat_prepared_samples.push(time_flat_prepared()?);
                    naive_samples.push(time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?);
                    flat_fresh_samples.push(time_flat_fresh_output(&ctx, &pipeline, flat_inputs, shape, config)?);
                },
                _ =>
                {
                    flat_prepared_samples.push(time_flat_prepared()?);
                    naive_samples.push(time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?);
                    flat_fresh_samples.push(time_flat_fresh_output(&ctx, &pipeline, flat_inputs, shape, config)?);
                    flat_reused_samples.push(time_flat_reused(&ctx, &pipeline, flat_inputs, &reused_output, shape, config)?);
                },
            }
        }

        let naive_median = median_ns(&naive_samples);
        let flat_fresh_median = median_ns(&flat_fresh_samples);
        let flat_reused_median = median_ns(&flat_reused_samples);
        let flat_prepared_median = median_ns(&flat_prepared_samples);
        let naive_over_flat_fresh = naive_median as f64 / flat_fresh_median.max(1) as f64;
        let naive_over_flat_reused = naive_median as f64 / flat_reused_median.max(1) as f64;
        let naive_over_flat_prepared = naive_median as f64 / flat_prepared_median.max(1) as f64;
        let flat_reused_over_prepared = flat_reused_median as f64 / flat_prepared_median.max(1) as f64;
'''
    text = text[:start] + timing + text[end:]

    print_start = text.index("        println!(\n", start)
    print_end = text.index("        );\n", print_start) + len("        );\n")
    new_print = r'''        println!(
            "{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{:.8},{:.8},{:.8},none",
            ctx.adapter_name().replace(',', ";"),
            ctx.adapter_backend().replace(',', ";"),
            causal,
            seq_len,
            head_dim,
            warmups,
            repeats,
            naive_median as f64 / 1_000.0,
            percentile_ns(&naive_samples, 95) as f64 / 1_000.0,
            flat_fresh_median as f64 / 1_000.0,
            percentile_ns(&flat_fresh_samples, 95) as f64 / 1_000.0,
            flat_reused_median as f64 / 1_000.0,
            percentile_ns(&flat_reused_samples, 95) as f64 / 1_000.0,
            flat_prepared_median as f64 / 1_000.0,
            percentile_ns(&flat_prepared_samples, 95) as f64 / 1_000.0,
            naive_over_flat_fresh,
            naive_over_flat_reused,
            naive_over_flat_prepared,
            flat_reused_over_prepared,
            max_abs_error(&naive_host, &expected.output).max(naive_parity),
            flat_parity,
            prepared_parity,
        );
'''
    text = text[:print_start] + new_print + text[print_end:]
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    patch_cargo()
    patch_benchmark()
    print(f"candidate_flat_revision={CANDIDATE_FLAT_REV}")
