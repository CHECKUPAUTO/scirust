# SCIAGENT evidence-driven optimization loop

This document defines the SciRust translation of the generate/compile/verify/benchmark/profile/rewrite pattern used by specialized kernel-optimization agents.

The implementation lives in `scirust-sciagent::optimization_agent`, the CLI is `sciagent-optimize`, and the agent contract is `scirust-sciagent/SKILL_OPTIMIZATION.md`.

## Why it belongs in SciRust

SciRust already contains the pieces that a separate optimization repository would otherwise have to duplicate: SCIAGENT, CUDA and WGPU backends, SIMD/SVE-capable Rust code, scientific correctness tests, execution attestation, benchmark examples, and the ElasticAutoTuner. The optimization loop therefore stays in the existing workspace and treats those systems as targets and oracles.

## Architecture

A task is a JSON manifest containing:

- a stable task id and scientific/performance goal;
- target crate and backend (`cpu`, `simd`, `sve`, `wgpu`, or `cuda`);
- the implementation paths the agent is allowed to modify;
- an iteration budget, minimum speedup, error tolerances, and command timeout;
- commands for baseline, generation, compilation, verification, benchmark, optional profiling, and optional rewriting.

The runner itself never invents a performance result. Baseline, verifier, and benchmark stages must write the canonical JSON metric files described in `SKILL_OPTIMIZATION.md`. Missing, malformed, non-finite, or non-positive timing data aborts the run.

## Loop

### 1. Freeze the target

Create a dedicated Git branch or worktree. Do not optimize directly on the protected branch. Record the workload, target hardware/backend, allowed implementation paths, correctness oracle, benchmark protocol, and promotion threshold.

### 2. Establish a baseline

The baseline command runs before any generation. It must use the same workload and timing boundaries later used for candidates and write:

```json
{"median_ns": 1000.0}
```

The runner freezes that value for the entire run.

### 3. Generate one candidate

The first iteration invokes `commands.generate`. The command receives the skill and a machine-readable context through environment variables. The context contains the goal, backend, allowed paths, tolerances, frozen baseline, current iteration, and all previous evidence.

The generator may be the local SCIAGENT checkpoint or another explicitly configured adapter. The intended production mode is a local SCIAGENT process that emits a focused patch from this context.

### 4. Compile

`commands.compile` must build/check the affected target using the real feature set. A non-zero exit code or timeout rejects the iteration immediately. Compilation failures are never converted into benchmark results.

### 5. Verify

`commands.verify` runs the scientific/numerical oracle and writes:

```json
{
  "passed": true,
  "max_abs_error": 1e-9,
  "max_rel_error": 2e-9,
  "notes": "domain-specific evidence"
}
```

A candidate cannot be promoted when `passed` is false or when a reported error exceeds the manifest tolerance.

### 6. Benchmark

`commands.benchmark` measures the candidate under the frozen protocol and writes:

```json
{"median_ns": 800.0}
```

The runner computes:

```text
speedup = baseline_median_ns / candidate_median_ns
```

The default promotion gate is 1.05x. Each manifest may choose a stricter threshold.

### 7. Decide

The state machine has three per-iteration outcomes:

- `promote`: correctness and performance gates both pass;
- `rewrite-for-correctness`: verification failed or exceeded tolerances;
- `rewrite-for-performance`: correctness passed but the speedup target was missed.

### 8. Profile when performance is the problem

When a candidate is correct but too slow and a profile command exists, the runner invokes it before the next rewrite. Profiling is not used to excuse a regression; it exists to give the next iteration evidence about memory bandwidth, launch/dispatch overhead, occupancy, cache behavior, vectorization, allocation, synchronization, or another actual bottleneck.

### 9. Rewrite from evidence

Iterations after the first call `commands.rewrite` when supplied, otherwise they call the generator again. `context.json` contains every previous verification result, timing, speedup, and decision so the next proposal can be different for a measurable reason.

### 10. Stop honestly

The runner stops immediately on `promote`. If the iteration budget is exhausted, it records `budget-exhausted` and retains the best verified speedup. It never relabels a near miss as success.

## Commands

Validate a manifest without executing it:

```bash
cargo run -p scirust-sciagent --bin sciagent-optimize -- \
  plan --manifest scirust-sciagent/examples/optimization_tasks/smoke.json
```

Run a task:

```bash
cargo run -p scirust-sciagent --bin sciagent-optimize -- \
  run \
  --manifest path/to/task.json \
  --workspace . \
  --run-root .sciagent-opt
```

Evaluate a measured candidate directly:

```bash
cargo run -p scirust-sciagent --bin sciagent-optimize -- \
  score \
  --baseline-ns 1000 \
  --candidate-ns 800 \
  --verified \
  --max-abs-error 1e-9 \
  --max-rel-error 2e-9 \
  --min-speedup 1.05
```

## Environment passed to stages

Every stage receives:

```text
SCIAGENT_OPT_TASK_ID
SCIAGENT_OPT_BACKEND
SCIAGENT_OPT_GOAL
SCIAGENT_OPT_ITERATION
SCIAGENT_OPT_RUN_DIR
SCIAGENT_OPT_CONTEXT
SCIAGENT_OPT_BASELINE_METRICS
SCIAGENT_OPT_VERIFY_METRICS
SCIAGENT_OPT_CANDIDATE_METRICS
SCIAGENT_OPT_SKILL_PATH
```

The runner removes known API, cloud, GitHub, wallet, and SciRust secret variables before spawning stage commands. A task that genuinely needs a credential must be designed explicitly rather than inheriting ambient secrets accidentally.

## Backend-specific strategy

For CPU workloads, start with algorithmic work, allocation removal, cache locality and branch behavior. For portable SIMD, preserve tail correctness and a scalar/reference oracle. On ARM SVE/SVE2, never assume a fixed vector length. For WGPU, focus on dispatches, transfers, workgroup geometry and workgroup-memory reuse. For CUDA, use profiler evidence to decide between fusion, block geometry, coalescing, vectorized access, warp primitives, shared memory, register-pressure reduction, Tensor Cores, CUDA Graphs, persistent execution or asynchronous pipelines.

Optimization is still subordinate to SciRust's numerical and execution contracts. A faster result that silently changes backend, precision, recurrence behavior, parity, attestation, or public semantics is a failed candidate.

## CI

`.github/workflows/sciagent-optimization-loop.yml` checks formatting, builds the new CLI, runs the optimization-agent unit tests, validates the smoke manifest, and executes a deterministic end-to-end smoke optimization. Hardware-specific CUDA/SVE benchmark tasks remain runner/self-hosted jobs because GitHub-hosted CPU CI cannot validate real accelerator performance.

## Next production targets

The framework is intentionally generic. The first useful tasks to attach to real hardware are:

1. SCIAGENT I250 CUDA batch-one decode;
2. FLAT attention prefill/decode kernels;
3. SciRust CUDA tensor kernels;
4. ARM SVE/SVE2 hot loops on Jetson Thor;
5. WGPU tensor/attention dispatch fusion;
6. `scirust-special`, solvers, sparse kernels, and tensor contraction CPU/SIMD hot paths.

Each production target should get its own manifest and hardware-specific benchmark/verifier adapter while keeping the same promotion protocol.
