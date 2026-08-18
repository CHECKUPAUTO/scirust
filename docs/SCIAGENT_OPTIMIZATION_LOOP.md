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

The runner itself never invents a performance result. Baseline, verifier, and benchmark stages must write the canonical JSON metric files described in `SKILL_OPTIMIZATION.md`. Missing, malformed, non-finite, or non-positive timing data fails closed.

Every stage gets its stdout and stderr captured in the run directory. Generation, compilation, verification-command, benchmark-command, and profile failures are recorded in the report with their stage log path. Those failures become part of the next iteration context instead of being discarded.

## Loop

### 1. Freeze the target

Create a dedicated Git branch or worktree. Do not optimize directly on the protected branch. Record the workload, target hardware/backend, allowed implementation paths, correctness oracle, benchmark protocol, and promotion threshold.

### 2. Establish a baseline

The baseline command runs before any generation. It must use the same workload and timing boundaries later used for candidates and write:

```json
{"median_ns": 1000.0}
```

The runner freezes that value for the entire run. A baseline failure aborts the run because there is no valid performance reference against which a candidate can be scored.

### 3. Generate one candidate

The first iteration invokes `commands.generate`. The command receives the skill and a machine-readable context through environment variables. The context contains the goal, backend, allowed paths, tolerances, frozen baseline, current iteration, previous measured candidates, and previous stage failures.

The generator may be the local SCIAGENT checkpoint or another explicitly configured adapter. The production adapter `scripts/sciagent-opt-local-model.sh` supplies the skill, current source, prior errors and their log tails, verification evidence, timing evidence, and available profiler text to the next SCIAGENT turn. It accepts only a unified Git patch and rejects patches touching paths outside `allowed_paths`.

If generation itself fails, that failure is recorded and the next iteration may retry/regenerate from the recorded evidence.

### 4. Compile

`commands.compile` must build/check the affected target using the real feature set. A non-zero exit code or timeout rejects that candidate before verification or benchmarking. The compiler output is retained in the stage log and supplied to the next rewrite so SCIAGENT can repair the candidate instead of terminating the whole optimization campaign.

Compilation failures are never converted into benchmark results.

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

A candidate fails the correctness gate when `passed` is false or when a reported error exceeds the manifest tolerance. A failed correctness gate is recorded and the candidate is sent back to rewriting.

**An incorrect candidate is not benchmarked.** This avoids rewarding invalid implementations and prevents performance measurements from becoming an optimization signal before semantics are restored.

### 6. Benchmark

Only a candidate that has compiled and passed the correctness gate reaches `commands.benchmark`. It is measured under the frozen protocol and writes:

```json
{"median_ns": 800.0}
```

The runner computes:

```text
speedup = baseline_median_ns / candidate_median_ns
```

The default promotion gate is 1.05x. Each manifest may choose a stricter threshold.

### 7. Decide

The state machine can emit:

- `promote`: correctness and performance gates both pass;
- `retry-generation`: the candidate-generation stage itself failed;
- `rewrite-for-compilation`: generated code failed its compile gate;
- `rewrite-for-correctness`: verification failed or exceeded tolerances;
- `rewrite-for-performance`: correctness passed but the speedup target was missed;
- `budget-exhausted`: no candidate cleared all gates within the iteration budget.

Only `promote` represents success.

### 8. Profile when performance is the problem

When a candidate is correct but too slow and a profile command exists, the runner invokes it before the next rewrite. Profiling is not used to excuse a regression; it exists to give the next iteration evidence about memory bandwidth, launch/dispatch overhead, occupancy, cache behavior, vectorization, allocation, synchronization, or another actual bottleneck.

For the I250 CUDA task, the adapter uses Nsight Systems when `nsys` exists and additionally extracts `nsys stats` into a textual artifact. The local SCIAGENT adapter includes the latest profiler logs/stats in the next prompt. If Nsight Systems is unavailable, the adapter says so explicitly and retains benchmark evidence instead of inventing profiler data.

### 9. Rewrite from evidence

Iterations after the first call `commands.rewrite` when supplied, otherwise they call the generator again. `context.json` carries every previous verified timing/decision plus recorded stage failures. The local model adapter also reads the associated failure-log tails and profiler artifacts. The next proposal therefore has concrete evidence explaining why the previous attempt failed.

### 10. Stop honestly

The runner stops immediately on `promote`. If the iteration budget is exhausted, it records `budget-exhausted` and retains the best verified speedup. It never relabels a near miss as success.

## Commands

Validate a manifest without executing it:

```bash
cargo +nightly-2026-07-02 run -p scirust-sciagent --bin sciagent-optimize -- \
  plan --manifest scirust-sciagent/examples/optimization_tasks/smoke.json
```

Run a task:

```bash
cargo +nightly-2026-07-02 run -p scirust-sciagent --bin sciagent-optimize -- \
  run \
  --manifest path/to/task.json \
  --workspace . \
  --run-root .sciagent-opt
```

Evaluate a measured candidate directly:

```bash
cargo +nightly-2026-07-02 run -p scirust-sciagent --bin sciagent-optimize -- \
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

## First production task: I250 CUDA decode

`scirust-sciagent/examples/optimization_tasks/i250_cuda_decode.json` is the first real hardware target. It optimizes only:

```text
scirust-sciagent/src/cuda_decode.rs
```

Its production contract is:

- Jetson Thor CUDA backend;
- batch-one SCIAGENT I250 resident decode;
- prompt length 128 and 64 generated tokens by default;
- five independent baseline samples and five independent candidate samples;
- median timing normalized to ns/token from the existing `fast_tok_s` benchmark output;
- exact cached-oracle token parity plus both canonical full-logits parity checks before benchmarking;
- default promotion threshold 1.05x;
- up to eight optimization iterations;
- optional Nsight Systems profiling and textual stats extraction;
- candidate generation/rewrite through the local SCIAGENT checkpoint identified by `SCIAGENT_CKPT`.

### Why this production task is launched from the root Thor shell

The persistent GitHub Actions runner on Thor is intentionally systemd-sandboxed and cannot see production checkpoints under `/root`. Existing SciRust production policy keeps that isolation. Consequently, GitHub Actions validates the optimization engine and deterministic recovery logic, while a real SCIAGENT-checkpoint optimization campaign is launched from a root shell/worktree on Thor.

The production trainer and optimization campaign must share the same physical GPU advisory lock:

```text
/dev/nvidia0
```

The wrapper `scripts/sciagent-opt-i250-thor-root.sh` enforces this contract and fails immediately instead of interfering with an active training/qualification owner of the GPU.

From a dedicated worktree on Thor:

```bash
cd /root/scirust-opt-i250
export SCIAGENT_CKPT=/root/scirust/checkpoints/bpe350m_v5_semantics_v2
bash scripts/sciagent-opt-i250-thor-root.sh
```

Do not repoint the production-training worktree while training is active. Create/use a separate worktree for optimization.

## Backend-specific strategy

For CPU workloads, start with algorithmic work, allocation removal, cache locality and branch behavior. For portable SIMD, preserve tail correctness and a scalar/reference oracle. On ARM SVE/SVE2, never assume a fixed vector length. For WGPU, focus on dispatches, transfers, workgroup geometry and workgroup-memory reuse. For CUDA, use profiler evidence to decide between fusion, block geometry, coalescing, vectorized access, warp primitives, shared memory, register-pressure reduction, Tensor Cores, CUDA Graphs, persistent execution or asynchronous pipelines.

Optimization is still subordinate to SciRust's numerical and execution contracts. A faster result that silently changes backend, precision, recurrence behavior, parity, attestation, or public semantics is a failed candidate.

## CI

`.github/workflows/sciagent-optimization-loop.yml` checks canonical formatting on the repository's pinned Rust nightly, builds the CLI, runs the optimization-agent unit tests, validates the deterministic and I250 manifests, runs a normal end-to-end promotion case, then runs a recovery case that intentionally forces a compilation failure at iteration 1 and a correctness failure at iteration 2. That recovery test forbids benchmarking until iteration 3, proving that invalid candidates cannot receive performance credit.

Hardware-specific CUDA/SVE performance qualification remains on the physical/self-hosted hardware; GitHub-hosted CPU CI cannot prove accelerator performance.

## Next production targets

After I250 decode, the same protocol can be attached to:

1. FLAT attention prefill/decode kernels;
2. SciRust CUDA tensor kernels;
3. ARM SVE/SVE2 hot loops on Jetson Thor;
4. WGPU tensor/attention dispatch fusion;
5. `scirust-special`, solvers, sparse kernels, and tensor contraction CPU/SIMD hot paths.

Each production target should get its own manifest and hardware-specific benchmark/verifier adapter while keeping the same promotion protocol.
