# Dream real-policy proof on Jetson AGX Thor

This protocol tests whether the synthetic `63.110794%` refresh-compute gain from
SciRust policy discovery transfers to real Dream 7B attention states.

## Evidence produced

For every reused attention layer, the probe executes two paths from the same
layer input state:

1. the competitor's cached K/V reuse path;
2. an exact full K/V refresh path.

It records the competitor's cosine trigger plus seven additional signals and a
bounded divergence between the two attention outputs. This is local
attention-output counterfactual evidence, not yet a GSM8K, HumanEval, task
accuracy, or wall-clock production claim.

## Jetson target

- NVIDIA Jetson AGX Thor
- Ubuntu 24.04
- CUDA 13.0
- compute capability 11.0
- 128 GB unified memory
- model: `Dream-org/Dream-v0-Instruct-7B`
- NVIDIA container: `nvcr.io/nvidia/pytorch:25.08-py3`
- validated image digest:
  `sha256:ace9a848c0ae543317e3c4763b6b4248961c47902625abfe3c77a0fb931c50fb`

The harness uses the NVIDIA NGC PyTorch container because the generic CUDA 13
PyTorch wheel detects Thor but fails BF16 cuBLAS GEMM on this platform. The host
Python, TensorRT-LLM installation, and system packages are not modified.

## Phase 1: first held-out Dream test

Run:

```bash
bash experiments/elastic-cache-policy/jetson/run_jetson_dream_proof.sh
```

Observed on the first real trace:

- 30 deterministic trajectories;
- 25,026 counterfactual observations;
- learned quality loss: `0.06384892`;
- quality budget: `0.05`;
- learned normalized refresh compute: `0.81045261`;
- raw compute reduction: `18.954739%`;
- best admissible fixed gamma: `inf` / always refresh;
- synthetic `63.11%` reproduction: false.

The raw compute reduction is not admissible because the learned policy exceeds
the registered quality budget.

## Phase 2: trajectory-balanced exploratory cross-validation

Run:

```bash
bash experiments/elastic-cache-policy/jetson/run_robust_cross_validation.sh
```

This phase reuses the first trace and applies:

- equal objective weight per trajectory;
- a calibration budget of `0.025` for a final budget of `0.05`;
- constraints on aggregate loss, mean trajectory loss, and the nearest-rank
  90th percentile of trajectory loss;
- five rotating train/calibration/test folds;
- a robust fixed-gamma comparison in every fold.

Observed exploratory result:

- folds meeting the quality budget: `5/5`;
- folds beating the constrained fixed-gamma baseline: `5/5`;
- mean relative compute improvement: `1.7591488%`;
- minimum fold improvement: `0.151849%`;
- maximum fold improvement: `3.636473%`;
- mean learned quality loss: `0.005368442`;
- maximum learned quality loss: `0.01064077`;
- maximum 90th-percentile trajectory loss: `0.03361345`;
- strict exploratory success: true.

This is not confirmatory because the robust method was designed after inspecting
the first split.

## Phase 3: frozen independent confirmation

Run:

```bash
bash experiments/elastic-cache-policy/jetson/run_confirmatory_dream_proof.sh
```

The confirmation protocol is frozen before collecting the new trace:

- new seed: `20260805`;
- 60 independent deterministic trajectories;
- no policy fitting or threshold calibration on the new trace;
- the five cross-validation policies are frozen as a bundle;
- the primary policy refreshes when at least three of five policies vote to
  refresh;
- primary baseline: always refresh;
- quality budget: `0.05` for aggregate, mean trajectory, and 90th-percentile
  trajectory loss;
- minimum mean compute improvement: `0.5%`;
- deterministic 10,000-sample trajectory bootstrap;
- the lower bound of the 95% compute-improvement interval must exceed zero.

Outputs:

```text
~/.local/share/scirust/dream-policy-proof/results/confirmatory-<UTC timestamp>/
├── dream_counterfactual_trace.csv
├── dream_trace_manifest.json
└── dream_frozen_policy_confirmatory_report.json
```

Exit status:

- `0`: every frozen confirmatory criterion is satisfied;
- `2`: the independent experiment is valid but at least one criterion fails;
- another nonzero status: environment, trace integrity, or evaluation failure.

## Metric boundary

The stale-loss target is:

```text
0.5 * bounded cosine loss + 0.5 * bounded relative MSE
```

between cached and fully refreshed attention outputs for the same layer input.
The refresh cost is the normalized number of downstream layers that must be
recomputed after a trigger.
