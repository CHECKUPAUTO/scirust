# Dream real-policy proof on Jetson AGX Thor

This protocol evaluates whether SciRust can replace Dream Elastic-Cache's fixed
cosine threshold with a deterministic eight-signal cache-refresh policy.

## Evidence boundary

For every reused attention layer, the trace probe executes both the cached K/V
path and an exact full K/V refresh path from the same layer input. It records a
bounded divergence between their attention outputs. The later GSM8K phases run
the frozen policy directly and measure exact-match accuracy and wall-clock
latency without the dual counterfactual path.

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

The NGC container is required because the generic CUDA 13 PyTorch wheel detects
Thor but fails BF16 cuBLAS GEMM. The host Python, TensorRT-LLM installation, and
system packages are not modified.

## Phase 1: first held-out Dream test

```bash
bash experiments/elastic-cache-policy/jetson/run_jetson_dream_proof.sh
```

Observed result:

- 30 trajectories and 25,026 counterfactual observations;
- learned stale-loss fraction: `0.06384892`;
- registered budget: `0.05`;
- raw refresh-compute reduction: `18.954739%`;
- best admissible fixed gamma: `inf` / always refresh;
- synthetic `63.110794%` reproduction: false.

The raw reduction is inadmissible because it exceeds the quality budget.

## Phase 2: trajectory-balanced exploratory cross-validation

```bash
bash experiments/elastic-cache-policy/jetson/run_robust_cross_validation.sh
```

The robust mode gives equal objective weight to every trajectory, calibrates at
`0.025` for a final `0.05` budget, constrains aggregate, mean, and 90th-percentile
trajectory loss, and rotates five train/calibration/test folds.

Observed exploratory result:

- folds meeting the budget: `5/5`;
- folds beating their constrained fixed-gamma baseline: `5/5`;
- mean relative refresh-compute improvement: `1.7591488%`;
- fold range: `0.151849%` to `3.636473%`;
- mean learned quality loss: `0.005368442`;
- maximum learned quality loss: `0.01064077`;
- maximum 90th-percentile trajectory loss: `0.03361345`;
- strict exploratory success: true.

## Phase 3: frozen independent confirmation

```bash
bash experiments/elastic-cache-policy/jetson/run_confirmatory_dream_proof.sh
```

The five cross-validation policies are frozen before collecting a new trace.
The primary ensemble refreshes when at least three of five policies vote to
refresh. No fitting or threshold calibration is performed on the new trace.

Observed independent result on seed `20260805`:

- 60 trajectories and 50,241 observations;
- trace SHA-256:
  `0ed76ee45bd8b612cc62a581e6629469354554abb12e213df5a7feaf8fdb1192`;
- aggregate stale-loss fraction: `0.0012919755`;
- mean trajectory stale-loss fraction: `0.0024553258`;
- 90th-percentile trajectory stale-loss fraction: `0.0063779950`;
- worst trajectory stale-loss fraction: `0.0354609929`;
- aggregate refresh-compute improvement: `1.1700108%`;
- mean per-trajectory improvement: `1.2360039%`;
- deterministic bootstrap 95% interval:
  `[1.1424517%, 1.3392975%]`;
- quality criteria: pass;
- compute criteria: pass;
- `confirmatory_success`: true.

This confirms a small but statistically stable local refresh-compute gain. It
does not revive the synthetic `63.11%` claim.

## Phase 4: registered paired GSM8K end-to-end benchmark

```bash
bash experiments/elastic-cache-policy/jetson/run_end_to_end_gsm8k.sh
```

The frozen 3-of-5 ensemble is compared with `always_refresh` on the same 60
GSM8K questions without fitting or calibration on GSM8K.

Observed registered result:

- accuracy delta: `+1.6667` percentage points, 95% interval `[0, 5]` points;
- mean wall-clock improvement: `0.2909%`;
- wall-clock 95% interval: `[-1.6496%, 2.1121%]`;
- mean total refresh-cost improvement: `0.5329%`;
- refresh-cost 95% interval: `[-1.4611%, 2.4059%]`;
- quality criterion: pass;
- latency criterion: fail;
- refresh-compute criterion: fail;
- `end_to_end_success`: false.

The registered negative verdict is final for this protocol and must not be
retroactively changed.

### Secondary diagnostic

```bash
bash experiments/elastic-cache-policy/jetson/finalize_end_to_end_diagnostic.sh
```

The diagnostic preserves the registered verdict and shows why the total metric
is noisy:

- exact response match rate: `70%`;
- same prediction rate: `96.67%`;
- same correctness rate: `98.33%`;
- the ensemble executes more cache decisions on `55/60` questions;
- mean ensemble/always decision ratio: `1.0779`;
- mean conditional refresh-cost saving: `7.3859%`, 95% interval
  `[6.7011%, 8.1040%]`;
- mean refresh-cost-per-decision improvement: `7.6675%`, 95% interval
  `[6.9636%, 8.4200%]`;
- mean latency-per-decision improvement: `7.4346%`, 95% interval
  `[6.7421%, 8.1682%]`.

The execution order is strongly confounded with question identity: ensemble-first
pairs show a positive mean while always-first pairs show a negative mean. Simple
alternation therefore does not fully remove thermal, carry-over, and trajectory
bias.

## Phase 5: repeated counterbalanced crossover

```bash
bash experiments/elastic-cache-policy/jetson/run_repeated_crossover_gsm8k.sh
```

This exploratory follow-up reuses exactly the same 60 registered GSM8K indices.
Each question is executed four times with the same generation seed:

- even question positions: `always, ensemble, ensemble, always` (`ABBA`);
- odd question positions: `ensemble, always, always, ensemble` (`BAAB`).

Each mode is summarized by its median of two repeats for that question. The
report bootstraps the 60 per-question median differences and separately records:

- end-to-end latency;
- total refresh cost;
- conditional refresh saving;
- refresh cost per decision;
- latency per decision;
- generation decision-count ratio;
- within-mode deterministic reproducibility.

This phase is explicitly post-hoc and exploratory. It can isolate an execution
signal but cannot replace the negative phase-4 verdict.

## Counterfactual metric

The local stale-loss target used in phases 1 to 3 is:

```text
0.5 * bounded cosine loss + 0.5 * bounded relative MSE
```

between cached and fully refreshed attention outputs for the same layer input.
The refresh cost is the normalized number of downstream layers recomputed after
a trigger.
