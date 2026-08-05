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

Each mode is summarized by its median of two repeats for that question.

Observed result:

- all 60 questions are exactly deterministic within each mode;
- accuracy delta: `+1.6667` percentage points, 95% interval `[0, 5]` points;
- exact response match rate: `70%`;
- same prediction rate: `96.67%`;
- mean end-to-end latency improvement: `0.1840%`;
- latency 95% interval: `[-1.7366%, 2.0011%]`;
- mean total refresh-cost improvement: `0.5329%`;
- refresh-cost 95% interval: `[-1.4611%, 2.4059%]`;
- mean conditional refresh-cost saving: `7.3859%`, 95% interval
  `[6.7082%, 8.0998%]`;
- mean refresh-cost-per-decision improvement: `7.6675%`, 95% interval
  `[6.9708%, 8.4193%]`;
- mean latency-per-decision improvement: `7.3303%`, 95% interval
  `[6.6464%, 8.0719%]`;
- mean ensemble/always decision ratio: `1.0779`, 95% interval
  `[1.0574, 1.0989]`;
- controlled latency signal: false;
- controlled refresh-cost signal: false;
- exploratory controlled signal: false.

The crossover removes the execution-order ambiguity and confirms the mechanism:
about `7.3%` to `7.7%` less work per decision is offset by about `7.8%` more
decisions because the cache policy perturbs the generation trajectory. This
phase remains post-hoc and does not replace the phase-4 verdict.

## Phase 6: trajectory-stability guard development

```bash
bash experiments/elastic-cache-policy/jetson/run_trajectory_stability_guard_selection.sh
```

This offline phase does not use GSM8K outcomes and does not require a GPU. It
repurposes the phase-3 local counterfactual trace as development data and sweeps
runtime-implementable guards over:

- minimum skip-risk margin;
- minimum saved downstream refresh cost;
- maximum skips per attention layer;
- mandatory refreshed decisions after a skip.

Selection constraints are intentionally stricter than the original local proof:
aggregate, mean, and 90th-percentile trajectory stale-loss must remain below
`0.01`, and mean local compute improvement must remain at least `0.8%`. Eligible
guards are ranked by the fewest and least concentrated skipped decisions before
compute improvement.

Observed selection result:

- candidates evaluated: `120`;
- eligible candidates: `6`;
- minimum skip margin: `0.000017300677036841128`;
- minimum normalized refresh cost: `0.9642857142857143` (`27/28`);
- maximum skips per attention layer: `4`;
- cooldown decisions: `0`;
- skipped decisions reduced from `312` to `226`;
- mean skipped decisions per trajectory reduced from `5.2` to `3.7667`;
- maximum skipped decisions per trajectory reduced from `8` to `4`;
- aggregate stale-loss fraction: `0.0005645126`;
- mean trajectory stale-loss fraction: `0.0011382795`;
- 90th-percentile trajectory stale-loss fraction: `0.0016884127`;
- worst trajectory stale-loss fraction: `0.0111111111`;
- mean local compute improvement: `0.8951668%`;
- local quality criterion: pass;
- local compute criterion: pass.

Because the normalized refresh-cost threshold is exactly `27/28`, this guard
permits skips only in the first attention layer. The four-skip cap is maintained
per layer and reset for every generation. The phase-3 trace is now development
data for the guard and can no longer confirm it independently.

## Phase 7: frozen guard on independent GSM8K prompts

```bash
bash experiments/elastic-cache-policy/jetson/run_guarded_independent_gsm8k.sh
```

The selected guard is implemented exactly in the patched Dream runtime and
frozen before task evaluation. This phase:

- uses seed `20260808`;
- selects 60 GSM8K questions disjoint from all 60 phase-4/5 indices;
- performs two runs per mode and question in counterbalanced `ABBA/BAAB` order;
- uses the same generation seed for all four runs of a question;
- fits or calibrates neither the five policies nor the guard;
- preserves the phase-4 negative verdict;
- bootstraps the 60 per-question median differences with 10,000 samples.

Pre-registered success requires:

- exact-match non-inferiority within five percentage points, including the lower
  bootstrap bound;
- mean latency improvement at least `0.5%` with lower 95% bound above zero;
- mean total refresh-cost improvement at least `0.5%` with lower 95% bound above
  zero;
- exact deterministic reproducibility within each mode.

The primary field is `independent_guard_validation_success`. Exit status `2`
means a valid negative scientific result, not a runtime failure.

## Counterfactual metric

The local stale-loss target used in phases 1 to 3 is:

```text
0.5 * bounded cosine loss + 0.5 * bounded relative MSE
```

between cached and fully refreshed attention outputs for the same layer input.
The refresh cost is the normalized number of downstream layers recomputed after
a trigger.
