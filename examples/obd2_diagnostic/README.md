# OBD2 automotive diagnostic assistant (SciRust example)

A **small AI** (an MLP neural network) specialized in automotive diagnostics: it
reads an **OBD2 fault code** + a few symptoms and proposes the **most likely
root cause**, ranks all hypotheses by probability, and suggests the repair
action.

It is the same engine as the `quickstart_v2` demo (autograd `Tape`, `Linear`/
`ReLU` layers, `Adam` optimizer), but **specialized** for a trade.

## Running

**Simple version** (25 cases, 5 causes) — pedagogical demo:

```bash
cargo run -p obd2_diagnostic --release
```

**Massive version** (10,000 synthetic cases, 10 causes) — production training:

```bash
cargo run -p obd2_diagnostic --release --bin obd2_massive
```

**Ultra-massive version** (100,000 synthetic cases, 10 causes, very deep model) — large-scale challenge:

```bash
cargo run -p obd2_diagnostic --release --bin obd2_ultra
```

**MEGAVERSE version** (1,000,000 synthetic cases, 1000 causes, extreme classification) — the ultimate challenge:

```bash
cargo run -p obd2_diagnostic --release --bin obd2_megaverse       # 8 epochs
cargo run -p obd2_diagnostic --release --bin obd2_megaverse -- 3  # number of epochs of your choice
```

**REAL DATA version** (43,139 records from a 2012 Opel Corsa, fuel-mixture anomaly detection):

```bash
cargo run -p obd2_diagnostic --release --bin obd2_real                # defaults: committed CSV, 40 epochs
cargo run -p obd2_diagnostic --release --bin obd2_real -- <csv> <ep>  # CSV + epochs of your choice
```

## What the program does

1. **Encodes** each workshop situation into 7 numbers (the fault code + records:
   long-term fuel trim, MAF air flow, idle…).
2. **Trains** on 25 cases of "validated" repairs.
3. **Diagnoses** new cases: hypotheses ranked by % + action.

## The key idea: disambiguating the root cause

The same code (`P0171`, mixture too lean) can have **several** causes. The AI
learns that the **air flow (MAF)** separates the two most frequent ones:

| Code | Fuel trim | MAF air flow | → Predicted cause |
|------|-----------|--------------|-------------------|
| P0171 | +21 % (high) | **normal** | Air intake leak / vacuum leak |
| P0171 | +18 % (high) | **low** | Faulty MAF sensor |

Same code, same main symptoms — a single record changes the diagnosis.

## Adapting it to YOUR data

- **Add a root cause**: add an entry in `CAUSES` and `ACTIONS`, increase
  `N_CLASSES`, and provide examples in `training_data()`.
- **Add a symptom** (e.g. engine speed, temperature): increase `N_FEATURES` and
  add the column to each row.
- **Real data**: replace `training_data()` with your history of validated
  repairs (one case = features + confirmed cause).

## Massive version (production training)

The `obd2_massive` binary includes:
- **10,000+ synthetic cases** split into train/val/test
- **10 root causes** (instead of 5)
- **Realistic noise** (~2 % during training, ~8 % at test)
- **Deeper model**: 10 → 64 → 32 → 10
- **Performance metrics**: train/val/test accuracy

Results on synthetic data:
- Train accuracy: 99.88 %
- Best validation accuracy: 79.80 %
- Test accuracy: 56.60 % (566 / 1000 noisy cases)

The 56.6 % over 10 classes reflects the real separability of the generated
patterns. With real workshop data (stronger causal signatures), the results
would be better.

## MEGAVERSE version (1M cases × 1000 causes)

The `obd2_megaverse` binary pushes the framework to scale:
- **1,000,000 synthetic cases** (800K train / 100K val / 100K test)
- **1000 root causes**, each with a unique signature of 8 abnormal sensors
  (high/low) among 20 — uniqueness verified at generation
- **Mini-batches of 256** via the native multi-batch support (batched matmul +
  integer-label CrossEntropy): 3,125 autodiff graphs per epoch instead of
  800,000
- **Fisher-Yates shuffle** of the example order at each epoch
- **Noise**: ±0.03 at training, ±0.05 at test (harder)

Measured results (model 20 → 256 → 128 → 1000, ~167K parameters,
Adam lr=0.001, seed 42):

| Metric | Value |
|--------|-------|
| Generation of the 1M cases | 0.07 s |
| Training (3 epochs) | 157 s (~52 s/epoch) |
| Validation | **100.00 %** from epoch 1 |
| **Test (100,000 cases)** | **100.00 %** (100000/100000) |
| Random baseline | 0.10 % |

The 100 % is explained: each cause has a **unique signature, well separated**
from the noise (signal gap ~0.3-0.45 vs noise ±0.05). The network just has to
learn 1000 decision regions in a 20-dimensional space — which 800K examples
make possible. This is a demonstration of the **framework's capacity and
scaling** (1M cases, 1000 classes, minutes of compute), not a measure of real
diagnostic difficulty.

The v1 of this binary plateaued at ~0.1 %: colliding signatures (periodicity
modulo 20 → 20 signatures for 1000 causes), never-shuffled data (catastrophic
forgetting) and one autodiff graph per example (~9 h per epoch). The header
comment of `main_megaverse.rs` details the three fixes.

## REAL DATA version (`obd2_real`)

No more synthetic: the `obd2_real` binary trains on **real workshop
telemetry** — 43,139 records from an Opel Corsa 1.2 (2012) captured via an
ELM327 adapter (Hugging Face dataset
[`PedroCuisinier2025/OBD2_panel_opel_2012`](https://huggingface.co/datasets/PedroCuisinier2025/OBD2_panel_opel_2012),
CC-BY-4.0 license; the sample committed in `data/opel_corsa_telemetry.csv`
is 1 record out of every 5 of the original 394,406-line dataset).

**Principle**: the model learns the *healthy* relationship between 10 sensors
(RPM, MAF, engine load, O2 sensors, pressures/temperatures…) and the
**long-term fuel trim** (`LONG_FUEL_TRIM_1`). At diagnosis, a residual
|observed trim − predicted trim| beyond the threshold (p99 of the validation
residuals) signals a **mixture anomaly** — the P0171 logic of the first
example, this time learned on real data.

Measured results (split by driving segments, no temporal leakage):

| Metric | Value |
|--------|-------|
| Train / Val / Test | 28,538 / 7,139 / 7,462 records (distinct segments) |
| Baseline MAE (mean) | 6.61 % trim |
| **Model MAE (test)** | **2.74 % trim** |
| Anomaly threshold (p99) | ±8.85 % trim |
| Training | 1.5 s (40 epochs, batch 256) |
| Simulated air intake leak (+14 % trim) | ⚠ detected (residual 14.8 %) |

A real-data anecdote: this Opel shows an average long-term trim of **+14.4 %** —
the actual car probably has a small air intake leak itself or an aging MAF.
That is exactly the kind of signal the model learns to contextualize.

Honest limitation: corrupting *a single* sensor (e.g. MAF −35 %) is not always
detected on an isolated record — the correlated sensors (load, pressure)
"compensate" in the prediction. This is precisely what
`POST /trip/{id}/reading` (see API section below) addresses: a persistent but
subtle bias becomes statistically visible over several records of the same
trip, where an isolated record misses it.

## Saved weights (safetensors)

The trained models are serialized in `models/` in the **safetensors** format
via `scirust_core::io::safetensors::save_state_dict`:

- `models/obd2_real_fueltrim.safetensors` (12 KB) — weights + **embedded
  metadata**: feature names, normalization means/standard deviations, anomaly
  threshold, data source. The file is **self-sufficient**: a future diagnostic
  API needs only it.
- `models/obd2_megaverse.safetensors` (~660 KB) — weights of the 1000-cause
  classifier + metadata (architecture, seed, test accuracy).

The round-trip is verified at every run: reloading into a blank model via
`load_state_dict` → maximum prediction gap = 0.

## Diagnostic API (`obd2_api`)

The `obd2_api` binary exposes **both trained models** (fueltrim + megaverse)
through an HTTP API **with no external dependency whatsoever** (`std::net::TcpListener`
server, minimal hand-written JSON). Each model is loaded from its
self-sufficient safetensors file: the network architecture is reconstructed
from the tensor shapes, and the normalization and anomaly threshold come from
the embedded metadata — nothing is hard-coded server-side.

Automatically tested (`cargo test -p obd2_diagnostic --bin obd2_api`): JSON
parsing, sliding-threshold logic (deterministic, without a model), and both
models against their real committed weights — 17 tests.

```bash
cargo run -p obd2_diagnostic --release --bin obd2_api            # port 8080
cargo run -p obd2_diagnostic --release --bin obd2_api -- 9090    # chosen port
```

| Endpoint | Description |
|----------|-------------|
| `GET /health` | service status, loaded models |
| `GET /model` | expected features, target, threshold (fueltrim model) |
| `GET /model/megaverse` | architecture, classes, accuracy (megaverse model) |
| `POST /diagnose` | JSON sensor records → predicted trim, residual, verdict |
| `POST /diagnose/megaverse` | 20 raw features → top-3 predicted causes |
| `POST /trip/start` | starts a trip → `trip_id` |
| `POST /trip/{id}/reading` | adds a record to the trip → sliding residual |
| `GET /trip/{id}/status` | trip stats without adding a record |
| `POST /feedback` | workshop-confirmed case → archived (JSONL) for future retraining |

Example (real healthy record from the CSV):

```bash
curl -s localhost:8080/diagnose -d '{"RPM":1898,"SPEED":39,
  "THROTTLE_POS":23.53,"MAF":2.66,"COOLANT_TEMP":93,"INTAKE_TEMP":27,
  "O2_B1S1":0.625,"ENGINE_LOAD":5.49,"INTAKE_PRESSURE":26,
  "O2_B1S2":0.055,"LONG_FUEL_TRIM_1":17.97}'
```

```json
{"trim_observe_pct":17.97,"trim_predit_pct":17.49,"residu_pct":0.48,
 "seuil_pct":8.85,"anomalie":false,"verdict":"sain",
 "interpretation":"The observed trim is consistent with the engine state…"}
```

The same record with a trim inflated by +14 % (air-intake-leak symptom) returns
`"verdict":"anomalie_melange_pauvre"` with the classic P0171 suspects; an
abnormally low trim returns `"anomalie_melange_riche"` (P0172 logic). Missing
field → HTTP 400 with the name of the expected sensor.

### Sliding residual over a trip (`/trip/*`)

A persistent but subtle bias (e.g. a MAF slightly underestimating) can remain
below the detection threshold on an isolated record. `POST /trip/{id}/reading`
accumulates the **signed** residual record after record and compares its mean
to a threshold tightened in 1/√n (the record-independent noise cancels out in a
mean, unlike a systematic bias):

```bash
curl -s -X POST localhost:8080/trip/start                # → {"trip_id":1}
# then, at each record of the trip (same schema as /diagnose):
curl -s localhost:8080/trip/1/reading -d '{...same fields as /diagnose...}'
```

Tested in practice: a constant bias of +3 % (well below the pointwise threshold
of 8.85 %) stays `"anomalie":false` for 8 records, then flips to
`"anomalie":true` at the 9th record — the effective threshold (8.85/√n, floor
1.0) drops below 3 % exactly at that moment.

### Multi-model and workshop feedback

`POST /diagnose/megaverse` queries the 1000-cause classifier (scaling demo, see
above) with 20 raw features; the response is identical to the predictions
obtained during training (verified: `{"features":[…class 200 signature…]}` →
`"cause_id":200` at 100 %).

`POST /feedback` archives a confirmed case (records + `cause_confirmee` +
optional `notes`) in `data/feedback.jsonl`, timestamped, one JSON per line —
the starting base for the retraining described below.

## Retraining from feedback (`obd2_retrain`)

Closes the loop: `obd2_retrain` reloads the workshop CSV **and**
`data/feedback.jsonl`, then trains **twice** the same model (identical
architecture and hyperparameters, same never-touched test split) — once on the
CSV alone (baseline), once on CSV + feedback (augmented) — to honestly compare
the two MAEs rather than assuming that more data always improves the result.

```bash
cargo run -p obd2_diagnostic --release --bin obd2_retrain
# or with custom paths/epochs:
cargo run -p obd2_diagnostic --release --bin obd2_retrain -- <csv> <feedback.jsonl> <epochs>
```

Only feedback cases with the 10 sensors **and** the confirmed trim are usable
for this regression (`cause_confirmee` is free text, not a numeric target — it
will await a future cause classifier trained on labeled workshop history).
Without a feedback file (first run), the augmented training is simply identical
to the baseline — not an error.

The augmented model overwrites `models/obd2_real_fueltrim.safetensors` while
preserving exactly the metadata schema `obd2_api` depends on (features,
normalization, threshold, source), plus two traceability keys:
`feedback_cases_used` and `baseline_mae_no_feedback_pct`. Round-trip verified
at each run, as for `obd2_real`.

Verified in practice (30 records duplicated from the CSV as dummy feedback
cases, only to validate the pipeline mechanics — not a measure of real gain
since the model had already seen them): loading, merging, retraining, saving
and reloading by `obd2_api` work end to end, with the `feedback_cases_used: 30`
metadata correctly embedded.

## Honesty about limitations

The massive/ultra/megaverse versions remain **synthetic**: the AI there learns
generated patterns. The `obd2_real` version trains on real data, but from
**a single healthy car**: it detects mixture anomalies relative to the learned
norm, it does not classify 1000 root causes on real data (that would require a
workshop history labeled "confirmed cause"). This example is **educational &
training-oriented**, not a certified diagnostic tool.
