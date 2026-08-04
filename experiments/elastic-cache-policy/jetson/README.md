# Dream real-policy proof on Jetson AGX Thor

This protocol tests whether the synthetic `63.110794%` refresh-compute gain from
SciRust policy discovery transfers to real Dream 7B attention states.

## Evidence produced

For every reused attention layer, the probe executes two paths from the same
layer input state:

1. the competitor's cached K/V reuse path;
2. an exact full K/V refresh path.

It records the competitor's cosine trigger plus seven additional signals and a
bounded divergence between the two attention outputs. SciRust then fits the
multi-signal policy on complete training trajectories, calibrates it on complete
validation trajectories, and compares it with the best fixed `gamma` on held-out
trajectories.

The run is considered a reproduction of the synthetic result only when:

- both policies satisfy the same `0.05` stale-loss budget;
- the SciRust policy strictly Pareto-dominates the best fixed `gamma`;
- the measured refresh-compute improvement lies within `63.110794% ± 8%`.

This is a real Dream hidden-state and attention-output result. It is not yet a
GSM8K, HumanEval, or end-to-end task-accuracy claim.

## Jetson target

- NVIDIA Jetson AGX Thor
- Ubuntu 24.04
- CUDA 13.0
- 128 GB unified memory
- model: `Dream-org/Dream-v0-Instruct-7B`
- model size: approximately 15.2 GB in BF16

The setup creates a virtual environment with `--system-site-packages` so the
NVIDIA-provided Jetson PyTorch build is preserved. Do not install a generic
PyPI `torch` wheel into this environment.

## Run

From the SciRust repository:

```bash
bash experiments/elastic-cache-policy/jetson/run_jetson_dream_proof.sh
```

The command checks CUDA, downloads the competitor and Dream checkpoint, collects
30 deterministic trajectories, invokes the Rust discovery binary, and writes:

```text
~/.local/share/scirust/dream-policy-proof/results/<UTC timestamp>/
├── dream_counterfactual_trace.csv
├── scirust_discovery_output.txt
└── dream_real_policy_report.json
```

Exit status:

- `0`: the 63.11% reproduction band is satisfied;
- `2`: the real Dream result is valid but does not reproduce the band;
- another nonzero status: environment, model, trace, or discovery failure.

## Metric boundary

The stale-loss target is:

```text
0.5 * bounded cosine loss + 0.5 * bounded relative MSE
```

between cached and fully refreshed attention outputs for the same layer input.
The refresh cost is the normalized number of downstream layers that must be
recomputed after a trigger.

A subsequent paper-grade phase must repeat the frozen comparison on task-level
metrics and independent datasets.
