# TDI-1 — Reproducible results

## Hypothesis tested

The prospective structure of future accessible states provides additional predictive signal over Shannon entropy alone for predicting the recovery of a deterministic system after perturbation.

## Experimental protocol

- Finite deterministic networks
- Width: 3 bits
- Possible states per system: 8
- Training: 12,000 systems
- Independent holdout: 4,000 systems
- Entropy horizon: 8
- TDI prospective horizon: 4
- Maximum recovery horizon: 32
- Perturbation: flip of bit 2
- Deterministic paired bootstrap: 2,000 replications

## Holdout results

| Model | Accuracy | Balanced accuracy | Brier | AUPRC |
|---|---|---:|---:|---:|
| Entropy only | 0.716250 | 0.500000 | 0.200117 | 0.760675 |
| TDI return profile | 0.812000 | 0.706496 | 0.144390 | 0.827000 |
| Entropy + TDI | 0.814750 | 0.692721 | 0.141205 | 0.854758 |

## Observed gains

- AUPRC gain of the TDI profile over entropy: `+0.066325`
- AUPRC gain of the combined model: `+0.094083`
- Brier score improvement with TDI: `+0.055728`
- Combined Brier score improvement: `+0.058912`

## 95 % bootstrap confidence intervals

| Comparison | 95 % CI | Median |
|---|---:|---:|
| TDI AUPRC gain | [0.051310, 0.080562] | 0.066138 |
| TDI Brier improvement | [0.050331, 0.061267] | 0.055780 |
| Combined AUPRC gain | [0.081111, 0.106507] | 0.093957 |
| Combined Brier improvement | [0.053399, 0.064276] | 0.058929 |

## TDI-1 preregistered criterion

Success required simultaneously:

1. an observed TDI AUPRC gain of at least `0.05`;
2. a strictly positive lower bound of the 95 % CI of the AUPRC gain;
3. a strictly positive lower bound of the 95 % CI of the Brier improvement.

```text
TDI-1 PREREGISTERED CRITERION: PASSED
```

## Conclusion limited to the available evidence

On the synthetic family studied, the TDI prospective return profile contains predictive information that is not preserved by Shannon entropy alone.

The experiment notably establishes:

- the existence of systems with the same entropy but different recovery behaviors;
- the separation of thousands of opposite pairs by the TDI profile;
- a predictive gain on an independent holdout set;
- strictly positive paired confidence intervals.

## Limitations

TDI-1 does not yet demonstrate:

- a general fundamental law of information;
- biological, physical or quantum validity;
- superiority over all existing dynamical measures;
- generalization to continuous, stochastic or real systems;
- complete independence from the chosen perturbation protocol.

## Orbital dynamic control

A stronger baseline uses the transient lengths and periods of the reference and perturbed orbits.

| Model | Accuracy | Balanced accuracy | Brier | AUPRC |
|---|---|---:|---:|---:|
| Orbital baseline | 0.926250 | 0.870044 | 0.053284 | 0.984074 |
| Orbit + TDI | 0.926250 | 0.870044 | 0.053284 | 0.984074 |
| Entropy + orbit | 0.917000 | 0.860927 | 0.057089 | 0.983091 |
| Entropy + orbit + TDI | 0.917000 | 0.860927 | 0.057089 | 0.983091 |

The incremental gains of TDI against the orbital baseline are exactly zero:

- AUPRC gain: `0.000000`;
- Brier improvement: `0.000000`;
- 95 % bootstrap CI: `[0.000000, 0.000000]`.

```text
ORIGINAL TDI-1 CRITERION VS ENTROPY : PASSED
NOVELTY CONTROL VS ORBIT            : FAILED
```

TDI-1 therefore refutes the sufficiency of scalar entropy alone, but does not yet demonstrate a dynamical invariant independent of classical orbital properties.
