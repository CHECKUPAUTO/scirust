# TDI-4 — Preregistered Target Geometry results

## Identity

- Branch: `tdi-4-target-geometry`
- Scientific freeze: `367bff80467d22f0aee0b3122de9dac893a0e0b3`
- CI of the freeze: `29233554774`
- Environment: Jetson AGX Thor, Linux AArch64
- Rust: `rustc 1.97.0`
- Duration: 148 seconds
- Raw result: `results/tdi-target-geometry.log`
- Complete console: `results/tdi4-first-complete-evaluation-console.log`
- SHA-256 of the result: `f1b67144b91f025e586cafc3a4b18b44360564055e46de2be1eecb18d04059a9`
- SHA-256 of the console: `11227c39c348d41a8514eed3eb8d0a48c181af8f554a660d931f1928f6f64ffc`

## Preregistered verdicts

    CRITÈRE PRINCIPAL TDI-4A : ÉCHOUÉ
    CRITÈRE TRANSFERT TDI-4B : ÉCHOUÉ

These verdicts are final for this protocol.

## Geometry of the target

No exact recovery at horizon 6 was observed.

| Population | Systems | Exact recoveries |
|---|---:|---:|
| Training widths 3 and 4 | 20,000 | 0 |
| Width-3 holdout | 5,000 | 0 |
| Width-4 holdout | 5,000 | 0 |
| OOD width-5 holdout | 5,000 | 0 |
| Total | 35,000 | 0 |

The binary-head target is therefore constant and equal to zero.

The baseline and the challenger both obtain a zero Brier score.
The Brier improvement and its bootstrap interval are exactly zero.

This condition formally prevents the success of TDI-4A and TDI-4B.

## Results of the continuous component

The conditional component uses:

\[
U=-\log_2(1-O_6).
\]

| Population | Baseline loss | TDI-4 loss | Reduction |
|---|---:|---:|---:|
| Width 3 | 0.241887657 | 0.179439894 | 25.816846 % |
| Width 4 | 0.085792849 | 0.065318515 | 23.864849 % |
| Combined 3 and 4 | 0.163840253 | 0.122379204 | 25.305777 % |
| OOD width 5 | 0.071703443 | 0.035827770 | 50.033403 % |

## Bootstrap of the composite loss

| Population | 95 % interval |
|---|---:|
| Width 3 | [0.055802305, 0.069284921] |
| Width 4 | [0.018695561, 0.022346104] |
| Combined 3 and 4 | [0.037685803, 0.044976080] |
| OOD width 5 | [0.034570495, 0.037174978] |

All the lower bounds of the composite loss improvement are strictly
positive.

## Width 3

| Conditional metric | Baseline | TDI-4 |
|---|---:|---:|
| MSE | 0.483775315 | 0.358879788 |
| R² | 0.450815004 | 0.592597247 |
| Spearman | 0.637248808 | 0.757689675 |

The reconstructed MSE decreases from 0.001712584 to 0.001497982.

The reconstructed MAE decreases from 0.011346318 to 0.010076199.

## Width 4

| Conditional metric | Baseline | TDI-4 |
|---|---:|---:|
| MSE | 0.171585697 | 0.130637029 |
| R² | 0.332284748 | 0.491633987 |
| Spearman | 0.565404261 | 0.696432192 |

The reconstructed MSE decreases from 0.000001282 to 0.000001099.

The reconstructed MAE decreases from 0.000479102 to 0.000426239.

## Out-of-distribution transfer — width 5

| Conditional metric | Baseline | TDI-4 |
|---|---:|---:|
| MSE | 0.143406887 | 0.071655540 |
| MAE | 0.318879657 | 0.217243811 |
| R² | -0.762078825 | 0.119549183 |
| Spearman | 0.467776309 | 0.615705212 |
| Bias | -0.284013178 | -0.146952367 |

| Reconstructed metric | Baseline | TDI-4 |
|---|---:|---:|
| MSE | 0.000000005 | 0.000000002 |
| MAE | 0.000061450 | 0.000038676 |
| R² | -0.641389205 | 0.243959397 |
| Spearman | 0.467776309 | 0.615705212 |
| Bias | -0.000050411 | -0.000019773 |

OOD bootstrap intervals:

- composite loss: [0.034570495, 0.037174978]
- conditional MSE: [0.069140990, 0.074349957]
- reconstructed MSE: [0.000000003, 0.000000003]
- reconstructed MAE: [0.000022019, 0.000023481]
- Brier: [0.000000000, 0.000000000]

## Analysis of the criteria

### TDI-4A

Passed conditions:

- combined composite reduction greater than 5 %;
- combined composite interval strictly positive;
- composite intervals of widths 3 and 4 strictly positive;
- point improvement of the reconstructed MSE;
- point improvement of the reconstructed MAE;
- positive conditional Spearman at widths 3 and 4.

Failed condition:

- lower bound of the Brier improvement strictly positive.

### TDI-4B

Passed conditions:

- OOD composite interval strictly positive;
- reconstructed MSE reduction greater than 5 %;
- reconstructed MSE interval strictly positive;
- challenger Spearman positive and greater than the baseline;
- challenger absolute bias lower than that of the baseline.

Failed condition:

- lower bound of the Brier improvement strictly positive.

## Scientific interpretation

The complete two-head protocol formally fails.

The exact-recovery head is degenerate in the studied populations, because
no positive example was observed.

The continuous deficit geometry is nevertheless strongly supported:

- improvement at widths 3 and 4;
- strictly positive composite intervals;
- large out-of-distribution transfer at width 5;
- positive and improved OOD Spearman;
- transition from a negative OOD R² to a positive R²;
- reduction of the reconstruction bias.

The retained interpretation is therefore:

1. the preregistered TDI-4 verdict remains negative;
2. the binary head is not identifiable in this regime;
3. the continuous component shows a robust signal;
4. the TDI variables provide cross-width predictive information;
5. a subsequent experiment must directly preregister the study of the
   continuous geometry, on new populations.
