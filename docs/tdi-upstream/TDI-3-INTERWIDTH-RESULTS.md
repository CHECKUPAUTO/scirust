# TDI-3 — Results of the preregistered inter-width evaluation

## Status

- **Primary criterion TDI-3A: FAILED**
- **Transfer criterion TDI-3B: FAILED**

This result is negative in the strict sense of the preregistered criteria.
It does not however mean that the TDI-3 variables are devoid of signal:
their effects are positive at some widths and negative or insufficiently
generalizable at others.

## Identity of the experiment

- branch: `tdi-3-interwidth`
- frozen scientific commit:
  `5c7fc5ecae57b8cfa10e6c400f9c03bbd030af4c`
- execution date: 12 July 2026
- duration: 118 seconds
- Rust: `rustc 1.97.0`
- architecture: Linux ARM64, Jetson AGX Thor
- result:
  `results/tdi-interwidth-continuous.log`
- SHA-256 of the result:
  `047a33708f691be20308fd51b9a428f98ee041ec1e5809516b5196de15ac7396`
- SHA-256 of the console:
  `633d36937f290bbe528e49111b609fff51166cfd5d40ee6cfa69992b53a12179`

The integrity of the preregistration, the evaluator, the scientific code and
the vendored dependencies was verified before execution.

## Populations

| Population | Retained systems | Excluded systems |
|---|---:|---:|
| training width 3 | 8,000 | 42 |
| holdout width 3 | 4,000 | 24 |
| training width 4 | 8,000 | 0 |
| holdout width 4 | 4,000 | 0 |
| OOD holdout width 5 | 4,000 | 0 |

The model was trained on 16,000 systems combining widths 3 and 4.
The criteria were evaluated on 8,000 in-distribution holdout systems and
4,000 out-of-distribution width-5 systems.

## Width 3 results

| Measure | Baseline | Baseline + TDI-3 |
|---|---:|---:|
| MSE | 0.002080792 | 0.001789824 |
| MAE | 0.018622116 | 0.017337850 |
| R² | 0.305464087 | 0.402584700 |
| Spearman | 0.591430188 | 0.697022323 |

- relative MSE reduction: **13.983526 %**
- MAE improvement: **0.001284266**
- 95 % CI of the MSE improvement:
  **[0.000195826, 0.000396785]**
- 95 % CI of the MAE improvement:
  **[0.000928742, 0.001645253]**

The TDI-3 variables here provide a clear improvement, statistically stable
and accompanied by better ranking quality.

## Width 4 results

| Measure | Baseline | Baseline + TDI-3 |
|---|---:|---:|
| MSE | 0.000025287 | 0.000039535 |
| MAE | 0.002883024 | 0.003628335 |
| R² | -22.139403817 | -35.176751612 |
| Spearman | 0.018318165 | 0.279302369 |

- relative MSE reduction: **-56.342626 %**
- MAE improvement: **-0.000745311**
- 95 % CI of the MSE improvement:
  **[-0.000017242, -0.000011408]**
- 95 % CI of the MAE improvement:
  **[-0.000870391, -0.000621827]**

The TDI-3 variables increase the ordinal signal, but significantly worsen
the MSE and MAE errors. The bootstrap intervals are entirely negative: this
is not a sampling fluctuation.

## Combined widths 3 and 4 holdout

| Measure | Baseline | Baseline + TDI-3 |
|---|---:|---:|
| MSE | 0.001053039 | 0.000914679 |
| MAE | 0.010752570 | 0.010483093 |
| R² | 0.324153855 | 0.412954221 |
| Spearman | 0.490755159 | 0.566936416 |

- relative MSE reduction: **13.139139 %**
- MAE improvement: **0.000269478**
- 95 % CI of the MSE improvement:
  **[0.000091703, 0.000190158]**
- 95 % CI of the MAE improvement:
  **[0.000079518, 0.000467571]**

The aggregated result is favorable and exceeds the 5 % threshold. The
TDI-3A criterion nevertheless fails, because the preregistration also
required a positive MSE improvement in each width. Width 4 violates this
condition in a statistically clear manner.

## Out-of-distribution width 5 transfer

| Measure | Baseline | Baseline + TDI-3 |
|---|---:|---:|
| MSE | 0.000767583 | 0.000483356 |
| MAE | 0.027293619 | 0.021295341 |
| R² | -251476.348326980 | -158357.366983186 |
| Spearman | -0.403040291 | -0.217928974 |
| mean bias | -0.027293619 | -0.021295341 |

- relative MSE reduction: **37.028775 %**
- MAE improvement: **0.005998278**
- 95 % CI of the MSE improvement:
  **[0.000279329, 0.000289017]**
- 95 % CI of the MAE improvement:
  **[0.005897616, 0.006096733]**

TDI-3 strongly reduces absolute errors and bias at width 5. However, the
R² and the Spearman correlation remain negative. The model therefore gets
closer to targets almost all near 1, without correctly learning their
individual variation. The TDI-3B criterion, which notably required positive
values of R² and Spearman, fails.

## Scientific interpretation

The results indicate three distinct facts:

1. **Real signal at width 3.**
   TDI-3 simultaneously improves error, ranking and explained variance.

2. **Instability between widths 3 and 4.**
   Width 4 presents a target extremely concentrated near 1. The TDI-3
   variables improve Spearman but worsen calibration and absolute errors.
   The learned signal is therefore not invariant to width under this
   formulation.

3. **Error reduction without structural generalization at width 5.**
   The 37 % MSE gain is real, but it comes mainly from a correction of the
   mean bias. The negative R² and Spearman values rule out concluding a
   successful structural prediction.

## Conclusion

TDI-3 does not validate the strong hypothesis of a universal inter-width
representation according to the preregistered criteria.

The experiment nevertheless provides constructive information: the TDI-3
features contain a predictive signal, but this signal is sensitive to width
and to the strong concentration of the target near 1. Any subsequent study
must be the subject of a new preregistration, without retroactively
modifying the TDI-3 verdict.
