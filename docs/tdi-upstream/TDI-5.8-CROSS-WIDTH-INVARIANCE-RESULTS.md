# TDI-5.8 — Cross-Width Invariance: Confirmatory Results

## Status

This document reports the single, real, preregistered TDI-5.8 run. The design
was frozen before execution
(`docs/TDI-5.8-CROSS-WIDTH-INVARIANCE-PREREGISTRATION.md`, SHA-256
`981dc709ae87f9191548bf6c31b4b0558b9550d196c6caa69220206101b9c0de`) and the run
was a deliberate one-time human action under the exact confirmation token. No
classification below may be rewritten.

**Headline.** The result splits cleanly in two, and the split is the finding.
**Within each width**, the overlap signal replicates completely: all 18
grid cells (widths 3, 4, 5 × horizons U₃…U₈) are *Beneficial*, and the effect
size is remarkably stable across widths — at U₆ the spread between widths is
0.6 percentage points. **Across widths**, transfer is another matter: a model
fitted at width 3 and applied to width 5 is badly **miscalibrated** at every
setting, with R² far below zero for both layouts. But it does not fail
uniformly. The overlaps preserve **rank ordering** across widths
(Spearman 0.695 at U₃) where the descriptor-only baseline preserves
**nothing** (Spearman −0.040). What transfers is the ordering, not the scale.

## 1. Provenance and integrity

| Item | Value |
|---|---|
| Run git commit | `4b6ef4796a3b326b9ea01a632626b27f4ed0c46c` |
| Evaluator (v58) SHA-256 | `e58d07e9ee01ab447be90fc90913661a0cbacd765e02f4670ada01965556f53a` |
| Preregistration SHA-256 | `981dc709ae87f9191548bf6c31b4b0558b9550d196c6caa69220206101b9c0de` |
| Scientific-manifest SHA-256 | `83d265843ce213adc9375a2f54b664e8a2e3362c171bede1559350e5d55bb926` |
| Result log SHA-256 | `d639d70ab0d2ee803a8b1c032537d990870f98b19661e44fdeb6a7e3a7157921` |
| Toolchain | rustc 1.97.1 (8bab26f4f 2026-07-14), cargo 1.97.1 (c980f4866 2026-06-30) |
| Host | Linux `tarek`, aarch64 (Jetson, ARM64) |
| Start / end (UTC) | 2026-07-29T18:54:32Z / 2026-07-29T21:20:18Z (2 h 26) |

`scripts/reproduce-tdi5.8.sh` verified the full frozen ancestor chain before
any generation and refuses a dirty repository. The committed log has been
independently rehashed in this working tree to
`d639d70ab0d2ee803a8b1c032537d990870f98b19661e44fdeb6a7e3a7157921`, matching
the value the run recorded for itself. Reproduction is byte-exact.

The run commit `4b6ef479` is the local commit that carried the TDI-6.4 run
artifacts; the working tree was clean and every frozen hash verified, so the
scientific code was intact.

## 2. Populations

TDI-5.8 is the widest experiment in the series: **nine seed blocks** — three
per width for widths 3, 4 and 5 — rather than the usual three, giving
**180,000 accepted records** (15,000 training + 5,000 holdout per block)
against 120,000 for every other experiment.

180,263 candidates were attempted for 180,000 accepted: **263 preregistered
exclusions**. Their distribution is itself informative:

| Width | Exclusions | Note |
|---|---:|---|
| 3 | 262 | 261 `observation-fully-recovered`, 1 `target-fully-recovered-h3` |
| 4 | 1 | a single `observation-fully-recovered` |
| 5 | **0** | none in any of the six populations |

Full recovery — the degeneracy that forces a candidate out — essentially stops
occurring once the state space is large enough. This is the same monotone
trend TDI-6.4 saw between widths 3 and 4, extended to width 5.

## 3. Criterion TDI-5.8A — in-width replication

SKT versus SK on combined holdout, per width, at the focal horizons:

| Width | U₃ | U₆ |
|---|---|---|
| 3 | **Beneficial** | **Beneficial** |
| 4 | **Beneficial** | **Beneficial** |
| 5 | **Beneficial** | **Beneficial** |

**`replication = yes`** — Beneficial at both focal horizons for all three
widths. Extending the dense grid to all six horizons, all **18 cells** are
Beneficial:

| Horizon | Width 3 | Width 4 | Width 5 |
|---|---:|---:|---:|
| U₃ | 47.01 % | 50.99 % | 50.52 % |
| U₄ | 36.29 % | 37.34 % | 37.33 % |
| U₅ | 29.34 % | 29.52 % | 28.82 % |
| U₆ | 23.98 % | 23.98 % | 23.37 % |
| U₇ | 20.11 % | 19.62 % | 19.37 % |
| U₈ | 17.20 % | 16.51 % | 16.00 % |

## 4. Criterion TDI-5.8C — the effect size is width-stable

| Horizon | min | max | spread | all three above the 2 % margin |
|---|---:|---:|---:|:---:|
| U₃ | 47.01 % | 50.99 % | 3.98 pp | yes |
| U₆ | 23.37 % | 23.98 % | **0.61 pp** | yes |

At U₆ the three widths agree to within six tenths of a percentage point. The
*relative* value of the overlaps is therefore close to width-invariant, which
is a genuinely strong form of the invariance the experiment set out to test —
and it makes the transfer failure below more, not less, interesting.

## 5. Criterion TDI-5.8B — transfer: calibration dies, ordering survives

A model fitted on width 3 and evaluated on width 5, aggregate:

| Horizon | Layout | MSE | R² | Spearman |
|---|---|---:|---:|---:|
| U₃ | SK (descriptors only) | 61.046 | −437.80 | **−0.040** |
| U₃ | SKT (+ overlaps) | 0.392 | −1.82 | **0.695** |
| U₆ | SK | 116.864 | −1276.58 | **−0.063** |
| U₆ | SKT | 18.238 | −198.38 | **0.351** |

Both criteria classify as *Beneficial*, and that label needs care: **it does
not mean transfer works.** Every R² here is far below zero, so both layouts
are worse than predicting the target's mean — the fitted scale simply does not
carry from width 3 to width 5. Reporting "Beneficial" without that context
would invert the finding.

But the Spearman column shows the label is not empty either. The
descriptor-only baseline transfers **no usable information at all**: its rank
correlation is −0.040 at U₃ and −0.063 at U₆, i.e. indistinguishable from
zero and slightly the wrong way. The overlaps transfer **substantial ordering
information**: 0.695 at U₃, still 0.351 at U₆. In reconstructed-O space the
baseline collapses entirely (R² −35031, Spearman exactly 0.000000000).

> **Correction, added while preparing TDI-6.8.** The last clause above is
> misleading and is corrected here rather than removed. That `Spearman exactly
> 0.000000000` is **not a measurement of lost ordering**: the same block's log
> line reports `fraction borne basse = 1.0`, meaning **every** prediction was
> clamped to the floor of the representable O range. The predictions are
> therefore a constant, and a constant has no ranks — `ρ = 0` follows by
> definition, whatever the underlying ordering was. The reconstruction
> `O_h = 1 − 2^(−U_h)` is strictly increasing and so *would* preserve rank
> exactly, but the saturation applied on top of it is monotone without being
> strictly increasing, and it manufactures ties. The correct reading of the
> zero is that the baseline's predicted levels fall entirely outside the
> representable range — which is a genuine and severe failure, but a failure of
> *level*, already reported by the `R²` beside it, not an independent failure
> of ordering.
>
> The Spearman values quoted elsewhere in this section (−0.040, −0.063, 0.695,
> 0.351) are standardized-U values and are unaffected by the saturation. They
> are, however, **aggregate** values, and TDI-6.7 §9.1 later showed that pooled
> rank statistics in this design are not trustworthy as criterion inputs. They
> were never criterion inputs here — TDI-5.8B classifies on relative MSE — and
> they are left as reported. TDI-6.8 is the experiment that puts a
> **per-block** rank statistic under a preregistered criterion; it forbids both
> the pooled and the reconstructed-O rank statistic from entering one, so
> neither reading can recur.

So the honest statement is three-part: cross-width **calibration** fails for
everything; cross-width **ranking** survives via the overlaps and not via the
exact descriptors; and the surviving ordering **degrades with horizon**
(0.695 → 0.351 from U₃ to U₆), as does the calibration (R² −1.8 → −198).

This confirms at scale what TDI-5.2D saw as a weak signal on width 4, and it
locates the failure precisely: not in the signal, but in the mapping from
signal to absolute deficit level.

## 6. Criterion TDI-5.8D — why transfer fails

Mean exact descriptor values by width:

| Descriptor | Width 3 | Width 4 | Width 5 | Range |
|---|---:|---:|---:|---:|
| δ | 0.955636 | 0.942474 | 0.880884 | 0.074752 |
| δ̄ | 0.594437 | 0.569776 | 0.549706 | 0.044732 |
| s₂ | 1.169223 | 1.071486 | 1.033160 | **0.136063** |
| s₃ | 1.031756 | 1.002611 | 1.000329 | 0.031428 |

All four drift **monotonically downward** with width, `s₂` most of all. A model
fitted at width 3 is therefore extrapolating outside its feature distribution
when applied at width 5, which is a sufficient mechanism for the calibration
failure of Section 5 without invoking anything about the overlaps themselves.

This is offered as a consistent explanation, not a demonstrated cause: the
experiment measures the drift and measures the transfer failure, but does not
test the link between them — that would need a further design (for example,
re-standardising with width-5 statistics before transfer).

## 7. Interpretation and boundaries

TDI-5.8 closes the most-repeated limitation of the series — "widths 3–4 only"
— in the direction that matters: the signal itself is width-robust, and its
*relative* magnitude is nearly width-invariant. It simultaneously establishes
that the program's models are **not** transportable across widths in absolute
terms.

What this does **not** establish:

- **transportable prediction.** Nothing here licenses fitting at one width and
  predicting deficit levels at another. The correct use of a cross-width model
  is ranking, not calibration, and even the ranking degrades with horizon;
- widths beyond 5 (6 remains untested at scale); a single generator;
  linear ridge machinery; no causal claim (TDI-6.4); no information
  decomposition (TDI-6.3). No universal law is claimed, and nothing here was
  tested outside small synthetic finite-state families.

The TDI-5.8A / TDI-5.8B / TDI-5.8C / TDI-5.8D summaries are frozen as
reported.

## 8. Reproduction

    TDI58_CONFIRM_FULL_RUN=I_ACCEPT_THE_TDI58_FREEZE_RULE \
      bash scripts/reproduce-tdi5.8.sh

The script refuses without the exact token, refuses a dirty repository,
verifies the full frozen hash chain before any generation, executes the
evaluator once with `--full`, verifies the final criterion lines, and writes
read-only artifacts under `results/tdi5.8-cross-width-invariance/`.
Determinism is exact: the log SHA-256
`d639d70ab0d2ee803a8b1c032537d990870f98b19661e44fdeb6a7e3a7157921` is
reproduced by any faithful re-run on the same toolchain and architecture.
Width 5 dominates the wall-clock; budget roughly 2.5 hours.
