# CCOS Core — Resolution Model

Resolution answers: *which claim is currently accepted, why, since when, based
on which source, under which policy?* (§11.6)

## Resolution states

| State | Meaning |
|---|---|
| accepted | `belief ≥ min_belief` and `conflict ≤ max_conflict` (validated) |
| rejected | `belief ≤ −min_belief` (refuted by the evidence fold) |
| temporary | validated under a decaying/eviction policy; re-evaluated each cycle |
| unresolved | `conflict` high — tension reported, no forced choice |
| contextualized | valid within a region/domain (Context Region Engine scope) |
| valid for an interval | temporal validity window (TEMPORAL_VALIDITY.md) |

## Mechanism

Resolution is a **fold with an explicit policy**:

1. the evidence set is fixed and enumerable;
2. the authority of each source is on the edge weight;
3. the policy (thresholds, decay, weights) is versioned **in the snapshot**
   (`scoring_weights` is serialised), so the *same* evidence under the *same*
   policy always yields the *same* resolution;
4. the outcome is explainable: `evidence_of` lists exactly which sources
   carried the decision and with what polarity.

## Honesty

When evidence is matched, `conflict` stays high and the claim is reported as
**unresolved**. Reporting genuine tension is a feature: it is the signal
similarity-only retrieval can never produce.
