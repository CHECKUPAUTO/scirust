//! Deterministic contamination-matrix benchmark for SRCC trust policies.
//!
//! One exact-source fixture family (two mirrored transport views) is replayed
//! over a contamination matrix:
//!
//! - corrupted observation count 0–4 then 6 against 5 clean per view (count
//!   fractions 0–44 % and a strict 55 % corrupt majority);
//! - corrupted-weight concentration: unit priors everywhere, or corrupt
//!   observations down-weighted to `0.1` (count-majority / weight-minority);
//! - anchor state: clean anchors, or one anchor corrupted (moved onto the
//!   contaminant target);
//! - attack duration: burst (corrupt samples carry a single prediction step)
//!   versus persistent (corrupt samples carry a full prediction history).
//!
//! Each cell is evaluated under five policies (unweighted, per-group bound,
//! independent views, trusted anchors, temporal persistence). The output row
//! records the typed outcome — `accepted:<target>`, `unidentifiable:<k>`,
//! or `rejected:<error>` — plus certificate fields (effective trusted weight,
//! winning support, runner-up support, gated count).
//!
//! The matrix deliberately includes cells where **no policy identifies the
//! truth** (the 50/50 persistent split) and cells where the unweighted
//! consensus silently elects the contaminant (`accepted:bad`): the honest
//! failure surface, printed rather than hidden. A tiny deterministic
//! [`SrccTrustEvidenceProvider`] shows the adapter interface end to end.
//!
//! Output is deterministic (`{:.17e}` floats, no timestamps); run twice and
//! compare byte-for-byte (`cmp` / SHA-256).

use scirust_srcc::{
    SrccConfig, SrccObservationTrust, SrccTransportSample, SrccTrustError, SrccTrustEvidence,
    SrccTrustEvidenceKind, SrccTrustEvidenceProvider, SrccTrustModel, SrccTrustPolicy,
    SrccTrustProviderId, SrccTrustedFitError, Vector16, basis_vector, collect_trust_evidence,
    fit_trusted_robust_srcc_projector_from_views,
};

const CLEAN_PER_VIEW: usize = 5;

fn config() -> SrccConfig {
    SrccConfig {
        novelty_threshold: 1.0e-10,
        resonance_threshold: 0.999,
        minimum_support: 2,
        maximum_dimension: 2,
        maximum_rounds: 2,
        energy_floor: 1.0e-30,
    }
}

/// Deterministic provider demonstrating the adapter interface: it flags the
/// first `persistent_prefix` samples of every view with a full prediction
/// history and every later sample with a single (burst) step.
struct PersistenceProvider {
    persistent_prefix: usize,
    steps: usize,
}

impl SrccTrustEvidenceProvider for PersistenceProvider {
    fn provider_id(&self) -> SrccTrustProviderId {
        SrccTrustProviderId(7)
    }

    fn evidence_for(
        &self,
        _view_index: usize,
        sample_index: usize,
        _sample: &SrccTransportSample,
    ) -> Result<Vec<SrccTrustEvidence>, SrccTrustError> {
        let steps = if sample_index < self.persistent_prefix
        {
            self.steps
        }
        else
        {
            1
        };

        Ok((0..steps)
            .map(|step| SrccTrustEvidence {
                kind: SrccTrustEvidenceKind::TemporalPrediction,
                provider: self.provider_id(),
                score: 0.01 * (step as f64 + 1.0),
            })
            .collect())
    }
}

struct Fixture {
    source: Vector16,
    clean_target: Vector16,
    bad_target: Vector16,
    positive: Vec<SrccTransportSample>,
    negative: Vec<SrccTransportSample>,
}

/// Two mirrored views: `clean` clean repetitions first, then `bad` corrupt
/// repetitions (caller indices `0..clean` are clean).
fn fixture(clean: usize, bad: usize) -> Fixture {
    let source = basis_vector(1).expect("valid basis index");
    let clean_target = basis_vector(2).expect("valid basis index");
    let bad_target = basis_vector(8).expect("valid basis index");
    let negative_clean = clean_target.map(|value| -value);
    let negative_bad = bad_target.map(|value| -value);

    let mut positive = Vec::new();
    let mut negative = Vec::new();

    for _ in 0..clean
    {
        positive.push(SrccTransportSample::new(source, clean_target));
        negative.push(SrccTransportSample::new(source, negative_clean));
    }

    for _ in 0..bad
    {
        positive.push(SrccTransportSample::new(source, bad_target));
        negative.push(SrccTransportSample::new(source, negative_bad));
    }

    Fixture {
        source,
        clean_target,
        bad_target,
        positive,
        negative,
    }
}

fn trust_error_name(error: &SrccTrustError) -> String {
    match error
    {
        SrccTrustError::UnidentifiableContamination {
            competing_hypotheses,
        } => format!("unidentifiable:{competing_hypotheses}"),
        SrccTrustError::InsufficientAnchorSupport { .. } =>
        {
            "rejected:insufficient_anchor_support".to_string()
        },
        SrccTrustError::ConflictingAnchors { .. } => "rejected:conflicting_anchors".to_string(),
        SrccTrustError::AllObservationsUntrusted { .. } =>
        {
            "rejected:all_observations_untrusted".to_string()
        },
        SrccTrustError::InsufficientViews { .. } => "rejected:insufficient_views".to_string(),
        _ => "rejected:invalid_model".to_string(),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_cell(
    label: &str,
    fixture: &Fixture,
    policy: SrccTrustPolicy,
    observations: Vec<SrccObservationTrust>,
    bad: usize,
    weight_minority: bool,
    anchors_corrupted: bool,
    persistent_attack: bool,
) {
    let views = [fixture.positive.as_slice(), fixture.negative.as_slice()];

    let model = SrccTrustModel {
        policy,
        observations,
    };

    let outcome = match fit_trusted_robust_srcc_projector_from_views(
        &[fixture.source],
        &views,
        &model,
        config(),
    )
    {
        Ok(result) =>
        {
            let elected =
                scirust_srcc::apply_linear_map(&result.fit.transports[0], &fixture.source);

            let name = if elected == fixture.clean_target
            {
                "accepted:clean"
            }
            else if elected == fixture.bad_target
            {
                "accepted:bad"
            }
            else
            {
                "accepted:other"
            };

            let group = &result.certificate.groups[0];

            format!(
                "{name},{:.17e},{:.17e},{:.17e},{}",
                result.certificate.effective_trusted_weight,
                group.winning_support,
                group.runner_up_support,
                result.certificate.gated_observation_count,
            )
        },
        Err(SrccTrustedFitError::Trust(error)) =>
        {
            format!("{},nan,nan,nan,nan", trust_error_name(&error))
        },
        Err(SrccTrustedFitError::Fit(_)) => "rejected:fit,nan,nan,nan,nan".to_string(),
    };

    println!("{bad},{weight_minority},{anchors_corrupted},{persistent_attack},{label},{outcome}");
}

fn main() {
    println!("# trusted_consensus_contamination deterministic benchmark");
    println!("# fixture: two mirrored views, 5 clean + {{0..4,6}} corrupt observations per view");
    println!(
        "# columns: corrupt_count,weight_minority,anchors_corrupted,persistent_attack,policy,\
outcome,effective_trusted_weight,winning_support,runner_up_support,gated_count"
    );

    for bad in [0usize, 1, 2, 3, 4, 6]
    {
        // The corrupt observations are appended after the five clean ones, so
        // per-view counts are 5 clean + `bad` corrupt.
        let fixture = fixture(CLEAN_PER_VIEW, bad);
        let total = CLEAN_PER_VIEW + bad;

        for weight_minority in [false, true]
        {
            let priors = |sample_index: usize| {
                if weight_minority && sample_index >= CLEAN_PER_VIEW
                {
                    0.1
                }
                else
                {
                    1.0
                }
            };

            let plain_records = |with_anchor_on_bad: bool| {
                let mut records = Vec::new();

                for view_index in 0..2
                {
                    for sample_index in 0..total
                    {
                        let mut evidence = Vec::new();

                        // Anchors: the first two clean samples, or (when the
                        // anchor is corrupted) one clean and one corrupt.
                        let is_anchor = if with_anchor_on_bad
                        {
                            sample_index == 0 || sample_index == CLEAN_PER_VIEW
                        }
                        else
                        {
                            sample_index < 2
                        };

                        if is_anchor && (sample_index < total)
                        {
                            evidence.push(SrccTrustEvidence {
                                kind: SrccTrustEvidenceKind::TrustedAnchor,
                                provider: SrccTrustProviderId(1),
                                score: 1.0,
                            });
                        }

                        records.push(SrccObservationTrust {
                            view_index,
                            sample_index,
                            prior_weight: priors(sample_index),
                            evidence,
                        });
                    }
                }

                records
            };

            for anchors_corrupted in [false, true]
            {
                // Skip the anchor-corruption axis when there is no corrupt
                // observation to move the anchor onto.
                if anchors_corrupted && bad == 0
                {
                    continue;
                }

                for persistent_attack in [false, true]
                {
                    // Temporal evidence via the provider adapter: clean
                    // samples always persist; corrupt samples persist only in
                    // the persistent-attack rows.
                    let provider = PersistenceProvider {
                        persistent_prefix: if persistent_attack
                        {
                            total
                        }
                        else
                        {
                            CLEAN_PER_VIEW
                        },
                        steps: 3,
                    };

                    let views = [fixture.positive.as_slice(), fixture.negative.as_slice()];

                    let mut temporal_records = collect_trust_evidence(&views, &[&provider])
                        .expect("deterministic provider cannot fail");

                    for record in &mut temporal_records
                    {
                        record.prior_weight = priors(record.sample_index);

                        if anchors_corrupted
                        {
                            if record.sample_index == 0 || record.sample_index == CLEAN_PER_VIEW
                            {
                                record.evidence.push(SrccTrustEvidence {
                                    kind: SrccTrustEvidenceKind::TrustedAnchor,
                                    provider: SrccTrustProviderId(1),
                                    score: 1.0,
                                });
                            }
                        }
                        else if record.sample_index < 2
                        {
                            record.evidence.push(SrccTrustEvidence {
                                kind: SrccTrustEvidenceKind::TrustedAnchor,
                                provider: SrccTrustProviderId(1),
                                score: 1.0,
                            });
                        }
                    }

                    run_cell(
                        "unweighted",
                        &fixture,
                        SrccTrustPolicy::Unweighted,
                        Vec::new(),
                        bad,
                        weight_minority,
                        anchors_corrupted,
                        persistent_attack,
                    );

                    run_cell(
                        "group_bound_0_3",
                        &fixture,
                        SrccTrustPolicy::GroupContaminationBound {
                            maximum_corrupted_weight_per_group: 0.3,
                        },
                        plain_records(anchors_corrupted),
                        bad,
                        weight_minority,
                        anchors_corrupted,
                        persistent_attack,
                    );

                    run_cell(
                        "independent_views_0_3",
                        &fixture,
                        SrccTrustPolicy::IndependentViews {
                            minimum_consistent_views: 2,
                            maximum_corrupted_weight_per_view: 0.3,
                        },
                        plain_records(anchors_corrupted),
                        bad,
                        weight_minority,
                        anchors_corrupted,
                        persistent_attack,
                    );

                    run_cell(
                        "trusted_anchors_2",
                        &fixture,
                        SrccTrustPolicy::TrustedAnchors {
                            minimum_anchor_support: 2,
                        },
                        plain_records(anchors_corrupted),
                        bad,
                        weight_minority,
                        anchors_corrupted,
                        persistent_attack,
                    );

                    run_cell(
                        "temporal_persistence_3",
                        &fixture,
                        SrccTrustPolicy::TemporalPersistence {
                            minimum_consistent_steps: 3,
                            maximum_prediction_error: 0.1,
                        },
                        temporal_records,
                        bad,
                        weight_minority,
                        anchors_corrupted,
                        persistent_attack,
                    );
                }
            }
        }
    }

    // The unidentifiable-by-construction cell: a 50/50 persistent split with
    // equal weights. No policy exposed here can (or should) pick a side.
    let split = fixture(5, 5);
    let views = [split.positive.as_slice(), split.negative.as_slice()];
    let provider = PersistenceProvider {
        persistent_prefix: 10,
        steps: 3,
    };
    let records =
        collect_trust_evidence(&views, &[&provider]).expect("deterministic provider cannot fail");

    run_cell(
        "temporal_persistence_3",
        &split,
        SrccTrustPolicy::TemporalPersistence {
            minimum_consistent_steps: 3,
            maximum_prediction_error: 0.1,
        },
        records,
        5,
        false,
        false,
        true,
    );

    run_cell(
        "unweighted",
        &split,
        SrccTrustPolicy::Unweighted,
        Vec::new(),
        5,
        false,
        false,
        true,
    );
}
