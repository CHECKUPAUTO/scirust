//! Bayes-factor theory ranking, end to end across the Invariant VIII boundary.
//!
//! `sos-theory` ranks rivals by a weight of evidence it refuses to compute;
//! this crate computes it from real likelihoods. Neither crate's own tests
//! can show the join — `sos-theory` cannot depend on a backend, and this
//! crate's unit tests stop at the number. Here both meet over a real store.

use scirust_stats::Poisson;
use sos_core::Author;
use sos_scirust::bayes::log10_bayes_factor;
use sos_store::{MemoryStore, TypedStore};
use sos_theory::{RankBasis, Scope, Theories, TheoryEngine, TheoryError, seal_theory};
use sos_theory::{Theory, WeightOfEvidence};

/// Two rivals over a shared domain, plus one that does not claim it.
fn world() -> (
    MemoryStore,
    sos_core::ObjectId,
    sos_core::ObjectId,
    sos_core::ObjectId,
) {
    let mut store = MemoryStore::new();
    let a = Author::engine("physics");
    let shared = Scope::from_predicates(["low-count-regime"]);

    // Distinct axioms, so the two rivals are genuinely different objects.
    // (Built identically they would share a content address and *be* the same
    // theory — which content addressing gets right, and which would make
    // "rank these two rivals" a meaningless request.)
    let axiom = |t: &[u8]| sos_core::ObjectId::compute(sos_core::HashAlgo::Sha256, b"axiom", t);
    let quiet = seal_theory(
        Theory::builder(shared.clone())
            .axioms(vec![axiom(b"rate-is-low")])
            .build(),
        a.clone(),
    );
    let loud = seal_theory(
        Theory::builder(shared)
            .axioms(vec![axiom(b"rate-is-high")])
            .build(),
        a.clone(),
    );
    // Claims a different domain entirely.
    let elsewhere = seal_theory(
        Theory::builder(Scope::from_predicates(["high-energy"])).build(),
        a,
    );
    store.put_object(&quiet).unwrap();
    store.put_object(&loud).unwrap();
    store.put_object(&elsewhere).unwrap();
    (store, quiet.id, loud.id, elsewhere.id)
}

fn scope() -> Scope {
    Scope::from_predicates(["low-count-regime"])
}

#[test]
fn real_data_ranks_the_theory_it_favours_first() {
    let (store, quiet, loud, _) = world();
    let engine = Theories::new(&store);

    // Counts clustered near 2 — far likelier under Poisson(2) than Poisson(10).
    let observed = [1_u64, 2, 2, 3, 1];
    let bf = log10_bayes_factor(&observed, &Poisson::new(2.0), &Poisson::new(10.0)).unwrap();
    assert!(
        bf.milli_bans > 0,
        "the data really does favour the quiet rival"
    );

    let ranking = engine
        .compare_by_evidence(
            &[quiet, loud],
            &scope(),
            &[
                WeightOfEvidence::new(quiet, bf.milli_bans),
                WeightOfEvidence::new(loud, -bf.milli_bans),
            ],
        )
        .unwrap();

    assert_eq!(ranking.basis, RankBasis::PosteriorOdds);
    assert_eq!(ranking.ranked[0].theory, quiet);
    assert_eq!(ranking.ranked[0].weight_milli_bans, Some(bf.milli_bans));
    assert_eq!(ranking.ranked[1].theory, loud);
}

#[test]
fn swapping_which_hypothesis_the_data_favours_swaps_the_ranking() {
    // The ranking tracks the evidence, not the argument order.
    let (store, quiet, loud, _) = world();
    let engine = Theories::new(&store);

    let noisy = [9_u64, 11, 10, 12];
    let bf = log10_bayes_factor(&noisy, &Poisson::new(2.0), &Poisson::new(10.0)).unwrap();
    assert!(bf.milli_bans < 0, "this data favours the loud rival");

    let ranking = engine
        .compare_by_evidence(
            &[quiet, loud],
            &scope(),
            &[
                WeightOfEvidence::new(quiet, bf.milli_bans),
                WeightOfEvidence::new(loud, -bf.milli_bans),
            ],
        )
        .unwrap();
    assert_eq!(ranking.ranked[0].theory, loud);
}

#[test]
fn a_rival_outside_the_scope_is_excluded_not_penalized() {
    // "Restricted to the shared domain": a theory that never claimed to apply
    // here is not losing an argument it did not enter, so it needs no weight
    // and does not appear.
    let (store, quiet, loud, elsewhere) = world();
    let engine = Theories::new(&store);

    let ranking = engine
        .compare_by_evidence(
            &[quiet, loud, elsewhere],
            &scope(),
            &[
                WeightOfEvidence::new(quiet, 500),
                WeightOfEvidence::new(loud, -500),
            ],
        )
        .unwrap();

    assert_eq!(ranking.ranked.len(), 2);
    assert!(ranking.ranked.iter().all(|r| r.theory != elsewhere));
}

#[test]
fn an_in_scope_rival_without_a_weight_is_an_error_not_a_zero() {
    // Absent is not neutral. Ranking `loud` as 0 millibans would assert the
    // evidence is balanced, which nobody computed.
    let (store, quiet, loud, _) = world();
    let engine = Theories::new(&store);

    assert!(matches!(
        engine.compare_by_evidence(
            &[quiet, loud],
            &scope(),
            &[WeightOfEvidence::new(quiet, 500)],
        ),
        Err(TheoryError::MissingWeight(id)) if id == loud
    ));
}

#[test]
fn a_weight_for_a_theory_that_is_not_a_rival_is_refused() {
    let (store, quiet, loud, elsewhere) = world();
    let engine = Theories::new(&store);

    assert!(matches!(
        engine.compare_by_evidence(
            &[quiet, loud],
            &scope(),
            &[
                WeightOfEvidence::new(quiet, 500),
                WeightOfEvidence::new(loud, -500),
                WeightOfEvidence::new(elsewhere, 100),
            ],
        ),
        Err(TheoryError::WeightForNonRival(id)) if id == elsewhere
    ));
}

#[test]
fn data_impossible_under_one_rival_is_decisive_and_ranks_it_last() {
    // Poisson has support on all of N, so use a distribution that genuinely
    // excludes the observation: Binomial(n=3) cannot produce a count of 9.
    use scirust_stats::Binomial;
    let (store, quiet, loud, _) = world();
    let engine = Theories::new(&store);

    let bf = log10_bayes_factor(&[9_u64], &Poisson::new(8.0), &Binomial::new(3, 0.5)).unwrap();
    assert!(bf.decisive, "the second rival is refuted by the data");
    assert_eq!(bf.milli_bans, i64::MAX);

    let ranking = engine
        .compare_by_evidence(
            &[quiet, loud],
            &scope(),
            &[
                WeightOfEvidence::new(quiet, bf.milli_bans),
                WeightOfEvidence::new(loud, i64::MIN),
            ],
        )
        .unwrap();
    assert_eq!(ranking.ranked[0].theory, quiet);
    assert_eq!(ranking.ranked[1].weight_milli_bans, Some(i64::MIN));
}

#[test]
fn the_evidential_balance_ranking_is_untouched_by_any_of_this() {
    // The pre-existing basis still works and still reports itself as such —
    // the new path is an addition, not a replacement.
    let (store, quiet, loud, _) = world();
    let engine = Theories::new(&store);
    let ranking = engine.compare(&[quiet, loud], &scope()).unwrap();
    assert_eq!(ranking.basis, RankBasis::EvidentialBalance);
    assert!(
        ranking.ranked.iter().all(|r| r.weight_milli_bans.is_none()),
        "no weight is invented for a basis that does not use one"
    );
}
