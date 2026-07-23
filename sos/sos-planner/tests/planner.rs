//! End-to-end planning: rank candidate experiments by utility, recommend ξ* or
//! report information exhaustion, seal a citable Plan, and evaluate stopping.

use sos_core::{Author, DeterminismLevel, HashAlgo, ObjectId};
use sos_planner::{
    Candidate, Cost, Estimate, GreedyPlanner, Planner, StopSignals, StopVerdict, StoppingRule,
    UtilityPolicy, seal_plan,
};

fn design(tag: &[u8]) -> ObjectId {
    ObjectId::compute(HashAlgo::default(), b"design", tag)
}

#[test]
fn ranks_by_utility_and_recommends_the_best() {
    let (a, b, c) = (design(b"A"), design(b"B"), design(b"C"));
    let candidates = [
        // A: cheap but uninformative.
        Candidate::new(
            a,
            Estimate::new(50, 10, DeterminismLevel::L2),
            Cost::new(1, 0, 0, 0),
        ),
        // B: informative, moderate cost — best utility.
        Candidate::new(
            b,
            Estimate::new(900, 50, DeterminismLevel::L2),
            Cost::new(2, 0, 0, 0),
        ),
        // C: very informative but very expensive.
        Candidate::new(
            c,
            Estimate::new(1500, 80, DeterminismLevel::L2),
            Cost::new(100, 0, 0, 0),
        ),
    ];

    let plan = GreedyPlanner::new()
        .recommend(&candidates, UtilityPolicy::EigPerCost, 10)
        .unwrap();

    // U_A = 50k, U_B = 450k, U_C = 15k ⇒ B, A, C.
    assert_eq!(plan.verdict, StopVerdict::Recommend(b));
    assert_eq!(
        plan.ranked.iter().map(|d| d.experiment).collect::<Vec<_>>(),
        vec![b, a, c]
    );
    assert_eq!(plan.best().unwrap().utility, 450_000);
}

#[test]
fn no_design_clearing_the_floor_is_information_exhausted() {
    let a = design(b"A");
    // 0.02 ± 0.03 bits — not significant, below any real floor.
    let candidates = [Candidate::new(
        a,
        Estimate::new(20, 30, DeterminismLevel::L1),
        Cost::new(1, 0, 0, 0),
    )];
    let plan = GreedyPlanner::new()
        .recommend(&candidates, UtilityPolicy::EigPerCost, 100)
        .unwrap();
    assert_eq!(plan.verdict, StopVerdict::InformationExhausted);
    assert!(!plan.recommends());
}

#[test]
fn a_high_utility_but_insignificant_design_does_not_get_recommended() {
    // A cheap, noise-dominated design has the highest raw utility yet must not be
    // recommended — its EIG is insignificant, so it teaches nothing.
    let (a, b) = (design(b"A"), design(b"B"));
    let candidates = [
        // A: U = 40*1000/1 = 40_000, but 40 ± 50 is insignificant and below floor.
        Candidate::new(
            a,
            Estimate::new(40, 50, DeterminismLevel::L1),
            Cost::new(1, 0, 0, 0),
        ),
        // B: U = 300*1000/10 = 30_000, significant and clears the floor.
        Candidate::new(
            b,
            Estimate::new(300, 20, DeterminismLevel::L2),
            Cost::new(10, 0, 0, 0),
        ),
    ];
    let plan = GreedyPlanner::new()
        .recommend(&candidates, UtilityPolicy::EigPerCost, 100)
        .unwrap();
    // A ranks first by raw utility, but B is the recommendation (A is below floor).
    assert_eq!(plan.ranked[0].experiment, a);
    assert_eq!(plan.verdict, StopVerdict::Recommend(b));
}

#[test]
fn budgeted_policy_excludes_over_budget_designs() {
    let (a, b) = (design(b"A"), design(b"B"));
    let candidates = [
        Candidate::new(
            a,
            Estimate::new(1500, 50, DeterminismLevel::L2),
            Cost::new(100, 0, 0, 0),
        ), // over budget
        Candidate::new(
            b,
            Estimate::new(400, 20, DeterminismLevel::L2),
            Cost::new(3, 0, 0, 0),
        ),
    ];
    // Budget 10: A (cost 100) is excluded; B wins despite lower raw EIG.
    let plan = GreedyPlanner::new()
        .recommend(&candidates, UtilityPolicy::EigBudgeted { budget: 10 }, 10)
        .unwrap();
    assert_eq!(plan.verdict, StopVerdict::Recommend(b));
    assert_eq!(plan.ranked[0].experiment, b);
    assert_eq!(plan.ranked[1].experiment, a); // present but excluded (utility EXCLUDED)
}

#[test]
fn planning_is_deterministic_and_the_plan_seals() {
    let (a, b) = (design(b"A"), design(b"B"));
    let candidates = [
        Candidate::new(
            a,
            Estimate::new(200, 10, DeterminismLevel::L2),
            Cost::new(1, 0, 0, 0),
        ),
        Candidate::new(
            b,
            Estimate::new(200, 10, DeterminismLevel::L2),
            Cost::new(1, 0, 0, 0),
        ),
    ];
    let planner = GreedyPlanner::new();
    let p1 = planner
        .recommend(&candidates, UtilityPolicy::EigPerCost, 10)
        .unwrap();
    let p2 = planner
        .recommend(&candidates, UtilityPolicy::EigPerCost, 10)
        .unwrap();
    assert_eq!(p1, p2); // deterministic
    // A tie in utility is broken by object id, deterministically.
    assert_eq!(p1.ranked[0].experiment, a.min(b));

    let obj = seal_plan(p1, Author::engine("sos-planner"));
    assert!(obj.verify_id());
    assert_eq!(obj.kind.name, "Plan");
}

#[test]
fn stopping_rules_compose_over_signals() {
    let signals = StopSignals {
        max_posterior_mass_milli: 995,
        best_eig_milli: 3,
        budget_spent: 10,
        budget_cap: 50,
    };
    // Stop if the leading hypothesis is near-certain OR information is exhausted.
    let rule = StoppingRule::Any(vec![
        StoppingRule::PosteriorMass {
            threshold_milli: 990,
        },
        StoppingRule::EigFloor { epsilon_milli: 5 },
    ]);
    assert!(rule.evaluate(&signals)); // posterior mass 995 >= 990
}
