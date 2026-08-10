use rand::rngs::StdRng;
use scirust_rsi::refine::{RefineTask, SelfRefiner};
use scirust_rsi::{Fitness, Guard};

struct SeededMixedStep;

impl RefineTask for SeededMixedStep
{
    type Solution = i64;

    fn initial(&self, _rng: &mut StdRng) -> Self::Solution
    {
        0
    }

    fn score(&self, solution: &Self::Solution) -> Fitness
    {
        *solution as Fitness
    }

    fn refine(&self, solution: &Self::Solution, rng: &mut StdRng) -> Self::Solution
    {
        use rand::Rng;
        let delta = if rng.gen_bool(0.55) { 2 } else { -3 };
        solution + delta
    }
}

struct AlwaysWorse;

impl RefineTask for AlwaysWorse
{
    type Solution = i64;

    fn initial(&self, _rng: &mut StdRng) -> Self::Solution
    {
        10
    }

    fn score(&self, solution: &Self::Solution) -> Fitness
    {
        *solution as Fitness
    }

    fn refine(&self, solution: &Self::Solution, _rng: &mut StdRng) -> Self::Solution
    {
        solution - 1
    }
}

#[test]
fn canonical_self_refiner_is_seed_reproducible()
{
    let guard = Guard::new().max_iters(64).patience(24);
    let (best_a, report_a) = SelfRefiner::new(0x5C1_2057).run(&SeededMixedStep, &guard);
    let (best_b, report_b) = SelfRefiner::new(0x5C1_2057).run(&SeededMixedStep, &guard);

    assert_eq!(best_a, best_b);
    assert_eq!(report_a.iterations, report_b.iterations);
    assert_eq!(report_a.accepted, report_b.accepted);
    assert_eq!(report_a.best_fitness, report_b.best_fitness);
    assert_eq!(report_a.history, report_b.history);
    assert_eq!(report_a.stop_reason, report_b.stop_reason);
    assert!(report_a.is_monotone());
}

#[test]
fn rejected_candidates_do_not_break_best_so_far_monotonicity()
{
    let guard = Guard::new().max_iters(8);
    let (best, report) = SelfRefiner::new(7).run(&AlwaysWorse, &guard);

    assert_eq!(best, 10);
    assert_eq!(report.iterations, 8);
    assert_eq!(report.accepted, 0);
    assert_eq!(report.best_fitness, 10.0);
    assert_eq!(report.history, vec![10.0; 8]);
    assert!(report.is_monotone());
}

#[test]
fn canonical_report_history_tracks_the_kept_incumbent()
{
    let guard = Guard::new().max_iters(32);
    let (_best, report) = SelfRefiner::new(42).run(&SeededMixedStep, &guard);

    assert_eq!(report.history.len(), report.iterations);
    assert_eq!(report.history.last().copied(), Some(report.best_fitness));
    assert!(report.history.windows(2).all(|pair| pair[1] >= pair[0]));
}
