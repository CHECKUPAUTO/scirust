//! **Phase C, kernel 2: quaternion orientation averaging.**
//!
//! (`docs/research/ANEE_ADAPTIVE_NUMERICAL_EXECUTION_ENGINE_2026-07-17.md`,
//! §13/§14 "replicate on 1–2 more kernels before generalizing" — the honest
//! next step after `crate::representation_graph`'s first kernel (scalar
//! compress-then-aggregate) survived its own kill criterion.) This module
//! tests whether the same finding — joint `(representation, accumulation)`
//! search beats sequential per-axis selection — replicates on a
//! *structurally different* kernel: hypercomplex orientation averaging
//! ([ATRA]'s own X5 experiment, `docs/research/atra_experiments/
//! atra_quaternion.py`), using [`scirust_simd::geometry::quaternion::Quaternion`]
//! (a real, generic, deterministic quaternion type already in this
//! workspace — not reimplemented here) rather than plain scalars.
//!
//! ## The task
//!
//! Average `N` noisy unit-quaternion observations of a fixed true
//! orientation (isotropic Gaussian-angle noise around a random axis per
//! sample, matching [ATRA X5]'s protocol) and measure the angular error
//! (degrees) of the estimated mean against the true orientation. Two charts
//! (the representation axis `R`, here called `Chart`) are compared:
//!
//! * [`Chart::Componentwise`] — average the raw `(w,x,y,z)` components in
//!   ambient `ℝ⁴`, then renormalize. [ATRA X5]'s "componentwise mean +
//!   renorm" baseline: known to degrade badly at high noise (hemisphere
//!   wrapping).
//! * [`Chart::LogChart`] — a fixed-iteration Karcher-style tangent-space
//!   mean: log-map each sample relative to a reference orientation via
//!   [`Quaternion::to_axis_angle`], average the tangent vectors, exp-map
//!   back via [`Quaternion::from_axis_angle`], repeat. [ATRA X5]'s "Karcher
//!   mean (log/exp chart)".
//!
//! [ATRA X5] additionally tested a third method (the chordal/Markley mean,
//! the largest-eigenvector of `Σqᵢqᵢᵀ`) that is **not** replicated here — it
//! needs a symmetric eigensolver this prototype does not have reason to add
//! just for this comparison, and the report's own discipline is to stay
//! narrow, not to acquire new machinery per kernel. This is reported as a
//! partial (2-of-3-method), not full, replication of ATRA X5.
//!
//! The **accumulation** axis `A` reuses [`crate::autotune_accumulate::AccumMethod`]
//! unchanged, applied to whichever three-or-four scalar component arrays
//! each chart produces per averaging step (narrowed to `f32` for the
//! accumulation itself, exactly [`crate::autotune_accumulate`]'s own
//! "compute wide, store narrow" pattern, reused here rather than
//! re-derived).
//!
//! ## Pre-registered kill criterion
//!
//! Written before this benchmark was run, using the **same bar** as kernel 1
//! ([`crate::representation_graph`]) for direct methodological consistency
//! across replications, not tuned per kernel:
//!
//! > On held-out data (3 fresh seeds beyond the dev/eval draw used for
//! > selection), joint `(Chart, AccumMethod)` search must reduce mean
//! > angular error versus the sequential baseline (always
//! > [`Chart::Componentwise`] — the cheapest, simplest default, mirroring
//! > kernel 1's `Identity`-always-wins cost tie-break — then `A` selected
//! > with the chart held fixed) by **at least 20% relative** on **at least
//! > 2 of the 3** tested noise levels (`σ ∈ {0.2, 0.8, 1.5}` rad, matching
//! > [ATRA X5]'s exact levels). If this is not met, the replication
//! > **fails** for this kernel, and that is reported as a genuine,
//! > equally-informative non-replication — not explained away.

use crate::autotune_accumulate::{AccumMethod, accumulate};
use crate::transform_autotune::{GenericAutotune, autotune_by};
use rand::Rng;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use scirust_simd::geometry::quaternion::Quaternion;

/// This module always works in `f64` (the chart math is deliberately kept
/// wide; only the accumulation step is narrowed to `f32` — see module docs).
type Quat = Quaternion<f64>;

// ---------------------------------------------------------------------------
// The chart (representation) axis
// ---------------------------------------------------------------------------

/// A representation ("chart") for averaging unit-quaternion samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Chart {
    /// Ambient `ℝ⁴` componentwise mean, then renormalize.
    Componentwise,
    /// Tangent-space (log/exp) mean, re-linearized `iterations` times.
    LogChart {
        /// Number of Gauss-Newton-style re-linearization passes.
        iterations: u32,
    },
}

impl Chart {
    /// Human-readable name.
    pub fn name(self) -> String {
        match self
        {
            Chart::Componentwise => "componentwise".to_string(),
            Chart::LogChart { iterations } => format!("log-chart(iters={iterations})"),
        }
    }

    /// Cost proxy: `Componentwise` is cheapest (no trigonometric inverse per
    /// sample); `LogChart` costs one `acos`+`sqrt` pair per sample per
    /// iteration.
    pub fn cost(self) -> u32 {
        match self
        {
            Chart::Componentwise => 0,
            Chart::LogChart { iterations } => 4 * iterations,
        }
    }
}

/// The default chart dictionary: both members described in the module docs.
pub fn default_chart_dictionary() -> Vec<Chart> {
    vec![Chart::Componentwise, Chart::LogChart { iterations: 2 }]
}

// ---------------------------------------------------------------------------
// Quaternion averaging pipeline
// ---------------------------------------------------------------------------

/// Align every sample's sign to the first sample's hemisphere
/// (`dot(q_i, q_0) >= 0`) — unit quaternions `q` and `-q` represent the same
/// rotation, and naive averaging without this step can catastrophically
/// cancel (the well-known "hemisphere wrapping" failure [ATRA X5]'s own
/// componentwise-mean baseline exhibits at high noise).
fn align_hemisphere(samples: &[Quat]) -> Vec<Quat> {
    let reference = samples[0];
    samples
        .iter()
        .map(|&q| {
            if q.dot(reference) < 0.0
            {
                q.scale(-1.0)
            }
            else
            {
                q
            }
        })
        .collect()
}

/// Log map of `q` relative to `reference` (both assumed unit): the tangent
/// vector at `reference` whose exp map recovers `q`, taking the shorter arc
/// (`d.w >= 0`) so the tangent vector's norm never exceeds `π`.
fn log_relative(reference: Quat, q: Quat) -> [f64; 3] {
    let d = reference.conjugate().mul_quat(q);
    let d = if d.w < 0.0
    {
        Quat::new(-d.w, -d.x, -d.y, -d.z)
    }
    else
    {
        d
    };
    let (axis, angle) = d.to_axis_angle();
    [axis[0] * angle, axis[1] * angle, axis[2] * angle]
}

/// Exp map of tangent vector `v` at `reference`, inverse of [`log_relative`].
fn exp_relative(reference: Quat, v: [f64; 3]) -> Quat {
    let angle = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let delta = if angle < 1e-12
    {
        Quat::identity()
    }
    else
    {
        Quat::from_axis_angle([v[0] / angle, v[1] / angle, v[2] / angle], angle)
    };
    reference.mul_quat(delta)
}

/// Accumulate three or four parallel component arrays via `accum` (narrowed
/// to `f32` for the accumulation step, widened back to `f64` after), then
/// divide by `n` — the shared "mean of components" building block both
/// charts use.
fn accumulate_mean(accum: AccumMethod, columns: &[Vec<f64>]) -> Vec<f64> {
    let n = columns[0].len() as f64;
    columns
        .iter()
        .map(|col| {
            let narrow: Vec<f32> = col.iter().map(|&v| v as f32).collect();
            accumulate(accum, &narrow) as f64 / n
        })
        .collect()
}

/// Average `samples` (assumed unit quaternions, at least one) using `chart`
/// for representation and `accum` for combining component/tangent arrays.
pub fn average(chart: Chart, accum: AccumMethod, samples: &[Quat]) -> Quat {
    let aligned = align_hemisphere(samples);
    match chart
    {
        Chart::Componentwise =>
        {
            let cols = vec![
                aligned.iter().map(|q| q.w).collect(),
                aligned.iter().map(|q| q.x).collect(),
                aligned.iter().map(|q| q.y).collect(),
                aligned.iter().map(|q| q.z).collect(),
            ];
            let m = accumulate_mean(accum, &cols);
            Quat::new(m[0], m[1], m[2], m[3]).normalize()
        },
        Chart::LogChart { iterations } =>
        {
            let n = aligned.len() as f64;
            let sw: f64 = aligned.iter().map(|q| q.w).sum::<f64>() / n;
            let sx: f64 = aligned.iter().map(|q| q.x).sum::<f64>() / n;
            let sy: f64 = aligned.iter().map(|q| q.y).sum::<f64>() / n;
            let sz: f64 = aligned.iter().map(|q| q.z).sum::<f64>() / n;
            let mut est = Quat::new(sw, sx, sy, sz).normalize();
            for _ in 0..iterations
            {
                let tangents: Vec<[f64; 3]> =
                    aligned.iter().map(|&q| log_relative(est, q)).collect();
                let cols = vec![
                    tangents.iter().map(|v| v[0]).collect(),
                    tangents.iter().map(|v| v[1]).collect(),
                    tangents.iter().map(|v| v[2]).collect(),
                ];
                let m = accumulate_mean(accum, &cols);
                est = exp_relative(est, [m[0], m[1], m[2]]);
            }
            est
        },
    }
}

/// Angular error (degrees) between two unit quaternions, robust to the
/// double-cover sign ambiguity (`|dot|`, clamped to `[-1,1]` for `acos`
/// safety against float round-off).
pub fn angular_error_degrees(a: Quat, b: Quat) -> f64 {
    let d = a.dot(b).abs().min(1.0);
    2.0 * d.acos() * (180.0 / core::f64::consts::PI)
}

// ---------------------------------------------------------------------------
// Noisy-sample generation
// ---------------------------------------------------------------------------

/// Draw a uniformly random unit vector on `S²` via 3 standard normals
/// (Marsaglia's method for the sphere).
fn random_unit_axis(rng: &mut StdRng) -> [f64; 3] {
    loop
    {
        let v = [
            rng.sample::<f64, _>(StandardNormal),
            rng.sample::<f64, _>(StandardNormal),
            rng.sample::<f64, _>(StandardNormal),
        ];
        let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if n2 > 1e-18
        {
            let inv = n2.sqrt().recip();
            return [v[0] * inv, v[1] * inv, v[2] * inv];
        }
    }
}

/// One trial: `n` noisy unit-quaternion observations of `truth`, each
/// perturbed by an independent rotation of angle `~ Normal(0, sigma)`
/// (isotropic, "von-Mises-like" per [ATRA X5]) about a uniformly random
/// axis.
pub fn noisy_trial(truth: Quat, sigma: f64, n: usize, rng: &mut StdRng) -> Vec<Quat> {
    (0..n)
        .map(|_| {
            let axis = random_unit_axis(rng);
            let angle: f64 = rng.sample::<f64, _>(StandardNormal) * sigma;
            let perturbation = Quat::from_axis_angle(axis, angle);
            perturbation.mul_quat(truth)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Sequential and joint search over (Chart, AccumMethod)
// ---------------------------------------------------------------------------

/// A chosen `(chart, accumulation)` pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuatPlan {
    /// Chosen chart.
    pub chart: Chart,
    /// Chosen accumulation strategy.
    pub accumulation: AccumMethod,
}

/// Score `plan` against `truth`: negative mean angular error (degrees) over
/// `score_on`'s trials (higher is better, matching [`autotune_by`]'s
/// convention). `fit` is unused — chart/accumulation have no fitted
/// parameter, exactly [`crate::autotune_accumulate`]'s own accumulation-only
/// autotune ("Accumulation has no fitted parameter, so the harness's 'fit'
/// set is unused").
pub fn score_plan(
    plan: QuatPlan,
    truth: Quat,
    _fit: &[Vec<Quat>],
    score_on: &[Vec<Quat>],
) -> Option<f64> {
    if score_on.is_empty()
    {
        return None;
    }
    let mean_err: f64 = score_on
        .iter()
        .map(|trial| {
            let est = average(plan.chart, plan.accumulation, trial);
            angular_error_degrees(est, truth)
        })
        .sum::<f64>()
        / score_on.len() as f64;
    Some(-mean_err)
}

/// Outcome of one search approach.
#[derive(Debug, Clone, Copy)]
pub struct QuatSearchReport {
    /// The chosen plan.
    pub plan: QuatPlan,
    /// Mean angular error (degrees) on the eval set used for selection.
    pub eval_mean_error_degrees: f64,
}

/// Sequential baseline: always [`Chart::Componentwise`] (cost 0, the
/// simplest default — mirrors kernel 1's `Identity`-always-wins cost
/// tie-break exactly), then `A` selected via [`autotune_by`] with the chart
/// held fixed.
pub fn sequential_baseline(
    truth: Quat,
    dev: &[Vec<Quat>],
    eval: &[Vec<Quat>],
    a_dict: &[AccumMethod],
) -> Option<QuatSearchReport> {
    let chart = Chart::Componentwise;
    let score = move |a: AccumMethod, fit: &[Vec<Quat>], scr: &[Vec<Quat>]| {
        score_plan(
            QuatPlan {
                chart,
                accumulation: a,
            },
            truth,
            fit,
            scr,
        )
    };
    let baseline = move |fit: &[Vec<Quat>], scr: &[Vec<Quat>]| {
        score_plan(
            QuatPlan {
                chart,
                accumulation: AccumMethod::NaiveF32,
            },
            truth,
            fit,
            scr,
        )
        .unwrap_or(f64::NEG_INFINITY)
    };
    let out: GenericAutotune<AccumMethod> = autotune_by(dev, eval, a_dict, score, baseline);
    let a = out.chosen?;
    Some(QuatSearchReport {
        plan: QuatPlan {
            chart,
            accumulation: a,
        },
        eval_mean_error_degrees: -out.chosen_eval_score,
    })
}

/// Joint search: the same [`autotune_by`] harness fed the Cartesian product
/// of `chart_dict × a_dict`.
pub fn joint_search(
    truth: Quat,
    dev: &[Vec<Quat>],
    eval: &[Vec<Quat>],
    chart_dict: &[Chart],
    a_dict: &[AccumMethod],
) -> Option<QuatSearchReport> {
    let candidates: Vec<QuatPlan> = chart_dict
        .iter()
        .flat_map(|&chart| {
            a_dict.iter().map(move |&a| QuatPlan {
                chart,
                accumulation: a,
            })
        })
        .collect();
    let score = move |plan: QuatPlan, fit: &[Vec<Quat>], scr: &[Vec<Quat>]| {
        score_plan(plan, truth, fit, scr)
    };
    let baseline = move |fit: &[Vec<Quat>], scr: &[Vec<Quat>]| {
        score_plan(
            QuatPlan {
                chart: Chart::Componentwise,
                accumulation: AccumMethod::NaiveF32,
            },
            truth,
            fit,
            scr,
        )
        .unwrap_or(f64::NEG_INFINITY)
    };
    let out: GenericAutotune<QuatPlan> = autotune_by(dev, eval, &candidates, score, baseline);
    let plan = out.chosen?;
    Some(QuatSearchReport {
        plan,
        eval_mean_error_degrees: -out.chosen_eval_score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn trials(
        truth: Quat,
        sigma: f64,
        seed: u64,
        n_trials: usize,
        n_per_trial: usize,
    ) -> Vec<Vec<Quat>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n_trials)
            .map(|_| noisy_trial(truth, sigma, n_per_trial, &mut rng))
            .collect()
    }

    #[test]
    fn zero_noise_recovers_truth_exactly_for_both_charts() {
        let truth = Quaternion::from_axis_angle([1.0, 1.0, 1.0], 0.7).normalize();
        let samples: Vec<Quat> = (0..10).map(|_| truth).collect();
        for &chart in &default_chart_dictionary()
        {
            let est = average(chart, AccumMethod::NeumaierF32, &samples);
            let err = angular_error_degrees(est, truth);
            assert!(
                err < 1e-3,
                "{}: err {err} too large at zero noise",
                chart.name()
            );
        }
    }

    #[test]
    fn hemisphere_flip_does_not_corrupt_the_componentwise_mean() {
        let truth = Quaternion::identity();
        let mut samples: Vec<Quat> = (0..6).map(|_| truth).collect();
        // Flip half the samples to their antipodal (same rotation) representative.
        for q in samples.iter_mut().take(3)
        {
            *q = Quat::new(-q.w, -q.x, -q.y, -q.z);
        }
        let est = average(Chart::Componentwise, AccumMethod::NaiveF32, &samples);
        assert!(angular_error_degrees(est, truth) < 1e-6);
    }

    #[test]
    fn log_relative_and_exp_relative_are_inverses() {
        let reference = Quaternion::from_axis_angle([0.0, 1.0, 0.0], 0.3).normalize();
        let q = Quaternion::from_axis_angle([1.0, 0.0, 0.0], 0.9).normalize();
        let v = log_relative(reference, q);
        let back = exp_relative(reference, v);
        assert!(angular_error_degrees(back, q) < 1e-6);
    }

    #[test]
    fn joint_search_eval_score_is_never_worse_than_sequential_on_dev() {
        let truth = Quaternion::from_axis_angle([1.0, 1.0, 1.0], 0.7).normalize();
        let dev = trials(truth, 0.8, 10, 20, 100);
        let eval = trials(truth, 0.8, 11, 20, 100);
        let chart_dict = default_chart_dictionary();
        let a_dict = crate::autotune_accumulate::default_accumulators();

        let seq = sequential_baseline(truth, &dev, &eval, &a_dict);
        let joint = joint_search(truth, &dev, &eval, &chart_dict, &a_dict);
        if let (Some(seq), Some(joint)) = (seq, joint)
        {
            let seq_dev = score_plan(seq.plan, truth, &dev, &dev);
            let joint_dev = score_plan(joint.plan, truth, &dev, &dev);
            if let (Some(seq_dev), Some(joint_dev)) = (seq_dev, joint_dev)
            {
                assert!(
                    joint_dev >= seq_dev - 1e-9,
                    "joint dev score {joint_dev} worse than sequential {seq_dev}: \
                     joint search must dominate on dev by construction"
                );
            }
        }
    }
}
