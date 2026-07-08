//! # scirust-tolerance — inertial tolerancing (*tolérancement inertiel*)
//!
//! Deterministic, pure-Rust building blocks for the inertial-tolerancing
//! methodology of M. Pillet and the SYMME lab (Adragna, Pillet, Formosa,
//! Samper — arXiv:1002.0270). Inertial tolerancing judges a characteristic by
//! its **inertia**
//!
//! ```text
//! I = √(δ² + σ²),   δ = μ − Target,   σ = std-dev
//! ```
//!
//! — the root-mean-square deviation from the *target*, equal to `√(E[Taguchi
//! loss]/k)` — rather than by distance to a `[LSL, USL]` interval. This lets an
//! off-centre, low-spread batch and a centred, higher-spread batch be judged
//! *equivalent* when they carry the same expected loss, and gives a real
//! guarantee on assembled products that root-sum-square-on-σ cannot.
//!
//! ## Modules
//!
//! - [`inertia`] — the [`Inertia`] type (estimate from a sample, Taguchi loss,
//!   the maximum-inertia budget `I_max`, the acceptance [`InertiaCone`]).
//! - [`capability`] — classical [`capability::cp`]/[`capability::cpk`]/
//!   [`capability::cpm`]/[`capability::cpmk`]/`Pp`/`Ppk`, the inertial index
//!   [`capability::cpi`], and conformity ([`capability::nonconformity_ppm`],
//!   [`capability::sigma_level`]).
//! - [`chain`] — 1D tolerance chains: [`chain::assembly_inertia_statistical`]/
//!   [`chain::assembly_inertia_worst_case`] analysis and [`chain::allocate`]
//!   synthesis (worst-case / statistical / weighted / guaranteed-`Cpk` /
//!   cost-optimal), plus traditional interval allocation for comparison.
//! - [`optimize`](mod@optimize) — minimum-cost tolerance synthesis under **several**
//!   functional requirements at once ([`optimize::optimize`]), by convex
//!   Lagrangian dual ascent, plus the cost–quality Pareto frontier.
//! - [`chart`] — the [`chart::PilotingChart`] (*carte de pilotage inertiel*),
//!   monitoring the inertia against an upper piloting limit.
//! - [`sampling`] — acceptance sampling by inertia: the [`sampling::SamplingPlan`]
//!   operating-characteristic curve and [`sampling::design_plan`] for a
//!   producer/consumer double specification.
//! - [`form`] — surface / form inertia ([`form::FormBatch`]): the whole-surface
//!   inertia as the quadratic mean of the per-point inertias.
//! - [`modal`] — modal decomposition of form defects ([`modal::ModalBasis`],
//!   DCT / eigenmode) and the modal inertias that partition the surface inertia.
//! - [`spatial`] — 3D inertial tolerancing by small-displacement torsors
//!   ([`spatial::Torsor`], [`spatial::Feature`]): normal deviation `e = G·θ`,
//!   best-fit torsor + form residual, and the surface inertia as the
//!   statistical combination of location and orientation via the geometry
//!   matrix `H`.
//! - [`nonnormal`] — non-normal statistical tolerancing: Cornish–Fisher
//!   quantiles, non-normal ppm, and Clements percentile capability from the
//!   first four moments (the inertia itself is distribution-free).
//! - [`position`] — GD&T / ISO positional tolerancing: true position, MMC
//!   bonus, `±`↔`Ø` conversion, and the positional inertia `√(Iₓ²+I_y²)`.
//! - [`geometry`] — the rest of the ISO 1101 characteristics: straightness /
//!   flatness / roundness / cylindricity (form), parallelism / perpendicularity
//!   / angularity (orientation), profile and runout, each with its inertial RMS.
//! - [`montecarlo`] — Monte-Carlo tolerance simulation: arbitrary component
//!   distributions through a non-linear transfer function → inertia, yield,
//!   ppm, percentiles (deterministic, seeded).
//! - [`correlated`] — correlated / non-linear chains: covariance-form inertia
//!   `√((α∘I)ᵀR(α∘I))`, finite-difference linearisation, second-order mean.
//! - [`sensitivity`] — per-component variance-contribution ranking of a chain.
//! - [`process`] — discrete-process (menu) cost allocation by exact
//!   multiple-choice knapsack.
//! - [`drift`] — short-vs-long-term capability: uniform mean-drift variance and
//!   the Motorola 1.5σ shift (`Cpk`↔`Ppk`).
//! - [`msa`] — measurement-system analysis: crossed Gage R&R by ANOVA
//!   (repeatability / reproducibility / part variance, %R&R, ndc).
//! - [`interval`] — statistical tolerance intervals (normal `k`-factors,
//!   coverage × confidence) and spec conformance.
//! - [`distfit`] — distribution fitting (normal / lognormal / Rayleigh /
//!   Weibull) and ISO 22514 percentile capability off the best-fit law.
//! - [`variables`] — acceptance sampling **by variables** (ISO 3951 /
//!   MIL-STD-414 Form-`k`): the OC curve `Φ(√n(z_p−k))` and two-point
//!   `(n, k)` design for known- and unknown-`σ` methods.
//! - [`sixsigma`] — Six-Sigma yield accounting: DPMO / DPU, throughput and
//!   rolled-throughput yield, and yield↔sigma-level↔DPMO conversions.
//! - [`attribution`] — data-driven root-cause attribution: least-squares
//!   variance-transmission decomposition of measured assembly variation onto
//!   co-measured components, with fitted sensitivities, `R²` and the
//!   unexplained remainder.
//! - [`attributes`] — acceptance sampling **by attributes** (ISO 2859-1):
//!   binomial OC `P(D≤c)` and two-point `(n, c)` design, plus average
//!   outgoing quality.
//! - [`interference`] — stress–strength interference and assembly-fit
//!   reliability: `R = Φ((μ_S−μ_L)/√(σ_S²+σ_L²))`, the reliability index, and
//!   clearance-fit probabilities for a random hole/shaft pair.
//! - [`subgroup`] — rational-subgroup capability study (AIAG / ISO 22514-2):
//!   within-subgroup `σ̂ = R̄/d₂ = s̄/c₄` driving `Cp`/`Cpk` vs the overall spread
//!   driving `Pp`/`Ppk`.
//! - [`fits`] — ISO 286 limits and fits: standard tolerance grades `ITn` from the
//!   tolerance factor `i`, shaft fundamental deviations `d..h`, and hole/shaft
//!   clearance-fit classification.
//! - [`sequential`] — multi-stage acceptance sampling: double-sampling OC / ASN
//!   and Wald's sequential probability ratio test.
//! - [`taguchi`] — the Taguchi quadratic loss and cost of non-quality: the
//!   `E[L] = k·I²` link to inertia and the economic manufacturing tolerance.
//! - [`special`] — error function / normal CDF / central & non-central χ².
//!
//! Beyond the single-characteristic core, [`inertia`] also covers **lot
//! mixing** ([`inertia::mix_lots`], `I_c² = Σ pᵢ Iᵢ²`), multi-DOF / 3D
//! combination ([`inertia::vector_inertia`]), and correcting an observed
//! inertia for measurement dispersion ([`inertia::correct_for_measurement`]).
//!
//! Pairs naturally with `scirust-spc` (Shewhart / EWMA / Hotelling charts) and
//! `scirust-metrology` (GUM measurement uncertainty) for a complete
//! measure → capability → monitor → tolerance-allocate loop.
//!
//! ## Quick start
//!
//! ```
//! use scirust_tolerance::inertia::{Inertia, InertiaCone, i_max_from_tolerance};
//! use scirust_tolerance::capability::cpi;
//!
//! // A gap toleranced 1 ± 0.5 mm. Cp = 1 inertia budget: I_max = IT/6.
//! let i_max = i_max_from_tolerance(1.0, 1.0);      // = 1/6 ≈ 0.1667
//! let cone = InertiaCone::new(i_max);
//!
//! // A batch off-target by 0.1 with σ = 0.08.
//! let batch = Inertia::new(0.1, 0.08);             // I ≈ 0.128
//! assert!(cone.accepts(&batch));                   // inside the cone
//! assert!(cpi(&batch, i_max) > 1.0);               // Cpi ≥ 1 ⇒ conforming
//! ```
//!
//! ```
//! use scirust_tolerance::chain::{allocate, Allocation};
//!
//! // Distribute a Cp=1 assembly budget (R_Y/6) over a five-link ±1 chain,
//! // guaranteeing Cpk ≥ 1 on the resultant.
//! let coeffs = [1.0, -1.0, -1.0, -1.0, -1.0];
//! let i_per = allocate(1.0 / 6.0, &coeffs, &Allocation::GuaranteedCpk(1.0)).unwrap();
//! assert!((i_per[0] - 0.0597).abs() < 1e-3);
//! ```

pub mod attributes;
pub mod attribution;
pub mod capability;
pub mod chain;
pub mod chart;
pub mod correlated;
pub mod distfit;
pub mod drift;
pub mod fits;
pub mod form;
pub mod geometry;
pub mod inertia;
pub mod interference;
pub mod interval;
pub mod modal;
pub mod montecarlo;
pub mod msa;
pub mod nonnormal;
pub mod optimize;
pub mod position;
pub mod process;
pub mod sampling;
pub mod sensitivity;
pub mod sequential;
pub mod sixsigma;
pub mod spatial;
pub mod special;
pub mod subgroup;
pub mod taguchi;
pub mod variables;

pub use attributes::{AttributesPlan, design_attributes_plan};
pub use attribution::{Attribution, AttributionReport, attribute};
pub use capability::{
    CapabilitySummary, cp, cp_confidence_interval, cpi, cpk, cpk_confidence_interval, cpm, cpmk,
    nonconformity_ppm,
};
pub use chain::{
    Allocation, Contributor, ContributorState, allocate, assembly_inertia_statistical,
    assembly_inertia_worst_case, assembly_state,
};
pub use chart::{PilotingAction, PilotingChart, PilotingSignal};
pub use correlated::{correlated_inertia, correlated_variance, gradient, second_order_mean};
pub use distfit::{FittedDistribution, best_fit, percentile_capability};
pub use drift::{cpk_to_ppk, long_term_inertia, long_term_ppm, long_term_sigma};
pub use fits::{
    Fit, FitType, fit_from_deviations, hole_basis_fit, it_grade_tolerance,
    shaft_fundamental_deviation,
};
pub use form::FormBatch;
pub use geometry::{
    angularity, cylindricity, flatness, parallelism, perpendicularity, profile, roundness,
    straightness, total_runout,
};
pub use inertia::{
    Inertia, InertiaCone, correct_for_measurement, i_max_from_tolerance, mix_lots, vector_inertia,
};
pub use interference::{FitAnalysis, clearance_fit, interference_reliability, reliability_index};
pub use interval::{ToleranceInterval, tolerance_factor_two_sided, tolerance_interval};
pub use modal::{ModalBasis, modal_inertias};
pub use montecarlo::{Distribution, SimResult, simulate};
pub use msa::{GageRnR, GageVerdict, gage_rr};
pub use nonnormal::{
    ClementsCapability, clements_capability, cornish_fisher_quantile, nonnormal_ppm,
};
pub use optimize::{Component, OptimizeResult, Requirement, cost_quality_frontier, optimize};
pub use position::{
    CompositePosition, FeatureType, datum_shift, positional_inertia, resultant_condition,
    total_position_tolerance, true_position, virtual_condition,
};
pub use process::{Combination, ProcessOption, allocate_discrete};
pub use sampling::{SamplingPlan, design_plan, plan_for_producer_risk};
pub use sensitivity::{Contribution, DualContribution, contributions, dual_contributions};
pub use sequential::{
    DoubleSamplingPlan, SequentialPlan, SequentialVerdict, design_sequential_plan,
};
pub use sixsigma::{
    ProcessReport, dpmo, dpmo_from_sigma, dpu, normalized_yield, process_report,
    rolled_throughput_yield, sigma_from_dpmo, sigma_from_yield, throughput_yield, yield_from_sigma,
};
pub use spatial::{Feature, Torsor, surface_inertia_from_torsors};
pub use subgroup::{SubgroupCapability, sigma_within_s_method, subgroup_capability};
pub use taguchi::{
    economic_tolerance, expected_loss, expected_loss_from_moments, larger_the_better_loss,
    loss_coefficient, quadratic_loss, smaller_the_better_loss,
};
pub use variables::{VariablesPlan, design_variables_plan};
