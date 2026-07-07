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

pub mod capability;
pub mod chain;
pub mod chart;
pub mod correlated;
pub mod distfit;
pub mod drift;
pub mod form;
pub mod geometry;
pub mod inertia;
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
pub mod spatial;
pub mod special;

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
pub use form::FormBatch;
pub use geometry::{
    angularity, cylindricity, flatness, parallelism, perpendicularity, profile, roundness,
    straightness, total_runout,
};
pub use inertia::{
    Inertia, InertiaCone, correct_for_measurement, i_max_from_tolerance, mix_lots, vector_inertia,
};
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
pub use spatial::{Feature, Torsor, surface_inertia_from_torsors};
