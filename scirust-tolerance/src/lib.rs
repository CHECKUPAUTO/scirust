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
//! - [`chart`] — the [`chart::PilotingChart`] (*carte de pilotage inertiel*),
//!   monitoring the inertia against an upper piloting limit.
//! - [`sampling`] — acceptance sampling by inertia: the [`sampling::SamplingPlan`]
//!   operating-characteristic curve and [`sampling::design_plan`] for a
//!   producer/consumer double specification.
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
pub mod inertia;
pub mod sampling;
pub mod special;

pub use capability::{CapabilitySummary, cp, cpi, cpk, cpm, cpmk, nonconformity_ppm};
pub use chain::{
    Allocation, Contributor, ContributorState, allocate, assembly_inertia_statistical,
    assembly_inertia_worst_case, assembly_state,
};
pub use chart::{PilotingAction, PilotingChart, PilotingSignal};
pub use inertia::{
    Inertia, InertiaCone, correct_for_measurement, i_max_from_tolerance, mix_lots, vector_inertia,
};
pub use sampling::{SamplingPlan, design_plan, plan_for_producer_risk};
