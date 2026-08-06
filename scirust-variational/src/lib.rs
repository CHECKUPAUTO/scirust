#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
pub mod action;
pub mod control;
pub mod error;
pub mod euler_lagrange;
pub mod learning;
pub mod mechanics;
pub mod pinn;
pub mod util;
pub mod verification;

pub mod prelude {
    pub use crate::action::{
        ActionProblem, BoundaryCondition, GeneralizedCoordinate, GeneralizedVelocity, TimeInterval,
    };
    pub use crate::control::direct::direct_shooting;
    pub use crate::control::problem::{ControlBounds, ControlSolution, OptimalControlProblem};
    pub use crate::error::{Result, VariationalError};
    pub use crate::euler_lagrange::ELEquation;
    pub use crate::euler_lagrange::autodiff::{
        AutodiffEulerLagrange, ELGradients, VelocityHessian,
    };
    pub use crate::euler_lagrange::constrained::ConstrainedEulerLagrange;
    pub use crate::euler_lagrange::symbolic::SymbolicEulerLagrange;
    pub use crate::learning::dataset::{TrajectoryDataset, TrajectorySample, trajectory_from_ode};
    pub use crate::learning::hnn::HamiltonianNetwork;
    pub use crate::learning::lnn::LagrangianNetwork;
    pub use crate::learning::losses::{LossValue, acceleration_loss, mse_loss};
    pub use crate::learning::trainer::{
        HasParameters, PhysicsTrainer, TrainingConfig, TrainingMetrics,
    };
    pub use crate::mechanics::hamiltonian::{HamiltonianDynamics, HamiltonianDynamicsConfig};
    pub use crate::mechanics::invariants::{ConservationReport, compute_energy};
    pub use crate::mechanics::lagrangian::{LagrangianDynamics, LagrangianDynamicsConfig};
    pub use crate::pinn::collocation::CollocationPoints;
    pub use crate::pinn::conditions::{Condition, ConditionConfig, ConditionKind};
    pub use crate::pinn::domain::Domain1D;
    pub use crate::pinn::residual::{
        DifferentialOperator, central_difference, central_difference_1d,
    };
    pub use crate::pinn::trainer::PinnNet;
    pub use crate::verification::finite_difference::{
        FDCheckReport, check_gradient, check_hessian,
    };
}
