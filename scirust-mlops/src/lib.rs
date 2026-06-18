//! SciRust MLOps — Industrial ML Operations
//!
//! Drift detection, shadow deployment, and OTA model distribution for
//! industrial AI applications.
//!
//! ## Modules
//! - **drift** — Data drift and model drift detection
//! - **shadow** — Shadow deployment (run new model alongside production)
//! - **ota** — Over-the-air model distribution with signed artifacts

pub mod drift;
pub mod ota;
pub mod shadow;

pub use drift::{DataDriftDetector, DriftReport, DriftType, ModelDriftDetector};
pub use ota::{ModelSignature, OtaResult, OtaUpdate, SigningKey};
pub use shadow::{ComparisonMetric, ShadowDeployment, ShadowResult};
