//! Reproducible industrial benchmark harness for the SRCC robust structural
//! intelligence program (phase 727).
//!
//! This crate is **protocol, not verdict**: it provides the machinery to test
//! industrial claims honestly — dataset manifests with content checksums,
//! deterministic anti-leakage splits, contamination generators that record
//! exactly what they did, a capability-declared adapter interface over the
//! workspace's estimators, deterministic paired bootstrap inference, and
//! `scirust-bench-schema::BenchRecord` output. It contains **no superiority
//! claims**; producing evidence is phase 728's job, under the preregistration
//! committed in `docs/research/SRCC_INDUSTRIAL_BENCHMARK_PREREGISTRATION.md`.
//!
//! Dependency direction (workspace integration contract): this crate depends
//! on method crates; method crates never depend on this crate.

#![forbid(unsafe_code)]

pub mod adapter;
pub mod contamination;
pub mod dataset;
pub mod loaders;
pub mod manifest;
pub mod metrics;
pub mod missing;
pub mod paired;
pub mod promotion;
pub mod records;
pub mod splits;
pub mod srcc_views;

pub use adapter::{
    AdapterError, AdapterOutput, BaselineAdapter, CusumAdapter, DbscanAdapter, EwmaAdapter,
    FittingProtocol, HotellingT2Adapter, IsolationForestAdapter, LofAdapter, MahalanobisAdapter,
    RobustRegressionAdapter, TaskKind,
};
pub use contamination::{
    ContaminationConfig, ContaminationError, ContaminationKind, ContaminationManifest,
    apply_contamination,
};
pub use dataset::{DatasetError, TabularDataset};
pub use loaders::{LoaderError, clip_rul_targets, parse_cmapss_training, parse_obd2, parse_secom};
pub use manifest::{DatasetManifest, FeatureDescriptor, ManifestError};
pub use metrics::{
    ConfusionCounts, DetectionOutcome, DetectionReport, MetricError, adjusted_rand_index, auroc,
    confusion_counts, detection_report, mean_absolute_error, median_absolute_error, rand_index,
    rmse, worst_absolute_error,
};
pub use missing::{DropReason, FittedImputer, MissingPolicyError, MissingValuePolicy};
pub use paired::{
    PairedBootstrapReport, PairedComparisonError, paired_bootstrap, paired_differences,
};
pub use promotion::{
    CriterionFinding, Decision, ExtendedPromotionGate, ExtendedPromotionReport, Guardrail,
    Orientation, PairedMetric, PrimaryCriterion, PromotionError, PromotionGate, PromotionReport,
    ShadowWindow, WeightedMetric, WindowFinding, WindowMetricValues,
};
pub use records::{
    RecordKey, RunMetadata, alarm_records, anomaly_label_records, anomaly_score_records,
    regression_records, sha256_hex,
};
pub use splits::{SplitAssignment, SplitError, SplitManifest, SplitStrategy, split_dataset};
pub use srcc_views::{TransportViewSpec, TransportViews, ViewError, build_transport_views};
