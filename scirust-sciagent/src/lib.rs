pub mod agentic;
pub mod artifact_provenance;
pub mod attention;
pub mod block;
pub mod bpe;
pub mod bpe_dispatch;
pub mod canonical_bpe_train;
pub mod ccos;
pub mod checkpointing;
pub mod communication;
pub mod config;
pub mod corpus_paths;
#[cfg(feature = "cuda")]
pub mod cuda_decode;
#[cfg(feature = "cuda")]
pub mod cuda_decode_execution_attestation;
#[cfg(feature = "cuda")]
pub mod cuda_model;
pub mod elastic_autotune;
pub mod elastic_calibration;
pub mod elastic_engine;
pub mod elastic_heap;
pub mod elastic_id;
pub mod elastic_indexed;
pub mod elastic_profile_fit;
pub mod elastic_profile_store;
mod elastic_rule_table;
pub mod elastic_text_tokenizer;
pub mod elastic_tiny;
pub mod elastic_tokenizer;
pub mod execution_attestation;
pub mod flash_attention;
pub mod generate;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod inference;
pub mod model;
pub mod norm;
pub mod planning;
pub mod quantize;
pub mod route_b_execution_attestation;
pub mod sha256;
pub mod swiglu;
pub mod tokenizer;
pub mod train;

pub use artifact_provenance::{
    artifact_sha256, builtin_byte_tokenizer_sha256, embedded_bpe_tokenizer_sha256,
};
pub use attention::GQAAttention;
pub use block::SciAgentBlock;
pub use bpe::{BpeTokenizer, BpeTrainer};
pub use bpe_dispatch::{BpeDispatchError, VersionedBpeTokenizer};
pub use canonical_bpe_train::{CanonicalBpeArtifact, CanonicalBpeTrainError, CanonicalBpeTrainer};
pub use ccos::CcosLog;
pub use communication::{RuntimeEndpoint, SciAgentEndpoint};
pub use config::SciAgentConfig;
#[cfg(feature = "cuda")]
pub use cuda_decode_execution_attestation::{
    CUDA_DECODE_NUMERIC_MODE_V1, CudaDecodeExecutionAttestationError,
    CudaDecodeExecutionAttestationExt, CudaDecodeExecutionAttestationInputs,
    cuda_decode_kernel_semantic_version,
};
pub use elastic_autotune::{
    AutotuneConfig, AutotuneError, AutotuneResult, CalibrationCase, ElasticAutotuner,
};
pub use elastic_calibration::{
    CalibrationError, CalibrationMeasurement, CalibrationReport, CalibrationWinner,
};
pub use elastic_engine::{ElasticBpeEngine, ElasticEncoding};
pub use elastic_heap::HeapBpe;
pub use elastic_id::{CompactWordError, PackedRule, PairKey, PairKeyError, PriorityKey};
pub use elastic_indexed::IndexedBpe;
pub use elastic_profile_fit::{ElasticProfileFitter, ProfileFitError};
pub use elastic_profile_store::{
    CANONICAL_BPE_SEMANTICS_V1, ELASTIC_PROFILE_SCHEMA_V1, ELASTIC_PROFILE_SCHEMA_V2,
    ElasticHardwareIdentity, ProfileStoreError, StoredElasticProfile, ordered_merges_fingerprint,
};
pub use elastic_text_tokenizer::{
    BpeMergeSemantics, ElasticTextTokenizer, ElasticTextTokenizerError,
};
pub use elastic_tiny::{TINY_SCAN_CAPACITY, TinyScanBpe};
pub use elastic_tokenizer::{
    BpeKernel, CanonicalBpeOracle, DuplicateMergeRule, ElasticProfile, ElasticThresholds,
    PieceClass, ThresholdError, TokenId,
};
pub use execution_attestation::{
    RuntimeExecutionAttestationInputs, build_runtime_execution_attestation, sha256_digest,
};
pub use generate::Generator;
pub use inference::SciAgentInference;
pub use model::SciAgentModel;
pub use norm::RMSNorm;
pub use route_b_execution_attestation::{
    ROUTE_B_CUDA_KERNEL_SEMANTICS_V1, ROUTE_B_CUDA_NUMERIC_MODE_V1,
    RouteBCudaExecutionAttestationInputs, build_route_b_cuda_execution_attestation,
};
pub use swiglu::SwiGLUFFN;
pub use tokenizer::SciAgentTokenizer;
