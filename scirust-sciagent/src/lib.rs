pub mod agentic;
pub mod attention;
pub mod block;
pub mod bpe;
pub mod ccos;
pub mod checkpointing;
pub mod communication;
pub mod config;
pub mod corpus_paths;
#[cfg(feature = "cuda")]
pub mod cuda_model;
pub mod elastic_calibration;
pub mod elastic_engine;
pub mod elastic_tiny;
pub mod elastic_tokenizer;
pub mod flash_attention;
pub mod generate;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod inference;
pub mod model;
pub mod norm;
pub mod planning;
pub mod quantize;
pub mod sha256;
pub mod swiglu;
pub mod tokenizer;
pub mod train;

pub use attention::GQAAttention;
pub use block::SciAgentBlock;
pub use bpe::{BpeTokenizer, BpeTrainer};
pub use ccos::CcosLog;
pub use communication::SciAgentEndpoint;
pub use config::SciAgentConfig;
pub use elastic_calibration::{
    CalibrationError, CalibrationMeasurement, CalibrationReport, CalibrationWinner,
};
pub use elastic_engine::{ElasticBpeEngine, ElasticEncoding};
pub use elastic_tiny::{TINY_SCAN_CAPACITY, TinyScanBpe};
pub use elastic_tokenizer::{
    BpeKernel, CanonicalBpeOracle, DuplicateMergeRule, ElasticProfile, ElasticThresholds,
    PieceClass, ThresholdError, TokenId,
};
pub use generate::Generator;
pub use inference::SciAgentInference;
pub use model::SciAgentModel;
pub use norm::RMSNorm;
pub use swiglu::SwiGLUFFN;
pub use tokenizer::SciAgentTokenizer;
