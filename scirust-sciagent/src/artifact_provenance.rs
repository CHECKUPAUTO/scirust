use scirust_agent_protocol::Sha256Digest;

use crate::execution_attestation::sha256_digest;

const BUILTIN_BYTE_TOKENIZER_SEMANTICS_V1: &[u8] = b"scirust.sciagent.byte-tokenizer.v1\0";
const EMBEDDED_BPE_JSON: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/tokenizer/bpe.json"));

/// Exact SHA-256 identity of an artifact byte sequence.
///
/// This helper is intentionally byte-oriented: callers that load an artifact
/// from a buffer can hash the exact bytes that are then parsed, avoiding a
/// time-of-check/time-of-use gap between provenance and execution state.
pub fn artifact_sha256(bytes: &[u8]) -> Sha256Digest {
    sha256_digest(bytes)
}

/// Stable semantic identity for SciAgent's built-in raw-byte tokenizer path.
///
/// There is no tokenizer file on that path, so its provenance is the hash of a
/// versioned semantic identifier rather than a placeholder or machine-local
/// path. Changing byte-tokenizer semantics requires a new identifier/version.
pub fn builtin_byte_tokenizer_sha256() -> Sha256Digest {
    artifact_sha256(BUILTIN_BYTE_TOKENIZER_SEMANTICS_V1)
}

/// Exact artifact identity of the tokenizer JSON embedded in the SciAgent
/// binary. The digest is over the same `include_bytes!` payload used by the
/// embedded BPE loader.
pub fn embedded_bpe_tokenizer_sha256() -> Sha256Digest {
    artifact_sha256(EMBEDDED_BPE_JSON)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_tokenizer_identity_is_stable_and_nonzero() {
        let a = builtin_byte_tokenizer_sha256();
        let b = builtin_byte_tokenizer_sha256();
        assert_eq!(a, b);
        assert_ne!(
            a.as_str(),
            "0000000000000000000000000000000000000000000000000000000000000000"
        );
    }

    #[test]
    fn embedded_bpe_identity_hashes_exact_embedded_bytes() {
        assert_eq!(
            embedded_bpe_tokenizer_sha256(),
            sha256_digest(EMBEDDED_BPE_JSON)
        );
    }

    #[test]
    fn artifact_hash_is_byte_sensitive() {
        assert_ne!(artifact_sha256(b"model-a"), artifact_sha256(b"model-b"));
    }
}
