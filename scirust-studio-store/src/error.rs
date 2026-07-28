//! Storage failures.

use std::path::{Path, PathBuf};

/// Something went wrong reading or writing the run store.
#[derive(Debug)]
pub enum StoreError {
    /// A filesystem operation failed, naming the path it was on — an
    /// `io::Error` alone ("permission denied") is not actionable without
    /// knowing *what* was denied.
    Io {
        /// The path the operation was on.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// No run with this id exists in the store.
    NoSuchRun {
        /// The id that was looked up.
        run_id: String,
    },
    /// The run exists but produced no result, because it was cancelled or
    /// failed.
    NoResult {
        /// The run.
        run_id: String,
        /// The status it ended in.
        status: String,
    },
    /// A stored file could not be decoded.
    Corrupt {
        /// The file.
        path: PathBuf,
        /// Why it could not be decoded.
        reason: String,
    },
    /// A stored file's content no longer matches the hash recorded for it.
    HashMismatch {
        /// The run.
        run_id: String,
        /// Which file changed.
        file: String,
        /// The hash the manifest recorded.
        expected: String,
        /// The hash the file has now.
        actual: String,
    },
    /// A stored result uses a schema version this build cannot read, or one
    /// the caller specifically required a different version of.
    ///
    /// Reported rather than worked around: the alternative is guessing what
    /// an unknown format meant, and for a legacy result the specific guess
    /// that matters — that its samples were evenly spaced — is false for any
    /// adaptive solver.
    UnsupportedResultSchema {
        /// The run.
        run_id: String,
        /// The version found in the file.
        found: u32,
        /// The versions acceptable here.
        supported: Vec<u32>,
    },
    /// Serializing a manifest or result failed.
    Encode(String),
    /// Every candidate run id derived from this base was already taken.
    RunIdExhausted {
        /// The base the ids were derived from.
        base: String,
    },
}

impl StoreError {
    /// Build an [`StoreError::Io`] naming the path involved.
    pub(crate) fn io(path: impl AsRef<Path>, source: std::io::Error) -> Self {
        StoreError::Io {
            path: path.as_ref().to_path_buf(),
            source,
        }
    }
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            StoreError::Io { path, source } =>
            {
                write!(f, "{}: {source}", path.display())
            },
            StoreError::NoSuchRun { run_id } => write!(f, "no run `{run_id}` in this store"),
            StoreError::NoResult { run_id, status } => write!(
                f,
                "run `{run_id}` has no result: it {status} before producing one"
            ),
            StoreError::Corrupt { path, reason } =>
            {
                write!(f, "{} is not readable: {reason}", path.display())
            },
            StoreError::HashMismatch {
                run_id,
                file,
                expected,
                actual,
            } => write!(
                f,
                "run `{run_id}` has been modified: {file} hashes to {actual}, \
                 but its manifest records {expected}"
            ),
            StoreError::UnsupportedResultSchema {
                run_id,
                found,
                supported,
            } =>
            {
                let list = supported
                    .iter()
                    .map(|v| format!("v{v}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "run `{run_id}` stores a v{found} result; this needs one of: {list}"
                )
            },
            StoreError::Encode(reason) => write!(f, "could not encode for storage: {reason}"),
            StoreError::RunIdExhausted { base } =>
            {
                write!(f, "could not allocate a free run id from `{base}`")
            },
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self
        {
            StoreError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_errors_name_the_path() {
        let err = StoreError::io("/tmp/x/manifest.json", std::io::Error::other("denied"));
        let text = err.to_string();
        assert!(text.contains("manifest.json"), "{text}");
        assert!(text.contains("denied"), "{text}");
    }

    #[test]
    fn a_hash_mismatch_names_both_hashes_and_the_file() {
        let text = StoreError::HashMismatch {
            run_id: "r1".to_string(),
            file: "result.json".to_string(),
            expected: "aaaa".to_string(),
            actual: "bbbb".to_string(),
        }
        .to_string();
        assert!(text.contains("r1"), "{text}");
        assert!(text.contains("result.json"), "{text}");
        assert!(text.contains("aaaa") && text.contains("bbbb"), "{text}");
    }

    #[test]
    fn a_missing_result_explains_why_rather_than_just_saying_missing() {
        let text = StoreError::NoResult {
            run_id: "r1".to_string(),
            status: "cancelled".to_string(),
        }
        .to_string();
        assert!(text.contains("cancelled"), "{text}");
    }
}
