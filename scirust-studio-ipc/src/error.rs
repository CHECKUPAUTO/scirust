//! Protocol-level failures.

/// Something went wrong reading, writing, or agreeing on the protocol.
///
/// These are failures of the *channel*, distinct from a capability failing
/// to run — a scenario that does not validate is a perfectly well-formed
/// [`crate::WorkerMessage::Failed`], not an `IpcError`.
#[derive(Debug)]
pub enum IpcError {
    /// The underlying stream failed.
    Io(std::io::Error),
    /// A message could not be encoded to JSON.
    Encode(serde_json::Error),
    /// A message exceeded the caller's size limit and was refused before
    /// being decoded.
    MessageTooLarge {
        /// The limit that was exceeded, in bytes.
        limit: usize,
    },
    /// The stream ended part-way through a message: the peer died mid-write.
    Truncated,
    /// A complete line arrived but was not a valid protocol message.
    Malformed(String),
    /// The peer speaks a protocol version this build cannot talk to.
    UnsupportedProtocolVersion {
        /// The version this build speaks.
        ours: u32,
        /// The version the peer announced.
        theirs: u32,
    },
}

impl std::fmt::Display for IpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            IpcError::Io(e) => write!(f, "worker channel I/O failed: {e}"),
            IpcError::Encode(e) => write!(f, "could not encode a protocol message: {e}"),
            IpcError::MessageTooLarge { limit } =>
            {
                write!(f, "protocol message exceeded the {limit}-byte limit")
            },
            IpcError::Truncated =>
            {
                write!(f, "the peer closed the channel part-way through a message")
            },
            IpcError::Malformed(msg) => write!(f, "malformed protocol message: {msg}"),
            IpcError::UnsupportedProtocolVersion { ours, theirs } => write!(
                f,
                "protocol version mismatch: this build speaks v{ours}, the peer speaks v{theirs}"
            ),
        }
    }
}

impl std::error::Error for IpcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self
        {
            IpcError::Io(e) => Some(e),
            IpcError::Encode(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for IpcError {
    fn from(e: std::io::Error) -> Self {
        IpcError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_names_the_limit_that_was_exceeded() {
        let text = IpcError::MessageTooLarge { limit: 1024 }.to_string();
        assert!(text.contains("1024"), "{text}");
    }

    #[test]
    fn version_mismatch_names_both_versions() {
        let text = IpcError::UnsupportedProtocolVersion { ours: 1, theirs: 9 }.to_string();
        assert!(text.contains("v1") && text.contains("v9"), "{text}");
    }

    #[test]
    fn io_errors_keep_their_source() {
        use std::error::Error;
        let err = IpcError::from(std::io::Error::other("boom"));
        assert!(err.source().is_some());
    }
}
