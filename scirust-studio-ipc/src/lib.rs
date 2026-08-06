//! `scirust-studio-ipc` — the protocol between a SciRust Studio client and
//! the out-of-process capability worker.
//!
//! This is Phase 2B of the SciRust Studio effort (see
//! `docs/studio/adr/0003-worker-process-and-ipc.md` and
//! `docs/studio/IPC_PROTOCOL.md`). The crate is deliberately transport-free:
//! it defines the messages and how to frame them on *any* byte stream, and
//! says nothing about pipes, sockets, or process spawning. The worker binary
//! (`scirust-studio-worker`) speaks it over stdin/stdout; a test speaks it
//! over an in-memory buffer; a future desktop shell can speak it over
//! whatever it likes without this crate changing.
//!
//! Three properties are load-bearing and each is tested here:
//!
//! 1. **Versioned.** [`PROTOCOL_VERSION`] is exchanged in the opening
//!    handshake, so a client and worker from different builds discover the
//!    mismatch immediately instead of misinterpreting each other's messages.
//! 2. **Bounded.** [`framing::read_message`] refuses a message larger than a
//!    caller-supplied limit, so neither side can be driven into unbounded
//!    allocation by a runaway or hostile peer.
//! 3. **Correlated.** Every worker message about a request carries the
//!    [`RequestId`] the client assigned, so responses are never matched by
//!    arrival order.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod error;
pub mod framing;
mod message;

pub use error::IpcError;
pub use framing::{DEFAULT_MAX_MESSAGE_BYTES, read_message, write_message};
pub use message::{ClientMessage, FailureKind, PROTOCOL_VERSION, RequestId, WorkerMessage};

/// Check a peer's announced protocol version against this build's.
///
/// Deliberately an exact-match check. A range check would be a promise this
/// project cannot yet keep — there is exactly one version in existence, so
/// there is no compatibility window to have tested, and pretending otherwise
/// would let a future mismatch through silently. When a v2 exists and a
/// tested v1-compatibility path exists with it, this is where that
/// negotiation goes.
pub fn check_protocol_version(theirs: u32) -> Result<(), IpcError> {
    if theirs == PROTOCOL_VERSION
    {
        Ok(())
    }
    else
    {
        Err(IpcError::UnsupportedProtocolVersion {
            ours: PROTOCOL_VERSION,
            theirs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_current_version_is_accepted() {
        assert!(check_protocol_version(PROTOCOL_VERSION).is_ok());
    }

    #[test]
    fn any_other_version_is_refused() {
        for other in [0, PROTOCOL_VERSION + 1, u32::MAX]
        {
            let err = check_protocol_version(other).unwrap_err();
            assert!(
                matches!(err, IpcError::UnsupportedProtocolVersion { .. }),
                "{err:?}"
            );
        }
    }

    /// A full request/response exchange over an in-memory stream — proof
    /// that the message types and the framing compose, without a process.
    #[test]
    fn a_client_and_worker_exchange_survives_a_round_trip() {
        let mut wire = Vec::new();
        write_message(
            &mut wire,
            &ClientMessage::Hello {
                protocol_version: PROTOCOL_VERSION,
            },
        )
        .unwrap();
        write_message(
            &mut wire,
            &ClientMessage::Run {
                request_id: RequestId(1),
                scenario_toml: "schema_version = 1\n".to_string(),
            },
        )
        .unwrap();
        write_message(&mut wire, &ClientMessage::Shutdown).unwrap();

        let mut cursor = wire.as_slice();
        let mut seen = Vec::new();
        while let Some(m) = read_message::<_, ClientMessage>(&mut cursor, 1 << 20).unwrap()
        {
            seen.push(m);
        }
        assert_eq!(seen.len(), 3);
        assert!(matches!(seen[0], ClientMessage::Hello { .. }));
        assert!(matches!(seen[1], ClientMessage::Run { .. }));
        assert_eq!(seen[2], ClientMessage::Shutdown);
    }
}
