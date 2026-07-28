//! The messages a Studio client and the capability worker exchange.

use serde::{Deserialize, Serialize};

use scirust_studio_runtime::{RunResult, RunWarning};

/// The protocol version this build speaks.
///
/// Bumped whenever a change would stop an older peer understanding a newer
/// one. Adding a field with a `#[serde(default)]` does not require a bump;
/// removing or renaming one, or changing a variant's meaning, does.
pub const PROTOCOL_VERSION: u32 = 1;

/// A client-assigned identifier correlating a request with every message
/// the worker sends back about it.
///
/// The worker never invents one: every response echoes the id of the
/// request that caused it, so a client that has several jobs in flight can
/// tell whose progress it is watching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RequestId(pub u64);

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// A message from the client to the worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Opens the session, announcing the client's protocol version. Must be
    /// the first message; the worker answers with
    /// [`WorkerMessage::Ready`] or refuses the version.
    Hello {
        /// The client's [`PROTOCOL_VERSION`].
        protocol_version: u32,
    },
    /// Ask for the capability catalogue.
    Catalog {
        /// Correlation id.
        request_id: RequestId,
    },
    /// Validate a scenario without running it.
    Validate {
        /// Correlation id.
        request_id: RequestId,
        /// The scenario's `.scirust.toml` text.
        scenario_toml: String,
    },
    /// Validate and run a scenario, streaming progress.
    Run {
        /// Correlation id.
        request_id: RequestId,
        /// The scenario's `.scirust.toml` text.
        scenario_toml: String,
    },
    /// Ask the worker to cancel an in-flight request.
    ///
    /// Cancelling a request that already finished, or one that was never
    /// started, is not an error — it is simply ignored, so a client racing
    /// a completion does not have to synchronise to avoid a spurious
    /// failure.
    Cancel {
        /// The request to cancel.
        request_id: RequestId,
    },
    /// Ask the worker to exit once the current request settles.
    Shutdown,
}

/// A message from the worker to the client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerMessage {
    /// The session is open.
    Ready {
        /// The worker's [`PROTOCOL_VERSION`].
        protocol_version: u32,
        /// The worker binary's crate version.
        worker_version: String,
    },
    /// The capability catalogue, as the registry's own JSON.
    Catalog {
        /// The request being answered.
        request_id: RequestId,
        /// `scirust_studio_registry::CapabilityRegistry::to_json` output,
        /// carried verbatim so the worker never re-describes capabilities in
        /// a second, drift-prone shape.
        catalog_json: String,
    },
    /// A scenario validated successfully.
    Validated {
        /// The request being answered.
        request_id: RequestId,
        /// The capability the scenario targets.
        capability_id: String,
    },
    /// Genuine progress: the run has covered `fraction` of its time span.
    Progress {
        /// The request this is about.
        request_id: RequestId,
        /// Fraction of the time span completed, in `0.0..=1.0`.
        fraction: f64,
        /// The simulation time reached.
        t: f64,
    },
    /// A non-fatal warning raised during the run.
    Warning {
        /// The request this is about.
        request_id: RequestId,
        /// The warning.
        warning: RunWarning,
    },
    /// The run finished successfully.
    ///
    /// Boxed because a [`RunResult`] carrying long time-courses is far
    /// larger than any other variant, and an un-boxed one would make every
    /// message in the enum that size.
    Completed {
        /// The request being answered.
        request_id: RequestId,
        /// The result.
        result: Box<RunResult>,
    },
    /// The run was cancelled before it finished.
    Cancelled {
        /// The request being answered.
        request_id: RequestId,
    },
    /// The request failed. `kind` classifies it so a client can map it to an
    /// exit code without parsing `message`.
    Failed {
        /// The request being answered.
        request_id: RequestId,
        /// What kind of failure this was.
        kind: FailureKind,
        /// Human-facing explanation, already formatted.
        message: String,
    },
    /// The channel itself is unusable — a malformed message, an
    /// unsupported version, a request before `Hello`. The worker exits
    /// after sending this.
    ProtocolError {
        /// What went wrong.
        message: String,
    },
    /// The worker is exiting in response to [`ClientMessage::Shutdown`].
    ShuttingDown,
}

/// How a request failed, mirroring the distinctions the CLI's exit codes
/// already draw (see `docs/studio/RUNTIME_CONTRACT.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    /// The scenario did not parse, or failed schema or capability
    /// validation.
    Validation,
    /// Integration failed numerically.
    Numerical,
    /// A bug in the worker or an adapter, not in the input.
    Internal,
}

impl FailureKind {
    /// The `scirust` exit code this failure maps to.
    pub fn exit_code(self) -> i32 {
        match self
        {
            FailureKind::Validation => 3,
            FailureKind::Numerical => 5,
            FailureKind::Internal => 7,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(message: &WorkerMessage) -> WorkerMessage {
        let json = serde_json::to_string(message).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn client_messages_round_trip() {
        let messages = vec![
            ClientMessage::Hello {
                protocol_version: PROTOCOL_VERSION,
            },
            ClientMessage::Catalog {
                request_id: RequestId(1),
            },
            ClientMessage::Validate {
                request_id: RequestId(2),
                scenario_toml: "schema_version = 1".to_string(),
            },
            ClientMessage::Run {
                request_id: RequestId(3),
                scenario_toml: "schema_version = 1".to_string(),
            },
            ClientMessage::Cancel {
                request_id: RequestId(3),
            },
            ClientMessage::Shutdown,
        ];
        for message in messages
        {
            let json = serde_json::to_string(&message).unwrap();
            let decoded: ClientMessage = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, message);
        }
    }

    #[test]
    fn worker_messages_round_trip() {
        use scirust_studio_runtime::WarningCategory;

        let messages = vec![
            WorkerMessage::Ready {
                protocol_version: PROTOCOL_VERSION,
                worker_version: "0.1.0".to_string(),
            },
            WorkerMessage::Catalog {
                request_id: RequestId(1),
                catalog_json: "[]".to_string(),
            },
            WorkerMessage::Validated {
                request_id: RequestId(2),
                capability_id: "sim.epidemiology.sir".to_string(),
            },
            WorkerMessage::Progress {
                request_id: RequestId(3),
                fraction: 0.25,
                t: 2.5,
            },
            WorkerMessage::Warning {
                request_id: RequestId(3),
                warning: RunWarning {
                    category: WarningCategory::Numerical,
                    message: "example".to_string(),
                },
            },
            WorkerMessage::Cancelled {
                request_id: RequestId(3),
            },
            WorkerMessage::Failed {
                request_id: RequestId(4),
                kind: FailureKind::Validation,
                message: "bad scenario".to_string(),
            },
            WorkerMessage::ProtocolError {
                message: "nope".to_string(),
            },
            WorkerMessage::ShuttingDown,
        ];
        for message in messages
        {
            assert_eq!(round_trip(&message), message);
        }
    }

    #[test]
    fn the_tag_is_a_stable_snake_case_type_field() {
        let json = serde_json::to_string(&ClientMessage::Shutdown).unwrap();
        assert_eq!(json, r#"{"type":"shutdown"}"#);

        let json = serde_json::to_string(&WorkerMessage::ShuttingDown).unwrap();
        assert_eq!(json, r#"{"type":"shutting_down"}"#);
    }

    #[test]
    fn request_ids_serialize_as_bare_numbers() {
        let json = serde_json::to_string(&RequestId(42)).unwrap();
        assert_eq!(json, "42");
    }

    #[test]
    fn request_id_displays_with_a_hash() {
        assert_eq!(RequestId(42).to_string(), "#42");
    }

    #[test]
    fn failure_kinds_map_to_the_documented_exit_codes() {
        // These must stay in step with docs/studio/RUNTIME_CONTRACT.md's
        // exit-code table; the CLI's own tests assert the same numbers.
        assert_eq!(FailureKind::Validation.exit_code(), 3);
        assert_eq!(FailureKind::Numerical.exit_code(), 5);
        assert_eq!(FailureKind::Internal.exit_code(), 7);
    }

    #[test]
    fn an_unknown_message_type_is_rejected_rather_than_silently_ignored() {
        let err = serde_json::from_str::<ClientMessage>(r#"{"type":"launch_missiles"}"#);
        assert!(err.is_err());
    }
}
