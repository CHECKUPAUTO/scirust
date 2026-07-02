use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum SisError {
    #[error(
        "architecture {m}oo{n} has no known PFDavg formula (supported: 1oo1, 1oo2, 2oo2, 2oo3, 1oo3)"
    )]
    UnsupportedArchitecture { m: u8, n: u8 },

    #[error("architecture {m}oo{n} is invalid: need 1 <= m <= n")]
    InvalidArchitecture { m: u8, n: u8 },

    #[error("expected {expected} channel votes, got {got}")]
    VoteCountMismatch { expected: u8, got: usize },

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error(
        "could not bracket a proof-test interval reaching PFDavg={target:.3e} after {tries} doublings (last t1={last_t1:.3e}h, pfd={last_pfd:.3e})"
    )]
    NoBracket {
        target: f64,
        tries: usize,
        last_t1: f64,
        last_pfd: f64,
    },

    #[error("root-finding failed while sizing the proof-test interval: {0}")]
    RootFindingFailed(String),

    #[error("cause '{0}' is not defined in this cause-and-effect matrix")]
    UnknownCause(String),

    #[error("effect '{0}' is not defined in this cause-and-effect matrix")]
    UnknownEffect(String),
}

pub type SisResult<T> = Result<T, SisError>;
