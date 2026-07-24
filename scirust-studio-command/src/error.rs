//! The stable `SRST-*` error catalogue.
//!
//! Codes are meant to be quoted in bug reports and help links: once a code is
//! assigned to a meaning by a producing crate, it must not be reassigned to a
//! different one.

use std::fmt;

/// One of the code families from the SciRust Studio error design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorFamily {
    /// `SRST-CLI-xxxx` — command-line usage errors.
    Cli,
    /// `SRST-VAL-xxxx` — scenario/input validation errors.
    Validation,
    /// `SRST-SIM-xxxx` — simulation-model errors.
    Simulation,
    /// `SRST-NUM-xxxx` — numerical-solver errors.
    Numerical,
    /// `SRST-RES-xxxx` — resource-limit errors.
    Resource,
    /// `SRST-IO-xxxx` — filesystem/import/export errors.
    Io,
    /// `SRST-IPC-xxxx` — worker/IPC errors. Not yet raised by anything (the
    /// worker process is Phase 2); reserved so the family exists up front.
    Ipc,
    /// `SRST-SEC-xxxx` — security/integrity errors.
    Security,
    /// `SRST-UPD-xxxx` — updater errors. Not yet raised by anything (the
    /// updater is Phase 10); reserved so the family exists up front.
    Updater,
    /// `SRST-INT-xxxx` — internal errors that indicate a bug, not bad input.
    Internal,
}

impl ErrorFamily {
    /// The three-letter code segment, e.g. `"VAL"`.
    pub const fn code_segment(self) -> &'static str {
        match self
        {
            ErrorFamily::Cli => "CLI",
            ErrorFamily::Validation => "VAL",
            ErrorFamily::Simulation => "SIM",
            ErrorFamily::Numerical => "NUM",
            ErrorFamily::Resource => "RES",
            ErrorFamily::Io => "IO",
            ErrorFamily::Ipc => "IPC",
            ErrorFamily::Security => "SEC",
            ErrorFamily::Updater => "UPD",
            ErrorFamily::Internal => "INT",
        }
    }
}

/// A stable, structured error code, e.g. `SRST-VAL-0001`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorCode {
    /// Which family the error belongs to.
    pub family: ErrorFamily,
    /// The number within the family.
    pub number: u16,
}

impl ErrorCode {
    /// Build a code, e.g. `ErrorCode::new(ErrorFamily::Validation, 1)`.
    pub const fn new(family: ErrorFamily, number: u16) -> Self {
        ErrorCode { family, number }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SRST-{}-{:04}", self.family.code_segment(), self.number)
    }
}

/// A user-facing error entry: a stable code, a short title, a plain-language
/// explanation, whether the user can recover from it, and what to try next —
/// not just a `Display` string with no structure behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogedError {
    /// The stable code, e.g. `SRST-VAL-0001`.
    pub code: ErrorCode,
    /// A short, human title, e.g. `"Unsupported schema version"`.
    pub title: String,
    /// A plain-language explanation of what went wrong.
    pub explanation: String,
    /// Whether the user can fix and retry, as opposed to this indicating a
    /// bug that needs a report.
    pub recoverable: bool,
    /// What the user should try next, if anything.
    pub suggested_action: Option<String>,
}

impl fmt::Display for CatalogedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}: {}", self.code, self.title, self.explanation)?;
        if let Some(action) = &self.suggested_action
        {
            write!(f, " ({action})")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_formats_with_zero_padding() {
        assert_eq!(
            ErrorCode::new(ErrorFamily::Validation, 1).to_string(),
            "SRST-VAL-0001"
        );
        assert_eq!(
            ErrorCode::new(ErrorFamily::Internal, 42).to_string(),
            "SRST-INT-0042"
        );
    }

    #[test]
    fn cataloged_error_display_includes_code_title_and_action() {
        let err = CatalogedError {
            code: ErrorCode::new(ErrorFamily::Validation, 3),
            title: "End before start".to_string(),
            explanation: "the solver's end time is not after its start time".to_string(),
            recoverable: true,
            suggested_action: Some("set `solver.end` greater than `solver.start`".to_string()),
        };
        let text = err.to_string();
        assert!(text.contains("SRST-VAL-0003"));
        assert!(text.contains("End before start"));
        assert!(text.contains("set `solver.end`"));
    }

    #[test]
    fn display_without_suggested_action_has_no_trailing_parens() {
        let err = CatalogedError {
            code: ErrorCode::new(ErrorFamily::Internal, 1),
            title: "Unexpected state".to_string(),
            explanation: "this should not happen".to_string(),
            recoverable: false,
            suggested_action: None,
        };
        assert!(!err.to_string().ends_with(')'));
    }
}
