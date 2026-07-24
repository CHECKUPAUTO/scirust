//! A minimal, hand-rolled argument parser: positionals plus `--flag value`
//! pairs. No third-party dependency — every `sos` subcommand's argument shape
//! is simple enough for this to stay honest and readable, matching this
//! workspace's established "no new third-party download" discipline
//! (`scirust-cli` at the repository root does the same).

use std::collections::BTreeMap;

use crate::error::{CliError, Result};

/// The arguments to one subcommand, split into positionals (in order) and
/// `--name value` flags.
#[derive(Debug, Clone, Default)]
pub struct Args {
    /// Positional arguments, in the order given.
    pub positional: Vec<String>,
    /// `--name value` flags, keyed by name (without the leading `--`).
    pub flags: BTreeMap<String, String>,
}

impl Args {
    /// Parse `argv` (a subcommand's arguments, program name and subcommand
    /// name already stripped): every `--name value` pair becomes a flag;
    /// everything else is positional, in order.
    ///
    /// # Errors
    /// [`CliError::Usage`] if a `--name` flag has no following value.
    pub fn parse(argv: &[String]) -> Result<Self> {
        let mut positional = Vec::new();
        let mut flags = BTreeMap::new();
        let mut it = argv.iter();
        while let Some(arg) = it.next()
        {
            if let Some(name) = arg.strip_prefix("--")
            {
                let value = it
                    .next()
                    .ok_or_else(|| CliError::Usage(format!("flag --{name} needs a value")))?;
                flags.insert(name.to_owned(), value.clone());
            }
            else
            {
                positional.push(arg.clone());
            }
        }
        Ok(Self { positional, flags })
    }

    /// The `n`-th positional argument (0-indexed).
    ///
    /// # Errors
    /// [`CliError::Usage`] naming `what` if there is no such positional.
    pub fn positional(&self, n: usize, what: &str) -> Result<&str> {
        self.positional
            .get(n)
            .map(String::as_str)
            .ok_or_else(|| CliError::Usage(format!("missing required argument: {what}")))
    }

    /// The `n`-th positional argument, or `None` if absent.
    #[must_use]
    pub fn positional_opt(&self, n: usize) -> Option<&str> {
        self.positional.get(n).map(String::as_str)
    }

    /// A named flag's value, or `None` if it was not given.
    #[must_use]
    pub fn flag(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(String::as_str)
    }

    /// A named flag parsed as `i64`, or `default` if absent.
    ///
    /// # Errors
    /// [`CliError::Usage`] if the flag was given but is not a valid integer.
    pub fn flag_i64(&self, name: &str, default: i64) -> Result<i64> {
        match self.flag(name)
        {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| CliError::Usage(format!("--{name} must be an integer, got `{v}`"))),
        }
    }
}

/// Parse a `sos1:`-prefixed (or bare) hex object id argument.
///
/// # Errors
/// [`CliError::Usage`] if `s` is not a valid object id.
pub fn parse_object_id(s: &str) -> Result<sos_core::ObjectId> {
    sos_core::ObjectId::parse(s)
        .map_err(|e| CliError::Usage(format!("invalid object id `{s}`: {e}")))
}
