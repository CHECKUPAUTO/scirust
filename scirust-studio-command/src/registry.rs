//! Command descriptors and the registry that holds them.

use std::fmt;

/// A stable identifier for a command, e.g. `"catalog"` or `"run"`. Distinct
/// from the dispatch path so a command's path can be renamed without
/// changing what help links, logs, and tests refer to it as.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CommandId(pub String);

impl CommandId {
    /// Build a command id from a string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        CommandId(id.into())
    }
}

impl fmt::Display for CommandId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// How much a command can change or reveal. The registry only carries this
/// classification; enforcing it (confirmation prompts, feature flags for
/// `ExternalCommunication`) is the caller's job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyClass {
    /// Cannot modify anything; safe to run automatically or repeatedly.
    ReadOnly,
    /// Creates a new artifact (a run, a report, an export) without touching
    /// existing ones.
    CreatesArtifact,
    /// Modifies existing workspace state (a scenario file, a setting).
    ModifiesWorkspace,
    /// Deletes or irreversibly overwrites something; must be confirmed.
    Destructive,
    /// Talks to the network; must stay off unless explicitly enabled.
    ExternalCommunication,
}

/// One argument a command accepts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgumentDescriptor {
    /// Argument name as typed by the user (e.g. `"--seed"` or a positional name).
    pub name: String,
    /// One-line help text.
    pub help: String,
    /// Whether omitting the argument is an error.
    pub required: bool,
}

/// A worked example shown in help output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandExample {
    /// The literal command line, without the leading `scirust`.
    pub command_line: String,
    /// What the example demonstrates.
    pub description: String,
}

/// Everything the registry and the help system need to know about one
/// command, so it is written once instead of by hand in a dispatcher, a help
/// listing, and a documentation page that can drift apart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandDescriptor {
    /// Stable identifier, distinct from the display path.
    pub id: CommandId,
    /// Dispatch path, e.g. `["catalog", "list"]` for `scirust catalog list`.
    pub path: Vec<String>,
    /// Alternate paths that dispatch to the same command.
    pub aliases: Vec<Vec<String>>,
    /// One-line summary shown in listings.
    pub summary: String,
    /// Longer help text shown by `scirust help <command>`.
    pub long_help: String,
    /// Accepted arguments.
    pub arguments: Vec<ArgumentDescriptor>,
    /// Worked examples.
    pub examples: Vec<CommandExample>,
    /// The safety classification.
    pub safety: SafetyClass,
}

/// An error registering a command. The registry is left unchanged whenever
/// one of these is returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// Another command already registered this exact id.
    DuplicateId(CommandId),
    /// Another command already registered this exact path, as either a
    /// primary path or an alias.
    DuplicatePath(Vec<String>),
    /// A command was registered with an empty dispatch path.
    EmptyPath(CommandId),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            RegistryError::DuplicateId(id) => write!(f, "command id `{id}` is already registered"),
            RegistryError::DuplicatePath(path) =>
            {
                write!(f, "command path `{}` is already registered", path.join(" "))
            },
            RegistryError::EmptyPath(id) =>
            {
                write!(
                    f,
                    "command `{id}` was registered with an empty dispatch path"
                )
            },
        }
    }
}

impl std::error::Error for RegistryError {}

/// A registry of [`CommandDescriptor`]s: the single source of truth that
/// help text, completion, and (eventually) the command palette are meant to
/// be generated from, rather than hand-duplicated.
#[derive(Debug, Clone, Default)]
pub struct CommandRegistry {
    commands: Vec<CommandDescriptor>,
}

impl CommandRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        CommandRegistry::default()
    }

    /// Register a command. Rejects a duplicate id, a duplicate path (primary
    /// or alias, checked against every previously registered command's
    /// primary path and aliases), or an empty path — leaving the registry
    /// unchanged on error.
    pub fn register(&mut self, descriptor: CommandDescriptor) -> Result<(), RegistryError> {
        if descriptor.path.is_empty()
        {
            return Err(RegistryError::EmptyPath(descriptor.id));
        }
        if self.commands.iter().any(|c| c.id == descriptor.id)
        {
            return Err(RegistryError::DuplicateId(descriptor.id));
        }
        let new_paths = std::iter::once(&descriptor.path).chain(descriptor.aliases.iter());
        for existing in &self.commands
        {
            let existing_paths = std::iter::once(&existing.path).chain(existing.aliases.iter());
            for candidate in existing_paths
            {
                if new_paths.clone().any(|p| p == candidate)
                {
                    return Err(RegistryError::DuplicatePath(candidate.clone()));
                }
            }
        }
        self.commands.push(descriptor);
        Ok(())
    }

    /// Every registered command, in registration order (deterministic).
    pub fn iter(&self) -> impl Iterator<Item = &CommandDescriptor> {
        self.commands.iter()
    }

    /// Number of registered commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Whether the registry has no commands.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Find a command by its exact dispatch path (primary path or an alias).
    pub fn find_by_path(&self, path: &[&str]) -> Option<&CommandDescriptor> {
        self.commands.iter().find(|c| {
            path_matches(&c.path, path) || c.aliases.iter().any(|a| path_matches(a, path))
        })
    }

    /// Find a command by its stable id.
    pub fn find_by_id(&self, id: &CommandId) -> Option<&CommandDescriptor> {
        self.commands.iter().find(|c| &c.id == id)
    }

    /// Deterministic, plain-text help listing: one line per command in
    /// registration order, `<path> — <summary>`. This is the foundation that
    /// `scirust help`, a future command palette, and offline reference
    /// generation are meant to render from, so it intentionally does not
    /// depend on terminal width or color support.
    pub fn render_help(&self) -> String {
        let mut out = String::new();
        for c in &self.commands
        {
            out.push_str(&c.path.join(" "));
            out.push_str(" — ");
            out.push_str(&c.summary);
            out.push('\n');
        }
        out
    }
}

fn path_matches(candidate: &[String], query: &[&str]) -> bool {
    candidate.len() == query.len() && candidate.iter().zip(query.iter()).all(|(a, b)| a == b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(id: &str, path: &[&str]) -> CommandDescriptor {
        CommandDescriptor {
            id: CommandId::new(id),
            path: path.iter().map(|s| s.to_string()).collect(),
            aliases: Vec::new(),
            summary: format!("{id} summary"),
            long_help: format!("{id} long help"),
            arguments: Vec::new(),
            examples: Vec::new(),
            safety: SafetyClass::ReadOnly,
        }
    }

    #[test]
    fn registers_and_finds_by_path_and_id() {
        let mut reg = CommandRegistry::new();
        reg.register(sample("catalog.list", &["catalog", "list"]))
            .unwrap();
        assert_eq!(reg.len(), 1);
        let found = reg.find_by_path(&["catalog", "list"]).unwrap();
        assert_eq!(found.id, CommandId::new("catalog.list"));
        assert!(reg.find_by_id(&CommandId::new("catalog.list")).is_some());
        assert!(reg.find_by_path(&["catalog"]).is_none());
    }

    #[test]
    fn rejects_duplicate_id() {
        let mut reg = CommandRegistry::new();
        reg.register(sample("run", &["run"])).unwrap();
        let err = reg.register(sample("run", &["run", "start"])).unwrap_err();
        assert_eq!(err, RegistryError::DuplicateId(CommandId::new("run")));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn rejects_duplicate_path_including_aliases() {
        let mut reg = CommandRegistry::new();
        let mut first = sample("catalog.list", &["catalog", "list"]);
        first.aliases.push(vec!["ls".to_string()]);
        reg.register(first).unwrap();
        let err = reg.register(sample("other", &["ls"])).unwrap_err();
        assert_eq!(err, RegistryError::DuplicatePath(vec!["ls".to_string()]));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn rejects_empty_path() {
        let mut reg = CommandRegistry::new();
        let err = reg.register(sample("nowhere", &[])).unwrap_err();
        assert_eq!(err, RegistryError::EmptyPath(CommandId::new("nowhere")));
        assert!(reg.is_empty());
    }

    #[test]
    fn help_rendering_is_deterministic_and_ordered_by_registration() {
        let mut reg = CommandRegistry::new();
        reg.register(sample("b", &["b"])).unwrap();
        reg.register(sample("a", &["a"])).unwrap();
        let help = reg.render_help();
        assert!(help.find("b —").unwrap() < help.find("a —").unwrap());
        assert!(help.contains("b summary"));
        assert!(help.contains("a summary"));
    }
}
