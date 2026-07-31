use crate::memory::{EdgeType, MemoryGraph, NodeId, NodeType};
use crate::util::sha256_hex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseResult {
    pub file_path: String,
    pub file_hash: String,
    pub modules: Vec<ModuleDecl>,
    pub use_statements: Vec<UseStatement>,
    pub symbols: Vec<Symbol>,
    /// In-body call-sites (Slice 1: single-segment free-function calls). `serde(default)` +
    /// skip-if-empty keeps the serialized form unchanged for the common (heuristic / no-call) case.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub call_sites: Vec<CallSite>,
    /// In-body references to module-level `static`/`const` items (data-flow Slice 1). Same
    /// skip-if-empty serde contract — empty on the heuristic path and for call-free bodies.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub data_refs: Vec<DataRef>,
    /// Renamed-import bindings — one per `use a::b as c` (including inside groups). Records the
    /// LOCAL name (`c`) and the ORIGINAL target path (`a::b`) so a later call to the alias (`c()`
    /// or qualified `c::X`) resolves to the real definition. Only the `syn` (real-AST) path can
    /// see a rename; the heuristic fallback leaves this empty (same skip-if-empty serde contract as
    /// `call_sites`/`data_refs`, so the common no-alias case serializes unchanged).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub import_aliases: Vec<ImportAlias>,
    /// Per-impl-block method ownership facts — one `(type_name, method)` per `impl Foo { fn bar }`
    /// whose Self type is a concrete name (Slice 3, `x.bar()` receiver-type inference). They are the
    /// inputs to the `(type, method) → symbol` index that
    /// [`crate::memory::MemoryGraph::resolve_symbol_calls`] uses to resolve a `Foo::bar` callee minted
    /// from an inferred receiver type. Only the `syn` path emits them (same skip-if-empty contract).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub method_defs: Vec<MethodDef>,
    pub generated_nodes: usize,
    pub generated_edges: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDecl {
    pub name: String,
    pub line: usize,
    pub is_public: bool,
    pub children: Vec<ModuleDecl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UseStatement {
    pub full_path: String,
    pub line: usize,
    pub is_import: bool,
    pub components: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub line: usize,
    pub kind: SymbolKind,
}

/// A **call-site**: a `callee()` invocation found inside `caller`'s body. `callee` is the
/// `::`-joined call path — a bare `foo` (Slice 1) or a qualified `a::b::foo` (Slice 2); method
/// calls `x.bar()` are Slice 3. Resolved to a definition symbol by
/// [`crate::memory::MemoryGraph::resolve_symbol_calls`], which adds a `caller → callee`
/// [`EdgeType::Calls`](crate::memory::EdgeType) edge — the fn→fn structure that imports alone
/// miss. Only emitted on the `syn` (real-AST) path; empty on the heuristic fallback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallSite {
    pub caller: String,
    pub callee: String,
    pub line: usize,
}

/// A **data-reference**: `reader`'s body mentions `name`, a value path whose last segment is in
/// `SCREAMING_SNAKE_CASE` (the Rust convention for a `static`/`const`). `name` is the full
/// `::`-joined path — a bare `FOO` (Slice 1) or a qualified `m::CONST` / `crate::limits::MAX`
/// (Slice 2). Resolved to a `static`/`const` definition symbol by
/// [`crate::memory::MemoryGraph::resolve_data_flow`], which adds a `reader → item`
/// [`EdgeType::DataFlow`](crate::memory::EdgeType) edge — the shared-global-state channel that call
/// and import edges miss. Only emitted on the `syn` (real-AST) path; empty on the heuristic fallback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRef {
    pub reader: String,
    pub name: String,
    pub line: usize,
}

/// A **renamed import**: `use a::b as c` binds the LOCAL name `local` (`c`) to the ORIGINAL target
/// path `target` (`a::b`). A later call to the alias — bare `c()` or qualified `c::X` — is resolved
/// by [`crate::memory::MemoryGraph::resolve_symbol_calls`] by rewriting the alias's leading segment
/// through `target`, so the edge lands on `a::b`'s real definition (never on a same-named sibling
/// module/symbol). The plain (non-renamed) `use a::b` keeps the local name `b`, so it needs no
/// entry here. Only emitted on the `syn` (real-AST) path; empty on the heuristic fallback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportAlias {
    pub local: String,
    pub target: String,
}

/// A **method-ownership fact**: `type_name` (the concrete Self type of an `impl` block, e.g. `Foo`)
/// defines method `method` (e.g. `bar`). Recorded once per `impl Foo { fn bar }` so
/// [`crate::memory::MemoryGraph::resolve_symbol_calls`] can build a `(type, method) → symbol` index and
/// resolve a `Foo::bar` callee — minted by receiver-type inference for `x.bar()` (where `x: Foo`) or by
/// an explicit `Foo::bar()` associated call — to the right method, **resolve-uniquely-or-skip** (a type
/// name shared by two impls makes the pair ambiguous, so it is dropped rather than guessed). Only the
/// `syn` (real-AST) path emits these; the heuristic fallback leaves them empty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodDef {
    pub type_name: String,
    pub method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Const,
    Static,
    Type,
    Macro,
    Other,
}

/// Everything `ASTParser::extract_all` produces for one source file.
type Extracted = (
    Vec<ModuleDecl>,
    Vec<UseStatement>,
    Vec<Symbol>,
    Vec<CallSite>,
    Vec<DataRef>,
    Vec<ImportAlias>,
    Vec<MethodDef>,
);

/// Deepest `(`/`[`/`{` nesting CCOS will hand to `syn`.
///
/// `syn` is a recursive-descent parser: one nested delimiter costs several stack
/// frames, and a stack overflow is a `SIGABRT` the process cannot catch — one
/// hostile (or merely generated) source file would abort the whole agent mid-run,
/// taking any un-checkpointed memory with it. Measured on this crate, a *block*
/// nest (`if true { … }`, the most expensive shape) overflows at depth ~96 on a
/// 2 MiB stack and between 512 and 1024 on the 8 MiB main stack in a debug build.
///
/// Real Rust stays far below that. Scanning 14 899 `.rs` files (this crate, the
/// `scirust` tree, and every crate in the local cargo registry) the deepest was 45
/// — a `syn` test fixture holding a pretty-printed AST — with the deepest
/// hand-written file at 20. 128 is therefore ~3× the worst real file observed and
/// still an order of magnitude below the overflow.
///
/// Over the limit, `ASTParser::extract_all` falls back to the line-based
/// heuristic, which is iterative and cannot overflow: a pathological file
/// degrades to a coarser graph instead of killing the process.
pub const MAX_PARSE_NESTING: usize = 128;

/// At or below this depth `syn` runs inline on the caller's stack — the case for
/// essentially every real file, so the common path pays nothing.
#[cfg(feature = "syn-parser")]
const INLINE_PARSE_NESTING: usize = 32;

/// Stack given to the dedicated parse thread used between [`INLINE_PARSE_NESTING`]
/// and [`MAX_PARSE_NESTING`]. Large enough that the limit above is safe *whatever*
/// stack the caller happens to run on (a 2 MiB worker thread, say), which is what
/// makes the threshold a property of CCOS rather than of its embedding.
#[cfg(feature = "syn-parser")]
const DEEP_PARSE_STACK: usize = 32 * 1024 * 1024;

fn utf8_width(first: u8) -> usize {
    match first {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

/// Index just past the `"` that closes a normal string starting at `from`.
fn string_end(b: &[u8], from: usize) -> usize {
    let mut i = from;
    while i < b.len() {
        match b[i] {
            b'\\' => i += 2,
            b'"' => return i + 1,
            _ => i += 1,
        }
    }
    b.len()
}

/// Index just past the `"###…` that closes a raw string opened with `hashes` hashes.
fn raw_string_end(b: &[u8], from: usize, hashes: usize) -> usize {
    let mut i = from;
    while i < b.len() {
        if b[i] == b'"'
            && b[i + 1..]
                .iter()
                .take(hashes)
                .filter(|c| **c == b'#')
                .count()
                == hashes
        {
            return i + 1 + hashes;
        }
        i += 1;
    }
    b.len()
}

/// Index just past a `'…'` char literal starting at `i`, or `i + 1` when the quote
/// opens a *lifetime* (`'a`) instead. Distinguishing the two is what keeps `'}'`
/// and `'"'` — legal char literals — from corrupting the depth count.
fn quote_end(b: &[u8], i: usize) -> usize {
    let start = i + 1;
    if start >= b.len() {
        return b.len();
    }
    if b[start] == b'\\' {
        let mut j = start + 1;
        while j < b.len() {
            match b[j] {
                b'\'' => return j + 1,
                b'\\' => j += 2,
                _ => j += 1,
            }
        }
        return b.len();
    }
    let w = utf8_width(b[start]);
    if start + w < b.len() && b[start + w] == b'\'' {
        return start + w + 1;
    }
    i + 1
}

/// Deepest `(`/`[`/`{` nesting in `source`, ignoring delimiters inside comments,
/// strings, raw strings, byte strings, char literals and lifetimes.
///
/// A single linear pass with no recursion, so it is safe on exactly the inputs it
/// exists to reject. It is a *bound*, not a parse: unbalanced closers saturate at
/// zero rather than going negative, which can only ever make the answer smaller —
/// and the answer is compared against a limit with an order of magnitude of slack.
pub fn max_nesting_depth(source: &str) -> usize {
    let b = source.as_bytes();
    let (mut i, mut depth, mut max) = (0usize, 0usize, 0usize);
    while i < b.len() {
        match b[i] {
            b'/' if b.get(i + 1) == Some(&b'/') => {
                i += 2;
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            // Rust block comments nest, so `/* /* */ */` is one comment.
            b'/' if b.get(i + 1) == Some(&b'*') => {
                let mut open = 1usize;
                i += 2;
                while i < b.len() && open > 0 {
                    if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                        open += 1;
                        i += 2;
                    } else if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                        open -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
            }
            // `b"…"`, `b'…'`, `r"…"`, `r#"…"#`, `br#"…"#` — but only when the byte
            // starts a token, so the `r` of `char` or `arr` is left alone.
            b'b' | b'r' if i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_') => {
                let mut j = i;
                let byte_prefix = b[j] == b'b';
                if byte_prefix {
                    j += 1;
                }
                let raw = b.get(j) == Some(&b'r');
                if raw {
                    j += 1;
                    let hash_start = j;
                    while b.get(j) == Some(&b'#') {
                        j += 1;
                    }
                    if b.get(j) == Some(&b'"') {
                        i = raw_string_end(b, j + 1, j - hash_start);
                        continue;
                    }
                } else if byte_prefix {
                    match b.get(j) {
                        Some(&b'"') => {
                            i = string_end(b, j + 1);
                            continue;
                        }
                        Some(&b'\'') => {
                            i = quote_end(b, j);
                            continue;
                        }
                        _ => {}
                    }
                }
                i += 1;
            }
            b'"' => i = string_end(b, i + 1),
            b'\'' => i = quote_end(b, i),
            b'(' | b'[' | b'{' => {
                depth += 1;
                max = max.max(depth);
                i += 1;
            }
            b')' | b']' | b'}' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            _ => i += 1,
        }
    }
    max
}

/// `syn` parse behind the nesting guard: refuse what would overflow, and give
/// anything unusually deep a stack of its own before parsing it.
#[cfg(feature = "syn-parser")]
fn parse_guarded(source: &str) -> Option<Extracted> {
    match max_nesting_depth(source) {
        d if d > MAX_PARSE_NESTING => None,
        d if d <= INLINE_PARSE_NESTING => syn_ast::parse(source),
        _ => {
            let src = source.to_string();
            std::thread::Builder::new()
                .stack_size(DEEP_PARSE_STACK)
                .spawn(move || syn_ast::parse(&src))
                .ok()?
                .join()
                .ok()?
        }
    }
}

#[derive(Debug, Clone)]
pub struct ASTParser;

impl ASTParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse_source(&self, file_path: &str, source_code: &str) -> ParseResult {
        let hash = Self::compute_hash(source_code);
        let (modules, use_statements, symbols, call_sites, data_refs, import_aliases, method_defs) =
            Self::extract_all(source_code);

        ParseResult {
            file_path: file_path.to_string(),
            file_hash: hash,
            generated_nodes: modules.len() + use_statements.len() + symbols.len(),
            generated_edges: use_statements.len() + modules.len(),
            modules,
            use_statements,
            symbols,
            call_sites,
            data_refs,
            import_aliases,
            method_defs,
        }
    }

    /// Extract modules / `use` statements / symbols from source.
    ///
    /// With the `syn-parser` feature enabled, this parses a real Rust AST (which
    /// captures nested-module bodies, multi-line signatures, grouped `use` and
    /// impl methods). If the feature is off — or the source does not parse as
    /// valid Rust — it falls back to the zero-dependency line-based heuristic.
    fn extract_all(source: &str) -> Extracted {
        #[cfg(feature = "syn-parser")]
        {
            if let Some(parsed) = parse_guarded(source) {
                return parsed;
            }
        }
        // Heuristic fallback emits no call-sites, data-refs, import aliases, or method-defs (it does
        // not parse expression trees, `use`-tree renames, or impl blocks), so the call/data-flow
        // graphs simply stay empty on that path — a build can only ever *omit* those edges.
        (
            Self::extract_modules(source),
            Self::extract_uses(source),
            Self::extract_symbols(source),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    /// Build the causal graph for one file, storing **granular** content on every
    /// node so recall never spends a whole file's budget on a single node (see
    /// `docs/DESIGN_symbol_granularity.md`): the file node carries a *header*
    /// (path + one signature line per symbol), each symbol node carries its own
    /// source span, modules carry their declaration line, and `use` nodes the
    /// import line. The whole-file source is kept by the caller (`ExternalMemory`)
    /// for explicit retrieval, not duplicated into every node.
    pub fn update_memory_graph(&self, result: &ParseResult, source: &str, graph: &mut MemoryGraph) {
        let lines: Vec<&str> = source.lines().collect();
        let line_at = |ln: usize| {
            lines
                .get(ln.saturating_sub(1))
                .map(|l| l.trim().to_string())
                .unwrap_or_default()
        };

        // File node = a thin header: the path and a signature line per symbol,
        // capped at `header_symbol_cap()` lines so a huge file (syn's `item.rs` has
        // ~88 symbols) does not spend a third of a recall budget on its index alone
        // (see `docs/DESIGN_recall_budget.md`). Capped-out symbols are still their
        // own span nodes; the header just teases the first N.
        let file_id = NodeId(format!("file:{}", result.file_path));
        let cap = header_symbol_cap();
        let mut header = format!(
            "// {} — {} symbols\n",
            result.file_path,
            result.symbols.len()
        );
        let mut shown = 0usize;
        for s in &result.symbols {
            if shown >= cap {
                break;
            }
            let sig = line_at(s.line);
            if !sig.is_empty() {
                header.push_str(&sig);
                header.push('\n');
                shown += 1;
            }
        }
        if result.symbols.len() > shown {
            header.push_str(&format!("// … (+{} more)\n", result.symbols.len() - shown));
        }
        graph.upsert_node(
            file_id.clone(),
            result.file_path.clone(),
            header,
            NodeType::Module,
        );

        // Module nodes = their declaration line only; the items inside become their
        // own symbol nodes, so carrying the body here would just duplicate them.
        for module in &result.modules {
            let mod_id = NodeId(format!("mod:{}:{}", result.file_path, module.name));
            graph.upsert_node(
                mod_id.clone(),
                module.name.clone(),
                line_at(module.line),
                NodeType::Module,
            );
            graph.add_edge(file_id.clone(), mod_id.clone(), 0.9, EdgeType::Contains);
            self.add_module_tree(graph, &mod_id, &module.children, &result.file_path, &lines);
        }

        // Use statements = the import line itself.
        for use_stmt in &result.use_statements {
            let use_id = NodeId(format!("use:{}:{}", result.file_path, use_stmt.full_path));
            graph.upsert_node(
                use_id.clone(),
                format!("use {}", use_stmt.full_path),
                line_at(use_stmt.line),
                NodeType::Symbol,
            );
            graph.add_edge(file_id.clone(), use_id.clone(), 0.5, EdgeType::DependsOn);

            // Create dependency edges based on use path components
            if let Some(root) = use_stmt.components.first() {
                let dep_id = NodeId(format!("dep:{}", root));
                graph.upsert_node(
                    dep_id.clone(),
                    root.clone(),
                    format!("External dependency: {}", root),
                    NodeType::Symbol,
                );
                graph.add_edge(use_id.clone(), dep_id, 0.7, EdgeType::DependsOn);
            }
        }

        // Symbol nodes = the symbol's own source span (the granular recall unit).
        for symbol in &result.symbols {
            let sym_id = NodeId(format!("sym:{}:{}", result.file_path, symbol.name));
            let (start, end) = symbol_span(&lines, symbol.line);
            // Clamp into the slice before indexing: with `--features syn-parser`
            // a span line can land past EOF (trailing-newline / CRLF edge cases),
            // and `lines[start-1..end]` would otherwise panic out of bounds.
            let body = if lines.is_empty() {
                String::new()
            } else {
                let s = start.clamp(1, lines.len());
                let e = end.clamp(s, lines.len());
                lines[s - 1..e].join("\n")
            };
            graph.upsert_node(sym_id.clone(), symbol.name.clone(), body, NodeType::Symbol);
            graph.add_edge(file_id.clone(), sym_id.clone(), 0.6, EdgeType::Contains);
            // A `static`/`const` is the only valid `DataFlow` target; mark it so the resolver can
            // tell it apart from a function (the graph node stores `NodeType`, not `SymbolKind`).
            if matches!(symbol.kind, SymbolKind::Static | SymbolKind::Const) {
                graph.mark_data_symbol(sym_id.clone());
            }
        }

        // Hand this file's in-body call-sites to the graph; they are resolved into `caller →
        // callee` Calls edges by the whole-graph `resolve_symbol_calls` pass once every file is
        // ingested (a call may target a symbol defined in a not-yet-seen file). Replaces any
        // prior entry for this file, so a re-ingest re-states (never duplicates) its calls.
        graph.set_pending_calls(
            &result.file_path,
            result
                .call_sites
                .iter()
                .map(|c| (c.caller.clone(), c.callee.clone(), c.line))
                .collect(),
        );
        // Likewise hand over this file's `static`/`const` references, resolved into `reader → item`
        // DataFlow edges by the whole-graph `resolve_data_flow` pass after call resolution.
        graph.set_pending_data_refs(
            &result.file_path,
            result
                .data_refs
                .iter()
                .map(|d| (d.reader.clone(), d.name.clone(), d.line))
                .collect(),
        );
        // And this file's renamed-import bindings (`use a::b as c` → local `c` ↦ target `a::b`),
        // consulted by `resolve_symbol_calls` so a call to the alias resolves to the real target.
        graph.set_pending_aliases(
            &result.file_path,
            result
                .import_aliases
                .iter()
                .map(|a| (a.local.clone(), a.target.clone()))
                .collect(),
        );
        // And this file's impl-block method-ownership facts (`impl Foo { fn bar }` ⇒ `("Foo","bar")`),
        // the inputs `resolve_symbol_calls` uses to build the `(type, method) → symbol` index that
        // resolves a `Foo::bar` callee (Slice 3, `x.bar()` receiver-type inference).
        graph.set_pending_methods(
            &result.file_path,
            result
                .method_defs
                .iter()
                .map(|m| (m.type_name.clone(), m.method.clone()))
                .collect(),
        );
    }

    #[allow(clippy::only_used_in_recursion)]
    fn add_module_tree(
        &self,
        graph: &mut MemoryGraph,
        parent_id: &NodeId,
        children: &[ModuleDecl],
        file_path: &str,
        lines: &[&str],
    ) {
        for child in children {
            let child_id = NodeId(format!("mod:{}:{}", file_path, child.name));
            let decl = lines
                .get(child.line.saturating_sub(1))
                .map(|l| l.trim().to_string())
                .unwrap_or_default();
            graph.upsert_node(child_id.clone(), child.name.clone(), decl, NodeType::Module);
            graph.add_edge(
                parent_id.clone(),
                child_id.clone(),
                0.85,
                EdgeType::Contains,
            );

            if !child.children.is_empty() {
                self.add_module_tree(graph, &child_id, &child.children, file_path, lines);
            }
        }
    }

    fn compute_hash(source: &str) -> String {
        sha256_hex(source)
    }

    fn extract_modules(source: &str) -> Vec<ModuleDecl> {
        let mut modules = Vec::new();

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            let stripped = strip_comments(trimmed);

            if stripped.is_empty() {
                continue;
            }

            let is_pub = stripped.starts_with("pub mod ");
            let is_mod = (stripped.starts_with("mod ") || stripped.starts_with("pub mod "))
                && (stripped.ends_with(';') || stripped.ends_with('{'));

            if is_mod {
                let name = if is_pub {
                    stripped
                        .strip_prefix("pub mod ")
                        .unwrap_or("")
                        .trim()
                        .split([' ', '{', ';'])
                        .next()
                        .unwrap_or("")
                } else {
                    stripped
                        .strip_prefix("mod ")
                        .unwrap_or("")
                        .trim()
                        .split([' ', '{', ';'])
                        .next()
                        .unwrap_or("")
                };

                if name.is_empty() || name.contains("//") {
                    continue;
                }

                modules.push(ModuleDecl {
                    name: name.to_string(),
                    line: line_num + 1,
                    is_public: is_pub,
                    children: Vec::new(),
                });
            }
        }
        modules
    }

    fn extract_uses(source: &str) -> Vec<UseStatement> {
        let mut uses = Vec::new();

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            let stripped = strip_comments(trimmed);

            if stripped.is_empty() {
                continue;
            }

            if stripped.starts_with("use ") {
                let raw_path = stripped.strip_prefix("use ").unwrap_or("").trim();
                let path = raw_path.trim_end_matches(';').trim();

                let components: Vec<String> = path
                    .split("::")
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();

                let full_path = components.join("::");

                if !full_path.is_empty() {
                    uses.push(UseStatement {
                        full_path,
                        line: line_num + 1,
                        is_import: true,
                        components,
                    });
                }
            }
        }
        uses
    }

    fn extract_symbols(source: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        let mut seen = HashSet::new();

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            let stripped = strip_comments(trimmed);

            if stripped.is_empty() {
                continue;
            }

            let kind_and_name = if stripped.starts_with("fn ") {
                stripped.strip_prefix("fn ").and_then(|rest| {
                    let name = rest
                        .split(['(', '<', '{', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() || name.starts_with("//") {
                        None
                    } else {
                        Some((SymbolKind::Function, name))
                    }
                })
            } else if stripped.starts_with("pub fn ") {
                stripped.strip_prefix("pub fn ").and_then(|rest| {
                    let name = rest
                        .split(['(', '<', '{', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Function, name))
                    }
                })
            } else if stripped.starts_with("struct ") {
                stripped.strip_prefix("struct ").and_then(|rest| {
                    let name = rest
                        .split(['<', '{', '(', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Struct, name))
                    }
                })
            } else if stripped.starts_with("pub struct ") {
                stripped.strip_prefix("pub struct ").and_then(|rest| {
                    let name = rest
                        .split(['<', '{', '(', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Struct, name))
                    }
                })
            } else if stripped.starts_with("enum ") {
                stripped.strip_prefix("enum ").and_then(|rest| {
                    let name = rest
                        .split(['<', '{', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Enum, name))
                    }
                })
            } else if stripped.starts_with("pub enum ") {
                stripped.strip_prefix("pub enum ").and_then(|rest| {
                    let name = rest
                        .split(['<', '{', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Enum, name))
                    }
                })
            } else if stripped.starts_with("trait ") {
                stripped.strip_prefix("trait ").and_then(|rest| {
                    let name = rest
                        .split(['<', '{', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Trait, name))
                    }
                })
            } else if stripped.starts_with("pub trait ") {
                stripped.strip_prefix("pub trait ").and_then(|rest| {
                    let name = rest
                        .split(['<', '{', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Trait, name))
                    }
                })
            } else if stripped.starts_with("impl ") {
                stripped.strip_prefix("impl ").and_then(|rest| {
                    let name = rest
                        .split(['<', '{', ' ', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Impl, name))
                    }
                })
            } else if stripped.starts_with("const ") {
                stripped.strip_prefix("const ").and_then(|rest| {
                    let name = rest
                        .split([':', '=', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Const, name))
                    }
                })
            } else if stripped.starts_with("pub const ") {
                stripped.strip_prefix("pub const ").and_then(|rest| {
                    let name = rest
                        .split([':', '=', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Const, name))
                    }
                })
            } else if stripped.starts_with("static ") {
                stripped.strip_prefix("static ").and_then(|rest| {
                    let name = rest
                        .split([':', '=', ';'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Static, name))
                    }
                })
            } else if stripped.starts_with("type ") {
                stripped.strip_prefix("type ").and_then(|rest| {
                    let name = rest
                        .split(['=', ';', '<'])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Type, name))
                    }
                })
            } else if stripped.starts_with("macro_rules!") {
                stripped.strip_prefix("macro_rules!").and_then(|rest| {
                    let name = rest
                        .trim()
                        .split(['{', '(', ' '])
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if name.is_empty() {
                        None
                    } else {
                        Some((SymbolKind::Macro, name))
                    }
                })
            } else {
                None
            };

            if let Some((kind, name)) = kind_and_name {
                if seen.insert((kind.clone(), name.clone())) {
                    symbols.push(Symbol {
                        name,
                        line: line_num + 1,
                        kind,
                    });
                }
            }
        }
        symbols.sort_by_key(|s| s.line);
        symbols
    }
}

/// Strip comments from a single source line, ignoring `//` and `/* … */` that
/// appear inside string literals. Inline block comments (`a(); /* c */ b();`)
/// are removed in place. A block comment left open at end-of-line is dropped to
/// the line end — multi-line `/* … */` spans are a known limitation of the
/// line-based parser (see `ROADMAP.md`).
fn strip_comments(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push(c);
            }
            // Line comment: the rest of the line is dropped.
            '/' if chars.peek() == Some(&'/') => break,
            // Block comment: skip until the closing `*/` (or end of line).
            '/' if chars.peek() == Some(&'*') => {
                chars.next(); // consume '*'
                let mut prev = '\0';
                for d in chars.by_ref() {
                    if prev == '*' && d == '/' {
                        break;
                    }
                    prev = d;
                }
            }
            _ => out.push(c),
        }
    }
    out.trim().to_string()
}

impl Default for ASTParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Max signature lines a file-header node lists. Default 24; override with
/// `CCOS_HEADER_SYMBOLS`. Caps the header footprint of very large files so it
/// cannot dominate a recall budget; the omitted symbols remain their own nodes.
fn header_symbol_cap() -> usize {
    std::env::var("CCOS_HEADER_SYMBOLS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|x| *x >= 1)
        .unwrap_or(24)
}

/// Lexer state that has to survive a line break: Rust block comments nest, and
/// both plain and raw strings can span lines. Carrying this between lines is what
/// lets [`symbol_span`] scan a file line-by-line without losing its place.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ScanState {
    Code,
    /// Inside `/* … */`, with the current nesting count (Rust's block comments nest).
    BlockComment(usize),
    /// Inside `"…"`.
    Str,
    /// Inside `r#…"…"#…`, with the number of hashes that will close it.
    RawStr(usize),
}

/// Feed the **code** bytes of `line` to `on_code`, skipping everything that only
/// looks like code: comments, strings, raw strings, byte strings and char
/// literals. Returns the state the next line starts in.
///
/// Same discipline as [`max_nesting_depth`], applied per line so the caller can
/// keep track of line numbers. Whitespace is not reported.
fn scan_code_line(line: &str, entry: ScanState, mut on_code: impl FnMut(u8)) -> ScanState {
    let b = line.as_bytes();
    let mut i = 0usize;
    let mut state = entry;
    while i < b.len() {
        match state {
            ScanState::BlockComment(depth) => {
                if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                    state = ScanState::BlockComment(depth + 1);
                    i += 2;
                } else if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                    state = if depth <= 1 {
                        ScanState::Code
                    } else {
                        ScanState::BlockComment(depth - 1)
                    };
                    i += 2;
                } else {
                    i += 1;
                }
            }
            ScanState::Str => match b[i] {
                b'\\' => i += 2,
                b'"' => {
                    state = ScanState::Code;
                    i += 1;
                }
                _ => i += 1,
            },
            ScanState::RawStr(hashes) => {
                if b[i] == b'"'
                    && b[i + 1..]
                        .iter()
                        .take(hashes)
                        .filter(|c| **c == b'#')
                        .count()
                        == hashes
                {
                    state = ScanState::Code;
                    i += 1 + hashes;
                } else {
                    i += 1;
                }
            }
            ScanState::Code => match b[i] {
                b'/' if b.get(i + 1) == Some(&b'/') => return ScanState::Code,
                b'/' if b.get(i + 1) == Some(&b'*') => {
                    state = ScanState::BlockComment(1);
                    i += 2;
                }
                b'b' | b'r'
                    if i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_') =>
                {
                    let mut j = i;
                    let byte_prefix = b[j] == b'b';
                    if byte_prefix {
                        j += 1;
                    }
                    if b.get(j) == Some(&b'r') {
                        j += 1;
                        let hash_start = j;
                        while b.get(j) == Some(&b'#') {
                            j += 1;
                        }
                        if b.get(j) == Some(&b'"') {
                            state = ScanState::RawStr(j - hash_start);
                            i = j + 1;
                            continue;
                        }
                    } else if byte_prefix {
                        match b.get(j) {
                            Some(&b'"') => {
                                state = ScanState::Str;
                                i = j + 1;
                                continue;
                            }
                            Some(&b'\'') => {
                                i = quote_end(b, j);
                                continue;
                            }
                            _ => {}
                        }
                    }
                    on_code(b[i]);
                    i += 1;
                }
                b'"' => {
                    state = ScanState::Str;
                    i += 1;
                }
                b'\'' => {
                    let end = quote_end(b, i);
                    // A lifetime is code; a char literal is not. `quote_end`
                    // returns `i + 1` for the former.
                    if end == i + 1 {
                        on_code(b[i]);
                    }
                    i = end;
                }
                c => {
                    if !c.is_ascii_whitespace() {
                        on_code(c);
                    }
                    i += 1;
                }
            },
        }
    }
    state
}

/// Inclusive 1-based `[start, end]` line span of the item beginning at
/// `start_line`. Brace-matched for `{}`-bodied items (fn/struct/enum/trait/impl);
/// semicolon-terminated for the rest (const/static/type/use); a lone start line
/// otherwise. Capped at end-of-file.
///
/// Counts only braces that are **code**. It used to strip line comments and then
/// count every remaining brace, so a `}` inside a string, a raw string or a char
/// literal closed the item early: `pub fn a() { let c = '}'; … }` reported a
/// one-line body and the graph stored a fragment of the function under the
/// function's name. A stray `{` in a literal made the opposite error, swallowing
/// whatever followed. Multi-line strings and `/* … */` are handled too — the
/// scanner carries its state across lines instead of restarting on each one.
fn symbol_span(lines: &[&str], start_line: usize) -> (usize, usize) {
    let n = lines.len();
    if start_line == 0 || start_line > n {
        // Out-of-range start (e.g. a syn span past EOF): return an in-bounds
        // lone line so callers can slice without panicking.
        let line = start_line.clamp(1, n.max(1));
        return (line, line);
    }
    let s0 = start_line - 1; // 0-based
    let mut state = ScanState::Code;
    let mut depth: i32 = 0;
    let mut opened = false;
    for (i, line) in lines.iter().enumerate().skip(s0) {
        // The body's opening brace is looked for in a short signature window; an
        // item that has not opened one by the end of it is a lone line.
        let in_signature_window = i < s0 + 8;
        if !opened && !in_signature_window {
            return (start_line, start_line);
        }
        let mut ends_here = None;
        let mut last_code = None;
        state = scan_code_line(line, state, |c| {
            last_code = Some(c);
            match c {
                b'{' => {
                    depth += 1;
                    opened = true;
                }
                b'}' if opened => {
                    depth -= 1;
                    if depth == 0 && ends_here.is_none() {
                        ends_here = Some(i + 1);
                    }
                }
                _ => {}
            }
        });
        if let Some(end) = ends_here {
            return (start_line, end);
        }
        // A brace-less item (const/static/type/use) ends at the line whose last
        // code character is its terminating semicolon.
        if !opened && last_code == Some(b';') {
            return (start_line, i + 1);
        }
    }
    (start_line, n)
}

/// Real-AST parsing via `syn` (enabled by the `syn-parser` feature). Produces
/// the same `(modules, uses, symbols)` triple as the heuristic parser, but
/// accurately: it descends into nested-module bodies and impl blocks, expands
/// grouped `use` trees, handles multi-line signatures, and ignores comments
/// natively. Returns `None` on a syntax error so the caller can fall back.
#[cfg(feature = "syn-parser")]
mod syn_ast {
    use super::{
        CallSite, DataRef, ImportAlias, MethodDef, ModuleDecl, Symbol, SymbolKind, UseStatement,
    };
    use proc_macro2::Span;
    use std::collections::HashSet;
    use syn::spanned::Spanned;
    use syn::visit::Visit;

    /// Callers must go through [`super::parse_guarded`], never here directly: this
    /// function inherits `syn`'s unbounded recursion and will abort the process on
    /// a deeply nested source.
    pub fn parse(source: &str) -> Option<super::Extracted> {
        let file = syn::parse_file(source).ok()?;
        let mut out = Collected::default();
        walk(&file.items, &mut out);
        Some((
            out.modules,
            out.uses,
            out.symbols,
            out.calls,
            out.data_refs,
            out.aliases,
            out.method_defs,
        ))
    }

    #[derive(Default)]
    struct Collected {
        modules: Vec<ModuleDecl>,
        uses: Vec<UseStatement>,
        symbols: Vec<Symbol>,
        calls: Vec<CallSite>,
        data_refs: Vec<DataRef>,
        aliases: Vec<ImportAlias>,
        method_defs: Vec<MethodDef>,
    }

    /// Collects single-segment free-function call-sites from a function body, in source
    /// (document) order — a pure function of the AST, so call extraction is deterministic.
    struct CallVisitor<'a> {
        caller: String,
        /// Method / associated-fn names defined on the enclosing impl or trait (empty for free
        /// functions). A `self.m()` / `Self::m()` is captured ONLY when `m` is in this set, so a
        /// Deref- or blanket-trait-provided method (not defined here) is never mislinked to a
        /// same-named free function in this module.
        own_methods: &'a std::collections::HashSet<String>,
        /// Names bound *locally* in this function — parameters, `let`s, and fn-local `const`/`static`
        /// items. A `SCREAMING_SNAKE` data-reference whose name is locally bound is skipped: it
        /// denotes the local (invisible as a graph symbol), not a same-named module-level
        /// `static`/`const`, so resolving it global-unique would be a false edge.
        local_bound: &'a std::collections::HashSet<String>,
        /// Local name → concrete receiver type (Slice 3 `x.bar()` inference; see
        /// [`local_receiver_types`]). A bare-ident receiver found here emits a `Type::method` callee.
        local_types: &'a std::collections::BTreeMap<String, String>,
        /// File-scope typing facts (Slice 4): struct-field types and declared return types,
        /// consumed by [`receiver_type_of`](Self::receiver_type_of) for field/chain receivers.
        scope: &'a ScopeTypes,
        /// The enclosing impl's concrete Self type (`None` for free fns, trait default bodies, and
        /// non-concrete impls) — resolves a `self`-rooted field receiver to its owning type.
        self_type: Option<&'a str>,
        calls: Vec<CallSite>,
        data_refs: Vec<DataRef>,
    }
    impl CallVisitor<'_> {
        /// The concrete type a **receiver expression** evaluates to, if syntactically certain —
        /// the Slice 4 extension of Slice 3's bare-ident lookup. Handles, recursively:
        /// `self` / typed locals (Slice 3's map), field accesses `base.field` (same-scope declared
        /// field types), calls `f()` / `T::m()` (declared returns / ctor convention), method calls
        /// `base.m()` (declared returns — single- and multi-hop chains), `&expr`, and parens.
        /// Anything else — trait objects, generics, wrappers, unknown fields — yields `None`:
        /// resolve-uniquely-or-skip, never a guess. A returned `"Self"` (a `Self`-typed local) is
        /// resolved to the enclosing concrete type or refused.
        fn receiver_type_of(&self, expr: &syn::Expr) -> Option<String> {
            let concrete = |ty: String| -> Option<String> {
                if ty == "Self" {
                    self.self_type.map(str::to_string)
                } else {
                    Some(ty)
                }
            };
            match expr {
                syn::Expr::Paren(p) => self.receiver_type_of(&p.expr),
                syn::Expr::Reference(r) => self.receiver_type_of(&r.expr),
                syn::Expr::Path(p) if p.qself.is_none() => {
                    let id = p.path.get_ident()?;
                    if id == "self" {
                        return self.self_type.map(str::to_string);
                    }
                    concrete(self.local_types.get(&id.to_string()).cloned()?)
                }
                syn::Expr::Field(f) => {
                    let syn::Member::Named(field) = &f.member else {
                        return None;
                    };
                    let owner = self.receiver_type_of(&f.base)?;
                    self.scope
                        .field_types
                        .get(&(owner, field.to_string()))
                        .cloned()
                }
                syn::Expr::Call(c) => {
                    // Method's own generics were already applied when the facts were built; the
                    // ctor-convention fallback only needs the wrapper/PascalCase gate.
                    call_return_type(c, &HashSet::new(), self.scope, self.self_type)
                }
                syn::Expr::MethodCall(mc) => {
                    let owner = self.receiver_type_of(&mc.receiver)?;
                    self.scope
                        .method_returns
                        .get(&(owner, mc.method.to_string()))
                        .cloned()
                }
                _ => None,
            }
        }
    }
    impl<'a, 'ast> Visit<'ast> for CallVisitor<'a> {
        fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
            if let syn::Expr::Path(p) = &*node.func {
                // Capture the call path as a `::`-joined callee: bare `foo` (Slice 1) and qualified
                // `a::b::foo` (Slice 2). `<T>::x` (qself) and `::abs::x` (leading `::`) are skipped.
                // A `Self::assoc()` is captured only when `assoc` is defined on this type (Slice 3) —
                // else it is a trait/blanket assoc fn that must not match a same-named free fn.
                if p.qself.is_none() && p.path.leading_colon.is_none() {
                    if let Some(last) = p.path.segments.last() {
                        let segs = p
                            .path
                            .segments
                            .iter()
                            .map(|s| s.ident.to_string())
                            .collect::<Vec<_>>();
                        let is_self = segs.first().is_some_and(|s| s == "Self");
                        if !is_self || self.own_methods.contains(&last.ident.to_string()) {
                            self.calls.push(CallSite {
                                caller: self.caller.clone(),
                                callee: segs.join("::"),
                                line: line_of(last.ident.span()),
                            });
                        }
                    }
                }
            }
            // Always recurse so calls nested in args, closures, match arms, blocks are seen.
            syn::visit::visit_expr_call(self, node);
        }
        fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
            // Slice 3 — a **bare-ident receiver** `recv.method()`. Two cases, both resolve-or-skip:
            //   * `self.method()` → `Self::method`, captured ONLY when `method` is on this type
            //     (`own_methods`), so a Deref/blanket method is never mislinked to a same-named free fn.
            //   * `x.method()` where `x`'s concrete type is known (`local_types`, from receiver-type
            //     inference) → `Type::method`, resolved by the `(type, method)` index (unique-or-skip).
            //     A `Self`-typed local routes through the same `Self::` same-module gate as `self`.
            // Slice 4 — a **compound receiver**: field access (`self.field.m()`, `x.field.m()`),
            //   a call (`f().m()`, `T::make().m()`), or a method chain (`x.a().m()`), typed via the
            //   same-scope declared facts ([`ScopeTypes`]) → `Type::method`, through the same index.
            // An un-inferable receiver still drops — never a guess.
            let method = node.method.to_string();
            let callee = if let syn::Expr::Path(p) = &*node.receiver {
                if p.qself.is_none() {
                    p.path.get_ident().and_then(|recv| {
                        if recv == "self" {
                            self.own_methods
                                .contains(&method)
                                .then(|| format!("Self::{method}"))
                        } else {
                            match self.local_types.get(&recv.to_string()).map(String::as_str) {
                                Some("Self") => self
                                    .own_methods
                                    .contains(&method)
                                    .then(|| format!("Self::{method}")),
                                Some(ty) => Some(format!("{ty}::{method}")),
                                None => None,
                            }
                        }
                    })
                } else {
                    None
                }
            } else {
                self.receiver_type_of(&node.receiver)
                    .map(|ty| format!("{ty}::{method}"))
            };
            if let Some(callee) = callee {
                self.calls.push(CallSite {
                    caller: self.caller.clone(),
                    callee,
                    line: line_of(node.method.span()),
                });
            }
            syn::visit::visit_expr_method_call(self, node);
        }
        fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
            // Data-flow — a value path whose LAST segment is `SCREAMING_SNAKE` is, by Rust
            // convention, a `static`/`const` use. Capture it carrying its FULL `::`-joined path:
            //   * bare `FOO` (Slice 1), and
            //   * qualified `m::CONST` / `crate::limits::MAX` / `self::FOO` (Slice 2),
            // resolved later by [`crate::memory::MemoryGraph::resolve_data_flow`] using the same
            // import/module machinery as qualified calls. The casing test on the LAST segment
            // excludes PascalCase types/variants and snake_case fns/locals. `<T>::X` (qself) and a
            // leading `::X` are skipped, matching qualified-CALL capture.
            //
            // Scope guard: skip when the HEAD segment is locally bound (a param / `let` / fn-local
            // `const` — `local_bound`). For a bare ref the head IS the name, so a local `FOO`
            // shadows the same-named global (the false edge the guard closes); for a qualified ref a
            // locally-bound head (`m::CONST` where `m` is a local) likewise must not be captured.
            // (Known residual: a bare SCREAMING-cased enum variant brought in by `use` can still
            // coincide with a global const — rare, since variants are conventionally PascalCase.)
            if node.qself.is_none() && node.path.leading_colon.is_none() {
                if let (Some(first), Some(last)) =
                    (node.path.segments.first(), node.path.segments.last())
                {
                    let head = first.ident.to_string();
                    let last_name = last.ident.to_string();
                    if is_screaming_snake(&last_name) && !self.local_bound.contains(&head) {
                        let full = node
                            .path
                            .segments
                            .iter()
                            .map(|s| s.ident.to_string())
                            .collect::<Vec<_>>()
                            .join("::");
                        self.data_refs.push(DataRef {
                            reader: self.caller.clone(),
                            name: full,
                            line: line_of(last.ident.span()),
                        });
                    }
                }
            }
            syn::visit::visit_expr_path(self, node);
        }
    }

    /// A `static`/`const`-style identifier: at least one ASCII upper-case letter and nothing but
    /// upper-case letters, digits, and underscores (the Rust `SCREAMING_SNAKE_CASE` convention).
    fn is_screaming_snake(s: &str) -> bool {
        s.chars().any(|c| c.is_ascii_uppercase())
            && s.chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
    }

    /// The **concrete Self type name** an `impl` block is for — the final path segment of its
    /// `self_ty` (`impl a::b::Bar` and `impl Bar<T>` both key under `Bar`) — used to gate and union
    /// per-type method sets so a `self.foo()` resolves to *this type's* `foo` even when `foo` is
    /// defined in a different impl block of the same type.
    ///
    /// Returns `None` for any shape where a concrete type name cannot be determined, so the caller
    /// falls back to same-block gating rather than guessing: a reference/tuple/slice/`dyn`/`impl
    /// Trait`/qself receiver (`impl Trait for &Bar`, `for [T]`, …), or a `self_ty` that is itself one
    /// of the impl's **generic parameters** (`impl<T> Trait for T` — a blanket impl, where keying
    /// under `T` would wrongly union two unrelated blanket impls). The generic-parameter guard is the
    /// key no-cross-link safeguard for blanket impls.
    fn impl_self_type_name(imp: &syn::ItemImpl) -> Option<String> {
        let syn::Type::Path(tp) = &*imp.self_ty else {
            return None; // &Bar / (A, B) / [T] / dyn X / impl Trait / fn(..) — not a named type
        };
        if tp.qself.is_some() {
            return None; // <T as Trait>::Assoc — no single concrete owning type
        }
        let last = tp.path.segments.last()?.ident.to_string();
        // A bare self_ty that names one of the impl's own generic type params is a type *variable*
        // (`impl<T> .. for T`), not a concrete type — refuse it so blanket impls never cross-link.
        if tp.path.segments.len() == 1 && impl_generic_type_params(imp).contains(&last) {
            return None;
        }
        Some(last)
    }

    /// The names of an `impl`'s generic **type** parameters (`impl<T, U>` → {`T`, `U`}). Lifetimes
    /// and const generics are irrelevant to the type-variable check in [`impl_self_type_name`].
    fn impl_generic_type_params(imp: &syn::ItemImpl) -> HashSet<String> {
        imp.generics
            .params
            .iter()
            .filter_map(|p| match p {
                syn::GenericParam::Type(t) => Some(t.ident.to_string()),
                _ => None,
            })
            .collect()
    }

    /// Union, per concrete Self type, of every method/associated-fn name across **all** impl blocks
    /// of that type at one scope (inherent `impl Bar` *and* trait impls `impl Trait for Bar`), keyed
    /// by the type's final-segment name (see [`impl_self_type_name`]). A method body's `self.m()` /
    /// `Self::m` is then captured when `m` is in its enclosing type's *full* set, so a call resolves
    /// across impl blocks of the same type — while two **different** types each owning a same-named
    /// method stay in separate sets and never cross-link. Impls whose Self type is not a concrete
    /// name are skipped here (their methods fall back to same-block gating). `BTree*` ⇒ deterministic.
    fn build_type_method_sets(
        items: &[syn::Item],
    ) -> std::collections::BTreeMap<String, std::collections::BTreeSet<String>> {
        let mut map: std::collections::BTreeMap<String, std::collections::BTreeSet<String>> =
            std::collections::BTreeMap::new();
        for item in items {
            if let syn::Item::Impl(imp) = item {
                if let Some(ty) = impl_self_type_name(imp) {
                    let set = map.entry(ty).or_default();
                    for ii in &imp.items {
                        if let syn::ImplItem::Fn(m) = ii {
                            set.insert(m.sig.ident.to_string());
                        }
                    }
                }
            }
        }
        map
    }

    /// File-scope typing facts for Slice 4 receiver inference — field receivers
    /// (`self.field.bar()`) and return-type chains (`f().bar()`, `x.m().bar()`). Collected once per
    /// scope from the same syntactically-certain annotations Slice 3 trusts ([`annotation_type`]),
    /// never persisted: they only enrich the `Type::method` callee strings minted during call
    /// extraction, and every minted callee still passes the graph-wide `(type, method)`
    /// unique-or-skip index. `BTree*` ⇒ deterministic.
    #[derive(Default)]
    struct ScopeTypes {
        /// `(type name, field name) → concrete field type` — from named-struct declarations whose
        /// field annotation is concrete (PascalCase, non-generic, non-wrapper; `&T` peeled).
        field_types: std::collections::BTreeMap<(String, String), String>,
        /// `free fn name → concrete return type` at this scope.
        fn_returns: std::collections::BTreeMap<String, String>,
        /// `(type, method) → concrete return type` (`-> Self` resolved to the impl's concrete
        /// type). Poison-on-conflict: an inherent method and a same-named trait method with
        /// different returns yield nothing.
        method_returns: std::collections::BTreeMap<(String, String), String>,
        /// Every `(type, method)` DEFINED at this scope, inferable return or not. When the
        /// definition is visible but its return is not concrete, the `new`/`default`/`with_*`
        /// returns-`Self` *convention* is contradicted by evidence — so it must not fire.
        method_known: std::collections::BTreeSet<(String, String)>,
    }

    /// The concrete type a signature returns, if syntactically certain: the same
    /// [`annotation_type`] gate as every other Slice 3/4 inference, with `-> Self` resolved to the
    /// enclosing impl's concrete type (or refused for a free fn / non-concrete impl).
    fn return_type(
        output: &syn::ReturnType,
        generics: &HashSet<String>,
        self_ty: Option<&str>,
    ) -> Option<String> {
        let syn::ReturnType::Type(_, ty) = output else {
            return None;
        };
        let t = annotation_type(ty, generics)?;
        if t == "Self" {
            self_ty.map(str::to_string)
        } else {
            Some(t)
        }
    }

    /// Collect the [`ScopeTypes`] facts for one scope's items (struct fields, free-fn returns,
    /// method returns). Same-scope name collisions for structs/free fns are impossible in valid
    /// Rust; the one legal collision — an inherent method and a trait method sharing a name on the
    /// same type — is poisoned unless their returns agree.
    fn build_scope_types(items: &[syn::Item]) -> ScopeTypes {
        let mut st = ScopeTypes::default();
        let mut poisoned: HashSet<(String, String)> = HashSet::new();
        for item in items {
            match item {
                syn::Item::Struct(s) => {
                    let sg = generic_type_params(&s.generics);
                    if let syn::Fields::Named(named) = &s.fields {
                        for f in &named.named {
                            if let (Some(id), Some(t)) = (&f.ident, annotation_type(&f.ty, &sg)) {
                                st.field_types
                                    .insert((s.ident.to_string(), id.to_string()), t);
                            }
                        }
                    }
                }
                syn::Item::Fn(f) => {
                    let fg = generic_type_params(&f.sig.generics);
                    if let Some(t) = return_type(&f.sig.output, &fg, None) {
                        st.fn_returns.insert(f.sig.ident.to_string(), t);
                    }
                }
                syn::Item::Impl(i) => {
                    let Some(ty) = impl_self_type_name(i) else {
                        continue;
                    };
                    let ig = impl_generic_type_params(i);
                    for ii in &i.items {
                        let syn::ImplItem::Fn(m) = ii else { continue };
                        let key = (ty.clone(), m.sig.ident.to_string());
                        st.method_known.insert(key.clone());
                        let mut mg = ig.clone();
                        mg.extend(generic_type_params(&m.sig.generics));
                        let ret = return_type(&m.sig.output, &mg, Some(&ty));
                        if poisoned.contains(&key) {
                            continue;
                        }
                        match (st.method_returns.get(&key), ret) {
                            (None, Some(r)) => {
                                st.method_returns.insert(key, r);
                            }
                            (Some(prev), Some(r)) if *prev != r => {
                                st.method_returns.remove(&key);
                                poisoned.insert(key);
                            }
                            (Some(_), None) => {
                                st.method_returns.remove(&key);
                                poisoned.insert(key);
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        st
    }

    /// Std container/smart-pointer type names we never treat as a method receiver's concrete type:
    /// `x.bar()` on a `Box<Foo>` / `Vec<Foo>` / … dispatches to the wrapper's (or auto-deref'd) method,
    /// not to a local `impl`, so inferring the wrapper — or unwrapping its argument — could only
    /// mislead the `(type, method)` index. Skipping them keeps receiver-type inference precise.
    const NON_RECEIVER_WRAPPERS: &[&str] = &[
        "Box", "Rc", "Arc", "Option", "Result", "Vec", "VecDeque", "RefCell", "Cell", "Mutex",
        "RwLock", "Cow", "Pin", "Weak", "HashMap", "BTreeMap", "HashSet", "BTreeSet",
    ];

    /// The generic **type-parameter** names declared by `generics` (`<T, U, const N: usize>` ⇒
    /// `{T, U}`). A receiver whose type names one of these is a type variable, not a concrete graph
    /// symbol, so it must never be inferred (it would cross-link to a same-named concrete type).
    fn generic_type_params(generics: &syn::Generics) -> HashSet<String> {
        generics
            .params
            .iter()
            .filter_map(|p| match p {
                syn::GenericParam::Type(t) => Some(t.ident.to_string()),
                _ => None,
            })
            .collect()
    }

    /// Accept `name` as a concrete receiver type only when it is **type-like** (PascalCase head — the
    /// Rust convention that separates a type `Foo` from a module `foo`, closing the `Foo::new()`
    /// type-constructor vs `foo::new()` module-function ambiguity), is **not a generic param**, and is
    /// **not a std wrapper**. `Self` passes (PascalCase) and is resolved same-module by the existing
    /// `Self::` path. Precision-first: anything else yields `None` (no inference, never a guess).
    fn receiver_type_name(name: String, generics: &HashSet<String>) -> Option<String> {
        let type_like = name.chars().next().is_some_and(|c| c.is_ascii_uppercase());
        if type_like && !generics.contains(&name) && !NON_RECEIVER_WRAPPERS.contains(&name.as_str())
        {
            Some(name)
        } else {
            None
        }
    }

    /// The concrete receiver type named by a `syn::Type` annotation, if syntactically certain: a plain
    /// `Type::Path` (no qself) whose final segment passes [`receiver_type_name`]. `Foo`, `a::b::Foo`,
    /// and `Foo<T>` all key under `Foo` (final segment, generics stripped) — matching how
    /// [`impl_self_type_name`] keys impl blocks, so a receiver type and its impl agree. References,
    /// wrappers, trait objects, tuples, `impl`/`dyn Trait`, and generic params → `None`.
    fn annotation_type(ty: &syn::Type, generics: &HashSet<String>) -> Option<String> {
        // Peel leading references: `x: &T` / `x: &mut T` call methods on `T` (auto-ref of `&self`), so
        // the receiver type is `T` — the same inference, and the same precision, as the owned `x: T`
        // case (the `(type, method)` index's cardinality guard still gates every resulting edge).
        if let syn::Type::Reference(r) = ty {
            return annotation_type(&r.elem, generics);
        }
        let syn::Type::Path(tp) = ty else {
            return None;
        };
        if tp.qself.is_some() {
            return None;
        }
        receiver_type_name(tp.path.segments.last()?.ident.to_string(), generics)
    }

    /// The concrete receiver type of a binding's **RHS expression**:
    ///
    /// * a **single-segment** struct literal `Foo { .. }` (the type is named literally; a
    ///   multi-segment path — `Enum::Variant { .. }`, `m::Struct { .. }` — is ambiguous, skipped);
    /// * a call — resolved by [`call_return_type`]: a same-scope free fn / method with a concrete
    ///   declared return (Slice 4), else the `Foo::new()`/`default()`/`with_*()`
    ///   returns-`Self`-by-convention idiom, which is *suppressed* when the definition is visible
    ///   at this scope and contradicts it.
    ///
    /// Everything else (method-chain RHS on an untyped receiver, `from`/`try_from`, arbitrary
    /// assoc fns) → `None`.
    fn rhs_type(
        expr: &syn::Expr,
        generics: &HashSet<String>,
        scope: &ScopeTypes,
        self_type: Option<&str>,
    ) -> Option<String> {
        match expr {
            syn::Expr::Call(call) => call_return_type(call, generics, scope, self_type),
            syn::Expr::Struct(s) if s.qself.is_none() && s.path.segments.len() == 1 => {
                receiver_type_name(s.path.segments[0].ident.to_string(), generics)
            }
            _ => None,
        }
    }

    /// The concrete type a call expression evaluates to, if certain. Single-segment `f(..)` reads
    /// the same-scope free-fn return facts; two-segment `T::m(..)` (with `Self::` resolved to the
    /// enclosing concrete type) prefers the **declared** same-scope return, refuses when the
    /// definition is visible but not concretely typed (evidence beats convention), and only then
    /// falls back to the `new`/`default`/`with_*` returns-`Self` convention for out-of-scope types.
    fn call_return_type(
        call: &syn::ExprCall,
        generics: &HashSet<String>,
        scope: &ScopeTypes,
        self_type: Option<&str>,
    ) -> Option<String> {
        let syn::Expr::Path(p) = &*call.func else {
            return None;
        };
        if p.qself.is_some() || p.path.leading_colon.is_some() {
            return None;
        }
        match p.path.segments.len() {
            1 => scope
                .fn_returns
                .get(&p.path.segments[0].ident.to_string())
                .cloned(),
            2 => {
                let head = p.path.segments[0].ident.to_string();
                let method = p.path.segments[1].ident.to_string();
                let owner = if head == "Self" {
                    self_type?.to_string()
                } else {
                    head
                };
                let key = (owner, method);
                if let Some(ret) = scope.method_returns.get(&key) {
                    return Some(ret.clone());
                }
                let (owner, method) = key;
                if scope
                    .method_known
                    .contains(&(owner.clone(), method.clone()))
                {
                    return None; // defined here with a non-concrete return — convention refuted
                }
                if method == "new" || method == "default" || method.starts_with("with_") {
                    receiver_type_name(owner, generics)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Collects the **receiver-type map** `local name → concrete type` for one function body — the
    /// inputs to `x.bar()` inference. Populated only from the high-confidence idioms ([`annotation_type`]
    /// / [`rhs_type`]). Flow-insensitive and **poison-on-conflict**: a name bound to two different
    /// types, re-`let` to an un-inferable type, or reassigned (`x = ..`) anywhere in the body is dropped
    /// entirely — so a shadow or rebind can never mint a false edge (the map only ever yields a type it
    /// is *certain* of, exactly the precision-first trade the call graph requires).
    struct ReceiverTypeCollector<'g> {
        generics: &'g HashSet<String>,
        scope: &'g ScopeTypes,
        self_type: Option<&'g str>,
        types: std::collections::BTreeMap<String, String>,
        poisoned: HashSet<String>,
    }
    impl ReceiverTypeCollector<'_> {
        fn record(&mut self, name: String, ty: Option<String>) {
            if self.poisoned.contains(&name) {
                return;
            }
            match ty {
                Some(t) => match self.types.get(&name) {
                    Some(prev) if *prev != t => {
                        self.types.remove(&name);
                        self.poisoned.insert(name);
                    }
                    Some(_) => {}
                    None => {
                        self.types.insert(name, t);
                    }
                },
                // A later rebind whose type we cannot infer invalidates an earlier inferred type.
                None => {
                    if self.types.remove(&name).is_some() {
                        self.poisoned.insert(name);
                    }
                }
            }
        }
    }
    impl<'ast> Visit<'ast> for ReceiverTypeCollector<'_> {
        fn visit_local(&mut self, l: &'ast syn::Local) {
            let (name, annot) = match &l.pat {
                syn::Pat::Ident(pi) if pi.subpat.is_none() => (Some(pi.ident.to_string()), None),
                syn::Pat::Type(pt) => match &*pt.pat {
                    syn::Pat::Ident(pi) if pi.subpat.is_none() => {
                        (Some(pi.ident.to_string()), Some(&*pt.ty))
                    }
                    _ => (None, None),
                },
                _ => (None, None),
            };
            if let Some(name) = name {
                // Annotation wins; otherwise infer from the initialiser expression.
                let ty = annot
                    .and_then(|t| annotation_type(t, self.generics))
                    .or_else(|| {
                        l.init.as_ref().and_then(|i| {
                            rhs_type(&i.expr, self.generics, self.scope, self.self_type)
                        })
                    });
                self.record(name, ty);
            }
            syn::visit::visit_local(self, l);
        }
        fn visit_expr_assign(&mut self, a: &'ast syn::ExprAssign) {
            // Flow-insensitive: any reassignment of a bare local drops its inferred type (the value
            // after `x = ..` may be a type we cannot track). Conservative, never guesses.
            if let syn::Expr::Path(p) = &*a.left {
                if let Some(id) = p.path.get_ident() {
                    self.types.remove(&id.to_string());
                    self.poisoned.insert(id.to_string());
                }
            }
            syn::visit::visit_expr_assign(self, a);
        }
    }

    /// The receiver-type map for `x.bar()` inference in one function body — see
    /// [`ReceiverTypeCollector`]. `enclosing_generics` carries the impl/trait block's type params (the
    /// method `sig`'s own params are unioned in here) so a generic receiver is never mistaken for a
    /// concrete type. Deterministic (`BTreeMap`, source-order traversal).
    fn local_receiver_types(
        sig: &syn::Signature,
        block: &syn::Block,
        enclosing_generics: &HashSet<String>,
        scope: &ScopeTypes,
        self_type: Option<&str>,
    ) -> std::collections::BTreeMap<String, String> {
        let mut generics = enclosing_generics.clone();
        generics.extend(generic_type_params(&sig.generics));
        let mut c = ReceiverTypeCollector {
            generics: &generics,
            scope,
            self_type,
            types: std::collections::BTreeMap::new(),
            poisoned: HashSet::new(),
        };
        // Typed value params (`fn f(x: Foo)`); `self`/destructured/wrapper params yield no entry.
        for input in &sig.inputs {
            if let syn::FnArg::Typed(pt) = input {
                if let syn::Pat::Ident(pi) = &*pt.pat {
                    if pi.subpat.is_none() {
                        let ty = annotation_type(&pt.ty, &generics);
                        c.record(pi.ident.to_string(), ty);
                    }
                }
            }
        }
        c.visit_block(block);
        c.types
    }

    /// Names bound *locally* in a function — its parameters plus every `let`, fn-local
    /// `const`/`static`, closure parameter, and pattern binding in its body. Used to suppress
    /// data-references that denote a local rather than a module-level `static`/`const`.
    /// **Conservative**: collects across the whole body regardless of nested scope, so it can only
    /// ever drop a data-edge, never invent one — exactly the precision-first trade we want.
    fn local_bound_names(sig: &syn::Signature, block: &syn::Block) -> HashSet<String> {
        struct BindingCollector {
            names: HashSet<String>,
        }
        impl<'ast> Visit<'ast> for BindingCollector {
            fn visit_pat_ident(&mut self, p: &'ast syn::PatIdent) {
                self.names.insert(p.ident.to_string());
                syn::visit::visit_pat_ident(self, p);
            }
            fn visit_item_const(&mut self, c: &'ast syn::ItemConst) {
                self.names.insert(c.ident.to_string());
                syn::visit::visit_item_const(self, c);
            }
            fn visit_item_static(&mut self, s: &'ast syn::ItemStatic) {
                self.names.insert(s.ident.to_string());
                syn::visit::visit_item_static(self, s);
            }
        }
        let mut c = BindingCollector {
            names: HashSet::new(),
        };
        for input in &sig.inputs {
            if let syn::FnArg::Typed(pt) = input {
                c.visit_pat(&pt.pat);
            }
        }
        c.visit_block(block);
        c.names
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_calls(
        caller: &str,
        own_methods: &std::collections::HashSet<String>,
        enclosing_generics: &std::collections::HashSet<String>,
        sig: &syn::Signature,
        block: &syn::Block,
        scope: &ScopeTypes,
        self_type: Option<&str>,
        calls_out: &mut Vec<CallSite>,
        refs_out: &mut Vec<DataRef>,
    ) {
        let local_bound = local_bound_names(sig, block);
        let local_types = local_receiver_types(sig, block, enclosing_generics, scope, self_type);
        let mut v = CallVisitor {
            caller: caller.to_string(),
            own_methods,
            local_bound: &local_bound,
            local_types: &local_types,
            scope,
            self_type,
            calls: Vec::new(),
            data_refs: Vec::new(),
        };
        v.visit_block(block);
        calls_out.append(&mut v.calls);
        refs_out.append(&mut v.data_refs);
    }

    /// 1-based source line; the `span-locations` feature guarantees real spans.
    fn line_of(span: Span) -> usize {
        span.start().line
    }

    fn is_pub(vis: &syn::Visibility) -> bool {
        matches!(vis, syn::Visibility::Public(_))
    }

    fn push_sym(out: &mut Collected, ident: &syn::Ident, kind: SymbolKind) {
        out.symbols.push(Symbol {
            name: ident.to_string(),
            line: line_of(ident.span()),
            kind,
        });
    }

    /// Walk a list of items at one scope. Nested modules become `children` of
    /// their parent; symbols and `use`s from nested scopes are surfaced into the
    /// flat lists (matching the line parser, which sees every line).
    fn walk(items: &[syn::Item], out: &mut Collected) {
        // Per-type method sets, unioned across every impl block of each concrete type at THIS scope
        // (inherent + trait impls). Used below to gate `self.m()` / `Self::m` capture on the type's
        // FULL method set, so a self-call resolves across sibling impl blocks of the same type.
        let type_methods = build_type_method_sets(items);
        // Declared field/return types at THIS scope (Slice 4 field/chain receiver inference).
        let scope_types = build_scope_types(items);
        for item in items {
            match item {
                syn::Item::Mod(m) => {
                    let mut child = Collected::default();
                    if let Some((_, inner)) = &m.content {
                        walk(inner, &mut child);
                    }
                    out.uses.append(&mut child.uses);
                    out.symbols.append(&mut child.symbols);
                    out.aliases.append(&mut child.aliases);
                    out.modules.push(ModuleDecl {
                        name: m.ident.to_string(),
                        line: line_of(m.ident.span()),
                        is_public: is_pub(&m.vis),
                        children: child.modules,
                    });
                }
                syn::Item::Use(u) => {
                    flatten_use(
                        &u.tree,
                        String::new(),
                        line_of(u.span()),
                        &mut out.uses,
                        &mut out.aliases,
                    );
                }
                syn::Item::Fn(f) => {
                    push_sym(out, &f.sig.ident, SymbolKind::Function);
                    // A free function has no `self`/`Self` methods in scope → empty own-method set, and
                    // no enclosing-block generics (its own `sig` generics are handled by collect_calls).
                    collect_calls(
                        &f.sig.ident.to_string(),
                        &HashSet::new(),
                        &HashSet::new(),
                        &f.sig,
                        &f.block,
                        &scope_types,
                        None,
                        &mut out.calls,
                        &mut out.data_refs,
                    );
                }
                syn::Item::Struct(s) => push_sym(out, &s.ident, SymbolKind::Struct),
                syn::Item::Enum(e) => push_sym(out, &e.ident, SymbolKind::Enum),
                syn::Item::Trait(t) => {
                    push_sym(out, &t.ident, SymbolKind::Trait);
                    let methods: HashSet<String> = t
                        .items
                        .iter()
                        .filter_map(|ti| match ti {
                            syn::TraitItem::Fn(m) => Some(m.sig.ident.to_string()),
                            _ => None,
                        })
                        .collect();
                    let trait_generics = generic_type_params(&t.generics);
                    for ti in &t.items {
                        if let syn::TraitItem::Fn(method) = ti {
                            push_sym(out, &method.sig.ident, SymbolKind::Function);
                            if let Some(body) = &method.default {
                                collect_calls(
                                    &method.sig.ident.to_string(),
                                    &methods,
                                    &trait_generics,
                                    &method.sig,
                                    body,
                                    &scope_types,
                                    None,
                                    &mut out.calls,
                                    &mut out.data_refs,
                                );
                            }
                        }
                    }
                }
                syn::Item::Const(c) => push_sym(out, &c.ident, SymbolKind::Const),
                syn::Item::Static(s) => push_sym(out, &s.ident, SymbolKind::Static),
                syn::Item::Type(t) => push_sym(out, &t.ident, SymbolKind::Type),
                syn::Item::Macro(m) => {
                    if let Some(ident) = &m.ident {
                        push_sym(out, ident, SymbolKind::Macro);
                    }
                }
                syn::Item::Impl(i) => {
                    // Gate `self.m()` / `Self::m` on the enclosing type's FULL (unioned) method set
                    // when the Self type is a concrete name — so a self-call resolves to a method
                    // defined in a *different* impl block (inherent or trait) of the same type. If
                    // the Self type is not a concrete name (generic param / reference / tuple / …),
                    // fall back to THIS block's own methods only — never guess a cross-block link.
                    let own_block: HashSet<String> = i
                        .items
                        .iter()
                        .filter_map(|ii| match ii {
                            syn::ImplItem::Fn(m) => Some(m.sig.ident.to_string()),
                            _ => None,
                        })
                        .collect();
                    let self_ty = impl_self_type_name(i);
                    let methods: HashSet<String> = match &self_ty {
                        Some(ty) => type_methods
                            .get(ty)
                            .map(|s| s.iter().cloned().collect())
                            .unwrap_or(own_block),
                        None => own_block,
                    };
                    let impl_generics = generic_type_params(&i.generics);
                    for ii in &i.items {
                        if let syn::ImplItem::Fn(method) = ii {
                            push_sym(out, &method.sig.ident, SymbolKind::Function);
                            // Index this method under its concrete Self type (Slice 3) — the input to
                            // the `(type, method)` resolution index for `x.bar()` / `Type::assoc()`.
                            // A non-concrete Self (generic param, reference, blanket impl) yields `None`
                            // and contributes nothing, so no unrelated receiver type is ever linked.
                            if let Some(ty) = &self_ty {
                                out.method_defs.push(MethodDef {
                                    type_name: ty.clone(),
                                    method: method.sig.ident.to_string(),
                                });
                            }
                            collect_calls(
                                &method.sig.ident.to_string(),
                                &methods,
                                &impl_generics,
                                &method.sig,
                                &method.block,
                                &scope_types,
                                self_ty.as_deref(),
                                &mut out.calls,
                                &mut out.data_refs,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Expand a (possibly grouped) `use` tree into one `UseStatement` per leaf
    /// path, e.g. `use a::{b, c::d}` → `a::b` and `a::c::d`. A renamed leaf
    /// (`use a::b as c`, including inside a group) additionally records an
    /// [`ImportAlias`] binding `c ↦ a::b` in `aliases`.
    fn flatten_use(
        tree: &syn::UseTree,
        prefix: String,
        line: usize,
        out: &mut Vec<UseStatement>,
        aliases: &mut Vec<ImportAlias>,
    ) {
        let join = |p: &str, s: &str| {
            if p.is_empty() {
                s.to_string()
            } else {
                format!("{p}::{s}")
            }
        };
        match tree {
            syn::UseTree::Path(p) => flatten_use(
                &p.tree,
                join(&prefix, &p.ident.to_string()),
                line,
                out,
                aliases,
            ),
            syn::UseTree::Name(n) => push_use(join(&prefix, &n.ident.to_string()), line, out),
            // A renamed import `use a::b as c` binds the target `a::b` under the LOCAL name `c`.
            // Record the ORIGINAL path (`a::b`) as the `UseStatement` so import-linking still
            // resolves the real module (identical to a plain `use a::b`), AND record the alias
            // binding `c ↦ a::b` so `resolve_symbol_calls` can rewrite a call to `c` onto the real
            // target — never onto a same-named sibling module/symbol `c`.
            syn::UseTree::Rename(r) => {
                let target = join(&prefix, &r.ident.to_string());
                aliases.push(ImportAlias {
                    local: r.rename.to_string(),
                    target: target.clone(),
                });
                push_use(target, line, out);
            }
            syn::UseTree::Glob(_) => push_use(join(&prefix, "*"), line, out),
            syn::UseTree::Group(g) => {
                for t in &g.items {
                    flatten_use(t, prefix.clone(), line, out, aliases);
                }
            }
        }
    }

    fn push_use(full_path: String, line: usize, out: &mut Vec<UseStatement>) {
        let components: Vec<String> = full_path.split("::").map(str::to_string).collect();
        out.push(UseStatement {
            full_path,
            line,
            is_import: true,
            components,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source file nested past what `syn`'s recursion can survive must degrade to
    /// the heuristic parser, not abort the process.
    ///
    /// Before the nesting guard this did not *fail* — it killed the whole test
    /// binary with `thread 'main' has overflowed its stack / fatal runtime error:
    /// stack overflow` (SIGABRT, rc=134), which is precisely the point: a stack
    /// overflow is not catchable, so one such file ingested by an agent takes the
    /// process — and any un-checkpointed memory — down with it.
    #[test]
    fn deeply_nested_source_falls_back_instead_of_overflowing() {
        for (open, close) in [("(", ")"), ("[", "]"), ("{ (", ") }")] {
            let source = format!(
                "pub fn deep() {{ let x = {}1{}; }}\n",
                open.repeat(37_251),
                close.repeat(37_251)
            );
            let result = ASTParser::new().parse_source("deep.rs", &source);
            assert!(
                result.symbols.iter().any(|s| s.name == "deep"),
                "heuristic fallback should still see the function for {open:?}"
            );
        }
    }

    /// Just over the limit, the AST is refused; just under, it is still used. The
    /// two are told apart by `method_defs`, which only the `syn` path ever emits.
    #[test]
    fn the_nesting_limit_is_where_the_ast_stops() {
        let nested = |depth: usize| {
            format!(
                "pub struct Foo;\npub fn top() {{ let x = {}1{}; }}\nimpl Foo {{ pub fn bar() {{}} }}\n",
                "(".repeat(depth),
                ")".repeat(depth)
            )
        };
        let under = ASTParser::new().parse_source("u.rs", &nested(MAX_PARSE_NESTING - 4));
        let over = ASTParser::new().parse_source("o.rs", &nested(MAX_PARSE_NESTING + 4));

        // Both paths keep finding the top-level symbols; only the depth of the analysis changes.
        assert!(under.symbols.iter().any(|s| s.name == "top"));
        assert!(over.symbols.iter().any(|s| s.name == "top"));
        #[cfg(feature = "syn-parser")]
        {
            // Under the limit the parse runs on the dedicated large-stack thread
            // (the depth is above INLINE_PARSE_NESTING) and still yields real AST facts.
            assert_eq!(
                under.method_defs.len(),
                1,
                "AST facts expected under the limit"
            );
            assert!(
                over.method_defs.is_empty(),
                "AST must be refused over the limit"
            );
        }
    }

    /// The depth scan is a lexer, not a byte count: delimiters inside comments,
    /// strings, raw strings, byte strings and char literals do not nest anything,
    /// and `'a` is a lifetime while `'{'` is a char. Getting this wrong would
    /// silently downgrade ordinary files to the heuristic parser.
    #[test]
    fn nesting_depth_ignores_delimiters_that_are_not_code() {
        assert_eq!(max_nesting_depth(""), 0);
        assert_eq!(max_nesting_depth("fn f() {}"), 1); // sibling delimiters, not nested
        assert_eq!(max_nesting_depth("fn f() { g(h(1)); }"), 3);

        for noise in [
            r#"let s = "((((((((((((";"#,
            r#"let s = r"((((((((((((";"#,
            r##"let s = r#"(((((((((((("#;"##,
            r#"let s = b"((((((((((((";"#,
            "// ((((((((((((\n",
            "/* (((((((((((( */",
            "/* /* (((( */ (((( */",
            r"let c = '(';",
            r"let c = '{';",
            r#"let c = '"';"#,
            r"let c = '\'';",
            r"let c = '\\';",
            r"let c = 'é';",
            r"let b = b'(';",
        ] {
            assert_eq!(
                max_nesting_depth(&format!("fn f() {{ {noise} }}")),
                1,
                "delimiters inside {noise:?} must not count"
            );
        }

        // A lifetime is not a char literal: the rest of the line must keep counting.
        assert_eq!(max_nesting_depth("fn f<'a>(x: &'a str) { g(1); }"), 2);
        // Identifiers ending in `b`/`r` must not be mistaken for string prefixes.
        assert_eq!(max_nesting_depth("let arr = sub(f(1));"), 2);
        // Unbalanced closers saturate at zero instead of wrapping around.
        assert_eq!(max_nesting_depth(")))))))(((("), 4);
    }

    #[test]
    fn test_extract_modules_basic() {
        let source = "mod foo;\nmod bar;\npub mod baz;";
        let modules = ASTParser::extract_modules(source);
        assert_eq!(modules.len(), 3);
        assert!(modules.iter().any(|m| m.name == "foo"));
        assert!(modules.iter().any(|m| m.name == "bar"));
        assert!(modules.iter().any(|m| m.name == "baz"));
    }

    #[test]
    fn test_extract_uses_basic() {
        let source = "use std::collections::HashMap;\nuse crate::foo::bar;";
        let uses = ASTParser::extract_uses(source);
        assert_eq!(uses.len(), 2);
        assert_eq!(uses[0].full_path, "std::collections::HashMap");
    }

    #[test]
    fn test_extract_symbols_functions() {
        let source = "fn main() {}\nfn helper() {}\npub fn public_fn() {}";
        let symbols = ASTParser::extract_symbols(source);
        assert!(symbols
            .iter()
            .any(|s| s.name == "main" && s.kind == SymbolKind::Function));
        assert!(symbols.iter().any(|s| s.name == "helper"));
        assert!(symbols.iter().any(|s| s.name == "public_fn"));
    }

    #[test]
    fn test_strip_comments() {
        assert_eq!(strip_comments("use foo; // bar"), "use foo;");
        assert_eq!(
            strip_comments("let x = \"//not_a_comment\"; // real"),
            "let x = \"//not_a_comment\";"
        );
    }

    #[test]
    fn test_strip_block_comments() {
        // Inline block comment removed in place.
        assert_eq!(
            strip_comments("pub fn a() {} /* fn fake() {} */"),
            "pub fn a() {}"
        );
        // Block comment between tokens.
        assert_eq!(strip_comments("mod /* x */ real;"), "mod  real;");
        // Unterminated block comment dropped to end of line.
        assert_eq!(strip_comments("use foo; /* trailing"), "use foo;");
        // `/*` inside a string is not a comment.
        assert_eq!(
            strip_comments("let u = \"http:/*not*/\";"),
            "let u = \"http:/*not*/\";"
        );
    }

    #[test]
    fn test_block_comment_hides_fake_symbols() {
        // A function hidden in a block comment must NOT be extracted as a symbol.
        let source = "fn real() {}\n/* fn fake() {} */ struct Keep;";
        let symbols = ASTParser::extract_symbols(source);
        assert!(symbols.iter().any(|s| s.name == "real"));
        assert!(symbols.iter().any(|s| s.name == "Keep"));
        assert!(
            !symbols.iter().any(|s| s.name == "fake"),
            "symbol inside a block comment must be ignored"
        );
    }

    #[test]
    fn test_extract_structs_and_enums() {
        let source = "struct Foo;\nenum Bar { A, B }\npub struct Baz<T> {}";
        let symbols = ASTParser::extract_symbols(source);
        assert!(symbols
            .iter()
            .any(|s| s.name == "Foo" && s.kind == SymbolKind::Struct));
        assert!(symbols
            .iter()
            .any(|s| s.name == "Bar" && s.kind == SymbolKind::Enum));
        assert!(symbols.iter().any(|s| s.name == "Baz"));
    }

    #[test]
    fn test_graph_update_from_parse() {
        let source = "mod foo;\nuse std::io;\nfn main() {}";
        let parser = ASTParser::new();
        let result = parser.parse_source("test.rs", source);
        let mut graph = MemoryGraph::default();
        parser.update_memory_graph(&result, source, &mut graph);
        assert!(graph.node_count() > 3);
    }

    #[test]
    fn file_header_caps_its_symbol_list() {
        let mut src = String::new();
        for i in 0..50 {
            src.push_str(&format!("pub fn f{i}() {{}}\n"));
        }
        let parser = ASTParser::new();
        let result = parser.parse_source("t.rs", &src);
        let mut graph = MemoryGraph::default();
        parser.update_memory_graph(&result, &src, &mut graph);
        let header = &graph
            .nodes
            .get(&NodeId("file:t.rs".to_string()))
            .expect("file node")
            .content;
        // Default cap is 24 lines + a "(+N more)" marker, not all 50 signatures.
        assert!(
            header.contains("(+26 more)"),
            "header must note the omitted symbols: {header}"
        );
        assert!(
            !header.contains("f49"),
            "header must not list every symbol of a large file"
        );
    }

    /// A brace that is not code must not close the item.
    ///
    /// The span used to be measured by stripping line comments and counting every
    /// remaining brace, so `'}'`, `"}"` and `r#"}"#` all ended the function at the
    /// line they appeared on. Measured before the fix, each of the first three
    /// cases below reported `(1, 2)` where the body runs to line 4 — the graph
    /// stored two lines of a four-line function under that function's name, and a
    /// recall returned the fragment with nothing marking it as one.
    #[test]
    fn a_brace_inside_a_literal_does_not_end_the_symbol() {
        let cases: [(&str, &str, (usize, usize)); 9] = [
            (
                "char literal",
                "pub fn a() {\n    let c = '}';\n    let d = 1;\n}\nfn after() {}\n",
                (1, 4),
            ),
            (
                "string",
                "pub fn a() {\n    let s = \"}\";\n    let d = 1;\n}\nfn after() {}\n",
                (1, 4),
            ),
            (
                "raw string",
                "pub fn a() {\n    let s = r#\"}\"#;\n    let d = 1;\n}\nfn after() {}\n",
                (1, 4),
            ),
            (
                "byte string",
                "pub fn a() {\n    let s = b\"}\";\n    let d = 1;\n}\nfn after() {}\n",
                (1, 4),
            ),
            (
                "byte char",
                "pub fn a() {\n    let s = b'}';\n    let d = 1;\n}\nfn after() {}\n",
                (1, 4),
            ),
            (
                // The opposite error: a stray `{` used to extend the span past the
                // real end and swallow whatever followed.
                "unbalanced opener in a literal",
                "pub fn a() {\n    let s = \"{\";\n}\nfn after() {}\n",
                (1, 3),
            ),
            (
                "escaped quote before a brace",
                "pub fn a() {\n    let s = \"\\\"}\";\n    let d = 1;\n}\nfn after() {}\n",
                (1, 4),
            ),
            (
                // A lifetime is code, and its quote must not open a char literal
                // that then eats the rest of the line.
                "lifetime, not a char literal",
                "pub fn a<'t>(x: &'t u8) {\n    let d = 1;\n}\nfn after() {}\n",
                (1, 3),
            ),
            (
                "nested block comment on one line",
                "pub fn a() {\n    /* /* } */ } */\n    let d = 1;\n}\nfn after() {}\n",
                (1, 4),
            ),
        ];
        for (name, src, want) in cases {
            let lines: Vec<&str> = src.lines().collect();
            assert_eq!(symbol_span(&lines, 1), want, "{name}: {src:?}");
        }
    }

    /// Multi-line literals must not lose the scanner its place either: the state
    /// is carried across lines rather than reset on each one.
    #[test]
    fn a_multi_line_literal_keeps_its_place() {
        let src =
            "pub fn a() {\n    let s = r#\"\n      }\n      }\n    \"#;\n    let d = 1;\n}\nfn after() {}\n";
        let lines: Vec<&str> = src.lines().collect();
        assert_eq!(symbol_span(&lines, 1), (1, 7));
    }

    #[test]
    fn symbol_span_brace_matches_multiline_and_single_line() {
        let src = "pub fn a() {\n    let x = 1;\n    x\n}\nfn b() {}\nconst K: u8 = 3;";
        let lines: Vec<&str> = src.lines().collect();
        assert_eq!(
            symbol_span(&lines, 1),
            (1, 4),
            "multi-line fn closes at its brace"
        );
        assert_eq!(
            symbol_span(&lines, 5),
            (5, 5),
            "one-line fn is a single line"
        );
        assert_eq!(
            symbol_span(&lines, 6),
            (6, 6),
            "a const ends at its semicolon line"
        );
    }

    #[test]
    fn symbol_span_keeps_nested_braces() {
        let src = "fn f() {\n    if x { a(); }\n    loop { break; }\n}";
        let lines: Vec<&str> = src.lines().collect();
        assert_eq!(
            symbol_span(&lines, 1),
            (1, 4),
            "nested braces must not close the span early"
        );
    }

    #[test]
    fn symbol_span_clamps_a_line_past_eof() {
        // A start line beyond EOF (a `--features syn-parser` span edge case) must
        // return an in-bounds span so `lines[start-1..end]` never panics.
        let lines = vec!["fn a() {}"];
        let (s, e) = symbol_span(&lines, 9);
        assert!(s >= 1 && s <= lines.len() && e >= s && e <= lines.len());
        // Empty input must not panic either.
        let empty: Vec<&str> = vec![];
        let _ = symbol_span(&empty, 3);
    }

    #[test]
    fn symbol_node_carries_its_span_and_file_node_is_a_header() {
        let src = "pub fn small() -> u8 { 7 }\npub fn big() {\n    let _ = 1;\n    let _ = 2;\n}";
        let parser = ASTParser::new();
        let result = parser.parse_source("t.rs", src);
        let mut graph = MemoryGraph::default();
        parser.update_memory_graph(&result, src, &mut graph);

        let small = graph
            .nodes
            .get(&NodeId("sym:t.rs:small".to_string()))
            .expect("small symbol node");
        assert_eq!(small.content, "pub fn small() -> u8 { 7 }");

        let big = graph
            .nodes
            .get(&NodeId("sym:t.rs:big".to_string()))
            .expect("big symbol node");
        assert!(big.content.starts_with("pub fn big()") && big.content.contains("let _ = 2;"));

        // The file node is a header (signatures), never the embedded bodies.
        let file = graph
            .nodes
            .get(&NodeId("file:t.rs".to_string()))
            .expect("file node");
        assert!(
            file.content.contains("pub fn small"),
            "header lists signatures"
        );
        assert!(
            !file.content.contains("let _ = 2;"),
            "file header must not embed symbol bodies"
        );
    }
}

/// Tests exercising the real-AST path (only compiled with `--features syn-parser`).
#[cfg(all(test, feature = "syn-parser"))]
mod syn_tests {
    use super::*;

    #[test]
    fn syn_captures_nested_module_tree() {
        let src = "pub mod outer { mod inner { fn deep() {} } }";
        let r = ASTParser::new().parse_source("t.rs", src);
        let outer = r
            .modules
            .iter()
            .find(|m| m.name == "outer")
            .expect("outer module");
        assert!(outer.is_public);
        assert!(
            outer.children.iter().any(|c| c.name == "inner"),
            "nested module must be a child (heuristic parser cannot do this)"
        );
        // The deeply-nested function is still surfaced into the flat symbol list.
        assert!(r.symbols.iter().any(|s| s.name == "deep"));
    }

    #[test]
    fn syn_captures_multiline_signature() {
        // The `fn` line does not end in `{`, so the line parser would miss it.
        let src = "fn wide(\n    a: i32,\n    b: i32,\n) -> i32 {\n    a + b\n}";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(r
            .symbols
            .iter()
            .any(|s| s.name == "wide" && s.kind == SymbolKind::Function));
    }

    #[test]
    fn syn_expands_grouped_use() {
        let src = "use std::collections::{HashMap, HashSet};";
        let r = ASTParser::new().parse_source("t.rs", src);
        let paths: Vec<&str> = r
            .use_statements
            .iter()
            .map(|u| u.full_path.as_str())
            .collect();
        assert!(paths.contains(&"std::collections::HashMap"));
        assert!(paths.contains(&"std::collections::HashSet"));
    }

    #[test]
    fn syn_captures_impl_methods() {
        let src = "struct S;\nimpl S {\n    fn a(&self) {}\n    fn b(&self) {}\n}";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(r
            .symbols
            .iter()
            .any(|s| s.name == "S" && s.kind == SymbolKind::Struct));
        assert!(r
            .symbols
            .iter()
            .any(|s| s.name == "a" && s.kind == SymbolKind::Function));
        assert!(r.symbols.iter().any(|s| s.name == "b"));
    }

    #[test]
    fn syn_captures_self_and_typed_receiver_method_calls() {
        // `self.helper()` is captured as `Self::helper`; a typed receiver `x: T` makes `x.helper()`
        // resolve to `T::helper` (Slice 3 receiver-type inference) — the older "only self" behaviour
        // is intentionally gone. A receiver with no syntactically-known type is still dropped.
        let src = "struct T;\nimpl T {\n  fn run(&self, x: T) { self.helper(); x.helper(); }\n  fn helper(&self) {}\n}";
        let r = ASTParser::new().parse_source("t.rs", src);
        let callees: Vec<String> = r
            .call_sites
            .iter()
            .filter(|c| c.caller == "run")
            .map(|c| c.callee.clone())
            .collect();
        assert!(
            callees.contains(&"Self::helper".to_string()),
            "self.helper() → Self::helper, got {callees:?}"
        );
        assert!(
            callees.contains(&"T::helper".to_string()),
            "x.helper() with x: T → T::helper, got {callees:?}"
        );
    }

    #[test]
    fn syn_captures_reference_typed_receiver_method_calls() {
        // A reference-typed receiver `x: &T` / `y: &mut T` calls methods on `T` (auto-ref of `&self`),
        // so `x.helper()` / `y.helper()` resolve to `T::helper` — leading `&`/`&mut` are peeled to the
        // underlying type, the same inference (and precision) as the owned `x: T` case.
        let src = "struct T;\nimpl T {\n  fn run(x: &T, y: &mut T) { x.helper(); y.helper(); }\n  fn helper(&self) {}\n}";
        let r = ASTParser::new().parse_source("t.rs", src);
        let n = r
            .call_sites
            .iter()
            .filter(|c| c.caller == "run" && c.callee == "T::helper")
            .count();
        assert_eq!(
            n,
            2,
            "both &T and &mut T receivers resolve x.helper() to T::helper, got {:?}",
            r.call_sites
                .iter()
                .filter(|c| c.caller == "run")
                .map(|c| &c.callee)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn syn_reference_to_non_type_receiver_still_dropped() {
        // Peeling references does not loosen precision: a `&dyn Trait` and a `&Box<T>` behind a
        // reference are still not concrete receiver types, so their method calls are dropped — no
        // false `T::m` edge.
        let src = "struct S;\nfn run(a: &dyn std::fmt::Debug, b: &Box<S>) { a.fmt(); b.get(); }";
        let r = ASTParser::new().parse_source("t.rs", src);
        let typed: Vec<&String> = r
            .call_sites
            .iter()
            .filter(|c| c.caller == "run" && c.callee.contains("::"))
            .map(|c| &c.callee)
            .collect();
        assert!(
            typed.is_empty(),
            "a &dyn Trait / &Box receiver is not a concrete type — dropped, got {typed:?}"
        );
    }

    // ── Slice 3: x.bar() receiver-type inference (precision-first) ───────────────
    /// The callees emitted for `caller` in `src`.
    fn callees_of(src: &str, caller: &str) -> Vec<String> {
        ASTParser::new()
            .parse_source("t.rs", src)
            .call_sites
            .into_iter()
            .filter(|c| c.caller == caller)
            .map(|c| c.callee)
            .collect()
    }

    #[test]
    fn infers_receiver_from_the_four_evidence_idioms() {
        // Typed param, let-annotation, the new/default/with_* constructors, and a single-segment
        // struct literal each pin a concrete receiver type → a `Foo::go` callee.
        for (src, want) in [
            ("fn f(x: Foo) { x.go(); }", "Foo::go"),
            ("fn f() { let x: Foo = make(); x.go(); }", "Foo::go"),
            ("fn f() { let x = Foo::new(); x.go(); }", "Foo::go"),
            ("fn f() { let x = Foo::default(); x.go(); }", "Foo::go"),
            (
                "fn f() { let x = Foo::with_capacity(4); x.go(); }",
                "Foo::go",
            ),
            ("fn f() { let x = Foo { a: 1 }; x.go(); }", "Foo::go"),
        ] {
            let cs = callees_of(src, "f");
            assert!(
                cs.contains(&want.to_string()),
                "{src:?} should infer {want}, got {cs:?}"
            );
        }
    }

    #[test]
    fn skips_shadowed_or_reassigned_receiver() {
        // A name bound to two different types, re-`let` to an un-inferable type, or reassigned is
        // POISONED — a shadow/rebind must never mint a false edge (drop-or-nothing).
        for src in [
            "fn f() { let x: A = a(); let x: B = b(); x.go(); }",
            "fn f(mut x: A) { x.go(); x = make(); }",
            "fn f() { let x: A = a(); let x = compute(); x.go(); }",
        ] {
            let cs = callees_of(src, "f");
            assert!(
                !cs.iter().any(|c| c.ends_with("::go")),
                "{src:?} must poison the receiver (no ::go edge), got {cs:?}"
            );
        }
    }

    #[test]
    fn skips_generic_wrapper_and_unknown_receivers() {
        // Generic params, std wrappers, multi-segment (enum-variant) struct literals, method-chain
        // returns, and non-constructor assoc fns are all dropped — a wrong type here is a false edge.
        for src in [
            "fn f<T>(x: T) { x.go(); }",                       // generic type param
            "fn f(x: Box<Foo>) { x.go(); }",                   // std wrapper (no unwrap)
            "fn f() { let x = Status::Ok { c: 0 }; x.go(); }", // enum-variant struct literal
            "fn f(a: A) { let x = a.build(); x.go(); }",       // method-chain return type unknown
            "fn f() { let x = String::from(\"s\"); x.go(); }", // from() is not a whitelisted ctor
        ] {
            let cs = callees_of(src, "f");
            assert!(
                !cs.iter().any(|c| c.ends_with("::go")),
                "{src:?} must not infer a receiver type, got {cs:?}"
            );
        }
    }

    // ── Slice 4: field receivers & return-type chains (precision-first) ──────────

    #[test]
    fn infers_field_receivers_from_declared_field_types() {
        // `self.field.m()` and `x.field.m()` type the receiver from the struct's own field
        // declaration — the paper's "field receivers" future-work item.
        let decls = "struct Db; impl Db { fn q(&self) {} }\nstruct S { db: Db }\n";
        for (src, caller) in [
            (
                format!("{decls}impl S {{ fn go(&self) {{ self.db.q(); }} }}"),
                "go",
            ),
            (format!("{decls}fn f(s: S) {{ s.db.q(); }}"), "f"),
            (format!("{decls}fn f(s: &S) {{ s.db.q(); }}"), "f"),
        ] {
            let cs = callees_of(&src, caller);
            assert!(
                cs.contains(&"Db::q".to_string()),
                "{src:?} should mint Db::q, got {cs:?}"
            );
        }
    }

    #[test]
    fn infers_call_and_chain_receivers_from_declared_returns() {
        // `f().m()`, `let x = f(); x.m()`, `T::assoc().m()` (incl. `-> Self`), and single-hop
        // method chains `x.a().m()` — all typed by same-scope declared returns.
        let decls = "struct A; struct B;\nimpl A { fn make() -> Self { A } fn b(&self) -> B { B } fn touch(&self) {} }\nimpl B { fn q(&self) {} }\nfn make_b() -> B { B }\n";
        for (body, want) in [
            ("fn f() { make_b().q(); }", "B::q"),
            ("fn f() { let x = make_b(); x.q(); }", "B::q"),
            ("fn f() { A::make().touch(); }", "A::touch"),
            ("fn f(a: A) { a.b().q(); }", "B::q"),
            ("fn f(a: &A) { a.b().q(); }", "B::q"),
            ("fn f(a: A) { a.b().q(); a.touch(); }", "A::touch"),
        ] {
            let src = format!("{decls}{body}");
            let cs = callees_of(&src, "f");
            assert!(
                cs.contains(&want.to_string()),
                "{src:?} should mint {want}, got {cs:?}"
            );
        }
        // Inside an impl, `Self::make()` resolves through the same declared-return facts.
        let src = format!("{decls}impl B {{ fn init() {{ let x = A::make(); x.touch(); }} }}");
        assert!(
            callees_of(&src, "init").contains(&"A::touch".to_string()),
            "A::make() -> Self types the local"
        );
    }

    #[test]
    fn field_and_chain_inference_skips_uncertain_shapes() {
        // Generic/wrapper/undeclared fields, non-concrete returns, and — the key precision rule —
        // a visible definition contradicting the ctor convention all drop (no ::q / ::go edge).
        for src in [
            // generic field type
            "struct S<T> { x: T }\nimpl S<i32> { fn f(&self) { self.x.go(); } }",
            // wrapper field type
            "struct Db; struct S { x: Vec<Db> }\nimpl S { fn f(&self) { self.x.go(); } }",
            // field not declared on the receiver's type (cross-file struct) — unknown
            "struct S;\nimpl S { fn f(&self) { self.db.go(); } }",
            // fn with a non-concrete (wrapper) return
            "struct C; fn make() -> Option<C> { None }\nfn f() { make().go(); }",
            // in-file definition refutes the `new` returns-Self convention
            "struct C; impl C { fn new() -> Option<C> { None } }\nfn f() { let x = C::new(); x.go(); }",
            // chain hop through a method with no declared return
            "struct A; impl A { fn b(&self) { } }\nfn f(a: A) { a.b().go(); }",
            // tuple-field receiver (unnamed member) — no field-type fact
            "struct P(i32);\nimpl P { fn f(&self) { self.0.go(); } }",
        ] {
            let cs = callees_of(src, "f");
            assert!(
                !cs.iter().any(|c| c.ends_with("::go") || c.ends_with("::q")),
                "{src:?} must not infer a receiver, got {cs:?}"
            );
        }
    }

    #[test]
    fn field_receiver_resolves_cross_file_via_the_type_method_index() {
        // The minted `Db::q` callee resolves through the graph-wide (type, method) index even when
        // `Db` and its impl live in ANOTHER file — only the field's declared type is same-file.
        let mut g = crate::memory::MemoryGraph::new(0.2, 5000);
        let parser = ASTParser::new();
        let src_a = "pub struct Db; impl Db { pub fn q(&self) {} }\n";
        let src_b = "use crate::a::Db;\npub struct S { db: Db }\nimpl S { pub fn go(&self) { self.db.q(); } }\n";
        for (file, src) in [("src/a.rs", src_a), ("src/b.rs", src_b)] {
            let parsed = parser.parse_source(file, src);
            parser.update_memory_graph(&parsed, src, &mut g);
        }
        g.resolve_all();
        let has = g.edges.iter().any(|e| {
            e.edge_type == crate::memory::EdgeType::Calls
                && e.source.0 == "sym:src/b.rs:go"
                && e.target.0 == "sym:src/a.rs:q"
        });
        assert!(
            has,
            "self.db.q() links go → Db::q across files, got {:?}",
            g.edges
                .iter()
                .filter(|e| e.edge_type == crate::memory::EdgeType::Calls)
                .map(|e| format!("{}→{}", e.source.0, e.target.0))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn emits_method_def_facts_for_concrete_impls_only() {
        // A concrete `impl Foo` contributes (Foo, m) ownership facts; a blanket `impl<T> Tr for T`
        // (Self type is a generic param) contributes nothing — it has no concrete owner to index.
        let src = "struct Foo;\nimpl Foo { fn a(&self) {} fn b(&self) {} }\ntrait Tr { fn show(&self); }\nimpl<T> Tr for T { fn show(&self) {} }";
        let defs = ASTParser::new().parse_source("t.rs", src).method_defs;
        let pairs: Vec<(String, String)> =
            defs.into_iter().map(|m| (m.type_name, m.method)).collect();
        assert!(pairs.contains(&("Foo".into(), "a".into())));
        assert!(pairs.contains(&("Foo".into(), "b".into())));
        assert!(
            !pairs.iter().any(|(t, _)| t == "T"),
            "the blanket impl's generic Self must not be indexed, got {pairs:?}"
        );
    }

    #[test]
    fn syn_self_method_skips_deref_or_external_method() {
        // `self.len()` where the type has NO `len` method (it would come from Deref / a trait) is
        // NOT captured — so it can never be mislinked to the same-named free `len`. This is the
        // exact false-edge the own-method-set guard closes.
        let src =
            "struct W;\nimpl W { fn run(&self) -> usize { self.len() } }\nfn len() -> usize { 0 }";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(
            !r.call_sites.iter().any(|c| c.callee == "Self::len"),
            "self.len() (not a method of W) is not captured, got {:?}",
            r.call_sites.iter().map(|c| &c.callee).collect::<Vec<_>>()
        );
    }

    #[test]
    fn syn_captures_screaming_snake_data_refs_only() {
        // A `SCREAMING_SNAKE` value reference is captured as a data-ref; snake_case (fn/local) and
        // PascalCase (type/variant) are not. The const *definition* is an item, not a value path,
        // so only the *use* inside `reader` is captured.
        let src = "const MAX_SIZE: usize = 10;\nfn reader() -> usize { let _c = Config; MAX_SIZE + helper() }\nfn helper() -> usize { 0 }\nstruct Config;";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(
            r.data_refs
                .iter()
                .any(|d| d.reader == "reader" && d.name == "MAX_SIZE"),
            "SCREAMING_SNAKE reference is captured, got {:?}",
            r.data_refs.iter().map(|d| &d.name).collect::<Vec<_>>()
        );
        assert!(
            !r.data_refs
                .iter()
                .any(|d| d.name == "helper" || d.name == "Config"),
            "snake_case (fn) and PascalCase (type) references are not data-refs"
        );
    }

    #[test]
    fn data_flow_end_to_end_cross_file() {
        // The whole wiring through the real parser: config.rs defines `const MAX_LIMIT`, api.rs's
        // `reader` references it. update_memory_graph marks the const + records the ref, and
        // resolve_data_flow links them across files — the channel imports/calls never see.
        let p = ASTParser::new();
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        for (path, src) in [
            ("src/config.rs", "pub const MAX_LIMIT: usize = 10;"),
            ("src/api.rs", "fn reader() -> usize { MAX_LIMIT + 1 }"),
        ] {
            let r = p.parse_source(path, src);
            p.update_memory_graph(&r, src, &mut g);
        }
        g.resolve_data_flow();
        let edges: Vec<(String, String)> = g
            .edges()
            .iter()
            .filter(|e| e.edge_type == EdgeType::DataFlow)
            .map(|e| (e.source.0.clone(), e.target.0.clone()))
            .collect();
        assert_eq!(
            edges,
            vec![(
                "sym:src/api.rs:reader".to_string(),
                "sym:src/config.rs:MAX_LIMIT".to_string()
            )],
            "reader → MAX_LIMIT cross-file data-flow edge, end to end"
        );
    }

    #[test]
    fn data_flow_qualified_end_to_end_cross_file() {
        // End-to-end through the real parser: cfg.rs defines `const MAX_RETRIES`; api.rs's `reader`
        // references it QUALIFIED as `crate::cfg::MAX_RETRIES`. The parser captures the full path,
        // and resolve_data_flow pins the prefix to cfg.rs and links to the marked const (Slice 2).
        let p = ASTParser::new();
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        for (path, src) in [
            ("src/cfg.rs", "pub const MAX_RETRIES: usize = 3;"),
            (
                "src/api.rs",
                "fn reader() -> usize { crate::cfg::MAX_RETRIES + 1 }",
            ),
        ] {
            let r = p.parse_source(path, src);
            p.update_memory_graph(&r, src, &mut g);
        }
        g.resolve_data_flow();
        let edges: Vec<(String, String)> = g
            .edges()
            .iter()
            .filter(|e| e.edge_type == EdgeType::DataFlow)
            .map(|e| (e.source.0.clone(), e.target.0.clone()))
            .collect();
        assert_eq!(
            edges,
            vec![(
                "sym:src/api.rs:reader".to_string(),
                "sym:src/cfg.rs:MAX_RETRIES".to_string()
            )],
            "qualified crate::cfg::MAX_RETRIES → cfg.rs const, end to end (Slice 2)"
        );
    }

    #[test]
    fn data_flow_qualified_unresolvable_end_to_end_skips() {
        // A qualified ref whose module prefix pins to NO file must skip end-to-end — never a false
        // edge. `crate::missing::MAX_RETRIES` (module `missing` has no file) is dropped even though a
        // unique `MAX_RETRIES` exists in cfg.rs (proving no fall-back to the bare global index).
        let p = ASTParser::new();
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        for (path, src) in [
            ("src/cfg.rs", "pub const MAX_RETRIES: usize = 3;"),
            (
                "src/api.rs",
                "fn reader() -> usize { crate::missing::MAX_RETRIES + 1 }",
            ),
        ] {
            let r = p.parse_source(path, src);
            p.update_memory_graph(&r, src, &mut g);
        }
        g.resolve_data_flow();
        assert!(
            !g.edges()
                .iter()
                .any(|e| e.edge_type == EdgeType::DataFlow),
            "unresolvable qualified ref emits no DataFlow edge (no fallback to the global MAX_RETRIES)"
        );
    }

    #[test]
    fn syn_data_refs_skip_locally_bound_names() {
        // A `SCREAMING_SNAKE` name bound locally — a parameter, a fn-local `const`, or a `let` — is
        // NOT a data-ref: it denotes the local, so resolving it global-unique to a same-named global
        // would be a false edge. Only the genuinely-free `MAX_LIMIT` is captured.
        let src = "fn r(PARAM_X: u8) -> u8 {\n  const LOCAL_C: u8 = 1;\n  let LET_V = 2u8;\n  PARAM_X + LOCAL_C + LET_V + MAX_LIMIT\n}";
        let r = ASTParser::new().parse_source("t.rs", src);
        let names: Vec<String> = r.data_refs.iter().map(|d| d.name.clone()).collect();
        assert!(
            names.contains(&"MAX_LIMIT".to_string()),
            "the free global reference is captured, got {names:?}"
        );
        for bound in ["PARAM_X", "LOCAL_C", "LET_V"] {
            assert!(
                !names.iter().any(|n| n == bound),
                "{bound} is locally bound — it must not be a data-ref (got {names:?})"
            );
        }
    }

    #[test]
    fn syn_captures_qualified_screaming_snake_data_ref_full_path() {
        // A qualified value path whose LAST segment is SCREAMING_SNAKE is captured as a data-ref
        // carrying its FULL `::`-joined path (Slice 2) — `config::MAX_RETRIES`,
        // `crate::limits::MAX`, `self::FOO`. A qualified path whose last segment is NOT screaming
        // (`config::helper`, a fn call) is not a data-ref.
        let src = "fn reader() -> usize {\n  config::MAX_RETRIES + crate::limits::MAX + self::FOO + config::helper()\n}";
        let r = ASTParser::new().parse_source("t.rs", src);
        let names: Vec<String> = r
            .data_refs
            .iter()
            .filter(|d| d.reader == "reader")
            .map(|d| d.name.clone())
            .collect();
        for want in ["config::MAX_RETRIES", "crate::limits::MAX", "self::FOO"] {
            assert!(
                names.iter().any(|n| n == want),
                "qualified ref {want} captured with full path, got {names:?}"
            );
        }
        assert!(
            !names.iter().any(|n| n == "config::helper" || n == "helper"),
            "a qualified call (non-screaming last segment) is not a data-ref, got {names:?}"
        );
    }

    #[test]
    fn syn_qualified_data_ref_skips_locally_bound_head() {
        // The scope guard extends to qualified paths: when the HEAD segment is locally bound, the
        // path denotes the local (e.g. a `let m = …; m::FOO` field/assoc access), not a module —
        // so it must NOT be captured even though it looks qualified. Only the free `cfg::LIMIT`
        // (head `cfg` is not a local) is captured.
        let src = "fn r(m: u8) -> u8 {\n  let mod_x = 1u8;\n  m::FOO + mod_x::BAR + cfg::LIMIT\n}";
        let r = ASTParser::new().parse_source("t.rs", src);
        let names: Vec<String> = r.data_refs.iter().map(|d| d.name.clone()).collect();
        assert!(
            names.iter().any(|n| n == "cfg::LIMIT"),
            "the free qualified ref is captured, got {names:?}"
        );
        for bound_head in ["m::FOO", "mod_x::BAR"] {
            assert!(
                !names.iter().any(|n| n == bound_head),
                "{bound_head} has a locally-bound head — it must not be a data-ref (got {names:?})"
            );
        }
    }

    #[test]
    fn syn_captures_free_function_call_sites() {
        let src = "fn caller() { helper(); ns::deep(); recur(); }\nfn helper() {}\nfn recur() { recur(); }";
        let r = ASTParser::new().parse_source("t.rs", src);
        // bare free-function calls are captured with their caller; a qualified call keeps its full
        // `::`-joined path (Slice 2). Method calls `x.bar()` are still not captured (Slice 3).
        assert!(r
            .call_sites
            .iter()
            .any(|c| c.caller == "caller" && c.callee == "helper"));
        assert!(r
            .call_sites
            .iter()
            .any(|c| c.caller == "recur" && c.callee == "recur"));
        assert!(
            r.call_sites
                .iter()
                .any(|c| c.caller == "caller" && c.callee == "ns::deep"),
            "a qualified-path call keeps its full path (Slice 2)"
        );
    }

    #[test]
    fn syn_falls_back_on_invalid_syntax() {
        // Not valid Rust → syn returns None → heuristic parser handles it, no panic.
        let src = "fn broken( this is not rust {{{";
        let r = ASTParser::new().parse_source("t.rs", src);
        // Should not panic; result is whatever the heuristic produced.
        let _ = r.symbols.len();
    }

    #[test]
    fn syn_ignores_commented_out_code() {
        let src = "fn real() {}\n// fn commented() {}\n/* fn blocked() {} */";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(r.symbols.iter().any(|s| s.name == "real"));
        assert!(!r.symbols.iter().any(|s| s.name == "commented"));
        assert!(!r.symbols.iter().any(|s| s.name == "blocked"));
    }

    // ── Renamed-import alias support (`use a::b as c`) ────────────────────────────────────────

    #[test]
    fn syn_records_top_level_rename_alias() {
        // `use a::b as c` records the LOCAL name `c` bound to the ORIGINAL target path `a::b`. The
        // `UseStatement` keeps the original path (so import-linking is unchanged from `use a::b`).
        let r = ASTParser::new().parse_source("t.rs", "use a::b as c;");
        assert_eq!(r.import_aliases.len(), 1);
        assert_eq!(r.import_aliases[0].local, "c");
        assert_eq!(r.import_aliases[0].target, "a::b");
        assert!(
            r.use_statements.iter().any(|u| u.full_path == "a::b"),
            "the original path is still recorded as a use statement, got {:?}",
            r.use_statements
                .iter()
                .map(|u| &u.full_path)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn syn_records_alias_inside_group_and_nested_group() {
        // Aliases inside a group — `use a::{b, c as d}` — and inside a NESTED group —
        // `use a::{e::{f as g}}` — must both be captured, each with the full target path.
        let r = ASTParser::new().parse_source("t.rs", "use a::{b, c as d, e::{f as g}};");
        let mut got: Vec<(String, String)> = r
            .import_aliases
            .iter()
            .map(|a| (a.local.clone(), a.target.clone()))
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                ("d".to_string(), "a::c".to_string()),
                ("g".to_string(), "a::e::f".to_string()),
            ],
            "group + nested-group aliases captured with full target paths"
        );
        // The non-renamed member `b` is a plain import, not an alias.
        assert!(r.use_statements.iter().any(|u| u.full_path == "a::b"));
    }

    #[test]
    fn syn_glob_import_still_works_and_is_not_an_alias() {
        // `use a::*` must keep producing an `a::*` use statement and record NO alias.
        let r = ASTParser::new().parse_source("t.rs", "use a::*;");
        assert!(
            r.use_statements.iter().any(|u| u.full_path == "a::*"),
            "glob import is preserved, got {:?}",
            r.use_statements
                .iter()
                .map(|u| &u.full_path)
                .collect::<Vec<_>>()
        );
        assert!(r.import_aliases.is_empty(), "a glob is not a rename");
    }

    /// Build a graph from `(path, src)` files through the real parser, then run the
    /// import/call resolution passes. Returns the resolved `Calls` edges as `(src_id, dst_id)`
    /// pairs (sorted), i.e. the exact fn→fn structure the call graph encodes.
    fn calls_edges_of(files: &[(&str, &str)]) -> Vec<(String, String)> {
        let p = ASTParser::new();
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        for (path, src) in files {
            let r = p.parse_source(path, src);
            p.update_memory_graph(&r, src, &mut g);
        }
        g.link_module_imports();
        g.resolve_symbol_calls();
        let mut edges: Vec<(String, String)> = g
            .edges()
            .iter()
            .filter(|e| e.edge_type == EdgeType::Calls)
            .map(|e| (e.source.0.clone(), e.target.0.clone()))
            .collect();
        edges.sort();
        edges
    }

    #[test]
    fn syn_alias_call_resolves_to_original_target() {
        // `use crate::a::b as c; ... c()` — the alias call resolves to `a::b`'s symbol, end to end.
        let edges = calls_edges_of(&[
            ("src/a.rs", "pub fn b() {}"),
            ("src/api.rs", "use crate::a::b as c;\nfn caller() { c(); }"),
        ]);
        assert!(
            edges.contains(&(
                "sym:src/api.rs:caller".to_string(),
                "sym:src/a.rs:b".to_string()
            )),
            "aliased call c() resolves to a::b, got {edges:?}"
        );
    }

    #[test]
    fn syn_alias_call_inside_group_resolves() {
        // The alias is declared inside a group — `use crate::a::{x as y}` — and `y()` resolves to x.
        let edges = calls_edges_of(&[
            ("src/a.rs", "pub fn x() {}"),
            (
                "src/api.rs",
                "use crate::a::{x as y};\nfn caller() { y(); }",
            ),
        ]);
        assert!(
            edges.contains(&(
                "sym:src/api.rs:caller".to_string(),
                "sym:src/a.rs:x".to_string()
            )),
            "grouped alias call y() resolves to a::x, got {edges:?}"
        );
    }

    #[test]
    fn syn_qualified_alias_call_resolves() {
        // A qualified call through the alias — `use crate::a::b as c; ... c::CONST` (rewritten to
        // `crate::a::b::CONST`) resolves to the symbol in module `a::b`.
        let edges = calls_edges_of(&[
            ("src/a/b.rs", "pub fn deep() {}"),
            (
                "src/api.rs",
                "use crate::a::b as c;\nfn caller() { c::deep(); }",
            ),
        ]);
        assert!(
            edges.contains(&(
                "sym:src/api.rs:caller".to_string(),
                "sym:src/a/b.rs:deep".to_string()
            )),
            "qualified alias call c::deep() resolves to a::b::deep, got {edges:?}"
        );
    }

    #[test]
    fn syn_alias_with_missing_target_is_skipped() {
        // The alias target `crate::a::nope` has NO resident symbol → resolve-uniquely-or-skip yields
        // NO edge (a false edge would be the bug). `a.rs` defines only an unrelated `other`.
        let edges = calls_edges_of(&[
            ("src/a.rs", "pub fn other() {}"),
            (
                "src/api.rs",
                "use crate::a::nope as c;\nfn caller() { c(); }",
            ),
        ]);
        assert!(
            !edges.iter().any(|(s, _)| s == "sym:src/api.rs:caller"),
            "alias to a non-existent target must add no Calls edge, got {edges:?}"
        );
    }

    #[test]
    fn syn_alias_does_not_collide_with_same_named_real_symbol() {
        // There is a REAL `fn c` in another module (`z`). `api.rs` aliases `crate::a::b as c` and
        // calls `c()`. The alias must win — the call resolves to `a::b`, NEVER cross-linking to the
        // unrelated real `z::c` (which is what bare global-unique resolution would have done).
        let edges = calls_edges_of(&[
            ("src/a.rs", "pub fn b() {}"),
            ("src/z.rs", "pub fn c() {}"),
            ("src/api.rs", "use crate::a::b as c;\nfn caller() { c(); }"),
        ]);
        assert!(
            edges.contains(&(
                "sym:src/api.rs:caller".to_string(),
                "sym:src/a.rs:b".to_string()
            )),
            "aliased c() resolves to a::b, got {edges:?}"
        );
        assert!(
            !edges.contains(&(
                "sym:src/api.rs:caller".to_string(),
                "sym:src/z.rs:c".to_string()
            )),
            "the alias must NOT cross-link to the same-named real z::c, got {edges:?}"
        );
    }

    #[test]
    fn syn_alias_resolution_is_deterministic() {
        // The same source must yield byte-for-byte identical Calls edges across runs (the replay
        // invariant): indices over sorted ids, sorted+deduped candidate edges.
        let files: &[(&str, &str)] = &[
            ("src/a.rs", "pub fn b() {}"),
            ("src/z.rs", "pub fn c() {}"),
            (
                "src/api.rs",
                "use crate::a::{b as c, b as e};\nfn caller() { c(); e(); }",
            ),
        ];
        let first = calls_edges_of(files);
        for _ in 0..5 {
            assert_eq!(first, calls_edges_of(files), "alias edges must be stable");
        }
        // Both aliases (`b as c`, `b as e`) point at the same target, so exactly one caller→b edge.
        assert_eq!(
            first,
            vec![(
                "sym:src/api.rs:caller".to_string(),
                "sym:src/a.rs:b".to_string()
            )],
            "two aliases to one target dedupe to a single edge, got {first:?}"
        );
    }

    #[test]
    fn syn_self_call_resolves_method_in_other_inherent_impl_block() {
        // `self.foo()` in one `impl Bar` block must resolve to `Bar::foo` even though `foo` is
        // defined in a SEPARATE `impl Bar` block — per-type method sets union across all blocks.
        let src =
            "struct Bar;\nimpl Bar { fn a(&self) { self.foo(); } }\nimpl Bar { fn foo(&self) {} }";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(
            r.call_sites
                .iter()
                .any(|c| c.caller == "a" && c.callee == "Self::foo"),
            "self.foo() must be captured via Bar's unioned method set across impl blocks, got {:?}",
            r.call_sites
                .iter()
                .map(|c| (&c.caller, &c.callee))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn syn_self_call_resolves_method_from_trait_impl_for_type() {
        // `foo` is provided by a TRAIT impl `impl SomeTrait for Bar`; a `self.foo()` in Bar's
        // inherent impl must still be captured — trait-impl methods are unioned into Bar's set.
        let src = "struct Bar;\ntrait SomeTrait { fn foo(&self); }\nimpl SomeTrait for Bar { fn foo(&self) {} }\nimpl Bar { fn a(&self) { self.foo(); } }";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(
            r.call_sites
                .iter()
                .any(|c| c.caller == "a" && c.callee == "Self::foo"),
            "self.foo() from a trait impl for Bar must be captured, got {:?}",
            r.call_sites
                .iter()
                .map(|c| (&c.caller, &c.callee))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn syn_self_call_two_types_same_method_name_do_not_cross_link() {
        // Bar has `foo`; Baz does NOT. `self.foo()` inside Baz must NOT be captured — gating is
        // STRICTLY by the enclosing impl's Self type, so two types' same-named methods never cross.
        let src = "struct Bar;\nstruct Baz;\nimpl Bar { fn foo(&self) {} }\nimpl Baz { fn a(&self) { self.foo(); } }";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(
            !r.call_sites
                .iter()
                .any(|c| c.caller == "a" && c.callee == "Self::foo"),
            "Baz has no foo — self.foo() inside Baz must not cross-link to Bar::foo, got {:?}",
            r.call_sites
                .iter()
                .map(|c| (&c.caller, &c.callee))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn syn_self_path_resolves_same_as_self_method_call_across_blocks() {
        // A `Self::foo` PATH (associated-fn form) must gate identically to `self.foo()` — captured
        // when `foo` is in the type's unioned set even if defined in another impl block.
        let src = "struct Bar;\nimpl Bar { fn a() { Self::foo(); } }\nimpl Bar { fn foo() {} }";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(
            r.call_sites
                .iter()
                .any(|c| c.caller == "a" && c.callee == "Self::foo"),
            "Self::foo path must resolve across impl blocks like self.foo(), got {:?}",
            r.call_sites
                .iter()
                .map(|c| (&c.caller, &c.callee))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn syn_self_call_blanket_impl_generic_self_falls_back_no_cross_link() {
        // `impl<T> Marker for T` has a type-VARIABLE Self (`T`), not a concrete type. It must fall
        // back to same-block gating: `self.foo()` (foo not in this block) is NOT captured, and a
        // generic blanket impl never unions a same-named method from an unrelated concrete type.
        let src = "trait Marker { fn a(&self); }\nstruct Bar;\nimpl Bar { fn foo(&self) {} }\nimpl<T> Marker for T { fn a(&self) { self.foo(); } }";
        let r = ASTParser::new().parse_source("t.rs", src);
        assert!(
            !r.call_sites
                .iter()
                .any(|c| c.caller == "a" && c.callee == "Self::foo"),
            "blanket impl<T> for T must fall back (no concrete type) and not cross-link, got {:?}",
            r.call_sites
                .iter()
                .map(|c| (&c.caller, &c.callee))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn syn_self_call_cross_impl_is_deterministic() {
        // Same source ⇒ identical captured edges (per-type sets are BTree-ordered; capture is a
        // pure function of the AST).
        let src = "struct Bar;\nimpl Bar { fn a(&self) { self.foo(); self.bar(); } }\nimpl Bar { fn foo(&self) {} }\nimpl SomeTrait for Bar { fn bar(&self) {} }\ntrait SomeTrait { fn bar(&self); }";
        let first = ASTParser::new().parse_source("t.rs", src);
        let second = ASTParser::new().parse_source("t.rs", src);
        let edges = |r: &ParseResult| -> Vec<(String, String, usize)> {
            r.call_sites
                .iter()
                .map(|c| (c.caller.clone(), c.callee.clone(), c.line))
                .collect()
        };
        assert_eq!(
            edges(&first),
            edges(&second),
            "cross-impl self-call capture must be deterministic"
        );
        assert!(
            first
                .call_sites
                .iter()
                .any(|c| c.caller == "a" && c.callee == "Self::foo")
                && first
                    .call_sites
                    .iter()
                    .any(|c| c.caller == "a" && c.callee == "Self::bar"),
            "both inherent (foo) and trait-impl (bar) methods resolve from Bar's union, got {:?}",
            first
                .call_sites
                .iter()
                .map(|c| (&c.caller, &c.callee))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn syn_self_call_cross_impl_end_to_end_resolves_edge() {
        // Full wiring: a self-call to a method in a DIFFERENT impl block of the same type resolves
        // to a real `Calls` edge through resolve_symbol_calls (Self::foo → same-module foo symbol).
        let p = ASTParser::new();
        let mut g = MemoryGraph::new(0.0, usize::MAX);
        let src =
            "struct Bar;\nimpl Bar { fn a(&self) { self.foo(); } }\nimpl Bar { fn foo(&self) {} }";
        let r = p.parse_source("src/lib.rs", src);
        p.update_memory_graph(&r, src, &mut g);
        g.resolve_symbol_calls();
        let edges: Vec<(String, String)> = g
            .edges()
            .iter()
            .filter(|e| e.edge_type == EdgeType::Calls)
            .map(|e| (e.source.0.clone(), e.target.0.clone()))
            .collect();
        assert_eq!(
            edges,
            vec![(
                "sym:src/lib.rs:a".to_string(),
                "sym:src/lib.rs:foo".to_string()
            )],
            "self.foo() across impl blocks resolves to Bar::foo, end to end"
        );
    }
}
