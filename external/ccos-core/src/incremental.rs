use crate::memory::{MemoryGraph, NodeId};
use crate::parser::{ASTParser, ParseResult};
use crate::util::sha256_hex as compute_hash;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileState {
    pub file_path: String,
    pub content_hash: String,
    pub last_parsed_at: u64,
    pub modules_count: usize,
    pub uses_count: usize,
    pub symbols_count: usize,
}

#[derive(Debug, Clone)]
pub struct IncrementalGraphEngine {
    pub file_states: HashMap<String, FileState>,
    pub parser: ASTParser,
    pub mutation_log: Vec<DeltaMutation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaMutation {
    pub timestamp: u64,
    pub file_path: String,
    pub operation: MutationOp,
    pub nodes_added: usize,
    pub nodes_removed: usize,
    pub edges_added: usize,
    pub edges_removed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MutationOp {
    FileAdded,
    FileModified,
    FileRemoved,
    NoChange,
}

impl IncrementalGraphEngine {
    pub fn new() -> Self {
        Self {
            file_states: HashMap::new(),
            parser: ASTParser::new(),
            mutation_log: Vec::new(),
        }
    }

    pub fn register_file(&mut self, file_path: &str, source_code: &str) -> ParseResult {
        let hash = compute_hash(source_code);
        let parse_result = self.parser.parse_source(file_path, source_code);

        // This only records file-level state (hashes, counts); applying the
        // parse result to the graph is the caller's responsibility (see
        // `ASTParser::update_memory_graph`). Use `process_delta` for the
        // combined parse + incremental graph update.
        self.file_states.insert(
            file_path.to_string(),
            FileState {
                file_path: file_path.to_string(),
                content_hash: hash,
                last_parsed_at: Self::now(),
                modules_count: parse_result.modules.len(),
                uses_count: parse_result.use_statements.len(),
                symbols_count: parse_result.symbols.len(),
            },
        );

        parse_result
    }

    pub fn process_delta(
        &mut self,
        file_path: &str,
        old_source: Option<&str>,
        new_source: &str,
        graph: &mut MemoryGraph,
    ) -> DeltaMutation {
        let new_hash = compute_hash(new_source);
        let timestamp = Self::now();

        // Determine operation type
        let op = match old_source {
            None => MutationOp::FileAdded,
            Some(old) if compute_hash(old) == new_hash => MutationOp::NoChange,
            Some(_) => MutationOp::FileModified,
        };

        let nodes_before = graph.node_count();
        let edges_before = graph.edge_count();

        match op {
            MutationOp::FileAdded => {
                let result = self.parser.parse_source(file_path, new_source);
                self.parser.update_memory_graph(&result, new_source, graph);
                self.store_file_state(file_path, new_hash, &result);
            }
            MutationOp::FileModified => {
                // O(Δ): only modify this file's contributions
                self.evict_file_nodes(file_path, graph);
                let result = self.parser.parse_source(file_path, new_source);
                self.parser.update_memory_graph(&result, new_source, graph);
                self.store_file_state(file_path, new_hash, &result);
            }
            MutationOp::NoChange => {
                // Nothing to do
            }
            MutationOp::FileRemoved => {
                self.evict_file_nodes(file_path, graph);
                self.file_states.remove(file_path);
            }
        }

        let nodes_after = graph.node_count();
        let edges_after = graph.edge_count();

        let nodes_added = nodes_after.saturating_sub(nodes_before);
        let nodes_removed = nodes_before.saturating_sub(nodes_after);
        let edges_added = edges_after.saturating_sub(edges_before);
        let edges_removed = edges_before.saturating_sub(edges_after);

        let mutation = DeltaMutation {
            timestamp,
            file_path: file_path.to_string(),
            operation: op,
            nodes_added,
            nodes_removed,
            edges_added,
            edges_removed,
        };

        self.mutation_log.push(mutation.clone());
        mutation
    }

    pub fn evict_file_nodes(&mut self, file_path: &str, graph: &mut MemoryGraph) {
        let prefix = format!("file:{}", file_path);
        let mod_prefix = format!("mod:{}:", file_path);
        let use_prefix = format!("use:{}:", file_path);
        let sym_prefix = format!("sym:{}:", file_path);

        let to_remove: Vec<NodeId> = graph
            .nodes
            .keys()
            .filter(|id| {
                let s = id.0.as_str();
                s == prefix
                    || s.starts_with(&mod_prefix)
                    || s.starts_with(&use_prefix)
                    || s.starts_with(&sym_prefix)
            })
            .cloned()
            .collect();

        for id in &to_remove {
            graph.remove_node(id);
        }
    }

    pub fn store_file_state(&mut self, file_path: &str, hash: String, result: &ParseResult) {
        self.file_states.insert(
            file_path.to_string(),
            FileState {
                file_path: file_path.to_string(),
                content_hash: hash,
                last_parsed_at: Self::now(),
                modules_count: result.modules.len(),
                uses_count: result.use_statements.len(),
                symbols_count: result.symbols.len(),
            },
        );
    }

    pub fn has_file_changed(&self, file_path: &str, new_source: &str) -> bool {
        let new_hash = compute_hash(new_source);
        match self.file_states.get(file_path) {
            Some(state) => state.content_hash != new_hash,
            None => true,
        }
    }

    pub fn changed_files(&self, current_sources: &HashMap<String, String>) -> Vec<String> {
        current_sources
            .iter()
            .filter_map(|(path, source)| {
                if self.has_file_changed(path, source) {
                    Some(path.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_state(&self, file_path: &str) -> Option<&FileState> {
        self.file_states.get(file_path)
    }

    pub fn total_mutations(&self) -> usize {
        self.mutation_log.len()
    }

    fn now() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

impl Default for IncrementalGraphEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_added_delta() {
        let mut engine = IncrementalGraphEngine::new();
        let mut graph = MemoryGraph::default();
        let mutation = engine.process_delta("test.rs", None, "mod foo;\nfn bar() {}", &mut graph);
        assert_eq!(mutation.operation, MutationOp::FileAdded);
        assert!(mutation.nodes_added > 0);
    }

    #[test]
    fn test_file_modified_delta() {
        let mut engine = IncrementalGraphEngine::new();
        let mut graph = MemoryGraph::default();

        let old_source = "mod foo;\nfn bar() {}";
        let new_source = "mod foo;\nmod baz;\nfn bar() {}\nfn qux() {}";

        engine.process_delta("test.rs", None, old_source, &mut graph);
        let first_count = graph.node_count();

        let mutation = engine.process_delta("test.rs", Some(old_source), new_source, &mut graph);
        assert_eq!(mutation.operation, MutationOp::FileModified);
        // Should have different node count due to added module and function
        assert!(mutation.nodes_added > 0 || graph.node_count() > first_count);
    }

    #[test]
    fn test_no_change_detected() {
        let mut engine = IncrementalGraphEngine::new();
        let source = "fn main() {}";

        engine.file_states.insert(
            "main.rs".into(),
            FileState {
                file_path: "main.rs".into(),
                content_hash: compute_hash(source),
                last_parsed_at: 0,
                modules_count: 0,
                uses_count: 0,
                symbols_count: 1,
            },
        );

        assert!(!engine.has_file_changed("main.rs", source));
    }

    #[test]
    fn test_change_detected() {
        let mut engine = IncrementalGraphEngine::new();
        let source = "fn main() {}";

        engine.file_states.insert(
            "main.rs".into(),
            FileState {
                file_path: "main.rs".into(),
                content_hash: compute_hash(source),
                last_parsed_at: 0,
                modules_count: 0,
                uses_count: 0,
                symbols_count: 1,
            },
        );

        assert!(engine.has_file_changed("main.rs", "fn main() { let x = 1; }"));
    }

    #[test]
    fn test_evict_file_nodes() {
        let mut engine = IncrementalGraphEngine::new();
        let mut graph = MemoryGraph::default();

        engine.process_delta("test.rs", None, "mod foo;\nfn bar() {}", &mut graph);
        let count_before = graph.node_count();
        assert!(count_before > 0);

        engine.evict_file_nodes("test.rs", &mut graph);
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_changed_files_bulk() {
        let mut engine = IncrementalGraphEngine::new();
        let mut current = HashMap::new();
        current.insert("a.rs".into(), "fn foo() {}".into());
        current.insert("b.rs".into(), "fn bar() {}".into());

        // Process files first
        let mut graph = MemoryGraph::default();
        engine.process_delta("a.rs", None, "fn foo() {}", &mut graph);

        let changed = engine.changed_files(&current);
        assert!(changed.contains(&"b.rs".to_string()));
    }

    /// The headline `O(Δ)` claim, guarded **deterministically** in CI (criterion's `delta_bench`
    /// only *times* it): editing one file must produce the same node/edge footprint no matter how
    /// large the surrounding graph already is. If `process_delta` ever did work proportional to the
    /// whole repo (re-linking everything), this footprint would drift with the background.
    #[test]
    fn process_delta_footprint_is_background_independent() {
        fn footprint(background: usize) -> (usize, usize, usize, usize) {
            let mut engine = IncrementalGraphEngine::new();
            let mut graph = MemoryGraph::new(0.0, usize::MAX);
            for f in 0..background {
                let src = format!("mod m{f};\npub fn func_{f}() {{}}\nstruct S{f};\n");
                engine.process_delta(&format!("bg/file_{f}.rs"), None, &src, &mut graph);
            }
            engine.process_delta(
                "hot/file.rs",
                None,
                "pub fn a() {}\nstruct A;\n",
                &mut graph,
            );
            let d = engine.process_delta(
                "hot/file.rs",
                Some("pub fn a() {}\nstruct A;\n"),
                "pub fn a() {}\npub fn b() {}\nstruct A;\n",
                &mut graph,
            );
            (
                d.nodes_added,
                d.nodes_removed,
                d.edges_added,
                d.edges_removed,
            )
        }
        assert_eq!(
            footprint(0),
            footprint(200),
            "per-edit footprint must not grow with the background graph (O(Δ))"
        );
    }
}
