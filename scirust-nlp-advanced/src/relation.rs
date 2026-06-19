//! Relation extraction from text.
//!
//! Provides:
//! - Pattern-based relation extraction with typed triples.
//! - Dependency-path feature computation (simplified dependency representation).
//! - Relation classification via feature overlap scoring.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A typed relation between two entity mentions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub evidence: String,
}

/// An entity mention with position in text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMention {
    pub text: String,
    pub entity_type: String,
    pub start: usize,
    pub end: usize,
}

/// A single dependency edge (simplified).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepEdge {
    pub governor: usize,
    pub dependent: usize,
    pub relation: String,
}

/// A simplified dependency tree.
#[derive(Debug, Clone)]
pub struct DependencyTree {
    pub tokens: Vec<String>,
    pub edges: Vec<DepEdge>,
}

impl DependencyTree {
    pub fn new(tokens: Vec<String>) -> Self {
        Self {
            tokens,
            edges: Vec::new(),
        }
    }

    pub fn add_edge(&mut self, governor: usize, dependent: usize, relation: &str) {
        self.edges.push(DepEdge {
            governor,
            dependent,
            relation: relation.to_string(),
        });
    }

    /// Get the shortest path between two token indices (BFS).
    pub fn shortest_path(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        if from == to
        {
            return Some(vec![from]);
        }
        let n = self.tokens.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for e in &self.edges
        {
            adj[e.governor].push(e.dependent);
            adj[e.dependent].push(e.governor);
        }
        let mut visited = vec![false; n];
        let mut parent = vec![usize::MAX; n];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(from);
        visited[from] = true;
        while let Some(cur) = queue.pop_front()
        {
            for &next in &adj[cur]
            {
                if !visited[next]
                {
                    visited[next] = true;
                    parent[next] = cur;
                    if next == to
                    {
                        // reconstruct path
                        let mut path = Vec::new();
                        let mut node = to;
                        while node != usize::MAX
                        {
                            path.push(node);
                            node = parent[node];
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(next);
                }
            }
        }
        None
    }

    /// Extract the dependency path between two token spans as a sequence of
    /// (token, edge-relation) pairs.
    pub fn dependency_path(&self, from: usize, to: usize) -> Option<Vec<(String, String)>> {
        let path_indices = self.shortest_path(from, to)?;
        let mut result = Vec::with_capacity(path_indices.len());
        for window in path_indices.windows(2)
        {
            let a = window[0];
            let b = window[1];
            // Find edge label
            let label = self
                .edges
                .iter()
                .find(|e| {
                    (e.governor == a && e.dependent == b) || (e.governor == b && e.dependent == a)
                })
                .map(|e| e.relation.clone())
                .unwrap_or_default();
            result.push((self.tokens[a].clone(), label));
        }
        // Add last token
        if let Some(&last) = path_indices.last()
        {
            result.push((self.tokens[last].clone(), String::new()));
        }
        Some(result)
    }
}

// ---------------------------------------------------------------------------
// Pattern-based relation extraction
// ---------------------------------------------------------------------------

/// A lexico-syntactic pattern for extracting relations.
///
/// Pattern syntax: tokens separated by spaces.  Special markers:
/// - `{SUBJ}` — placeholder for subject entity
/// - `{OBJ}` — placeholder for object entity
/// - `*` — wildcard (matches any single token)
///
/// Example: `"{SUBJ} was born in {OBJ}"` matches
/// "Einstein was born in Germany".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationPattern {
    /// Pattern tokens (including {SUBJ}, {OBJ}, *).
    pub tokens: Vec<String>,
    /// Predicate label to assign to extracted relations.
    pub predicate: String,
}

/// Find the starting index in `sentence` where all `fixed` tokens match
/// (case-insensitive, `*` matches any token).  Returns the first index
/// where the entire fixed sequence fits.
fn find_fixed_start(sentence: &[String], fixed: &[&str]) -> Option<usize> {
    if fixed.is_empty()
    {
        return Some(0);
    }
    for start in 0..=sentence.len().saturating_sub(fixed.len())
    {
        let mut ok = true;
        for (fi, ft) in fixed.iter().enumerate()
        {
            if *ft != "*" && ft.to_lowercase() != sentence[start + fi].to_lowercase()
            {
                ok = false;
                break;
            }
        }
        if ok
        {
            return Some(start);
        }
    }
    None
}

/// Build the match result, returning None if either side is empty.
fn make_result(subj_words: Vec<String>, obj_words: Vec<String>) -> Option<(String, String)> {
    let subj = subj_words.join(" ");
    let obj = obj_words.join(" ");
    if subj.is_empty() && obj.is_empty()
    {
        None
    }
    else
    {
        Some((subj, obj))
    }
}

impl RelationPattern {
    pub fn new(pattern: &str, predicate: &str) -> Self {
        let tokens: Vec<String> = pattern.split_whitespace().map(String::from).collect();
        Self {
            tokens,
            predicate: predicate.to_string(),
        }
    }

    /// Check whether `sentence` matches this pattern and extract the
    /// subject and object spans.
    ///
    /// Pattern tokens like `{SUBJ}` and `{OBJ}` are placeholders that
    /// capture one or more tokens.  `*` matches exactly one token.
    /// All other tokens must match literally (case-insensitive).
    pub fn matches(&self, sentence: &[String]) -> Option<(String, String)> {
        let subj_pos = self.tokens.iter().position(|t| t == "{SUBJ}");
        let obj_pos = self.tokens.iter().position(|t| t == "{OBJ}");

        match (subj_pos, obj_pos)
        {
            (Some(sp), Some(op)) if sp < op =>
            {
                // {SUBJ} ... fixed ... {OBJ}
                let fixed_between: Vec<&str> =
                    self.tokens[sp + 1..op].iter().map(|s| s.as_str()).collect();
                let fixed_start = find_fixed_start(sentence, &fixed_between)?;
                let subj_words: Vec<String> = sentence[..fixed_start].to_vec();
                let fixed_end = fixed_start + fixed_between.len();
                let obj_words: Vec<String> = sentence[fixed_end..].to_vec();
                make_result(subj_words, obj_words)
            },
            (Some(sp), Some(op)) if op < sp =>
            {
                // {OBJ} ... fixed ... {SUBJ}
                let fixed_between: Vec<&str> =
                    self.tokens[op + 1..sp].iter().map(|s| s.as_str()).collect();
                let fixed_start = find_fixed_start(sentence, &fixed_between)?;
                let obj_words: Vec<String> = sentence[..fixed_start].to_vec();
                let fixed_end = fixed_start + fixed_between.len();
                let subj_words: Vec<String> = sentence[fixed_end..].to_vec();
                make_result(subj_words, obj_words)
            },
            (Some(sp), None) =>
            {
                // Only {SUBJ}: everything before sp are the fixed tokens after SUBJ.
                // Pattern: {SUBJ} fixed_tokens...
                // Fixed tokens are after {SUBJ}
                let fixed_after: Vec<&str> =
                    self.tokens[sp + 1..].iter().map(|s| s.as_str()).collect();
                if fixed_after.is_empty()
                {
                    // {SUBJ} alone → entire sentence is subject
                    let subj = sentence.join(" ");
                    Some((subj, String::new()))
                }
                else
                {
                    let fixed_start = find_fixed_start(sentence, &fixed_after)?;
                    let subj_words: Vec<String> = sentence[..fixed_start].to_vec();
                    let subj = subj_words.join(" ");
                    Some((subj, String::new()))
                }
            },
            (None, Some(op)) =>
            {
                // Only {OBJ}: everything before op are fixed tokens before OBJ.
                let fixed_before: Vec<&str> =
                    self.tokens[..op].iter().map(|s| s.as_str()).collect();
                if fixed_before.is_empty()
                {
                    let obj = sentence.join(" ");
                    Some((String::new(), obj))
                }
                else
                {
                    let fixed_start = find_fixed_start(sentence, &fixed_before)?;
                    let fixed_end = fixed_start + fixed_before.len();
                    let obj_words: Vec<String> = sentence[fixed_end..].to_vec();
                    let obj = obj_words.join(" ");
                    Some((String::new(), obj))
                }
            },
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Relation extractor
// ---------------------------------------------------------------------------

/// Configuration for relation extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationExtractorConfig {
    /// Minimum confidence to report a relation.
    pub min_confidence: f64,
    /// Maximum window size between entities for proximity-based extraction.
    pub max_entity_distance: usize,
}

impl Default for RelationExtractorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            max_entity_distance: 20,
        }
    }
}

/// Pattern-based relation extractor.
pub struct RelationExtractor {
    pub patterns: Vec<RelationPattern>,
    pub config: RelationExtractorConfig,
}

impl RelationExtractor {
    pub fn new(config: RelationExtractorConfig) -> Self {
        Self {
            patterns: Vec::new(),
            config,
        }
    }

    /// Add a pattern.
    pub fn add_pattern(&mut self, pattern: RelationPattern) {
        self.patterns.push(pattern);
    }

    /// Add multiple patterns at once.
    pub fn add_patterns(&mut self, patterns: Vec<RelationPattern>) {
        self.patterns.extend(patterns);
    }

    /// Extract relations from a sentence given entity mentions.
    pub fn extract(&self, sentence: &[String], entities: &[EntityMention]) -> Vec<Relation> {
        let mut relations = Vec::new();

        // 1. Pattern-based extraction
        for pattern in &self.patterns
        {
            if let Some((subj, obj)) = pattern.matches(sentence)
            {
                // Find evidence span
                let evidence = sentence.join(" ");
                relations.push(Relation {
                    subject: subj,
                    predicate: pattern.predicate.clone(),
                    object: obj,
                    confidence: 0.8,
                    evidence,
                });
            }
        }

        // 2. Proximity-based extraction: if two entities of different types
        //    are close, extract a generic relation.
        for (i, e1) in entities.iter().enumerate()
        {
            for (j, e2) in entities.iter().enumerate()
            {
                if i >= j
                {
                    continue;
                }
                let dist = if e1.end <= e2.start
                {
                    e2.start - e1.end
                }
                else
                {
                    e1.start.saturating_sub(e2.end)
                };
                if dist <= self.config.max_entity_distance
                {
                    let confidence =
                        1.0 - (dist as f64 / self.config.max_entity_distance as f64) * 0.5;
                    if confidence >= self.config.min_confidence
                    {
                        relations.push(Relation {
                            subject: e1.text.clone(),
                            predicate: "RELATED_TO".to_string(),
                            object: e2.text.clone(),
                            confidence,
                            evidence: sentence.join(" "),
                        });
                    }
                }
            }
        }

        relations
    }
}

// ---------------------------------------------------------------------------
// Dependency-path features
// ---------------------------------------------------------------------------

/// Compute features from the dependency path between two entity mention
/// spans.  Returns a feature map suitable for downstream classification.
pub fn dependency_path_features(
    tree: &DependencyTree,
    subj_start: usize,
    _subj_end: usize,
    obj_start: usize,
    _obj_end: usize,
) -> HashMap<String, f64> {
    let mut features = HashMap::new();

    let path = tree.shortest_path(subj_start, obj_start);
    if let Some(indices) = path
    {
        features.insert("path_length".to_string(), indices.len() as f64);

        // Collect edge labels along the path
        let mut edge_labels: Vec<String> = Vec::new();
        for window in indices.windows(2)
        {
            let a = window[0];
            let b = window[1];
            if let Some(edge) = tree.edges.iter().find(|e| {
                (e.governor == a && e.dependent == b) || (e.governor == b && e.dependent == a)
            })
            {
                edge_labels.push(edge.relation.clone());
            }
        }
        features.insert(
            "num_noun_edges".to_string(),
            edge_labels
                .iter()
                .filter(|l| l.contains("NOUN") || l.contains("nsubj"))
                .count() as f64,
        );
        features.insert(
            "num_verb_edges".to_string(),
            edge_labels
                .iter()
                .filter(|l| l.contains("VERB") || l.contains("dobj"))
                .count() as f64,
        );
        features.insert(
            "num_prep_edges".to_string(),
            edge_labels.iter().filter(|l| l.contains("prep")).count() as f64,
        );

        // Has direct connection
        features.insert(
            "direct_connection".to_string(),
            if indices.len() <= 2 { 1.0 } else { 0.0 },
        );
        // Has verb on path
        features.insert(
            "has_verb_on_path".to_string(),
            if edge_labels
                .iter()
                .any(|l| l.contains("VERB") || l.contains("ROOT"))
            {
                1.0
            }
            else
            {
                0.0
            },
        );
    }
    else
    {
        features.insert("path_length".to_string(), -1.0);
    }

    features
}

// ---------------------------------------------------------------------------
// Relation classifier
// ---------------------------------------------------------------------------

/// A simple relation classifier based on pattern + feature overlap.
pub struct RelationClassifier {
    /// Known patterns: (subject-type, object-type, predicate) → weight
    pub pattern_weights: HashMap<(String, String, String), f64>,
}

impl RelationClassifier {
    pub fn new() -> Self {
        Self {
            pattern_weights: HashMap::new(),
        }
    }

    /// Train from gold relations.
    pub fn train(&mut self, gold: &[Relation]) {
        for rel in gold
        {
            let key = (
                rel.subject.clone(),
                rel.object.clone(),
                rel.predicate.clone(),
            );
            *self.pattern_weights.entry(key).or_insert(0.0) += rel.confidence;
        }
    }

    /// Classify candidate relations, returning scores.
    pub fn classify(&self, candidates: &[Relation]) -> Vec<(Relation, f64)> {
        candidates
            .iter()
            .map(|rel| {
                let key = (
                    rel.subject.clone(),
                    rel.object.clone(),
                    rel.predicate.clone(),
                );
                let score = self.pattern_weights.get(&key).copied().unwrap_or(0.0);
                (rel.clone(), score)
            })
            .collect()
    }

    /// Predict the most likely predicate for a (subject, object) pair.
    pub fn predict_predicate(&self, subject: &str, object: &str) -> Option<(String, f64)> {
        let mut best: Option<(String, f64)> = None;
        for ((s, o, pred), &weight) in &self.pattern_weights
        {
            if s == subject && o == object && best.as_ref().is_none_or(|(_, w)| weight > *w)
            {
                best = Some((pred.clone(), weight));
            }
        }
        best
    }
}

impl Default for RelationClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_matching() {
        let pat = RelationPattern::new("{SUBJ} was born in {OBJ}", "born_in");
        let sentence: Vec<String> = "Einstein was born in Germany"
            .split_whitespace()
            .map(String::from)
            .collect();
        let result = pat.matches(&sentence);
        assert!(result.is_some());
        let (subj, obj) = result.unwrap();
        assert_eq!(subj, "Einstein");
        assert_eq!(obj, "Germany");
    }

    #[test]
    fn test_pattern_no_match() {
        let pat = RelationPattern::new("{SUBJ} invented {OBJ}", "invented");
        let sentence: Vec<String> = "Einstein was born in Germany"
            .split_whitespace()
            .map(String::from)
            .collect();
        assert!(pat.matches(&sentence).is_none());
    }

    #[test]
    fn test_pattern_wildcard() {
        let pat = RelationPattern::new("{SUBJ} * works at {OBJ}", "works_at");
        let sentence: Vec<String> = "Alice currently works at Google"
            .split_whitespace()
            .map(String::from)
            .collect();
        let result = pat.matches(&sentence);
        assert!(result.is_some());
        let (subj, obj) = result.unwrap();
        // * matches "currently" (one token, not captured)
        assert_eq!(subj, "Alice");
        assert_eq!(obj, "Google");
    }

    #[test]
    fn test_dependency_tree_shortest_path() {
        let mut tree = DependencyTree::new(vec![
            "Einstein".into(),
            "was".into(),
            "born".into(),
            "in".into(),
            "Germany".into(),
        ]);
        tree.add_edge(2, 0, "nsubj");
        tree.add_edge(2, 4, "prep");
        tree.add_edge(4, 3, "det"); // in -> Germany (simplified)
        let path = tree.shortest_path(0, 4);
        assert!(path.is_some());
        let p = path.unwrap();
        assert!(p.contains(&0) && p.contains(&4));
    }

    #[test]
    fn test_dependency_path_features() {
        let mut tree = DependencyTree::new(vec![
            "Alice".into(),
            "works".into(),
            "at".into(),
            "Google".into(),
        ]);
        tree.add_edge(1, 0, "nsubj");
        tree.add_edge(1, 2, "prep");
        tree.add_edge(2, 3, "pobj");
        let features = dependency_path_features(&tree, 0, 1, 3, 4);
        assert!(features.contains_key("path_length"));
        assert!(features.contains_key("direct_connection"));
    }

    #[test]
    fn test_relation_extractor() {
        let mut extractor = RelationExtractor::new(RelationExtractorConfig::default());
        extractor.add_pattern(RelationPattern::new("{SUBJ} was born in {OBJ}", "born_in"));
        let sentence: Vec<String> = "Einstein was born in Germany"
            .split_whitespace()
            .map(String::from)
            .collect();
        let entities = vec![
            EntityMention {
                text: "Einstein".into(),
                entity_type: "PER".into(),
                start: 0,
                end: 8,
            },
            EntityMention {
                text: "Germany".into(),
                entity_type: "LOC".into(),
                start: 23,
                end: 30,
            },
        ];
        let relations = extractor.extract(&sentence, &entities);
        assert!(
            relations.iter().any(|r| r.predicate == "born_in"
                && r.subject == "Einstein"
                && r.object == "Germany")
        );
    }

    #[test]
    fn test_relation_classifier() {
        let mut clf = RelationClassifier::new();
        clf.train(&[Relation {
            subject: "Einstein".into(),
            predicate: "born_in".into(),
            object: "Germany".into(),
            confidence: 1.0,
            evidence: String::new(),
        }]);
        let pred = clf.predict_predicate("Einstein", "Germany");
        assert!(pred.is_some());
        assert_eq!(pred.unwrap().0, "born_in");
    }

    #[test]
    fn test_relation_classifier_default() {
        let clf = RelationClassifier::default();
        assert!(clf.pattern_weights.is_empty());
    }
}
