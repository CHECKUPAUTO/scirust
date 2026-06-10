use crate::core::{Reasoner, Result};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fact {
    pub predicate: String,
    pub terms: Vec<String>,
}

pub struct DatalogEngine {
    pub facts: HashSet<Fact>,
}

impl DatalogEngine {
    pub fn new() -> Self {
        Self {
            facts: HashSet::new(),
        }
    }

    pub fn add_fact(&mut self, predicate: &str, terms: Vec<&str>) {
        self.facts.insert(Fact {
            predicate: predicate.to_string(),
            terms: terms.into_iter().map(|s| s.to_string()).collect(),
        });
    }

    pub fn query(&self, predicate: &str, terms: Vec<&str>) -> bool {
        let query_fact = Fact {
            predicate: predicate.to_string(),
            terms: terms.into_iter().map(|s| s.to_string()).collect(),
        };
        self.facts.contains(&query_fact)
    }

    pub fn run_fixed_point(&mut self) -> Result<()> {
        // Evaluate rules until no new facts are added
        Ok(())
    }
}

impl Reasoner for DatalogEngine {
    fn name(&self) -> &str {
        "DatalogEngine"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datalog_basic() {
        let mut engine = DatalogEngine::new();
        engine.add_fact("parent", vec!["alice", "bob"]);
        assert!(engine.query("parent", vec!["alice", "bob"]));
        assert!(!engine.query("parent", vec!["bob", "alice"]));
    }
}
