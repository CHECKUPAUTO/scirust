use crate::core::{Reasoner, Result};
use std::collections::HashSet;

pub struct Rule {
    pub head: String,
    pub body: Vec<String>,
}

/// A propositional production-rule engine with **forward chaining**: a rule
/// fires (deriving its head) once every atom in its body is known.
pub struct RuleEngine {
    pub rules: Vec<Rule>,
    pub facts: HashSet<String>,
}

impl RuleEngine {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            facts: HashSet::new(),
        }
    }

    pub fn add_rule(&mut self, head: &str, body: Vec<&str>) {
        self.rules.push(Rule {
            head: head.to_string(),
            body: body.into_iter().map(|s| s.to_string()).collect(),
        });
    }

    pub fn add_fact(&mut self, fact: &str) {
        self.facts.insert(fact.to_string());
    }

    /// Runs forward chaining to a fixpoint and returns the list of *newly
    /// derived* atoms (excludes the initial facts), in derivation order.
    pub fn forward_chain(&self) -> Result<Vec<String>> {
        let mut known = self.facts.clone();
        let mut derived = Vec::new();
        loop {
            let mut changed = false;
            for rule in &self.rules {
                if !known.contains(&rule.head) && rule.body.iter().all(|b| known.contains(b)) {
                    known.insert(rule.head.clone());
                    derived.push(rule.head.clone());
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Ok(derived)
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Reasoner for RuleEngine {
    fn name(&self) -> &str {
        "RuleEngine"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_chaining_derives_transitive_consequences() {
        let mut re = RuleEngine::new();
        re.add_fact("a");
        re.add_fact("b");
        re.add_rule("c", vec!["a", "b"]);
        re.add_rule("d", vec!["c"]);
        let derived = re.forward_chain().unwrap();
        assert!(derived.contains(&"c".to_string()));
        assert!(derived.contains(&"d".to_string()));
    }

    #[test]
    fn no_derivation_without_complete_body() {
        let mut re = RuleEngine::new();
        re.add_fact("a");
        re.add_rule("c", vec!["a", "b"]); // b unknown
        assert!(re.forward_chain().unwrap().is_empty());
    }
}
