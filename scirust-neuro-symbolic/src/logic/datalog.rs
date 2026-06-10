use crate::core::{Reasoner, Result};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fact {
    pub predicate: String,
    pub terms: Vec<String>,
}

/// A term in a rule: either a (capitalised-by-convention) variable or a constant.
#[derive(Debug, Clone)]
pub enum Term {
    Var(String),
    Const(String),
}

impl Term {
    pub fn var(name: &str) -> Self {
        Term::Var(name.to_string())
    }
    pub fn con(name: &str) -> Self {
        Term::Const(name.to_string())
    }
}

/// A (possibly non-ground) atom used in rule heads and bodies.
#[derive(Debug, Clone)]
pub struct Atom {
    pub predicate: String,
    pub terms: Vec<Term>,
}

impl Atom {
    pub fn new(predicate: &str, terms: Vec<Term>) -> Self {
        Self {
            predicate: predicate.to_string(),
            terms,
        }
    }
}

/// A Datalog rule `head :- body`.
pub struct DatalogRule {
    pub head: Atom,
    pub body: Vec<Atom>,
}

pub struct DatalogEngine {
    pub facts: HashSet<Fact>,
    pub rules: Vec<DatalogRule>,
}

type Subst = HashMap<String, String>;

impl DatalogEngine {
    pub fn new() -> Self {
        Self {
            facts: HashSet::new(),
            rules: Vec::new(),
        }
    }

    pub fn add_fact(&mut self, predicate: &str, terms: Vec<&str>) {
        self.facts.insert(Fact {
            predicate: predicate.to_string(),
            terms: terms.into_iter().map(|s| s.to_string()).collect(),
        });
    }

    pub fn add_rule(&mut self, head: Atom, body: Vec<Atom>) {
        self.rules.push(DatalogRule { head, body });
    }

    pub fn query(&self, predicate: &str, terms: Vec<&str>) -> bool {
        let query_fact = Fact {
            predicate: predicate.to_string(),
            terms: terms.into_iter().map(|s| s.to_string()).collect(),
        };
        self.facts.contains(&query_fact)
    }

    /// Naive bottom-up evaluation: repeatedly apply every rule, deriving new
    /// ground facts, until no rule produces anything new (a fixpoint).
    pub fn run_fixed_point(&mut self) -> Result<()> {
        loop {
            let mut new_facts: Vec<Fact> = Vec::new();
            for rule in &self.rules {
                for sub in self.eval_body(&rule.body) {
                    if let Some(fact) = instantiate(&rule.head, &sub) {
                        if !self.facts.contains(&fact) {
                            new_facts.push(fact);
                        }
                    }
                }
            }
            if new_facts.is_empty() {
                break;
            }
            for f in new_facts {
                self.facts.insert(f);
            }
        }
        Ok(())
    }

    /// All substitutions over the current facts that satisfy the whole body.
    fn eval_body(&self, body: &[Atom]) -> Vec<Subst> {
        let mut subs = vec![Subst::new()];
        for atom in body {
            let mut next = Vec::new();
            for sub in &subs {
                for fact in &self.facts {
                    if fact.predicate != atom.predicate || fact.terms.len() != atom.terms.len() {
                        continue;
                    }
                    if let Some(s2) = unify(atom, fact, sub) {
                        next.push(s2);
                    }
                }
            }
            subs = next;
            if subs.is_empty() {
                break;
            }
        }
        subs
    }
}

fn unify(atom: &Atom, fact: &Fact, sub: &Subst) -> Option<Subst> {
    let mut s = sub.clone();
    for (t, val) in atom.terms.iter().zip(&fact.terms) {
        match t {
            Term::Const(c) => {
                if c != val {
                    return None;
                }
            }
            Term::Var(v) => match s.get(v) {
                Some(bound) if bound != val => return None,
                Some(_) => {}
                None => {
                    s.insert(v.clone(), val.clone());
                }
            },
        }
    }
    Some(s)
}

fn instantiate(head: &Atom, sub: &Subst) -> Option<Fact> {
    let mut terms = Vec::with_capacity(head.terms.len());
    for t in &head.terms {
        match t {
            Term::Const(c) => terms.push(c.clone()),
            Term::Var(v) => terms.push(sub.get(v)?.clone()),
        }
    }
    Some(Fact {
        predicate: head.predicate.clone(),
        terms,
    })
}

impl Default for DatalogEngine {
    fn default() -> Self {
        Self::new()
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

    #[test]
    fn derives_grandparent_via_rule() {
        let mut e = DatalogEngine::new();
        e.add_fact("parent", vec!["alice", "bob"]);
        e.add_fact("parent", vec!["bob", "carol"]);
        // grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        e.add_rule(
            Atom::new("grandparent", vec![Term::var("X"), Term::var("Z")]),
            vec![
                Atom::new("parent", vec![Term::var("X"), Term::var("Y")]),
                Atom::new("parent", vec![Term::var("Y"), Term::var("Z")]),
            ],
        );
        e.run_fixed_point().unwrap();
        assert!(e.query("grandparent", vec!["alice", "carol"]));
        assert!(!e.query("grandparent", vec!["alice", "bob"]));
    }

    #[test]
    fn computes_transitive_closure() {
        let mut e = DatalogEngine::new();
        e.add_fact("edge", vec!["a", "b"]);
        e.add_fact("edge", vec!["b", "c"]);
        e.add_fact("edge", vec!["c", "d"]);
        // path(X,Y) :- edge(X,Y).
        e.add_rule(
            Atom::new("path", vec![Term::var("X"), Term::var("Y")]),
            vec![Atom::new("edge", vec![Term::var("X"), Term::var("Y")])],
        );
        // path(X,Z) :- edge(X,Y), path(Y,Z).
        e.add_rule(
            Atom::new("path", vec![Term::var("X"), Term::var("Z")]),
            vec![
                Atom::new("edge", vec![Term::var("X"), Term::var("Y")]),
                Atom::new("path", vec![Term::var("Y"), Term::var("Z")]),
            ],
        );
        e.run_fixed_point().unwrap();
        assert!(e.query("path", vec!["a", "d"]));
    }
}
