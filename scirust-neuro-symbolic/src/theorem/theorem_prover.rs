use crate::core::{Reasoner, Result};
use std::collections::HashSet;

/// A propositional theorem prover based on **forward chaining**.
///
/// Premises are either atoms (`"a"`) or definite implications written
/// `"a & b -> c"`. The prover repeatedly fires rules whose entire body is known
/// until a fixpoint, then reports whether the goal atom was derived.
///
/// Note: the `iterations` field caps the number of inference rounds. The
/// "neural-guided" proof search advertised historically is **not** implemented;
/// this is an honest symbolic forward-chaining prover.
pub struct NeuralTheoremProver {
    pub iterations: usize,
}

impl NeuralTheoremProver {
    pub fn new(iterations: usize) -> Self {
        Self { iterations }
    }

    /// Returns `Ok(true)` iff `goal` is entailed by `premises` via forward
    /// chaining, within `self.iterations` rounds (0 ⇒ unbounded until fixpoint).
    pub fn prove(&self, goal: &str, premises: &[&str]) -> Result<bool> {
        let mut facts: HashSet<String> = HashSet::new();
        let mut rules: Vec<(Vec<String>, String)> = Vec::new();

        for p in premises {
            if let Some((body, head)) = p.split_once("->") {
                let body_atoms: Vec<String> = body
                    .split('&')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                rules.push((body_atoms, head.trim().to_string()));
            } else {
                facts.insert(p.trim().to_string());
            }
        }

        let goal = goal.trim().to_string();
        let max_rounds = if self.iterations == 0 {
            usize::MAX
        } else {
            self.iterations
        };

        let mut round = 0;
        loop {
            if facts.contains(&goal) {
                return Ok(true);
            }
            if round >= max_rounds {
                break;
            }
            let mut changed = false;
            for (body, head) in &rules {
                if !facts.contains(head) && body.iter().all(|b| facts.contains(b)) {
                    facts.insert(head.clone());
                    changed = true;
                }
            }
            round += 1;
            if !changed {
                break;
            }
        }
        Ok(facts.contains(&goal))
    }
}

impl Reasoner for NeuralTheoremProver {
    fn name(&self) -> &str {
        "NeuralTheoremProver"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proves_via_modus_ponens_chain() {
        let prover = NeuralTheoremProver::new(0);
        // a, b, (a & b -> c), (c -> d)  ⊢  d
        let premises = ["a", "b", "a & b -> c", "c -> d"];
        assert!(prover.prove("d", &premises).unwrap());
    }

    #[test]
    fn unprovable_goal_returns_false() {
        let prover = NeuralTheoremProver::new(0);
        assert!(!prover.prove("z", &["a", "a -> b"]).unwrap());
    }
}
