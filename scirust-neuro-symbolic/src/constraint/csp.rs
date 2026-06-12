use crate::core::Reasoner;
use std::collections::HashMap;

/// A finite-domain constraint-satisfaction solver using **backtracking search**.
/// Constraints are arbitrary predicates over a complete assignment; they are
/// evaluated once every variable is bound.
pub struct CspSolver {
    pub domains: HashMap<String, Vec<i32>>,
    #[allow(clippy::type_complexity)]
    pub constraints: Vec<Box<dyn Fn(&HashMap<String, i32>) -> bool>>,
}

impl CspSolver {
    pub fn new() -> Self {
        Self {
            domains: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    pub fn add_variable(&mut self, name: &str, domain: Vec<i32>) {
        self.domains.insert(name.to_string(), domain);
    }

    pub fn add_constraint<F>(&mut self, constraint: F)
    where
        F: Fn(&HashMap<String, i32>) -> bool + 'static,
    {
        self.constraints.push(Box::new(constraint));
    }

    /// Returns a satisfying assignment if one exists, else `None`.
    pub fn solve(&self) -> Option<HashMap<String, i32>> {
        let vars: Vec<String> = {
            let mut v: Vec<String> = self.domains.keys().cloned().collect();
            v.sort(); // deterministic variable order
            v
        };
        let mut assignment = HashMap::new();
        if self.backtrack(&vars, 0, &mut assignment)
        {
            Some(assignment)
        }
        else
        {
            None
        }
    }

    fn satisfies_all(&self, assignment: &HashMap<String, i32>) -> bool {
        self.constraints.iter().all(|c| c(assignment))
    }

    fn backtrack(
        &self,
        vars: &[String],
        idx: usize,
        assignment: &mut HashMap<String, i32>,
    ) -> bool {
        if idx == vars.len()
        {
            return self.satisfies_all(assignment);
        }
        let var = &vars[idx];
        for &val in &self.domains[var]
        {
            assignment.insert(var.clone(), val);
            if self.backtrack(vars, idx + 1, assignment)
            {
                return true;
            }
            assignment.remove(var);
        }
        false
    }
}

impl Default for CspSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Reasoner for CspSolver {
    fn name(&self) -> &str {
        "CspSolver"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_assignment_under_constraints() {
        let mut csp = CspSolver::new();
        csp.add_variable("a", vec![1, 2, 3]);
        csp.add_variable("b", vec![1, 2, 3]);
        csp.add_constraint(|asg| asg["a"] < asg["b"]);
        csp.add_constraint(|asg| asg["a"] + asg["b"] == 4);
        let sol = csp.solve().expect("should be solvable");
        assert!(sol["a"] < sol["b"]);
        assert_eq!(sol["a"] + sol["b"], 4);
    }

    #[test]
    fn returns_none_when_unsatisfiable() {
        let mut csp = CspSolver::new();
        csp.add_variable("a", vec![1, 2]);
        csp.add_constraint(|asg| asg["a"] > 5);
        assert!(csp.solve().is_none());
    }
}
