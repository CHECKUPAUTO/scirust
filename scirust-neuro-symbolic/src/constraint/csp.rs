use crate::core::Reasoner;
use std::collections::HashMap;

pub struct CspSolver {
    pub domains: HashMap<String, Vec<i32>>,
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

    pub fn solve(&self) -> Option<HashMap<String, i32>> {
        // Backtracking search implementation placeholder
        None
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
    fn test_csp_solver_name() {
        let solver = CspSolver::new();
        assert_eq!(solver.name(), "CspSolver");
    }
}
