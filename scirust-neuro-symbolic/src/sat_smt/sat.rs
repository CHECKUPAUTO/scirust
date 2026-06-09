use crate::core::{Reasoner, Result};

pub type Literal = i32;
pub type Clause = Vec<Literal>;

/// A propositional SAT solver implementing the **DPLL** algorithm
/// (unit propagation + backtracking search). Variables are 1-indexed; a literal
/// `v` means variable `v` is true and `-v` means it is false.
pub struct SatSolver {
    pub clauses: Vec<Clause>,
}

impl SatSolver {
    pub fn new() -> Self {
        Self { clauses: Vec::new() }
    }

    pub fn add_clause(&mut self, clause: Clause) {
        self.clauses.push(clause);
    }

    /// Returns `Ok(Some(model))` if satisfiable (where `model[i]` is the value of
    /// variable `i + 1`), or `Ok(None)` if unsatisfiable.
    pub fn solve(&self) -> Result<Option<Vec<bool>>> {
        let n = self.num_vars();
        let mut assign: Vec<Option<bool>> = vec![None; n + 1];
        if dpll(&self.clauses, &mut assign) {
            Ok(Some((1..=n).map(|v| assign[v].unwrap_or(false)).collect()))
        } else {
            Ok(None)
        }
    }

    fn num_vars(&self) -> usize {
        self.clauses
            .iter()
            .flatten()
            .map(|l| l.unsigned_abs() as usize)
            .max()
            .unwrap_or(0)
    }
}

impl Default for SatSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// DPLL with unit propagation. `assign` is indexed by variable id (1-based).
fn dpll(clauses: &[Clause], assign: &mut Vec<Option<bool>>) -> bool {
    // --- Unit propagation to a fixpoint ---
    loop {
        let mut progressed = false;
        for clause in clauses {
            let mut unassigned: Option<Literal> = None;
            let mut unassigned_count = 0usize;
            let mut satisfied = false;
            for &lit in clause {
                let v = lit.unsigned_abs() as usize;
                match assign[v] {
                    Some(b) if b == (lit > 0) => {
                        satisfied = true;
                        break;
                    }
                    Some(_) => {}
                    None => {
                        unassigned = Some(lit);
                        unassigned_count += 1;
                    }
                }
            }
            if satisfied {
                continue;
            }
            if unassigned_count == 0 {
                return false; // conflict: clause fully assigned and false
            }
            if unassigned_count == 1 {
                let lit = unassigned.unwrap();
                assign[lit.unsigned_abs() as usize] = Some(lit > 0);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }

    // --- Pick the next unassigned variable ---
    let next = (1..assign.len()).find(|&v| assign[v].is_none());
    match next {
        None => clauses.iter().all(|c| {
            c.iter()
                .any(|&lit| assign[lit.unsigned_abs() as usize] == Some(lit > 0))
        }),
        Some(v) => {
            for val in [true, false] {
                let mut branch = assign.clone();
                branch[v] = Some(val);
                if dpll(clauses, &mut branch) {
                    *assign = branch;
                    return true;
                }
            }
            false
        }
    }
}

impl Reasoner for SatSolver {
    fn name(&self) -> &str {
        "SatSolver"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_satisfiable_instance() {
        // (x1 ∨ x2) ∧ (¬x1) ⇒ x1=false, x2=true
        let mut s = SatSolver::new();
        s.add_clause(vec![1, 2]);
        s.add_clause(vec![-1]);
        let model = s.solve().unwrap().expect("should be SAT");
        assert!(!model[0]); // x1 = false
        assert!(model[1]); // x2 = true
    }

    #[test]
    fn detects_unsatisfiable_instance() {
        // (x1) ∧ (¬x1) ⇒ UNSAT
        let mut s = SatSolver::new();
        s.add_clause(vec![1]);
        s.add_clause(vec![-1]);
        assert!(s.solve().unwrap().is_none());
    }

    #[test]
    fn solves_3var_chain() {
        // (x1 ∨ ¬x2) ∧ (x2 ∨ ¬x3) ∧ (x3) — forces x3,x2,x1 = true
        let mut s = SatSolver::new();
        s.add_clause(vec![1, -2]);
        s.add_clause(vec![2, -3]);
        s.add_clause(vec![3]);
        let m = s.solve().unwrap().unwrap();
        assert!(m[0] && m[1] && m[2]);
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(SatSolver::new().name(), "SatSolver");
    }
}
