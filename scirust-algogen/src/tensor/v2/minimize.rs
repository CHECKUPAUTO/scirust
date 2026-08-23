//! Deterministic bounded minimization of research programs.
//!
//! Reduces a program to a smaller program that still satisfies a caller
//! predicate (typically "still fails verification" or "still diverges from
//! the oracle"), using leaf-stripping delta debugging:
//!
//! 1. candidate nodes are leaves — defined values no surviving op reads and
//!    that are not section roots;
//! 2. all current leaves are removed in one deterministic sweep; the removal
//!    is kept only when the predicate still holds on the rewritten program;
//! 3. sweeps repeat until none removes anything or
//!    [`MAX_MINIMIZATION_PASSES`] runs out.
//!
//! The result is deterministic for a given input program and predicate.
//! Minimization never changes signatures, steps, semantics, or root counts;
//! it only removes dead weight so humans and archives can read what remains.

use super::ir::{Op, Ref, ResearchProgram, Section, ValueId};
use super::verify::VerificationLimits;

/// Hard bound on greedy sweeps; realistic programs minimize long before this,
/// and stopping early is always safe (merely less reduced).
pub const MAX_MINIMIZATION_PASSES: usize = 64;

/// Why a program cannot be minimized. Generated research programs are
/// untrusted input: structural malformation is rejected with a structured
/// error instead of panicking mid-traversal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MinimizationError {
    /// A section references a local id that does not exist in it.
    LocalReferenceOutOfRange {
        /// `"init"`, `"step"` or `"finalize"`.
        section: &'static str,
        id: ValueId,
        section_len: usize,
    },
    /// A root list (init_state / next_state / outputs) points outside its
    /// section.
    RootOutOfRange {
        /// `"init_state"`, `"next_state"` or `"outputs"`.
        role: &'static str,
        id: ValueId,
        section_len: usize,
    },
}

impl std::fmt::Display for MinimizationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::LocalReferenceOutOfRange {
                section,
                id,
                section_len,
            } => write!(
                formatter,
                "{section} section references local {id} but has only {section_len} ops"
            ),
            Self::RootOutOfRange {
                role,
                id,
                section_len,
            } => write!(
                formatter,
                "{role} root {id} lies outside its section of {section_len} ops"
            ),
        }
    }
}

impl std::error::Error for MinimizationError {}

/// Statistics of one minimization run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MinimizationStats {
    /// Nodes removed relative to the input program.
    pub removed_nodes: usize,
    /// Sweeps executed.
    pub passes: usize,
    /// Node count of the final program.
    pub final_node_count: usize,
}

/// Reduce `program` while `predicate` keeps holding.
///
/// The predicate receives each trial program and must itself decide validity
/// (wrap [`super::verify_program`] inside it if validity must survive).
/// Signature fields (`inputs`, `items`, `state`, `steps`, `semantics`) are
/// preserved verbatim; only unreferenced non-root nodes may disappear.
///
/// Trust boundary: the program's reference structure is validated once,
/// up front, so arbitrary malformed IR is rejected with a structured
/// [`MinimizationError`] instead of panicking. Removals can never break the
/// established invariant (deleted nodes are by construction unreferenced),
/// so one validation suffices for every sweep.
pub fn minimize_program(
    program: &ResearchProgram,
    predicate: &mut dyn FnMut(&ResearchProgram) -> bool,
) -> Result<(ResearchProgram, MinimizationStats), MinimizationError> {
    validate_structure(program)?;
    let mut current = program.clone();
    let mut stats = MinimizationStats::default();

    for _ in 0..MAX_MINIMIZATION_PASSES
    {
        stats.passes += 1;
        let mut changed = false;

        // One greedy sweep over all three sections, in fixed order.
        if let Some(reduced) = strip_section(&current.init, &current.init_state, SectionKind::Init)
        {
            if let Some(trial) = apply(&current, &reduced)
            {
                if predicate(&trial)
                {
                    stats.removed_nodes += reduced.removed.len();
                    current = trial;
                    changed = true;
                }
            }
        }
        if let Some(reduced) = strip_section(&current.step, &current.next_state, SectionKind::Step)
        {
            if let Some(trial) = apply(&current, &reduced)
            {
                if predicate(&trial)
                {
                    stats.removed_nodes += reduced.removed.len();
                    current = trial;
                    changed = true;
                }
            }
        }
        if let Some(reduced) =
            strip_section(&current.finalize, &current.outputs, SectionKind::Finalize)
        {
            if let Some(trial) = apply(&current, &reduced)
            {
                if predicate(&trial)
                {
                    stats.removed_nodes += reduced.removed.len();
                    current = trial;
                    changed = true;
                }
            }
        }

        if !changed
        {
            break;
        }
    }

    stats.final_node_count = current.node_count();
    Ok((current, stats))
}

/// Establish the `Ref::Local(id) ⇒ id < section.ops.len()` and
/// `root id ⇒ id < section.ops.len()` invariants once, for all sections.
fn validate_structure(program: &ResearchProgram) -> Result<(), MinimizationError> {
    let sections = [
        (
            "init",
            &program.init,
            program.init_state.as_slice(),
            "init_state",
        ),
        (
            "step",
            &program.step,
            program.next_state.as_slice(),
            "next_state",
        ),
        (
            "finalize",
            &program.finalize,
            program.outputs.as_slice(),
            "outputs",
        ),
    ];
    for (name, section, roots, root_role) in sections
    {
        let len = section.ops.len();
        let mut malformed: Option<MinimizationError> = None;
        for op in &section.ops
        {
            op.for_each_ref(|reference| {
                if malformed.is_none()
                {
                    if let Ref::Local(id) = reference
                    {
                        if id >= len
                        {
                            malformed = Some(MinimizationError::LocalReferenceOutOfRange {
                                section: name,
                                id,
                                section_len: len,
                            });
                        }
                    }
                }
            });
            if let Some(error) = malformed
            {
                return Err(error);
            }
        }
        for &root in roots
        {
            if root >= len
            {
                return Err(MinimizationError::RootOutOfRange {
                    role: root_role,
                    id: root,
                    section_len: len,
                });
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum SectionKind {
    Init,
    Step,
    Finalize,
}

struct StripOutcome {
    ops: Vec<Op>,
    roots: Vec<ValueId>,
    removed: Vec<ValueId>,
    kind: SectionKind,
}

/// Compute the section that results from deleting every leaf at once, with
/// ids remapped; returns `None` when nothing is removable.
fn strip_section(section: &Section, roots: &[ValueId], kind: SectionKind) -> Option<StripOutcome> {
    let count = section.ops.len();
    if count == 0
    {
        return None;
    }
    let mut referenced = vec![false; count];
    for op in &section.ops
    {
        op.for_each_ref(|reference| {
            if let Ref::Local(id) = reference
            {
                referenced[id] = true;
            }
        });
    }
    let candidates: Vec<ValueId> = (0..count)
        .filter(|&id| !referenced[id] && !roots.contains(&id))
        .collect();
    if candidates.is_empty()
    {
        return None;
    }

    let removed: std::collections::BTreeSet<ValueId> = candidates.into_iter().collect();
    let remap = |id: ValueId| -> ValueId { id - removed.range(..id).count() };
    let mut ops: Vec<Op> = section
        .ops
        .iter()
        .enumerate()
        .filter(|(id, _)| !removed.contains(id))
        .map(|(_, op)| op.clone())
        .collect();
    for op in &mut ops
    {
        op.map_refs(|reference| match reference
        {
            Ref::Local(id) => Ref::Local(remap(id)),
            other => other,
        });
    }
    Some(StripOutcome {
        ops,
        roots: roots.iter().copied().map(remap).collect(),
        removed: removed.into_iter().collect(),
        kind,
    })
}

fn apply(program: &ResearchProgram, outcome: &StripOutcome) -> Option<ResearchProgram> {
    let mut trial = program.clone();
    match outcome.kind
    {
        SectionKind::Init =>
        {
            trial.init = Section::new(outcome.ops.clone());
            trial.init_state = outcome.roots.clone();
        },
        SectionKind::Step =>
        {
            trial.step = Section::new(outcome.ops.clone());
            trial.next_state = outcome.roots.clone();
        },
        SectionKind::Finalize =>
        {
            trial.finalize = Section::new(outcome.ops.clone());
            trial.outputs = outcome.roots.clone();
        },
    }
    Some(trial)
}

/// Convenience predicate: the minimized program must still verify.
pub fn verifying_predicate(limits: VerificationLimits) -> impl FnMut(&ResearchProgram) -> bool {
    move |program: &ResearchProgram| super::verify_program(program, limits).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::ir::{Bin, Un};
    use crate::tensor::v2::types::{DType, ScalarValue, ValueType};

    #[test]
    fn dead_leaves_are_removed_and_validity_survives() {
        // abs(x), sin(x) [dead], cos(x) [dead], plus an isolated constant
        // chain feeding nothing live.
        let program = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![
                Op::Abs(Un::new(Ref::Input(0))),  // 0 live
                Op::Sin(Un::new(Ref::Input(0))),  // 1 dead leaf
                Op::Cos(Un::new(Ref::Local(1))),  // 2 becomes a leaf after 1 dies
                Op::Const(ScalarValue::F64(7.0)), // 3 dead root-less const
            ]),
            vec![0],
        );
        let (minimized, stats) = minimize_program(
            &program,
            &mut verifying_predicate(VerificationLimits::default()),
        )
        .unwrap();
        assert_eq!(minimized.finalize.ops.len(), 1);
        assert_eq!(stats.removed_nodes, 3);
        assert!(stats.passes >= 2, "chained leaves need two sweeps");
        super::super::verify_program(&minimized, VerificationLimits::default()).unwrap();
        assert_eq!(minimized.outputs, vec![0]);
    }

    #[test]
    fn roots_are_never_stripped() {
        let program = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
            vec![0],
        );
        let (minimized, stats) = minimize_program(
            &program,
            &mut verifying_predicate(VerificationLimits::default()),
        )
        .unwrap();
        assert_eq!(stats.removed_nodes, 0);
        assert_eq!(minimized.finalize.ops.len(), 1);
        assert_eq!(minimized.outputs, vec![0]);
    }

    #[test]
    fn predicate_gates_removal() {
        let program = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![
                Op::Abs(Un::new(Ref::Input(0))),
                Op::Exp(Un::new(Ref::Input(0))),
            ]),
            vec![0],
        );
        // Predicate refuses any change: nothing may be removed even though
        // node 1 is a leaf.
        let mut conservative = |program: &ResearchProgram| program.node_count() == 2;
        let (minimized, stats) = minimize_program(&program, &mut conservative).unwrap();
        assert_eq!(stats.removed_nodes, 0);
        assert_eq!(minimized.node_count(), 2);
    }

    #[test]
    fn recurrence_sections_minimize_with_roots_intact() {
        let program = super::super::reference::welford_recurrence(2);
        let before_outputs = program.outputs.clone();
        let (minimized, _) = minimize_program(
            &program,
            &mut verifying_predicate(VerificationLimits::default()),
        )
        .unwrap();
        assert_eq!(minimized.outputs, before_outputs);
        assert_eq!(minimized.steps, program.steps);
        assert_eq!(minimized.state, program.state);
        super::super::verify_program(&minimized, VerificationLimits::default()).unwrap();
    }

    #[test]
    fn add_node_shape_is_preserved_when_live() {
        let program = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![
                Op::Add(Bin::new(Ref::Input(0), Ref::Input(0))),
                Op::Tanh(Un::new(Ref::Input(0))),
            ]),
            vec![0],
        );
        let (minimized, stats) = minimize_program(
            &program,
            &mut verifying_predicate(VerificationLimits::default()),
        )
        .unwrap();
        assert_eq!(stats.removed_nodes, 1);
        assert!(matches!(minimized.finalize.ops.first(), Some(Op::Add(_))));
    }

    /// Regression: a hostile `Ref::Local` beyond the section used to index
    /// straight out of bounds and panic; it must now be a structured error.
    #[test]
    fn malformed_local_reference_is_rejected_not_panic() {
        let program = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![
                Op::Abs(Un::new(Ref::Input(0))),
                Op::Neg(Un::new(Ref::Local(99))),
            ]),
            vec![0],
        );
        let error = minimize_program(&program, &mut |_| true)
            .expect_err("malformed reference must be rejected");
        assert_eq!(
            error,
            MinimizationError::LocalReferenceOutOfRange {
                section: "finalize",
                id: 99,
                section_len: 2,
            }
        );
    }

    /// A root pointing outside its section is equally rejected.
    #[test]
    fn malformed_root_is_rejected_not_panic() {
        let program = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
            vec![7],
        );
        let error = minimize_program(&program, &mut |_| true)
            .expect_err("out-of-range root must be rejected");
        assert_eq!(
            error,
            MinimizationError::RootOutOfRange {
                role: "outputs",
                id: 7,
                section_len: 1,
            }
        );
    }

    /// Malformed recurrence sections are covered by the same boundary.
    #[test]
    fn malformed_step_section_is_rejected() {
        let mut program = super::super::reference::welford_recurrence(2);
        program.step = Section::new(vec![]);
        program.next_state = vec![3];
        let error = minimize_program(&program, &mut |_| true)
            .expect_err("empty step section with a root must be rejected");
        assert!(matches!(
            error,
            MinimizationError::RootOutOfRange {
                role: "next_state",
                ..
            }
        ));
    }
}
