//! Canonicalization: deterministic simplification to a canonical form.
//!
//! The pass pipeline runs to a fixed point (bounded by
//! [`MAX_SIMPLIFY_PASSES`]) over four rule families, in fixed order, followed
//! by dead-code elimination against the live roots (`init_state`,
//! `next_state`, `outputs`):
//!
//! 1. **Safe constant folding** — operators over constant operands evaluate
//!    natively at build time *only when the float result is finite* (Boolean
//!    results always fold). Anything else stays symbolic so the runtime
//!    regime decides.
//! 2. **Identity rewrites** — every rule's validity domain is documented
//!    below; matches become aliases (the node is replaced by its target).
//! 3. **Commutative operand normalization** — operands of provably
//!    commutative operators sort under a fixed reference order.
//! 4. **Common-subexpression elimination** — duplicate definitions (by exact
//!    structural key) become aliases.
//!
//! # Validity domains (normative)
//!
//! All rules hold under the **canonical numeric regime**: IEEE-754 arithmetic,
//! a finite-valued defined domain, *signed-zero insensitivity* (rewrites may
//! merge value flows that differ only in ±0 propagation), NaN excluded by the
//! runtime gate.
//!
//! Applied:
//! * `Add(x, 0) -> x`, `Sub(x, 0) -> x` — exact except `-0 + 0 = +0`;
//!   signed-zero flows are merged by contract.
//! * `Mul(x, 1) -> x`, `Div(x, 1) -> x` — bit-exact for all finite `x`.
//! * `Min(x, x) -> x`, `Max(x, x) -> x`, `Dot(x, x)` kept (not an identity),
//!   `Eq(x, x)`/`Ne(x, x)` kept (observable Boolean) — only genuinely
//!   value-preserving identities alias.
//! * `Neg(Neg(x)) -> x`, `Not(Not(b)) -> b` — double negation.
//! * `Select(_, v, v) -> v`; `And(b, true) -> b`; `Or(b, false) -> b`.
//! * Commutative normalization for `Add, Mul, Min, Max, And, Or, Dot, Eq, Ne`
//!   — IEEE addition/multiplication/comparison/min-max and dot products are
//!   commutative bit-for-bit on the defined domain.
//!
//! **Deliberately rejected** (unsound under IEEE):
//! * `Mul(x, 0) -> 0` — fails for `x = ±Infinity`/NaN.
//! * `Sub(x, x) -> 0`, `Div(x, x) -> 1` — fail for infinite `x`.
//! * Associativity or distributivity of any kind — floating-point rounding.
//! * `Log(exp(x)) -> x`, `Sqrt(Mul(x, x)) -> Abs(x)` — rounding gaps.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::ir::{Bin, Op, Ref, ResearchProgram, Section, ShapeTo, ValueId};
use super::types::ScalarValue;
use super::verify::{
    ProgramError, VerificationLimits, VerifiedProgram, analyze_section_active, verify_program,
};

/// Hard bound on canonicalization passes; realistic programs reach their fixed
/// point long before this, and stopping early is always safe (the result is
/// merely not fully reduced).
pub const MAX_SIMPLIFY_PASSES: usize = 16;

/// Statistics of one canonicalization run (structural diagnostics only).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimplifyStats {
    /// Completed passes (including the confirming final pass).
    pub passes: usize,
    /// Nodes removed as dead code, summed over passes.
    pub dead_nodes_removed: usize,
    /// Constants folded.
    pub constants_folded: usize,
    /// Identity rewrites applied (aliasing rewrites included).
    pub identities_applied: usize,
    /// Duplicate definitions eliminated by CSE.
    pub duplicates_removed: usize,
}

/// The outcome of canonicalizing a program.
#[derive(Debug, Clone, PartialEq)]
pub struct Canonicalized {
    pub program: ResearchProgram,
    pub stats: SimplifyStats,
}

/// Reduce `program` to its canonical form.
///
/// The input must verify (checked first). The output verifies against the same
/// limits and executes identically under the default numerical regime.
pub fn canonicalize(
    program: &ResearchProgram,
    limits: VerificationLimits,
) -> Result<Canonicalized, ProgramError> {
    let mut verified = verify_program(program, limits)?;
    let mut current = program.clone();
    let mut stats = SimplifyStats::default();

    for _ in 0..MAX_SIMPLIFY_PASSES
    {
        let (next, delta) = simplify_once(&current, &verified)?;
        stats.passes += 1;
        stats.dead_nodes_removed += delta.dead_nodes_removed;
        stats.constants_folded += delta.constants_folded;
        stats.identities_applied += delta.identities_applied;
        stats.duplicates_removed += delta.duplicates_removed;

        if !delta.changed()
        {
            break;
        }
        current = next;
        verified = verify_program(&current, limits)?;
    }

    Ok(Canonicalized {
        program: current,
        stats,
    })
}

#[derive(Default)]
struct Delta {
    dead_nodes_removed: usize,
    constants_folded: usize,
    identities_applied: usize,
    duplicates_removed: usize,
    normalized: usize,
}

impl Delta {
    fn changed(&self) -> bool {
        self.dead_nodes_removed > 0
            || self.constants_folded > 0
            || self.identities_applied > 0
            || self.duplicates_removed > 0
            || self.normalized > 0
    }
}

/// One bounded simplification sweep over all sections.
fn simplify_once(
    program: &ResearchProgram,
    verified: &VerifiedProgram,
) -> Result<(ResearchProgram, Delta), ProgramError> {
    let mut delta = Delta::default();

    let (init_ops, init_aliases, init_remap) = simplify_section(&program.init, &mut delta);
    let (step_ops, step_aliases, step_remap) = simplify_section(&program.step, &mut delta);
    let (finalize_ops, finalize_aliases, finalize_remap) =
        simplify_section(&program.finalize, &mut delta);

    // Rebind section roots through alias chains / compaction; rooted aliases
    // become explicit passthrough reshapes appended to their section.
    let (init_ops, init_state) = rebind_roots(
        init_ops,
        &program.init_state,
        &init_aliases,
        &init_remap,
        &verified.init_types,
    );
    let (step_ops, next_state) = rebind_roots(
        step_ops,
        &program.next_state,
        &step_aliases,
        &step_remap,
        &verified.step_types,
    );
    let (finalize_ops, outputs) = rebind_roots(
        finalize_ops,
        &program.outputs,
        &finalize_aliases,
        &finalize_remap,
        &verified.finalize_types,
    );

    let mut next = ResearchProgram {
        inputs: program.inputs.clone(),
        items: program.items.clone(),
        state: program.state.clone(),
        steps: program.steps,
        init: Section::new(init_ops),
        init_state,
        step: Section::new(step_ops),
        next_state,
        finalize: Section::new(finalize_ops),
        outputs,
    };

    // Dead-code elimination per section with index remapping.
    let before = program.node_count();
    let (init, init_state) = dce_section(std::mem::take(&mut next.init), &next.init_state);
    next.init = init;
    next.init_state = init_state;
    let (step, next_state) = dce_section(std::mem::take(&mut next.step), &next.next_state);
    next.step = step;
    next.next_state = next_state;
    let (finalize, outputs) = dce_section(std::mem::take(&mut next.finalize), &next.outputs);
    next.finalize = finalize;
    next.outputs = outputs;
    let removed = before.saturating_sub(next.node_count());
    delta.dead_nodes_removed += removed;

    Ok((next, delta))
}

// ---------------------------------------------------------------------------
// Per-section machinery
// ---------------------------------------------------------------------------

/// Fully resolve `reference` through an alias chain.
fn resolve(aliases: &[Option<Ref>], reference: Ref) -> Ref {
    let mut current = reference;
    let mut hops = 0usize;
    loop
    {
        match current
        {
            Ref::Local(id) => match aliases.get(id).copied().flatten()
            {
                Some(target) =>
                {
                    current = target;
                    hops += 1;
                    if hops > aliases.len()
                    {
                        // Cannot happen (aliases always target strictly
                        // earlier definitions); defensive bail-out.
                        return current;
                    }
                },
                None => return current,
            },
            other => return other,
        }
    }
}

/// Total order over references used by commutative normalization.
///
/// Any fixed total order yields a consistent canonical form; ranks group
/// external references before locals so common patterns normalise stably.
fn ref_order(reference: Ref) -> (u8, u64) {
    match reference
    {
        Ref::Input(index) => (0, index as u64),
        Ref::Item(index) => (1, index as u64),
        Ref::StatePrev(slot) => (2, slot as u64),
        Ref::StateFinal(slot) => (3, slot as u64),
        Ref::Local(id) => (4, id as u64),
    }
}

/// Constant view of each definition in the section (`None` when not a
/// compile-time constant).
fn const_views(ops: &[Op]) -> Vec<Option<ScalarValue>> {
    ops.iter()
        .map(|op| match op
        {
            Op::Const(value) => Some(*value),
            _ => None,
        })
        .collect()
}

fn is_zero(value: Option<ScalarValue>) -> bool {
    matches!(value, Some(ScalarValue::F32(v)) if v == 0.0)
        || matches!(value, Some(ScalarValue::F64(v)) if v == 0.0)
}

fn is_one(value: Option<ScalarValue>) -> bool {
    matches!(value, Some(ScalarValue::F32(v)) if v == 1.0)
        || matches!(value, Some(ScalarValue::F64(v)) if v == 1.0)
}

fn is_true(value: Option<ScalarValue>) -> bool {
    matches!(value, Some(ScalarValue::Bool(true)))
}

/// Simplify one section. Returns rewritten ops, the alias map in *old-id*
/// space (targets already resolved into the compacted id space), and the
/// declared types indexed by old ids (needed for passthrough materialisation).
#[allow(clippy::type_complexity)]
fn simplify_section(
    section: &Section,
    delta: &mut Delta,
) -> (Vec<Op>, Vec<Option<Ref>>, Vec<usize>) {
    let ops = &section.ops;
    let mut aliases: Vec<Option<Ref>> = vec![None; ops.len()];
    let mut consts = const_views(ops);

    // ---- folding, identities, normalization ---------------------------------
    let mut working: Vec<Op> = ops.clone();
    for node in 0..working.len()
    {
        // Resolve current refs through aliases built so far.
        working[node].map_refs(|reference| resolve(&aliases, reference));

        // Safe constant folding.
        if let Some(folded) = try_fold(&working[node], &consts, &aliases)
        {
            working[node] = Op::Const(folded);
            consts[node] = Some(folded);
            delta.constants_folded += 1;
            continue;
        }

        // Identity rewrites.
        if let Some(target) = identity_target(&working[node], &consts)
            .or_else(|| double_negation_target(&working, node))
        {
            aliases[node] = Some(target);
            delta.identities_applied += 1;
            continue;
        }

        // Commutative normalization.
        if normalize_commutative(&mut working[node])
        {
            delta.normalized += 1;
        }
    }

    // ---- common-subexpression elimination -----------------------------------
    let mut seen: HashMap<Vec<u8>, ValueId> = HashMap::new();
    for node in 0..working.len()
    {
        if aliases[node].is_some()
        {
            continue;
        }
        let key = cse_key(&working[node]);
        match seen.get(&key)
        {
            Some(&first) =>
            {
                aliases[node] = Some(Ref::Local(first));
                delta.duplicates_removed += 1;
            },
            None =>
            {
                seen.insert(key, node);
            },
        }
    }

    // Thread aliases through surviving concrete ops.
    for op in working.iter_mut()
    {
        op.map_refs(|reference| resolve(&aliases, reference));
    }

    // ---- compaction ----------------------------------------------------------
    let keep: Vec<bool> = aliases.iter().map(|alias| alias.is_none()).collect();
    let (compacted, remap) = compact(&working, &keep);

    // Translate the alias map into the compacted id space, resolving chains so
    // callers can rebind roots with a single lookup.
    let translated: Vec<Option<Ref>> = (0..ops.len())
        .map(|old_id| {
            aliases[old_id]
                .map(|target| resolve(&aliases, target))
                .and_then(|target| match target
                {
                    Ref::Local(resolved_old) => remap
                        .get(resolved_old)
                        .copied()
                        .filter(|&new_id| new_id != usize::MAX)
                        .map(Ref::Local),
                    external => Some(external),
                })
        })
        .collect();

    (compacted, translated, remap)
}

/// Remove aliased/dead placeholder entries, keeping live ops contiguous.
fn compact(ops: &[Op], keep: &[bool]) -> (Vec<Op>, Vec<usize>) {
    let mut remap = vec![usize::MAX; ops.len()];
    let mut out = Vec::new();
    for (index, (op, &kept)) in ops.iter().zip(keep).enumerate()
    {
        if kept
        {
            remap[index] = out.len();
            out.push(op.clone());
        }
    }
    for op in out.iter_mut()
    {
        op.map_refs(|reference| match reference
        {
            Ref::Local(id) =>
            {
                let mapped = remap[id];
                debug_assert_ne!(mapped, usize::MAX, "dangling local after compaction");
                Ref::Local(mapped)
            },
            other => other,
        });
    }
    (out, remap)
}

/// Rebind section roots through alias chains. A rooted alias becomes an
/// explicit passthrough `Reshape` to its own type shape (a transparent
/// row-major copy) appended to the section, keeping outputs concrete values.
fn rebind_roots(
    mut ops: Vec<Op>,
    roots: &[ValueId],
    aliases: &[Option<Ref>],
    remap: &[usize],
    old_types: &[super::types::ValueType],
) -> (Vec<Op>, Vec<ValueId>) {
    let mut rebound = Vec::with_capacity(roots.len());
    for &root in roots
    {
        match aliases.get(root).copied().flatten()
        {
            Some(replacement) =>
            {
                // The root was rewritten away; materialise it as an explicit
                // passthrough reshape (transparent row-major copy).
                let shape = old_types
                    .get(root)
                    .map(|value_type| value_type.shape.clone())
                    .unwrap_or_default();
                ops.push(Op::Reshape(ShapeTo {
                    src: replacement,
                    shape,
                }));
                rebound.push(ops.len() - 1);
            },
            None =>
            {
                let mapped = remap
                    .get(root)
                    .copied()
                    .filter(|&id| id != usize::MAX)
                    .unwrap_or(root);
                rebound.push(mapped);
            },
        }
    }
    (ops, rebound)
}

/// Backward liveness DCE for one section, remapping its roots.
fn dce_section(section: Section, roots: &[ValueId]) -> (Section, Vec<ValueId>) {
    let active = analyze_section_active(&section.ops, roots);
    let (compacted, remap) = compact(&section.ops, &active);
    let new_roots = roots.iter().map(|&root| remap[root]).collect();
    (Section::new(compacted), new_roots)
}

// ---------------------------------------------------------------------------
// Rule engines
// ---------------------------------------------------------------------------

/// Look up the constant an operand reference designates, if any.
fn const_of(
    reference: Ref,
    consts: &[Option<ScalarValue>],
    aliases: &[Option<Ref>],
) -> Option<ScalarValue> {
    match resolve(aliases, reference)
    {
        Ref::Local(id) => consts.get(id).copied().flatten(),
        _ => None,
    }
}

/// Native-dtype evaluation of fully constant operations. Returns `None` when
/// the op is not foldable or the float result is non-finite (left symbolic so
/// the runtime regime decides).
#[allow(clippy::too_many_lines)]
fn try_fold(
    op: &Op,
    consts: &[Option<ScalarValue>],
    aliases: &[Option<Ref>],
) -> Option<ScalarValue> {
    // Gather operand constants in reference order.
    let mut operands: Vec<Option<ScalarValue>> = Vec::with_capacity(3);
    op.for_each_ref(|reference| {
        operands.push(const_of(reference, consts, aliases));
    });

    let all_const = operands.iter().all(Option::is_some);
    if !all_const
    {
        return None;
    }
    let values: Vec<ScalarValue> = operands.into_iter().flatten().collect();

    use ScalarValue::*;
    macro_rules! f32_bin {
        ($a:expr, $b:expr, $f:expr) => {{
            let r: f32 = $f($a, $b);
            if r.is_finite() { Some(F32(r)) } else { None }
        }};
    }
    macro_rules! f64_bin {
        ($a:expr, $b:expr, $f:expr) => {{
            let r: f64 = $f($a, $b);
            if r.is_finite() { Some(F64(r)) } else { None }
        }};
    }

    let binary_float = |a: ScalarValue, b: ScalarValue, op: &Op| -> Option<ScalarValue> {
        match (a, b)
        {
            (F32(x), F32(y)) => match op
            {
                Op::Add(_) => f32_bin!(x, y, |a, b| a + b),
                Op::Sub(_) => f32_bin!(x, y, |a, b| a - b),
                Op::Mul(_) => f32_bin!(x, y, |a, b| a * b),
                Op::Div(_) => f32_bin!(x, y, |a, b| a / b),
                Op::Pow(_) => f32_bin!(x, y, f32::powf),
                Op::Min(_) => Some(F32(x.min(y))),
                Op::Max(_) => Some(F32(x.max(y))),
                _ => None,
            },
            (F64(x), F64(y)) => match op
            {
                Op::Add(_) => f64_bin!(x, y, |a, b| a + b),
                Op::Sub(_) => f64_bin!(x, y, |a, b| a - b),
                Op::Mul(_) => f64_bin!(x, y, |a, b| a * b),
                Op::Div(_) => f64_bin!(x, y, |a, b| a / b),
                Op::Pow(_) => f64_bin!(x, y, f64::powf),
                Op::Min(_) => Some(F64(x.min(y))),
                Op::Max(_) => Some(F64(x.max(y))),
                _ => None,
            },
            _ => None,
        }
    };

    let unary_float = |a: ScalarValue, op: &Op| -> Option<ScalarValue> {
        match a
        {
            F32(x) =>
            {
                let r: f32 = match op
                {
                    Op::Neg(_) => -x,
                    Op::Abs(_) => x.abs(),
                    Op::Exp(_) => x.exp(),
                    Op::Exp2(_) => x.exp2(),
                    Op::Expm1(_) => x.exp_m1(),
                    Op::Log(_) => x.ln(),
                    Op::Log2(_) => x.log2(),
                    Op::Log1p(_) => x.ln_1p(),
                    Op::Sqrt(_) => x.sqrt(),
                    Op::Rsqrt(_) => 1.0 / x.sqrt(),
                    Op::Sin(_) => x.sin(),
                    Op::Cos(_) => x.cos(),
                    Op::Tanh(_) => x.tanh(),
                    _ => return None,
                };
                if r.is_finite() { Some(F32(r)) } else { None }
            },
            F64(x) =>
            {
                let r: f64 = match op
                {
                    Op::Neg(_) => -x,
                    Op::Abs(_) => x.abs(),
                    Op::Exp(_) => x.exp(),
                    Op::Exp2(_) => x.exp2(),
                    Op::Expm1(_) => x.exp_m1(),
                    Op::Log(_) => x.ln(),
                    Op::Log2(_) => x.log2(),
                    Op::Log1p(_) => x.ln_1p(),
                    Op::Sqrt(_) => x.sqrt(),
                    Op::Rsqrt(_) => 1.0 / x.sqrt(),
                    Op::Sin(_) => x.sin(),
                    Op::Cos(_) => x.cos(),
                    Op::Tanh(_) => x.tanh(),
                    _ => return None,
                };
                if r.is_finite() { Some(F64(r)) } else { None }
            },
            _ => None,
        }
    };

    match op
    {
        Op::Add(_)
        | Op::Sub(_)
        | Op::Mul(_)
        | Op::Div(_)
        | Op::Pow(_)
        | Op::Min(_)
        | Op::Max(_) => binary_float(values[0], values[1], op),

        Op::Eq(_) | Op::Ne(_) | Op::Lt(_) | Op::Le(_) | Op::Gt(_) | Op::Ge(_) =>
        {
            let ord = match (values[0], values[1])
            {
                (F32(a), F32(b)) => a.partial_cmp(&b),
                (F64(a), F64(b)) => a.partial_cmp(&b),
                _ => None,
            }?;
            let bit = match op
            {
                Op::Eq(_) => ord == std::cmp::Ordering::Equal,
                Op::Ne(_) => ord != std::cmp::Ordering::Equal,
                Op::Lt(_) => ord == std::cmp::Ordering::Less,
                Op::Le(_) => ord != std::cmp::Ordering::Greater,
                Op::Gt(_) => ord == std::cmp::Ordering::Greater,
                Op::Ge(_) => ord != std::cmp::Ordering::Less,
                _ => return None,
            };
            Some(Bool(bit))
        },

        Op::And(_) | Op::Or(_) =>
        {
            let (a, b) = match (values[0], values[1])
            {
                (Bool(a), Bool(b)) => (a, b),
                _ => return None,
            };
            Some(Bool(
                if matches!(op, Op::And(_))
                {
                    a && b
                }
                else
                {
                    a || b
                },
            ))
        },
        Op::Not(_) => match values[0]
        {
            Bool(bit) => Some(Bool(!bit)),
            _ => None,
        },

        Op::MulAdd(_) => match (values[0], values[1], values[2])
        {
            (F32(a), F32(b), F32(c)) =>
            {
                let r = a.mul_add(b, c);
                if r.is_finite() { Some(F32(r)) } else { None }
            },
            (F64(a), F64(b), F64(c)) =>
            {
                let r = a.mul_add(b, c);
                if r.is_finite() { Some(F64(r)) } else { None }
            },
            _ => None,
        },

        Op::Neg(_)
        | Op::Abs(_)
        | Op::Exp(_)
        | Op::Exp2(_)
        | Op::Expm1(_)
        | Op::Log(_)
        | Op::Log2(_)
        | Op::Log1p(_)
        | Op::Sqrt(_)
        | Op::Rsqrt(_)
        | Op::Sin(_)
        | Op::Cos(_)
        | Op::Tanh(_) => unary_float(values[0], op),

        Op::Clamp(_) => match (values[0], values[1], values[2])
        {
            (F32(a), F32(b), F32(c)) =>
            {
                let r = a.min(c).max(b);
                if r.is_finite() { Some(F32(r)) } else { None }
            },
            (F64(a), F64(b), F64(c)) =>
            {
                let r = a.min(c).max(b);
                if r.is_finite() { Some(F64(r)) } else { None }
            },
            _ => None,
        },

        _ => None,
    }
}

/// Identity-rewrite table. `consts` views the *original* section (aliases
/// never point at constants directly because folding runs first).
fn identity_target(op: &Op, consts: &[Option<ScalarValue>]) -> Option<Ref> {
    match op
    {
        Op::Add(bin) =>
        {
            if is_zero(direct_const(bin.lhs, consts))
            {
                return Some(bin.rhs);
            }
            if is_zero(direct_const(bin.rhs, consts))
            {
                return Some(bin.lhs);
            }
            None
        },
        Op::Sub(bin) =>
        {
            if is_zero(direct_const(bin.rhs, consts))
            {
                return Some(bin.lhs);
            }
            None
        },
        Op::Mul(bin) =>
        {
            if is_one(direct_const(bin.lhs, consts))
            {
                return Some(bin.rhs);
            }
            if is_one(direct_const(bin.rhs, consts))
            {
                return Some(bin.lhs);
            }
            None
        },
        Op::Div(bin) =>
        {
            if is_one(direct_const(bin.rhs, consts))
            {
                return Some(bin.lhs);
            }
            None
        },
        Op::Min(bin) | Op::Max(bin) if bin.lhs == bin.rhs => Some(bin.lhs),
        Op::And(bin) =>
        {
            if is_true(direct_const(bin.lhs, consts))
            {
                return Some(bin.rhs);
            }
            if is_true(direct_const(bin.rhs, consts))
            {
                return Some(bin.lhs);
            }
            None
        },
        Op::Or(bin) =>
        {
            if !is_true(direct_const(bin.lhs, consts)) && direct_is_bool_false(bin.lhs, consts)
            {
                return Some(bin.rhs);
            }
            if direct_is_bool_false(bin.rhs, consts)
            {
                return Some(bin.lhs);
            }
            None
        },
        Op::Select(ter) =>
        {
            if ter.b == ter.c
            {
                return Some(ter.b);
            }
            None
        },
        _ => None,
    }
}

/// Double-negation identities: `Neg(Neg(x)) -> x`, `Not(Not(b)) -> b`.
///
/// Validity: exact for finite values (signaling-NaN quieting is unreachable
/// under the runtime gate).
fn double_negation_target(working: &[Op], node: usize) -> Option<Ref> {
    match &working[node]
    {
        Op::Neg(un) => match un.src
        {
            Ref::Local(inner_id) => match working.get(inner_id)?
            {
                Op::Neg(inner) => Some(inner.src),
                _ => None,
            },
            _ => None,
        },
        Op::Not(un) => match un.src
        {
            Ref::Local(inner_id) => match working.get(inner_id)?
            {
                Op::Not(inner) => Some(inner.src),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

/// Constant designated by a *direct* `Ref::Local` (no alias chasing needed:
/// folding has already replaced foldable predecessors in the same sweep).
fn direct_const(reference: Ref, consts: &[Option<ScalarValue>]) -> Option<ScalarValue> {
    match reference
    {
        Ref::Local(id) => consts.get(id).copied().flatten(),
        _ => None,
    }
}

fn direct_is_bool_false(reference: Ref, consts: &[Option<ScalarValue>]) -> bool {
    matches!(
        direct_const(reference, consts),
        Some(ScalarValue::Bool(false))
    )
}

/// Reorder operands of provably commutative operators under [`ref_order`].
fn normalize_commutative(op: &mut Op) -> bool {
    fn swap_pair(bin: &mut Bin) -> bool {
        if ref_order(bin.rhs) < ref_order(bin.lhs)
        {
            std::mem::swap(&mut bin.lhs, &mut bin.rhs);
            true
        }
        else
        {
            false
        }
    }

    match op
    {
        Op::Add(bin)
        | Op::Mul(bin)
        | Op::Min(bin)
        | Op::Max(bin)
        | Op::And(bin)
        | Op::Or(bin)
        | Op::Eq(bin)
        | Op::Ne(bin)
        | Op::Dot(bin) => swap_pair(bin),
        _ => false,
    }
}

/// Exact structural key for CSE: opcode tag, operand refs and attributes.
fn cse_key(op: &Op) -> Vec<u8> {
    fn push_u64(bytes: &mut Vec<u8>, value: usize) {
        bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }
    fn push_ref(bytes: &mut Vec<u8>, reference: Ref) {
        let (rank, index) = ref_order(reference);
        bytes.push(rank);
        push_u64(bytes, index as usize);
    }

    let mut key = Vec::new();
    key.extend_from_slice(&op.tag().to_le_bytes());
    op.for_each_ref(|reference| push_ref(&mut key, reference));
    match op
    {
        Op::Const(value) => match *value
        {
            ScalarValue::F32(inner) =>
            {
                key.push(0);
                key.extend_from_slice(&inner.to_bits().to_le_bytes());
            },
            ScalarValue::F64(inner) =>
            {
                key.push(1);
                key.extend_from_slice(&inner.to_bits().to_le_bytes());
            },
            ScalarValue::Bool(inner) =>
            {
                key.push(2);
                key.push(u8::from(inner));
            },
        },
        Op::ReduceSum(reduce)
        | Op::ReduceProd(reduce)
        | Op::ReduceMax(reduce)
        | Op::ReduceMin(reduce)
        | Op::ReduceMean(reduce) => match reduce.axis
        {
            None => key.push(0),
            Some(axis) =>
            {
                key.push(1);
                push_u64(&mut key, axis);
            },
        },
        Op::Reshape(to) | Op::BroadcastTo(to) =>
        {
            push_u64(&mut key, to.shape.len());
            for &dimension in &to.shape
            {
                push_u64(&mut key, dimension);
            }
        },
        Op::Squeeze(axis_op) | Op::Unsqueeze(axis_op) => push_u64(&mut key, axis_op.axis),
        Op::Transpose(permute) =>
        {
            push_u64(&mut key, permute.perm.len());
            for &axis in &permute.perm
            {
                push_u64(&mut key, axis);
            }
        },
        Op::Concat { axis, .. } => push_u64(&mut key, *axis),
        Op::Narrow(narrow) =>
        {
            push_u64(&mut key, narrow.axis);
            push_u64(&mut key, narrow.start);
            push_u64(&mut key, narrow.len);
        },
        _ =>
        {},
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::interpret::{ExecutionPolicy, ValueTensor, execute_program};
    use crate::tensor::v2::ir::{Reduce, Ter, Un};
    use crate::tensor::v2::types::ValueType;
    use crate::tensor::v2::{DType, program_digest};

    fn f32_type(shape: &[usize]) -> ValueType {
        ValueType::new(DType::F32, shape.to_vec())
    }

    fn tensor_f32(data: &[f32], shape: &[usize]) -> ValueTensor {
        ValueTensor::new(
            DType::F32,
            shape.to_vec(),
            data.iter().map(|&value| value as f64).collect(),
        )
        .unwrap()
    }

    fn run(program: &ResearchProgram) -> Vec<ValueTensor> {
        execute_program(
            program,
            &[tensor_f32(&[1.5, -2.0, 0.25, 3.0], &[4])],
            &[],
            ExecutionPolicy::default(),
            VerificationLimits::default(),
        )
        .unwrap()
        .outputs
    }

    /// Differential helper: canonicalization must preserve execution
    /// bit-for-bit and verify afterwards.
    fn assert_semantics_preserved(program: &ResearchProgram) -> Canonicalized {
        let before = run(program);
        let canonical = canonicalize(program, VerificationLimits::default()).unwrap();
        let after = run(&canonical.program);
        assert_eq!(before.len(), after.len());
        for (index, (a, b)) in before.iter().zip(&after).enumerate()
        {
            assert_eq!(a.shape, b.shape, "output {index} shape");
            for (x, y) in a.data.iter().zip(&b.data)
            {
                assert_eq!(x.to_bits(), y.to_bits(), "output {index}");
            }
        }
        verify_program(&canonical.program, VerificationLimits::default())
            .expect("canonicalized program must verify");
        canonical
    }

    #[test]
    fn add_zero_sub_zero_mul_one_div_one_are_eliminated() {
        let program = ResearchProgram::expression(
            vec![f32_type(&[4])],
            Section::new(vec![
                Op::Const(ScalarValue::F32(0.0)),                // 0
                Op::Add(Bin::new(Ref::Input(0), Ref::Local(0))), // 1: x+0
                Op::Const(ScalarValue::F32(1.0)),                // 2
                Op::Mul(Bin::new(Ref::Local(1), Ref::Local(2))), // 3: *1
                Op::Sub(Bin::new(Ref::Local(3), Ref::Local(0))), // 4: -0
                Op::Const(ScalarValue::F32(1.0)),                // 5
                Op::Div(Bin::new(Ref::Local(4), Ref::Local(5))), // 6: /1
            ]),
            vec![6],
        );
        let canonical = assert_semantics_preserved(&program);
        // x survives as one passthrough reshape.
        assert_eq!(canonical.program.finalize.len(), 1);
        assert!(canonical.stats.identities_applied >= 4);
    }

    #[test]
    fn min_max_self_and_select_same_branches_collapse() {
        let program = ResearchProgram::expression(
            vec![f32_type(&[4])],
            Section::new(vec![
                Op::Const(ScalarValue::F32(0.0)),                // 0
                Op::Lt(Bin::new(Ref::Input(0), Ref::Local(0))),  // 1: mask
                Op::Max(Bin::new(Ref::Input(0), Ref::Input(0))), // 2 -> x
                Op::Min(Bin::new(Ref::Local(2), Ref::Local(2))), // 3 -> x
                Op::Select(Ter::new(Ref::Local(1), Ref::Local(3), Ref::Local(3))), // 4 -> x
                Op::Abs(Un::new(Ref::Local(4))),                 // 5
            ]),
            vec![5],
        );
        let canonical = assert_semantics_preserved(&program);
        // Once Select collapses onto x its mask loses every consumer and is
        // removed by DCE; only abs(x) remains.
        assert_eq!(canonical.program.finalize.len(), 1);
        assert!(matches!(
            canonical.program.finalize.ops.last(),
            Some(Op::Abs(_))
        ));
    }

    #[test]
    fn double_negation_and_double_not_collapse() {
        let program = ResearchProgram::expression(
            vec![f32_type(&[4])],
            Section::new(vec![
                Op::Neg(Un::new(Ref::Input(0))),
                Op::Neg(Un::new(Ref::Local(0))),
                Op::Lt(Bin::new(Ref::Input(0), Ref::Input(0))),
                Op::Not(Un::new(Ref::Local(2))),
                Op::Not(Un::new(Ref::Local(3))),
                Op::Select(Ter::new(Ref::Local(4), Ref::Input(0), Ref::Local(1))),
            ]),
            vec![5],
        );
        let canonical = assert_semantics_preserved(&program);
        assert!(canonical.stats.identities_applied >= 2);
    }

    #[test]
    fn mul_by_zero_is_deliberately_not_folded() {
        let program = ResearchProgram::expression(
            vec![f32_type(&[4])],
            Section::new(vec![
                Op::Const(ScalarValue::F32(0.0)),
                Op::Mul(Bin::new(Ref::Input(0), Ref::Local(0))),
            ]),
            vec![1],
        );
        let canonical = assert_semantics_preserved(&program);
        // The multiplication must survive (IEEE-unsound to remove).
        assert!(
            canonical
                .program
                .finalize
                .ops
                .iter()
                .any(|op| matches!(op, Op::Mul(_)))
        );
    }

    #[test]
    fn sub_x_x_is_deliberately_not_folded() {
        let program = ResearchProgram::expression(
            vec![f32_type(&[4])],
            Section::new(vec![Op::Sub(Bin::new(Ref::Input(0), Ref::Input(0)))]),
            vec![0],
        );
        let canonical = assert_semantics_preserved(&program);
        assert!(matches!(
            canonical.program.finalize.ops.last(),
            Some(Op::Sub(_))
        ));
    }

    #[test]
    fn constant_expressions_fold_to_single_constants() {
        let program = ResearchProgram::expression(
            vec![f32_type(&[4])],
            Section::new(vec![
                Op::Const(ScalarValue::F32(2.0)),
                Op::Const(ScalarValue::F32(3.0)),
                Op::Add(Bin::new(Ref::Local(0), Ref::Local(1))),
                Op::Exp(Un::new(Ref::Local(2))),
                Op::Mul(Bin::new(Ref::Local(3), Ref::Local(3))),
                Op::ReduceSum(Reduce {
                    src: Ref::Input(0),
                    axis: None,
                }),
                Op::Add(Bin::new(Ref::Local(4), Ref::Local(5))),
            ]),
            vec![6],
        );
        let canonical = assert_semantics_preserved(&program);
        assert!(canonical.stats.constants_folded >= 3);
        // The folded scalar constant, the input reduction, and their addition.
        assert_eq!(canonical.program.finalize.len(), 3);
    }

    #[test]
    fn exp_of_negative_infinity_folds_to_zero_but_division_by_zero_stays() {
        // exp(-Inf) = 0 (finite): folds.
        let folds = ResearchProgram::expression(
            vec![],
            Section::new(vec![
                Op::Const(ScalarValue::F64(f64::NEG_INFINITY)),
                Op::Exp(Un::new(Ref::Local(0))),
            ]),
            vec![1],
        );
        let canonical = canonicalize(&folds, VerificationLimits::default()).unwrap();
        assert!(matches!(
            canonical.program.finalize.ops.last(),
            Some(Op::Const(ScalarValue::F64(zero))) if *zero == 0.0
        ));

        // 1/0 = +Inf: stays symbolic; execution reports NonFiniteOutput.
        let symbolic = ResearchProgram::expression(
            vec![],
            Section::new(vec![
                Op::Const(ScalarValue::F64(1.0)),
                Op::Const(ScalarValue::F64(0.0)),
                Op::Div(Bin::new(Ref::Local(0), Ref::Local(1))),
            ]),
            vec![2],
        );
        let canonical = canonicalize(&symbolic, VerificationLimits::default()).unwrap();
        assert!(matches!(
            canonical.program.finalize.ops.last(),
            Some(Op::Div(_))
        ));
    }

    #[test]
    fn common_subexpressions_merge_and_execution_is_preserved() {
        let program = ResearchProgram::expression(
            vec![f32_type(&[4])],
            Section::new(vec![
                Op::Tanh(Un::new(Ref::Input(0))), // 0
                Op::Tanh(Un::new(Ref::Input(0))), // 1 duplicate of 0
                Op::Abs(Un::new(Ref::Local(0))),  // 2
                Op::Abs(Un::new(Ref::Local(1))),  // 3 duplicate of 2 once 1 merges
                Op::Add(Bin::new(Ref::Local(2), Ref::Local(3))),
            ]),
            vec![4],
        );
        let canonical = assert_semantics_preserved(&program);
        assert!(canonical.stats.duplicates_removed >= 1);
        // After merging, only tanh, abs and the add remain.
        assert_eq!(canonical.program.finalize.len(), 3);
    }

    #[test]
    fn commutative_operands_normalize_to_identical_digests() {
        let build = |swap: bool| {
            ResearchProgram::expression(
                vec![f32_type(&[4]), f32_type(&[4])],
                Section::new(vec![
                    Op::Tanh(Un::new(Ref::Input(0))),
                    Op::Tanh(Un::new(Ref::Input(1))),
                    Op::Add(
                        if swap
                        {
                            Bin::new(Ref::Local(1), Ref::Local(0))
                        }
                        else
                        {
                            Bin::new(Ref::Local(0), Ref::Local(1))
                        },
                    ),
                ]),
                vec![2],
            )
        };
        let left = canonicalize(&build(false), VerificationLimits::default()).unwrap();
        let right = canonicalize(&build(true), VerificationLimits::default()).unwrap();
        assert!(super::super::canonical::canonical_equal(
            &left.program,
            &right.program
        ));
        assert_eq!(
            program_digest(&left.program),
            program_digest(&right.program)
        );
    }

    #[test]
    fn non_commutative_operands_keep_operand_order() {
        let program = ResearchProgram::expression(
            vec![f32_type(&[4])],
            Section::new(vec![
                Op::Const(ScalarValue::F32(2.0)),
                Op::Sub(Bin::new(Ref::Local(0), Ref::Input(0))),
            ]),
            vec![1],
        );
        let canonical = canonicalize(&program, VerificationLimits::default()).unwrap();
        match canonical.program.finalize.ops.last()
        {
            Some(Op::Sub(bin)) => assert_eq!(bin.lhs, Ref::Local(0)),
            other => panic!("expected surviving Sub, got {other:?}"),
        }
    }

    #[test]
    fn dce_keeps_every_output_dependency_in_multi_output_programs() {
        let program = ResearchProgram::expression(
            vec![f32_type(&[4])],
            Section::new(vec![
                Op::Tanh(Un::new(Ref::Input(0))), // shared by both outputs
                Op::Abs(Un::new(Ref::Local(0))),  // output A
                Op::Exp(Un::new(Ref::Local(0))),  // output B
                Op::Sin(Un::new(Ref::Input(0))),  // dead
            ]),
            vec![1, 2],
        );
        let canonical = assert_semantics_preserved(&program);
        assert_eq!(canonical.program.outputs, vec![1, 2]);
        // Shared dependency preserved exactly once; dead Sin removed.
        assert_eq!(canonical.program.finalize.len(), 3);
    }

    #[test]
    fn recurrence_programs_survive_canonicalization_with_state_intact() {
        let program = ResearchProgram {
            inputs: vec![],
            items: vec![ValueType::scalar(DType::F64)],
            state: vec![ValueType::scalar(DType::F64); 2],
            steps: 3,
            init: Section::new(vec![
                Op::Const(ScalarValue::F64(f64::NEG_INFINITY)),
                Op::Const(ScalarValue::F64(0.0)),
            ]),
            init_state: vec![0, 1],
            step: Section::new(vec![
                Op::Max(Bin::new(Ref::StatePrev(0), Ref::Item(0))),
                Op::Sub(Bin::new(Ref::StatePrev(0), Ref::Local(0))),
                Op::Exp(Un::new(Ref::Local(1))),
                Op::Sub(Bin::new(Ref::Item(0), Ref::Local(0))),
                Op::Exp(Un::new(Ref::Local(3))),
                Op::Mul(Bin::new(Ref::StatePrev(1), Ref::Local(2))),
                Op::Add(Bin::new(Ref::Local(5), Ref::Local(4))),
                // Dead branch that must be removed:
                Op::Cos(Un::new(Ref::Item(0))),
            ]),
            next_state: vec![0, 6],
            finalize: Section::new(vec![
                Op::Reshape(ShapeTo {
                    src: Ref::StateFinal(0),
                    shape: vec![],
                }),
                Op::Reshape(ShapeTo {
                    src: Ref::StateFinal(1),
                    shape: vec![],
                }),
                Op::Div(Bin::new(Ref::Local(1), Ref::Local(1))),
            ]),
            outputs: vec![0, 1, 2],
        };

        // Independent oracle on the original program.
        let items: Vec<ValueTensor> = [0.5f64, 2.0, -1.0]
            .iter()
            .map(|&value| ValueTensor::new(DType::F64, vec![], vec![value]).unwrap())
            .collect();

        let original = super::super::interpret::execute_program(
            &program,
            &[],
            &items,
            ExecutionPolicy::default(),
            VerificationLimits::default(),
        )
        .unwrap();

        let canonical = canonicalize(&program.clone(), VerificationLimits::default()).unwrap();
        assert_eq!(canonical.program.step.ops.len(), 7); // dead Cos removed

        let reduced = super::super::interpret::execute_program(
            &canonical.program,
            &[],
            &items,
            ExecutionPolicy::default(),
            VerificationLimits::default(),
        )
        .unwrap();

        assert_eq!(original.outputs, reduced.outputs);
        // Dead code never executes, even before canonicalization.
        assert_eq!(original.executed_nodes, reduced.executed_nodes);
    }

    #[test]
    fn canonicalization_is_idempotent_with_stable_digest() {
        let program = ResearchProgram::expression(
            vec![f32_type(&[4]), f32_type(&[4])],
            Section::new(vec![
                Op::Const(ScalarValue::F32(1.0)),
                Op::Add(Bin::new(Ref::Input(0), Ref::Local(0))),
                Op::Mul(Bin::new(Ref::Local(1), Ref::Input(1))),
                Op::Add(Bin::new(Ref::Input(1), Ref::Local(2))), // commutes with 2
                Op::Tanh(Un::new(Ref::Input(0))),
                Op::Tanh(Un::new(Ref::Input(0))),
            ]),
            vec![3],
        );
        let once = canonicalize(&program, VerificationLimits::default()).unwrap();
        let twice = canonicalize(&once.program, VerificationLimits::default()).unwrap();
        assert!(super::super::canonical::canonical_equal(
            &once.program,
            &twice.program
        ));
        assert_eq!(twice.stats.passes, 1); // second run converges immediately
        assert_eq!(
            program_digest(&once.program),
            program_digest(&twice.program)
        );
    }

    #[test]
    fn chained_identities_terminate_within_the_pass_budget() {
        // ((x + 0) + 0) + 0 ... requires several sweeps because the
        // eliminations chain through intermediate definitions.
        let depth = 12;
        let mut ops = vec![
            Op::Const(ScalarValue::F32(0.0)),
            Op::Add(Bin::new(Ref::Input(0), Ref::Local(0))),
        ];
        for _ in 1..depth
        {
            let prev = ops.len() - 1;
            ops.push(Op::Add(Bin::new(Ref::Local(prev), Ref::Local(0))));
        }
        let output = ops.len() - 1;
        let program =
            ResearchProgram::expression(vec![f32_type(&[4])], Section::new(ops), vec![output]);
        let canonical = canonicalize(&program, VerificationLimits::default()).unwrap();
        assert!(canonical.program.finalize.len() <= 2);
        assert!(canonical.stats.passes <= MAX_SIMPLIFY_PASSES);
        assert_semantics_preserved(&program);
    }
}
