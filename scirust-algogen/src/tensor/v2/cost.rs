//! Deterministic structural cost model and phase-resolved liveness.
//!
//! Every metric is a **pure function of program structure and statically
//! inferred types** — never wall-clock time — so costs are reproducible and
//! safe as search objectives. All metrics are *structural/logical*: they do
//! not claim hardware throughput. Integer accumulation saturates.
//!
//! Logical-FLOP weights (per produced/consumed element as noted):
//!
//! | class | weight |
//! |---|---|
//! | add/sub/neg/abs/min/max/clamp/select/logic/compare | 1 / element |
//! | mul | 1 / element |
//! | fma | 2 / element |
//! | div/pow | 4 / element |
//! | exp/exp2/expm1, log/log2/log1p | 10 / element |
//! | sqrt/rsqrt | 6 / element |
//! | sin/cos/tanh | 20 / element |
//! | sum/prod/max/min reduction | 1 per reduced element |
//! | mean reduction | 1 per reduced element + 1 per output element |
//! | dot/matvec/vecmat | 2 × inner length |
//! | matmul/batched-matmul | 2 × m·k·n (× batch) |
//! | outer | m·n |
//!
//! Shape ops move data (counted in reads/writes) but perform no arithmetic;
//! constants cost nothing.

use serde::{Deserialize, Serialize};

use super::ir::{Op, Ref, ResearchProgram};
use super::types::{ValueType, shape_elements};
use super::verify::{ProgramError, VerificationLimits, VerifiedProgram, verify_program};

/// Deterministic structural cost of a research program.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostReport {
    // ---- structure -----------------------------------------------------------
    /// Active nodes actually executed across all sections and scan steps.
    pub executed_nodes: u64,
    /// Active nodes counted once (step repetition excluded).
    pub active_nodes: u64,
    /// Dead nodes (defined but unreachable from outputs/state bindings).
    pub dead_nodes: usize,

    // ---- operator classes (active occurrences; scan steps included) ----------
    pub add_count: u64,
    pub mul_count: u64,
    pub fma_count: u64,
    pub div_count: u64,
    pub pow_count: u64,
    pub exp_count: u64,
    pub log_count: u64,
    pub sqrt_count: u64,
    pub trig_count: u64,
    pub minmax_count: u64,
    pub compare_count: u64,
    pub select_count: u64,
    pub bool_logic_count: u64,
    pub reduction_count: u64,
    pub linalg_count: u64,
    pub shape_op_count: u64,

    // ---- logical work ----------------------------------------------------------
    /// Weighted logical scalar operations over the whole execution.
    pub logical_flops: u64,
    /// Step-section contribution alone (per-step streaming cost).
    pub per_step_logical_flops: u64,
    /// Longest value-dependency chain inside the step section.
    pub update_depth: usize,
    /// Longest dependency chain inside the finalize section.
    pub finalize_depth: usize,

    // ---- memory ------------------------------------------------------------------
    /// Elements read by active operators over the whole execution.
    pub elements_read: u64,
    /// Elements written by active operators over the whole execution.
    pub elements_written: u64,
    /// Logical bytes read/written using declared dtype widths.
    pub logical_bytes_read: u64,
    pub logical_bytes_written: u64,
    /// Summed elements of active values that are neither observable outputs
    /// nor state bindings (each counted once).
    pub intermediate_elements: u64,
    /// Logical byte footprint of active non-output/non-state temporaries.
    pub intermediate_bytes: u64,
    /// Peak simultaneously-live elements during one scan step, including the
    /// resident state.
    pub peak_live_elements_in_step: u64,
    /// Peak live elements during finalize, including the final state.
    pub peak_live_elements_in_finalize: u64,
    /// Peak live value counts (state components included in scan/finalize).
    pub peak_live_values_in_step: u64,
    pub peak_live_values_in_finalize: u64,
    /// Peak live logical bytes by phase and across the whole program.
    pub peak_live_bytes_in_step: u64,
    pub peak_live_bytes_in_finalize: u64,
    pub peak_live_bytes: u64,
    /// Elements held by one instance of the recurrence state.
    pub state_elements: u64,
    pub state_bytes: u64,
    pub output_elements: u64,
    pub output_bytes: u64,
    /// Externally materialized per-step item sequence.
    pub stream_input_elements: u64,
    pub stream_input_bytes: u64,
    /// This V2 scan is a fold: it does not materialize per-step outputs.
    pub materialized_sequence_elements: u64,
    /// Static trip count of the scan.
    pub steps: u32,
}

impl CostReport {
    /// Worst-case placeholder for programs that cannot be verified.
    pub fn unevaluable(total_nodes: usize, steps: u32) -> Self {
        Self {
            executed_nodes: (total_nodes as u64).saturating_mul(u64::from(steps.max(1))),
            logical_flops: u64::MAX,
            dead_nodes: total_nodes,
            steps,
            ..Self::default()
        }
    }
}

/// Compute the structural cost, verifying first.
pub fn estimate_cost(
    program: &ResearchProgram,
    limits: VerificationLimits,
) -> Result<CostReport, ProgramError> {
    let verified = verify_program(program, limits)?;
    Ok(estimate_cost_verified(program, &verified))
}

/// Compute the structural cost from an existing verification result.
#[allow(clippy::too_many_lines)]
pub fn estimate_cost_verified(program: &ResearchProgram, verified: &VerifiedProgram) -> CostReport {
    let mut report = CostReport {
        steps: program.steps,
        state_elements: program.state_elements(),
        state_bytes: program
            .state
            .iter()
            .fold(0u64, |sum, value| sum.saturating_add(value.logical_bytes())),
        output_elements: verified
            .output_types
            .iter()
            .fold(0u64, |sum, value| sum.saturating_add(value.elements())),
        output_bytes: verified
            .output_types
            .iter()
            .fold(0u64, |sum, value| sum.saturating_add(value.logical_bytes())),
        stream_input_elements: verified.stream_input_elements as u64,
        stream_input_bytes: program
            .items
            .iter()
            .fold(0u64, |sum, value| sum.saturating_add(value.logical_bytes()))
            .saturating_mul(u64::from(program.steps)),
        ..CostReport::default()
    };

    let step_scale = u64::from(program.steps);

    let init = tally_section(
        &program.init.ops,
        &verified.init_active,
        &verified.init_types,
        program,
        1,
        &mut report,
    );
    let step = tally_section(
        &program.step.ops,
        &verified.step_active,
        &verified.step_types,
        program,
        step_scale,
        &mut report,
    );
    let finalize = tally_section(
        &program.finalize.ops,
        &verified.finalize_active,
        &verified.finalize_types,
        program,
        1,
        &mut report,
    );

    report.active_nodes = (init.active + step.active + finalize.active) as u64;
    report.executed_nodes =
        init.active as u64 + step.active as u64 * step_scale + finalize.active as u64;
    report.dead_nodes = program.node_count() - (init.active + step.active + finalize.active);
    report.per_step_logical_flops = step.flops;
    report.logical_flops = init
        .flops
        .saturating_add(step.flops.saturating_mul(step_scale))
        .saturating_add(finalize.flops);
    report.update_depth = dependency_depth(&program.step.ops, &verified.step_active);
    report.finalize_depth = dependency_depth(&program.finalize.ops, &verified.finalize_active);

    report.elements_read = init
        .reads
        .saturating_add(step.reads.saturating_mul(step_scale))
        .saturating_add(finalize.reads);
    report.elements_written = init
        .writes
        .saturating_add(step.writes.saturating_mul(step_scale))
        .saturating_add(finalize.writes);
    report.logical_bytes_read = init
        .read_bytes
        .saturating_add(step.read_bytes.saturating_mul(step_scale))
        .saturating_add(finalize.read_bytes);
    report.logical_bytes_written = init
        .write_bytes
        .saturating_add(step.write_bytes.saturating_mul(step_scale))
        .saturating_add(finalize.write_bytes);

    // Intermediates: active values that are neither observable outputs nor
    // next-state bindings, each counted once.
    let outputs: std::collections::HashSet<usize> = program.outputs.iter().copied().collect();
    let state_bindings: std::collections::HashSet<usize> =
        program.next_state.iter().copied().collect();
    let mut intermediates = 0u64;
    let mut intermediate_bytes = 0u64;
    for (node, value_type) in verified.init_types.iter().enumerate()
    {
        if verified.init_active[node]
        {
            intermediates = intermediates.saturating_add(elements(value_type));
            intermediate_bytes = intermediate_bytes.saturating_add(value_type.logical_bytes());
        }
    }
    for (node, value_type) in verified.step_types.iter().enumerate()
    {
        if verified.step_active[node] && !state_bindings.contains(&node)
        {
            intermediates = intermediates.saturating_add(elements(value_type));
            intermediate_bytes = intermediate_bytes.saturating_add(value_type.logical_bytes());
        }
    }
    for (node, value_type) in verified.finalize_types.iter().enumerate()
    {
        if verified.finalize_active[node] && !outputs.contains(&node)
        {
            intermediates = intermediates.saturating_add(elements(value_type));
            intermediate_bytes = intermediate_bytes.saturating_add(value_type.logical_bytes());
        }
    }
    report.intermediate_elements = intermediates;
    report.intermediate_bytes = intermediate_bytes;

    report.peak_live_elements_in_step = step.peak_live.saturating_add(report.state_elements);
    report.peak_live_elements_in_finalize =
        finalize.peak_live.saturating_add(report.state_elements);
    report.peak_live_values_in_step = step
        .peak_live_values
        .saturating_add(program.state.len() as u64);
    report.peak_live_values_in_finalize = finalize
        .peak_live_values
        .saturating_add(program.state.len() as u64);
    report.peak_live_bytes_in_step = step.peak_live_bytes.saturating_add(report.state_bytes);
    report.peak_live_bytes_in_finalize =
        finalize.peak_live_bytes.saturating_add(report.state_bytes);
    report.peak_live_bytes = init
        .peak_live_bytes
        .max(report.peak_live_bytes_in_step)
        .max(report.peak_live_bytes_in_finalize);

    report
}

// ---------------------------------------------------------------------------
// Section machinery
// ---------------------------------------------------------------------------

#[derive(Default)]
struct SectionTally {
    active: usize,
    flops: u64,
    reads: u64,
    writes: u64,
    read_bytes: u64,
    write_bytes: u64,
    peak_live: u64,
    peak_live_values: u64,
    peak_live_bytes: u64,
}

fn elements(value_type: &ValueType) -> u64 {
    shape_elements(&value_type.shape).unwrap_or(0) as u64
}

fn opt_elements(value_type: Option<&ValueType>) -> u64 {
    value_type.map(elements).unwrap_or(0)
}

/// Shape resolver for references inside one section.
fn ref_shape<'a>(
    reference: Ref,
    locals: &'a [ValueType],
    program: &'a ResearchProgram,
) -> Vec<usize> {
    let declared: &[ValueType] = match reference
    {
        Ref::Input(_) => &program.inputs,
        Ref::Item(_) => &program.items,
        Ref::StatePrev(_) | Ref::StateFinal(_) => &program.state,
        Ref::Local(_) => locals,
    };
    let index = match reference
    {
        Ref::Input(index) | Ref::Item(index) | Ref::StatePrev(index) | Ref::StateFinal(index) =>
        {
            Some(index)
        },
        Ref::Local(id) => locals.get(id).map(|_| id),
    };
    index
        .and_then(|index| declared.get(index))
        .map(|value_type| value_type.shape.clone())
        .unwrap_or_default()
}

fn ref_element_count(reference: Ref, locals: &[ValueType], program: &ResearchProgram) -> u64 {
    shape_elements(&ref_shape(reference, locals, program)).unwrap_or(0) as u64
}

/// Per-section accumulation. `scale` repeats effects (the scan trip count).
#[allow(clippy::too_many_lines)]
fn tally_section(
    ops: &[Op],
    active: &[bool],
    types: &[ValueType],
    program: &ResearchProgram,
    scale: u64,
    report: &mut CostReport,
) -> SectionTally {
    let mut tally = SectionTally::default();

    macro_rules! count_class {
        ($field:ident) => {{
            report.$field = report.$field.saturating_add(scale);
        }};
    }

    // Reads of one binary broadcast kernel = stored sizes of both operands.
    let binary_reads = |lhs: Ref, rhs: Ref| -> u64 {
        ref_element_count(lhs, types, program)
            .saturating_add(ref_element_count(rhs, types, program))
    };

    for (node, op) in ops.iter().enumerate()
    {
        if !active.get(node).copied().unwrap_or(false)
        {
            continue;
        }
        tally.active += 1;

        let out_elements = opt_elements(types.get(node));

        let (reads_here, flops_here): (u64, u64) = match op
        {
            Op::Const(_) => (0, 0),

            Op::Add(b) | Op::Sub(b) =>
            {
                count_class!(add_count);
                (binary_reads(b.lhs, b.rhs), out_elements)
            },
            Op::Mul(b) =>
            {
                count_class!(mul_count);
                (binary_reads(b.lhs, b.rhs), out_elements)
            },
            Op::MulAdd(t) =>
            {
                count_class!(fma_count);
                (
                    ref_element_count(t.a, types, program)
                        .saturating_add(ref_element_count(t.b, types, program))
                        .saturating_add(ref_element_count(t.c, types, program)),
                    out_elements.saturating_mul(2),
                )
            },
            Op::Div(b) =>
            {
                count_class!(div_count);
                (binary_reads(b.lhs, b.rhs), out_elements.saturating_mul(4))
            },
            Op::Pow(b) =>
            {
                count_class!(pow_count);
                (binary_reads(b.lhs, b.rhs), out_elements.saturating_mul(4))
            },

            Op::Neg(_) | Op::Abs(_) => (out_elements, out_elements),
            Op::Exp(_) | Op::Exp2(_) | Op::Expm1(_) =>
            {
                count_class!(exp_count);
                (out_elements, out_elements.saturating_mul(10))
            },
            Op::Log(_) | Op::Log2(_) | Op::Log1p(_) =>
            {
                count_class!(log_count);
                (out_elements, out_elements.saturating_mul(10))
            },
            Op::Sqrt(_) | Op::Rsqrt(_) =>
            {
                count_class!(sqrt_count);
                (out_elements, out_elements.saturating_mul(6))
            },
            Op::Sin(_) | Op::Cos(_) | Op::Tanh(_) =>
            {
                count_class!(trig_count);
                (out_elements, out_elements.saturating_mul(20))
            },

            Op::Min(b) | Op::Max(b) =>
            {
                count_class!(minmax_count);
                (binary_reads(b.lhs, b.rhs), out_elements)
            },
            Op::Clamp(t) =>
            {
                count_class!(minmax_count);
                (
                    ref_element_count(t.a, types, program)
                        .saturating_add(ref_element_count(t.b, types, program))
                        .saturating_add(ref_element_count(t.c, types, program)),
                    out_elements,
                )
            },
            Op::Eq(b) | Op::Ne(b) | Op::Lt(b) | Op::Le(b) | Op::Gt(b) | Op::Ge(b) =>
            {
                count_class!(compare_count);
                (binary_reads(b.lhs, b.rhs), out_elements)
            },
            Op::Select(t) =>
            {
                count_class!(select_count);
                (
                    ref_element_count(t.a, types, program)
                        .saturating_add(out_elements.saturating_mul(2)),
                    out_elements,
                )
            },
            Op::And(b) | Op::Or(b) =>
            {
                count_class!(bool_logic_count);
                (binary_reads(b.lhs, b.rhs), out_elements)
            },
            Op::Not(_) =>
            {
                count_class!(bool_logic_count);
                (out_elements, out_elements)
            },

            Op::ReduceSum(r)
            | Op::ReduceProd(r)
            | Op::ReduceMin(r)
            | Op::ReduceMax(r)
            | Op::ReduceMean(r) =>
            {
                count_class!(reduction_count);
                let source_shape = ref_shape(r.src, types, program);
                let source_elements = shape_elements(&source_shape).unwrap_or(0) as u64;
                let divisions = matches!(op, Op::ReduceMean(_))
                    .then_some(out_elements)
                    .unwrap_or(0);
                (source_elements, source_elements.saturating_add(divisions))
            },

            Op::Dot(b) =>
            {
                count_class!(linalg_count);
                let n = ref_element_count(b.lhs, types, program);
                (n.saturating_mul(2), n.saturating_mul(2))
            },
            Op::MatVec(b) =>
            {
                count_class!(linalg_count);
                // [m, k] x [k]: 2 * m * k = 2 * matrix elements.
                let matrix = ref_element_count(b.lhs, types, program);
                (
                    matrix.saturating_add(ref_element_count(b.rhs, types, program)),
                    matrix.saturating_mul(2),
                )
            },
            Op::VecMat(b) =>
            {
                count_class!(linalg_count);
                // [k] x [k, n]: 2 * k * n = 2 * matrix elements.
                let matrix = ref_element_count(b.rhs, types, program);
                (
                    matrix.saturating_add(ref_element_count(b.lhs, types, program)),
                    matrix.saturating_mul(2),
                )
            },
            Op::MatMul(b) =>
            {
                count_class!(linalg_count);
                matmul_work(b.lhs, b.rhs, types, program, 1)
            },
            Op::BatchedMatMul(b) =>
            {
                count_class!(linalg_count);
                let lhs_shape = ref_shape(b.lhs, types, program);
                let batch = lhs_shape.first().copied().unwrap_or(1) as u64;
                matmul_work(b.lhs, b.rhs, types, program, batch)
            },
            Op::Outer(b) =>
            {
                count_class!(linalg_count);
                let work = out_elements;
                (
                    ref_element_count(b.lhs, types, program)
                        .saturating_add(ref_element_count(b.rhs, types, program)),
                    work,
                )
            },

            Op::Reshape(_)
            | Op::Squeeze(_)
            | Op::Unsqueeze(_)
            | Op::Transpose(_)
            | Op::BroadcastTo(_)
            | Op::Concat { .. }
            | Op::Narrow(_) =>
            {
                count_class!(shape_op_count);
                (out_elements, 0)
            },
        };

        // Section tallies are per execution. The caller applies scan scaling
        // exactly once; class occurrence counters above intentionally include
        // `scale` directly.
        tally.flops = tally.flops.saturating_add(flops_here);
        tally.reads = tally.reads.saturating_add(reads_here);
        tally.writes = tally.writes.saturating_add(out_elements);
        let out_bytes = types.get(node).map(ValueType::logical_bytes).unwrap_or(0);
        tally.write_bytes = tally.write_bytes.saturating_add(out_bytes);
        // A conservative dtype-aware read estimate: scale logical element
        // reads by the widest operand dtype participating in the op.
        let mut widest = 1u64;
        op.for_each_ref(|reference| {
            widest = widest
                .max(ref_type(reference, types, program).map_or(1, |v| v.dtype.logical_bytes()));
        });
        tally.read_bytes = tally
            .read_bytes
            .saturating_add(reads_here.saturating_mul(widest));
    }

    // ---- peak-live sweep -------------------------------------------------------
    //
    // A register becomes live at its definition and dies after its last active
    // use; roots (no active consumer) stay live until the section ends. The
    // recurrence state itself is added by the caller.
    let end = (0..ops.len())
        .rfind(|&node| active.get(node).copied().unwrap_or(false))
        .unwrap_or(0);
    let last_use = section_last_use(ops, active, end);

    let mut live = 0u64;
    let mut live_values = 0u64;
    let mut live_bytes = 0u64;
    for node in 0..ops.len()
    {
        if !active.get(node).copied().unwrap_or(false)
        {
            continue;
        }
        live = live.saturating_add(opt_elements(types.get(node)));
        live_values = live_values.saturating_add(1);
        live_bytes =
            live_bytes.saturating_add(types.get(node).map(ValueType::logical_bytes).unwrap_or(0));
        tally.peak_live = tally.peak_live.max(live);
        tally.peak_live_values = tally.peak_live_values.max(live_values);
        tally.peak_live_bytes = tally.peak_live_bytes.max(live_bytes);
        if node == end
        {
            break;
        }

        for (candidate, &last) in last_use.iter().enumerate()
        {
            if last == node
            {
                live = live.saturating_sub(opt_elements(types.get(candidate)));
                live_values = live_values.saturating_sub(1);
                live_bytes = live_bytes.saturating_sub(
                    types
                        .get(candidate)
                        .map(ValueType::logical_bytes)
                        .unwrap_or(0),
                );
            }
        }
    }

    tally
}

fn ref_type<'a>(
    reference: Ref,
    locals: &'a [ValueType],
    program: &'a ResearchProgram,
) -> Option<&'a ValueType> {
    match reference
    {
        Ref::Input(index) => program.inputs.get(index),
        Ref::Item(index) => program.items.get(index),
        Ref::StatePrev(index) | Ref::StateFinal(index) => program.state.get(index),
        Ref::Local(index) => locals.get(index),
    }
}

/// Last active consumer per definition; roots map to the section end.
fn section_last_use(ops: &[Op], active: &[bool], end: usize) -> Vec<usize> {
    let mut map = vec![end; ops.len()];
    for (node, op) in ops.iter().enumerate()
    {
        if !active.get(node).copied().unwrap_or(false)
        {
            continue;
        }
        op.for_each_ref(|reference| {
            if let Ref::Local(source) = reference
            {
                if source < node && active[source]
                {
                    map[source] = node;
                }
            }
        });
    }
    // Definitions never consumed stay at `end` only if they are roots; dead
    // definitions must not participate (they are skipped by callers anyway).
    for (node, last) in map.iter_mut().enumerate()
    {
        if !active.get(node).copied().unwrap_or(false)
        {
            *last = usize::MAX;
        }
    }
    map
}

/// Longest active dependency chain inside a section.
fn dependency_depth(ops: &[Op], active: &[bool]) -> usize {
    let mut depth = vec![0usize; ops.len()];
    let mut best = 0usize;
    for (node, op) in ops.iter().enumerate()
    {
        if !active.get(node).copied().unwrap_or(false)
        {
            continue;
        }
        let mut longest_predecessor = 0usize;
        op.for_each_ref(|reference| {
            if let Ref::Local(source) = reference
            {
                longest_predecessor = longest_predecessor.max(depth[source]);
            }
        });
        depth[node] = longest_predecessor + 1;
        best = best.max(depth[node]);
    }
    best
}

/// `2 · m · k · n` from declared operand shapes (× batch), plus read volume.
fn matmul_work(
    lhs: Ref,
    rhs: Ref,
    types: &[ValueType],
    program: &ResearchProgram,
    batch: u64,
) -> (u64, u64) {
    let lhs_shape = ref_shape(lhs, types, program);
    let rhs_shape = ref_shape(rhs, types, program);
    let reads = shape_elements(&lhs_shape)
        .unwrap_or(0)
        .saturating_add(shape_elements(&rhs_shape).unwrap_or(0)) as u64;

    // [b,] m, k times [b,] k, n.
    let from_end = |shape: &[usize], offset: usize| -> u64 {
        shape
            .len()
            .checked_sub(offset)
            .and_then(|index| shape.get(index))
            .copied()
            .unwrap_or(1) as u64
    };
    let m = from_end(&lhs_shape, 2);
    let k = from_end(&lhs_shape, 1);
    let n = from_end(&rhs_shape, 1);

    let work = batch
        .saturating_mul(2)
        .saturating_mul(m)
        .saturating_mul(k)
        .saturating_mul(n);
    (reads, work)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::{matrix_multiplication_program, online_softmax_recurrence};

    #[test]
    fn recurrence_cost_scales_step_work_exactly_once() {
        let program = online_softmax_recurrence(3);
        let report = estimate_cost(&program, VerificationLimits::default()).unwrap();
        assert_eq!(report.per_step_logical_flops, 25);
        assert_eq!(report.logical_flops, 75);
        assert_eq!(report.executed_nodes, 25);
        assert_eq!(report.exp_count, 6);
        assert_eq!(report.state_elements, 2);
        assert_eq!(report.state_bytes, 16);
        assert_eq!(report.output_bytes, 16);
        assert_eq!(report.stream_input_elements, 3);
        assert_eq!(report.stream_input_bytes, 24);
        assert_eq!(report.materialized_sequence_elements, 0);
        assert!(report.peak_live_bytes_in_step >= report.state_bytes);
    }

    #[test]
    fn matrix_multiplication_flops_use_m_k_n_dimensions() {
        let program = matrix_multiplication_program(2, 3, 4);
        let report = estimate_cost(&program, VerificationLimits::default()).unwrap();
        assert_eq!(report.logical_flops, 48);
        assert_eq!(report.linalg_count, 1);
        assert_eq!(report.output_elements, 8);
        assert_eq!(report.output_bytes, 64);
    }

    #[test]
    fn cost_is_a_bit_identical_pure_structural_analysis() {
        let program = online_softmax_recurrence(7);
        assert_eq!(
            estimate_cost(&program, VerificationLimits::default()).unwrap(),
            estimate_cost(&program, VerificationLimits::default()).unwrap()
        );
    }
}
