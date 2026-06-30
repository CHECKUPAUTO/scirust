use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LocalDefId;

use super::MirPass;

pub struct FusionPass;

impl<'tcx> MirPass<'tcx> for FusionPass {
    fn name(&self) -> &'static str {
        "scirust_fusion"
    }

    fn should_run(
        &self, _tcx: TyCtxt<'tcx>, _def_id: LocalDefId, _body: &Body<'tcx>
    ) -> bool {
        true
    }

    fn run(
        &mut self, tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &Body<'tcx>
    ) {
        eprintln!("[fusion] Analyzing MIR for fusion opportunities in {:?}", def_id);

        let mut opportunities = Vec::new();

        for (bb_idx, bb_data) in body.basic_blocks.iter_enumerated() {
            // Pattern 1: MatMul (Call) followed by Activation (Statement or Call)
            if let Some(term) = &bb_data.terminator {
                if let TerminatorKind::Call { func, destination, target, .. } = &term.kind {
                    let func_ty = func.ty(body, tcx);
                    let func_name = format!("{:?}", func_ty);

                    if func_name.contains("matmul") || func_name.contains("linear") {
                        if let Some(next_bb) = target {
                            if let Some(act) = self.find_activation_in_bb(tcx, body, *next_bb, *destination) {
                                opportunities.push((bb_idx, *next_bb, act));
                            }
                        }
                    }
                }
            }
        }

        for (matmul_bb, act_bb, act_type) in opportunities {
            eprintln!("[fusion] Found candidate: BB{:?} (MatMul) -> BB{:?} ({:?})", matmul_bb, act_bb, act_type);
            // Transformation strategy:
            // 1. Locate the MatMul call in matmul_bb.
            // 2. Identify the fused kernel corresponding to act_type (e.g., KernelType::MatmulRelu).
            // 3. Replace the original call with a call to the fused kernel.
            // 4. Remove the activation statement/call in act_bb.
            // 5. Update local variable usage to bypass the intermediate un-activated result.
        }
    }
}

#[derive(Debug)]
pub enum ActivationType {
    ReLU,
    SiLU,
    Sigmoid,
}

impl FusionPass {
    fn find_activation_in_bb<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        bb: BasicBlock,
        result_place: Place<'tcx>,
    ) -> Option<ActivationType> {
        let bb_data = &body.basic_blocks[bb];
        for stmt in &bb_data.statements {
            if let StatementKind::Assign(assign) = &stmt.kind {
                let (_, rvalue) = &**assign;

                // Check for ReLU pattern: max(0, x)
                // In MIR this might be a specific intrinsic or a branch.
                // Simplified: check for UnaryOp or custom calls.
                if let Rvalue::UnaryOp(_, operand) = rvalue {
                     if operand == &Operand::Copy(result_place) || operand == &Operand::Move(result_place) {
                         return Some(ActivationType::ReLU);
                     }
                }
            }
        }

        if let Some(term) = &bb_data.terminator {
            if let TerminatorKind::Call { func, args, .. } = &term.kind {
                let func_ty = func.ty(body, tcx);
                let func_name = format!("{:?}", func_ty);
                let is_consuming_result = args.iter().any(|arg| {
                    if let Operand::Copy(p) | Operand::Move(p) = &arg.node {
                        *p == result_place
                    } else {
                        false
                    }
                });

                if is_consuming_result {
                    if func_name.contains("relu") { return Some(ActivationType::ReLU); }
                    if func_name.contains("silu") { return Some(ActivationType::SiLU); }
                    if func_name.contains("sigmoid") { return Some(ActivationType::Sigmoid); }
                }
            }
        }
        None
    }
}
