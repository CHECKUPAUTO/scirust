use rustc_middle::mir::*;
use rustc_middle::ty::{FloatTy, TyCtxt, TyKind};
use rustc_span::def_id::LocalDefId;

use super::MirPass;

pub struct AutodiffPass;

impl<'tcx> MirPass<'tcx> for AutodiffPass {
    fn name(&self) -> &'static str {
        "scirust_autodiff"
    }

    fn should_run(
        &self, _tcx: TyCtxt<'tcx>, _def_id: LocalDefId, _body: &Body<'tcx>
    ) -> bool {
        true
    }

    fn run(
        &mut self, _tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &Body<'tcx>
    ) {
        eprintln!("[autodiff] Analysing MIR for {:?}", def_id);

        // 1. Count f64 locals (primal variables we can differentiate).
        let mut f64_locals = 0;
        for (_local, decl) in body.local_decls.iter_enumerated() {
            if let TyKind::Float(FloatTy::F64) = decl.ty.kind() {
                f64_locals += 1;
            }
        }

        // 2. Scan basic blocks for supported binary ops.
        let mut total_binops = 0;
        let mut supported_binops = 0;
        let mut return_count = 0;

        for bb_data in body.basic_blocks.iter() {
            for stmt in &bb_data.statements {
                if let StatementKind::Assign(assign) = &stmt.kind {
                    let (_, rvalue) = &**assign;
                    if let Rvalue::BinaryOp(op, _) = rvalue {
                        total_binops += 1;
                        match op {
                            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                                supported_binops += 1;
                            }
                            _ => {}
                        }
                    }
                }
            }

            if let Some(term) = &bb_data.terminator {
                if let TerminatorKind::Return = term.kind {
                    return_count += 1;
                }
            }
        }

        eprintln!(
            "[autodiff]   f64 locals: {} | total BinaryOp: {} | supported (add/sub/mul/div): {} | returns: {}",
            f64_locals, total_binops, supported_binops, return_count
        );

        if f64_locals > 0 && supported_binops > 0 {
            eprintln!(
                "[autodiff]   => Candidate for _grad derivative extraction (forward-mode dual numbers)"
            );
        }
    }
}
