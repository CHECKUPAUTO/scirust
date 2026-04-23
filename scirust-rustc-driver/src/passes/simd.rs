use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LocalDefId;

use super::MirPass;

pub struct SimdPass;

impl<'tcx> MirPass<'tcx> for SimdPass {
    fn name(&self) -> &'static str {
        "scirust_simd"
    }

    fn should_run(
        &self, _tcx: TyCtxt<'tcx>, _def_id: LocalDefId, _body: &Body<'tcx>
    ) -> bool {
        true
    }

    fn run(
        &mut self, _tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &Body<'tcx>
    ) {
        eprintln!("[simd] Analysing MIR for vectorisable loops in {:?}", def_id);

        let mut loop_headers = 0;
        let mut vectorisable_ops = 0;

        for (bb_idx, bb_data) in body.basic_blocks.iter_enumerated() {
            // Detect backward Goto edges = loop headers.
            if let Some(term) = &bb_data.terminator {
                match &term.kind {
                    TerminatorKind::Goto { target } => {
                        if *target <= bb_idx {
                            loop_headers += 1;
                            eprintln!(
                                "[simd]   Loop header detected at BB{:?} -> BB{:?}",
                                bb_idx, target
                            );
                        }
                    }
                    _ => {}
                }
            }

            // Count scalar arithmetic ops inside loop bodies (simplified: count all).
            for stmt in &bb_data.statements {
                if let StatementKind::Assign(assign) = &stmt.kind {
                    let (_, rvalue) = &**assign;
                    if let Rvalue::BinaryOp(op, _) = rvalue {
                        match op {
                            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                                vectorisable_ops += 1;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        eprintln!(
            "[simd]   Loop headers: {} | Scalar arithmetic ops: {}",
            loop_headers, vectorisable_ops
        );

        if loop_headers > 0 && vectorisable_ops > 0 {
            eprintln!(
                "[simd]   => Candidate for SIMD vectorisation (f64x4 / f32x8 / avx2 / neon)"
            );
        }
    }
}
