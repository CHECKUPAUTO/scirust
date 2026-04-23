use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LocalDefId;

use super::MirPass;

pub struct GpuPass;

impl<'tcx> MirPass<'tcx> for GpuPass {
    fn name(&self) -> &'static str {
        "scirust_gpu"
    }

    fn should_run(
        &self, _tcx: TyCtxt<'tcx>, _def_id: LocalDefId, _body: &Body<'tcx>
    ) -> bool {
        true
    }

    fn run(
        &mut self, _tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &Body<'tcx>
    ) {
        eprintln!("[gpu] Extracting GPU kernel candidates from {:?}", def_id);

        let mut slice_index_ops = 0;
        let mut loop_headers = 0;

        for (bb_idx, bb_data) in body.basic_blocks.iter_enumerated() {
            // Detect loops (same logic as SIMD: backward Goto edges).
            if let Some(term) = &bb_data.terminator {
                if let TerminatorKind::Goto { target } = &term.kind {
                    if *target <= bb_idx {
                        loop_headers += 1;
                    }
                }
            }

            // Look for slice element accesses via Index projection.
            for stmt in &bb_data.statements {
                if let StatementKind::Assign(assign) = &stmt.kind {
                    let (place, _rvalue) = &**assign;
                    for elem in place.projection.iter() {
                        if let PlaceElem::Index(_) = elem {
                            slice_index_ops += 1;
                        }
                    }
                }
            }
        }

        eprintln!(
            "[gpu]   Loops: {} | Slice index ops: {}",
            loop_headers, slice_index_ops
        );

        if loop_headers > 0 && slice_index_ops > 0 {
            eprintln!(
                "[gpu]   => Candidate for GPU offload (element-wise slice mutation -> CUDA/PTX/SPIR-V)"
            );
        }
    }
}
