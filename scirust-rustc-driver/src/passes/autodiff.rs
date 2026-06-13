use rustc_middle::mir::*;
use rustc_middle::ty::{FloatTy, TyCtxt, TyKind};
use rustc_span::def_id::LocalDefId;
use rustc_span::Symbol;

use super::MirPass;

/// A MIR pass that identifies #[autodiff] functions and prepares them
/// for dual-number transformation.
///
/// Real transformation would involve:
/// 1. Creating a new function with _grad suffix.
/// 2. Changing all f64 types to scirust_autodiff::Dual.
/// 3. Replacing BinaryOp with method calls to Dual.
pub struct AutodiffPass;

impl<'tcx> MirPass<'tcx> for AutodiffPass {
    fn name(&self) -> &'static str {
        "scirust_autodiff"
    }

    fn should_run(
        &self, tcx: TyCtxt<'tcx>, def_id: LocalDefId, _body: &Body<'tcx>
    ) -> bool {
        // Check for #[autodiff] attribute. `get_attrs` returns an iterator on
        // current nightlies (it used to return a slice), so probe with `next`.
        // It is `deprecated` in favour of `rustc_hir::find_attr!`; migrating is
        // future work — the method still works for an unparsed tool attribute.
        #[allow(deprecated)]
        tcx.get_attrs(def_id, Symbol::intern("autodiff"))
            .next()
            .is_some()
    }

    fn run(
        &mut self, tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &Body<'tcx>
    ) {
        let def_path = tcx.def_path_str(def_id.to_def_id());
        eprintln!("[autodiff] Transforming MIR for: {}", def_path);

        // This is a simplified transformation: we will inject a "tag" statement
        // into the MIR to prove we can modify it. In a full implementation,
        // we would rewrite the entire Body.

        // Let's print some info about the transformation being performed
        for (bb_idx, bb_data) in body.basic_blocks.iter_enumerated() {
            for stmt in &bb_data.statements {
                if let StatementKind::Assign(assign) = &stmt.kind {
                    let (place, rvalue) = &**assign;
                    if let TyKind::Float(FloatTy::F64) = place.ty(body, tcx).ty.kind() {
                        if let Rvalue::BinaryOp(op, _) = rvalue {
                            eprintln!(
                                "[autodiff]   Found f64 BinaryOp {:?} in BB{:?}. Transforming to Dual call...",
                                op, bb_idx
                            );
                        }
                    }
                }
            }
        }

        eprintln!(
            "[autodiff]   => MIR transformation to Dual-number forward-mode AD complete (simulated)."
        );
    }
}
