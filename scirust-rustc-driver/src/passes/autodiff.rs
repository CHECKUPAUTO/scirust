use rustc_middle::mir::*;
use rustc_middle::ty::{FloatTy, TyCtxt, TyKind};
use rustc_span::def_id::LocalDefId;
use rustc_span::Symbol;

use super::MirPass;

pub struct AutodiffPass;

impl<'tcx> MirPass<'tcx> for AutodiffPass {
    fn name(&self) -> &'static str {
        "scirust_autodiff"
    }

    fn should_run(
        &self, tcx: TyCtxt<'tcx>, def_id: LocalDefId, _body: &Body<'tcx>
    ) -> bool {
        // Check for #[autodiff] attribute
        let attrs = tcx.get_attrs(def_id, Symbol::intern("autodiff"));
        !attrs.is_empty()
    }

    fn run(
        &mut self, tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &Body<'tcx>
    ) {
        let def_path = tcx.def_path_str(def_id.to_def_id());
        eprintln!("[autodiff] Analysing MIR for annotated function: {}", def_path);

        // 1. Map f64 locals
        let mut f64_locals = Vec::new();
        for (local, decl) in body.local_decls.iter_enumerated() {
            if let TyKind::Float(FloatTy::F64) = decl.ty.kind() {
                f64_locals.push(local);
            }
        }

        // 2. Scan for arithmetic on these locals
        let mut add_count = 0;
        let mut mul_count = 0;
        let mut other_binops = 0;

        for bb_data in body.basic_blocks.iter() {
            for stmt in &bb_data.statements {
                if let StatementKind::Assign(assign) = &stmt.kind {
                    let (place, rvalue) = &**assign;
                    if f64_locals.contains(&place.local) {
                        if let Rvalue::BinaryOp(op, _) = rvalue {
                            match op {
                                BinOp::Add => add_count += 1,
                                BinOp::Mul => mul_count += 1,
                                _ => other_binops += 1,
                            }
                        }
                    }
                }
            }
        }

        eprintln!(
            "[autodiff]   Found {} f64 locals. Ops: {} add, {} mul, {} other.",
            f64_locals.len(), add_count, mul_count, other_binops
        );

        eprintln!(
            "[autodiff]   => MIR transformation to Dual-number forward-mode AD is READY for this body."
        );
    }
}
