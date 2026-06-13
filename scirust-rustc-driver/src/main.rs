#![feature(rustc_private)]
#![allow(unused_extern_crates)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

use rustc_driver::Callbacks;
use rustc_interface::interface;
use rustc_middle::ty::TyCtxt;

mod passes;
use passes::{AutodiffPass, GpuPass, SciRustPassManager, SimdPass};

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

struct SciRustCallbacks;

impl Callbacks for SciRustCallbacks {
    fn after_analysis<'tcx>(
        &mut self,
        _compiler: &interface::Compiler,
        tcx: TyCtxt<'tcx>,
    ) -> rustc_driver::Compilation {
        eprintln!("[scirustc] === SciRust MIR Transformation Phase ===");

        let mut pass_mgr = SciRustPassManager::new(tcx);
        pass_mgr.register(Box::new(AutodiffPass));
        pass_mgr.register(Box::new(SimdPass));
        pass_mgr.register(Box::new(GpuPass));

        for &def_id in tcx.mir_keys(()) {
            let body = tcx.optimized_mir(def_id);
            pass_mgr.analyze_mir(def_id, body);
        }

        eprintln!("[scirustc] === MIR transformations complete ===");
        rustc_driver::Compilation::Continue
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    eprintln!("[scirustc] SciRust compiler driver v0.3.0");
    eprintln!("[scirustc] Invoking rustc with SciRust MIR passes enabled...");
    rustc_driver::run_compiler(&args, &mut SciRustCallbacks);
}
