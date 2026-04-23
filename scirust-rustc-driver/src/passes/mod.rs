pub mod autodiff;
pub mod simd;
pub mod gpu;

pub use autodiff::AutodiffPass;
pub use simd::SimdPass;
pub use gpu::GpuPass;

use rustc_middle::mir::Body;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LocalDefId;

pub trait MirPass<'tcx> {
    fn name(&self) -> &'static str;
    fn should_run(&self, tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &Body<'tcx>) -> bool;
    fn run(&mut self, tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &Body<'tcx>);
}

pub struct SciRustPassManager<'tcx> {
    tcx: TyCtxt<'tcx>,
    passes: Vec<Box<dyn MirPass<'tcx>>>,
}

impl<'tcx> SciRustPassManager<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx, passes: Vec::new() }
    }

    pub fn register(&mut self, pass: Box<dyn MirPass<'tcx>>) {
        eprintln!("[scirustc] Registered pass: {}", pass.name());
        self.passes.push(pass);
    }

    /// Run all registered passes on a MIR body.
    pub fn analyze_mir(
        &mut self, def_id: LocalDefId, body: &Body<'tcx>
    ) {
        for pass in &mut self.passes {
            if pass.should_run(self.tcx, def_id, body) {
                eprintln!("[scirustc] Running {} on {:?}", pass.name(), def_id);
                pass.run(self.tcx, def_id, body);
            }
        }
    }
}
