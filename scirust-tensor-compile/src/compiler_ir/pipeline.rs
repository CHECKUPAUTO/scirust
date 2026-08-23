use crate::MemoryPlan;

use super::{CompilerIr, CompilerIrError, PassManager, PassManagerStats};

impl CompilerIr {
    /// Run the ordered compiler-pass pipeline, then derive logical memory from
    /// the transformed SSA program.
    ///
    /// This makes the intended compiler ordering explicit: rewrites and other
    /// compiler passes complete before liveness drives buffer reuse. The legacy
    /// memory plan embedded in [`crate::ExecutionPlan`] is not consulted.
    pub fn run_passes_and_plan_memory(
        &mut self,
        passes: &mut PassManager,
    ) -> Result<(PassManagerStats, MemoryPlan), CompilerIrError> {
        let stats = passes.run(self)?;
        let memory = MemoryPlan::from_compiler_ir(self)?;
        Ok((stats, memory))
    }
}

#[cfg(test)]
mod tests {
    use scirust_tensor_ir::{DType, Graph, Operation, Scalar, Shape, TensorType};

    use crate::{
        CanonicalCompiler, CompilerIr, PassManager, PassManagerStats, ScaleZeroCanonicalizationPass,
    };

    fn ty() -> TensorType {
        TensorType::new(DType::F32, Shape::new(vec![4]))
    }

    #[test]
    fn passes_run_before_compiler_ir_memory_planning() {
        let mut graph = Graph::new();
        let input = graph.add_input("input", ty()).unwrap();
        let zero = graph
            .add_node(
                Operation::Scale {
                    factor: Scalar::f32(0.0),
                },
                vec![input],
                ty(),
            )
            .unwrap();
        let output = graph.add_node(Operation::Relu, vec![zero], ty()).unwrap();
        graph.set_outputs(vec![output]).unwrap();

        let execution = CanonicalCompiler::new().compile(&graph).unwrap();
        let mut ir = CompilerIr::from_execution_plan(&execution).unwrap();
        let mut passes = PassManager::new();
        passes.push(ScaleZeroCanonicalizationPass);

        let (stats, memory) = ir.run_passes_and_plan_memory(&mut passes).unwrap();

        assert_eq!(
            stats,
            PassManagerStats {
                passes_run: 1,
                passes_changed: 1,
            }
        );
        assert!(matches!(
            ir.operations()[1].operation(),
            Operation::ZerosLike
        ));
        assert_eq!(&memory, execution.memory_plan());
    }
}
