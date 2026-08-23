use super::{CompilerAnalysis, CompilerIr, IrValueId};
use crate::IrOperationId;

/// Live range of one SSA value in the current linear operation order.
///
/// `last_operation_use` excludes the program-output boundary. A value with
/// `live_out == true` must remain available after the final compiler operation
/// even when it has no later operation use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearLiveRange {
    definition: IrOperationId,
    last_operation_use: Option<IrOperationId>,
    live_out: bool,
}

impl LinearLiveRange {
    pub const fn definition(&self) -> IrOperationId {
        self.definition
    }

    pub const fn last_operation_use(&self) -> Option<IrOperationId> {
        self.last_operation_use
    }

    pub const fn is_live_out(&self) -> bool {
        self.live_out
    }
}

/// Linear liveness result indexed by [`IrValueId`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearLiveness {
    ranges: Vec<LinearLiveRange>,
}

impl LinearLiveness {
    pub fn get(&self, value: IrValueId) -> Option<&LinearLiveRange> {
        usize::try_from(value.get())
            .ok()
            .and_then(|index| self.ranges.get(index))
    }

    pub fn as_slice(&self) -> &[LinearLiveRange] {
        &self.ranges
    }
}

/// Liveness analysis for the compiler IR's current single-block linear form.
///
/// The analysis deliberately does not claim CFG semantics. When compiler IR
/// gains branches and block arguments, a separate control-flow-aware liveness
/// analysis can replace or coexist with this one without changing the meaning
/// of these linear ranges.
#[derive(Debug, Clone, Copy, Default)]
pub struct LinearLivenessAnalysis;

impl CompilerAnalysis for LinearLivenessAnalysis {
    type Output = LinearLiveness;

    fn name() -> &'static str {
        "linear-liveness"
    }

    fn run(ir: &CompilerIr) -> Self::Output {
        let mut ranges = ir
            .values()
            .iter()
            .map(|value| LinearLiveRange {
                definition: value.defining_operation(),
                last_operation_use: None,
                live_out: false,
            })
            .collect::<Vec<_>>();

        for operation in ir.operations()
        {
            for &operand in operation.operands()
            {
                if let Some(range) = usize::try_from(operand.get())
                    .ok()
                    .and_then(|index| ranges.get_mut(index))
                {
                    range.last_operation_use = Some(operation.id());
                }
            }
        }

        for &output in ir.outputs()
        {
            if let Some(range) = usize::try_from(output.get())
                .ok()
                .and_then(|index| ranges.get_mut(index))
            {
                range.live_out = true;
            }
        }

        LinearLiveness { ranges }
    }
}

#[cfg(test)]
mod tests {
    use scirust_tensor_ir::{DType, Graph, Operation, Shape, TensorType};

    use crate::{CanonicalCompiler, CompilerIr};

    use super::*;

    fn ty() -> TensorType {
        TensorType::new(DType::F32, Shape::new(vec![4]))
    }

    #[test]
    fn linear_liveness_tracks_last_operation_use_and_live_out() {
        let mut graph = Graph::new();
        let input = graph.add_input("input", ty()).unwrap();
        let relu = graph.add_node(Operation::Relu, vec![input], ty()).unwrap();
        let output = graph
            .add_node(Operation::Add, vec![relu, input], ty())
            .unwrap();
        graph.set_outputs(vec![output]).unwrap();

        let plan = CanonicalCompiler::new().compile(&graph).unwrap();
        let ir = CompilerIr::from_execution_plan(&plan).unwrap();
        let mut analyses = super::super::AnalysisManager::new(&ir);
        let liveness = analyses.get::<LinearLivenessAnalysis>().unwrap();

        let input_range = liveness.get(IrValueId::new(0)).unwrap();
        assert_eq!(input_range.definition(), IrOperationId::new(0));
        assert_eq!(
            input_range.last_operation_use(),
            Some(IrOperationId::new(2))
        );
        assert!(!input_range.is_live_out());

        let relu_range = liveness.get(IrValueId::new(1)).unwrap();
        assert_eq!(relu_range.definition(), IrOperationId::new(1));
        assert_eq!(
            relu_range.last_operation_use(),
            Some(IrOperationId::new(2))
        );
        assert!(!relu_range.is_live_out());

        let output_range = liveness.get(IrValueId::new(2)).unwrap();
        assert_eq!(output_range.definition(), IrOperationId::new(2));
        assert_eq!(output_range.last_operation_use(), None);
        assert!(output_range.is_live_out());
    }
}
