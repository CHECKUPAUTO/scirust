use core::any::{Any, TypeId};
use core::fmt;

use super::super::ids::IrValueId;
use super::CompilerIr;

/// One stateless analysis over immutable [`CompilerIr`].
///
/// Analyses are keyed by their Rust type. Configured analyses with runtime
/// parameters are intentionally deferred until the compiler needs them.
pub trait CompilerAnalysis: 'static {
    type Output: 'static;

    fn name() -> &'static str;

    fn run(ir: &CompilerIr) -> Self::Output;
}

/// Typed failures from the heterogeneous analysis cache.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AnalysisManagerError {
    CacheTypeMismatch { analysis: &'static str },
}

impl fmt::Display for AnalysisManagerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::CacheTypeMismatch { analysis } =>
            {
                write!(formatter, "analysis cache type mismatch for {analysis}")
            },
        }
    }
}

impl std::error::Error for AnalysisManagerError {}

struct AnalysisCacheEntry {
    analysis: TypeId,
    output: Box<dyn Any>,
}

/// Deterministic, type-indexed cache bound to one immutable compiler IR.
///
/// Holding the IR borrow for the manager's entire lifetime prevents a cached
/// result from being reused accidentally with another or concurrently-mutated
/// program. A vector is used deliberately instead of a hash map so analysis
/// registration and lookup cannot introduce randomized iteration order.
pub struct AnalysisManager<'ir> {
    ir: &'ir CompilerIr,
    entries: Vec<AnalysisCacheEntry>,
}

impl<'ir> AnalysisManager<'ir> {
    pub fn new(ir: &'ir CompilerIr) -> Self {
        Self {
            ir,
            entries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get<A>(&mut self) -> Result<&A::Output, AnalysisManagerError>
    where
        A: CompilerAnalysis,
    {
        let analysis = TypeId::of::<A>();

        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.analysis == analysis)
        {
            return self.entries[index]
                .output
                .downcast_ref::<A::Output>()
                .ok_or(AnalysisManagerError::CacheTypeMismatch {
                    analysis: A::name(),
                });
        }

        self.entries.push(AnalysisCacheEntry {
            analysis,
            output: Box::new(A::run(self.ir)),
        });

        self.entries
            .last()
            .and_then(|entry| entry.output.downcast_ref::<A::Output>())
            .ok_or(AnalysisManagerError::CacheTypeMismatch {
                analysis: A::name(),
            })
    }

    pub fn invalidate<A>(&mut self)
    where
        A: CompilerAnalysis,
    {
        let analysis = TypeId::of::<A>();
        self.entries.retain(|entry| entry.analysis != analysis);
    }

    pub fn invalidate_all(&mut self) {
        self.entries.clear();
    }
}

/// Number of uses of each SSA value.
///
/// Operand occurrences are counted individually, including repeated operands.
/// Declared compiler outputs count as uses as well, which makes the result
/// suitable as a future seed for dead-value and liveness analyses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UseCounts {
    counts: Vec<usize>,
}

impl UseCounts {
    pub fn get(&self, value: IrValueId) -> Option<usize> {
        usize::try_from(value.get())
            .ok()
            .and_then(|index| self.counts.get(index))
            .copied()
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.counts
    }
}

/// Deterministic SSA use-count analysis.
#[derive(Debug, Clone, Copy, Default)]
pub struct UseCountAnalysis;

impl CompilerAnalysis for UseCountAnalysis {
    type Output = UseCounts;

    fn name() -> &'static str {
        "use-count"
    }

    fn run(ir: &CompilerIr) -> Self::Output {
        let mut counts = vec![0usize; ir.values().len()];

        for operation in ir.operations()
        {
            for &operand in operation.operands()
            {
                if let Some(count) = usize::try_from(operand.get())
                    .ok()
                    .and_then(|index| counts.get_mut(index))
                {
                    *count += 1;
                }
            }
        }

        for &output in ir.outputs()
        {
            if let Some(count) = usize::try_from(output.get())
                .ok()
                .and_then(|index| counts.get_mut(index))
            {
                *count += 1;
            }
        }

        UseCounts { counts }
    }
}

#[cfg(test)]
mod tests {
    use scirust_tensor_ir::{DType, Graph, Operation, Shape, TensorType};

    use crate::CanonicalCompiler;

    use super::*;

    fn ty() -> TensorType {
        TensorType::new(DType::F32, Shape::new(vec![4]))
    }

    fn repeated_operand_ir() -> CompilerIr {
        let mut graph = Graph::new();
        let input = graph.add_input("input", ty()).unwrap();
        let doubled = graph
            .add_node(Operation::Add, vec![input, input], ty())
            .unwrap();
        graph.set_outputs(vec![doubled]).unwrap();

        let plan = CanonicalCompiler::new().compile(&graph).unwrap();
        CompilerIr::from_execution_plan(&plan).unwrap()
    }

    #[test]
    fn use_count_analysis_counts_repeated_operands_and_outputs() {
        let ir = repeated_operand_ir();
        let mut analyses = AnalysisManager::new(&ir);

        let counts = analyses.get::<UseCountAnalysis>().unwrap();

        assert_eq!(counts.as_slice(), &[2, 1]);
        assert_eq!(counts.get(IrValueId::new(0)), Some(2));
        assert_eq!(counts.get(IrValueId::new(1)), Some(1));
    }

    #[test]
    fn analysis_manager_caches_and_invalidates_by_analysis_type() {
        let ir = repeated_operand_ir();
        let mut analyses = AnalysisManager::new(&ir);

        assert!(analyses.is_empty());
        assert_eq!(
            analyses.get::<UseCountAnalysis>().unwrap().as_slice(),
            &[2, 1]
        );
        assert_eq!(analyses.len(), 1);
        assert_eq!(
            analyses.get::<UseCountAnalysis>().unwrap().as_slice(),
            &[2, 1]
        );
        assert_eq!(analyses.len(), 1);

        analyses.invalidate::<UseCountAnalysis>();
        assert!(analyses.is_empty());

        assert_eq!(
            analyses.get::<UseCountAnalysis>().unwrap().as_slice(),
            &[2, 1]
        );
        analyses.invalidate_all();
        assert!(analyses.is_empty());
    }
}
