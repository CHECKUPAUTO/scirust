//! Dataset Builder for SciRust Ownership Model (SOM).
//! Converts PCG (Place Capability Graph) to labeled samples for ML.

use scirust_som_pcg::{Pcg, PcgNode, PcgEdge};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomSample {
    pub graph: Pcg,
    pub labels: SomLabels,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SomLabels {
    pub ownership: Vec<OwnershipLabel>,
    pub borrow_type: Vec<BorrowLabel>,
    pub lifetime_group: Vec<usize>,
    pub aliasing: Vec<bool>,
    pub escape: Vec<bool>,
    pub mutability: Vec<bool>,
    pub unsafe_probability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OwnershipLabel {
    Owned,
    Borrowed,
    Moved,
    Dropped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BorrowLabel {
    Shared,
    Mutable,
    None,
}

pub struct DatasetBuilder;

impl Default for DatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DatasetBuilder {
    pub fn new() -> Self {
        Self
    }

    pub fn generate_sample(&self, pcg: Pcg) -> SomSample {
        let mut labels = SomLabels::default();

        // Basic labeling logic based on PCG edges
        for (i, node) in pcg.nodes.iter().enumerate() {
            if let PcgNode::Variable(_) = node {
                // Determine ownership label
                let mut label = OwnershipLabel::Owned;
                if pcg.edges.iter().any(|(f, _, e)| *f == i && *e == PcgEdge::Moves) {
                    label = OwnershipLabel::Moved;
                } else if pcg.edges.iter().any(|(_, t, e)| *t == i && *e == PcgEdge::Drops) {
                    label = OwnershipLabel::Dropped;
                } else if pcg.edges.iter().any(|(_, t, e)| *t == i && (*e == PcgEdge::Borrows || *e == PcgEdge::MutBorrows)) {
                    label = OwnershipLabel::Borrowed;
                }
                labels.ownership.push(label);

                // Determine borrow label
                let mut b_label = BorrowLabel::None;
                if pcg.edges.iter().any(|(_, t, e)| *t == i && *e == PcgEdge::MutBorrows) {
                    b_label = BorrowLabel::Mutable;
                } else if pcg.edges.iter().any(|(_, t, e)| *t == i && *e == PcgEdge::Borrows) {
                    b_label = BorrowLabel::Shared;
                }
                labels.borrow_type.push(b_label);

                // Placeholder for other labels
                labels.lifetime_group.push(0);
                labels.aliasing.push(false);
                labels.escape.push(false);
                labels.mutability.push(true);
            }
        }

        SomSample { graph: pcg, labels }
    }

    pub fn export_jsonl(&self, samples: &[SomSample]) -> String {
        samples.iter()
            .map(|s| serde_json::to_string(s).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_som_pcg::ast::*;
    use scirust_som_pcg::PcgBuilder;

    #[test]
    fn test_sample_generation() {
        let prog = SomAst::Program(vec![Function {
            name: "test".to_string(),
            params: vec![],
            body: vec![
                Statement::VarDecl {
                    name: "x".to_string(),
                    ty: Type::Int,
                    init: Some(Expression::Literal(Literal::Int(1))),
                },
                Statement::VarDecl {
                    name: "y".to_string(),
                    ty: Type::Int,
                    init: Some(Expression::Variable("x".to_string())),
                },
            ],
        }]);

        let mut pcg_builder = PcgBuilder::new();
        pcg_builder.build(&prog);
        let pcg = pcg_builder.pcg;

        let db = DatasetBuilder::new();
        let sample = db.generate_sample(pcg);

        assert_eq!(sample.labels.ownership.len(), 2);
        // x is moved to y
        assert!(matches!(sample.labels.ownership[0], OwnershipLabel::Moved));
        // y is owned
        assert!(matches!(sample.labels.ownership[1], OwnershipLabel::Owned));
    }
}
