//! Dataset Builder for SciRust Ownership Model (SOM).
//! Converts PCG (Place Capability Graph) to labeled samples for ML.
//!
//! The [`generate`] module produces seeded random toy programs and turns
//! them into per-token training samples labelled by the deterministic
//! ownership oracle of `scirust-som-symbolic`.

pub mod batch;
pub mod generate;

pub use batch::{Batch, PaddedSample, SplitDataset, batch_samples, pad_sample, train_val_split};
pub use generate::{GeneratorConfig, ProgramGenerator, TrainingSample, build_training_set};

use scirust_som_pcg::{Pcg, PcgEdge, PcgNode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomSample {
    pub graph: Pcg,
    pub labels: SomLabels,
}

/// Per-variable structural labels read directly off the PCG.
///
/// Every vector is indexed by the order in which `PcgNode::Variable` nodes
/// appear in [`Pcg::nodes`], so position `k` describes the `k`-th variable
/// node. Each field is a deterministic structural query over the graph — none
/// are placeholders:
///
/// - `ownership` / `borrow_type`: derived from the move/borrow/drop edges.
/// - `lifetime_group`: the node index of the region that `Owns` the variable;
///   variables sharing a scope share a group id.
/// - `aliasing`: the variable has two or more incoming borrows (or an explicit
///   `Aliases` edge), i.e. it is reachable through more than one reference.
/// - `escape`: the variable flows into the function's `return` place, by move
///   (`var -> return`) or by borrow (`return -> var`).
/// - `mutability`: the variable is mutably borrowed (`&mut` taken of it).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SomLabels {
    pub ownership: Vec<OwnershipLabel>,
    pub borrow_type: Vec<BorrowLabel>,
    pub lifetime_group: Vec<usize>,
    pub aliasing: Vec<bool>,
    pub escape: Vec<bool>,
    pub mutability: Vec<bool>,
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
        for (i, node) in pcg.nodes.iter().enumerate()
        {
            if let PcgNode::Variable(_) = node
            {
                // Determine ownership label
                let mut label = OwnershipLabel::Owned;
                if pcg
                    .edges
                    .iter()
                    .any(|(f, _, e)| *f == i && *e == PcgEdge::Moves)
                {
                    label = OwnershipLabel::Moved;
                }
                else if pcg
                    .edges
                    .iter()
                    .any(|(_, t, e)| *t == i && *e == PcgEdge::Drops)
                {
                    label = OwnershipLabel::Dropped;
                }
                else if pcg.edges.iter().any(|(_, t, e)| {
                    *t == i && (*e == PcgEdge::Borrows || *e == PcgEdge::MutBorrows)
                })
                {
                    label = OwnershipLabel::Borrowed;
                }
                labels.ownership.push(label);

                // Determine borrow label
                let mut b_label = BorrowLabel::None;
                if pcg
                    .edges
                    .iter()
                    .any(|(_, t, e)| *t == i && *e == PcgEdge::MutBorrows)
                {
                    b_label = BorrowLabel::Mutable;
                }
                else if pcg
                    .edges
                    .iter()
                    .any(|(_, t, e)| *t == i && *e == PcgEdge::Borrows)
                {
                    b_label = BorrowLabel::Shared;
                }
                labels.borrow_type.push(b_label);

                // Lifetime group: the region node that owns this variable.
                // Variables in the same scope share their owner region, hence
                // the same group id. Falls back to the variable's own index
                // when no owner edge exists (should not happen for a binding).
                let lifetime_group = pcg
                    .edges
                    .iter()
                    .find(|(_, t, e)| *t == i && *e == PcgEdge::Owns)
                    .map(|(f, _, _)| *f)
                    .unwrap_or(i);
                labels.lifetime_group.push(lifetime_group);

                // Aliasing: reachable through more than one reference, i.e.
                // two or more borrows target it, or an explicit alias edge.
                let incoming_borrows = pcg
                    .edges
                    .iter()
                    .filter(|(_, t, e)| {
                        *t == i && (*e == PcgEdge::Borrows || *e == PcgEdge::MutBorrows)
                    })
                    .count();
                let aliased = pcg
                    .edges
                    .iter()
                    .any(|(f, t, e)| (*f == i || *t == i) && *e == PcgEdge::Aliases);
                labels.aliasing.push(incoming_borrows >= 2 || aliased);

                // Escape: the variable flows into the function's return place,
                // by move (`var -> return`) or by borrow (`return -> var`).
                let escape = pcg.edges.iter().any(|(f, t, e)| {
                    (*e == PcgEdge::Moves && *f == i && Self::is_return_place(&pcg.nodes[*t]))
                        || ((*e == PcgEdge::Borrows || *e == PcgEdge::MutBorrows)
                            && *t == i
                            && Self::is_return_place(&pcg.nodes[*f]))
                });
                labels.escape.push(escape);

                // Mutability: a `&mut` is taken of this variable.
                let mutability = pcg
                    .edges
                    .iter()
                    .any(|(_, t, e)| *t == i && *e == PcgEdge::MutBorrows);
                labels.mutability.push(mutability);
            }
        }

        SomSample { graph: pcg, labels }
    }

    /// True for the synthetic place a `return <expr>` flows into.
    fn is_return_place(node: &PcgNode) -> bool {
        matches!(node, PcgNode::MemoryLocation(name) if name == "return")
    }

    pub fn export_jsonl(&self, samples: &[SomSample]) -> String {
        samples
            .iter()
            .map(|s| serde_json::to_string(s).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_som_pcg::PcgBuilder;
    use scirust_som_pcg::ast::*;

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

        // No borrows, no return, no nested scope: the structural fields are
        // all the neutral value, and both vars share the one region (group 1).
        assert!(matches!(sample.labels.borrow_type[0], BorrowLabel::None));
        assert!(matches!(sample.labels.borrow_type[1], BorrowLabel::None));
        assert_eq!(sample.labels.lifetime_group, vec![1, 1]);
        assert_eq!(sample.labels.aliasing, vec![false, false]);
        assert_eq!(sample.labels.escape, vec![false, false]);
        assert_eq!(sample.labels.mutability, vec![false, false]);
    }

    /// `var_index(pcg, "name")` → position of that variable among the
    /// `PcgNode::Variable` nodes, i.e. its index into the label vectors.
    fn var_label_index(pcg: &Pcg, name: &str) -> usize {
        pcg.nodes
            .iter()
            .filter(|n| matches!(n, PcgNode::Variable(_)))
            .position(|n| matches!(n, PcgNode::Variable(v) if v == name))
            .expect("variable present")
    }

    fn build(prog: &SomAst) -> Pcg {
        let mut b = PcgBuilder::new();
        b.build(prog);
        b.pcg
    }

    #[test]
    fn structural_labels_borrows_alias_escape_mutability() {
        // fn f() {
        //   let a = "s"; let b = "s";
        //   let c = &b; let d = &b;   // two shared borrows of b
        //   let e = &mut a;           // mut borrow of a
        //   return &a;                // a escapes via the return place
        // }
        // Hand-derived expectations (variable order a,b,c,d,e):
        //   ownership   = [Borrowed, Borrowed, Owned, Owned, Owned]
        //   borrow_type = [Mutable,  Shared,   None,  None,  None ]
        //   lifetime    = [1,1,1,1,1]  (all owned by region_f, node 1)
        //   aliasing    = [true, true, false, false, false]  (a:&mut+&ret, b:&+&)
        //   escape      = [true, false, false, false, false] (return &a)
        //   mutability  = [true, false, false, false, false] (&mut a)
        let s_ref = |of: &str, mutable: bool| Expression::Reference {
            name: of.to_string(),
            mutable,
        };
        let decl_str = |name: &str| Statement::VarDecl {
            name: name.to_string(),
            ty: Type::Str,
            init: Some(Expression::Literal(Literal::Str("s".to_string()))),
        };
        let decl_ref = |name: &str, of: &str, mutable: bool| Statement::VarDecl {
            name: name.to_string(),
            ty: Type::Ref(Box::new(Type::Int), mutable),
            init: Some(s_ref(of, mutable)),
        };
        let prog = SomAst::Program(vec![Function {
            name: "f".to_string(),
            params: vec![],
            body: vec![
                decl_str("a"),
                decl_str("b"),
                decl_ref("c", "b", false),
                decl_ref("d", "b", false),
                decl_ref("e", "a", true),
                Statement::Return(Some(s_ref("a", false))),
            ],
        }]);
        let pcg = build(&prog);
        let sample = DatasetBuilder::new().generate_sample(pcg.clone());
        let lab = &sample.labels;

        // Variable nodes come out in declaration order a,b,c,d,e at 0..5.
        assert_eq!(var_label_index(&pcg, "a"), 0);
        assert_eq!(var_label_index(&pcg, "e"), 4);
        assert_eq!(lab.ownership.len(), 5);

        assert!(matches!(lab.ownership[0], OwnershipLabel::Borrowed)); // a
        assert!(matches!(lab.ownership[1], OwnershipLabel::Borrowed)); // b
        assert!(matches!(lab.ownership[2], OwnershipLabel::Owned)); // c
        assert!(matches!(lab.ownership[3], OwnershipLabel::Owned)); // d
        assert!(matches!(lab.ownership[4], OwnershipLabel::Owned)); // e

        assert!(matches!(lab.borrow_type[0], BorrowLabel::Mutable)); // a: &mut
        assert!(matches!(lab.borrow_type[1], BorrowLabel::Shared)); // b: &
        assert!(matches!(lab.borrow_type[2], BorrowLabel::None));
        assert!(matches!(lab.borrow_type[3], BorrowLabel::None));
        assert!(matches!(lab.borrow_type[4], BorrowLabel::None));

        assert_eq!(lab.lifetime_group, vec![1, 1, 1, 1, 1]);
        assert_eq!(lab.aliasing, vec![true, true, false, false, false]);
        assert_eq!(lab.escape, vec![true, false, false, false, false]);
        assert_eq!(lab.mutability, vec![true, false, false, false, false]);
    }

    #[test]
    fn structural_labels_nested_scope_groups_and_move_drop() {
        // fn g() { let p = "s"; { let q = p; } }
        // Variable order p,q. q lives in the nested scope region.
        //   ownership      = [Moved, Dropped]   (p moves into q; q drops)
        //   lifetime_group = [1, 3]             (p: region_f=1, q: scope=3)
        //   borrow/alias/escape/mutability all neutral.
        let prog = SomAst::Program(vec![Function {
            name: "g".to_string(),
            params: vec![],
            body: vec![
                Statement::VarDecl {
                    name: "p".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Literal(Literal::Str("s".to_string()))),
                },
                Statement::Scope(vec![Statement::VarDecl {
                    name: "q".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Variable("p".to_string())),
                }]),
            ],
        }]);
        let pcg = build(&prog);
        let sample = DatasetBuilder::new().generate_sample(pcg.clone());
        let lab = &sample.labels;

        assert_eq!(var_label_index(&pcg, "p"), 0);
        assert_eq!(var_label_index(&pcg, "q"), 1);
        assert_eq!(lab.ownership.len(), 2);

        assert!(matches!(lab.ownership[0], OwnershipLabel::Moved)); // p
        assert!(matches!(lab.ownership[1], OwnershipLabel::Dropped)); // q

        // p and q live in different regions ⇒ different lifetime groups.
        assert_eq!(lab.lifetime_group.len(), 2);
        assert_ne!(lab.lifetime_group[0], lab.lifetime_group[1]);
        assert_eq!(lab.lifetime_group, vec![1, 3]);

        assert_eq!(lab.aliasing, vec![false, false]);
        assert_eq!(lab.escape, vec![false, false]);
        assert_eq!(lab.mutability, vec![false, false]);
        assert!(matches!(lab.borrow_type[0], BorrowLabel::None));
        assert!(matches!(lab.borrow_type[1], BorrowLabel::None));
    }

    #[test]
    fn export_jsonl_roundtrips_labels() {
        // export_jsonl must emit one parseable JSON object per sample whose
        // labels survive a serde round-trip unchanged.
        let prog = SomAst::Program(vec![Function {
            name: "f".to_string(),
            params: vec![],
            body: vec![
                Statement::VarDecl {
                    name: "a".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Literal(Literal::Str("s".to_string()))),
                },
                Statement::VarDecl {
                    name: "b".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Variable("a".to_string())),
                },
            ],
        }]);
        let pcg = build(&prog);
        let db = DatasetBuilder::new();
        let s0 = db.generate_sample(pcg.clone());
        let s1 = db.generate_sample(pcg);
        let jsonl = db.export_jsonl(&[s0.clone(), s1]);

        let lines: Vec<&str> = jsonl.lines().collect();
        assert_eq!(lines.len(), 2, "one JSON object per sample");
        let parsed: SomSample = serde_json::from_str(lines[0]).expect("valid JSON");
        // a moves into b ⇒ a Moved, b Owned, both Str so no borrow labels.
        assert_eq!(parsed.labels.ownership.len(), s0.labels.ownership.len());
        assert!(matches!(parsed.labels.ownership[0], OwnershipLabel::Moved));
        assert!(matches!(parsed.labels.ownership[1], OwnershipLabel::Owned));
        assert_eq!(parsed.labels.lifetime_group, s0.labels.lifetime_group);
        assert_eq!(parsed.graph.nodes.len(), s0.graph.nodes.len());
    }
}
