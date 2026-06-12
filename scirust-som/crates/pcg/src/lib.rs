//! PCG (Place Capability Graph) Engine
pub mod ast;

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PcgNode {
    Variable(String),
    MemoryLocation(String),
    Function(String),
    Region(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PcgEdge {
    Owns,
    Borrows,
    MutBorrows,
    Moves,
    Aliases,
    Drops,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Pcg {
    pub nodes: Vec<PcgNode>,
    pub edges: Vec<(usize, usize, PcgEdge)>, // (from_idx, to_idx, edge_type)
}

impl Pcg {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, node: PcgNode) -> usize {
        if let Some(pos) = self.nodes.iter().position(|n| n == &node) {
            return pos;
        }
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    pub fn add_edge(&mut self, from: usize, to: usize, edge: PcgEdge) {
        if !self.edges.iter().any(|(f, t, e)| *f == from && *t == to && *e == edge) {
            self.edges.push((from, to, edge));
        }
    }

    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph PCG {\n");
        for (i, node) in self.nodes.iter().enumerate() {
            dot.push_str(&format!("  node_{} [label=\"{:?}\"];\n", i, node));
        }
        for (from, to, edge) in &self.edges {
            dot.push_str(&format!("  node_{} -> node_{} [label=\"{:?}\"];\n", from, to, edge));
        }
        dot.push_str("}\n");
        dot
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

pub struct PcgBuilder {
    pub pcg: Pcg,
    current_region: Option<usize>,
}

impl Default for PcgBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PcgBuilder {
    pub fn new() -> Self {
        Self {
            pcg: Pcg::new(),
            current_region: None,
        }
    }

    pub fn build(&mut self, ast: &ast::SomAst) {
        match ast {
            ast::SomAst::Program(functions) => {
                for func in functions {
                    self.process_function(func);
                }
            }
        }
    }

    fn process_function(&mut self, func: &ast::Function) {
        let func_node = self.pcg.add_node(PcgNode::Function(func.name.clone()));
        let region_node = self.pcg.add_node(PcgNode::Region(format!("region_{}", func.name)));
        self.pcg.add_edge(func_node, region_node, PcgEdge::Owns);

        let old_region = self.current_region;
        self.current_region = Some(region_node);

        for param in &func.params {
            let param_node = self.pcg.add_node(PcgNode::Variable(param.name.clone()));
            self.pcg.add_edge(region_node, param_node, PcgEdge::Owns);
        }

        for stmt in &func.body {
            self.process_statement(stmt);
        }

        self.current_region = old_region;
    }

    fn process_statement(&mut self, stmt: &ast::Statement) {
        match stmt {
            ast::Statement::VarDecl { name, ty: _, init } => {
                let var_node = self.pcg.add_node(PcgNode::Variable(name.clone()));
                if let Some(region) = self.current_region {
                    self.pcg.add_edge(region, var_node, PcgEdge::Owns);
                }

                if let Some(init_expr) = init {
                    self.process_expression(var_node, init_expr);
                }
            }
            ast::Statement::Assignment { lhs, rhs } => {
                let lhs_node = self.pcg.add_node(PcgNode::Variable(lhs.clone()));
                self.process_expression(lhs_node, rhs);
            }
            ast::Statement::Scope(statements) => {
                let old_region = self.current_region;
                let scope_region = self.pcg.add_node(PcgNode::Region(format!("scope_{}", self.pcg.nodes.len())));
                if let Some(parent) = old_region {
                    self.pcg.add_edge(parent, scope_region, PcgEdge::Owns);
                }
                self.current_region = Some(scope_region);

                for s in statements {
                    self.process_statement(s);
                }

                for i in 0..self.pcg.nodes.len() {
                    if let PcgNode::Variable(_) = &self.pcg.nodes[i] {
                        if self.pcg.edges.iter().any(|(f, t, e)| *f == scope_region && *t == i && *e == PcgEdge::Owns) {
                             self.pcg.add_edge(scope_region, i, PcgEdge::Drops);
                        }
                    }
                }

                self.current_region = old_region;
            }
            ast::Statement::Expression(expr) => {
                let dummy = self.pcg.add_node(PcgNode::MemoryLocation("temp".to_string()));
                self.process_expression(dummy, expr);
            }
            ast::Statement::Return(Some(expr)) => {
                let dummy = self.pcg.add_node(PcgNode::MemoryLocation("return".to_string()));
                self.process_expression(dummy, expr);
            }
            _ => {}
        }
    }

    fn process_expression(&mut self, target_node: usize, expr: &ast::Expression) {
        match expr {
            ast::Expression::Variable(name) => {
                let src_node = self.pcg.add_node(PcgNode::Variable(name.clone()));
                self.pcg.add_edge(src_node, target_node, PcgEdge::Moves);
            }
            ast::Expression::Reference { name, mutable } => {
                let src_node = self.pcg.add_node(PcgNode::Variable(name.clone()));
                let edge = if *mutable { PcgEdge::MutBorrows } else { PcgEdge::Borrows };
                self.pcg.add_edge(target_node, src_node, edge);
            }
            ast::Expression::BinaryOp { left, op: _, right } => {
                self.process_expression(target_node, left);
                self.process_expression(target_node, right);
            }
            ast::Expression::Call { name: _, args } => {
                for arg in args {
                    self.process_expression(target_node, arg);
                }
            }
            ast::Expression::Dereference(inner) => {
                self.process_expression(target_node, inner);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    #[test]
    fn test_pcg_move() {
        let prog = SomAst::Program(vec![Function {
            name: "test_move".to_string(),
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

        let mut builder = PcgBuilder::new();
        builder.build(&prog);
        let pcg = builder.pcg;

        let x_idx = pcg.nodes.iter().position(|n| matches!(n, PcgNode::Variable(name) if name == "x")).unwrap();
        let y_idx = pcg.nodes.iter().position(|n| matches!(n, PcgNode::Variable(name) if name == "y")).unwrap();

        assert!(pcg.edges.iter().any(|(f, t, e)| *f == x_idx && *t == y_idx && *e == PcgEdge::Moves));
    }

    #[test]
    fn test_pcg_borrow() {
        let prog = SomAst::Program(vec![Function {
            name: "test_borrow".to_string(),
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
                    init: Some(Expression::Reference { name: "x".to_string(), mutable: false }),
                },
            ],
        }]);

        let mut builder = PcgBuilder::new();
        builder.build(&prog);
        let pcg = builder.pcg;

        let x_idx = pcg.nodes.iter().position(|n| matches!(n, PcgNode::Variable(name) if name == "x")).unwrap();
        let y_idx = pcg.nodes.iter().position(|n| matches!(n, PcgNode::Variable(name) if name == "y")).unwrap();

        assert!(pcg.edges.iter().any(|(f, t, e)| *f == y_idx && *t == x_idx && *e == PcgEdge::Borrows));
    }

    #[test]
    fn test_pcg_scope_drop() {
        let prog = SomAst::Program(vec![Function {
            name: "test_scope".to_string(),
            params: vec![],
            body: vec![
                Statement::Scope(vec![
                    Statement::VarDecl {
                        name: "x".to_string(),
                        ty: Type::Int,
                        init: Some(Expression::Literal(Literal::Int(1))),
                    },
                ]),
            ],
        }]);

        let mut builder = PcgBuilder::new();
        builder.build(&prog);
        let pcg = builder.pcg;

        let x_idx = pcg.nodes.iter().position(|n| matches!(n, PcgNode::Variable(name) if name == "x")).unwrap();
        let scope_idx = pcg.nodes.iter().position(|n| matches!(n, PcgNode::Region(name) if name.starts_with("scope_"))).unwrap();

        assert!(pcg.edges.iter().any(|(f, t, e)| *f == scope_idx && *t == x_idx && *e == PcgEdge::Owns));
        assert!(pcg.edges.iter().any(|(f, t, e)| *f == scope_idx && *t == x_idx && *e == PcgEdge::Drops));
    }
}
