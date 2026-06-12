//! Structured Tokenizer for SOM (Non-NLP).
//! Converts AST and PCG elements into a token sequence for ML model input.

use scirust_som_pcg::ast::*;
use scirust_som_pcg::{Pcg, PcgNode, PcgEdge};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SomToken {
    // AST Tokens
    FnDecl(String),
    Param(String),
    VarDecl(String),
    Assign(String),
    Return,
    ScopeStart,
    ScopeEnd,
    // PCG Tokens
    Node(PcgNode),
    Edge(PcgEdge),
    // Structural
    Sep,
}

pub struct StructuredTokenizer;

impl StructuredTokenizer {
    pub fn new() -> Self {
        Self
    }

    pub fn tokenize_ast(&self, ast: &SomAst) -> Vec<SomToken> {
        let mut tokens = Vec::new();
        match ast {
            SomAst::Program(functions) => {
                for func in functions {
                    tokens.push(SomToken::FnDecl(func.name.clone()));
                    for param in &func.params {
                        tokens.push(SomToken::Param(param.name.clone()));
                    }
                    self.tokenize_body(&func.body, &mut tokens);
                }
            }
        }
        tokens
    }

    fn tokenize_body(&self, body: &[Statement], tokens: &mut Vec<SomToken>) {
        for stmt in body {
            match stmt {
                Statement::VarDecl { name, .. } => tokens.push(SomToken::VarDecl(name.clone())),
                Statement::Assignment { lhs, .. } => tokens.push(SomToken::Assign(lhs.clone())),
                Statement::Return(_) => tokens.push(SomToken::Return),
                Statement::Scope(inner) => {
                    tokens.push(SomToken::ScopeStart);
                    self.tokenize_body(inner, tokens);
                    tokens.push(SomToken::ScopeEnd);
                }
                _ => {}
            }
        }
    }

    pub fn tokenize_pcg(&self, pcg: &Pcg) -> Vec<SomToken> {
        let mut tokens = Vec::new();
        for node in &pcg.nodes {
            tokens.push(SomToken::Node(node.clone()));
        }
        tokens.push(SomToken::Sep);
        for (f, t, e) in &pcg.edges {
            tokens.push(SomToken::Node(pcg.nodes[*f].clone()));
            tokens.push(SomToken::Edge(e.clone()));
            tokens.push(SomToken::Node(pcg.nodes[*t].clone()));
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_tokenization() {
        let prog = SomAst::Program(vec![Function {
            name: "main".to_string(),
            params: vec![],
            body: vec![
                Statement::VarDecl {
                    name: "x".to_string(),
                    ty: Type::Int,
                    init: None,
                },
                Statement::Scope(vec![
                    Statement::VarDecl { name: "y".to_string(), ty: Type::Int, init: None }
                ]),
            ],
        }]);

        let tokenizer = StructuredTokenizer::new();
        let tokens = tokenizer.tokenize_ast(&prog);

        assert_eq!(tokens[0], SomToken::FnDecl("main".to_string()));
        assert_eq!(tokens[1], SomToken::VarDecl("x".to_string()));
        assert_eq!(tokens[2], SomToken::ScopeStart);
        assert_eq!(tokens[3], SomToken::VarDecl("y".to_string()));
        assert_eq!(tokens[4], SomToken::ScopeEnd);
    }
}
