//! Structured Tokenizer for SOM (Non-NLP).
//! Converts AST and PCG elements into a token sequence for ML model input.
//!
//! Two AST linearizations are provided:
//! - [`StructuredTokenizer::tokenize_ast`] — declarations and structure only
//!   (historical stream, kept for compatibility).
//! - [`StructuredTokenizer::tokenize_ast_with_drops`] — the full training
//!   stream: expression effects (`Use`/`Ref`/`MutRef`) are emitted in
//!   evaluation order *before* the binding token, and `Drop` tokens are
//!   emitted for every binding of a scope (reverse declaration order) just
//!   before the scope closes. This stream is what the ownership oracle in
//!   `scirust-som-symbolic` labels, token for token.
//!
//! [`SomVocab`] maps tokens to a closed, deterministic integer vocabulary:
//! variable names are replaced by first-occurrence slots (`0..MAX_VARS`), so
//! the model never sees raw identifiers and the id space is fixed.

use scirust_som_pcg::ast::*;
use scirust_som_pcg::{Pcg, PcgEdge, PcgNode};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SomToken {
    // AST Tokens
    FnDecl(String),
    Param(String),
    VarDecl(String),
    Assign(String),
    /// Use of a variable as a value (move semantics in the toy language).
    Use(String),
    /// Shared borrow `&x`.
    Ref(String),
    /// Mutable borrow `&mut x`.
    MutRef(String),
    /// End-of-scope destruction point of a binding.
    Drop(String),
    Return,
    ScopeStart,
    ScopeEnd,
    // PCG Tokens
    Node(PcgNode),
    Edge(PcgEdge),
    // Structural
    Sep,
}

impl SomToken {
    /// Variable name carried by the token, if any.
    pub fn var_name(&self) -> Option<&str> {
        match self
        {
            SomToken::Param(n)
            | SomToken::VarDecl(n)
            | SomToken::Assign(n)
            | SomToken::Use(n)
            | SomToken::Ref(n)
            | SomToken::MutRef(n)
            | SomToken::Drop(n) => Some(n),
            _ => None,
        }
    }
}

pub struct StructuredTokenizer;

impl Default for StructuredTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl StructuredTokenizer {
    pub fn new() -> Self {
        Self
    }

    /// Historical stream: declarations and structure, no drops.
    pub fn tokenize_ast(&self, ast: &SomAst) -> Vec<SomToken> {
        self.walk(ast, false)
    }

    /// Full training stream: expression tokens + `Drop` tokens.
    ///
    /// Scope bookkeeping is purely syntactic: a binding enters the current
    /// scope on `VarDecl`/`Param`, and on `Assignment` to a name not yet
    /// bound (implicit declaration, mirroring the PCG builder). Drops are
    /// emitted in reverse declaration order before the scope closes; the
    /// function scope drops before the stream ends.
    pub fn tokenize_ast_with_drops(&self, ast: &SomAst) -> Vec<SomToken> {
        self.walk(ast, true)
    }

    fn walk(&self, ast: &SomAst, drops: bool) -> Vec<SomToken> {
        let mut tokens = Vec::new();
        let SomAst::Program(functions) = ast;
        for func in functions
        {
            tokens.push(SomToken::FnDecl(func.name.clone()));
            let mut scopes: Vec<Vec<String>> = vec![Vec::new()];
            for param in &func.params
            {
                tokens.push(SomToken::Param(param.name.clone()));
                scopes.last_mut().expect("scope").push(param.name.clone());
            }
            for stmt in &func.body
            {
                self.walk_stmt(stmt, &mut tokens, &mut scopes, drops);
            }
            let frame = scopes.pop().expect("scope");
            if drops
            {
                for name in frame.iter().rev()
                {
                    tokens.push(SomToken::Drop(name.clone()));
                }
            }
        }
        tokens
    }

    fn walk_stmt(
        &self,
        stmt: &Statement,
        tokens: &mut Vec<SomToken>,
        scopes: &mut Vec<Vec<String>>,
        drops: bool,
    ) {
        match stmt
        {
            Statement::VarDecl { name, init, .. } =>
            {
                if let Some(expr) = init
                {
                    self.walk_expr(expr, tokens);
                }
                tokens.push(SomToken::VarDecl(name.clone()));
                scopes.last_mut().expect("scope").push(name.clone());
            },
            Statement::Assignment { lhs, rhs } =>
            {
                self.walk_expr(rhs, tokens);
                tokens.push(SomToken::Assign(lhs.clone()));
                let bound = scopes.iter().any(|s| s.iter().any(|n| n == lhs));
                if !bound
                {
                    scopes.last_mut().expect("scope").push(lhs.clone());
                }
            },
            Statement::Expression(expr) =>
            {
                self.walk_expr(expr, tokens);
            },
            Statement::Scope(inner) =>
            {
                tokens.push(SomToken::ScopeStart);
                scopes.push(Vec::new());
                for s in inner
                {
                    self.walk_stmt(s, tokens, scopes, drops);
                }
                let frame = scopes.pop().expect("scope");
                if drops
                {
                    for name in frame.iter().rev()
                    {
                        tokens.push(SomToken::Drop(name.clone()));
                    }
                }
                tokens.push(SomToken::ScopeEnd);
            },
            Statement::Return(expr) =>
            {
                if let Some(e) = expr
                {
                    self.walk_expr(e, tokens);
                }
                tokens.push(SomToken::Return);
            },
        }
    }

    fn walk_expr(&self, expr: &Expression, tokens: &mut Vec<SomToken>) {
        match expr
        {
            Expression::Literal(_) =>
            {},
            Expression::Variable(name) => tokens.push(SomToken::Use(name.clone())),
            Expression::Reference { name, mutable } =>
            {
                if *mutable
                {
                    tokens.push(SomToken::MutRef(name.clone()));
                }
                else
                {
                    tokens.push(SomToken::Ref(name.clone()));
                }
            },
            Expression::BinaryOp { left, right, .. } =>
            {
                self.walk_expr(left, tokens);
                self.walk_expr(right, tokens);
            },
            Expression::Call { args, .. } =>
            {
                for arg in args
                {
                    self.walk_expr(arg, tokens);
                }
            },
            Expression::Dereference(inner) => self.walk_expr(inner, tokens),
        }
    }

    pub fn tokenize_pcg(&self, pcg: &Pcg) -> Vec<SomToken> {
        let mut tokens = Vec::new();
        for node in &pcg.nodes
        {
            tokens.push(SomToken::Node(node.clone()));
        }
        tokens.push(SomToken::Sep);
        for (f, t, e) in &pcg.edges
        {
            tokens.push(SomToken::Node(pcg.nodes[*f].clone()));
            tokens.push(SomToken::Edge(e.clone()));
            tokens.push(SomToken::Node(pcg.nodes[*t].clone()));
        }
        tokens
    }
}

/// Maximum number of distinct variable slots in a single token stream.
pub const MAX_VARS: usize = 8;

const VAR_KINDS: usize = 7; // Param, VarDecl, Assign, Use, Ref, MutRef, Drop

/// Closed, deterministic integer vocabulary over [`SomToken`] streams.
///
/// Layout:
/// - `0..=6` : PAD, UNK, FnDecl, Return, ScopeStart, ScopeEnd, Sep
/// - `7..`   : `7 + kind * MAX_VARS + slot` for variable-bearing tokens,
///   with kinds ordered Param, VarDecl, Assign, Use, Ref, MutRef, Drop.
///
/// Slots are assigned by first occurrence of a variable name in the stream;
/// names beyond [`MAX_VARS`] encode to UNK. PCG tokens encode to UNK.
pub struct SomVocab;

impl SomVocab {
    pub const PAD: usize = 0;
    pub const UNK: usize = 1;
    pub const FN_DECL: usize = 2;
    pub const RETURN: usize = 3;
    pub const SCOPE_START: usize = 4;
    pub const SCOPE_END: usize = 5;
    pub const SEP: usize = 6;
    const VAR_BASE: usize = 7;

    /// Total number of token ids.
    pub fn vocab_size() -> usize {
        Self::VAR_BASE + VAR_KINDS * MAX_VARS
    }

    fn kind_index(token: &SomToken) -> Option<usize> {
        match token
        {
            SomToken::Param(_) => Some(0),
            SomToken::VarDecl(_) => Some(1),
            SomToken::Assign(_) => Some(2),
            SomToken::Use(_) => Some(3),
            SomToken::Ref(_) => Some(4),
            SomToken::MutRef(_) => Some(5),
            SomToken::Drop(_) => Some(6),
            _ => None,
        }
    }

    /// Encode a token stream to integer ids (deterministic).
    pub fn encode(tokens: &[SomToken]) -> Vec<usize> {
        let mut slots: HashMap<String, usize> = HashMap::new();
        let mut ids = Vec::with_capacity(tokens.len());
        for token in tokens
        {
            let id = match token
            {
                SomToken::FnDecl(_) => Self::FN_DECL,
                SomToken::Return => Self::RETURN,
                SomToken::ScopeStart => Self::SCOPE_START,
                SomToken::ScopeEnd => Self::SCOPE_END,
                SomToken::Sep => Self::SEP,
                SomToken::Node(_) | SomToken::Edge(_) => Self::UNK,
                _ =>
                {
                    let kind = Self::kind_index(token).expect("var token kind");
                    let name = token.var_name().expect("var token name");
                    let next = slots.len();
                    let slot = *slots.entry(name.to_string()).or_insert(next);
                    if slot < MAX_VARS
                    {
                        Self::VAR_BASE + kind * MAX_VARS + slot
                    }
                    else
                    {
                        Self::UNK
                    }
                },
            };
            ids.push(id);
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn move_then_use_program() -> SomAst {
        // fn main() { let x = 1; let y = x; { let z = &y; } }
        SomAst::Program(vec![Function {
            name: "main".to_string(),
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
                Statement::Scope(vec![Statement::VarDecl {
                    name: "z".to_string(),
                    ty: Type::Ref(Box::new(Type::Int), false),
                    init: Some(Expression::Reference {
                        name: "y".to_string(),
                        mutable: false,
                    }),
                }]),
            ],
        }])
    }

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
                Statement::Scope(vec![Statement::VarDecl {
                    name: "y".to_string(),
                    ty: Type::Int,
                    init: None,
                }]),
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

    #[test]
    fn test_full_stream_evaluation_order_and_drops() {
        let tokens = StructuredTokenizer::new().tokenize_ast_with_drops(&move_then_use_program());
        let expected = vec![
            SomToken::FnDecl("main".into()),
            SomToken::VarDecl("x".into()),
            // RHS of `let y = x` evaluates before the binding token:
            SomToken::Use("x".into()),
            SomToken::VarDecl("y".into()),
            SomToken::ScopeStart,
            SomToken::Ref("y".into()),
            SomToken::VarDecl("z".into()),
            SomToken::Drop("z".into()),
            SomToken::ScopeEnd,
            SomToken::Drop("y".into()),
            SomToken::Drop("x".into()),
        ];
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_vocab_encode_deterministic_and_in_range() {
        let tokens = StructuredTokenizer::new().tokenize_ast_with_drops(&move_then_use_program());
        let ids_a = SomVocab::encode(&tokens);
        let ids_b = SomVocab::encode(&tokens);
        assert_eq!(ids_a, ids_b, "encoding must be deterministic");
        assert!(ids_a.iter().all(|&id| id < SomVocab::vocab_size()));
        // Same kind + same slot ⇒ same id; different kinds ⇒ different ids.
        let use_x = SomVocab::encode(&[SomToken::Use("x".into())])[0];
        let decl_x = SomVocab::encode(&[SomToken::VarDecl("x".into())])[0];
        assert_ne!(use_x, decl_x);
    }

    #[test]
    fn test_vocab_overflow_maps_to_unk() {
        let tokens: Vec<SomToken> = (0..MAX_VARS + 2)
            .map(|i| SomToken::Use(format!("v{i}")))
            .collect();
        let ids = SomVocab::encode(&tokens);
        assert_eq!(ids[MAX_VARS], SomVocab::UNK);
        assert_eq!(ids[MAX_VARS + 1], SomVocab::UNK);
    }
}
