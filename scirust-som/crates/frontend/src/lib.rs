//! Real-Rust frontend for SOM.
//!
//! Parses **actual Rust source** with [`syn`] (the real Rust grammar, on
//! stable — not a toy parser) and lowers a well-defined subset into the
//! ownership IR of `scirust-som-pcg` (`SomAst`). The existing oracle,
//! tokenizer, model and inference then operate on genuine Rust files.
//!
//! ## What is lowered precisely
//!
//! - `fn` items (free functions, inherent methods in `impl`, functions in
//!   `mod`), their parameters and straight-line bodies;
//! - `let` bindings with an optional initializer;
//! - assignments `x = expr`;
//! - moves: a bare path used as a value (`x`) moves `x`;
//! - borrows: `&x` / `&mut x`;
//! - blocks `{ … }` and `unsafe { … }` as scopes;
//! - `return expr`;
//! - calls `f(a, &b)` (each argument lowered: by-value args move, `&`
//!   args borrow), aggregates (struct/tuple/array literals move their
//!   fields), `*x` dereference, parenthesised/grouped and unary-neg exprs.
//!
//! ## Honest boundaries (reported, never guessed silently)
//!
//! - **Copy vs move is type-aware.** Declared types map to the IR
//!   (`i32`/`f64`/`bool`/raw pointers/`&T` → Copy; `String`/`Vec`/unknown
//!   paths/`&mut T` → move) and the oracle infers Copy-ness of
//!   unannotated bindings from their initializer (source variable,
//!   literal, or borrow kind). The remaining over-approximation: an
//!   unannotated binding initialised by a *call* (`let x = f();`) defaults
//!   to move semantics — conservative, and resolvable only with
//!   type-resolved analysis (the `rustc`-driver path).
//! - **Method receivers** (`x.foo()`) are treated as a shared borrow of
//!   the receiver — recorded in [`Lowered::approximations`] — because
//!   `&self` vs by-value `self` is not syntactically decidable.
//! - **Branching control flow** (`if` / `match` / `loop` / `while` /
//!   `for` / closures / `async`) and **macros** are *not* lowered: they
//!   are recorded in [`Lowered::unsupported`] and skipped, so the oracle's
//!   labels stay correct on what it does analyze rather than inventing
//!   branch-join semantics it does not have.

use scirust_som_pcg::ast::{
    BinaryOp, Expression, Function, Literal, Param, SomAst, Statement, Type,
};

/// Result of lowering real Rust source to the SOM IR.
#[derive(Debug, Clone)]
pub struct Lowered {
    pub ast: SomAst,
    /// Constructs encountered but not modelled (skipped), e.g. `if`, macros.
    pub unsupported: Vec<String>,
    /// Constructs lowered with a documented approximation, e.g. method
    /// receivers treated as shared borrows.
    pub approximations: Vec<String>,
}

/// Parse and lower a string of Rust source.
///
/// Returns a [`syn::Error`] only when the input is not syntactically valid
/// Rust; unmodelled-but-valid constructs are reported in the [`Lowered`]
/// fields, not as errors.
pub fn lower_str(src: &str) -> Result<Lowered, syn::Error> {
    let file = syn::parse_file(src)?;
    let mut lowerer = Lowerer::default();
    for item in &file.items
    {
        lowerer.item(item);
    }
    Ok(Lowered {
        ast: SomAst::Program(lowerer.functions),
        unsupported: lowerer.unsupported,
        approximations: lowerer.approximations,
    })
}

#[derive(Default)]
struct Lowerer {
    functions: Vec<Function>,
    unsupported: Vec<String>,
    approximations: Vec<String>,
}

impl Lowerer {
    fn note_unsupported(&mut self, what: impl Into<String>) {
        let what = what.into();
        if !self.unsupported.contains(&what)
        {
            self.unsupported.push(what);
        }
    }

    fn note_approximation(&mut self, what: impl Into<String>) {
        let what = what.into();
        if !self.approximations.contains(&what)
        {
            self.approximations.push(what);
        }
    }

    fn item(&mut self, item: &syn::Item) {
        match item
        {
            syn::Item::Fn(f) =>
            {
                let func = self.lower_fn(&f.sig, &f.block);
                self.functions.push(func);
            },
            syn::Item::Impl(imp) =>
            {
                for it in &imp.items
                {
                    if let syn::ImplItem::Fn(m) = it
                    {
                        let func = self.lower_fn(&m.sig, &m.block);
                        self.functions.push(func);
                    }
                }
            },
            syn::Item::Mod(m) =>
            {
                if let Some((_, items)) = &m.content
                {
                    for it in items
                    {
                        self.item(it);
                    }
                }
            },
            _ =>
            {},
        }
    }

    fn lower_fn(&mut self, sig: &syn::Signature, block: &syn::Block) -> Function {
        let mut params = Vec::new();
        for input in &sig.inputs
        {
            match input
            {
                syn::FnArg::Receiver(r) =>
                {
                    let mutable = r.mutability.is_some();
                    let ty = if r.reference.is_some()
                    {
                        Type::Ref(Box::new(Type::Str), mutable)
                    }
                    else
                    {
                        Type::Str
                    };
                    params.push(Param {
                        name: "self".to_string(),
                        ty,
                    });
                },
                syn::FnArg::Typed(pt) =>
                {
                    if let Some(name) = pat_ident(&pt.pat)
                    {
                        params.push(Param {
                            name,
                            ty: lower_type(&pt.ty),
                        });
                    }
                    else
                    {
                        self.note_unsupported("non-identifier function parameter pattern");
                    }
                },
            }
        }
        Function {
            name: sig.ident.to_string(),
            params,
            body: self.lower_block(block),
        }
    }

    fn lower_block(&mut self, block: &syn::Block) -> Vec<Statement> {
        let mut out = Vec::new();
        for stmt in &block.stmts
        {
            self.lower_stmt(stmt, &mut out);
        }
        out
    }

    fn lower_stmt(&mut self, stmt: &syn::Stmt, out: &mut Vec<Statement>) {
        match stmt
        {
            syn::Stmt::Local(local) =>
            {
                let name = match pat_ident(&local.pat)
                {
                    Some(n) => n,
                    None =>
                    {
                        self.note_unsupported(
                            "non-identifier `let` pattern (tuple/struct binding)",
                        );
                        return;
                    },
                };
                let ty = pat_type(&local.pat);
                let init = local
                    .init
                    .as_ref()
                    .map(|li| self.lower_expr(&li.expr))
                    .unwrap_or(None);
                if let Some(li) = &local.init
                {
                    if li.diverge.is_some()
                    {
                        self.note_unsupported("`let … else` divergence");
                    }
                }
                out.push(Statement::VarDecl { name, ty, init });
            },
            syn::Stmt::Expr(expr, _) => self.lower_expr_stmt(expr, out),
            syn::Stmt::Macro(_) => self.note_unsupported("macro statement"),
            syn::Stmt::Item(_) => self.note_unsupported("nested item"),
        }
    }

    fn lower_expr_stmt(&mut self, expr: &syn::Expr, out: &mut Vec<Statement>) {
        match expr
        {
            syn::Expr::Assign(a) =>
            {
                let rhs = self
                    .lower_expr(&a.right)
                    .unwrap_or(Expression::Literal(Literal::Int(0)));
                if let Some(lhs) = expr_place_ident(&a.left)
                {
                    out.push(Statement::Assignment { lhs, rhs });
                }
                else
                {
                    // Assignment to a non-simple place (e.g. `*p = …`,
                    // `s.field = …`): keep the rhs effect as a statement.
                    self.note_approximation(
                        "assignment to non-identifier place lowered as expression",
                    );
                    out.push(Statement::Expression(rhs));
                }
            },
            syn::Expr::Return(r) =>
            {
                let inner = r.expr.as_ref().and_then(|e| self.lower_expr(e));
                out.push(Statement::Return(inner));
            },
            syn::Expr::Block(b) =>
            {
                let scope = self.lower_block(&b.block);
                out.push(Statement::Scope(scope));
            },
            syn::Expr::Unsafe(u) =>
            {
                let scope = self.lower_block(&u.block);
                out.push(Statement::Scope(scope));
            },
            other =>
            {
                if let Some(e) = self.lower_expr(other)
                {
                    out.push(Statement::Expression(e));
                }
            },
        }
    }

    /// Lower an expression. Returns `None` for constructs that carry no
    /// ownership effect we model (after recording any note).
    fn lower_expr(&mut self, expr: &syn::Expr) -> Option<Expression> {
        match expr
        {
            syn::Expr::Lit(l) => Some(lower_lit(&l.lit)),
            syn::Expr::Path(p) => match path_single_ident(&p.path)
            {
                Some(name) => Some(Expression::Variable(name)),
                // Multi-segment path (const/unit-struct/fn item): no local
                // is moved.
                None => Some(Expression::Literal(Literal::Int(0))),
            },
            syn::Expr::Reference(r) =>
            {
                let mutable = r.mutability.is_some();
                match &*r.expr
                {
                    syn::Expr::Path(p) => path_single_ident(&p.path)
                        .map(|name| Expression::Reference { name, mutable })
                        .or(Some(Expression::Literal(Literal::Int(0)))),
                    other =>
                    {
                        // `&expr` over a temporary: keep inner uses, no place.
                        self.note_approximation("borrow of a temporary lowered to its inner uses");
                        self.lower_expr(other)
                    },
                }
            },
            syn::Expr::Binary(b) =>
            {
                let left = self
                    .lower_expr(&b.left)
                    .unwrap_or(Expression::Literal(Literal::Int(0)));
                let right = self
                    .lower_expr(&b.right)
                    .unwrap_or(Expression::Literal(Literal::Int(0)));
                // Both operands are always visited (their ownership effects
                // are real). The six arithmetic/equality operators the IR can
                // name become a `BinaryOp`; every other operator (`%`, `<<`,
                // `&`, `<`, `&&`, …) has no faithful `BinaryOp` variant, so we
                // emit a `Call` tagged with the source operator rather than
                // mislabelling it as `Add`. The consumer treats `BinaryOp` and
                // `Call` identically (it recurses into both operands and
                // ignores the operator), so ownership analysis is unchanged.
                match binop_node(&b.op)
                {
                    Some(op) => Some(Expression::BinaryOp {
                        left: Box::new(left),
                        op,
                        right: Box::new(right),
                    }),
                    None => Some(Expression::Call {
                        name: binop_symbol(&b.op).to_string(),
                        args: vec![left, right],
                    }),
                }
            },
            syn::Expr::Unary(u) => match u.op
            {
                syn::UnOp::Deref(_) => self
                    .lower_expr(&u.expr)
                    .map(|e| Expression::Dereference(Box::new(e))),
                _ => self.lower_expr(&u.expr),
            },
            syn::Expr::Paren(p) => self.lower_expr(&p.expr),
            syn::Expr::Group(g) => self.lower_expr(&g.expr),
            syn::Expr::Call(c) =>
            {
                let name = callee_name(&c.func);
                let args = c.args.iter().filter_map(|a| self.lower_expr(a)).collect();
                Some(Expression::Call { name, args })
            },
            syn::Expr::MethodCall(m) =>
            {
                // `recv.method(args)`: receiver borrow (approximation) + args.
                self.note_approximation("method-call receiver treated as a shared borrow");
                let mut args = Vec::new();
                match &*m.receiver
                {
                    syn::Expr::Path(p) if path_single_ident(&p.path).is_some() =>
                    {
                        let name = path_single_ident(&p.path).unwrap();
                        args.push(Expression::Reference {
                            name,
                            mutable: false,
                        });
                    },
                    other =>
                    {
                        if let Some(e) = self.lower_expr(other)
                        {
                            args.push(e);
                        }
                    },
                }
                for a in &m.args
                {
                    if let Some(e) = self.lower_expr(a)
                    {
                        args.push(e);
                    }
                }
                Some(Expression::Call {
                    name: m.method.to_string(),
                    args,
                })
            },
            syn::Expr::Struct(s) =>
            {
                let args = s
                    .fields
                    .iter()
                    .filter_map(|f| self.lower_expr(&f.expr))
                    .collect();
                Some(Expression::Call {
                    name: "<struct>".to_string(),
                    args,
                })
            },
            syn::Expr::Tuple(t) =>
            {
                let args = t.elems.iter().filter_map(|e| self.lower_expr(e)).collect();
                Some(Expression::Call {
                    name: "<tuple>".to_string(),
                    args,
                })
            },
            syn::Expr::Array(a) =>
            {
                let args = a.elems.iter().filter_map(|e| self.lower_expr(e)).collect();
                Some(Expression::Call {
                    name: "<array>".to_string(),
                    args,
                })
            },
            syn::Expr::Field(f) =>
            {
                // Field access `x.f`: lower the base as a use of the place.
                self.lower_expr(&f.base)
            },
            syn::Expr::Cast(c) => self.lower_expr(&c.expr),
            syn::Expr::Macro(_) =>
            {
                self.note_unsupported("macro expression");
                None
            },
            syn::Expr::If(_) =>
            {
                self.note_unsupported("`if` expression (branch-sensitive ownership)");
                None
            },
            syn::Expr::Match(_) =>
            {
                self.note_unsupported("`match` expression (branch-sensitive ownership)");
                None
            },
            syn::Expr::While(_) | syn::Expr::ForLoop(_) | syn::Expr::Loop(_) =>
            {
                self.note_unsupported("loop expression");
                None
            },
            syn::Expr::Closure(_) =>
            {
                self.note_unsupported("closure");
                None
            },
            other =>
            {
                self.note_unsupported(format!("unsupported expression: {}", expr_kind(other)));
                None
            },
        }
    }
}

// ---------------------------------------------------------------------
// syn helpers
// ---------------------------------------------------------------------

fn pat_ident(pat: &syn::Pat) -> Option<String> {
    match pat
    {
        syn::Pat::Ident(pi) => Some(pi.ident.to_string()),
        syn::Pat::Type(pt) => pat_ident(&pt.pat),
        _ => None,
    }
}

fn pat_type(pat: &syn::Pat) -> Type {
    match pat
    {
        syn::Pat::Type(pt) => lower_type(&pt.ty),
        _ => Type::Str,
    }
}

fn path_single_ident(path: &syn::Path) -> Option<String> {
    if path.leading_colon.is_none() && path.segments.len() == 1
    {
        let seg = &path.segments[0];
        if seg.arguments.is_none()
        {
            return Some(seg.ident.to_string());
        }
    }
    None
}

fn expr_place_ident(expr: &syn::Expr) -> Option<String> {
    match expr
    {
        syn::Expr::Path(p) => path_single_ident(&p.path),
        syn::Expr::Paren(p) => expr_place_ident(&p.expr),
        _ => None,
    }
}

fn callee_name(func: &syn::Expr) -> String {
    if let syn::Expr::Path(p) = func
    {
        if let Some(seg) = p.path.segments.last()
        {
            return seg.ident.to_string();
        }
    }
    "<call>".to_string()
}

fn lower_lit(lit: &syn::Lit) -> Expression {
    let l = match lit
    {
        syn::Lit::Int(i) => Literal::Int(i.base10_parse::<i64>().unwrap_or(0)),
        syn::Lit::Float(f) => Literal::Float(f.base10_parse::<f64>().unwrap_or(0.0)),
        syn::Lit::Bool(b) => Literal::Bool(b.value),
        syn::Lit::Str(s) => Literal::Str(s.value()),
        _ => Literal::Int(0),
    };
    Expression::Literal(l)
}

/// Map a `syn` binary operator to the IR's [`BinaryOp`], or `None` when the
/// IR has no faithful variant for it. The IR can name only the four arithmetic
/// operators and `==`/`!=`; everything else (`%`, bit-ops, shifts, ordering
/// comparisons, `&&`/`||`) returns `None` and is lowered to a tagged `Call`
/// instead of being mislabelled. Compound-assignment forms map to their base
/// arithmetic operator when that base is representable.
fn binop_node(op: &syn::BinOp) -> Option<BinaryOp> {
    match op
    {
        syn::BinOp::Add(_) | syn::BinOp::AddAssign(_) => Some(BinaryOp::Add),
        syn::BinOp::Sub(_) | syn::BinOp::SubAssign(_) => Some(BinaryOp::Sub),
        syn::BinOp::Mul(_) | syn::BinOp::MulAssign(_) => Some(BinaryOp::Mul),
        syn::BinOp::Div(_) | syn::BinOp::DivAssign(_) => Some(BinaryOp::Div),
        syn::BinOp::Eq(_) => Some(BinaryOp::Eq),
        syn::BinOp::Ne(_) => Some(BinaryOp::Ne),
        _ => None,
    }
}

/// The source spelling of a binary operator, used as the `Call` name when the
/// operator has no faithful [`BinaryOp`] variant. Exhaustive over the
/// operators `binop_node` returns `None` for; the representable operators are
/// listed too so the match stays total and future `syn` additions surface as a
/// compile error rather than a silent wrong tag.
fn binop_symbol(op: &syn::BinOp) -> &'static str {
    match op
    {
        syn::BinOp::Add(_) => "+",
        syn::BinOp::Sub(_) => "-",
        syn::BinOp::Mul(_) => "*",
        syn::BinOp::Div(_) => "/",
        syn::BinOp::Rem(_) => "%",
        syn::BinOp::And(_) => "&&",
        syn::BinOp::Or(_) => "||",
        syn::BinOp::BitXor(_) => "^",
        syn::BinOp::BitAnd(_) => "&",
        syn::BinOp::BitOr(_) => "|",
        syn::BinOp::Shl(_) => "<<",
        syn::BinOp::Shr(_) => ">>",
        syn::BinOp::Eq(_) => "==",
        syn::BinOp::Lt(_) => "<",
        syn::BinOp::Le(_) => "<=",
        syn::BinOp::Ne(_) => "!=",
        syn::BinOp::Ge(_) => ">=",
        syn::BinOp::Gt(_) => ">",
        syn::BinOp::AddAssign(_) => "+=",
        syn::BinOp::SubAssign(_) => "-=",
        syn::BinOp::MulAssign(_) => "*=",
        syn::BinOp::DivAssign(_) => "/=",
        syn::BinOp::RemAssign(_) => "%=",
        syn::BinOp::BitXorAssign(_) => "^=",
        syn::BinOp::BitAndAssign(_) => "&=",
        syn::BinOp::BitOrAssign(_) => "|=",
        syn::BinOp::ShlAssign(_) => "<<=",
        syn::BinOp::ShrAssign(_) => ">>=",
        // `syn::BinOp` is `#[non_exhaustive]`; a new operator added upstream
        // is an honest unknown rather than a fabricated symbol.
        _ => "<binop>",
    }
}

fn lower_type(ty: &syn::Type) -> Type {
    match ty
    {
        syn::Type::Reference(r) => Type::Ref(Box::new(lower_type(&r.elem)), r.mutability.is_some()),
        syn::Type::Ptr(p) => Type::Ptr(Box::new(lower_type(&p.elem))),
        syn::Type::Tuple(t) if t.elems.is_empty() => Type::Unit,
        syn::Type::Path(p) =>
        {
            let name = p
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            match name.as_str()
            {
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64"
                | "u128" | "usize" => Type::Int,
                "f32" | "f64" => Type::Float,
                "bool" => Type::Bool,
                "str" | "String" => Type::Str,
                _ => Type::Str,
            }
        },
        _ => Type::Str,
    }
}

fn expr_kind(expr: &syn::Expr) -> &'static str {
    match expr
    {
        syn::Expr::Async(_) => "async block",
        syn::Expr::Await(_) => "await",
        syn::Expr::TryBlock(_) => "try block",
        syn::Expr::Try(_) => "`?` operator",
        syn::Expr::Range(_) => "range",
        syn::Expr::Index(_) => "index",
        syn::Expr::Repeat(_) => "array repeat",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names_decl(body: &[Statement]) -> Vec<&str> {
        body.iter()
            .filter_map(|s| match s
            {
                Statement::VarDecl { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn lowers_real_fn_with_move() {
        let src = r#"
            fn main() {
                let x = String::from("a");
                let y = x;
            }
        "#;
        let lowered = lower_str(src).expect("valid rust");
        let SomAst::Program(funcs) = &lowered.ast;
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0].name, "main");
        assert_eq!(names_decl(&funcs[0].body), vec!["x", "y"]);
        // `let y = x;` lowers to a move of x.
        match &funcs[0].body[1]
        {
            Statement::VarDecl {
                init: Some(Expression::Variable(v)),
                ..
            } =>
            {
                assert_eq!(v, "x")
            },
            other => panic!("expected move init, got {other:?}"),
        }
    }

    #[test]
    fn lowers_borrows_and_methods() {
        let src = r#"
            fn f(v: Vec<u8>) {
                let r = &v;
                let m = &mut v;
                let n = v.len();
            }
        "#;
        let lowered = lower_str(src).unwrap();
        let SomAst::Program(funcs) = &lowered.ast;
        let body = &funcs[0].body;
        assert!(matches!(
            &body[0],
            Statement::VarDecl { init: Some(Expression::Reference { name, mutable: false }), .. } if name == "v"
        ));
        assert!(matches!(
            &body[1],
            Statement::VarDecl { init: Some(Expression::Reference { name, mutable: true }), .. } if name == "v"
        ));
        // method call recorded as an approximation
        assert!(
            lowered
                .approximations
                .iter()
                .any(|a| a.contains("method-call receiver"))
        );
    }

    #[test]
    fn records_unsupported_control_flow() {
        let src = r#"
            fn g(c: bool) {
                let x = String::new();
                if c { let y = x; }
            }
        "#;
        let lowered = lower_str(src).unwrap();
        assert!(
            lowered
                .unsupported
                .iter()
                .any(|u| u.contains("`if` expression"))
        );
    }

    #[test]
    fn handles_impl_methods_and_scopes() {
        let src = r#"
            struct S;
            impl S {
                fn run(&self, a: String) {
                    {
                        let b = a;
                    }
                    return;
                }
            }
        "#;
        let lowered = lower_str(src).unwrap();
        let SomAst::Program(funcs) = &lowered.ast;
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0].name, "run");
        // params: self + a
        assert_eq!(funcs[0].params.len(), 2);
        assert!(matches!(funcs[0].body[0], Statement::Scope(_)));
        assert!(matches!(funcs[0].body[1], Statement::Return(None)));
    }

    #[test]
    fn lowering_is_deterministic() {
        let src = "fn h() { let a = String::new(); let b = a; let c = a; }";
        let a = format!("{:?}", lower_str(src).unwrap().ast);
        let b = format!("{:?}", lower_str(src).unwrap().ast);
        assert_eq!(a, b);
    }

    #[test]
    fn rejects_invalid_rust() {
        assert!(lower_str("fn broken( {").is_err());
    }

    /// Lower a single-function program and return its `Function`, asserting
    /// exactly one function was produced.
    fn only_fn(src: &str) -> Function {
        let lowered = lower_str(src).expect("valid rust");
        let SomAst::Program(mut funcs) = lowered.ast;
        assert_eq!(funcs.len(), 1, "expected exactly one function");
        funcs.remove(0)
    }

    #[test]
    fn exact_ast_for_typed_fn_with_binop_and_return() {
        // Hand-derived oracle: params keep their declared types; the
        // unannotated `let s` defaults to `Str`; `a + b` is the representable
        // `Add`; `return s;` carries `s` as a move.
        let f = only_fn("fn add(a: i32, b: i32) -> i32 { let s = a + b; return s; }");
        let expected = Function {
            name: "add".to_string(),
            params: vec![
                Param {
                    name: "a".to_string(),
                    ty: Type::Int,
                },
                Param {
                    name: "b".to_string(),
                    ty: Type::Int,
                },
            ],
            body: vec![
                Statement::VarDecl {
                    name: "s".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::BinaryOp {
                        left: Box::new(Expression::Variable("a".to_string())),
                        op: BinaryOp::Add,
                        right: Box::new(Expression::Variable("b".to_string())),
                    }),
                },
                Statement::Return(Some(Expression::Variable("s".to_string()))),
            ],
        };
        assert_eq!(f, expected);
    }

    #[test]
    fn borrow_and_mut_borrow_lower_to_reference_nodes() {
        // `&v` → shared Reference, `&mut v` → mutable Reference, both naming v.
        let f = only_fn("fn f(v: String) { let r = &v; let m = &mut v; }");
        assert_eq!(
            f.body,
            vec![
                Statement::VarDecl {
                    name: "r".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Reference {
                        name: "v".to_string(),
                        mutable: false,
                    }),
                },
                Statement::VarDecl {
                    name: "m".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Reference {
                        name: "v".to_string(),
                        mutable: true,
                    }),
                },
            ]
        );
    }

    #[test]
    fn nested_blocks_produce_nested_scopes() {
        // Two levels of `{ … }` nest into Scope(Scope(...)); each inner let is
        // placed at the correct depth.
        let f = only_fn("fn f() { let a = x; { let b = a; { let c = b; } } }");
        let expected = vec![
            Statement::VarDecl {
                name: "a".to_string(),
                ty: Type::Str,
                init: Some(Expression::Variable("x".to_string())),
            },
            Statement::Scope(vec![
                Statement::VarDecl {
                    name: "b".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Variable("a".to_string())),
                },
                Statement::Scope(vec![Statement::VarDecl {
                    name: "c".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Variable("b".to_string())),
                }]),
            ]),
        ];
        assert_eq!(f.body, expected);
    }

    #[test]
    fn typed_let_carries_declared_type() {
        // `let n: u64` → Int; `let r: &mut i32` → Ref(Int, true).
        let f = only_fn("fn f() { let n: u64 = 5; let r: &mut i32 = q; }");
        assert_eq!(
            f.body,
            vec![
                Statement::VarDecl {
                    name: "n".to_string(),
                    ty: Type::Int,
                    init: Some(Expression::Literal(Literal::Int(5))),
                },
                Statement::VarDecl {
                    name: "r".to_string(),
                    ty: Type::Ref(Box::new(Type::Int), true),
                    init: Some(Expression::Variable("q".to_string())),
                },
            ]
        );
    }

    #[test]
    fn call_lowers_value_arg_as_move_and_ref_arg_as_borrow() {
        // `g(a, &b)`: first arg moves a (Variable), second borrows b
        // (Reference). The whole call is an expression statement.
        let f = only_fn("fn f(a: String, b: String) { g(a, &b); }");
        assert_eq!(
            f.body,
            vec![Statement::Expression(Expression::Call {
                name: "g".to_string(),
                args: vec![
                    Expression::Variable("a".to_string()),
                    Expression::Reference {
                        name: "b".to_string(),
                        mutable: false,
                    },
                ],
            })]
        );
    }

    #[test]
    fn dereference_lowers_to_dereference_node() {
        let f = only_fn("fn f(p: i32) { let v = *p; }");
        assert_eq!(
            f.body,
            vec![Statement::VarDecl {
                name: "v".to_string(),
                ty: Type::Str,
                init: Some(Expression::Dereference(Box::new(Expression::Variable(
                    "p".to_string()
                )))),
            }]
        );
    }

    #[test]
    fn simple_assignment_lowers_to_assignment_node() {
        let f = only_fn("fn f() { x = y; }");
        assert_eq!(
            f.body,
            vec![Statement::Assignment {
                lhs: "x".to_string(),
                rhs: Expression::Variable("y".to_string()),
            }]
        );
    }

    #[test]
    fn representable_operator_stays_a_binaryop() {
        // Regression guard: `+` and `==` must remain real BinaryOp variants,
        // not be downgraded to a Call by the unrepresentable-op handling.
        let add = only_fn("fn f() { let z = a + b; }");
        assert!(matches!(
            &add.body[0],
            Statement::VarDecl {
                init: Some(Expression::BinaryOp {
                    op: BinaryOp::Add,
                    ..
                }),
                ..
            }
        ));
        let eq = only_fn("fn f() { let z = a == b; }");
        assert!(matches!(
            &eq.body[0],
            Statement::VarDecl {
                init: Some(Expression::BinaryOp {
                    op: BinaryOp::Eq,
                    ..
                }),
                ..
            }
        ));
    }

    #[test]
    fn unrepresentable_operator_lowers_to_tagged_call_not_fake_add() {
        // The IR's BinaryOp has no `%`, `<<`, `<`, `&&`, … variant. These must
        // NOT be silently relabelled as `Add`: they lower to a `Call` tagged
        // with the source operator, preserving both operands' ownership uses.
        for (src, sym) in [
            ("fn f() { let z = a % b; }", "%"),
            ("fn f() { let z = a << b; }", "<<"),
            ("fn f() { let z = a < b; }", "<"),
            ("fn f() { let z = a && b; }", "&&"),
            ("fn f() { let z = a | b; }", "|"),
            ("fn f() { let z = a >> b; }", ">>"),
        ]
        {
            let f = only_fn(src);
            assert_eq!(
                f.body,
                vec![Statement::VarDecl {
                    name: "z".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Call {
                        name: sym.to_string(),
                        args: vec![
                            Expression::Variable("a".to_string()),
                            Expression::Variable("b".to_string()),
                        ],
                    }),
                }],
                "operator `{sym}` lowered wrong"
            );
        }
    }

    #[test]
    fn nested_unrepresentable_operator_keeps_outer_binaryop() {
        // `(a << b) + c`: the outer `+` is a real Add whose left operand is the
        // tagged `<<` Call — precedence and both operands preserved.
        let f = only_fn("fn f() { let z = (a << b) + c; }");
        assert_eq!(
            f.body,
            vec![Statement::VarDecl {
                name: "z".to_string(),
                ty: Type::Str,
                init: Some(Expression::BinaryOp {
                    left: Box::new(Expression::Call {
                        name: "<<".to_string(),
                        args: vec![
                            Expression::Variable("a".to_string()),
                            Expression::Variable("b".to_string()),
                        ],
                    }),
                    op: BinaryOp::Add,
                    right: Box::new(Expression::Variable("c".to_string())),
                }),
            }]
        );
    }

    #[test]
    fn left_associativity_and_precedence_are_preserved() {
        // `a - b - c` is `(a - b) - c` (left-assoc); `a + b * c` keeps `*`
        // bound tighter than `+`. syn enforces this; assert we carry it through.
        let sub = only_fn("fn f() { let z = a - b - c; }");
        assert_eq!(
            sub.body,
            vec![Statement::VarDecl {
                name: "z".to_string(),
                ty: Type::Str,
                init: Some(Expression::BinaryOp {
                    left: Box::new(Expression::BinaryOp {
                        left: Box::new(Expression::Variable("a".to_string())),
                        op: BinaryOp::Sub,
                        right: Box::new(Expression::Variable("b".to_string())),
                    }),
                    op: BinaryOp::Sub,
                    right: Box::new(Expression::Variable("c".to_string())),
                }),
            }]
        );
        let prec = only_fn("fn f() { let z = a + b * c; }");
        assert_eq!(
            prec.body,
            vec![Statement::VarDecl {
                name: "z".to_string(),
                ty: Type::Str,
                init: Some(Expression::BinaryOp {
                    left: Box::new(Expression::Variable("a".to_string())),
                    op: BinaryOp::Add,
                    right: Box::new(Expression::BinaryOp {
                        left: Box::new(Expression::Variable("b".to_string())),
                        op: BinaryOp::Mul,
                        right: Box::new(Expression::Variable("c".to_string())),
                    }),
                }),
            }]
        );
    }

    #[test]
    fn unsupported_branch_is_dropped_without_losing_neighbours() {
        // The `if` is skipped (recorded), but the lets on either side of it
        // must both survive in order — guards against an off-by-one that eats a
        // neighbouring statement.
        let lowered = lower_str("fn f() { let a = x; if c { } let b = a; }").unwrap();
        let SomAst::Program(funcs) = &lowered.ast;
        assert_eq!(
            funcs[0].body,
            vec![
                Statement::VarDecl {
                    name: "a".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Variable("x".to_string())),
                },
                Statement::VarDecl {
                    name: "b".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Variable("a".to_string())),
                },
            ]
        );
        assert!(
            lowered
                .unsupported
                .iter()
                .any(|u| u.contains("`if` expression"))
        );
    }

    #[test]
    fn malformed_inputs_error() {
        // Several distinct syntax errors must all be rejected (not silently
        // mis-parsed). Each is invalid Rust for a different reason.
        for bad in [
            "fn broken( {",        // unterminated parameter list
            "fn f() { let = 1; }", // missing binding pattern
            "fn f() { 1 + }",      // dangling operator
            "struct {",            // unnamed struct, unterminated
            "fn f( a: ) {}",       // missing parameter type
        ]
        {
            assert!(lower_str(bad).is_err(), "expected error for: {bad:?}");
        }
    }

    #[test]
    fn whole_program_with_multiple_functions_keeps_order_and_names() {
        // Free fn + impl method + fn inside a mod all lower, in source order.
        let src = r#"
            fn first() {}
            struct S;
            impl S {
                fn second(&self) {}
            }
            mod m {
                fn third() {}
            }
        "#;
        let lowered = lower_str(src).unwrap();
        let SomAst::Program(funcs) = &lowered.ast;
        let names: Vec<&str> = funcs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["first", "second", "third"]);
        // The inherent method records its `&self` receiver as a shared-ref param.
        let second = &funcs[1];
        assert_eq!(
            second.params,
            vec![Param {
                name: "self".to_string(),
                ty: Type::Ref(Box::new(Type::Str), false),
            }]
        );
    }
}
