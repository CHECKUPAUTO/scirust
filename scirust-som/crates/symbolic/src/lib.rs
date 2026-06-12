//! Deterministic ownership oracle for SOM.
//!
//! An abstract interpreter over the toy AST of `scirust-som-pcg` that emits
//! the *same* token stream as
//! `StructuredTokenizer::tokenize_ast_with_drops` and labels every token
//! with the ground-truth ownership facts. It is the single source of truth
//! for SOM training labels and the oracle against which the neural model is
//! validated — no randomness, no floats, bit-stable output.
//!
//! ## Typed semantics (documented contract)
//!
//! The oracle is **type-aware**, matching Rust's Copy/move split on the
//! IR's type vocabulary:
//!
//! - **Copy types** — `Int`, `Float`, `Bool`, `Unit`, raw pointers and
//!   shared references `&T`: using the variable as a value *copies* it;
//!   the variable stays usable. Copying while a `&mut` borrow is
//!   outstanding is still a fault ([`FaultKind::UseWhileMutBorrowed`]).
//! - **Move types** — `Str` (the stand-in for `String`/`Vec`/unknown
//!   owner types) and `&mut T`: any value use *moves* the variable.
//!   Unannotated bindings infer locally: from the source variable for
//!   `let y = x;`, from the literal for `let n = 1;`, from the borrow kind
//!   for `let r = &x;`; otherwise they default to **move** (conservative).
//! - `&x` / `&mut x` take borrows: any number of shared XOR one mutable.
//!   A borrow granted in a `VarDecl` initializer or `Assignment` RHS is
//!   held by the bound variable and released when it drops, moves or is
//!   reassigned; borrows in expression statements / `return` end with the
//!   statement.
//! - Bindings drop in reverse declaration order at scope end; moved-out
//!   bindings do not drop (their `Drop` token is labelled `Moved`).
//! - Assignment re-initializes: a moved variable becomes `Owned` again
//!   (Rust re-initialization). Assigning to an undeclared name implicitly
//!   declares it (flagged), mirroring the PCG builder.
//! - `return &local` is flagged as an escaping borrow.

use scirust_som_pcg::ast::{Expression, Function, Literal, SomAst, Statement, Type};
use scirust_som_tokenizer::SomToken;

/// Whether values of `ty` have Copy semantics in the oracle's model.
///
/// Mirrors Rust for the IR's type vocabulary: scalars, `()`, raw pointers
/// and shared references are `Copy`; `Str` (owner types) and `&mut T` are
/// move-only.
pub fn type_is_copy(ty: &Type) -> bool {
    match ty
    {
        Type::Int | Type::Float | Type::Bool | Type::Unit | Type::Ptr(_) => true,
        Type::Ref(_, mutable) => !mutable,
        Type::Str => false,
    }
}

// ---------------------------------------------------------------------
// Label space
// ---------------------------------------------------------------------

/// Ownership classes (per token).
pub const OWNERSHIP_NA: usize = 0;
pub const OWNERSHIP_OWNED: usize = 1;
pub const OWNERSHIP_BORROWED: usize = 2;
pub const OWNERSHIP_MOVED: usize = 3;
pub const OWNERSHIP_DROPPED: usize = 4;
pub const OWNERSHIP_CLASSES: usize = 5;

/// Borrow classes (per token): outstanding borrows *on* the variable.
pub const BORROW_NA: usize = 0;
pub const BORROW_NONE: usize = 1;
pub const BORROW_SHARED: usize = 2;
pub const BORROW_MUT: usize = 3;
pub const BORROW_CLASSES: usize = 4;

pub fn ownership_name(id: usize) -> &'static str {
    match id
    {
        OWNERSHIP_OWNED => "Owned",
        OWNERSHIP_BORROWED => "Borrowed",
        OWNERSHIP_MOVED => "Moved",
        OWNERSHIP_DROPPED => "Dropped",
        _ => "-",
    }
}

pub fn borrow_name(id: usize) -> &'static str {
    match id
    {
        BORROW_NONE => "None",
        BORROW_SHARED => "Shared",
        BORROW_MUT => "Mut",
        _ => "-",
    }
}

/// Ground-truth label attached to one token of the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenLabel {
    /// Ownership state of the mentioned variable *after* the token's effect.
    pub ownership: usize,
    /// Outstanding borrows on the mentioned variable after the effect.
    pub borrow: usize,
    /// True when the token itself is a fault (use-after-move, conflict…).
    pub invalid: bool,
}

const NA_LABEL: TokenLabel = TokenLabel {
    ownership: OWNERSHIP_NA,
    borrow: BORROW_NA,
    invalid: false,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultKind {
    UseOfUndeclared,
    UseAfterMove,
    MoveWhileBorrowed,
    /// Copying a `Copy` value while a `&mut` borrow on it is outstanding.
    UseWhileMutBorrowed,
    BorrowOfMoved,
    BorrowConflict,
    AssignWhileBorrowed,
    AssignToUndeclared,
    EscapingBorrow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// Index of the offending token in [`Analysis::tokens`].
    pub token_index: usize,
    pub var: String,
    pub kind: FaultKind,
}

/// Output of the oracle: aligned tokens + labels, plus diagnostics.
#[derive(Debug, Clone, Default)]
pub struct Analysis {
    pub tokens: Vec<SomToken>,
    pub labels: Vec<TokenLabel>,
    pub diagnostics: Vec<Diagnostic>,
}

impl Analysis {
    pub fn ownership_ids(&self) -> Vec<usize> {
        self.labels.iter().map(|l| l.ownership).collect()
    }
    pub fn borrow_ids(&self) -> Vec<usize> {
        self.labels.iter().map(|l| l.borrow).collect()
    }
    pub fn invalid_flags(&self) -> Vec<f32> {
        self.labels
            .iter()
            .map(|l| if l.invalid { 1.0 } else { 0.0 })
            .collect()
    }
}

// ---------------------------------------------------------------------
// Interpreter state
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VarState {
    Owned,
    Borrowed,
    Moved,
    Dropped,
}

#[derive(Debug)]
struct VarRecord {
    name: String,
    state: VarState,
    shared: u32,
    muted: bool,
    /// Copy semantics (see [`type_is_copy`]): value uses copy, never move.
    copyable: bool,
    /// Borrows this binding holds on other variables: (target id, is_mut).
    holds: Vec<(usize, bool)>,
}

#[derive(Default)]
struct ScopeFrame {
    bindings: Vec<(String, usize)>,
    declared: Vec<usize>,
}

/// The deterministic ownership oracle.
#[derive(Default)]
pub struct OwnershipOracle;

impl OwnershipOracle {
    pub fn new() -> Self {
        Self
    }

    pub fn analyze(&self, ast: &SomAst) -> Analysis {
        let mut interp = Interp::default();
        let SomAst::Program(functions) = ast;
        for func in functions
        {
            interp.function(func);
        }
        interp.out
    }
}

#[derive(Default)]
struct Interp {
    vars: Vec<VarRecord>,
    scopes: Vec<ScopeFrame>,
    out: Analysis,
}

impl Interp {
    fn emit(&mut self, token: SomToken, label: TokenLabel) {
        self.out.tokens.push(token);
        self.out.labels.push(label);
    }

    /// Record a diagnostic for the token about to be emitted.
    fn fault(&mut self, var: &str, kind: FaultKind) {
        self.out.diagnostics.push(Diagnostic {
            token_index: self.out.tokens.len(),
            var: var.to_string(),
            kind,
        });
    }

    fn resolve(&self, name: &str) -> Option<usize> {
        for frame in self.scopes.iter().rev()
        {
            if let Some((_, id)) = frame.bindings.iter().rev().find(|(n, _)| n == name)
            {
                return Some(*id);
            }
        }
        None
    }

    fn declare(&mut self, name: &str, copyable: bool) -> usize {
        let id = self.vars.len();
        self.vars.push(VarRecord {
            name: name.to_string(),
            state: VarState::Owned,
            shared: 0,
            muted: false,
            copyable,
            holds: Vec::new(),
        });
        let frame = self.scopes.last_mut().expect("scope");
        frame.bindings.push((name.to_string(), id));
        frame.declared.push(id);
        id
    }

    /// Copy-ness of a new binding: explicit type wins; an unannotated
    /// binding (`Str` is also the frontend's "unknown" marker) infers from
    /// its initializer; anything else defaults to move (conservative).
    fn binding_copyable(&self, ty: &Type, init: Option<&Expression>) -> bool {
        if !matches!(ty, Type::Str)
        {
            return type_is_copy(ty);
        }
        match init
        {
            Some(Expression::Variable(src)) => self
                .resolve(src)
                .map(|id| self.vars[id].copyable)
                .unwrap_or(false),
            Some(Expression::Literal(Literal::Int(_)))
            | Some(Expression::Literal(Literal::Float(_)))
            | Some(Expression::Literal(Literal::Bool(_))) => true,
            Some(Expression::Reference { mutable, .. }) => !mutable,
            _ => false,
        }
    }

    fn state_label(&self, id: usize) -> usize {
        match self.vars[id].state
        {
            VarState::Owned => OWNERSHIP_OWNED,
            VarState::Borrowed => OWNERSHIP_BORROWED,
            VarState::Moved => OWNERSHIP_MOVED,
            VarState::Dropped => OWNERSHIP_DROPPED,
        }
    }

    fn borrow_label(&self, id: usize) -> usize {
        let v = &self.vars[id];
        if v.muted
        {
            BORROW_MUT
        }
        else if v.shared > 0
        {
            BORROW_SHARED
        }
        else
        {
            BORROW_NONE
        }
    }

    fn is_borrowed(&self, id: usize) -> bool {
        self.vars[id].muted || self.vars[id].shared > 0
    }

    /// Release one granted borrow on `target`.
    fn release_one(&mut self, target: usize, is_mut: bool) {
        let v = &mut self.vars[target];
        if is_mut
        {
            v.muted = false;
        }
        else
        {
            v.shared = v.shared.saturating_sub(1);
        }
        if !v.muted && v.shared == 0 && v.state == VarState::Borrowed
        {
            v.state = VarState::Owned;
        }
    }

    /// Release every borrow held *by* `id`.
    fn release_holds(&mut self, id: usize) {
        let holds = std::mem::take(&mut self.vars[id].holds);
        for (target, is_mut) in holds
        {
            self.release_one(target, is_mut);
        }
    }

    fn release_temps(&mut self, temps: Vec<(usize, bool)>) {
        for (target, is_mut) in temps
        {
            self.release_one(target, is_mut);
        }
    }

    // -----------------------------------------------------------------
    // Walk
    // -----------------------------------------------------------------

    fn function(&mut self, func: &Function) {
        self.emit(SomToken::FnDecl(func.name.clone()), NA_LABEL);
        self.scopes.push(ScopeFrame::default());
        for param in &func.params
        {
            self.declare(&param.name, type_is_copy(&param.ty));
            self.emit(
                SomToken::Param(param.name.clone()),
                TokenLabel {
                    ownership: OWNERSHIP_OWNED,
                    borrow: BORROW_NONE,
                    invalid: false,
                },
            );
        }
        for stmt in &func.body
        {
            self.statement(stmt);
        }
        self.end_scope();
    }

    fn end_scope(&mut self) {
        let frame = self.scopes.pop().expect("scope");
        for &id in frame.declared.iter().rev()
        {
            self.release_holds(id);
            if self.vars[id].state != VarState::Moved
            {
                self.vars[id].state = VarState::Dropped;
            }
            let label = TokenLabel {
                ownership: self.state_label(id),
                borrow: self.borrow_label(id),
                invalid: false,
            };
            self.emit(SomToken::Drop(self.vars[id].name.clone()), label);
        }
    }

    fn statement(&mut self, stmt: &Statement) {
        match stmt
        {
            Statement::VarDecl { name, ty, init } =>
            {
                // Copy-ness must read the *outer* environment (`let y = x;`
                // inherits from the pre-existing `x`), so infer before the
                // initializer's effects run.
                let copyable = self.binding_copyable(ty, init.as_ref());
                let temps = match init
                {
                    Some(expr) => self.expression(expr, false),
                    None => Vec::new(),
                };
                // The binding only becomes visible after its initializer
                // ran, so `let x = x;` resolves the outer `x`.
                let id = self.declare(name, copyable);
                self.vars[id].holds = temps;
                self.emit(
                    SomToken::VarDecl(name.clone()),
                    TokenLabel {
                        ownership: OWNERSHIP_OWNED,
                        borrow: BORROW_NONE,
                        invalid: false,
                    },
                );
            },
            Statement::Assignment { lhs, rhs } =>
            {
                let temps = self.expression(rhs, false);
                match self.resolve(lhs)
                {
                    Some(id) =>
                    {
                        let invalid = self.is_borrowed(id);
                        if invalid
                        {
                            self.fault(lhs, FaultKind::AssignWhileBorrowed);
                        }
                        self.release_holds(id);
                        self.vars[id].holds = temps;
                        // Re-initialization: a moved variable becomes owned
                        // again after assignment.
                        self.vars[id].state = if self.is_borrowed(id)
                        {
                            VarState::Borrowed
                        }
                        else
                        {
                            VarState::Owned
                        };
                        self.emit(
                            SomToken::Assign(lhs.clone()),
                            TokenLabel {
                                ownership: self.state_label(id),
                                borrow: self.borrow_label(id),
                                invalid,
                            },
                        );
                    },
                    None =>
                    {
                        self.fault(lhs, FaultKind::AssignToUndeclared);
                        let id = self.declare(lhs, false);
                        self.vars[id].holds = temps;
                        self.emit(
                            SomToken::Assign(lhs.clone()),
                            TokenLabel {
                                ownership: OWNERSHIP_OWNED,
                                borrow: BORROW_NONE,
                                invalid: true,
                            },
                        );
                    },
                }
            },
            Statement::Expression(expr) =>
            {
                let temps = self.expression(expr, false);
                self.release_temps(temps);
            },
            Statement::Scope(inner) =>
            {
                self.emit(SomToken::ScopeStart, NA_LABEL);
                self.scopes.push(ScopeFrame::default());
                for s in inner
                {
                    self.statement(s);
                }
                self.end_scope();
                self.emit(SomToken::ScopeEnd, NA_LABEL);
            },
            Statement::Return(expr) =>
            {
                if let Some(e) = expr
                {
                    let temps = self.expression(e, true);
                    self.release_temps(temps);
                }
                self.emit(SomToken::Return, NA_LABEL);
            },
        }
    }

    /// Interpret an expression, emitting its tokens and labels.
    /// Returns the borrows granted to the surrounding binding context.
    fn expression(&mut self, expr: &Expression, in_return: bool) -> Vec<(usize, bool)> {
        match expr
        {
            Expression::Literal(_) => Vec::new(),
            Expression::Variable(name) =>
            {
                match self.resolve(name)
                {
                    None =>
                    {
                        self.fault(name, FaultKind::UseOfUndeclared);
                        self.emit(
                            SomToken::Use(name.clone()),
                            TokenLabel {
                                ownership: OWNERSHIP_NA,
                                borrow: BORROW_NA,
                                invalid: true,
                            },
                        );
                    },
                    Some(id) =>
                    {
                        let mut invalid = false;
                        if self.vars[id].copyable
                        {
                            // Copy semantics: the value is duplicated, the
                            // variable stays usable. Only reading through an
                            // outstanding `&mut` borrow is a fault.
                            if self.vars[id].muted
                            {
                                self.fault(name, FaultKind::UseWhileMutBorrowed);
                                invalid = true;
                            }
                        }
                        else
                        {
                            match self.vars[id].state
                            {
                                VarState::Moved | VarState::Dropped =>
                                {
                                    self.fault(name, FaultKind::UseAfterMove);
                                    invalid = true;
                                },
                                VarState::Owned | VarState::Borrowed =>
                                {
                                    if self.is_borrowed(id)
                                    {
                                        self.fault(name, FaultKind::MoveWhileBorrowed);
                                        invalid = true;
                                    }
                                    self.release_holds(id);
                                    self.vars[id].state = VarState::Moved;
                                },
                            }
                        }
                        self.emit(
                            SomToken::Use(name.clone()),
                            TokenLabel {
                                ownership: self.state_label(id),
                                borrow: self.borrow_label(id),
                                invalid,
                            },
                        );
                    },
                }
                Vec::new()
            },
            Expression::Reference { name, mutable } =>
            {
                let token = if *mutable
                {
                    SomToken::MutRef(name.clone())
                }
                else
                {
                    SomToken::Ref(name.clone())
                };
                match self.resolve(name)
                {
                    None =>
                    {
                        self.fault(name, FaultKind::UseOfUndeclared);
                        self.emit(
                            token,
                            TokenLabel {
                                ownership: OWNERSHIP_NA,
                                borrow: BORROW_NA,
                                invalid: true,
                            },
                        );
                        Vec::new()
                    },
                    Some(id) =>
                    {
                        let mut invalid = false;
                        if matches!(self.vars[id].state, VarState::Moved | VarState::Dropped)
                        {
                            self.fault(name, FaultKind::BorrowOfMoved);
                            invalid = true;
                        }
                        if *mutable
                        {
                            if self.is_borrowed(id)
                            {
                                self.fault(name, FaultKind::BorrowConflict);
                                invalid = true;
                            }
                            self.vars[id].muted = true;
                        }
                        else
                        {
                            if self.vars[id].muted
                            {
                                self.fault(name, FaultKind::BorrowConflict);
                                invalid = true;
                            }
                            self.vars[id].shared += 1;
                        }
                        if self.vars[id].state == VarState::Owned
                        {
                            self.vars[id].state = VarState::Borrowed;
                        }
                        if in_return
                        {
                            self.fault(name, FaultKind::EscapingBorrow);
                            invalid = true;
                        }
                        self.emit(
                            token,
                            TokenLabel {
                                ownership: self.state_label(id),
                                borrow: self.borrow_label(id),
                                invalid,
                            },
                        );
                        vec![(id, *mutable)]
                    },
                }
            },
            Expression::BinaryOp { left, right, .. } =>
            {
                let mut temps = self.expression(left, in_return);
                temps.extend(self.expression(right, in_return));
                temps
            },
            Expression::Call { args, .. } =>
            {
                let mut temps = Vec::new();
                for arg in args
                {
                    temps.extend(self.expression(arg, in_return));
                }
                temps
            },
            Expression::Dereference(inner) => self.expression(inner, in_return),
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_som_pcg::ast::{BinaryOp, Literal, Type};
    use scirust_som_tokenizer::StructuredTokenizer;

    /// Declare an owner-typed (move-semantics) binding.
    fn decl_lit(name: &str) -> Statement {
        Statement::VarDecl {
            name: name.to_string(),
            ty: Type::Str,
            init: Some(Expression::Literal(Literal::Str("s".to_string()))),
        }
    }

    /// Declare a Copy-typed binding (`i64` semantics).
    fn decl_copy(name: &str) -> Statement {
        Statement::VarDecl {
            name: name.to_string(),
            ty: Type::Int,
            init: Some(Expression::Literal(Literal::Int(1))),
        }
    }

    fn decl_move(name: &str, from: &str) -> Statement {
        Statement::VarDecl {
            name: name.to_string(),
            ty: Type::Str,
            init: Some(Expression::Variable(from.to_string())),
        }
    }

    fn decl_ref(name: &str, of: &str, mutable: bool) -> Statement {
        Statement::VarDecl {
            name: name.to_string(),
            ty: Type::Ref(Box::new(Type::Int), mutable),
            init: Some(Expression::Reference {
                name: of.to_string(),
                mutable,
            }),
        }
    }

    fn program(body: Vec<Statement>) -> SomAst {
        SomAst::Program(vec![Function {
            name: "main".to_string(),
            params: vec![],
            body,
        }])
    }

    fn label_of<'a>(a: &'a Analysis, token: &SomToken) -> &'a TokenLabel {
        let i = a.tokens.iter().position(|t| t == token).expect("token");
        &a.labels[i]
    }

    #[test]
    fn use_after_move_is_flagged() {
        // let x = 1; let y = x; let z = x;
        let ast = program(vec![
            decl_lit("x"),
            decl_move("y", "x"),
            decl_move("z", "x"),
        ]);
        let a = OwnershipOracle::new().analyze(&ast);

        assert!(
            a.diagnostics
                .iter()
                .any(|d| d.kind == FaultKind::UseAfterMove && d.var == "x")
        );
        // First Use(x) is the legal move, second is the fault.
        let uses: Vec<usize> = a
            .tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t, SomToken::Use(n) if n == "x"))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(uses.len(), 2);
        assert!(!a.labels[uses[0]].invalid);
        assert_eq!(a.labels[uses[0]].ownership, OWNERSHIP_MOVED);
        assert!(a.labels[uses[1]].invalid);
        assert_eq!(a.labels[uses[1]].ownership, OWNERSHIP_MOVED);
    }

    #[test]
    fn borrow_rules_shared_ok_mut_conflicts() {
        // let x = 1; let r1 = &x; let r2 = &x; let m = &mut x;
        let ast = program(vec![
            decl_lit("x"),
            decl_ref("r1", "x", false),
            decl_ref("r2", "x", false),
            decl_ref("m", "x", true),
        ]);
        let a = OwnershipOracle::new().analyze(&ast);

        let conflicts: Vec<_> = a
            .diagnostics
            .iter()
            .filter(|d| d.kind == FaultKind::BorrowConflict)
            .collect();
        assert_eq!(conflicts.len(), 1, "only the &mut after two & conflicts");
        let mutref = label_of(&a, &SomToken::MutRef("x".into()));
        assert!(mutref.invalid);
        assert_eq!(mutref.borrow, BORROW_MUT);
        let r2 = label_of(&a, &SomToken::Ref("x".into()));
        assert_eq!(r2.ownership, OWNERSHIP_BORROWED);
    }

    #[test]
    fn scope_drop_labels() {
        // { let x = 1; }  → Drop(x) labelled Dropped
        // let y = 1; let z = y; → Drop(y) labelled Moved (no drop runs)
        let ast = program(vec![
            Statement::Scope(vec![decl_lit("x")]),
            decl_lit("y"),
            decl_move("z", "y"),
        ]);
        let a = OwnershipOracle::new().analyze(&ast);

        assert_eq!(
            label_of(&a, &SomToken::Drop("x".into())).ownership,
            OWNERSHIP_DROPPED
        );
        assert_eq!(
            label_of(&a, &SomToken::Drop("y".into())).ownership,
            OWNERSHIP_MOVED
        );
        assert_eq!(
            label_of(&a, &SomToken::Drop("z".into())).ownership,
            OWNERSHIP_DROPPED
        );
        assert!(a.diagnostics.is_empty());
    }

    #[test]
    fn reassignment_heals_moved() {
        // let x = 1; let y = x; x = 2; let z = x;  → no fault on last use
        let ast = program(vec![
            decl_lit("x"),
            decl_move("y", "x"),
            Statement::Assignment {
                lhs: "x".to_string(),
                rhs: Expression::Literal(Literal::Int(2)),
            },
            decl_move("z", "x"),
        ]);
        let a = OwnershipOracle::new().analyze(&ast);
        assert!(
            a.diagnostics.is_empty(),
            "unexpected diagnostics: {:?}",
            a.diagnostics
        );
        assert_eq!(
            label_of(&a, &SomToken::Assign("x".into())).ownership,
            OWNERSHIP_OWNED
        );
    }

    #[test]
    fn move_while_borrowed_and_escaping_borrow() {
        // let x = 1; let r = &x; let y = x;   → MoveWhileBorrowed
        // return &x;                          → EscapingBorrow
        let ast = program(vec![
            decl_lit("x"),
            decl_ref("r", "x", false),
            decl_move("y", "x"),
            Statement::Return(Some(Expression::Reference {
                name: "x".to_string(),
                mutable: false,
            })),
        ]);
        let a = OwnershipOracle::new().analyze(&ast);
        assert!(
            a.diagnostics
                .iter()
                .any(|d| d.kind == FaultKind::MoveWhileBorrowed)
        );
        assert!(
            a.diagnostics
                .iter()
                .any(|d| d.kind == FaultKind::EscapingBorrow)
        );
    }

    #[test]
    fn undeclared_use_and_assign() {
        let ast = program(vec![
            Statement::Expression(Expression::Variable("ghost".to_string())),
            Statement::Assignment {
                lhs: "w".to_string(),
                rhs: Expression::Literal(Literal::Int(0)),
            },
        ]);
        let a = OwnershipOracle::new().analyze(&ast);
        assert!(
            a.diagnostics
                .iter()
                .any(|d| d.kind == FaultKind::UseOfUndeclared)
        );
        assert!(
            a.diagnostics
                .iter()
                .any(|d| d.kind == FaultKind::AssignToUndeclared)
        );
        // The implicit declaration drops at function end.
        assert!(a.tokens.contains(&SomToken::Drop("w".into())));
    }

    #[test]
    fn binary_op_double_move_faults() {
        // let a = 1; let b = a + a; → second operand is use-after-move
        let ast = program(vec![
            decl_lit("a"),
            Statement::VarDecl {
                name: "b".to_string(),
                ty: Type::Str,
                init: Some(Expression::BinaryOp {
                    left: Box::new(Expression::Variable("a".to_string())),
                    op: BinaryOp::Add,
                    right: Box::new(Expression::Variable("a".to_string())),
                }),
            },
        ]);
        let a = OwnershipOracle::new().analyze(&ast);
        assert_eq!(
            a.diagnostics
                .iter()
                .filter(|d| d.kind == FaultKind::UseAfterMove)
                .count(),
            1
        );
    }

    #[test]
    fn tokens_align_with_tokenizer_stream() {
        let ast = program(vec![
            decl_lit("x"),
            decl_move("y", "x"),
            Statement::Scope(vec![
                decl_ref("r", "y", false),
                Statement::Expression(Expression::Variable("r".to_string())),
            ]),
            Statement::Assignment {
                lhs: "x".to_string(),
                rhs: Expression::Literal(Literal::Int(2)),
            },
            Statement::Return(Some(Expression::Variable("x".to_string()))),
        ]);
        let a = OwnershipOracle::new().analyze(&ast);
        let stream = StructuredTokenizer::new().tokenize_ast_with_drops(&ast);
        assert_eq!(a.tokens, stream, "oracle and tokenizer streams must match");
        assert_eq!(a.tokens.len(), a.labels.len());
    }

    #[test]
    fn analysis_is_deterministic() {
        let ast = program(vec![
            decl_lit("x"),
            decl_ref("r", "x", true),
            decl_move("y", "x"),
        ]);
        let a1 = OwnershipOracle::new().analyze(&ast);
        let a2 = OwnershipOracle::new().analyze(&ast);
        assert_eq!(format!("{a1:?}"), format!("{a2:?}"));
    }
    #[test]
    fn copy_types_are_not_moved() {
        // let a: i64 = 1; let b = a; let c = a;  → all legal, `a` stays Owned
        let ast = program(vec![
            decl_copy("a"),
            Statement::VarDecl {
                name: "b".to_string(),
                ty: Type::Str,
                init: Some(Expression::Variable("a".to_string())),
            },
            Statement::VarDecl {
                name: "c".to_string(),
                ty: Type::Str,
                init: Some(Expression::Variable("a".to_string())),
            },
        ]);
        let a = OwnershipOracle::new().analyze(&ast);
        assert!(
            a.diagnostics.is_empty(),
            "copy uses must not fault: {:?}",
            a.diagnostics
        );
        // Both uses of `a` are labelled Owned (copied, not moved).
        for (t, l) in a.tokens.iter().zip(&a.labels)
        {
            if matches!(t, SomToken::Use(n) if n == "a")
            {
                assert_eq!(l.ownership, OWNERSHIP_OWNED);
            }
        }
        // Unannotated `b`/`c` inherited Copy-ness from `a`.
        assert_eq!(
            label_of(&a, &SomToken::Drop("b".into())).ownership,
            OWNERSHIP_DROPPED
        );
    }

    #[test]
    fn copy_use_under_mut_borrow_faults() {
        // let a: i64 = 1; let m = &mut a; let b = a;  → E0503-style fault
        let ast = program(vec![
            decl_copy("a"),
            decl_ref("m", "a", true),
            Statement::VarDecl {
                name: "b".to_string(),
                ty: Type::Str,
                init: Some(Expression::Variable("a".to_string())),
            },
        ]);
        let a = OwnershipOracle::new().analyze(&ast);
        assert!(
            a.diagnostics
                .iter()
                .any(|d| d.kind == FaultKind::UseWhileMutBorrowed)
        );
    }

    #[test]
    fn copy_use_under_shared_borrow_is_legal() {
        // let a: i64 = 1; let r = &a; let b = a;  → legal in Rust
        let ast = program(vec![
            decl_copy("a"),
            decl_ref("r", "a", false),
            Statement::VarDecl {
                name: "b".to_string(),
                ty: Type::Str,
                init: Some(Expression::Variable("a".to_string())),
            },
        ]);
        let a = OwnershipOracle::new().analyze(&ast);
        assert!(
            a.diagnostics.is_empty(),
            "copy under shared borrow is legal: {:?}",
            a.diagnostics
        );
    }
}
