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
    use scirust_som_pcg::ast::{BinaryOp, Literal, Param, Type};
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

    /// `let name: &mut T = &mut of;` — a `&mut` reference binding, which is
    /// itself a *move* type (so the binding can be moved on later).
    fn decl_mutref_binding(name: &str, of: &str) -> Statement {
        Statement::VarDecl {
            name: name.to_string(),
            ty: Type::Ref(Box::new(Type::Str), true),
            init: Some(Expression::Reference {
                name: of.to_string(),
                mutable: true,
            }),
        }
    }

    /// `expr;` — an expression statement (its borrows end with the statement).
    fn stmt_expr(e: Expression) -> Statement {
        Statement::Expression(e)
    }

    fn var(name: &str) -> Expression {
        Expression::Variable(name.to_string())
    }

    fn borrow(name: &str, mutable: bool) -> Expression {
        Expression::Reference {
            name: name.to_string(),
            mutable,
        }
    }

    /// All `Use(name)` token indices in evaluation order.
    fn uses_of(a: &Analysis, name: &str) -> Vec<usize> {
        a.tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t, SomToken::Use(n) if n == name))
            .map(|(i, _)| i)
            .collect()
    }

    fn count_faults(a: &Analysis, kind: FaultKind) -> usize {
        a.diagnostics.iter().filter(|d| d.kind == kind).count()
    }

    /// Assert that the token at `idx` has exactly this ownership/borrow/invalid
    /// triple — the strongest possible per-token oracle assertion.
    fn assert_label(a: &Analysis, idx: usize, own: usize, bor: usize, invalid: bool) {
        let l = &a.labels[idx];
        assert_eq!(
            (l.ownership, l.borrow, l.invalid),
            (own, bor, invalid),
            "token[{idx}] = {:?}: expected (own={}, borrow={}, invalid={}) got (own={}, borrow={}, invalid={})",
            a.tokens[idx],
            ownership_name(own),
            borrow_name(bor),
            invalid,
            ownership_name(l.ownership),
            borrow_name(l.borrow),
            l.invalid,
        );
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

    // -----------------------------------------------------------------
    // Hand-labelled oracle tests: every label on a small program is
    // derived by hand from Rust's borrow/move semantics, then the oracle
    // is asserted to reproduce exactly that label (not merely a fault
    // count). A divergence here means the oracle is wrong.
    // -----------------------------------------------------------------

    /// `let x = owner; let y = x; use(x);`
    ///   Hand-derived: the first `Use(x)` is the legal move (x → Moved); the
    ///   second `Use(x)` is use-after-move and must be flagged. The end-scope
    ///   `Drop(x)` is labelled Moved (no destructor runs on a moved binding).
    #[test]
    fn move_then_use_full_labels() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            decl_move("y", "x"),
            stmt_expr(var("x")),
        ]));
        // Tokens: FnDecl, VarDecl(x), Use(x)#move, VarDecl(y), Use(x)#fault,
        //         Drop(y), Drop(x)
        let uses = uses_of(&a, "x");
        assert_eq!(uses.len(), 2);
        assert_label(&a, uses[0], OWNERSHIP_MOVED, BORROW_NONE, false);
        assert_label(&a, uses[1], OWNERSHIP_MOVED, BORROW_NONE, true);
        assert_label(
            &a,
            a.tokens
                .iter()
                .position(|t| matches!(t, SomToken::Drop(n) if n == "x"))
                .unwrap(),
            OWNERSHIP_MOVED,
            BORROW_NONE,
            false,
        );
        assert_eq!(count_faults(&a, FaultKind::UseAfterMove), 1);
        // Exactly one diagnostic overall — the second use.
        assert_eq!(a.diagnostics.len(), 1);
    }

    /// A shared borrow taken inside an inner scope is released when that scope
    /// ends, so moving the (outer) owner afterwards is legal. This is the
    /// "borrow does not outlive its scope" case: the inner `Drop(r)` heals the
    /// owner back to `Owned`.
    #[test]
    fn borrow_released_at_inner_scope_end_then_move_ok() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            Statement::Scope(vec![decl_ref("r", "x", false)]),
            stmt_expr(var("x")),
        ]));
        assert!(
            a.diagnostics.is_empty(),
            "borrow ended with inner scope; move must be clean: {:?}",
            a.diagnostics
        );
        // The borrow token: x is Borrowed/Shared while r lives.
        assert_label(
            &a,
            a.tokens
                .iter()
                .position(|t| matches!(t, SomToken::Ref(n) if n == "x"))
                .unwrap(),
            OWNERSHIP_BORROWED,
            BORROW_SHARED,
            false,
        );
        // After the scope, the lone Use(x) is a clean move.
        let u = uses_of(&a, "x");
        assert_eq!(u.len(), 1);
        assert_label(&a, u[0], OWNERSHIP_MOVED, BORROW_NONE, false);
    }

    /// Moving an owner while a shared borrow is still outstanding is rejected
    /// by Rust (E0505). The move token must carry the live `Shared` borrow and
    /// be flagged.
    #[test]
    fn move_while_shared_borrowed_is_flagged() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            decl_ref("r", "x", false), // r holds &x for the rest of the scope
            decl_move("y", "x"),       // move x while borrowed -> E0505
        ]));
        let u = uses_of(&a, "x");
        assert_eq!(u.len(), 1);
        assert_label(&a, u[0], OWNERSHIP_MOVED, BORROW_SHARED, true);
        assert_eq!(count_faults(&a, FaultKind::MoveWhileBorrowed), 1);
    }

    /// Drops run in reverse declaration order. With three owners x, y, z the
    /// drop tokens must be Drop(z), Drop(y), Drop(x) — and each labelled
    /// Dropped.
    #[test]
    fn drop_order_is_reverse_declaration() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            decl_lit("y"),
            decl_lit("z"),
        ]));
        let drops: Vec<&str> = a
            .tokens
            .iter()
            .filter_map(|t| match t
            {
                SomToken::Drop(n) => Some(n.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(drops, vec!["z", "y", "x"]);
        for name in ["x", "y", "z"]
        {
            assert_eq!(
                label_of(&a, &SomToken::Drop(name.into())).ownership,
                OWNERSHIP_DROPPED
            );
        }
    }

    /// A binding that borrows a same-scope owner drops *before* the owner
    /// (reverse declaration order), releasing the borrow so the owner's own
    /// `Drop` is clean and labelled Dropped (never Borrowed).
    #[test]
    fn borrower_drops_before_target_clean() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            decl_ref("r", "x", true), // r: &mut x
        ]));
        let drops: Vec<&str> = a
            .tokens
            .iter()
            .filter_map(|t| match t
            {
                SomToken::Drop(n) => Some(n.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(drops, vec!["r", "x"], "borrower drops before its target");
        // x's drop is clean: the &mut was released by Drop(r).
        assert_label(
            &a,
            a.tokens
                .iter()
                .position(|t| matches!(t, SomToken::Drop(n) if n == "x"))
                .unwrap(),
            OWNERSHIP_DROPPED,
            BORROW_NONE,
            false,
        );
        assert!(a.diagnostics.is_empty());
    }

    /// `&x` then `&mut x` while the shared borrow is live is E0502. Only the
    /// `&mut` is at fault; the preceding `&x` is clean.
    #[test]
    fn shared_then_mut_conflict_exact() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            decl_ref("r", "x", false),
            decl_ref("m", "x", true),
        ]));
        assert_label(
            &a,
            a.tokens
                .iter()
                .position(|t| matches!(t, SomToken::Ref(n) if n == "x"))
                .unwrap(),
            OWNERSHIP_BORROWED,
            BORROW_SHARED,
            false,
        );
        // The &mut: conflict, and it reports the (now dominant) Mut borrow.
        assert_label(
            &a,
            a.tokens
                .iter()
                .position(|t| matches!(t, SomToken::MutRef(n) if n == "x"))
                .unwrap(),
            OWNERSHIP_BORROWED,
            BORROW_MUT,
            true,
        );
        assert_eq!(count_faults(&a, FaultKind::BorrowConflict), 1);
    }

    /// Two simultaneous `&mut x` borrows are E0499. The second is the fault.
    #[test]
    fn two_mut_borrows_conflict() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            decl_ref("m1", "x", true),
            decl_ref("m2", "x", true),
        ]));
        let mutrefs: Vec<usize> = a
            .tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t, SomToken::MutRef(n) if n == "x"))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(mutrefs.len(), 2);
        assert!(!a.labels[mutrefs[0]].invalid, "first &mut is clean");
        assert!(a.labels[mutrefs[1]].invalid, "second &mut conflicts");
        assert_eq!(count_faults(&a, FaultKind::BorrowConflict), 1);
    }

    /// Two temporary shared borrows in one call (`f(&x, &x)`) are both legal;
    /// they end with the statement, so a following move of the owner is clean.
    #[test]
    fn two_shared_temps_then_move_ok() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            stmt_expr(Expression::Call {
                name: "f".to_string(),
                args: vec![borrow("x", false), borrow("x", false)],
            }),
            stmt_expr(var("x")), // temps released -> clean move
        ]));
        assert!(
            a.diagnostics.is_empty(),
            "two shared temps then move is clean: {:?}",
            a.diagnostics
        );
        let u = uses_of(&a, "x");
        assert_eq!(u.len(), 1);
        assert_label(&a, u[0], OWNERSHIP_MOVED, BORROW_NONE, false);
    }

    /// `f(&x, &mut x)` holds a shared and a mutable borrow simultaneously
    /// (within one statement) — E0502. The `&mut` argument is the fault.
    #[test]
    fn shared_and_mut_in_one_call_conflict() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            stmt_expr(Expression::Call {
                name: "f".to_string(),
                args: vec![borrow("x", false), borrow("x", true)],
            }),
        ]));
        assert_label(
            &a,
            a.tokens
                .iter()
                .position(|t| matches!(t, SomToken::MutRef(n) if n == "x"))
                .unwrap(),
            OWNERSHIP_BORROWED,
            BORROW_MUT,
            true,
        );
        assert_eq!(count_faults(&a, FaultKind::BorrowConflict), 1);
    }

    /// Returning a *reference* to a local escapes (E0515-style); returning a
    /// *value* by move does not. The two cases must be distinguished.
    #[test]
    fn return_reference_escapes_but_value_does_not() {
        // return &x;  -> EscapingBorrow on the Ref token.
        let esc = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            Statement::Return(Some(borrow("x", false))),
        ]));
        assert_label(
            &esc,
            esc.tokens
                .iter()
                .position(|t| matches!(t, SomToken::Ref(n) if n == "x"))
                .unwrap(),
            OWNERSHIP_BORROWED,
            BORROW_SHARED,
            true,
        );
        assert_eq!(count_faults(&esc, FaultKind::EscapingBorrow), 1);

        // return x;  -> clean move out, no escaping borrow.
        let mv = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            Statement::Return(Some(var("x"))),
        ]));
        assert!(
            mv.diagnostics.is_empty(),
            "returning a value is a clean move: {:?}",
            mv.diagnostics
        );
        let u = uses_of(&mv, "x");
        assert_eq!(u.len(), 1);
        assert_label(&mv, u[0], OWNERSHIP_MOVED, BORROW_NONE, false);
    }

    /// An escaping borrow nested inside a returned expression is still caught
    /// (the `in_return` flag propagates through `BinaryOp`).
    #[test]
    fn escaping_borrow_through_binary_op_in_return() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            Statement::Return(Some(Expression::BinaryOp {
                left: Box::new(borrow("x", false)),
                op: BinaryOp::Eq,
                right: Box::new(borrow("x", false)),
            })),
        ]));
        assert_eq!(
            count_faults(&a, FaultKind::EscapingBorrow),
            2,
            "both nested references escape"
        );
    }

    /// Reassignment re-initialises a moved binding: after `x = lit;` the
    /// binding is `Owned` again and may be borrowed/used without fault. The
    /// `Assign` token itself is clean and labelled Owned.
    #[test]
    fn reassign_heals_then_borrow_is_clean() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            decl_move("y", "x"), // x -> Moved
            Statement::Assignment {
                lhs: "x".to_string(),
                rhs: Expression::Literal(Literal::Str("t".to_string())),
            },
            decl_ref("r", "x", false), // legal: x healed to Owned
        ]));
        assert!(
            a.diagnostics.is_empty(),
            "reassign should heal the move: {:?}",
            a.diagnostics
        );
        assert_label(
            &a,
            a.tokens
                .iter()
                .position(|t| matches!(t, SomToken::Assign(n) if n == "x"))
                .unwrap(),
            OWNERSHIP_OWNED,
            BORROW_NONE,
            false,
        );
        assert_label(
            &a,
            a.tokens
                .iter()
                .position(|t| matches!(t, SomToken::Ref(n) if n == "x"))
                .unwrap(),
            OWNERSHIP_BORROWED,
            BORROW_SHARED,
            false,
        );
    }

    /// Assigning to a variable while it is borrowed is E0506. The `Assign`
    /// token is flagged and the variable stays `Borrowed`.
    #[test]
    fn assign_while_borrowed_is_flagged() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            decl_ref("r", "x", false), // r borrows x for the scope
            Statement::Assignment {
                lhs: "x".to_string(),
                rhs: Expression::Literal(Literal::Str("t".to_string())),
            },
        ]));
        assert_label(
            &a,
            a.tokens
                .iter()
                .position(|t| matches!(t, SomToken::Assign(n) if n == "x"))
                .unwrap(),
            OWNERSHIP_BORROWED,
            BORROW_SHARED,
            true,
        );
        assert_eq!(count_faults(&a, FaultKind::AssignWhileBorrowed), 1);
    }

    /// A binding whose RHS *is a borrow* holds that borrow until it drops, so
    /// the borrowed owner is rejected from being moved meanwhile (E0505), and
    /// the owner heals once the holder drops at scope end.
    #[test]
    fn assignment_rhs_borrow_is_held_by_binding() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            decl_lit("p"), // p starts as an owner
            Statement::Assignment {
                lhs: "p".to_string(),
                rhs: borrow("x", false), // p now holds &x
            },
            decl_move("y", "x"), // move x while p borrows it -> E0505
        ]));
        let u = uses_of(&a, "x");
        assert_eq!(u.len(), 1);
        assert_label(&a, u[0], OWNERSHIP_MOVED, BORROW_SHARED, true);
        assert_eq!(count_faults(&a, FaultKind::MoveWhileBorrowed), 1);
        // The borrow itself (in the assignment RHS) was clean.
        assert_label(
            &a,
            a.tokens
                .iter()
                .position(|t| matches!(t, SomToken::Ref(n) if n == "x"))
                .unwrap(),
            OWNERSHIP_BORROWED,
            BORROW_SHARED,
            false,
        );
    }

    /// Shadowing: an inner-scope binding with the same name as a moved outer
    /// owner is a *fresh* binding. Using the inner name is legal; the inner
    /// binding's own `Drop` reflects its own state; after the scope, the outer
    /// (still moved) binding faults on use.
    #[test]
    fn inner_shadow_does_not_heal_outer() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            decl_move("y", "x"), // outer x -> Moved
            Statement::Scope(vec![
                decl_lit("x"),       // fresh inner x (Owned)
                decl_move("z", "x"), // moves inner x (legal)
            ]),
            stmt_expr(var("x")), // refers to OUTER x (still Moved) -> fault
        ]));
        // Three Use(x) tokens, in order:
        //   [a] outer `let y = x`  -> legal move of the outer x   (clean)
        //   [b] inner `let z = x`  -> legal move of the inner x   (clean)
        //   [c] post-scope use     -> use-after-move on outer x   (fault)
        let u = uses_of(&a, "x");
        assert_eq!(u.len(), 3);
        assert_label(&a, u[0], OWNERSHIP_MOVED, BORROW_NONE, false);
        assert_label(&a, u[1], OWNERSHIP_MOVED, BORROW_NONE, false);
        assert_label(&a, u[2], OWNERSHIP_MOVED, BORROW_NONE, true);
        // Exactly one use-after-move overall: only the post-scope outer use.
        assert_eq!(count_faults(&a, FaultKind::UseAfterMove), 1);
        assert_eq!(a.diagnostics.len(), 1);
    }

    /// Copy types are never moved: `let a: i64 = 1; let b = a; let c = a;` is
    /// fully legal, and every `Use(a)` stays `Owned` (value copied).
    #[test]
    fn copy_value_uses_stay_owned() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_copy("a"),
            decl_move("b", "a"), // unannotated; infers Copy-ness from `a`
            decl_move("c", "a"),
        ]));
        assert!(
            a.diagnostics.is_empty(),
            "copy uses never fault: {:?}",
            a.diagnostics
        );
        for i in uses_of(&a, "a")
        {
            assert_label(&a, i, OWNERSHIP_OWNED, BORROW_NONE, false);
        }
        // `a` is Copy, so its end-scope Drop is a real Dropped (Copy values do
        // run their — trivial — drop glue and were never moved out).
        assert_eq!(
            label_of(&a, &SomToken::Drop("a".into())).ownership,
            OWNERSHIP_DROPPED
        );
    }

    /// Reading a Copy value while it is mutably borrowed is E0503; reading it
    /// while only *shared*-borrowed is legal. Both directions asserted.
    #[test]
    fn copy_read_under_mut_vs_shared() {
        // &mut a then read a -> fault, read reports the live Mut borrow.
        let bad = OwnershipOracle::new().analyze(&program(vec![
            decl_copy("a"),
            decl_ref("m", "a", true),
            decl_move("b", "a"), // copy-read while &mut live
        ]));
        let u = uses_of(&bad, "a");
        assert_eq!(u.len(), 1);
        assert_label(&bad, u[0], OWNERSHIP_BORROWED, BORROW_MUT, true);
        assert_eq!(count_faults(&bad, FaultKind::UseWhileMutBorrowed), 1);

        // &a then read a -> legal.
        let ok = OwnershipOracle::new().analyze(&program(vec![
            decl_copy("a"),
            decl_ref("r", "a", false),
            decl_move("b", "a"),
        ]));
        assert!(
            ok.diagnostics.is_empty(),
            "copy read under shared borrow is legal: {:?}",
            ok.diagnostics
        );
        let u = uses_of(&ok, "a");
        assert_eq!(u.len(), 1);
        assert_label(&ok, u[0], OWNERSHIP_BORROWED, BORROW_SHARED, false);
    }

    /// Borrowing or dereferencing a moved value is a fault (use-after-move /
    /// borrow-of-moved), and the moved owner stays `Moved` throughout.
    #[test]
    fn borrow_and_deref_of_moved_are_faults() {
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            decl_move("y", "x"),                                    // x -> Moved
            stmt_expr(Expression::Dereference(Box::new(var("x")))), // use-after-move
            stmt_expr(borrow("x", false)),                          // borrow-of-moved
        ]));
        assert_eq!(count_faults(&a, FaultKind::UseAfterMove), 1);
        assert_eq!(count_faults(&a, FaultKind::BorrowOfMoved), 1);
        // The deref's inner Use(x) is flagged and stays Moved.
        let faulting_use = uses_of(&a, "x")
            .into_iter()
            .find(|&i| a.labels[i].invalid)
            .unwrap();
        assert_label(&a, faulting_use, OWNERSHIP_MOVED, BORROW_NONE, true);
        // The borrow-of-moved token also stays Moved.
        assert_label(
            &a,
            a.tokens
                .iter()
                .position(|t| matches!(t, SomToken::Ref(n) if n == "x"))
                .unwrap(),
            OWNERSHIP_MOVED,
            BORROW_SHARED,
            true,
        );
    }

    /// A function parameter of owner type is `Owned` on entry; using it twice
    /// moves then use-after-moves it, and its `Drop` at function end is Moved.
    #[test]
    fn owner_param_double_use_after_move() {
        let ast = SomAst::Program(vec![Function {
            name: "f".to_string(),
            params: vec![Param {
                name: "p".to_string(),
                ty: Type::Str,
            }],
            body: vec![decl_move("a", "p"), decl_move("b", "p")],
        }]);
        let a = OwnershipOracle::new().analyze(&ast);
        assert_label(
            &a,
            a.tokens
                .iter()
                .position(|t| matches!(t, SomToken::Param(n) if n == "p"))
                .unwrap(),
            OWNERSHIP_OWNED,
            BORROW_NONE,
            false,
        );
        let u = uses_of(&a, "p");
        assert_eq!(u.len(), 2);
        assert_label(&a, u[0], OWNERSHIP_MOVED, BORROW_NONE, false);
        assert_label(&a, u[1], OWNERSHIP_MOVED, BORROW_NONE, true);
        assert_eq!(count_faults(&a, FaultKind::UseAfterMove), 1);
        assert_eq!(
            label_of(&a, &SomToken::Drop("p".into())).ownership,
            OWNERSHIP_MOVED
        );
    }

    /// Moving the *holder* of a borrow releases that borrow under the oracle's
    /// documented model ("released when it drops, **moves** or is reassigned",
    /// see the module contract). `&mut`-reference bindings are move types, so
    /// `let s = r;` ends r's borrow on its target. This pins the documented
    /// (intentionally reborrow-transfer-free) behaviour.
    #[test]
    fn moving_a_reference_binding_releases_its_borrow() {
        // let x = owner; let r = &mut x; let s = r;
        // `r` is a `&mut` binding (a move type), and `let s = r;` moves it.
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            decl_mutref_binding("r", "x"), // r: &mut x  (move type)
            Statement::VarDecl {
                name: "s".to_string(),
                ty: Type::Ref(Box::new(Type::Str), true),
                init: Some(var("r")), // move r into s
            },
        ]));
        // After moving r, the documented model releases r's &mut on x, so x is
        // back to Owned and its end-scope Drop is clean (not Borrowed).
        assert_label(
            &a,
            a.tokens
                .iter()
                .position(|t| matches!(t, SomToken::Drop(n) if n == "x"))
                .unwrap(),
            OWNERSHIP_DROPPED,
            BORROW_NONE,
            false,
        );
        // r was moved out, so its Drop is labelled Moved.
        assert_eq!(
            label_of(&a, &SomToken::Drop("r".into())).ownership,
            OWNERSHIP_MOVED
        );
    }

    /// Full per-token lock-down of a representative mixed program, asserting
    /// the entire (ownership, borrow, invalid) vector. This is the strongest
    /// regression guard: any change to a single transition trips it.
    #[test]
    fn full_program_label_vector_locked() {
        // let x = owner;         // VarDecl(x)               Owned/None
        // let y = x;             // Use(x) move; VarDecl(y)
        // { let r = &x; }        // (x already moved) Ref(x) borrow-of-moved!
        // To keep the program clean, borrow y instead:
        // let x; let y = x; { let r = &y; r; } return; (+ drops)
        let a = OwnershipOracle::new().analyze(&program(vec![
            decl_lit("x"),
            decl_move("y", "x"),
            Statement::Scope(vec![decl_ref("r", "y", false), stmt_expr(var("r"))]),
            Statement::Return(None),
        ]));
        // Hand-derived label vector, token by token:
        let expected: Vec<(SomToken, usize, usize, bool)> = vec![
            (
                SomToken::FnDecl("main".into()),
                OWNERSHIP_NA,
                BORROW_NA,
                false,
            ),
            (
                SomToken::VarDecl("x".into()),
                OWNERSHIP_OWNED,
                BORROW_NONE,
                false,
            ),
            // RHS of `let y = x` moves x:
            (
                SomToken::Use("x".into()),
                OWNERSHIP_MOVED,
                BORROW_NONE,
                false,
            ),
            (
                SomToken::VarDecl("y".into()),
                OWNERSHIP_OWNED,
                BORROW_NONE,
                false,
            ),
            (SomToken::ScopeStart, OWNERSHIP_NA, BORROW_NA, false),
            // &y -> y becomes Borrowed/Shared:
            (
                SomToken::Ref("y".into()),
                OWNERSHIP_BORROWED,
                BORROW_SHARED,
                false,
            ),
            (
                SomToken::VarDecl("r".into()),
                OWNERSHIP_OWNED,
                BORROW_NONE,
                false,
            ),
            // r is a Copy (&T) binding: using it copies, r stays Owned:
            (
                SomToken::Use("r".into()),
                OWNERSHIP_OWNED,
                BORROW_NONE,
                false,
            ),
            // inner scope ends: Drop(r) releases &y; r itself Dropped:
            (
                SomToken::Drop("r".into()),
                OWNERSHIP_DROPPED,
                BORROW_NONE,
                false,
            ),
            (SomToken::ScopeEnd, OWNERSHIP_NA, BORROW_NA, false),
            (SomToken::Return, OWNERSHIP_NA, BORROW_NA, false),
            // function scope drops y (now healed to Owned -> Dropped) then x
            // (moved out -> Moved):
            (
                SomToken::Drop("y".into()),
                OWNERSHIP_DROPPED,
                BORROW_NONE,
                false,
            ),
            (
                SomToken::Drop("x".into()),
                OWNERSHIP_MOVED,
                BORROW_NONE,
                false,
            ),
        ];
        assert_eq!(
            a.tokens.len(),
            expected.len(),
            "token count mismatch: {:?}",
            a.tokens
        );
        for (i, (tok, own, bor, inv)) in expected.into_iter().enumerate()
        {
            assert_eq!(a.tokens[i], tok, "token[{i}] identity");
            assert_label(&a, i, own, bor, inv);
        }
        assert!(
            a.diagnostics.is_empty(),
            "clean program: {:?}",
            a.diagnostics
        );
    }
}
