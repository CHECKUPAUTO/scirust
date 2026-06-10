//! Simplified AST for ownership and borrowing analysis.

#[derive(Debug, Clone, PartialEq)]
pub enum SomAst {
    Program(Vec<Function>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub params: Vec<Param>,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    VarDecl {
        name: String,
        ty: Type,
        init: Option<Expression>,
    },
    Assignment {
        lhs: String,
        rhs: Expression,
    },
    Expression(Expression),
    Scope(Vec<Statement>),
    Return(Option<Expression>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Literal(Literal),
    Variable(String),
    BinaryOp {
        left: Box<Expression>,
        op: BinaryOp,
        right: Box<Expression>,
    },
    Call {
        name: String,
        args: Vec<Expression>,
    },
    Reference {
        name: String,
        mutable: bool,
    },
    Dereference(Box<Expression>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Float,
    Bool,
    Str,
    Ref(Box<Type>, bool), // Type, is_mutable
    Ptr(Box<Type>),
    Unit,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
}
