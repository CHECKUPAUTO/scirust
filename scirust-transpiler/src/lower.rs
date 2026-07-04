//! Lower the Python subset AST into the typed [`crate::sir`] IR.
//!
//! This performs type/shape inference (scalar vs 1-D array vs integer index),
//! resolves NumPy intrinsics to SIR nodes, and enforces the subset contract —
//! refusing constructs it cannot prove correct rather than guessing.

use crate::front_python::ast::*;
use crate::sir::*;
use std::collections::HashMap;

pub fn lower_module(m: &PyModule) -> Result<SirModule, String> {
    let funcs = m
        .funcs
        .iter()
        .map(lower_func)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SirModule { funcs })
}

fn lower_func(f: &PyFunc) -> Result<SirFunc, String> {
    let mut env: HashMap<String, Ty> = HashMap::new();
    let mut params = Vec::new();
    for p in &f.params
    {
        let ty = match p.hint
        {
            Some(TypeHint::Array) => Ty::Array,
            Some(TypeHint::Int) => Ty::Int,
            Some(TypeHint::Float) => Ty::Scalar,
            None => infer_param_ty(&p.name, &f.body),
        };
        env.insert(p.name.clone(), ty);
        params.push((p.name.clone(), ty));
    }

    let mut declared: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
    let mut ret: Option<Ty> = None;
    let body = lower_block(&f.body, &mut env, &mut declared, true, &mut ret)?;

    let ret = match (f.ret_hint, ret)
    {
        (Some(TypeHint::Array), _) => Ty::Array,
        (Some(TypeHint::Float), _) | (Some(TypeHint::Int), _) => Ty::Scalar,
        (None, Some(t)) =>
        {
            if t == Ty::Int
            {
                Ty::Scalar
            }
            else
            {
                t
            }
        },
        (None, None) => Ty::Scalar,
    };

    Ok(SirFunc {
        name: f.name.clone(),
        params,
        ret,
        body,
    })
}

fn lower_block(
    stmts: &[PyStmt],
    env: &mut HashMap<String, Ty>,
    declared: &mut Vec<String>,
    top_level: bool,
    ret: &mut Option<Ty>,
) -> Result<Vec<SirStmt>, String> {
    let mut out = Vec::new();
    for s in stmts
    {
        out.push(lower_stmt(s, env, declared, top_level, ret)?);
    }
    Ok(out)
}

fn lower_stmt(
    s: &PyStmt,
    env: &mut HashMap<String, Ty>,
    declared: &mut Vec<String>,
    top_level: bool,
    ret: &mut Option<Ty>,
) -> Result<SirStmt, String> {
    match s
    {
        PyStmt::Assign { target, value } =>
        {
            let v = lower_scalar(value, env)?;
            let ty = normalize_ty(v.ty());
            if declared.contains(target)
            {
                // Re-assignment: type must be consistent.
                let prev = *env.get(target).unwrap();
                if normalize_ty(prev) != ty
                {
                    return Err(format!(
                        "variable `{}` reassigned with a different type ({:?} vs {:?})",
                        target, prev, ty
                    ));
                }
                Ok(SirStmt::Reassign {
                    name: target.clone(),
                    value: v,
                })
            }
            else
            {
                if !top_level
                {
                    return Err(format!(
                        "`{}` is first assigned inside a loop; initialise it before the loop \
                         (Python's function scope requires a hoisted binding)",
                        target
                    ));
                }
                env.insert(target.clone(), ty);
                declared.push(target.clone());
                Ok(SirStmt::Let {
                    name: target.clone(),
                    ty,
                    value: v,
                })
            }
        },
        PyStmt::AssignIndex {
            target,
            index,
            value,
        } =>
        {
            match env.get(target)
            {
                Some(Ty::Array) =>
                {},
                Some(other) =>
                {
                    return Err(format!(
                        "cannot index-assign into non-array `{}` ({:?})",
                        target, other
                    ));
                },
                None =>
                {
                    return Err(format!(
                        "`{}` is not defined before index-assignment",
                        target
                    ));
                },
            }
            let idx = lower_int(index, env)?;
            let v = lower_scalar(value, env)?;
            Ok(SirStmt::SetIndex {
                name: target.clone(),
                index: idx,
                value: v,
            })
        },
        PyStmt::For {
            var,
            start,
            end,
            body,
        } =>
        {
            let start = lower_int(start, env)?;
            let end = lower_int(end, env)?;
            let had = env.insert(var.clone(), Ty::Int);
            let body = lower_block(body, env, declared, false, ret)?;
            match had
            {
                Some(t) =>
                {
                    env.insert(var.clone(), t);
                },
                None =>
                {
                    env.remove(var);
                },
            }
            Ok(SirStmt::For {
                var: var.clone(),
                start,
                end,
                body,
            })
        },
        PyStmt::If { cond, then, els } =>
        {
            let cond = lower_condition(cond, env)?;
            // Branches are nested scopes: they may reassign already-declared
            // names or `return`, but cannot first-declare a name expected to
            // survive the `if` (same rule as loops — hoist before the `if`).
            let then = lower_block(then, env, declared, false, ret)?;
            let els = lower_block(els, env, declared, false, ret)?;
            Ok(SirStmt::If { cond, then, els })
        },
        PyStmt::While { cond, body } =>
        {
            let cond = lower_condition(cond, env)?;
            let body = lower_block(body, env, declared, false, ret)?;
            Ok(SirStmt::While { cond, body })
        },
        PyStmt::Return(Some(e)) =>
        {
            let v = lower_scalar(e, env)?;
            let t = normalize_ty(v.ty());
            merge_ret(ret, t)?;
            Ok(SirStmt::Return(v))
        },
        PyStmt::Return(None) =>
        {
            Err("bare `return` (no value) is not supported in this subset".into())
        },
    }
}

fn merge_ret(ret: &mut Option<Ty>, t: Ty) -> Result<(), String> {
    match ret
    {
        None =>
        {
            *ret = Some(t);
            Ok(())
        },
        Some(prev) if *prev == t => Ok(()),
        Some(prev) => Err(format!(
            "function returns inconsistent types ({:?} vs {:?})",
            prev, t
        )),
    }
}

/// `Int` is a valid scalar once it leaves the index domain; the emitter coerces.
fn normalize_ty(t: Ty) -> Ty {
    match t
    {
        Ty::Int => Ty::Scalar,
        other => other,
    }
}

// ---- integer (index/range/length) domain ----------------------------------

fn lower_int(e: &PyExpr, env: &HashMap<String, Ty>) -> Result<SirExpr, String> {
    match e
    {
        PyExpr::Int(v) => Ok(SirExpr::IntLit(*v)),
        PyExpr::Name(n) => match env.get(n)
        {
            Some(Ty::Int) => Ok(SirExpr::Var {
                name: n.clone(),
                ty: Ty::Int,
            }),
            Some(other) => Err(format!(
                "`{}` ({:?}) used where an integer index/length is required",
                n, other
            )),
            None => Err(format!("undefined name `{}` in integer context", n)),
        },
        PyExpr::Bin { op, l, r } =>
        {
            let op = int_op(*op)?;
            Ok(SirExpr::IntBin {
                op,
                l: Box::new(lower_int(l, env)?),
                r: Box::new(lower_int(r, env)?),
            })
        },
        PyExpr::Call { func, args } =>
        {
            if func == "len"
            {
                need_args(func, args, 1)?;
                let a = lower_scalar(&args[0], env)?;
                expect_array(&a, "len")?;
                Ok(SirExpr::Len(Box::new(a)))
            }
            else
            {
                Err(format!(
                    "call `{}` is not valid in an integer index/length context",
                    func
                ))
            }
        },
        other => Err(format!("unsupported integer expression: {:?}", other)),
    }
}

fn int_op(op: BinOp) -> Result<Op, String> {
    match op
    {
        BinOp::Add => Ok(Op::Add),
        BinOp::Sub => Ok(Op::Sub),
        BinOp::Mul => Ok(Op::Mul),
        BinOp::Div => Ok(Op::Div),
        BinOp::Pow | BinOp::MatMul =>
        {
            Err("`**`/`@` are not supported in an integer index context".into())
        },
    }
}

// ---- scalar / array (numeric) domain --------------------------------------

fn lower_scalar(e: &PyExpr, env: &HashMap<String, Ty>) -> Result<SirExpr, String> {
    match e
    {
        PyExpr::Float(v) => Ok(SirExpr::ScalarLit(*v)),
        PyExpr::Int(v) => Ok(SirExpr::ScalarLit(*v as f64)),
        PyExpr::Name(n) => match env.get(n)
        {
            Some(ty) => Ok(SirExpr::Var {
                name: n.clone(),
                ty: *ty,
            }),
            None => Err(format!("undefined name `{}`", n)),
        },
        PyExpr::Neg(inner) =>
        {
            let v = lower_scalar(inner, env)?;
            match v.ty()
            {
                Ty::Array => Ok(SirExpr::ScalarBroadcast {
                    op: Op::Mul,
                    scalar: Box::new(SirExpr::ScalarLit(-1.0)),
                    arr: Box::new(v),
                    arr_is_left: false,
                }),
                _ => Ok(SirExpr::ScalarNeg(Box::new(v))),
            }
        },
        PyExpr::Index { base, index } =>
        {
            let b = lower_scalar(base, env)?;
            expect_array(&b, "index")?;
            let idx = lower_int(index, env)?;
            Ok(SirExpr::Index {
                base: Box::new(b),
                idx: Box::new(idx),
            })
        },
        PyExpr::Bin { op, l, r } => lower_bin(*op, l, r, env),
        PyExpr::Call { func, args } => lower_call(func, args, env),
        PyExpr::Cmp { .. } => Err("a comparison is only valid as an `if`/`elif` condition".into()),
    }
}

/// Lower a boolean condition: a single scalar comparison.
fn lower_condition(e: &PyExpr, env: &HashMap<String, Ty>) -> Result<SirExpr, String> {
    match e
    {
        PyExpr::Cmp { op, l, r } =>
        {
            let lv = lower_scalar(l, env)?;
            let rv = lower_scalar(r, env)?;
            if lv.ty() == Ty::Array || rv.ty() == Ty::Array
            {
                return Err("array comparisons are not supported in conditions".into());
            }
            Ok(SirExpr::Cmp {
                op: cmp_op(*op),
                l: Box::new(lv),
                r: Box::new(rv),
            })
        },
        _ => Err("`if`/`elif` condition must be a comparison".into()),
    }
}

fn cmp_op(op: crate::front_python::ast::CmpOp) -> crate::sir::CmpOp {
    use crate::front_python::ast::CmpOp as A;
    use crate::sir::CmpOp as S;
    match op
    {
        A::Lt => S::Lt,
        A::Le => S::Le,
        A::Gt => S::Gt,
        A::Ge => S::Ge,
        A::Eq => S::Eq,
        A::Ne => S::Ne,
    }
}

fn lower_bin(
    op: BinOp,
    l: &PyExpr,
    r: &PyExpr,
    env: &HashMap<String, Ty>,
) -> Result<SirExpr, String> {
    if op == BinOp::Pow
    {
        let base = lower_scalar(l, env)?;
        let exp = lower_scalar(r, env)?;
        if base.ty() == Ty::Array || exp.ty() == Ty::Array
        {
            return Err("`**` on arrays is not supported in this subset".into());
        }
        return Ok(SirExpr::ScalarPow {
            base: Box::new(base),
            exp: Box::new(exp),
        });
    }
    if op == BinOp::MatMul
    {
        // `A @ b` : matrix-vector product (Matrix @ Array -> Array).
        let a = lower_scalar(l, env)?;
        let b = lower_scalar(r, env)?;
        if a.ty() != Ty::Matrix
        {
            return Err("`@` expects a 2-D matrix on the left".into());
        }
        expect_array(&b, "`@`")?;
        return Ok(SirExpr::Matvec {
            a: Box::new(a),
            b: Box::new(b),
        });
    }
    let sop = num_op(op);
    let lv = lower_scalar(l, env)?;
    let rv = lower_scalar(r, env)?;
    let la = lv.ty() == Ty::Array;
    let ra = rv.ty() == Ty::Array;
    match (la, ra)
    {
        (false, false) => Ok(SirExpr::ScalarBin {
            op: sop,
            l: Box::new(lv),
            r: Box::new(rv),
        }),
        (true, true) => Ok(SirExpr::EwBin {
            op: sop,
            l: Box::new(lv),
            r: Box::new(rv),
        }),
        (true, false) => Ok(SirExpr::ScalarBroadcast {
            op: sop,
            scalar: Box::new(rv),
            arr: Box::new(lv),
            arr_is_left: true,
        }),
        (false, true) => Ok(SirExpr::ScalarBroadcast {
            op: sop,
            scalar: Box::new(lv),
            arr: Box::new(rv),
            arr_is_left: false,
        }),
    }
}

fn num_op(op: BinOp) -> Op {
    match op
    {
        BinOp::Add => Op::Add,
        BinOp::Sub => Op::Sub,
        BinOp::Mul => Op::Mul,
        BinOp::Div => Op::Div,
        BinOp::Pow | BinOp::MatMul => unreachable!(),
    }
}

fn lower_call(func: &str, args: &[PyExpr], env: &HashMap<String, Ty>) -> Result<SirExpr, String> {
    let base = strip_np(func);
    match base
    {
        "sum" =>
        {
            need_args(func, args, 1)?;
            let a = lower_scalar(&args[0], env)?;
            expect_array(&a, "np.sum")?;
            Ok(SirExpr::Sum(Box::new(a)))
        },
        "dot" =>
        {
            need_args(func, args, 2)?;
            let a = lower_scalar(&args[0], env)?;
            let b = lower_scalar(&args[1], env)?;
            expect_array(&a, "np.dot")?;
            expect_array(&b, "np.dot")?;
            Ok(SirExpr::Dot(Box::new(a), Box::new(b)))
        },
        "linalg.solve" =>
        {
            // np.linalg.solve(A, b): A an n×n matrix, b an n vector.
            // Routed to the verified LU solver in `scirust-solvers`.
            need_args(func, args, 2)?;
            let a = lower_scalar(&args[0], env)?;
            let b = lower_scalar(&args[1], env)?;
            if a.ty() != Ty::Matrix
            {
                return Err("np.linalg.solve expects a 2-D matrix as its first \
                            argument (hint it as `A: np.ndarray` and use it only \
                            as solve's matrix)"
                    .into());
            }
            expect_array(&b, "np.linalg.solve")?;
            Ok(SirExpr::LinSolve {
                a: Box::new(a),
                b: Box::new(b),
            })
        },
        "linalg.det" =>
        {
            // np.linalg.det(A): determinant of an n×n matrix, routed to
            // `scirust-solvers` (LU-based).
            need_args(func, args, 1)?;
            let a = lower_scalar(&args[0], env)?;
            if a.ty() != Ty::Matrix
            {
                return Err("np.linalg.det expects a 2-D matrix argument".into());
            }
            Ok(SirExpr::Det(Box::new(a)))
        },
        "linalg.eigvalsh" =>
        {
            // np.linalg.eigvalsh(A): eigenvalues of a symmetric matrix (sorted
            // ascending), routed to `scirust-solvers::eigen_symmetric`.
            need_args(func, args, 1)?;
            let a = lower_scalar(&args[0], env)?;
            if a.ty() != Ty::Matrix
            {
                return Err("np.linalg.eigvalsh expects a 2-D matrix argument".into());
            }
            Ok(SirExpr::Eigvalsh(Box::new(a)))
        },
        "fft.fft" =>
        {
            // np.fft.fft(x): full complex DFT of a real signal.
            need_args(func, args, 1)?;
            let a = lower_scalar(&args[0], env)?;
            expect_array(&a, "np.fft.fft")?;
            Ok(SirExpr::Fft(Box::new(a)))
        },
        "fft.rfft" =>
        {
            // np.fft.rfft(x): real FFT (positive-frequency half spectrum).
            need_args(func, args, 1)?;
            let a = lower_scalar(&args[0], env)?;
            expect_array(&a, "np.fft.rfft")?;
            Ok(SirExpr::Rfft(Box::new(a)))
        },
        "fft.ifft" =>
        {
            // np.fft.ifft(c): inverse DFT of a complex spectrum.
            need_args(func, args, 1)?;
            let a = lower_scalar(&args[0], env)?;
            if a.ty() != Ty::ComplexArray
            {
                return Err("np.fft.ifft expects a complex array (e.g. the result \
                            of np.fft.fft)"
                    .into());
            }
            Ok(SirExpr::Ifft(Box::new(a)))
        },
        "zeros" =>
        {
            need_args(func, args, 1)?;
            Ok(SirExpr::Zeros(Box::new(lower_int(&args[0], env)?)))
        },
        "ones" =>
        {
            need_args(func, args, 1)?;
            Ok(SirExpr::Ones(Box::new(lower_int(&args[0], env)?)))
        },
        "len" =>
        {
            need_args(func, args, 1)?;
            let a = lower_scalar(&args[0], env)?;
            expect_array(&a, "len")?;
            Ok(SirExpr::Len(Box::new(a)))
        },
        "sqrt" | "exp" | "sin" | "cos" | "abs" | "tanh" =>
        {
            need_args(func, args, 1)?;
            let a = lower_scalar(&args[0], env)?;
            let mf = match base
            {
                "sqrt" => MathFn::Sqrt,
                "exp" => MathFn::Exp,
                "sin" => MathFn::Sin,
                "cos" => MathFn::Cos,
                "abs" => MathFn::Abs,
                "tanh" => MathFn::Tanh,
                _ => unreachable!(),
            };
            if mf == MathFn::Abs && a.ty() == Ty::ComplexArray
            {
                // |z| over a complex array -> real magnitude array.
                Ok(SirExpr::ComplexAbs(Box::new(a)))
            }
            else if a.ty() == Ty::Array
            {
                Ok(SirExpr::ArrayUnaryFn {
                    func: mf,
                    arg: Box::new(a),
                })
            }
            else
            {
                Ok(SirExpr::ScalarUnaryFn {
                    func: mf,
                    arg: Box::new(a),
                })
            }
        },
        other => Err(format!(
            "unsupported function `{}` (subset supports np.sum/dot/zeros/ones/sqrt/exp/sin/cos/abs/tanh, len)",
            other
        )),
    }
}

fn strip_np(func: &str) -> &str {
    func.strip_prefix("np.")
        .or_else(|| func.strip_prefix("numpy."))
        .or_else(|| func.strip_prefix("math."))
        .unwrap_or(func)
}

fn need_args(func: &str, args: &[PyExpr], n: usize) -> Result<(), String> {
    if args.len() == n
    {
        Ok(())
    }
    else
    {
        Err(format!(
            "`{}` expects {} argument(s), got {}",
            func,
            n,
            args.len()
        ))
    }
}

fn expect_array(e: &SirExpr, ctx: &str) -> Result<(), String> {
    if e.ty() == Ty::Array
    {
        Ok(())
    }
    else
    {
        Err(format!("{} expects an array argument", ctx))
    }
}

// ---- param type inference (no hint) ---------------------------------------

fn infer_param_ty(name: &str, body: &[PyStmt]) -> Ty {
    if matrix_evidence_block(name, body)
    {
        Ty::Matrix
    }
    else if array_evidence_block(name, body)
    {
        Ty::Array
    }
    else
    {
        Ty::Scalar
    }
}

/// A param is a matrix if it is the first argument of `np.linalg.solve`.
fn matrix_evidence_block(name: &str, stmts: &[PyStmt]) -> bool {
    fn expr(name: &str, e: &PyExpr) -> bool {
        match e
        {
            PyExpr::Call { func, args } =>
            {
                // First argument of a matrix-taking linalg routine is a matrix.
                (matches!(
                    strip_np(func),
                    "linalg.solve" | "linalg.det" | "linalg.eigvalsh"
                ) && matches!(args.first(), Some(PyExpr::Name(n)) if n == name))
                    || args.iter().any(|a| expr(name, a))
            },
            // Left operand of `@` (matrix-multiplication) is a matrix.
            PyExpr::Bin {
                op: BinOp::MatMul,
                l,
                r,
            } =>
            {
                matches!(l.as_ref(), PyExpr::Name(n) if n == name) || expr(name, l) || expr(name, r)
            },
            PyExpr::Bin { l, r, .. } | PyExpr::Cmp { l, r, .. } => expr(name, l) || expr(name, r),
            PyExpr::Neg(inner) => expr(name, inner),
            PyExpr::Index { base, index } => expr(name, base) || expr(name, index),
            _ => false,
        }
    }
    fn block(name: &str, stmts: &[PyStmt]) -> bool {
        stmts.iter().any(|s| match s
        {
            PyStmt::Assign { value, .. } => expr(name, value),
            PyStmt::AssignIndex { index, value, .. } => expr(name, index) || expr(name, value),
            PyStmt::For {
                start, end, body, ..
            } => expr(name, start) || expr(name, end) || block(name, body),
            PyStmt::If { cond, then, els } =>
            {
                expr(name, cond) || block(name, then) || block(name, els)
            },
            PyStmt::While { cond, body } => expr(name, cond) || block(name, body),
            PyStmt::Return(Some(e)) => expr(name, e),
            PyStmt::Return(None) => false,
        })
    }
    block(name, stmts)
}

fn array_evidence_block(name: &str, stmts: &[PyStmt]) -> bool {
    stmts.iter().any(|s| match s
    {
        PyStmt::Assign { value, .. } => array_evidence_expr(name, value),
        PyStmt::AssignIndex {
            target,
            index,
            value,
        } => target == name || array_evidence_expr(name, index) || array_evidence_expr(name, value),
        PyStmt::For {
            start, end, body, ..
        } =>
        {
            array_evidence_expr(name, start)
                || array_evidence_expr(name, end)
                || array_evidence_block(name, body)
        },
        PyStmt::If { cond, then, els } =>
        {
            array_evidence_expr(name, cond)
                || array_evidence_block(name, then)
                || array_evidence_block(name, els)
        },
        PyStmt::While { cond, body } =>
        {
            array_evidence_expr(name, cond) || array_evidence_block(name, body)
        },
        PyStmt::Return(Some(e)) => array_evidence_expr(name, e),
        PyStmt::Return(None) => false,
    })
}

fn array_evidence_expr(name: &str, e: &PyExpr) -> bool {
    match e
    {
        PyExpr::Index { base, index } =>
        {
            matches!(base.as_ref(), PyExpr::Name(n) if n == name)
                || array_evidence_expr(name, base)
                || array_evidence_expr(name, index)
        },
        PyExpr::Call { func, args } =>
        {
            let is_array_consumer = matches!(strip_np(func), "sum" | "dot" | "len");
            // `np.linalg.solve(A, b)` — the *second* argument `b` is a vector.
            let solve_rhs = strip_np(func) == "linalg.solve"
                && matches!(args.get(1), Some(PyExpr::Name(n)) if n == name);
            (is_array_consumer
                && args
                    .iter()
                    .any(|a| matches!(a, PyExpr::Name(n) if n == name)))
                || solve_rhs
                || args.iter().any(|a| array_evidence_expr(name, a))
        },
        // Right operand of `@` (matrix @ vector) is a vector.
        PyExpr::Bin {
            op: BinOp::MatMul,
            l,
            r,
        } =>
        {
            matches!(r.as_ref(), PyExpr::Name(n) if n == name)
                || array_evidence_expr(name, l)
                || array_evidence_expr(name, r)
        },
        PyExpr::Bin { l, r, .. } => array_evidence_expr(name, l) || array_evidence_expr(name, r),
        PyExpr::Cmp { l, r, .. } => array_evidence_expr(name, l) || array_evidence_expr(name, r),
        PyExpr::Neg(inner) => array_evidence_expr(name, inner),
        _ => false,
    }
}
