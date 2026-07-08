//! Lower the MATLAB subset AST into the shared typed [`crate::sir`] IR.
//!
//! MATLAB-specific semantics handled here: 1-based indexing (`a(i)` -> `a[i-1]`),
//! inclusive `for` ranges (`1:n` -> `1..n+1`), element-wise `.*`/`./` vs scalar
//! `*`/`/`, and returning the function's *output variable*. Everything then flows
//! through the same emitter and differential oracle as the Python front-end.

use crate::front_matlab::ast::*;
use crate::sir::*;
use std::collections::{HashMap, HashSet};

const MATH_FNS: &[&str] = &[
    "sqrt", "exp", "sin", "cos", "abs", "tanh", "log", "log10", "floor", "ceil", "sinh", "cosh",
    "atan", "round", "fix",
];

pub fn lower_module(m: &MModule) -> Result<SirModule, String> {
    let funcs = m
        .funcs
        .iter()
        .map(lower_func)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SirModule { funcs })
}

/// Per-function lowering state threaded through the statement walk.
struct Lower<'a> {
    env: HashMap<String, Ty>,
    /// Names already bound (params + everything assigned so far).
    declared: Vec<String>,
    /// Variables whose *first* assignment is inside a nested block; these are
    /// pre-declared (uninitialised) at the top of the function.
    hoisted: &'a HashSet<String>,
    /// Hoisted declarations collected in first-assignment order.
    declares: Vec<(String, Ty)>,
}

fn lower_func(f: &MFunc) -> Result<SirFunc, String> {
    let mut env: HashMap<String, Ty> = HashMap::new();
    let mut params = Vec::new();
    for p in &f.params
    {
        let ty = infer_param_ty(p, &f.body);
        env.insert(p.clone(), ty);
        params.push((p.clone(), ty));
    }

    // Variables whose first assignment is inside an `if`/`for`/`while` block
    // must be hoisted to an uninitialised `let mut` at the top so subsequent
    // `Reassign`s are valid Rust. Rust's definite-assignment analysis then
    // verifies every control-flow path writes them before any read.
    let mut seen: HashSet<String> = f.params.iter().cloned().collect();
    let mut hoisted: HashSet<String> = HashSet::new();
    collect_hoisted(&f.body, &mut seen, 0, &mut hoisted);

    let mut lo = Lower {
        env,
        declared: f.params.clone(),
        hoisted: &hoisted,
        declares: Vec::new(),
    };
    let mut body = Vec::new();
    lower_block(&f.body, &mut lo, true, &mut body)?;

    // MATLAB returns the output variable(s)' final value(s). A single output
    // is a plain return; multiple outputs `[o1, o2]` become a tuple return
    // (mapped onto the same `RetTy::Tuple`/`ReturnTuple` machinery as Python's
    // `return a, b`).
    let mut out_vars = Vec::with_capacity(f.outs.len());
    let mut out_tys = Vec::with_capacity(f.outs.len());
    for o in &f.outs
    {
        let ty = *lo
            .env
            .get(o)
            .ok_or_else(|| format!("output variable `{}` is never assigned", o))?;
        // `Int` accumulators leave the index domain on return -> `Scalar`.
        let ret_ty = if ty == Ty::Int { Ty::Scalar } else { ty };
        out_vars.push(SirExpr::Var {
            name: o.clone(),
            ty,
        });
        out_tys.push(ret_ty);
    }
    let ret = if out_tys.len() == 1
    {
        body.push(SirStmt::Return(out_vars.into_iter().next().unwrap()));
        RetTy::Single(out_tys[0])
    }
    else
    {
        body.push(SirStmt::ReturnTuple(out_vars));
        RetTy::Tuple(out_tys)
    };

    // Prepend hoisted declarations (first-assignment order) to the body.
    let mut full = Vec::with_capacity(lo.declares.len() + body.len());
    for (name, ty) in lo.declares
    {
        full.push(SirStmt::Declare { name, ty });
    }
    full.extend(body);

    Ok(SirFunc {
        name: f.name.clone(),
        params,
        ret,
        body: full,
    })
}

/// Determine which variables have their first assignment nested inside a block.
fn collect_hoisted(
    stmts: &[MStmt],
    seen: &mut HashSet<String>,
    depth: usize,
    hoisted: &mut HashSet<String>,
) {
    for s in stmts
    {
        match s
        {
            MStmt::Assign { target, .. } =>
            {
                if !seen.contains(target)
                {
                    seen.insert(target.clone());
                    if depth > 0
                    {
                        hoisted.insert(target.clone());
                    }
                }
            },
            // Index-assignment never *introduces* a binding (the array must
            // already exist), so it cannot make a variable hoisted.
            MStmt::AssignIndex { .. } =>
            {},
            MStmt::For { var, body, .. } =>
            {
                let had = seen.contains(var);
                seen.insert(var.clone());
                collect_hoisted(body, seen, depth + 1, hoisted);
                if !had
                {
                    seen.remove(var);
                }
            },
            MStmt::If { then, els, .. } =>
            {
                collect_hoisted(then, seen, depth + 1, hoisted);
                collect_hoisted(els, seen, depth + 1, hoisted);
            },
            MStmt::While { body, .. } => collect_hoisted(body, seen, depth + 1, hoisted),
        }
    }
}

fn lower_block(
    stmts: &[MStmt],
    lo: &mut Lower,
    top_level: bool,
    out: &mut Vec<SirStmt>,
) -> Result<(), String> {
    for s in stmts
    {
        lower_stmt(s, lo, top_level, out)?;
    }
    Ok(())
}

fn lower_stmt(
    s: &MStmt,
    lo: &mut Lower,
    top_level: bool,
    out: &mut Vec<SirStmt>,
) -> Result<(), String> {
    match s
    {
        MStmt::Assign { target, value } =>
        {
            let v = lower_scalar(value, &lo.env)?;
            let ty = normalize_ty(v.ty());
            if lo.declared.contains(target)
            {
                let prev = normalize_ty(*lo.env.get(target).unwrap());
                if prev != ty
                {
                    return Err(format!(
                        "variable `{}` reassigned with a different type ({:?} vs {:?})",
                        target, prev, ty
                    ));
                }
                out.push(SirStmt::Reassign {
                    name: target.clone(),
                    value: v,
                });
            }
            else if lo.hoisted.contains(target)
            {
                // First assignment of a hoisted variable: record its type for
                // the top-of-function `Declare`, then emit a plain reassignment.
                lo.env.insert(target.clone(), ty);
                lo.declared.push(target.clone());
                lo.declares.push((target.clone(), ty));
                out.push(SirStmt::Reassign {
                    name: target.clone(),
                    value: v,
                });
            }
            else
            {
                if !top_level
                {
                    return Err(format!(
                        "`{}` is first assigned inside a block; initialise it before the loop",
                        target
                    ));
                }
                lo.env.insert(target.clone(), ty);
                lo.declared.push(target.clone());
                out.push(SirStmt::Let {
                    name: target.clone(),
                    ty,
                    value: v,
                });
            }
            Ok(())
        },
        MStmt::AssignIndex {
            target,
            index,
            value,
        } =>
        {
            match lo.env.get(target)
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
            let idx = one_based(lower_int(index, &lo.env)?);
            let v = lower_scalar(value, &lo.env)?;
            out.push(SirStmt::SetIndex {
                name: target.clone(),
                index: idx,
                value: v,
            });
            Ok(())
        },
        MStmt::For {
            var,
            lo: mlo,
            hi,
            body,
        } =>
        {
            let start = lower_int(mlo, &lo.env)?;
            // Inclusive `lo:hi` -> exclusive `start..hi+1`.
            let end = SirExpr::IntBin {
                op: Op::Add,
                l: Box::new(lower_int(hi, &lo.env)?),
                r: Box::new(SirExpr::IntLit(1)),
            };
            let had = lo.env.insert(var.clone(), Ty::Int);
            let mut inner = Vec::new();
            lower_block(body, lo, false, &mut inner)?;
            match had
            {
                Some(t) =>
                {
                    lo.env.insert(var.clone(), t);
                },
                None =>
                {
                    lo.env.remove(var);
                },
            }
            out.push(SirStmt::For {
                var: var.clone(),
                start,
                end,
                body: inner,
            });
            Ok(())
        },
        MStmt::If { cond, then, els } =>
        {
            let cond = lower_condition(cond, &lo.env)?;
            let mut t = Vec::new();
            lower_block(then, lo, false, &mut t)?;
            let mut e = Vec::new();
            lower_block(els, lo, false, &mut e)?;
            out.push(SirStmt::If {
                cond,
                then: t,
                els: e,
            });
            Ok(())
        },
        MStmt::While { cond, body } =>
        {
            let cond = lower_condition(cond, &lo.env)?;
            let mut inner = Vec::new();
            lower_block(body, lo, false, &mut inner)?;
            out.push(SirStmt::While { cond, body: inner });
            Ok(())
        },
    }
}

/// Subtract 1 from a 1-based index to get a 0-based one.
fn one_based(idx: SirExpr) -> SirExpr {
    SirExpr::IntBin {
        op: Op::Sub,
        l: Box::new(idx),
        r: Box::new(SirExpr::IntLit(1)),
    }
}

fn normalize_ty(t: Ty) -> Ty {
    match t
    {
        Ty::Int => Ty::Scalar,
        other => other,
    }
}

fn lower_int(e: &MExpr, env: &HashMap<String, Ty>) -> Result<SirExpr, String> {
    match e
    {
        MExpr::Num(v) if v.fract() == 0.0 => Ok(SirExpr::IntLit(*v as i64)),
        MExpr::Ident(n) => match env.get(n)
        {
            Some(Ty::Int) => Ok(SirExpr::Var {
                name: n.clone(),
                ty: Ty::Int,
            }),
            Some(other) => Err(format!(
                "`{}` ({:?}) used where an integer index is required",
                n, other
            )),
            None => Err(format!("undefined name `{}` in integer context", n)),
        },
        MExpr::Bin { op, l, r } =>
        {
            let op = int_op(*op)?;
            Ok(SirExpr::IntBin {
                op,
                l: Box::new(lower_int(l, env)?),
                r: Box::new(lower_int(r, env)?),
            })
        },
        MExpr::Call { func, args } if func == "length" =>
        {
            need_args(func, args, 1)?;
            let a = lower_scalar(&args[0], env)?;
            expect_array(&a, "length")?;
            Ok(SirExpr::Len(Box::new(a)))
        },
        other => Err(format!("unsupported integer expression: {:?}", other)),
    }
}

fn int_op(op: MBinOp) -> Result<Op, String> {
    match op
    {
        MBinOp::Add => Ok(Op::Add),
        MBinOp::Sub => Ok(Op::Sub),
        MBinOp::Mul => Ok(Op::Mul),
        MBinOp::Div => Ok(Op::Div),
        _ => Err("that operator is not valid in an integer index context".into()),
    }
}

fn lower_scalar(e: &MExpr, env: &HashMap<String, Ty>) -> Result<SirExpr, String> {
    match e
    {
        MExpr::Num(v) => Ok(SirExpr::ScalarLit(*v)),
        MExpr::Ident(n) => match env.get(n)
        {
            Some(ty) => Ok(SirExpr::Var {
                name: n.clone(),
                ty: *ty,
            }),
            None => Err(format!("undefined name `{}`", n)),
        },
        MExpr::Neg(inner) =>
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
        MExpr::Bin { op, l, r } => lower_bin(*op, l, r, env),
        MExpr::Transpose(inner) =>
        {
            let v = lower_scalar(inner, env)?;
            if !is_matrixish(v.ty())
            {
                return Err("transpose `'` expects a matrix".into());
            }
            Ok(SirExpr::Transpose(Box::new(v)))
        },
        MExpr::Cmp { .. } => Err("a comparison is only valid as an `if`/`while` condition".into()),
        MExpr::Index { .. } => unreachable!("parser produces Call, not Index"),
        MExpr::Call { func, args } => lower_call(func, args, env),
    }
}

fn lower_bin(
    op: MBinOp,
    l: &MExpr,
    r: &MExpr,
    env: &HashMap<String, Ty>,
) -> Result<SirExpr, String> {
    let lv = lower_scalar(l, env)?;
    let rv = lower_scalar(r, env)?;
    let la = lv.ty() == Ty::Array;
    let ra = rv.ty() == Ty::Array;
    match op
    {
        // Matrix power (`^`) — scalars only in this subset (`A^2` on a matrix is
        // `mpower`, not supported; use `.^` for elementwise).
        MBinOp::Pow =>
        {
            if la || ra
            {
                return Err(
                    "`^` (matrix power) on arrays is not supported — use `.^` for \
                     elementwise power"
                        .into(),
                );
            }
            Ok(SirExpr::ScalarPow {
                base: Box::new(lv),
                exp: Box::new(rv),
            })
        },
        // Elementwise power (`.^`): scalar∘scalar, array∘array, or broadcast.
        MBinOp::EPow => Ok(match (la, ra)
        {
            (false, false) => SirExpr::ScalarPow {
                base: Box::new(lv),
                exp: Box::new(rv),
            },
            (true, true) => SirExpr::EwBinFn {
                func: MathFn2::Powf,
                l: Box::new(lv),
                r: Box::new(rv),
            },
            (true, false) => SirExpr::BroadcastFn {
                func: MathFn2::Powf,
                scalar: Box::new(rv),
                arr: Box::new(lv),
                arr_is_left: true,
            },
            (false, true) => SirExpr::BroadcastFn {
                func: MathFn2::Powf,
                scalar: Box::new(lv),
                arr: Box::new(rv),
                arr_is_left: false,
            },
        }),
        // Element-wise ops: array∘array, or scalar broadcast, or plain scalar.
        MBinOp::EMul | MBinOp::EDiv =>
        {
            let sop = if matches!(op, MBinOp::EMul)
            {
                Op::Mul
            }
            else
            {
                Op::Div
            };
            Ok(ew_or_broadcast(sop, lv, rv, la, ra))
        },
        // `* / + -`: matrix ops unsupported, so array∘array is rejected for
        // `*`/`/`; `+`/`-` are element-wise in MATLAB and mapped as such.
        MBinOp::Add | MBinOp::Sub =>
        {
            let sop = if matches!(op, MBinOp::Add)
            {
                Op::Add
            }
            else
            {
                Op::Sub
            };
            Ok(ew_or_broadcast(sop, lv, rv, la, ra))
        },
        // `*` — matrix multiplication in MATLAB. When the left operand is a
        // *matrix* (inferred from `det`/`inv`/`eig`/`\`), route to the verified
        // matrix-vector / matrix-matrix product in `scirust-solvers`; otherwise
        // it is scalar multiply or scalar↔array broadcast (`.*` is elementwise).
        MBinOp::Mul =>
        {
            let lm = is_matrixish(lv.ty());
            let rm = is_matrixish(rv.ty());
            if lm || rm
            {
                if lm && rv.ty() == Ty::Array
                {
                    return Ok(SirExpr::Matvec {
                        a: Box::new(lv),
                        b: Box::new(rv),
                    });
                }
                if lm && rm
                {
                    return Ok(SirExpr::Matmul {
                        a: Box::new(lv),
                        b: Box::new(rv),
                    });
                }
                return Err("unsupported matrix `*` form (supported: matrix*vector, \
                            matrix*matrix; the matrix operand must be on the left)"
                    .into());
            }
            if la && ra
            {
                return Err("matrix `*` on two arrays is not supported — use `.*`".into());
            }
            Ok(ew_or_broadcast(Op::Mul, lv, rv, la, ra))
        },
        MBinOp::Div =>
        {
            if is_matrixish(lv.ty()) || is_matrixish(rv.ty())
            {
                return Err("matrix `/` is not supported in this subset".into());
            }
            if la && ra
            {
                return Err("matrix `/` on two arrays is not supported — use `./`".into());
            }
            Ok(ew_or_broadcast(Op::Div, lv, rv, la, ra))
        },
        // `A \ b` — MATLAB left division (solve `A x = b`), routed to the
        // verified LU solver in `scirust-solvers`.
        MBinOp::LDiv =>
        {
            if !is_matrixish(lv.ty())
            {
                return Err("`\\` (left division / solve) expects a matrix on the left".into());
            }
            if rv.ty() != Ty::Array
            {
                return Err("`\\` (left division / solve) expects a vector on the right".into());
            }
            Ok(SirExpr::LinSolve {
                a: Box::new(lv),
                b: Box::new(rv),
            })
        },
    }
}

/// A matrix operand: either a flat `&[f64]` parameter or a produced `Matrix`.
fn is_matrixish(t: Ty) -> bool {
    matches!(t, Ty::Matrix | Ty::MatrixVal)
}

fn ew_or_broadcast(op: Op, lv: SirExpr, rv: SirExpr, la: bool, ra: bool) -> SirExpr {
    match (la, ra)
    {
        (false, false) => SirExpr::ScalarBin {
            op,
            l: Box::new(lv),
            r: Box::new(rv),
        },
        (true, true) => SirExpr::EwBin {
            op,
            l: Box::new(lv),
            r: Box::new(rv),
        },
        (true, false) => SirExpr::ScalarBroadcast {
            op,
            scalar: Box::new(rv),
            arr: Box::new(lv),
            arr_is_left: true,
        },
        (false, true) => SirExpr::ScalarBroadcast {
            op,
            scalar: Box::new(lv),
            arr: Box::new(rv),
            arr_is_left: false,
        },
    }
}

fn lower_call(func: &str, args: &[MExpr], env: &HashMap<String, Ty>) -> Result<SirExpr, String> {
    if MATH_FNS.contains(&func)
    {
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        let mf = match func
        {
            "sqrt" => MathFn::Sqrt,
            "exp" => MathFn::Exp,
            "sin" => MathFn::Sin,
            "cos" => MathFn::Cos,
            "abs" => MathFn::Abs,
            "tanh" => MathFn::Tanh,
            "log" => MathFn::Ln,
            "log10" => MathFn::Log10,
            "floor" => MathFn::Floor,
            "ceil" => MathFn::Ceil,
            "sinh" => MathFn::Sinh,
            "cosh" => MathFn::Cosh,
            "atan" => MathFn::Atan,
            "round" => MathFn::Round,
            "fix" => MathFn::Trunc,
            _ => unreachable!(),
        };
        return Ok(
            if a.ty() == Ty::Array
            {
                SirExpr::ArrayUnaryFn {
                    func: mf,
                    arg: Box::new(a),
                }
            }
            else
            {
                SirExpr::ScalarUnaryFn {
                    func: mf,
                    arg: Box::new(a),
                }
            },
        );
    }
    if func == "sum"
    {
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        expect_array(&a, "sum")?;
        return Ok(SirExpr::Sum(Box::new(a)));
    }
    if func == "prod"
    {
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        expect_array(&a, "prod")?;
        return Ok(SirExpr::Prod(Box::new(a)));
    }
    if (func == "max" || func == "min") && args.len() == 2
    {
        // Two-argument form: `max(a, b)` / `min(a, b)` over two scalars.
        let l = lower_scalar(&args[0], env)?;
        let r = lower_scalar(&args[1], env)?;
        expect_scalar(&l, func)?;
        expect_scalar(&r, func)?;
        let mf = if func == "max"
        {
            MathFn2::Max
        }
        else
        {
            MathFn2::Min
        };
        return Ok(SirExpr::ScalarBinFn {
            func: mf,
            l: Box::new(l),
            r: Box::new(r),
        });
    }
    if func == "max"
    {
        // One-argument form: reduction over a vector.
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        expect_array(&a, "max")?;
        return Ok(SirExpr::Max(Box::new(a)));
    }
    if func == "min"
    {
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        expect_array(&a, "min")?;
        return Ok(SirExpr::Min(Box::new(a)));
    }
    if func == "mean"
    {
        // mean(x) = sum(x) / length(x) — reuses existing nodes.
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        expect_array(&a, "mean")?;
        return Ok(SirExpr::ScalarBin {
            op: Op::Div,
            l: Box::new(SirExpr::Sum(Box::new(a.clone()))),
            r: Box::new(SirExpr::Len(Box::new(a))),
        });
    }
    if matches!(func, "var" | "std" | "median")
    {
        // Reduction statistics (array -> scalar): MATLAB sample variance
        // (`N-1`), its square root, and the median.
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        expect_array(&a, func)?;
        let boxed = Box::new(a);
        return Ok(match func
        {
            "var" => SirExpr::Variance(boxed),
            "std" => SirExpr::Stdev(boxed),
            _ => SirExpr::Median(boxed),
        });
    }
    if func == "norm"
    {
        // norm(v) — Euclidean 2-norm of a *vector* = sqrt(sum(v .* v)).
        // Restricted to a vector argument (a matrix `norm` is the spectral norm,
        // a different quantity); built from existing verified nodes.
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        expect_array(&a, "norm")?;
        let sq = SirExpr::EwBin {
            op: Op::Mul,
            l: Box::new(a.clone()),
            r: Box::new(a),
        };
        return Ok(SirExpr::ScalarUnaryFn {
            func: MathFn::Sqrt,
            arg: Box::new(SirExpr::Sum(Box::new(sq))),
        });
    }
    if func == "dot"
    {
        // dot(a, b) — inner product, routed to the fixed-order `np::dot` prelude
        // (bit-reproducible reduction order).
        need_args(func, args, 2)?;
        let a = lower_scalar(&args[0], env)?;
        let b = lower_scalar(&args[1], env)?;
        expect_array(&a, "dot")?;
        expect_array(&b, "dot")?;
        return Ok(SirExpr::Dot(Box::new(a), Box::new(b)));
    }
    if func == "cross"
    {
        // cross(a, b) — the 3-vector cross product (both operands are vectors).
        need_args(func, args, 2)?;
        let a = lower_scalar(&args[0], env)?;
        let b = lower_scalar(&args[1], env)?;
        expect_array(&a, "cross")?;
        expect_array(&b, "cross")?;
        return Ok(SirExpr::Cross(Box::new(a), Box::new(b)));
    }
    if func == "linspace"
    {
        // linspace(a, b, n) — n evenly-spaced points from a to b (a, b scalars;
        // n an integer count, e.g. a literal or `length(x)`).
        need_args(func, args, 3)?;
        let a = lower_scalar(&args[0], env)?;
        let b = lower_scalar(&args[1], env)?;
        expect_scalar(&a, "linspace")?;
        expect_scalar(&b, "linspace")?;
        let n = lower_int(&args[2], env)?;
        return Ok(SirExpr::Linspace {
            a: Box::new(a),
            b: Box::new(b),
            n: Box::new(n),
        });
    }
    if func == "diag"
    {
        // MATLAB's overloaded `diag`, dispatched on the operand type:
        //   diag(A) with A a matrix  -> extract the diagonal (a vector);
        //   diag(v) with v a vector  -> construct a diagonal matrix.
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        if is_matrixish(a.ty())
        {
            return Ok(SirExpr::DiagExtract(Box::new(a)));
        }
        if a.ty() == Ty::Array
        {
            return Ok(SirExpr::Diag(Box::new(a)));
        }
        return Err("diag expects a matrix (extract the diagonal) or a vector \
                    (construct a diagonal matrix)"
            .into());
    }
    if func == "trapz"
    {
        // trapz(v) — trapezoidal integration with unit spacing.
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        expect_array(&a, "trapz")?;
        return Ok(SirExpr::Trapz(Box::new(a)));
    }
    if matches!(
        func,
        "cumsum" | "cumprod" | "cummax" | "cummin" | "diff" | "sort" | "flip"
    )
    {
        // Vector -> vector builtins (array in, array out).
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        expect_array(&a, func)?;
        let boxed = Box::new(a);
        return Ok(match func
        {
            "cumsum" => SirExpr::Cumsum(boxed),
            "cumprod" => SirExpr::Cumprod(boxed),
            "cummax" => SirExpr::Cummax(boxed),
            "cummin" => SirExpr::Cummin(boxed),
            "diff" => SirExpr::Diff(boxed),
            "sort" => SirExpr::Sort(boxed),
            _ => SirExpr::Flip(boxed),
        });
    }
    if func == "mod" || func == "rem"
    {
        // MATLAB scalar modulo/remainder, built from existing nodes:
        //   mod(a, b) = a - b * floor(a / b)   (result follows the divisor sign)
        //   rem(a, b) = a - b * fix(a / b)     (result follows the dividend sign)
        // `fix` is truncate-toward-zero.
        need_args(func, args, 2)?;
        let a = lower_scalar(&args[0], env)?;
        let b = lower_scalar(&args[1], env)?;
        expect_scalar(&a, func)?;
        expect_scalar(&b, func)?;
        let round_fn = if func == "mod"
        {
            MathFn::Floor
        }
        else
        {
            MathFn::Trunc
        };
        let quotient = SirExpr::ScalarBin {
            op: Op::Div,
            l: Box::new(a.clone()),
            r: Box::new(b.clone()),
        };
        let rounded = SirExpr::ScalarUnaryFn {
            func: round_fn,
            arg: Box::new(quotient),
        };
        let scaled = SirExpr::ScalarBin {
            op: Op::Mul,
            l: Box::new(b),
            r: Box::new(rounded),
        };
        return Ok(SirExpr::ScalarBin {
            op: Op::Sub,
            l: Box::new(a),
            r: Box::new(scaled),
        });
    }
    if func == "sign"
    {
        // sign(x) -> -1/0/+1 (MATLAB semantics; sign(0) == 0). Scalar only.
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        expect_scalar(&a, "sign")?;
        return Ok(SirExpr::Sign(Box::new(a)));
    }
    if func == "atan2" || func == "hypot"
    {
        // Two-argument scalar math: atan2(y, x) / hypot(a, b). Scalar operands.
        need_args(func, args, 2)?;
        let l = lower_scalar(&args[0], env)?;
        let r = lower_scalar(&args[1], env)?;
        expect_scalar(&l, func)?;
        expect_scalar(&r, func)?;
        let mf = if func == "atan2"
        {
            MathFn2::Atan2
        }
        else
        {
            MathFn2::Hypot
        };
        return Ok(SirExpr::ScalarBinFn {
            func: mf,
            l: Box::new(l),
            r: Box::new(r),
        });
    }
    if func == "power"
    {
        // power(a, b) — the functional form of `a ^ b` (scalar exponentiation),
        // sharing the `^` lowering (integer exponents fold to `powi`).
        need_args(func, args, 2)?;
        let base = lower_scalar(&args[0], env)?;
        let exp = lower_scalar(&args[1], env)?;
        expect_scalar(&base, "power")?;
        expect_scalar(&exp, "power")?;
        return Ok(SirExpr::ScalarPow {
            base: Box::new(base),
            exp: Box::new(exp),
        });
    }
    if func == "det"
    {
        // det(A) -> routed to the verified LU-based determinant.
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        if !is_matrixish(a.ty())
        {
            return Err("det expects a matrix argument".into());
        }
        return Ok(SirExpr::Det(Box::new(a)));
    }
    if func == "inv"
    {
        // inv(A) -> routed to the verified matrix inverse (returns a matrix).
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        if !is_matrixish(a.ty())
        {
            return Err("inv expects a matrix argument".into());
        }
        return Ok(SirExpr::Inv(Box::new(a)));
    }
    if func == "trace"
    {
        // trace(A) -> sum of the diagonal of a matrix.
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        if !is_matrixish(a.ty())
        {
            return Err("trace expects a matrix argument".into());
        }
        return Ok(SirExpr::Trace(Box::new(a)));
    }
    if func == "eig"
    {
        // eig(A) -> eigenvalues (ascending), routed to the verified symmetric
        // eigensolver. Octave's `eig` returns ascending real eigenvalues for a
        // symmetric matrix, matching `scirust_solvers::eigen_symmetric`; this
        // route is therefore proven on symmetric inputs (see the oracle).
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        if !is_matrixish(a.ty())
        {
            return Err("eig expects a matrix argument".into());
        }
        return Ok(SirExpr::Eigvalsh(Box::new(a)));
    }
    if func == "length"
    {
        need_args(func, args, 1)?;
        let a = lower_scalar(&args[0], env)?;
        expect_array(&a, "length")?;
        return Ok(SirExpr::Len(Box::new(a)));
    }
    // Otherwise `name(idx)` is 1-based indexing of a variable.
    match env.get(func)
    {
        Some(Ty::Array) =>
        {
            need_args(func, args, 1)?;
            let idx = one_based(lower_int(&args[0], env)?);
            Ok(SirExpr::Index {
                base: Box::new(SirExpr::Var {
                    name: func.to_string(),
                    ty: Ty::Array,
                }),
                idx: Box::new(idx),
            })
        },
        Some(other) => Err(format!("cannot index non-array `{}` ({:?})", func, other)),
        None => Err(format!(
            "unknown function or variable `{}` (supported intrinsics: \
             sqrt/exp/log/log10/sin/cos/sinh/cosh/tanh/abs/floor/ceil/atan/round/fix, \
             mod/rem/sign/atan2/hypot/power, \
             sum/prod/mean/max/min/var/std/median/norm/dot/cross/trapz, \
             cumsum/cumprod/cummax/cummin/diff/sort/flip/diag, linspace, length, det/inv/eig/trace)",
            func
        )),
    }
}

fn lower_condition(e: &MExpr, env: &HashMap<String, Ty>) -> Result<SirExpr, String> {
    match e
    {
        MExpr::Cmp { op, l, r } =>
        {
            let lv = lower_scalar(l, env)?;
            let rv = lower_scalar(r, env)?;
            if lv.ty() == Ty::Array || rv.ty() == Ty::Array
            {
                return Err("array comparisons are not supported in conditions".into());
            }
            let cop = match op
            {
                MCmpOp::Lt => CmpOp::Lt,
                MCmpOp::Le => CmpOp::Le,
                MCmpOp::Gt => CmpOp::Gt,
                MCmpOp::Ge => CmpOp::Ge,
                MCmpOp::Eq => CmpOp::Eq,
                MCmpOp::Ne => CmpOp::Ne,
            };
            Ok(SirExpr::Cmp {
                op: cop,
                l: Box::new(lv),
                r: Box::new(rv),
            })
        },
        _ => Err("`if`/`while` condition must be a comparison".into()),
    }
}

fn need_args(func: &str, args: &[MExpr], n: usize) -> Result<(), String> {
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

fn expect_scalar(e: &SirExpr, ctx: &str) -> Result<(), String> {
    if e.ty() == Ty::Scalar
    {
        Ok(())
    }
    else
    {
        Err(format!("{} expects a scalar argument in this subset", ctx))
    }
}

// ---- param type inference -------------------------------------------------

fn infer_param_ty(name: &str, body: &[MStmt]) -> Ty {
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

/// A MATLAB param is a matrix if it is the argument of `det`/`inv` (or another
/// matrix-taking intrinsic) or the *left* operand of `\` (left division).
fn matrix_evidence_block(name: &str, stmts: &[MStmt]) -> bool {
    stmts.iter().any(|s| match s
    {
        MStmt::Assign { value, .. } => matrix_evidence_expr(name, value),
        MStmt::AssignIndex { index, value, .. } =>
        {
            matrix_evidence_expr(name, index) || matrix_evidence_expr(name, value)
        },
        MStmt::For { lo, hi, body, .. } =>
        {
            matrix_evidence_expr(name, lo)
                || matrix_evidence_expr(name, hi)
                || matrix_evidence_block(name, body)
        },
        MStmt::If { cond, then, els } =>
        {
            matrix_evidence_expr(name, cond)
                || matrix_evidence_block(name, then)
                || matrix_evidence_block(name, els)
        },
        MStmt::While { cond, body } =>
        {
            matrix_evidence_expr(name, cond) || matrix_evidence_block(name, body)
        },
    })
}

fn matrix_evidence_expr(name: &str, e: &MExpr) -> bool {
    match e
    {
        MExpr::Call { func, args } =>
        {
            (matches!(func.as_str(), "det" | "inv" | "eig" | "trace")
                && matches!(args.first(), Some(MExpr::Ident(n)) if n == name))
                || args.iter().any(|a| matrix_evidence_expr(name, a))
        },
        // `name \ b` — the left operand of left-division is a matrix.
        MExpr::Bin {
            op: MBinOp::LDiv,
            l,
            r,
        } => is_ident(name, l) || matrix_evidence_expr(name, l) || matrix_evidence_expr(name, r),
        MExpr::Bin { l, r, .. } | MExpr::Cmp { l, r, .. } =>
        {
            matrix_evidence_expr(name, l) || matrix_evidence_expr(name, r)
        },
        MExpr::Neg(inner) => matrix_evidence_expr(name, inner),
        // `name'` — only a matrix can be transposed in this subset.
        MExpr::Transpose(inner) => is_ident(name, inner) || matrix_evidence_expr(name, inner),
        _ => false,
    }
}

fn array_evidence_block(name: &str, stmts: &[MStmt]) -> bool {
    stmts.iter().any(|s| match s
    {
        MStmt::Assign { value, .. } => array_evidence_expr(name, value),
        MStmt::AssignIndex {
            target,
            index,
            value,
        } => target == name || array_evidence_expr(name, index) || array_evidence_expr(name, value),
        MStmt::For { lo, hi, body, .. } =>
        {
            array_evidence_expr(name, lo)
                || array_evidence_expr(name, hi)
                || array_evidence_block(name, body)
        },
        MStmt::If { cond, then, els } =>
        {
            array_evidence_expr(name, cond)
                || array_evidence_block(name, then)
                || array_evidence_block(name, els)
        },
        MStmt::While { cond, body } =>
        {
            array_evidence_expr(name, cond) || array_evidence_block(name, body)
        },
    })
}

fn array_evidence_expr(name: &str, e: &MExpr) -> bool {
    match e
    {
        // `name(idx)` — indexing a variable makes it an array; a reduction such
        // as `sum(name)` / `length(name)` / `min(name)` also implies an array.
        MExpr::Call { func, args } =>
        {
            (func == name && !MATH_FNS.contains(&func.as_str()) && !is_reduction(func))
                // A reduction implies an array argument only in its 1-arg form;
                // `max(a, b)` / `min(a, b)` are two-scalar intrinsics, not
                // reductions, so they must not mark their operands as arrays.
                || (is_reduction(func)
                    && args.len() == 1
                    && matches!(args.first(), Some(MExpr::Ident(n)) if n == name))
                // `dot(a, b)` / `cross(a, b)` — both operands are vectors, so
                // either being the name is evidence (the generic reduction arm
                // checks only the first argument).
                || (matches!(func.as_str(), "dot" | "cross")
                    && args.iter().any(|a| is_ident(name, a)))
                // Vector -> vector builtins whose (single) argument is a vector.
                || (matches!(
                    func.as_str(),
                    "cumsum" | "cumprod" | "cummax" | "cummin" | "diff" | "sort" | "flip"
                ) && matches!(args.first(), Some(MExpr::Ident(n)) if n == name))
                || args.iter().any(|a| array_evidence_expr(name, a))
        },
        // Element-wise operators (`.*`, `./`, `.^`) imply their operands are
        // arrays, so a bare `name` operand counts as evidence.
        MExpr::Bin {
            op: MBinOp::EMul | MBinOp::EDiv | MBinOp::EPow,
            l,
            r,
        } =>
        {
            is_ident(name, l)
                || is_ident(name, r)
                || array_evidence_expr(name, l)
                || array_evidence_expr(name, r)
        },
        // `A \ b` — left division: the *right* operand is a vector.
        MExpr::Bin {
            op: MBinOp::LDiv,
            l: _,
            r,
        } => is_ident(name, r) || array_evidence_expr(name, r),
        MExpr::Bin { l, r, .. } | MExpr::Cmp { l, r, .. } =>
        {
            array_evidence_expr(name, l) || array_evidence_expr(name, r)
        },
        MExpr::Neg(inner) => array_evidence_expr(name, inner),
        _ => false,
    }
}

fn is_ident(name: &str, e: &MExpr) -> bool {
    matches!(e, MExpr::Ident(n) if n == name)
}

/// A reduction intrinsic that consumes an array and yields a scalar/length.
/// (`dot` is a two-argument reduction and is handled separately where both
/// operands must count as evidence.)
fn is_reduction(func: &str) -> bool {
    matches!(
        func,
        "sum"
            | "prod"
            | "mean"
            | "max"
            | "min"
            | "length"
            | "norm"
            | "var"
            | "std"
            | "median"
            | "trapz"
    )
}
