//! Lower the MATLAB subset AST into the shared typed [`crate::sir`] IR.
//!
//! MATLAB-specific semantics handled here: 1-based indexing (`a(i)` -> `a[i-1]`),
//! inclusive `for` ranges (`1:n` -> `1..n+1`), element-wise `.*`/`./` vs scalar
//! `*`/`/`, and returning the function's *output variable*. Everything then flows
//! through the same emitter and differential oracle as the Python front-end.

use crate::front_matlab::ast::*;
use crate::sir::*;
use std::collections::{HashMap, HashSet};

const MATH_FNS: &[&str] = &["sqrt", "exp", "sin", "cos", "abs", "tanh"];

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

    // MATLAB returns the output variable's final value.
    let out_ty = *lo
        .env
        .get(&f.out)
        .ok_or_else(|| format!("output variable `{}` is never assigned", f.out))?;
    body.push(SirStmt::Return(SirExpr::Var {
        name: f.out.clone(),
        ty: out_ty,
    }));
    let ret = if out_ty == Ty::Int
    {
        Ty::Scalar
    }
    else
    {
        out_ty
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
        // Scalar power (`^`) — scalars only.
        MBinOp::Pow | MBinOp::EPow =>
        {
            if la || ra
            {
                return Err("`^`/`.^` on arrays is not supported in this subset".into());
            }
            Ok(SirExpr::ScalarPow {
                base: Box::new(lv),
                exp: Box::new(rv),
            })
        },
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
        MBinOp::Mul | MBinOp::Div =>
        {
            let sop = if matches!(op, MBinOp::Mul)
            {
                Op::Mul
            }
            else
            {
                Op::Div
            };
            if la && ra
            {
                return Err("matrix `*`/`/` on two arrays is not supported — use `.*`/`./`".into());
            }
            Ok(ew_or_broadcast(sop, lv, rv, la, ra))
        },
    }
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
             sqrt/exp/sin/cos/abs/tanh, sum, length)",
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

// ---- param type inference -------------------------------------------------

fn infer_param_ty(name: &str, body: &[MStmt]) -> Ty {
    if array_evidence_block(name, body)
    {
        Ty::Array
    }
    else
    {
        Ty::Scalar
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
        // `name(idx)` — indexing a variable makes it an array; `sum(name)` /
        // `length(name)` also imply an array.
        MExpr::Call { func, args } =>
        {
            (func == name
                && !MATH_FNS.contains(&func.as_str())
                && func != "sum"
                && func != "length")
                || (matches!(func.as_str(), "sum" | "length")
                    && matches!(args.first(), Some(MExpr::Ident(n)) if n == name))
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
